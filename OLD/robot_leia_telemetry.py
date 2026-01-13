"""
# robot_leia_telemetry.py
-----------------
Handles the World Model and Logging for Robot Leia.
"""
import time
import json
import os
import threading
from enum import Enum

import telemetry_brick
import telemetry_robot
import telemetry_wall

class ObjectiveState(Enum):
    FIND_WALL = "FIND_WALL"
    FIND = "FIND"
    ALIGN = "ALIGN"
    SCOOP = "SCOOP"
    LIFT = "LIFT"
    CARRY = "CARRY"
    PLACE = "PLACE"
    RETREAT = "RETREAT"

class MotionEvent:
    def __init__(self, action_type, power, duration_ms):
        self.action_type = action_type
        self.power = power
        self.duration_ms = duration_ms
        self.timestamp = time.time()

    def to_dict(self):
        return {
            "type": self.action_type,
            "power": self.power,
            "duration_ms": self.duration_ms,
            "timestamp": round(self.timestamp, 3)
        }

MOTION_EVENT_TYPES = {
    "forward",
    "backward",
    "left_turn",
    "right_turn",
    "mast_up",
    "mast_down"
}

from pathlib import Path
WORLD_MODEL_PROCESS_FILE = Path(__file__).parent / "world_model_process.json"
WORLD_MODEL_BRICK_FILE = Path(__file__).parent / "world_model_brick.json"

class WorldModel:
    def __init__(self):
        # Load Process Rules
        self.process_rules = {}
        if WORLD_MODEL_PROCESS_FILE.exists():
            try:
                with open(WORLD_MODEL_PROCESS_FILE, 'r') as f:
                    self.process_rules = json.load(f).get("objectives", {})
            except: pass
        self.rules = self.process_rules
            
        self.learned_rules = {} # Rules derived from demo analysis
            
        # Robot Pose (Dead Reckoning)
        self.x = 0.0 # mm
        self.y = 0.0 # mm
        self.theta = 0.0 # degrees

        # Wall Model
        self.wall_model = telemetry_wall.load_wall_model()
        self.wall_envelope = telemetry_wall.build_envelope(self.wall_model)
        self.wall = telemetry_wall.init_wall_state(self.wall_envelope)

        # Brick Data
        self.brick = {
            "visible": False,
            "id": None,
            "dist": 0,
            "angle": 0,
            "offset_x": 0,
            "confidence": 0,
            "held": False,
            "seated": False,
            "height_mm": None,
            "brickAbove": False,
            "brickBelow": False
        }

        # Forklift
        self.lift_height = 0.0 # mm (estimated)
        self.camera_height_anchor = None

        # Objective
        self._objective_state = None
        self._objective_start_time = 0
        self.objective_state = ObjectiveState.FIND
        self.attempt_status = "NORMAL" # NORMAL, FAIL, RECOVERY
        self.run_id = "unset"
        self.attempt_id = 0
        self.recording_active = False # For HUD prompt logic (Idle vs Success phase)
        
        # Alignment & Stability
        self.align_tol_angle = 5.0    # +/- Degrees
        self.align_tol_offset = 12.0  # +/- mm
        self.align_tol_dist_min = 30.0 # mm (Too close)
        self.align_tol_dist_max = 500.0 # mm (Too far)
        self.scoop_commit_offset_factor = 1.2
        self.scoop_success_offset_factor = 1.2
        self.stability_count = 0
        self.stability_threshold = 10  # 10 frames @ 20Hz = 0.5 seconds
        
        self.last_dist = 999.0 # Track last distance for seated heuristic
        self.last_align_time = None
        self.last_align_dist = None
        self.last_visible_time = None
        self.scoop_desired_offset_x = 0.0
        self.scoop_lateral_drift = 0.0
        self.scoop_forward_preferred = False
        
        self.last_image_file = None
        
        # Wiggle Verification
        self.verification_stage = "IDLE" # IDLE, BACK, LEFT, RIGHT
        self.verify_dist_mm = 0.0
        self.verify_turn_deg = 0.0
        self.verify_vision_hits = 0

        # Internal physics constants for dead reckoning (Calibration needed!)
        self.mm_per_sec_full_speed = 200.0 
        self.deg_per_sec_full_speed = 90.0
        self.lift_mm_per_sec = 23.5 # Adjusted for better dead reckoning
        self.lift_height_anchor = None # The Vision height at Mast=0mm

    @property
    def objective_state(self):
        return self._objective_state

    @objective_state.setter
    def objective_state(self, value):
        if self._objective_state == value:
            return
        self._objective_state = value
        self._objective_start_time = time.time()
        self.last_align_time = None
        self.last_align_dist = None
        self.last_visible_time = None
        # print(f"[WORLD] Objective changed to {value}, timer reset.", flush=True)

    @property
    def wall_origin(self):
        return self.wall.get("origin")

    @wall_origin.setter
    def wall_origin(self, value):
        self.wall["origin"] = value
        self.wall["valid"] = value is not None

    def update_from_motion(self, event):
        """
        Updates pose based on motion events (Dead Reckoning).
        Also manages "Wiggle Verification" state machine.
        """
        delta = telemetry_robot.update_from_motion(self, event)
        telemetry_brick.update_from_motion(self, event, delta)

    def update_vision(self, found, dist, angle, conf, offset_x=0, cam_h=0, brick_above=False, brick_below=False):
        brick_height = telemetry_brick.update_from_vision(
            self,
            found,
            dist,
            angle,
            conf,
            offset_x,
            cam_h,
            brick_above,
            brick_below,
        )
        telemetry_robot.update_lift_from_vision(self, cam_h, brick_height, conf)
        telemetry_wall.update_from_vision(self, found, dist, angle, conf, self.wall_envelope)

    def get_scoop_corridor_limits(self, dist):
        return telemetry_brick.get_scoop_corridor_limits(self, dist)

    def compute_brick_world_xy(self, dist, angle_deg):
        return telemetry_brick.compute_brick_world_xy(self, dist, angle_deg)

    def is_aligned(self):
        """Returns True if metrics have been stable and centered."""
        return self.stability_count >= self.stability_threshold

    def check_objective_complete(self):
        """Checks if success criteria are met using learned rules from demos."""
        wall_check = telemetry_wall.evaluate_success_gates(self, self.objective_state, self.wall_envelope)
        if not wall_check.ok:
            return False
        obj_name = self.objective_state.value
        if obj_name == ObjectiveState.SCOOP.value:
            return bool(self.brick.get("seated")) and self.verification_stage == "IDLE"

        gates = self.learned_rules.get(obj_name, {}).get("gates", {})
        success_metrics = gates.get("success", {}).get("metrics", {})
        if success_metrics:
            brick = self.brick or {}
            brick_visible = bool(brick.get("visible"))
            for metric, stats in success_metrics.items():
                if metric in ("angle_abs", "offset_abs", "dist", "confidence") and not brick_visible:
                    return False
                if metric == "angle_abs":
                    if abs(brick.get("angle", 0.0)) > stats.get("max", 0.0):
                        return False
                elif metric == "offset_abs":
                    if abs(brick.get("offset_x", 0.0)) > stats.get("max", 0.0):
                        return False
                elif metric == "dist":
                    if brick.get("dist", 0.0) > stats.get("max", 0.0):
                        return False
                elif metric == "confidence":
                    if brick.get("confidence", 0.0) < stats.get("min", 0.0):
                        return False
                elif metric == "visible":
                    if (1.0 if brick_visible else 0.0) < stats.get("min", 0.0):
                        return False
                elif metric == "lift_height":
                    lift = self.lift_height
                    if lift < stats.get("min", lift) or lift > stats.get("max", lift):
                        return False
            return True

        learned = self.learned_rules.get(obj_name, {})
        if not learned:
            return False

        target_vis = learned.get("final_visibility", True)
        if self.brick["visible"] != target_vis:
            return False

        if target_vis:
            max_x = learned.get("max_offset_x", 0)
            if abs(self.brick["offset_x"]) > max_x:
                return False
            max_ang = learned.get("max_angle", 0)
            if abs(self.brick["angle"]) > max_ang:
                return False

        return True

    def next_objective(self):
        """Cycles through the mission: FIND -> ALIGN -> SCOOP -> LIFT -> CARRY -> PLACE -> RETREAT"""
        if self.objective_state == ObjectiveState.FIND:
            self.objective_state = ObjectiveState.ALIGN
        elif self.objective_state == ObjectiveState.ALIGN:
            self.objective_state = ObjectiveState.SCOOP
        elif self.objective_state == ObjectiveState.SCOOP:
            self.objective_state = ObjectiveState.LIFT
        elif self.objective_state == ObjectiveState.LIFT:
            self.objective_state = ObjectiveState.CARRY
        elif self.objective_state == ObjectiveState.CARRY:
            self.objective_state = ObjectiveState.PLACE
        elif self.objective_state == ObjectiveState.PLACE:
            self.objective_state = ObjectiveState.RETREAT
        else:
            self.objective_state = ObjectiveState.FIND
            self.brick["seated"] = False # Reset on new cycle
            self.brick["held"] = False
            self.verification_stage = "IDLE"

        return self.objective_state.value

    def get_next_objective_label(self):
        """Returns the string label of the next objective in sequence."""
        objs = [o.value for o in ObjectiveState]
        curr_idx = objs.index(self.objective_state.value)
        next_idx = (curr_idx + 1) % len(objs)
        return objs[next_idx]

    def reset_mission(self):
        """Resets the objective state and all mission-specific flags."""
        self.objective_state = ObjectiveState.FIND
        self.brick["seated"] = False
        self.brick["held"] = False
        self.stability_count = 0
        self.verification_stage = "IDLE"
        self.verify_dist_mm = 0.0
        self.verify_turn_deg = 0.0
        self.verify_vision_hits = 0
        self.last_visible_time = None
        return self.objective_state.value

    def to_dict(self):
        # Format Brick Data
        brick_fmt = self.brick.copy()
        if brick_fmt.get("visible"):
            if brick_fmt.get("dist") is not None:
                brick_fmt['dist'] = round(brick_fmt['dist'], 2)
            if brick_fmt.get("angle") is not None:
                brick_fmt['angle'] = round(brick_fmt['angle'], 3)
            if brick_fmt.get("offset_x") is not None:
                brick_fmt['offset_x'] = round(brick_fmt['offset_x'], 2)
            if brick_fmt.get("confidence") is not None:
                brick_fmt['confidence'] = int(brick_fmt['confidence'])
            if brick_fmt.get("height_mm") is not None:
                brick_fmt['height_mm'] = round(brick_fmt['height_mm'], 2)
        else:
            brick_fmt['dist'] = None
            brick_fmt['angle'] = None
            brick_fmt['offset_x'] = None
            brick_fmt['confidence'] = None

        # Format Wall Origin
        wall_fmt = None
        if self.wall.get("origin"):
            wall_fmt = {
                'x': round(self.wall["origin"]['x'], 2),
                'y': round(self.wall["origin"]['y'], 2),
                'theta': round(self.wall["origin"]['theta'], 3)
            }
        wall_state = {
            "origin": wall_fmt,
            "angle_deg": round(self.wall.get("angle_deg", 0.0), 3),
            "valid": bool(self.wall.get("valid", False)),
            "source": self.wall.get("source"),
            "contradiction": self.wall.get("contradiction_reason"),
        }

        return {
            "type": "state",
            "timestamp": round(time.time(), 3),
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "robot_pose": {
                "x": round(self.x, 2), 
                "y": round(self.y, 2), 
                "theta": round(self.theta, 3)
            },
            "wall_origin": wall_fmt,
            "wall": wall_state,
            "brick": brick_fmt,
            "lift_height": round(self.lift_height, 2)
        }

class TelemetryLogger:
    def __init__(self, filename="leia_log.json"):
        self.filename = filename
        self.lock = threading.Lock()
        self.enabled = False # Don't log state until first keyframe
        # Clear old log
        with open(self.filename, 'w') as f:
            f.write("[\n") # Start JSON array
        self.first_entry = True

    def log_state(self, world_model: WorldModel):
        if not self.enabled:
            return
        data = world_model.to_dict()
        self._write_row(data)

    def log_keyframe(self, marker, objective=None, timestamp=None):
        self.enabled = True # Start recording state once we have a semantic marker
        if timestamp is None:
            timestamp = time.time()
        
        data = {
            "type": "keyframe",
            "timestamp": round(timestamp, 3),
            "marker": marker
        }
        if objective:
            data["objective"] = objective
            
        self._write_row(data)

    def _write_row(self, data):
        with self.lock:
            with open(self.filename, 'a') as f:
                if not self.first_entry:
                    f.write(",\n")
                json.dump(data, f)
                self.first_entry = False

    def log_event(self, event: MotionEvent, objective=None):
        semantic_events = ['FAIL', 'RECOVERY_START', 'OBJECTIVE_SUCCESS', 'JOB_SUCCESS', 'JOB_START']
        if event.action_type in semantic_events:
            self.log_keyframe(event.action_type, objective, event.timestamp)
            return

        if not self.enabled:
            return

        if event.action_type not in MOTION_EVENT_TYPES:
            return

        data = {
            "type": "action",
            "timestamp": round(event.timestamp, 3),
            "command": event.action_type,
            "power": int(event.power),
            "duration_ms": int(event.duration_ms)
        }

        self._write_row(data)

    def close(self):
        """
        Consolidated close method that handles JSON array termination.
        Robustly handles crashes by searching backward for the last valid '}'.
        """
        with self.lock:
            if not os.path.exists(self.filename):
                return
                
            try:
                with open(self.filename, 'rb+') as f:
                    f.seek(0, os.SEEK_END)
                    pos = f.tell()
                    
                    found_last_brace = False
                    # Search backwards for the last '}'
                    while pos > 0:
                        pos -= 1
                        f.seek(pos)
                        char = f.read(1)
                        if char == b'}':
                            # Found the end of a valid JSON object.
                            # Keep this row, truncate after it.
                            f.seek(pos + 1)
                            f.truncate()
                            found_last_brace = True
                            break
                        elif char == b'[': 
                            # Empty array case
                            f.seek(pos + 1)
                            f.truncate()
                            break
                    
                    # Ensure any trailing garbage (like a loose comma) is gone
                    # We already truncated at '}', so we are good.
                    
                    # Add final closing bracket
                    f.seek(0, os.SEEK_END)
                    if found_last_brace:
                        f.write(b"\n]\n")
                    else:
                        # If list was totally empty or malformed
                        f.write(b"]\n")
                        
                print(f"[LOGGER] Log closed and sanitized: {self.filename}")
            except Exception as e:
                print(f"[LOGGER] Error closing log: {e}")

    def _print_terminal(self, data):
        p = data.get('robot_pose', {'x':0, 'y':0, 'theta':0})
        b = data.get('brick', {})
        wall = "SET" if data.get('wall_origin') else "UNSET"
        print(f"{'='*40}")
        print(f"TIME: {data.get('timestamp', 0):.2f}s")
        if 'objective' in data:
            print(f"OBJECTIVE: {data['objective']}")
        print(f"WALL: {wall}")
        print(f"{'-'*40}")
        print(f"POSE:")
        print(f"  X: {p['x']:.2f} mm")
        print(f"  Y: {p['y']:.2f} mm")
        print(f"  Heading: {p['theta']:.2f}°")
        print(f"  Lift: {data.get('lift_height', 0):.2f} mm")
        print(f"{'-'*40}")
        print(f"BRICK:")
        print(f"  Visible: {b.get('visible', False)}")
        if b.get('visible'):
            print(f"  Distance: {b.get('dist', 0):.2f} mm")
            print(f"  Angle: {b.get('angle', 0):.2f}°")
            print(f"  Offset: {b.get('offset_x', 0):.2f} mm")
            print(f"  Confidence: {b.get('confidence', 0):.2f}%")
        print(f"{'-'*40}")
        
        print(f"{'='*40}")

# --- SHARED VISUALIZATION ---
import cv2

def draw_telemetry_overlay(frame, wm: WorldModel, extra_messages=None, reminders=None, gear=None, show_prompt=True):
    """
    Simplified HUD renderer.
    - Merged objective/checklist/status into single-line prompt.
    - Unified CONTROLS section (White).
    - Removed gear logic.
    """
    h, w = frame.shape[:2]
    
    # --- COLORS (BGR) ---
    GREEN = (0, 255, 0)
    WHITE = (255, 255, 255)
    ORANGE = (0, 165, 255)
    YELLOW = (0, 255, 255)
    
    # 0. Center Alignment Line
    cal_offset = 0
    if WORLD_MODEL_BRICK_FILE.exists():
        try:
            with open(WORLD_MODEL_BRICK_FILE, 'r') as f:
                cal_offset = json.load(f).get('calibration', {}).get('camera_center_offset_px', 0)
        except: pass
    cv2.line(frame, (w//2 + cal_offset, 0), (w//2 + cal_offset, h), (60, 60, 60), 1)

    # 1. Background Panel (Left Side)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (220, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # 2. Text Setup
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.38
    thickness = 1 # No bolding, as it gets fuzzy
    x_base = 12
    y_cur = 25
    line_h = 20
    
    def put_line(txt, c=WHITE, s=scale, th=thickness, thickness=None):
        nonlocal y_cur
        if thickness is not None:
            th = thickness
        cv2.putText(frame, txt, (x_base, y_cur), font, s, c, th)
        y_cur += line_h

    # 3. MERGED STATE & PROMPT
    state_label = wm.objective_state.value
    status_label = f" ({wm.attempt_status})" if wm.attempt_status != "NORMAL" else ""
    put_line(f"OBJ: {state_label}{status_label}", GREEN, 0.45, 1) # Objective Header
    
    if show_prompt:
        # Prompts based on attempt status and recording state
        if wm.attempt_status == "NORMAL":
            if not wm.recording_active:
                prompt = f"Press 'f' to BEGIN {state_label} (FAIL version)"
            else:
                prompt = f"Show clean {state_label} (+ Press 'f' when done)"
        elif wm.attempt_status == "FAIL":
            prompt = f"Press 'f' to BEGIN RECOVERY for {state_label}"
        elif wm.attempt_status == "RECOVERY":
            prompt = f"Press 'f' to finish recovery & start SUCCESS demo"
        else:
            prompt = f"Current focus: {state_label}"
        
        # Override with specific override if provided (e.g. from Keyboard Demo)
        if extra_messages:
            if isinstance(extra_messages, list): prompt = extra_messages[-1]
            else: prompt = extra_messages

        put_line(prompt, (0, 255, 255), 0.38, 1)
        y_cur += 10
    else:
        y_cur += 5

    # 4. Reminders
    if reminders:
        put_line("--- REMINDERS ---", WHITE, 0.35, 1)
        if isinstance(reminders, list):
            for msg in reminders:
                put_line(str(msg), WHITE, 0.35, 1)
        else:
            put_line(str(reminders), WHITE, 0.35, 1)
        y_cur += 5

    # 5. Position Info
    put_line("--- BRICK[0] TELEMETRY ---", WHITE, 0.35, 1)
    put_line(f"OFFSET: {wm.brick['offset_x']:.1f} mm", GREEN, 0.38, 1)
    put_line(f"ANGLE:  {wm.brick['angle']:.1f} deg", GREEN, 0.38, 1)
    put_line(f"DIST:   {wm.brick['dist']:.0f} mm", GREEN, 0.38, 1)
    above_txt = "YES" if wm.brick.get("brickAbove") else "NO"
    below_txt = "YES" if wm.brick.get("brickBelow") else "NO"
    put_line(f"ABOVE:  {above_txt}", GREEN, 0.38, 1)
    put_line(f"BELOW:  {below_txt}", GREEN, 0.38, 1)
    
    y_cur += 5
    put_line("--- LEIA TELEMETRY ---", WHITE, 0.35, 1)
    put_line(f"X:      {wm.x:.1f} mm", (200, 200, 255), 0.38, 1)
    put_line(f"Y:      {wm.y:.1f} mm", (200, 200, 255), 0.38, 1)
    put_line(f"THETA:  {wm.theta:.1f} deg", (200, 200, 255), 0.38, 1)
    put_line(f"LIFT:   {wm.lift_height:.0f} mm", (200, 200, 255), 0.38, 1)
    
    # 6. CONTROLS (Moved up below telemetry)
    y_cur += 10
    put_line("--- CONTROLS ---", WHITE, 0.35, 1)
    put_line("W/S: DRIVE (Fwd/Bwd)", WHITE, 0.35, 1)
    put_line("A/D: TURN (Left/Right, Slow)", WHITE, 0.35, 1)
    put_line("Z/C: TURN (Left/Right, Fast)", WHITE, 0.35, 1)
    put_line("P/L: MAST (Down/Up)", WHITE, 0.35, 1)
    put_line("F: NEXT ACTION Cycle", WHITE, 0.35, 1)
    put_line("Q: QUIT", WHITE, 0.35, 1)

    # 7. Vision Info
    y_cur += 15
    vis_txt = "VISION: LOCKED" if wm.brick['visible'] else "VISION: SEARCHING"
    vis_col = ORANGE if wm.brick['visible'] else (0, 0, 255)
    put_line(vis_txt, vis_col, 0.38, 1)
    
    # 8. Verification Progress
    if wm.verification_stage != "IDLE":
        put_line(f"VERIFY: {wm.verification_stage}", YELLOW, thickness=1)

    y_cur += 8 # Spacer

    # 9. Extra Messages (Banners -> Moved to Sidebar)
    if extra_messages:
        y_cur = h - 20
        for msg in extra_messages:
             put_line(f"! {msg}", (0, 0, 255), 0.4, 2)

    # 10. GEAR Display
    if gear:
        cv2.putText(frame, f"GEAR: {gear}", (x_base, h - 35), font, 0.4, YELLOW, 2)
