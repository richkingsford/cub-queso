"""
# robot_leia_telemetry.py
-----------------
Handles the World Model and Logging for Robot Leia.
"""
import time
import json
import math
import os
import threading
from enum import Enum

class ObjectiveState(Enum):
    FIND = "FIND"
    SCOOP = "SCOOP"
    LIFT = "LIFT"
    PLACE = "PLACE"

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

from pathlib import Path
WORLD_MODEL_FILE = Path(__file__).parent / "world_model.json"

class WorldModel:
    def __init__(self):
        # Load Rules
        self.rules = {}
        if WORLD_MODEL_FILE.exists():
            try:
                with open(WORLD_MODEL_FILE, 'r') as f:
                    self.rules = json.load(f).get('objectives', {})
            except: pass
            
        self.learned_rules = {} # Rules derived from demo analysis
            
        # Robot Pose (Dead Reckoning)
        self.x = 0.0 # mm
        self.y = 0.0 # mm
        self.theta = 0.0 # degrees

        # Wall Origin (set on first valid brick)
        self.wall_origin = None # {'x':, 'y':, 'theta':}

        # Brick Data
        self.brick = {
            "visible": False,
            "id": None,
            "dist": 0,
            "angle": 0,
            "offset_x": 0,
            "confidence": 0,
            "held": False,
            "seated": False
        }

        # Forklift
        self.lift_height = 0.0 # mm (estimated)

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
        self.stability_count = 0
        self.stability_threshold = 10  # 10 frames @ 20Hz = 0.5 seconds
        
        self.last_dist = 999.0 # Track last distance for seated heuristic
        
        # Helper for Event Tracking
        self.last_event = None
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
        # print(f"[WORLD] Objective changed to {value}, timer reset.", flush=True)

    def update_from_motion(self, event):
        """
        Updates pose based on motion events (Dead Reckoning).
        Also manages "Wiggle Verification" state machine.
        """
        self.last_event = event # Track last event
        
        dt = event.duration_ms / 1000.0
        power_ratio = event.power / 255.0 
        dist_pulse = 0.0
        rot_pulse = 0.0
        
        if event.action_type == "forward":
            dist_pulse = self.mm_per_sec_full_speed * power_ratio * dt
            rad = math.radians(self.theta)
            self.x += dist_pulse * math.cos(rad)
            self.y += dist_pulse * math.sin(rad)
            
        elif event.action_type == "backward":
            dist_pulse = self.mm_per_sec_full_speed * power_ratio * dt
            rad = math.radians(self.theta)
            self.x -= dist_pulse * math.cos(rad)
            self.y -= dist_pulse * math.sin(rad)
            
            if self.verification_stage == "BACK":
                self.verify_dist_mm += dist_pulse
                if self.verify_dist_mm >= 100:
                    self.verification_stage = "LEFT"
            
        elif event.action_type == "left_turn":
            rot_pulse = self.deg_per_sec_full_speed * power_ratio * dt
            self.theta += rot_pulse
            
            if self.verification_stage == "LEFT":
                self.verify_turn_deg += rot_pulse
                if self.verify_turn_deg >= 20: # ~50mm wiggle
                    self.verification_stage = "RIGHT"
                    self.verify_turn_deg = 0 # Reset for next side
            
        elif event.action_type == "right_turn":
            rot_pulse = self.deg_per_sec_full_speed * power_ratio * dt
            self.theta -= rot_pulse
            
            if self.verification_stage == "RIGHT":
                self.verify_turn_deg += rot_pulse
                if self.verify_turn_deg >= 20:
                    # VERIFICATION COMPLETE
                    seen = self.verify_vision_hits > 3 # Buffer for noise
                    
                    if self.objective_state == ObjectiveState.SCOOP:
                        # SUCCESS if NOT seen
                        self.brick["seated"] = not seen
                    elif self.objective_state == ObjectiveState.PLACE:
                        # SUCCESS if IS seen (left behind)
                        if seen: 
                            self.brick["seated"] = False
                            self.brick["held"] = False
                    
                    self.verification_stage = "IDLE"
                    self.verify_dist_mm = 0.0
                    self.verify_turn_deg = 0.0
                    self.verify_vision_hits = 0

        elif event.action_type == "mast_up":
            self.lift_height += self.lift_mm_per_sec * power_ratio * dt

        elif event.action_type == "mast_down":
            self.lift_height -= self.lift_mm_per_sec * power_ratio * dt
            if self.lift_height < 0: self.lift_height = 0

    def update_vision(self, found, dist, angle, conf, offset_x=0, cam_h=0):
        self.brick["visible"] = bool(found)
        self.brick["dist"] = float(dist)
        self.brick["angle"] = float(angle)
        self.brick["confidence"] = float(conf)
        self.brick["offset_x"] = float(offset_x)
        
        # --- LIFT HEIGHT FUSION ---
        # If we have a high-confidence vision height (PnP lock), use it to anchor and refine
        if found and conf > 50 and cam_h > 0:
            if self.lift_height_anchor is None:
                # Capture the anchor: VisionHeight - DeadReckonHeight
                # This assumes dead reckoning is 0 at startup
                self.lift_height_anchor = cam_h - self.lift_height
                print(f"[WORLD] Lift Anchor Set: {self.lift_height_anchor:.1f}mm")
            
            # Estimated Lift = Vision Altitude - Baseline Altitude
            vis_lift = cam_h - self.lift_height_anchor
            
            # FUSE: 90% Dead Reckon, 10% Vision (Smooth pull)
            # This keeps the motion fluid but corrects drift/scale errors
            self.lift_height = (0.9 * self.lift_height) + (0.1 * vis_lift)
        
        # Intelligent Alignment Check
        was_aligned = self.is_aligned()
        
        # --- PERFECT ALIGNMENT (DATA DRIVEN) ---
        self.brick["perfect_align"] = False
        if found and conf >= 25:
            # Use learned thresholds if available, otherwise defaults
            align_rules = self.learned_rules.get("ALIGN", {})
            tol_off = align_rules.get("max_offset_x", self.align_tol_offset)
            tol_ang = align_rules.get("max_angle", self.align_tol_angle)
            
            # Add a small buffer (10%) to the learned threshold for robustness
            tol_off *= 1.1 
            tol_ang *= 1.1
            
            angle_ok = abs(angle) <= tol_ang
            offset_ok = abs(offset_x) <= tol_off
            dist_ok = self.align_tol_dist_min <= dist <= self.align_tol_dist_max
            
            if angle_ok and offset_ok and dist_ok:
                self.stability_count += 1
                # Final gate: strict stability or ultra-centered
                if self.stability_count >= self.stability_threshold:
                    self.brick["perfect_align"] = True
            else:
                self.stability_count = 0
            
            self.last_dist = dist
        else:
            # SEATED HEURISTIC (Vision Lost)
            if was_aligned and self.last_dist < 100 and self.objective_state == ObjectiveState.SCOOP:
                 self.brick["seated"] = True
                 if self.verification_stage == "IDLE":
                     self.verification_stage = "BACK"

            self.stability_count = 0
            
        # Track Vision Hits during Wiggle Verification
        if found and conf > 40:
            if self.verification_stage in ["BACK", "LEFT", "RIGHT"]:
                self.verify_vision_hits += 1
        
        # Set wall origin if not set and we see a good brick
        if found and self.wall_origin is None and conf > 80:
            # Simple assumption: The first brick we see is the origin
            # In reality, you'd calculate this based on robot pose + brick relative pose
            self.wall_origin = {
                'x': self.x + (dist * math.cos(math.radians(self.theta + angle))),
                'y': self.y + (dist * math.sin(math.radians(self.theta + angle))),
                'theta': 0 # Align wall to world 0 for now
            }

    def is_aligned(self):
        """Returns True if metrics have been stable and centered."""
        return self.stability_count >= self.stability_threshold

    def check_objective_complete(self):
        """Checks if success criteria are met using ONLY learned rules from demos."""
        obj_name = self.objective_state.value
        learned = self.learned_rules.get(obj_name, {})
        
        if not learned:
            # If we don't have learned data for this objective (e.g. PLACE at the very end)
            # just return False to let it run its recorded duration.
            return False
            
        # 1. Logic-based Criteria
        if obj_name == "ALIGN":
            # ALIGN is complete when the data-driven 'perfect_align' flag is set
            if self.brick.get("perfect_align"):
                return True
        
        elif "final_visibility" in learned:
            # For FIND / SCOOP, check if we've reached the desired visibility state
            if self.brick["visible"] == learned["final_visibility"]:
                return True

        return False

    def next_objective(self):
        """Cycles through the 4-step mission: FIND -> SCOOP -> LIFT -> PLACE"""
        if self.objective_state == ObjectiveState.FIND:
            self.objective_state = ObjectiveState.SCOOP
        elif self.objective_state == ObjectiveState.SCOOP:
            self.objective_state = ObjectiveState.LIFT
        elif self.objective_state == ObjectiveState.LIFT:
            self.objective_state = ObjectiveState.PLACE
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
        return self.objective_state.value

    def to_dict(self):
        # Format Brick Data
        brick_fmt = self.brick.copy()
        brick_fmt['dist'] = round(brick_fmt['dist'], 2)
        brick_fmt['angle'] = round(brick_fmt['angle'], 3)
        brick_fmt['offset_x'] = round(brick_fmt['offset_x'], 2)
        brick_fmt['confidence'] = int(brick_fmt['confidence'])

        # Format Wall Origin
        wall_fmt = None
        if self.wall_origin:
            wall_fmt = {
                'x': round(self.wall_origin['x'], 2),
                'y': round(self.wall_origin['y'], 2),
                'theta': round(self.wall_origin['theta'], 3)
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
        """Deprecated: Use log_keyframe or rely on log_state for continuous motion."""
        # For now, we'll map motion events to keyframes if they are semantic
        semantic_events = ['FAIL', 'RECOVERY_START', 'OBJECTIVE_SUCCESS', 'JOB_SUCCESS', 'JOB_START']
        if event.action_type in semantic_events:
            self.log_keyframe(event.action_type, objective, event.timestamp)
        else:
            # Low-level motion is already captured in the high-frequency state logs
            pass

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
        evt = data.get('last_event')
        
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
        
        if evt:
            age = data.get('timestamp', 0) - evt.get('timestamp', 0)
            print(f"LAST EVENT: {evt.get('type', 'unknown')}")
            print(f"  Power: {evt.get('power', 0)}")
            print(f"  Duration: {evt.get('duration_ms', 0):.2f}ms")
            print(f"  Age: {age:.2f}s")
        else:
            print(f"LAST EVENT: None")
        print(f"{'='*40}")

# --- SHARED VISUALIZATION ---
import cv2

def draw_telemetry_overlay(frame, wm: WorldModel, extra_messages=None, reminders=None, gear=None):
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
    
    # 0. Center Alignment Line
    cal_offset = 0
    if WORLD_MODEL_FILE.exists():
        try:
            with open(WORLD_MODEL_FILE, 'r') as f:
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
    
    def put_line(txt, c=WHITE, s=scale, th=thickness):
        nonlocal y_cur
        cv2.putText(frame, txt, (x_base, y_cur), font, s, c, th)
        y_cur += line_h

    # 3. MERGED STATE & PROMPT
    state_label = wm.objective_state.value
    status_label = f" ({wm.attempt_status})" if wm.attempt_status != "NORMAL" else ""
    put_line(f"OBJ: {state_label}{status_label}", GREEN, 0.45, 1) # Objective Header
    
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

    # 4. Position Info
    put_line("--- TELEMETRY ---", WHITE, 0.35, 1)
    put_line(f"OFFSET: {wm.brick['offset_x']:.1f} mm", GREEN, 0.38, 1)
    put_line(f"ANGLE:  {wm.brick['angle']:.1f} deg", GREEN, 0.38, 1)
    put_line(f"DIST:   {wm.brick['dist']:.0f} mm", GREEN, 0.38, 1)
    put_line(f"LIFT:   {wm.lift_height:.0f} mm", GREEN, 0.38, 1)
    
    # 5. CONTROLS (Moved up below telemetry)
    y_cur += 10
    put_line("--- CONTROLS ---", WHITE, 0.35, 1)
    put_line("W/S: DRIVE (Fwd/Bwd)", WHITE, 0.35, 1)
    put_line("A/D: TURN (Left/Right)", WHITE, 0.35, 1)
    put_line("P/L: MAST (Up/Down)", WHITE, 0.35, 1)
    put_line("F: NEXT ACTION Cycle", WHITE, 0.35, 1)
    put_line("Q: QUIT", WHITE, 0.35, 1)

    # 6. Vision Info
    y_cur += 15
    vis_txt = "VISION: LOCKED" if wm.brick['visible'] else "VISION: SEARCHING"
    vis_col = ORANGE if wm.brick['visible'] else (0, 0, 255)
    put_line(vis_txt, vis_col, 0.38, 1)
    
    # 7. Action Tracking
    y_cur += 10
    if wm.last_event:
        put_line(f"ACT: {wm.last_event.action_type}", (255, 0, 255), 0.35, 1)
    else:
        put_line("ACT: IDLE", (100, 100, 100), 0.35, 1)

    # 7b. Verification Progress
    if wm.verification_stage != "IDLE":
        put_line(f"VERIFY: {wm.verification_stage}", YELLOW, thickness=1)

    y_cur += 8 # Spacer

    # 8. Extra Messages (Banners -> Moved to Sidebar)
    if extra_messages:
        y_cur = h - 20
        for msg in extra_messages:
             put_line(f"! {msg}", (0, 0, 255), 0.4, 2)

    # 9. GEAR Display
    if gear:
        cv2.putText(frame, f"GEAR: {gear}", (x_base, h - 35), font, 0.4, YELLOW, 2)
