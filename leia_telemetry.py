"""
leia_telemetry.py
-----------------
Handles the World Model and Logging for Robot Leia.
"""
import time
import json
import math
import os
from enum import Enum

class ObjectiveState(Enum):
    FIND = "FIND"
    ALIGN = "ALIGN"
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
            "timestamp": self.timestamp
        }

class WorldModel:
    def __init__(self):
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
            "seated": False,
            "perfect_align": False
        }

        # Forklift
        self.lift_height = 0.0 # mm (estimated)

        # Objective
        self.objective_state = ObjectiveState.FIND
        
        # Alignment & Stability
        self.align_tol_angle = 5.0    # +/- Degrees
        self.align_tol_offset = 12.0  # +/- mm
        self.align_tol_dist_min = 30.0 # mm (Too close)
        self.align_tol_dist_max = 500.0 # mm (Too far)
        self.stability_count = 0
        self.stability_threshold = 5  # Frames required for a "lock"
        
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
        self.lift_mm_per_sec = 20.0

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

    def update_vision(self, found, dist, angle, conf, offset_x=0, max_y=0):
        self.brick["visible"] = found
        self.brick["dist"] = dist
        self.brick["angle"] = angle
        self.brick["confidence"] = conf
        self.brick["offset_x"] = offset_x
        self.brick["max_y"] = max_y
        
        # Intelligent Alignment Check
        was_aligned = self.is_aligned()
        
        # Perfect Alignment Hint: Angle ~ 0, Offset ~ 0, Notch at bottom of frame
        self.brick["perfect_align"] = False
        if found and conf >= 80:
            angle_ok = abs(angle) <= self.align_tol_angle
            offset_ok = abs(offset_x) <= self.align_tol_offset
            dist_ok = self.align_tol_dist_min <= dist <= self.align_tol_dist_max
            
            # Additional hint: Notch near bottom (assuming 480px height)
            depth_ok = max_y > 450
            
            if angle_ok and offset_ok and dist_ok:
                self.stability_count += 1
                if abs(angle) < 1.5 and abs(offset_x) < 5.0 and depth_ok:
                    self.brick["perfect_align"] = True
            else:
                self.stability_count = 0
            
            self.last_dist = dist
        else:
            # SEATED HEURISTIC (Vision Lost)
            if was_aligned and self.last_dist < 100 and self.objective_state == ObjectiveState.SCOOP:
                 self.brick["seated"] = True
                 # Trigger Wiggle Verification automatically?
                 # No, user manually wiggles, we just track it.
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

    def next_objective(self):
        """Cycles through the 5-step mission: FIND -> ALIGN -> SCOOP -> LIFT -> PLACE"""
        if self.objective_state == ObjectiveState.FIND:
            self.objective_state = ObjectiveState.ALIGN
        elif self.objective_state == ObjectiveState.ALIGN:
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

    def reset_mission(self):
        """Resets the objective state and all mission-specific flags."""
        self.objective_state = ObjectiveState.FIND
        self.brick["seated"] = False
        self.brick["held"] = False
        self.brick["perfect_align"] = False
        self.stability_count = 0
        self.verification_stage = "IDLE"
        self.verify_dist_mm = 0.0
        self.verify_turn_deg = 0.0
        self.verify_vision_hits = 0
        return self.objective_state.value

    def to_dict(self):
        last_evt_dict = self.last_event.to_dict() if self.last_event else None
        return {
            "timestamp": time.time(),
            "robot_pose": {"x": self.x, "y": self.y, "theta": self.theta},
            "wall_origin": self.wall_origin,
            "brick": self.brick,
            "lift_height": self.lift_height,
            "objective": self.objective_state.value,
            "last_event": self.last_event.to_dict() if self.last_event else None,
            "image_file": self.last_image_file
        }

class TelemetryLogger:
    def __init__(self, filename="leia_log.json"):
        self.filename = filename
        # Clear old log
        with open(self.filename, 'w') as f:
            f.write("[\n") # Start JSON array
        self.first_entry = True

    def log_state(self, world_model: WorldModel):
        data = world_model.to_dict()
        
        # 1. Terminal Output (Human Readable)
        self._print_terminal(data)
        
        # 2. JSON Log
        with open(self.filename, 'a') as f:
            if not self.first_entry:
                f.write(",\n")
            json.dump(data, f)
            self.first_entry = False

    def log_event(self, event: MotionEvent):
        # We also append events to the stream or a separate file?
        # User asked for "State logs (continuous)" and "Motion events".
        # Let's embed motion events in the continuous log or strictly continuous snapshot?
        # User said "two aligned logging streams".
        # For simplicity, I'll log motion events as a special entry in the same JSON list 
        # but with a different 'record_type' field if I could, but strictly the request 
        # implies state snapshots at ticks.
        # I will just log motion events to terminal for now, or add them to the NEXT tick?
        # Let's write them to the JSON as a separate object for now.
        
        event_data = event.to_dict()
        event_data['record_type'] = 'event'
        
        with open(self.filename, 'a') as f:
            if not self.first_entry:
                f.write(",\n")
            json.dump(event_data, f)
            self.first_entry = False

    def _print_terminal(self, data):
        # Clear screen code (optional, maybe too flashy)
        # print("\033[H\033[J", end="") 
        
        p = data['robot_pose']
        b = data['brick']
        wall = "SET" if data['wall_origin'] else "UNSET"
        evt = data.get('last_event')
        
        print(f"| T={data['timestamp']:.2f} | STATE={data['objective']} | WALL={wall}")
        print(f"| POSE: X={p['x']:.0f} Y={p['y']:.0f} H={p['theta']:.0f}° | LIFT={data['lift_height']:.0f}")
        print(f"| BRICK: Vis={b['visible']} D={b['dist']:.0f} A={b['angle']:.0f} Conf={b['confidence']:.0f}%")
        
        if evt:
            age = data['timestamp'] - evt['timestamp']
            print(f"| LAST EVENT: {evt['type']} | Power: {evt['power']} | Dur: {evt['duration_ms']}ms | Age: {age:.1f}s")
        else:
            print(f"| LAST EVENT: None")
            
        print("-" * 60)

    def close(self):
        # Remove trailing comma and add closing bracket
        if os.path.exists(self.filename):
            with open(self.filename, 'rb+') as f:
                f.seek(0, os.SEEK_END)
                pos = f.tell()
                # Search backwards for the last comma
                while pos > 0:
                    pos -= 1
                    f.seek(pos)
                    if f.read(1) == b',':
                        f.seek(pos)
                        f.truncate()
                        break
                f.write(b"\n]")

# --- SHARED VISUALIZATION ---
import cv2

def draw_telemetry_overlay(frame, wm: WorldModel, extra_messages=None, controls=None, gear=None):
    """
    Miniaturized HUD renderer with alignment centerline.
    """
    h, w = frame.shape[:2]
    
    # --- COLORS (BGR) ---
    MAGENTA = (255, 0, 255)
    GREEN = (0, 255, 0)
    YELLOW = (0, 255, 255)
    ORANGE = (0, 165, 255) # Bright Orange
    WHITE = (255, 255, 255)
    
    # 0. Center Alignment Line (Subtle)
    cv2.line(frame, (w//2, 0), (w//2, h), (100, 100, 100), 1)

    # 1. Background Panel (Left Side) - Darker
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (220, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # 2. Text Config
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.35 # ~30% smaller than 0.5
    bold = 1 # Boldness is more sensitive at smaller scales
    x_base = 12
    y_cur = 25
    line_h = 18 # Tighter spacing
    
    def put_line(txt, c=WHITE, s=scale, thickness=bold):
        nonlocal y_cur
        cv2.putText(frame, txt, (x_base, y_cur), font, s, c, thickness)
        y_cur += line_h

    # 3. Objective Dashboard (Top - Magenta)
    put_line("OBJECTIVES:", MAGENTA, 0.4, 1)
    
    objectives = [
        (ObjectiveState.FIND,  "Objective (R): FIND"),
        (ObjectiveState.ALIGN, "Objective (T): ALIGN"),
        (ObjectiveState.SCOOP, "Objective (Y): SCOOP"),
        (ObjectiveState.LIFT,  "Objective (U): LIFT"),
        (ObjectiveState.PLACE, "Objective (I): PLACE")
    ]
    
    for state, label in objectives:
        is_active = (wm.objective_state == state)
        color = MAGENTA if is_active else (80, 80, 80)
        prefix = "> " if is_active else "  "
        put_line(f"{prefix}{label}", color, 0.32, 1)
    
    y_cur += 8 # Spacer
    
    # 4. Pose & Lift (Green)
    put_line(f"X: {wm.x:.1f} mm", GREEN)
    put_line(f"Y: {wm.y:.1f} mm", GREEN)
    put_line(f"H: {wm.theta:.1f} deg", GREEN)
    put_line(f"LIFT: {wm.lift_height:.1f} mm", GREEN)
    
    y_cur += 8 # Spacer
    
    # 5. Vision Data (Orange)
    vis_txt = "VISION: LOCKED" if wm.brick['visible'] else "VISION: SEARCHING"
    vis_col = ORANGE if wm.brick['visible'] else (0, 0, 255)
    put_line(vis_txt, vis_col, 0.35, 1)
    
    if wm.brick['visible']:
        put_line(f" CONF: {wm.brick['confidence']}%", ORANGE)
        put_line(f" DIST: {wm.brick['dist']:.0f} mm", ORANGE)
        put_line(f" ANG: {wm.brick['angle']:.1f} deg", ORANGE)
        put_line(f" OFF: {wm.brick['offset_x']:.1f} mm", ORANGE)
    
    y_cur += 8 # Spacer
    
    # 6. Last Activity (Magenta)
    if wm.last_event:
        evt = wm.last_event
        put_line(f"ACT: {evt.action_type}", MAGENTA)
        put_line(f"PWR: {evt.power} ({evt.duration_ms}ms)", MAGENTA)
    else:
        put_line("ACT: IDLE", MAGENTA)
    
    # 7. Alignment / Seated Banners (Center)
    if wm.brick.get("perfect_align"):
        banner_text = "PERFECT ALIGNMENT"
        tw, th = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)[0]
        bx = (w - tw) // 2
        by = 100
        cv2.rectangle(frame, (bx-15, by-th-15), (bx+tw+15, by+15), (0, 215, 255), -1) # Gold
        cv2.putText(frame, banner_text, (bx, by), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 3)
    elif wm.is_aligned():
        banner_text = "ALIGNED & LOCKED"
        tw, th = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0]
        bx = (w - tw) // 2
        by = 80
        cv2.rectangle(frame, (bx-10, by-th-10), (bx+tw+10, by+10), (0, 150, 0), -1)
        cv2.putText(frame, banner_text, (bx, by), cv2.FONT_HERSHEY_DUPLEX, 1.0, WHITE, 2)

    if wm.brick["seated"]:
        banner_text = "BRICK SEATED"
        tw, th = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0]
        bx = (w - tw) // 2
        if wm.brick.get("perfect_align") or wm.is_aligned(): by = 140
        else: by = 80
        cv2.rectangle(frame, (bx-10, by-th-10), (bx+tw+10, by+10), (0, 0, 150), -1)
        cv2.putText(frame, banner_text, (bx, by), cv2.FONT_HERSHEY_DUPLEX, 1.0, WHITE, 2)

    # 7b. Verification Progress
    if wm.verification_stage != "IDLE":
        v_msg = f"VERIFYING: {wm.verification_stage}..."
        cv2.putText(frame, v_msg, (w-250, h-20), font, 0.5, YELLOW, 2)

    # 8. Extra Messages (Banners)
    if extra_messages:
        y_center = h // 2
        for msg in extra_messages:
            text_size = cv2.getTextSize(msg, font, 1.2, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(frame, msg, (text_x, y_center), font, 1.2, GREEN, 2)
            y_center += 50

    # 9. GEAR & CONTROLS (Yellow - Bottom Left)
    y_cur = h - 15
    if controls:
        for msg in reversed(controls):
            cv2.putText(frame, msg, (x_base, y_cur), font, 0.35, YELLOW, 1)
            y_cur -= 15
