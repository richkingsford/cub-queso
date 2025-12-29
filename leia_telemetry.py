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
    PICK = "PICK"
    CARRY = "CARRY"
    PLACE = "PLACE"
    DONE = "DONE"

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
            "confidence": 0,
            "held": False
        }

        # Forklift
        self.lift_height = 0.0 # mm (estimated)

        # Objective
        self.objective_state = ObjectiveState.FIND
        
        # Helper for Event Tracking
        self.last_event = None
        self.last_image_file = None

        # Internal physics constants for dead reckoning (Calibration needed!)
        self.mm_per_sec_full_speed = 200.0 
        self.deg_per_sec_full_speed = 90.0
        self.lift_mm_per_sec = 20.0

    def update_from_motion(self, event: MotionEvent):
        """
        Updates pose based on motion events (Dead Reckoning).
        """
        self.last_event = event # Track last event
        
        dt = event.duration_ms / 1000.0
        power_ratio = event.power / 255.0 # Assuming 255 is max
        
        if event.action_type == "forward":
            dist = self.mm_per_sec_full_speed * power_ratio * dt
            rad = math.radians(self.theta)
            self.x += dist * math.cos(rad)
            self.y += dist * math.sin(rad)
            
        elif event.action_type == "backward":
            dist = self.mm_per_sec_full_speed * power_ratio * dt
            rad = math.radians(self.theta)
            self.x -= dist * math.cos(rad)
            self.y -= dist * math.sin(rad)
            
        elif event.action_type == "left_turn":
            # In place turn
            rot = self.deg_per_sec_full_speed * power_ratio * dt
            self.theta += rot
            
        elif event.action_type == "right_turn":
            rot = self.deg_per_sec_full_speed * power_ratio * dt
            self.theta -= rot

        elif event.action_type == "mast_up":
            self.lift_height += self.lift_mm_per_sec * power_ratio * dt

        elif event.action_type == "mast_down":
            self.lift_height -= self.lift_mm_per_sec * power_ratio * dt
            if self.lift_height < 0: self.lift_height = 0

    def update_vision(self, found, dist, angle, conf):
        self.brick['visible'] = found
        self.brick['dist'] = dist
        self.brick['angle'] = angle
        self.brick['confidence'] = conf
        
        # Set wall origin if not set and we see a good brick
        if found and self.wall_origin is None and conf > 80:
            # Simple assumption: The first brick we see is the origin
            # In reality, you'd calculate this based on robot pose + brick relative pose
            self.wall_origin = {
                'x': self.x + (dist * math.cos(math.radians(self.theta + angle))),
                'y': self.y + (dist * math.sin(math.radians(self.theta + angle))),
                'theta': 0 # Align wall to world 0 for now
            }

    def to_dict(self):
        last_evt_dict = self.last_event.to_dict() if self.last_event else None
        return {
            "timestamp": time.time(),
            "robot_pose": {"x": self.x, "y": self.y, "theta": self.theta},
            "wall_origin": self.wall_origin,
            "brick": self.brick,
            "lift_height": self.lift_height,
            "objective": self.objective_state.value,
            "last_event": last_evt_dict,
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
        with open(self.filename, 'a') as f:
            f.write("\n]") # Close JSON array

# --- SHARED VISUALIZATION ---
import cv2

def draw_telemetry_overlay(frame, wm: WorldModel, extra_messages=None):
    """
    Shared HUD renderer.
    wm: WorldModel instance
    extra_messages: list of strings to display prominently (e.g. ["JOB STARTED"])
    """
    h, w = frame.shape[:2]
    
    # 1. Background Panel for Telemetry
    overlay = frame.copy()
    cv2.rectangle(overlay, (w-250, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # 2. Text Config
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (200, 200, 200)
    x_base = w - 240
    y_cur = 30
    line_h = 25
    
    def put_line(txt, c=color):
        nonlocal y_cur
        cv2.putText(frame, txt, (x_base, y_cur), font, scale, c, 1)
        y_cur += line_h

    # 3. Print Standard Data
    put_line(f"OBJ: {wm.objective_state.value}", (255, 200, 0))
    put_line("")
    put_line(f"POSE X: {wm.x:.1f}")
    put_line(f"POSE Y: {wm.y:.1f}")
    put_line(f"HEAD: {wm.theta:.1f} deg")
    put_line(f"LIFT: {wm.lift_height:.1f} mm")
    put_line("")
    
    put_line("--- VISION ---")
    if wm.brick['visible']:
        put_line(f"DIST: {wm.brick['dist']:.0f} mm", (0, 255, 0))
        put_line(f"ANG: {wm.brick['angle']:.1f}", (0, 255, 0))
    else:
        put_line("SEARCHING...", (0, 0, 255))
        
    # 4. Last Event
    if wm.last_event:
        put_line("")
        evt = wm.last_event
        put_line(f"ACT: {evt.action_type}", (255, 100, 255))
        put_line(f"PWR: {evt.power} ({evt.duration_ms}ms)")

    # 5. Wall Origin
    if wm.wall_origin:
        put_line("")
        put_line("WALL SET", (0, 255, 255))
        
    # 6. Extra Messages (Banners)
    if extra_messages:
        # Draw large centered text
        y_center = h // 2
        for msg in extra_messages:
            text_size = cv2.getTextSize(msg, font, 1.5, 3)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(frame, msg, (text_x, y_center), font, 1.5, (0, 255, 0), 3)
            y_center += 60
