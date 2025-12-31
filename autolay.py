#!/usr/bin/env python3
import json
import os
import sys
import time
import threading
import cv2
from flask import Flask, Response

from robot_control import Robot
from brick_vision import BrickDetector
from leia_telemetry import WorldModel, TelemetryLogger, MotionEvent, ObjectiveState, draw_telemetry_overlay

# --- CONFIG ---
GEAR_1_SPEED = 0.32
WEB_PORT = 5001  # Different port to avoid conflict if both run
HEARTBEAT_RATE = 20 # Hz for internal loop

class AutoplayState:
    def __init__(self):
        self.running = True
        self.lock = threading.Lock()
        self.current_frame = None
        self.world = WorldModel()
        self.robot = None
        self.vision = None
        self.active_objective = "UNKNOWN"
        self.status_msg = "Initializing..."

app_state = AutoplayState()
flask_app = Flask(__name__)

def generate_frames():
    while True:
        with app_state.lock:
            if app_state.current_frame is None:
                frame_to_send = None
            else:
                frame_to_send = app_state.current_frame.copy()
        
        if frame_to_send is None:
            time.sleep(0.05)
            continue

        (flag, encodedImage) = cv2.imencode(".jpg", frame_to_send)
        if not flag: continue
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')

@flask_app.route("/")
def index():
    return "<html><body style='background:#111; color:#eee; font-family:sans-serif; text-align:center;'><h1>Robot Eyes (AUTOPILOT)</h1><img src='/video_feed' style='border:2px solid #555; border-radius:10px;'></body></html>"

@flask_app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

def vision_thread():
    app_state.vision = BrickDetector(debug=True, speed_optimize=False)
    while app_state.running:
        found, angle, dist, offset_x, max_y = app_state.vision.read()
        conf = 100 if found else 0
        
        with app_state.lock:
            app_state.world.update_vision(found, dist, angle, conf, offset_x, max_y)
            
            # Draw HUD
            frame = app_state.vision.current_frame.copy()
            messages = [f"AUTO: {app_state.active_objective}", app_state.status_msg]
            reminders = ["Ctrl+C to ABORT"]
            draw_telemetry_overlay(frame, app_state.world, messages, reminders, gear=1)
            app_state.current_frame = frame

def load_demo(session_name):
    path = f"demos/{session_name}/a_log.json"
    if not os.path.exists(path):
        path = f"demos/{session_name}/log.json"
    
    if not os.path.exists(path):
        print(f"Error: Session {session_name} not found.")
        return None

    events = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line in ["[", "]"]: continue
            if line.endswith(","): line = line[:-1]
            data = json.loads(line)
            
            # We only care about entries that had a command
            if data.get('last_event'):
                events.append({
                    'obj': data.get('objective'),
                    'cmd': data['last_event']['type'],
                    'duration': data['last_event']['duration_ms'] / 1000.0
                })
    return events

def main_autoplay(session_name):
    events = load_demo(session_name)
    if not events: return

    app_state.robot = Robot()
    
    # 1. Start Vision/Web Server
    threading.Thread(target=vision_thread, daemon=True).start()
    threading.Thread(target=lambda: flask_app.run(host='0.0.0.0', port=WEB_PORT, debug=False, use_reloader=False), daemon=True).start()
    
    print(f"\n[AUTOPLAY] Loaded {len(events)} events from {session_name}")
    print(f"[AUTOPLAY] Safety Check: Defaulting all speeds to GEAR 1 ({GEAR_1_SPEED})")
    print(f"[AUTOPLAY] Web Stream: http://<robot-ip>:{WEB_PORT}")
    time.sleep(2)

    # Group by Objective to show progress
    current_obj = None
    for i, ev in enumerate(events):
        if not app_state.running: break
        
        if ev['obj'] != current_obj:
            current_obj = ev['obj']
            with app_state.lock:
                app_state.active_objective = current_obj
                app_state.world.objective_state = ObjectiveState[current_obj]
            print(f"\n>>> objective: {current_obj}")

        cmd = ev['cmd']
        # Special handling for mast boost (4x as previously requested)
        speed = GEAR_1_SPEED
        if cmd in ('u', 'd'):
            speed = min(1.0, speed * 4.0)
            
        duration = ev['duration']
        
        with app_state.lock:
            app_state.status_msg = f"Executing: {cmd} ({duration:.2f}s)"
        
        print(f"  [{i+1}/{len(events)}] {cmd} for {duration:.2f}s")
        app_state.robot.send_command(cmd, speed)
        
        # Log this motion in our active world model for HUD/Telemetry
        m_evt = MotionEvent(cmd, int(speed*255), int(duration*1000))
        app_state.world.update_from_motion(m_evt)
        
        time.sleep(duration)
        app_state.robot.stop()

    print("\n[AUTOPLAY] Sequence Complete.")
    app_state.status_msg = "SEQUENCE COMPLETE"
    app_state.running = False
    time.sleep(2)

def find_sessions():
    demos_dir = os.path.join(os.getcwd(), "demos")
    if not os.path.exists(demos_dir):
        return []
    sessions = [d for d in os.listdir(demos_dir) if os.path.isdir(os.path.join(demos_dir, d))]
    sessions.sort(reverse=True)
    return sessions

if __name__ == "__main__":
    session = None
    if len(sys.argv) >= 2:
        session = sys.argv[1]
    else:
        # Try to find the latest
        available = find_sessions()
        if available:
            session = available[0]
            print(f"[AUTOPLAY] No session specified. Defaulting to latest: {session}")
        else:
            print("Error: No sessions found in demos/ folder.")
            sys.exit(1)
    
    try:
        main_autoplay(session)
    except KeyboardInterrupt:
        print("\n[AUTOPLAY] ABORTED by user.")
        if app_state.robot: app_state.robot.stop()
        app_state.running = False
