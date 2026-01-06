import argparse
import json
import os
import sys
import time
import threading
import cv2
import numpy as np
from flask import Flask, Response

from robot_control import Robot
from train_brick_vision import BrickDetector
from robot_leia_telemetry import WorldModel, TelemetryLogger, MotionEvent, ObjectiveState, draw_telemetry_overlay

# --- CONFIG ---
GEAR_1_SPEED = 0.32
WEB_PORT = 5000  # Port 5000 to match recording scripts
HEARTBEAT_RATE = 20 # Hz for internal loop
SLOWDOWN_FACTOR = 1.0 # default to real time
SMART_REPLAY = True
STREAM_ENABLED = True # Default, will be updated by args
DEBUG_MODE = False

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
                # Create NO SIGNAL placeholder
                frame_to_send = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame_to_send, "NO SIGNAL", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
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
    consecutive_failures = 0
    while app_state.running:
        # 1. Get the latest sensor data
        found, angle, dist, offset_x, conf, cam_h = app_state.vision.read()
        
        # dist == -1 is our sentinel for "Hardware Read Error"
        if not found and dist == -1:
            consecutive_failures += 1
            print(f"[VISION] WARNING: Camera read failed! ({consecutive_failures}/5)", flush=True)
            if consecutive_failures >= 5:
                # print("[VISION] FATAL: Camera connection lost. Aborting script.", flush=True)
                app_state.running = False
                os._exit(1)
        else:
            consecutive_failures = 0
            
        with app_state.lock:
            # 3. Telemetry Update
            app_state.world.update_vision(found, dist, angle, conf, offset_x, cam_h)
            
            # --- HUD PROCESSING (OPTIONAL) ---
            if STREAM_ENABLED:
                if app_state.vision.current_frame is not None:
                    frame = app_state.vision.current_frame.copy()
                    messages = [f"AUTO: {app_state.active_objective}", app_state.status_msg]
                    reminders = ["Ctrl+C to ABORT"]
                    draw_telemetry_overlay(frame, app_state.world, messages, reminders)
                    app_state.current_frame = frame
        
        time.sleep(0.05)  # ~20Hz update rate

def load_demo(session_name):
    path = f"demos/{session_name}/a_log.json"
    if not os.path.exists(path):
        path = f"demos/{session_name}/log.json"
    
    if not os.path.exists(path):
        print(f"Error: Session {session_name} not found.")
        return None

    raw_events = []
    try:
        with open(path, 'r') as f:
            log_data = json.load(f)
    except json.JSONDecodeError:
        # Fallback: The log might be truncated (missing closing ']') if the recorder crashed.
        try:
            with open(path, 'r') as f:
                content = f.read().strip()
            if not content.endswith(']'):
                # Try to "healing" the JSON by adding the closing bracket
                log_data = json.loads(content + '\n]')
            else:
                raise # Re-raise if it's not a simple truncation
        except Exception as e:
            return None
    
    # Clean up any nulls or invalid entries
    log_data = [x for x in log_data if x is not None and isinstance(x, dict)]

    # --- SUCCESS HEURISTIC EXTRACTION ---
    # Find the vision state at the moment of transition for each objective
    heuristics = {}
    current_obj = None
    last_state_entry = None
    
    for entry in log_data:
        etype = entry.get('type')
        
        if etype == 'keyframe':
            obj_field = entry.get('objective')
            if obj_field:
                current_obj = obj_field
            
            marker = entry.get('marker')
            if marker == 'OBJ_SUCCESS' and current_obj and last_state_entry:
                # --- Success Visibility / Precision ---
                prev_state = last_state_entry.get('brick') or {}
                
                if current_obj not in heuristics:
                    heuristics[current_obj] = {
                        'max_offset_x': 0.0,
                        'max_angle': 0.0,
                        'final_visibility': prev_state.get('visible', True),
                        'max_duration_ms': 0.0,
                        'samples': 0
                    }
                
                h = heuristics[current_obj]
                h['max_offset_x'] = max(h['max_offset_x'], abs(prev_state.get('offset_x', 0)))
                h['max_angle'] = max(h['max_angle'], abs(prev_state.get('angle', 0)))
                if prev_state.get('visible') is not None:
                    h['final_visibility'] = prev_state.get('visible')
                
                h['samples'] += 1
                # Note: Duration logging logic could be restored here if needed by tracking OBJ_START timestamp
                
        elif etype == 'state':
            last_state_entry = entry

    # Final Tuning: If we have heuristics, print them
    print("\n[LEARNED] Objective Success Gates from Demo:", flush=True)
    for obj, h in heuristics.items():
        if h['samples'] > 0:
            print(f"  {obj:6} > visible={str(h['final_visibility']):5} | offset < {h['max_offset_x']:.1f}px | timeout={h['max_duration_ms']/1000.0:.1f}s", flush=True)

    for entry in log_data:
        if entry.get('last_event'):
            raw_events.append({
                'obj': entry.get('objective'),
                'cmd': entry['last_event']['type'],
                'duration': entry['last_event']['duration_ms'] / 1000.0
            })
    
    if not raw_events: return [], {}
    
    merged_events = []
    current_event = raw_events[0].copy()
    for next_event in raw_events[1:]:
        if (next_event['obj'] == current_event['obj'] and next_event['cmd'] == current_event['cmd']):
            current_event['duration'] += next_event['duration']
        else:
            merged_events.append(current_event)
            current_event = next_event.copy()
    merged_events.append(current_event)
    
    return merged_events, heuristics

def event_type_to_cmd(event_type):
    """Convert logged event type to single-character robot command."""
    mapping = {
        'forward': 'f',
        'backward': 'b',
        'left_turn': 'l',
        'right_turn': 'r',
        'mast_up': 'u',
        'mast_down': 'd'
    }
    return mapping.get(event_type, None)

def main_autoplay(session_name):
    events, learned_rules = load_demo(session_name)
    if not events: return

    app_state.robot = Robot()
    app_state.world.learned_rules = learned_rules
    
    # 1. Start Vision/Web Server
    threading.Thread(target=vision_thread, daemon=True).start()
    
    if STREAM_ENABLED:
        threading.Thread(target=lambda: flask_app.run(host='0.0.0.0', port=WEB_PORT, debug=False, use_reloader=False), daemon=True).start()
        print(f"[AUTOPLAY] Web Stream: http://<robot-ip>:{WEB_PORT}", flush=True)
    else:
        print("[AUTOPLAY] Web Stream: DISABLED (--no-stream)", flush=True)
    
    print(f"\n[AUTOPLAY] Loaded {len(events)} events from {session_name}", flush=True)
    print(f"[AUTOPLAY] Safety Check: Defaulting all speeds to GEAR 1 ({GEAR_1_SPEED})", flush=True)
    
    # --- VISION WARMUP ---
    # Wait for the first valid frame or 2 seconds max
    print("[AUTOPLAY] Warming up vision...", end="", flush=True)
    warmup_start = time.time()
    while time.time() - warmup_start < 2.0:
        with app_state.lock:
            if app_state.current_frame is not None:
                print(" OK!", flush=True)
                break
        time.sleep(0.1)
    else:
        print(" (Camera still warming up, proceeding)", flush=True)

    # Group by Objective to show progress
    current_obj = None
    skip_current_objective = False
    
    for i, ev in enumerate(events):
        if not app_state.running: break
        
        # (3s pause removed for rapid replay)
        
        # If we are skipping, stay in skip mode until the next objective appears
        if skip_current_objective and ev['obj'] == current_obj:
            continue
            
        if ev['obj'] != current_obj:
            current_obj = ev['obj']
            skip_current_objective = False # Reset skip flag for new objective
            with app_state.lock:
                app_state.active_objective = current_obj
                app_state.world.objective_state = ObjectiveState[current_obj]
            
            # --- THINK BEFORE MOVING ---
            # Check if this new objective is already done!
            if SMART_REPLAY:
                with app_state.lock:
                    if app_state.world.check_objective_complete():
                        print(f"\n>>> OBJECTIVE: {current_obj} (ALREADY COMPLETE)", flush=True)
                        if DEBUG_MODE:
                            print(f"[DEBUG] Objective satisfied. Pausing for 5s...", flush=True)
                            app_state.robot.stop()
                            time.sleep(5.0)
                        print(f"[SMART] Skipping {current_obj} entirely.", flush=True)
                        skip_current_objective = True
                        continue

            print(f"\n>>> OBJECTIVE: {current_obj}", flush=True)

        event_type = ev['cmd']
        cmd = event_type_to_cmd(event_type)
        
        if cmd is None:
            print(f"  [{i+1}/{len(events)}] Skipping unknown event: {event_type}")
            continue
            
        # Special handling for mast boost (4x as previously requested)
        speed = GEAR_1_SPEED
        if cmd in ('u', 'd'):
            speed = min(1.0, speed * 4.0)
            
        duration = ev['duration'] * SLOWDOWN_FACTOR
        
        with app_state.lock:
            app_state.status_msg = f"Executing: {event_type} ({duration:.2f}s)"
        
        print(f"\n[{current_obj}] > {event_type} ({duration:.2f}s)", flush=True)
        
        # Send commands continuously during the duration
        start_time = time.time()
        command_interval = 0.1
        last_visible_time = time.time()
        
        while True:
            if not app_state.running:
                break
            
            # --- SMART REPLAY CHECK (PRE-EMPTIVE) ---
            if SMART_REPLAY:
                with app_state.lock:
                    if app_state.world.check_objective_complete():
                        print(f"\n[SMART] Objective {current_obj} Criteria Met! Skipping remaining commands.", flush=True)
                        if DEBUG_MODE:
                            print(f"[DEBUG] Objective satisfied. Pausing for 5s...", flush=True)
                            app_state.robot.stop()
                            time.sleep(5.0)
                        skip_current_objective = True
                        break

            elapsed = time.time() - start_time
            if elapsed >= duration:
                break

            # --- INTELLIGENT ALIGNMENT (CLOSED LOOP) ---
            # If in ALIGN, use the World Model to pick the best direction
            active_cmd = cmd
            active_speed = speed
            
            if current_obj == "SCOOP":
                with app_state.lock:
                    off_x = app_state.world.brick.get('offset_x', 0)
                    visible = app_state.world.brick['visible']
                    # Use learned threshold or default
                    align_rules = app_state.world.learned_rules.get("SCOOP", {})
                    tol_off = align_rules.get("max_offset_x", 10.0) * 1.1 # 10% buffer
                
                if visible:
                    # Slow down for precision
                    active_speed = GEAR_1_SPEED * 0.6
                    
                    if off_x > tol_off: # Brick is to the right
                        active_cmd = 'r'
                    elif off_x < -tol_off: # Brick is to the left
                        active_cmd = 'l'
                    else:
                        active_cmd = None # Centered relative to our learned rules!
                        with app_state.lock:
                            app_state.status_msg = f"[CENTERED] (Offset: {off_x:.1f}px) Holding stability..."
            
            if active_cmd:
                app_state.robot.send_command(active_cmd, active_speed)
            else:
                app_state.robot.stop()
            
            # Log this motion in our active world model for HUD/Telemetry
            m_evt = MotionEvent(event_type, int(active_speed*255), int(command_interval*1000))
            app_state.world.update_from_motion(m_evt)
            
            # Sleep for command interval or remaining time, whichever is shorter
            remaining = duration - elapsed
            sleep_time = min(command_interval, remaining)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Stop after the duration
        app_state.robot.stop()

    print("\n[AUTOPLAY] Sequence Complete.")
    app_state.status_msg = "SEQUENCE COMPLETE"
    app_state.running = False
    time.sleep(2)



def find_sessions():
    demos_dir = "demos"
    if not os.path.exists(demos_dir):
        return []
    sessions = [d for d in os.listdir(demos_dir) if os.path.isdir(os.path.join(demos_dir, d))]
    sessions.sort(reverse=True)
    return sessions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Brick Laying Replay")
    parser.add_argument("session", nargs="?", help="Name of the session to replay (e.g. kbd_...)")
    parser.add_argument("--no-stream", action="store_true", help="Disable web stream and HUD processing")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (5s pause after success)")
    args = parser.parse_args()

    STREAM_ENABLED = not args.no_stream
    DEBUG_MODE = args.debug
    
    session = args.session
    if not session:
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
