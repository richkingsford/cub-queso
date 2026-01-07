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
        found, angle, dist, offset_x, conf, cam_h = app_state.vision.read()
        
        if not found and dist == -1:
            consecutive_failures += 1
            if consecutive_failures >= 5:
                app_state.running = False
                os._exit(1)
        else:
            consecutive_failures = 0
            
        with app_state.lock:
            app_state.world.update_vision(found, dist, angle, conf, offset_x, cam_h)
            if STREAM_ENABLED and app_state.vision.current_frame is not None:
                frame = app_state.vision.current_frame.copy()
                messages = [f"AUTO: {app_state.active_objective}", app_state.status_msg]
                draw_telemetry_overlay(frame, app_state.world, messages)
                app_state.current_frame = frame
        
        time.sleep(0.05)

def load_demo(session_name):
    # Support both directory-style (old) and flat-file-style (new)
    path = f"demos/{session_name}"
    if not os.path.exists(path):
        if not session_name.endswith(".json"):
            path = f"demos/{session_name}.json"
        else:
            path = f"demos/{session_name}"

    if os.path.isdir(path):
        base_path = path
        path = os.path.join(base_path, "a_log.json")
        if not os.path.exists(path):
            path = os.path.join(base_path, "log.json")
    
    if not os.path.exists(path):
        print(f"Error: Session {session_name} not found.")
        return None, None

    try:
        with open(path, 'r') as f:
            log_data = json.load(f)
    except:
        # Simple healing for truncated JSON
        try:
            with open(path, 'r') as f:
                content = f.read().strip()
            if not content.endswith(']'):
                log_data = json.loads(content + '\n]')
            else:
                raise 
        except:
            return None, None
    
    log_data = [x for x in log_data if x is not None and isinstance(x, dict)]

    # --- HEURISTIC EXTRACTION ---
    heuristics = {}
    current_obj = None
    last_state_entry = None
    raw_events = []
    
    for entry in log_data:
        etype = entry.get('type')
        if etype == 'keyframe':
            obj_field = entry.get('objective')
            if obj_field: current_obj = obj_field
            marker = entry.get('marker')
            if marker == 'OBJ_SUCCESS' and current_obj and last_state_entry:
                prev_state = last_state_entry.get('brick') or {}
                if current_obj not in heuristics:
                    heuristics[current_obj] = {'max_offset_x': 0.0, 'max_angle': 0.0, 'final_visibility': prev_state.get('visible', True), 'samples': 0}
                h = heuristics[current_obj]
                h['max_offset_x'] = max(h['max_offset_x'], abs(prev_state.get('offset_x', 0)))
                h['max_angle'] = max(h['max_angle'], abs(prev_state.get('angle', 0)))
                if prev_state.get('visible') is not None: h['final_visibility'] = prev_state.get('visible')
                h['samples'] += 1
        elif etype == 'state':
            last_state_entry = entry
            if entry.get('last_event'):
                raw_events.append({
                    'obj': current_obj or entry.get('objective'),
                    'cmd': event_type_to_cmd(entry['last_event']['type']),
                    'duration': entry['last_event']['duration_ms'] / 1000.0
                })

    if heuristics:
        print("\n[LEARNED] Success Gates from Demo:")
        for obj, h in heuristics.items():
            print(f"  {obj:<8} | OffX: {h['max_offset_x']:4.1f} | Ang: {h['max_angle']:4.1f} | Vis: {h['final_visibility']} | n={h['samples']}")
    
    if not raw_events: return [], heuristics
    
    merged_events = []
    curr = raw_events[0].copy()
    for nxt in raw_events[1:]:
        if (nxt.get('obj') == curr.get('obj') and nxt['cmd'] == curr['cmd']):
            curr['duration'] += nxt['duration']
        else:
            merged_events.append(curr)
            curr = nxt.copy()
    merged_events.append(curr)
    
    return merged_events, heuristics

def event_type_to_cmd(event_type):
    mapping = {
        'forward': 'f', 'backward': 'b', 'left_turn': 'l', 'right_turn': 'r', 
        'mast_up': 'u', 'mast_down': 'd'
    }
    return mapping.get(event_type, None)

def validate_log_integrity(log_path):
    try:
        with open(log_path, 'r') as f:
            log_data = json.load(f)
    except:
        return False, {}, ["Failed to parse JSON"]

    stack = []
    errors = []
    stats = {}
    for entry in log_data:
        if entry.get('type') != 'keyframe': continue
        marker = entry.get('marker', '')
        obj = entry.get('objective', 'GLOBAL')
        if obj not in stats:
            stats[obj] = {"FAIL": 0, "RECOVER": 0, "COMPLETE": 0}
        
        if marker.endswith("_START"):
            stack.append((marker.replace("_START", ""), obj))
        elif marker.endswith("_END"):
            m_type = marker.replace("_END", "")
            if not stack:
                errors.append(f"Unexpected {marker}")
            else:
                top_type, top_obj = stack.pop()
                if top_type != m_type:
                    # Put it back and report error
                    stack.append((top_type, top_obj))
                    errors.append(f"Mismatched: {top_type} vs {m_type}")
                else:
                    if m_type in stats[obj]: stats[obj][m_type] += 1
        elif marker == "OBJ_SUCCESS":
             stats[obj]["COMPLETE"] += 1
             # Pop any OBJ or OBJ related starts from stack
             while stack and stack[-1][0] in ["OBJ", "FAIL", "RECOVER", "SUCCESS"]:
                 stack.pop()
        elif marker == "JOB_SUCCESS":
             # Pop everything until the JOB marker
             while stack:
                 top_type, _ = stack.pop()
                 if top_type == "JOB":
                     break
             
    for m_type, m_obj in stack:
        if m_type != "JOB": # Ignore trailing JOB starts (we auto-start the next run)
            errors.append(f"Unclosed {m_type}")
        
    return len(errors) == 0, stats, errors

def summarize_all_demos():
    sessions = find_sessions()
    if not sessions:
        print("[STATS] No demos found.")
        return

    print(f"\n{'='*90}")
    print(f"{'DEMO SESSION':<35} | {'OBJ':<8} | {'FAIL':<5} | {'RECV':<5} | {'DONE':<5} | {'STATUS'}")
    print(f"{'-'*90}")

    for s in sessions:
        path = f"demos/{s}"
        if os.path.isdir(path):
             p = os.path.join(path, "a_log.json")
             if not os.path.exists(p): p = os.path.join(path, "log.json")
             path = p
        elif not path.endswith(".json"):
             path += ".json"
        
        if not os.path.exists(path): continue
        ok, stats, _ = validate_log_integrity(path)
        if not stats: continue

        first = True
        for obj, sdata in stats.items():
            if obj == "GLOBAL": continue
            prefix = f"{s[:33]:<35}" if first else " " * 35
            status = "PASS" if ok else "FAIL"
            if not first: status = ""
            print(f"{prefix} | {obj:<8} | {sdata.get('FAIL',0):<5} | {sdata.get('RECOVER',0):<5} | {sdata.get('COMPLETE',0):<5} | {status}")
            first = False
    print(f"{'='*90}\n")

def find_sessions():
    demos_dir = "demos"
    if not os.path.exists(demos_dir): return []
    items = os.listdir(demos_dir)
    sessions = [f for f in items if f.endswith(".json") or os.path.isdir(os.path.join(demos_dir, f))]
    sessions.sort(reverse=True)
    return sessions

def main_autoplay(session_name):
    events, learned_rules = load_demo(session_name)
    if not events: return

    app_state.robot = Robot()
    app_state.world.learned_rules = learned_rules
    
    threading.Thread(target=vision_thread, daemon=True).start()
    if STREAM_ENABLED:
        threading.Thread(target=lambda: flask_app.run(host='0.0.0.0', port=WEB_PORT, debug=False, use_reloader=False), daemon=True).start()
    
    print(f"[AUTOPLAY] Loaded {len(events)} events. Warming up vision...")
    warmup_start = time.time()
    while time.time() - warmup_start < 2.0:
        with app_state.lock:
            if app_state.current_frame is not None: break
        time.sleep(0.1)

    current_obj = None
    skip_mode = False
    for i, ev in enumerate(events):
        if not app_state.running: break
        if skip_mode and ev.get('obj') == current_obj: continue
            
        if ev.get('obj') != current_obj:
            current_obj = ev.get('obj')
            skip_mode = False
            if current_obj in ObjectiveState.__members__:
                with app_state.lock:
                    app_state.active_objective = current_obj
                    app_state.world.objective_state = ObjectiveState[current_obj]
            
            if SMART_REPLAY:
                with app_state.lock:
                    if app_state.world.check_objective_complete():
                        print(f"\n>>> {current_obj} (ALREADY COMPLETE)")
                        if DEBUG_MODE: time.sleep(5.0)
                        skip_mode = True
                        continue
            print(f"\n>>> {current_obj}")

        cmd = ev['cmd']
        duration = ev['duration'] * SLOWDOWN_FACTOR
        print(f"  [{i+1}/{len(events)}] {ev['obj']} > {cmd} for {duration:.2f}s")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            if not app_state.running: break
            if SMART_REPLAY:
                with app_state.lock:
                    if app_state.world.check_objective_complete():
                        print(f"  [SMART] {current_obj} Criteria Met!")
                        if DEBUG_MODE: time.sleep(5.0)
                        skip_mode = True
                        break
            
            speed = GEAR_1_SPEED
            if cmd in ('u', 'd'): speed = min(1.0, speed * 4.0)
            app_state.robot.move(cmd, speed)
            time.sleep(0.05)
        app_state.robot.stop()

    print("\n[AUTOPLAY] Finished.")
    app_state.running = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("session", nargs="?")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    STREAM_ENABLED = not args.no_stream
    DEBUG_MODE = args.debug
    
    summarize_all_demos()
    
    session = args.session or (find_sessions()[0] if find_sessions() else None)
    if not session:
        print("No sessions found.")
        sys.exit(1)
    
    try:
        main_autoplay(session)
    except KeyboardInterrupt:
        if app_state.robot: app_state.robot.stop()
        app_state.running = False
