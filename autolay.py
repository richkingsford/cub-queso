import argparse
import json
import os
import sys
import time
import threading
import cv2
import numpy as np
from flask import Flask, Response
from pathlib import Path

from robot_control import Robot
from train_brick_vision import BrickDetector
from robot_leia_telemetry import WorldModel, TelemetryLogger, MotionEvent, ObjectiveState, draw_telemetry_overlay

# --- CONFIG ---
GEAR_1_SPEED = 1.0  # Full speed instead of 0.32
WEB_PORT = 5000  # Port 5000 to match recording scripts
HEARTBEAT_RATE = 20 # Hz for internal loop
SLOWDOWN_FACTOR = 1.0 # default to real time
SMART_REPLAY = True
STREAM_ENABLED = True # Default, will be updated by args
DEBUG_MODE = False
DEMOS_DIR = Path(__file__).resolve().parent / "demos"

# Continuous motion config - never stop moving
SCAN_SPEED = 0.4  # Constant slow scanning speed
BRICK_SPOTTED_SPEED = 0.15  # Even slower when brick is visible  
ALIGNMENT_SPEED = 0.2  # Speed for fine adjustments

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
        
        # Logging
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.demos_dir = DEMOS_DIR
        self.demos_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.demos_dir / f"auto_{timestamp}.json"
        self.logger = TelemetryLogger(log_path)
        print(f"[SESSION] Recording Auto-Log to: {log_path}")

app_state = AutoplayState()
flask_app = Flask(__name__)

def generate_frames():
    while True:
        with app_state.lock:
            if app_state.current_frame is None:
                # Placeholder if no frame yet
                frame_to_send = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame_to_send, "WAITING FOR CAMERA...", (120, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
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
    return """
    <html>
        <head>
            <title>Robot Leia - Autopilot</title>
            <style>
                body { background: #1a1a1a; color: #eee; font-family: sans-serif; text-align: center; margin-top: 50px; }
                .stream-container { display: inline-block; border: 5px solid #333; border-radius: 8px; overflow: hidden; }
                h1 { color: #f0ad4e; }
            </style>
        </head>
        <body>
            <h1>Robot Leia - Autopilot</h1>
            <div class="stream-container">
                <img src="/video_feed" width="800">
            </div>
            <p>Robot is running autonomously. Keep this window open to monitor status.</p>
        </body>
    </html>
    """

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
            
            # Log State (Auto-Log)
            app_state.logger.log_state(app_state.world)

            if STREAM_ENABLED and app_state.vision.current_frame is not None:
                frame = app_state.vision.current_frame.copy()
                messages = [f"AUTO: {app_state.active_objective}", app_state.status_msg]
                draw_telemetry_overlay(frame, app_state.world, messages)
                app_state.current_frame = frame
        
        time.sleep(0.05)

def load_demo(session_name, include_attempts=False):
    # Support both directory-style (old) and flat-file-style (new)
    path = DEMOS_DIR / session_name
    if not path.exists():
        if not session_name.endswith(".json"):
            path = DEMOS_DIR / f"{session_name}.json"
        else:
            path = DEMOS_DIR / session_name

    if path.is_dir():
        base_path = path
        path = base_path / "a_log.json"
        if not path.exists():
            path = base_path / "log.json"
    
    if not path.exists():
        print(f"Error: Session {session_name} not found.")
        return (None, None, None) if include_attempts else (None, None)

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
            return (None, None, None) if include_attempts else (None, None)
    
    log_data = [x for x in log_data if x is not None and isinstance(x, dict)]

    attempts = extract_demo_attempts(log_data) if include_attempts else None

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
        elif etype in ('action', 'event'):
            if etype == 'action':
                cmd_name = entry.get('command')
                power = entry.get('power', 0)
                duration_ms = entry.get('duration_ms')
            else:
                event = entry.get('event') or {}
                cmd_name = event.get('type')
                power = event.get('power', 0)
                duration_ms = event.get('duration_ms')

            event_obj = entry.get('objective')
            if event_obj:
                current_obj = event_obj
            cmd = event_type_to_cmd(cmd_name)
            if cmd and duration_ms:
                raw_events.append({
                    'obj': current_obj,
                    'cmd': cmd,
                    'duration': duration_ms / 1000.0,
                    'speed': (power or 0) / 255.0
                })

    if heuristics:
        print("\n[LEARNED] Success Gates from Demo:")
        for obj, h in heuristics.items():
            print(f"  {obj:<8} | OffX: {h['max_offset_x']:4.1f} | Ang: {h['max_angle']:4.1f} | Vis: {h['final_visibility']} | n={h['samples']}")
    
    if not raw_events:
        return ([], heuristics, attempts) if include_attempts else ([], heuristics)
    
    merged_events = merge_demo_events(raw_events)
    return (merged_events, heuristics, attempts) if include_attempts else (merged_events, heuristics)

ATTEMPT_START_MARKERS = {
    "FAIL_START": "FAIL",
    "RECOVER_START": "RECOVER",
    "SUCCESS_START": "SUCCESS"
}

ATTEMPT_END_MARKERS = {
    "FAIL_END": "FAIL",
    "RECOVER_END": "RECOVER",
    "SUCCESS_END": "SUCCESS"
}

def merge_demo_events(events, speed_tol=0.02):
    if not events:
        return []
    merged = []
    for evt in events:
        if not merged:
            merged.append(evt.copy())
            continue
        prev = merged[-1]
        same_cmd = evt.get('cmd') == prev.get('cmd')
        same_obj = evt.get('obj') == prev.get('obj')
        speed_match = abs(evt.get('speed', 0) - prev.get('speed', 0)) <= speed_tol
        if same_cmd and same_obj and speed_match:
            prev['duration'] += evt.get('duration', 0)
        else:
            merged.append(evt.copy())
    return merged

def extract_demo_attempts(log_data):
    attempts = {"FAIL": [], "RECOVER": [], "SUCCESS": []}
    current_obj = None
    current_type = None
    current_events = []

    def flush_attempt():
        nonlocal current_events
        if current_type and current_events:
            attempts[current_type].append(merge_demo_events(current_events))
        current_events = []

    for entry in log_data:
        if not isinstance(entry, dict):
            continue
        etype = entry.get('type')
        if etype == 'keyframe':
            obj_field = entry.get('objective')
            if obj_field:
                current_obj = obj_field
            marker = entry.get('marker')
            if marker in ATTEMPT_START_MARKERS:
                if current_type != ATTEMPT_START_MARKERS[marker]:
                    flush_attempt()
                current_type = ATTEMPT_START_MARKERS[marker]
            elif marker in ATTEMPT_END_MARKERS:
                if current_type == ATTEMPT_END_MARKERS[marker]:
                    flush_attempt()
                current_type = None
            continue

        if etype not in ('action', 'event'):
            continue

        if etype == 'action':
            cmd_name = entry.get('command')
            power = entry.get('power', 0)
            duration_ms = entry.get('duration_ms')
        else:
            event = entry.get('event') or {}
            cmd_name = event.get('type')
            power = event.get('power', 0)
            duration_ms = event.get('duration_ms')

        event_obj = entry.get('objective')
        if event_obj:
            current_obj = event_obj
        cmd = event_type_to_cmd(cmd_name)
        if cmd and duration_ms and current_type:
            current_events.append({
                'obj': current_obj,
                'cmd': cmd,
                'duration': duration_ms / 1000.0,
                'speed': (power or 0) / 255.0
            })

    flush_attempt()
    return attempts

def pick_best_attempt(attempts, attempt_type):
    if not attempts:
        return None
    sequences = attempts.get(attempt_type, [])
    if not sequences:
        return None
    return max(sequences, key=lambda seq: sum(evt.get('duration', 0) for evt in seq))

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
        path = DEMOS_DIR / s
        if path.is_dir():
             p = path / "a_log.json"
             if not p.exists(): p = path / "log.json"
             path = p
        elif path.suffix != ".json":
             path = path.with_suffix(".json")
        
        if not path.exists(): continue
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
    demos_dir = DEMOS_DIR
    if not demos_dir.exists(): return []
    items = os.listdir(demos_dir)
    sessions = [f for f in items if f.endswith(".json") or (demos_dir / f).is_dir()]
    sessions.sort(reverse=True)
    return sessions



def aggregate_heuristics():
    sessions = find_sessions()
    merged_heuristics = {}
    
    print(f"[LEARNING] Scanning {len(sessions)} logs for heuristics...")
    
    for s in sessions:
        # Resolve Path
        path = DEMOS_DIR / s
        if path.is_dir():
             p = path / "a_log.json"
             if not p.exists(): p = path / "log.json"
             path = p
        elif path.suffix != ".json":
             path = path.with_suffix(".json")
        
        if not path.exists(): continue
        
        # Load Data
        try:
            with open(path, 'r') as f:
                log_data = json.load(f)
        except: continue
        
        # Extract Specs
        current_obj = None
        last_state_entry = None
        
        for entry in log_data:
            if not isinstance(entry, dict): continue
            
            etype = entry.get('type')
            if etype == 'keyframe':
                obj_field = entry.get('objective')
                if obj_field: current_obj = obj_field
                
                marker = entry.get('marker')
                # We learn from both explicit OBJ_SUCCESS and implicit (next OBJ_START means prev success?)
                # Sticking to explicit OBJ_SUCCESS for safety.
                if marker == 'OBJ_SUCCESS' and current_obj and last_state_entry:
                    prev_state = last_state_entry.get('brick') or {}
                    
                    if current_obj not in merged_heuristics:
                        merged_heuristics[current_obj] = {
                            'max_offset_x': 0.0, 
                            'max_angle': 0.0, 
                            'final_visibility': prev_state.get('visible', True), 
                            'samples': 0
                        }
                    
                    h = merged_heuristics[current_obj]
                    
                    # Update MAX tolerances (Efficiency: If we succeeded with X, X is acceptable)
                    curr_off = abs(prev_state.get('offset_x', 0))
                    curr_ang = abs(prev_state.get('angle', 0))
                    
                    h['max_offset_x'] = max(h['max_offset_x'], curr_off)
                    h['max_angle'] = max(h['max_angle'], curr_ang)
                    h['samples'] += 1
                    
            elif etype == 'state':
                last_state_entry = entry

    if merged_heuristics:
        print("\n[LEARNED] Aggregated Success Gates:")
        for obj, h in merged_heuristics.items():
            print(f"  {obj:<8} | OffX: ~0-{h['max_offset_x']:.1f} | Ang: ~0-{h['max_angle']:.1f} | Vis: {h['final_visibility']} | n={h['samples']}")
            
    return merged_heuristics



def cmd_to_motion_type(cmd):
    m = {'f': 'forward', 'b': 'backward', 'l': 'left_turn', 'r': 'right_turn', 'u': 'mast_up', 'd': 'mast_down'}
    return m.get(cmd, 'wait')

def execute_and_track_smooth(cmd, duration, base_speed):
    """Executes command with smooth continuous motion, dynamically adjusting speed based on brick detection"""
    # 1. Update World (Dead Reckoning)
    m_type = cmd_to_motion_type(cmd)
    
    # Calculate approximate PWM same as Robot class
    pwm = int(60 + (255 - 60) * abs(base_speed))
    if cmd in ['u', 'd', 'wait']: pwm = 255 # Simple Approx
    
    evt = MotionEvent(m_type, pwm, int(duration * 1000))
    app_state.world.update_from_motion(evt)
    app_state.logger.log_event(evt, app_state.world.objective_state.value)
    
    # 2. Execute with dynamic speed adjustment - NO PAUSES
    start_time = time.time()
    last_send = 0
    
    while time.time() - start_time < duration:
        if not app_state.running: break
        
        # Dynamic speed: slow down if brick is detected
        with app_state.lock:
            brick_visible = app_state.world.brick['visible']
        
        current_speed = base_speed * BRICK_DETECTED_SLOWDOWN if brick_visible else base_speed
        
        # Send command continuously without any sleep
        current_time = time.time()
        app_state.robot.send_command(cmd, current_speed)
        last_send = current_time 

def replay_demo_attempt(events, objective, attempt_status, start_marker, end_marker, tick_hz=20):
    if not events:
        return False

    app_state.logger.log_keyframe(start_marker, objective)
    with app_state.lock:
        app_state.world.attempt_status = attempt_status
        app_state.world.recording_active = True

    dt = 1.0 / tick_hz
    for evt in events:
        if not app_state.running:
            break
        cmd = evt.get('cmd')
        speed = max(0.0, min(1.0, evt.get('speed', 0.0)))
        duration = evt.get('duration', 0.0) * SLOWDOWN_FACTOR
        if not cmd or duration <= 0:
            continue
        elapsed = 0.0
        while elapsed < duration and app_state.running:
            step = min(dt, duration - elapsed)
            app_state.robot.send_command(cmd, speed)
            motion_evt = MotionEvent(cmd_to_motion_type(cmd), int(speed * 255), int(step * 1000))
            with app_state.lock:
                app_state.world.update_from_motion(motion_evt)
            app_state.logger.log_event(motion_evt, objective)
            time.sleep(step)
            elapsed += step

    app_state.robot.stop()
    app_state.logger.log_keyframe(end_marker, objective)
    with app_state.lock:
        app_state.world.attempt_status = "NORMAL"
        app_state.world.recording_active = False
    return True



def decide_next_action(world, attempt_speed):
    """
    Decide what the robot should do based on current state.
    Returns: (command, duration, reason)
    """
    brick = world.brick
    
    # If brick is visible, work on alignment
    if brick['visible']:
        angle = brick['angle']
        offset_x = brick['offset_x']
        
        # Check if already aligned
        if abs(angle) <= 5.0 and abs(offset_x) <= 12.0:
            return None, 0, "ALIGNED - OBJECTIVE COMPLETE"
        
        # Prioritize angle correction
        if abs(angle) > 5.0:
            # Turn toward the brick
            cmd = 'l' if angle > 0 else 'r'
            # Duration proportional to error (but cap at 2 seconds)
            duration = min(abs(angle) / 90.0, 2.0)
            return cmd, duration, f"Correcting angle: {angle:.1f}°"
        
        # Then correct offset
        if abs(offset_x) > 12.0:
            # Assume we need to turn slightly to adjust offset
            # This is a simplification - could be improved with better kinematics
            cmd = 'l' if offset_x > 0 else 'r'
            duration = min(abs(offset_x) / 100.0, 2.0)
            return cmd, duration, f"Correcting offset: {offset_x:.1f}mm"
    
    # If no brick visible, search for one
    # Default: turn left to scan
    return 'l', 2.0, "Searching for brick..."

def smooth_velocity(current_vel, target_vel, alpha=0.3):
    """Exponential moving average for smooth velocity transitions"""
    return {
        'linear': alpha * target_vel['linear'] + (1 - alpha) * current_vel['linear'],
        'angular': alpha * target_vel['angular'] + (1 - alpha) * current_vel['angular']
    }

def compute_target_velocity(brick_visible, angle, offset_x, dist):
    """
    Compute target velocity vector based on current perception.
    Returns: {'linear': forward_speed, 'angular': turn_speed}
    Range: -1.0 to 1.0 for both
    """
    MIN_SPEED = 0.1  # Always moving at least 10%
    
    if brick_visible:
        # Brick detected - align while moving forward slowly
        linear = 0.15  # Slow forward motion
        
        # Angular velocity proportional to angle error
        angular = 0.0
        if abs(angle) > 5.0:
            # Turn to align (proportional control)
            angular = max(-0.4, min(0.4, angle / 90.0))
        elif abs(offset_x) > 12.0:
            # Adjust for offset
            angular = max(-0.2, min(0.2, offset_x / 100.0))
        
        return {'linear': linear, 'angular': angular}
    else:
        # No brick - slow scanning turn
        return {'linear': 0.0, 'angular': 0.3}  # Turn left to scan

def velocity_to_command(velocity):
    """
    Convert velocity vector to robot command.
    Returns: (cmd, speed) where cmd is 'f', 'b', 'l', 'r' and speed is 0-1
    """
    linear = velocity['linear']
    angular = velocity['angular']
    
    # Prioritize angular over linear (can be changed)
    if abs(angular) > 0.05:
        # Turning
        if angular > 0:
            return 'l', abs(angular)
        else:
            return 'r', abs(angular)
    elif abs(linear) > 0.05:
        # Moving forward/backward
        if linear > 0:
            return 'f', abs(linear)
        else:
            return 'b', abs(linear)
    else:
        # Default: very slow left turn (never fully stop)
        return 'l', 0.1

def continuous_autoplay(learned_rules):
    """
    True continuous velocity-based control.
    Robot flows toward target using smooth velocity corrections.
    Runs at fixed Hz with parallel perception - no blocking, no stops.
    """
    app_state.robot = Robot()
    app_state.world.learned_rules = learned_rules
    
    threading.Thread(target=vision_thread, daemon=True).start()
    if STREAM_ENABLED:
        threading.Thread(target=lambda: flask_app.run(host='0.0.0.0', port=WEB_PORT, debug=False, use_reloader=False), daemon=True).start()
    
    print(f"[AUTOPLAY] Continuous velocity control @ 20Hz")
    warmup_start = time.time()
    while time.time() - warmup_start < 2.0:
        with app_state.lock:
            if app_state.current_frame is not None: break
        time.sleep(0.1)

    print(f"\n{'='*60}")
    print(f"CONTINUOUS VELOCITY CONTROL - Smooth flow toward target")
    print(f"{'='*60}")
    
    # Set objective
    with app_state.lock:
        app_state.active_objective = "FIND"
        app_state.world.objective_state = ObjectiveState.FIND
    
    app_state.logger.log_keyframe("JOB_START")
    app_state.logger.log_keyframe("OBJ_START", "FIND")
    
    # Control parameters
    CONTROL_HZ = 20  # 20 Hz control loop
    CONTROL_DT = 1.0 / CONTROL_HZ
    
    # Failure detection parameters
    MAX_RUN_DURATION = 60.0  # Max time before timeout failure
    MAX_BRICK_LOST_TIME = 10.0  # Max time without seeing brick
    
    # State variables
    current_velocity = {'linear': 0.0, 'angular': 0.1}  # Start with slow turn
    run_start = time.time()
    cycle_count = 0
    last_log_time = run_start
    
    # Failure tracking
    brick_first_seen_time = None
    brick_last_seen_time = None
    failure_reason = None
    
    print(f"  └─ Running at {CONTROL_HZ}Hz | Min speed: 10% | Velocity smoothing: EMA")
    print(f"  └─ Failure detection: Timeout={MAX_RUN_DURATION}s | Brick lost={MAX_BRICK_LOST_TIME}s")
    
    # MAIN CONTROL LOOP - Fixed frequency
    while time.time() - run_start < MAX_RUN_DURATION:
        cycle_start = time.time()
        
        if not app_state.running:
            break
        
        # 1. GET TELEMETRY (non-blocking read from shared state)
        with app_state.lock:
            # Check completion
            if app_state.world.check_objective_complete():
                print(f"\n✓ FIND Objective Complete! (after {cycle_count} cycles)")
                app_state.logger.log_keyframe("OBJ_SUCCESS", "FIND")
                app_state.robot.stop()
                break
            
            # Get current perception
            brick_visible = app_state.world.brick['visible']
            angle = app_state.world.brick['angle']
            offset_x = app_state.world.brick['offset_x']
            dist = app_state.world.brick['dist']
        
        # Track brick visibility for failure detection
        current_time = time.time()
        if brick_visible:
            if brick_first_seen_time is None:
                brick_first_seen_time = current_time
                print(f"  ✓ First brick detection at {cycle_count} cycles")
            brick_last_seen_time = current_time
        
        # FAILURE DETECTION: Lost brick for too long
        if brick_first_seen_time is not None and brick_last_seen_time is not None:
            time_since_brick_seen = current_time - brick_last_seen_time
            if time_since_brick_seen > MAX_BRICK_LOST_TIME:
                failure_reason = f"Lost brick for {time_since_brick_seen:.1f}s"
                break
        
        # 2. COMPUTE TARGET VELOCITY VECTOR
        target_velocity = compute_target_velocity(brick_visible, angle, offset_x, dist)
        
        # 3. SMOOTH VELOCITY (blend with previous)
        smoothed_velocity = smooth_velocity(current_velocity, target_velocity, alpha=0.3)
        
        # 4. CONVERT TO COMMAND
        cmd, speed = velocity_to_command(smoothed_velocity)
        
        # 5. SEND COMMAND (non-blocking)
        actual_speed = GEAR_1_SPEED * speed
        app_state.robot.send_command(cmd, actual_speed)
        
        # Update state
        current_velocity = smoothed_velocity
        cycle_count += 1
        
        # Periodic logging (every 2 seconds)
        if time.time() - last_log_time > 2.0:
            status = "🟡 ALIGNING" if brick_visible else "🟢 SCANNING"
            vel_str = f"v=({smoothed_velocity['linear']:.2f}, {smoothed_velocity['angular']:.2f})"
            print(f"  {status} | {cmd}@{speed:.2f} | {vel_str} | {cycle_count} cycles")
            last_log_time = time.time()
        
        # 6. MAINTAIN FIXED FREQUENCY (non-blocking sleep)
        cycle_elapsed = time.time() - cycle_start
        sleep_time = CONTROL_DT - cycle_elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    # Check for timeout failure
    total_time = time.time() - run_start
    if total_time >= MAX_RUN_DURATION and failure_reason is None:
        failure_reason = f"Timeout: {total_time:.1f}s"
    
    # Clean shutdown with failure logging
    app_state.robot.stop()
    
    if failure_reason:
        print(f"\n✗ FIND Objective FAILED: {failure_reason}")
        app_state.logger.log_keyframe("FAIL_START", "FIND")
        app_state.logger.log_keyframe("FAIL_END", "FIND")
    
    app_state.logger.log_keyframe("JOB_SUCCESS")  # Job completes even if objective failed
    
    avg_hz = cycle_count / total_time if total_time > 0 else 0
    status_icon = "✗" if failure_reason else "✓"
    print(f"\n[AUTOPLAY] {status_icon} Run complete: {cycle_count} cycles in {total_time:.1f}s (avg {avg_hz:.1f}Hz)")
    app_state.logger.close()
    app_state.running = False

def main_autoplay(session_name):
    # Multi-scenario training: SUCCESS → FAIL → RECOVER → SUCCESS
    print("\n[AUTOPLAY] Multi-scenario training mode")
    print("  Sequence: SUCCESS → FAIL → RECOVER → SUCCESS\n")
    
    learned_rules = aggregate_heuristics()
    demo_attempts = None
    demo_fail = None
    demo_recover = None
    if SMART_REPLAY and session_name:
        _, _, demo_attempts = load_demo(session_name, include_attempts=True)
        demo_fail = pick_best_attempt(demo_attempts, "FAIL")
        demo_recover = pick_best_attempt(demo_attempts, "RECOVER")
    
    # Initialize (done once for all scenarios)
    app_state.robot = Robot()
    app_state.world.learned_rules = learned_rules
    threading.Thread(target=vision_thread, daemon=True).start()
    if STREAM_ENABLED:
        threading.Thread(target=lambda: flask_app.run(host='0.0.0.0', port=WEB_PORT, debug=False, use_reloader=False), daemon=True).start()
    
    # Warmup
    print("Warming up vision...")
    warmup_start = time.time()
    while time.time() - warmup_start < 2.0:
        with app_state.lock:
            if app_state.current_frame is not None: break
        time.sleep(0.1)
    
    # Run 4 training scenarios
    for idx, (name, timeout) in enumerate([("SUCCESS #1", 60), ("FAIL", 5), ("RECOVER", 60), ("SUCCESS #2", 60)], 1):
        scenario_type = "recover" if name == "RECOVER" else ("fail" if name == "FAIL" else "normal")
        min_success_time = 1.0
        min_success_cycles = 5
        use_demo_fail = scenario_type == "fail" and demo_fail
        use_demo_recover = scenario_type == "recover" and demo_recover
        
        print(f"{'='*70}")
        print(f"SCENARIO {idx}/4: {name} (timeout={timeout}s, type={scenario_type})")
        print(f"{'='*70}\n")
        
        # Log start
        if scenario_type == "recover":
            if not use_demo_recover:
                app_state.logger.log_keyframe("RECOVER_START", "FIND")
        else:
            app_state.logger.log_keyframe("OBJ_START", "FIND")
        
        with app_state.lock:
            app_state.world.reset_mission()
            app_state.world.objective_state = ObjectiveState.FIND
            if use_demo_fail:
                app_state.world.attempt_status = "FAIL"
            elif use_demo_recover:
                app_state.world.attempt_status = "RECOVERY"
        
        if use_demo_fail:
            print("  → Replaying FAIL demo attempt")
            replay_demo_attempt(demo_fail, "FIND", "FAIL", "FAIL_START", "FAIL_END")
        elif use_demo_recover:
            print("  → Replaying RECOVERY demo attempt")
            replay_demo_attempt(demo_recover, "FIND", "RECOVERY", "RECOVER_START", "RECOVER_END")
        else:
            # Run scenario
            current_velocity = {'linear': 0.0, 'angular': 0.1}
            run_start = time.time()
            cycle_count = 0
            success_flag = False
            
            while time.time() - run_start < timeout:
                with app_state.lock:
                    allow_success = cycle_count >= min_success_cycles and (time.time() - run_start) >= min_success_time
                    if allow_success and app_state.world.check_objective_complete():
                        print(f"  ✓ Objective complete! ({cycle_count} cycles)\n")
                        app_state.logger.log_keyframe("OBJ_SUCCESS", "FIND")
                        success_flag = True
                        break
                    
                    brick_visible = app_state.world.brick['visible']
                    angle = app_state.world.brick['angle']
                    offset_x = app_state.world.brick['offset_x']
                    dist = app_state.world.brick['dist']
                
                target_velocity = compute_target_velocity(brick_visible, angle, offset_x, dist)
                smoothed_velocity = smooth_velocity(current_velocity, target_velocity, alpha=0.3)
                cmd, speed = velocity_to_command(smoothed_velocity)
                app_state.robot.send_command(cmd, GEAR_1_SPEED * speed)
                current_velocity = smoothed_velocity
                cycle_count += 1
                time.sleep(0.05)  # 20Hz
            
            total_time = time.time() - run_start
            app_state.robot.stop()
            
            # Log result
            if not success_flag:
                print(f"  ✗ Failed: Timeout {total_time:.1f}s\n")
                app_state.logger.log_keyframe("FAIL_START", "FIND")
                app_state.logger.log_keyframe("FAIL_END", "FIND")
            elif scenario_type == "recover":
                app_state.logger.log_keyframe("RECOVER_END", "FIND")
        
        # Pause between scenarios
        if idx < 4:
            print("  → Pausing 2s before next scenario...\n")
            time.sleep(2.0)
    
    app_state.logger.close()
    app_state.running = False
    
    print(f"\n{'='*70}")
    print("✓ ALL TRAINING SCENARIOS COMPLETE") 
    print(f"{'='*70}\n")

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
