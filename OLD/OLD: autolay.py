import argparse
import json
import math
import os
import statistics
import sys
import time
import threading
import cv2
import numpy as np
from flask import Flask, Response
from pathlib import Path

from helper_robot_control import Robot
from helper_brick_vision import BrickDetector
from robot_leia_telemetry import WorldModel, TelemetryLogger, MotionEvent, StepState, draw_telemetry_overlay

# --- CONFIG ---
GEAR_1_SPEED = 1.0  # Full speed instead of 0.32
WEB_PORT = 5000  # Port 5000 to match recording scripts
HEARTBEAT_RATE = 20 # Hz for internal loop
SLOWDOWN_FACTOR = 1.0 # default to real time
SMART_REPLAY = True
STREAM_ENABLED = True # Default, will be updated by args
DEBUG_MODE = False
DEMOS_DIR = Path(__file__).resolve().parent / "demos"
STEP_TIMEOUT = 10.0  # Seconds before declaring step failure
RECOVER_TIMEOUT = 60.0  # Seconds to attempt recovery
MAX_STEP_ATTEMPTS = 5
BRICK_LOST_FAIL_TIME = 10.0
TURN_SIGN = -1  # Invert turn direction for angle/offset alignment

# Continuous motion config - never stop moving
SCAN_SPEED = 0.4  # Constant slow scanning speed
BRICK_SPOTTED_SPEED = 0.15  # Even slower when brick is visible  
ALIGNMENT_SPEED = 0.2  # Speed for fine adjustments
SCOOP_APPROACH_SPEED = 0.25  # Forward push when scooping
SCOOP_ALIGN_SPEED = 0.18  # Approach speed while correcting alignment
SCOOP_VERIFY_SPEED = 0.25  # Wiggle verification speed
LIFT_ACTION_SPEED = 0.6  # Mast up speed during carry
LIFT_ACTION_DURATION = 2.0  # Seconds to lift
PLACE_ACTION_SPEED = 0.6  # Mast down speed during place
PLACE_ACTION_DURATION = 2.0  # Seconds to lower
SCOOP_CORRIDOR_BIN_MM = 25.0
SCOOP_CORRIDOR_PCT = 0.9
SCOOP_CORRIDOR_MIN_SAMPLES = 8
SCOOP_COMMIT_PCT = 0.9
GATE_MIN_SAMPLES = 5
FAIL_TIME_OVERRIDE_S = 60.0
ANGLE_TOLERANCE_MULTIPLIER = 3.0
START_GATE_MIN_CONFIDENCE = 25.0

METRIC_DIRECTIONS = {
    "angle_abs": "low",
    "offset_abs": "low",
    "dist": "low",
    "visible": "high",
    "lift_height": "band",
}

STEP_METRICS = {
    "FIND": ("angle_abs", "offset_abs", "dist", "visible"),
    "SCOOP": ("angle_abs", "offset_abs", "dist", "visible"),
    "LIFT": ("lift_height",),
    "PLACE": ("lift_height",),
}

VISIBILITY_REQUIRED_METRICS = {"angle_abs", "offset_abs", "dist"}

class AutoplayState:
    def __init__(self):
        self.running = True
        self.lock = threading.Lock()
        self.current_frame = None
        self.world = WorldModel()
        self.robot = None
        self.vision = None
        self.active_step = "UNKNOWN"
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
                messages = [f"AUTO: {app_state.active_step}", app_state.status_msg]
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
            obj_field = entry.get('step')
            if obj_field: current_obj = obj_field
            marker = entry.get('marker')
            if marker == 'OBJ_SUCCESS' and current_obj and last_state_entry:
                prev_state = last_state_entry.get('brick') or {}
                if current_obj not in heuristics:
                    heuristics[current_obj] = {'max_offset_x': 0.0, 'max_angle': 0.0, 'final_visibility': prev_state.get('visible', True), 'samples': 0}
                h = heuristics[current_obj]
                prev_offset = prev_state.get('offset_x')
                if prev_offset is not None:
                    h['max_offset_x'] = max(h['max_offset_x'], abs(prev_offset))
                prev_angle = prev_state.get('angle')
                if prev_angle is not None:
                    h['max_angle'] = max(h['max_angle'], abs(prev_angle))
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

            event_obj = entry.get('step')
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
            obj_field = entry.get('step')
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

        event_obj = entry.get('step')
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

def extract_attempt_segments(log_data):
    segments = []
    current_obj = None
    current = None

    def close_segment(end_time=None):
        nonlocal current
        if not current:
            return
        if end_time is not None:
            current["end"] = end_time
        if current.get("start") is None and current.get("states"):
            current["start"] = current["states"][0].get("timestamp")
        if current.get("end") is None and current.get("states"):
            current["end"] = current["states"][-1].get("timestamp")
        if current.get("start") is not None and current.get("end") is not None:
            segments.append(current)
        current = None

    for entry in log_data:
        if not isinstance(entry, dict):
            continue
        etype = entry.get("type")
        if etype == "keyframe":
            obj_field = entry.get("step")
            if obj_field:
                current_obj = obj_field
            marker = entry.get("marker")
            if marker in ATTEMPT_START_MARKERS:
                close_segment()
                current = {
                    "step": current_obj,
                    "type": ATTEMPT_START_MARKERS[marker],
                    "start": entry.get("timestamp"),
                    "end": None,
                    "states": [],
                    "events": [],
                }
            elif marker in ATTEMPT_END_MARKERS:
                if current and current.get("type") == ATTEMPT_END_MARKERS[marker]:
                    close_segment(entry.get("timestamp"))
            continue

        if not current:
            continue
        if etype == "state":
            current["states"].append(entry)
            continue

        if etype in ("action", "event"):
            if etype == "action":
                cmd_name = entry.get("command")
                power = entry.get("power", 0)
                duration_ms = entry.get("duration_ms")
                timestamp = entry.get("timestamp")
            else:
                event = entry.get("event") or {}
                cmd_name = event.get("type")
                power = event.get("power", 0)
                duration_ms = event.get("duration_ms")
                timestamp = entry.get("timestamp") or event.get("timestamp")

            cmd = event_type_to_cmd(cmd_name)
            if cmd and duration_ms:
                current["events"].append({
                    "cmd": cmd,
                    "duration": duration_ms / 1000.0,
                    "speed": (power or 0) / 255.0,
                    "timestamp": timestamp,
                })

    close_segment()
    return segments

def build_stat_band(values, non_negative=False, clamp_min=None, clamp_max=None):
    if not values:
        return None
    mu = statistics.mean(values)
    sigma = statistics.pstdev(values) if len(values) > 1 else 0.0
    min_val = mu - sigma
    max_val = mu + sigma
    if non_negative:
        min_val = max(0.0, min_val)
        max_val = max(0.0, max_val)
    if clamp_min is not None:
        min_val = max(clamp_min, min_val)
    if clamp_max is not None:
        max_val = min(clamp_max, max_val)
    return {
        "mu": mu,
        "sigma": sigma,
        "min": min_val,
        "max": max_val,
        "samples": len(values),
    }

def extract_metrics_from_states(states, step):
    metrics = {m: [] for m in STEP_METRICS.get(step, ())}
    if not metrics:
        return metrics

    for state in states:
        brick = state.get("brick") or {}
        visible = bool(brick.get("visible"))
        for metric in metrics:
            if metric in VISIBILITY_REQUIRED_METRICS and not visible:
                continue
            if metric == "angle_abs":
                metrics[metric].append(abs(brick.get("angle", 0.0)))
            elif metric == "offset_abs":
                metrics[metric].append(abs(brick.get("offset_x", 0.0)))
            elif metric == "dist":
                dist = brick.get("dist")
                if dist is None or dist <= 0:
                    continue
                metrics[metric].append(float(dist))
            elif metric == "confidence":
                conf = brick.get("confidence")
                if conf is None:
                    continue
                metrics[metric].append(float(conf))
            elif metric == "visible":
                metrics[metric].append(1.0 if visible else 0.0)
            elif metric == "lift_height":
                lift = state.get("lift_height")
                if lift is None:
                    continue
                metrics[metric].append(float(lift))

    return metrics

def average_forward_speed(events, start_time, end_time):
    if not events or start_time is None or end_time is None or end_time <= start_time:
        return None
    weighted = 0.0
    total = 0.0
    for evt in events:
        if evt.get("cmd") != "f":
            continue
        evt_start = evt.get("timestamp")
        if evt_start is None:
            continue
        evt_end = evt_start + evt.get("duration", 0.0)
        overlap = max(0.0, min(evt_end, end_time) - max(evt_start, start_time))
        if overlap <= 0:
            continue
        speed = evt.get("speed", 0.0)
        weighted += speed * overlap
        total += overlap
    if total <= 0:
        return None
    return weighted / total

def pick_best_attempt(attempts, attempt_type):
    if not attempts:
        return None
    sequences = attempts.get(attempt_type, [])
    if not sequences:
        return None
    return max(sequences, key=lambda seq: sum(evt.get('duration', 0) for evt in seq))

def pick_best_attempt_for_step(attempts, attempt_type, step):
    if not attempts:
        return None
    sequences = attempts.get(attempt_type, [])
    if not sequences:
        return None

    best_seq = None
    best_duration = 0.0
    for seq in sequences:
        obj_events = [evt for evt in seq if evt.get('obj') == step]
        duration = sum(evt.get('duration', 0) for evt in obj_events)
        if obj_events and duration > best_duration:
            best_duration = duration
            best_seq = merge_demo_events(obj_events)

    return best_seq

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
        return False, {}, ["Failed to parse JSON"], []

    stack = []
    errors = []
    warnings = []
    stats = {}
    attempt_types = {"FAIL", "RECOVER", "SUCCESS"}
    for entry in log_data:
        if entry.get('type') != 'keyframe': continue
        marker = entry.get('marker', '')
        obj = entry.get('step', 'GLOBAL')
        if obj not in stats:
            stats[obj] = {"FAIL": 0, "RECOVER": 0, "COMPLETE": 0}
        
        if marker.endswith("_START"):
            stack.append((marker.replace("_START", ""), obj))
        elif marker.endswith("_END"):
            m_type = marker.replace("_END", "")
            if not stack:
                msg = f"Unexpected {marker} for {obj}"
                if m_type in attempt_types:
                    warnings.append(msg)
                else:
                    errors.append(msg)
            else:
                top_type, top_obj = stack.pop()
                if top_type != m_type:
                    # Put it back and report error
                    stack.append((top_type, top_obj))
                    msg = f"Mismatched: {top_type} vs {m_type} for {obj}"
                    if m_type in attempt_types:
                        warnings.append(msg)
                    else:
                        errors.append(msg)
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
            msg = f"Unclosed {m_type} for {m_obj}"
            if m_type in attempt_types or m_type == "OBJ":
                warnings.append(msg)
            else:
                errors.append(msg)
        
    return len(errors) == 0, stats, errors, warnings

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
        ok, stats, errors, warnings = validate_log_integrity(path)
        if not stats:
            status = "PASS" if ok else "FAIL"
            print(f"{s[:33]:<35} | {'-':<8} | {'-':<5} | {'-':<5} | {'-':<5} | {status}")
            for err in errors:
                print(f"  [ISSUE] {err}")
            for warn in warnings:
                print(f"  [WARN] {warn}")
            continue

        first = True
        for obj, sdata in stats.items():
            if obj == "GLOBAL": continue
            prefix = f"{s[:33]:<35}" if first else " " * 35
            status = "PASS" if ok else "FAIL"
            if not first: status = ""
            print(f"{prefix} | {obj:<8} | {sdata.get('FAIL',0):<5} | {sdata.get('RECOVER',0):<5} | {sdata.get('COMPLETE',0):<5} | {status}")
            first = False
        if errors:
            for err in errors:
                print(f"  [ISSUE] {err}")
        if warnings:
            for warn in warnings:
                print(f"  [WARN] {warn}")
    print(f"{'='*90}\n")

def find_sessions():
    demos_dir = DEMOS_DIR
    if not demos_dir.exists(): return []
    items = os.listdir(demos_dir)
    sessions = [f for f in items if f.endswith(".json") or (demos_dir / f).is_dir()]
    sessions.sort(reverse=True)
    return sessions


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    idx = int(math.ceil((len(values) - 1) * pct))
    return values[idx]

def build_scoop_corridor(samples):
    bins = {}
    for dist, off, ang in samples:
        if dist <= 0:
            continue
        bin_idx = int(dist // SCOOP_CORRIDOR_BIN_MM)
        bucket = bins.setdefault(bin_idx, {"offsets": [], "angles": []})
        bucket["offsets"].append(off)
        bucket["angles"].append(ang)

    corridor = []
    for bin_idx in sorted(bins):
        offsets = bins[bin_idx]["offsets"]
        angles = bins[bin_idx]["angles"]
        if len(offsets) < SCOOP_CORRIDOR_MIN_SAMPLES:
            continue
        corridor.append({
            "dist_min": bin_idx * SCOOP_CORRIDOR_BIN_MM,
            "dist_max": (bin_idx + 1) * SCOOP_CORRIDOR_BIN_MM,
            "max_offset_x": percentile(offsets, SCOOP_CORRIDOR_PCT),
            "max_angle": percentile(angles, SCOOP_CORRIDOR_PCT),
            "samples": len(offsets),
        })

    return corridor

def corridor_limits(corridor, dist):
    if not corridor or dist is None:
        return None
    for row in corridor:
        if row["dist_min"] <= dist < row["dist_max"]:
            return row
    if dist < corridor[0]["dist_min"]:
        return corridor[0]
    return corridor[-1]

def build_scoop_commit(corridor, segments):
    if not corridor or not segments:
        return None
    commit_times = []
    commit_dists = []
    commit_speeds = []
    for seg in segments:
        success_time = seg.get("end") or seg.get("success_time")
        if not success_time:
            continue
        last_aligned = None
        for state in seg.get("states", []):
            brick = state.get("brick") or {}
            if not brick.get("visible"):
                continue
            dist = brick.get("dist", 0)
            limits = corridor_limits(corridor, dist)
            if not limits:
                continue
            if abs(brick.get("offset_x", 0)) <= limits["max_offset_x"] and abs(brick.get("angle", 0)) <= limits["max_angle"]:
                last_aligned = state
        if last_aligned:
            dt = success_time - last_aligned.get("timestamp", success_time)
            if dt >= 0:
                commit_times.append(dt)
                commit_dists.append(last_aligned.get("brick", {}).get("dist", 0))
                speed = average_forward_speed(seg.get("events", []), last_aligned.get("timestamp"), success_time)
                if speed is not None:
                    commit_speeds.append(speed)

    if not commit_times:
        return None

    commit = {
        "time_s": percentile(commit_times, SCOOP_COMMIT_PCT),
        "max_dist": percentile(commit_dists, SCOOP_COMMIT_PCT),
        "samples": len(commit_times),
    }
    if commit_speeds:
        commit["speed"] = percentile(commit_speeds, SCOOP_COMMIT_PCT)
        commit["speed_samples"] = len(commit_speeds)
    return commit


def aggregate_heuristics():
    sessions = find_sessions()
    merged_heuristics = {}
    scoop_samples = []
    scoop_segments = []
    scoop_blind_times = []
    scoop_blind_speeds = []
    gate_acc = {
        "success": {},
        "failure": {},
        "success_times": {},
        "failure_times": {},
    }

    print(f"[LEARNING] Scanning {len(sessions)} logs for heuristics...")

    for s in sessions:
        # Resolve Path
        path = DEMOS_DIR / s
        if path.is_dir():
            p = path / "a_log.json"
            if not p.exists():
                p = path / "log.json"
            path = p
        elif path.suffix != ".json":
            path = path.with_suffix(".json")

        if not path.exists():
            continue

        # Load Data
        try:
            with open(path, 'r') as f:
                log_data = json.load(f)
        except:
            continue

        segments = extract_attempt_segments(log_data)
        for seg in segments:
            obj = seg.get("step")
            seg_type = seg.get("type")
            if not obj or not seg_type:
                continue
            duration = None
            if seg.get("start") is not None and seg.get("end") is not None:
                duration = seg["end"] - seg["start"]

            metrics = extract_metrics_from_states(seg.get("states", []), obj)

            if seg_type == "SUCCESS":
                if duration is not None:
                    gate_acc["success_times"].setdefault(obj, []).append(duration)
                for metric, values in metrics.items():
                    if values:
                        gate_acc["success"].setdefault(obj, {}).setdefault(metric, []).extend(values)

                if obj == "SCOOP":
                    scoop_segments.append(seg)
                    for state in seg.get("states", []):
                        brick = state.get("brick") or {}
                        if brick.get("visible") and brick.get("confidence", 0) >= 25:
                            dist = brick.get("dist", 0)
                            scoop_samples.append((dist, abs(brick.get("offset_x", 0)), abs(brick.get("angle", 0))))

                    last_visible = None
                    for state in seg.get("states", []):
                        brick = state.get("brick") or {}
                        if brick.get("visible"):
                            last_visible = state.get("timestamp")
                    if last_visible and seg.get("end") and seg.get("end") >= last_visible:
                        scoop_blind_times.append(seg["end"] - last_visible)
                        speed = average_forward_speed(seg.get("events", []), last_visible, seg.get("end"))
                        if speed is not None:
                            scoop_blind_speeds.append(speed)

            elif seg_type == "FAIL":
                if duration is not None:
                    gate_acc["failure_times"].setdefault(obj, []).append(duration)
                for metric, values in metrics.items():
                    if values:
                        gate_acc["failure"].setdefault(obj, {}).setdefault(metric, []).extend(values)

    for obj, metric_list in STEP_METRICS.items():
        success_metrics = {}
        failure_metrics = {}
        for metric in metric_list:
            success_vals = gate_acc["success"].get(obj, {}).get(metric, [])
            failure_vals = gate_acc["failure"].get(obj, {}).get(metric, [])
            non_negative = metric != "visible"
            clamp_min = 0.0 if non_negative else 0.0
            clamp_max = 1.0 if metric == "visible" else None
            success_stats = build_stat_band(success_vals, non_negative=non_negative, clamp_min=clamp_min, clamp_max=clamp_max)
            failure_stats = build_stat_band(failure_vals, non_negative=non_negative, clamp_min=clamp_min, clamp_max=clamp_max)
            if success_stats:
                success_metrics[metric] = success_stats
            if failure_stats:
                failure_metrics[metric] = failure_stats

        success_time = build_stat_band(gate_acc["success_times"].get(obj, []), non_negative=True)
        failure_time = build_stat_band(gate_acc["failure_times"].get(obj, []), non_negative=True)

        if success_metrics or failure_metrics or success_time or failure_time:
            merged_heuristics.setdefault(obj, {})
            merged_heuristics[obj]["gates"] = {
                "success": {"metrics": success_metrics, "time_s": success_time},
                "failure": {"metrics": failure_metrics, "time_s": failure_time},
                "temporal": {"success_time_s": success_time, "fail_time_s": failure_time},
            }

            if "offset_abs" in success_metrics:
                merged_heuristics[obj]["max_offset_x"] = success_metrics["offset_abs"]["max"]
            if "angle_abs" in success_metrics:
                merged_heuristics[obj]["max_angle"] = success_metrics["angle_abs"]["max"]
            if "visible" in success_metrics:
                merged_heuristics[obj]["final_visibility"] = success_metrics["visible"]["mu"] >= 0.5
            merged_heuristics[obj]["samples"] = max(
                (stat.get("samples", 0) for stat in success_metrics.values()),
                default=0
            )

    if merged_heuristics:
        print("\n[LEARNED] Success Envelopes:")
        for obj, h in merged_heuristics.items():
            gates = h.get("gates", {}).get("success", {}).get("metrics", {})
            off = gates.get("offset_abs", {}).get("max")
            ang = gates.get("angle_abs", {}).get("max")
            vis = h.get("final_visibility")
            samples = h.get("samples", 0)
            if off is not None or ang is not None:
                print(f"  {obj:<8} | OffX: ~0-{off if off is not None else 0:.1f} | Ang: ~0-{ang if ang is not None else 0:.1f} | Vis: {vis} | n={samples}")

    scoop_corridor = build_scoop_corridor(scoop_samples)
    if scoop_corridor:
        merged_heuristics.setdefault("SCOOP", {})
        merged_heuristics["SCOOP"]["corridor"] = scoop_corridor

        commit = build_scoop_commit(scoop_corridor, scoop_segments)
        if commit:
            merged_heuristics["SCOOP"]["commit"] = commit
            print(f"[LEARNED] SCOOP commit: dist<=~{commit['max_dist']:.1f}mm, time<=~{commit['time_s']:.2f}s (n={commit['samples']})")

    if scoop_blind_times:
        blind_time = build_stat_band(scoop_blind_times, non_negative=True)
        blind_speed = build_stat_band(scoop_blind_speeds, non_negative=True) if scoop_blind_speeds else None
        merged_heuristics.setdefault("SCOOP", {})
        merged_heuristics["SCOOP"]["blind"] = {
            "time_s": blind_time,
            "speed": blind_speed,
        }
        if blind_time:
            mean_blind = blind_time.get("mu", 0.0)
            print(f"[LEARNED] SCOOP blind window: ~{mean_blind:.2f}s (n={blind_time.get('samples', 0)})")

    return merged_heuristics



def cmd_to_motion_type(cmd):
    m = {'f': 'forward', 'b': 'backward', 'l': 'left_turn', 'r': 'right_turn', 'u': 'mast_up', 'd': 'mast_down'}
    return m.get(cmd, 'wait')

def step_complete(world):
    if world.step_state == StepState.SCOOP:
        if world.verification_stage == "IDLE" and world.brick.get("seated"):
            return True
    return world.check_step_complete()

def get_scoop_verification_command(world):
    if world.step_state != StepState.SCOOP:
        return None
    stage = world.verification_stage
    if stage == "BACK":
        return 'b', SCOOP_VERIFY_SPEED, "VERIFY_BACK"
    if stage == "LEFT":
        return 'l', SCOOP_VERIFY_SPEED, "VERIFY_LEFT"
    if stage == "RIGHT":
        return 'r', SCOOP_VERIFY_SPEED, "VERIFY_RIGHT"
    return None

def execute_and_track_smooth(cmd, duration, base_speed):
    """Executes command with smooth continuous motion, dynamically adjusting speed based on brick detection"""
    # 1. Update World (Dead Reckoning)
    m_type = cmd_to_motion_type(cmd)
    
    # Calculate approximate PWM same as Robot class
    pwm = int(60 + (255 - 60) * abs(base_speed))
    if cmd in ['u', 'd', 'wait']: pwm = 255 # Simple Approx
    
    evt = MotionEvent(m_type, pwm, int(duration * 1000))
    app_state.world.update_from_motion(evt)
    app_state.logger.log_event(evt, app_state.world.step_state.value)
    
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

def replay_demo_attempt(events, step, attempt_status, start_marker, end_marker, tick_hz=20):
    if not events:
        return False

    app_state.logger.log_keyframe(start_marker, step)
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
            app_state.logger.log_event(motion_evt, step)
            time.sleep(step)
            elapsed += step

    app_state.robot.stop()
    app_state.logger.log_keyframe(end_marker, step)
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
            return None, 0, "ALIGNED - STEP COMPLETE"
        
        # Prioritize angle correction
        if abs(angle) > 5.0:
            # Turn toward the brick
            cmd = turn_cmd_from_error(angle)
            # Duration proportional to error (but cap at 2 seconds)
            duration = min(abs(angle) / 90.0, 2.0)
            return cmd, duration, f"Correcting angle: {angle:.1f}°"
        
        # Then correct offset
        if abs(offset_x) > 12.0:
            # Assume we need to turn slightly to adjust offset
            # This is a simplification - could be improved with better kinematics
            cmd = turn_cmd_from_error(offset_x)
            duration = min(abs(offset_x) / 100.0, 2.0)
            return cmd, duration, f"Correcting offset: {offset_x:.1f}mm"
    
    # If no brick visible, search for one
    # Default: turn to scan
    scan_cmd = 'l' if TURN_SIGN > 0 else 'r'
    return scan_cmd, 2.0, "Searching for brick..."

def smooth_velocity(current_vel, target_vel, alpha=0.3):
    """Exponential moving average for smooth velocity transitions"""
    return {
        'linear': alpha * target_vel['linear'] + (1 - alpha) * current_vel['linear'],
        'angular': alpha * target_vel['angular'] + (1 - alpha) * current_vel['angular']
    }

def turn_cmd_from_error(error):
    return 'l' if error * TURN_SIGN > 0 else 'r'

def step_display_label(obj_enum):
    return "CARRY" if obj_enum == StepState.LIFT else obj_enum.value

def _format_percent(delta, baseline):
    if baseline is None or baseline == 0:
        return None
    return (delta / baseline) * 100.0

def humanize_failure_reason(reason):
    if not reason:
        return "failure gate triggered"
    mapping = {
        "angle_abs gate": "angle outside learned envelope",
        "offset_abs gate": "x-offset outside learned envelope",
        "dist gate": "distance outside learned envelope",
        "confidence gate": "confidence below learned envelope",
        "visible gate (time-to-fail)": "brick not visible within learned time-to-fail",
        "lost visibility gate": "brick visibility lost too long",
        "time-to-fail gate": "exceeded learned time-to-fail",
        "lift_height gate": "lift height outside learned envelope",
    }
    parts = [p.strip() for p in str(reason).split(";")]
    rendered = [mapping.get(part, part) for part in parts if part]
    return "; ".join(rendered) if rendered else "failure gate triggered"

def format_failure_details(world, step, gate_eval, learned_rules, elapsed_s):
    obj_name = step.value if isinstance(step, StepState) else step
    gates = learned_rules.get(obj_name, {}).get("gates", {})
    success_metrics = gates.get("success", {}).get("metrics", {})
    temporal = gates.get("temporal", {})
    reasons = [p.strip() for p in str(gate_eval.get("reason") or "").split(";") if p.strip()]
    if not reasons:
        return []

    brick = world.brick or {}
    brick_visible = bool(brick.get("visible"))
    values = {
        "angle_abs": abs(brick.get("angle", 0.0)),
        "offset_abs": abs(brick.get("offset_x", 0.0)),
        "dist": brick.get("dist", 0.0),
        "confidence": brick.get("confidence", 0.0),
        "visible": 1.0 if brick_visible else 0.0,
        "lift_height": getattr(world, "lift_height", 0.0),
    }
    details = []
    fail_time_max = (temporal.get("fail_time_s") or {}).get("max")
    if fail_time_max is not None:
        fail_time_max = max(fail_time_max, FAIL_TIME_OVERRIDE_S)

    for reason in reasons:
        if reason.startswith("visible gate"):
            if fail_time_max is not None:
                details.append(f"visible: no detection for {elapsed_s:.1f}s (limit {fail_time_max:.1f}s)")
            else:
                details.append(f"visible: no detection (elapsed {elapsed_s:.1f}s)")
            continue
        if reason == "lost visibility gate":
            lost_s = None
            if world.last_visible_time is not None:
                lost_s = time.time() - world.last_visible_time
            if lost_s is not None and fail_time_max is not None:
                details.append(f"visible: lost for {lost_s:.1f}s (limit {fail_time_max:.1f}s)")
            elif lost_s is not None:
                details.append(f"visible: lost for {lost_s:.1f}s")
            else:
                details.append("visible: lost for too long")
            continue
        if reason == "time-to-fail gate":
            if fail_time_max is not None:
                details.append(f"time-to-fail: {elapsed_s:.1f}s > {fail_time_max:.1f}s")
            else:
                details.append(f"time-to-fail: {elapsed_s:.1f}s")
            continue

        metric = reason.replace(" gate", "")
        stats = success_metrics.get(metric)
        value = values.get(metric)
        if stats is None or value is None:
            details.append(reason)
            continue

        if metric == "offset_abs":
            max_val = stats.get("max")
            if max_val is not None:
                delta = value - max_val
                pct = _format_percent(delta, max_val)
                pct_str = f"{pct:.0f}% outside tolerance" if pct is not None else "outside tolerance"
                details.append(f"x-offset within \u00b1{max_val:.1f}mm, got {value:.1f}mm ({pct_str})")
            continue
        if metric == "angle_abs":
            max_val = stats.get("max")
            if max_val is not None:
                max_val *= ANGLE_TOLERANCE_MULTIPLIER
                delta = value - max_val
                pct = _format_percent(delta, max_val)
                pct_str = f"{pct:.0f}% outside tolerance" if pct is not None else "outside tolerance"
                details.append(f"angle within \u00b1{max_val:.1f}\u00b0, got {value:.1f}\u00b0 ({pct_str})")
            continue
        if metric == "dist":
            max_val = stats.get("max")
            if max_val is not None:
                delta = value - max_val
                pct = _format_percent(delta, max_val)
                pct_str = f"{pct:.0f}% outside tolerance" if pct is not None else "outside tolerance"
                details.append(f"distance \u2264 {max_val:.0f}mm, got {value:.0f}mm ({pct_str})")
            continue
        if metric == "confidence":
            min_val = stats.get("min")
            if min_val is not None:
                delta = min_val - value
                pct = _format_percent(delta, min_val)
                pct_str = f"{pct:.0f}% below tolerance" if pct is not None else "below tolerance"
                details.append(f"confidence \u2265 {min_val:.0f}, got {value:.0f} ({pct_str})")
            continue
        if metric == "lift_height":
            min_val = stats.get("min")
            max_val = stats.get("max")
            if min_val is not None and max_val is not None:
                details.append(f"lift height {min_val:.1f}–{max_val:.1f}mm, got {value:.1f}mm")
            continue

        details.append(reason)

    return details

def metric_status(value, success_stats, failure_stats, direction):
    if success_stats is None or direction is None:
        return "unknown"
    if direction == "low":
        success_max = success_stats.get("max")
        failure_max = failure_stats.get("max") if failure_stats else None
        if success_max is not None and value <= success_max:
            return "success"
        if failure_max is not None and success_max is not None and failure_max <= success_max:
            failure_max = success_max * 1.25
        if failure_max is not None and value >= failure_max:
            return "fail"
        return "correct"
    if direction == "high":
        success_min = success_stats.get("min")
        failure_max = failure_stats.get("max") if failure_stats else None
        if success_min is not None and value >= success_min:
            return "success"
        if failure_max is not None and value <= failure_max:
            return "fail"
        return "correct"

    # band
    success_min = success_stats.get("min")
    success_max = success_stats.get("max")
    if success_min is not None and success_max is not None and success_min <= value <= success_max:
        return "success"
    if failure_stats:
        fail_min = failure_stats.get("min")
        fail_max = failure_stats.get("max")
        if fail_min is not None and fail_max is not None and (value < fail_min or value > fail_max):
            return "fail"
    return "correct"

def _resolve_start_gate_metrics(learned_rules, obj_name, fallback_obj=None):
    metrics = learned_rules.get(obj_name, {}).get("gates", {}).get("success", {}).get("metrics", {})
    if metrics:
        return metrics
    if fallback_obj:
        return learned_rules.get(fallback_obj, {}).get("gates", {}).get("success", {}).get("metrics", {})
    return {}

def _format_start_gate_reasons(reasons):
    return "; ".join(reasons) if reasons else ""

def evaluate_start_gates(world, step, learned_rules):
    obj_name = step.value if isinstance(step, StepState) else step
    reasons = []
    brick = world.brick or {}
    visible = bool(brick.get("visible"))
    confidence = brick.get("confidence", 0.0) or 0.0

    if obj_name in ("ALIGN", "SCOOP"):
        if not visible:
            reasons.append("brick not visible")
        elif confidence < START_GATE_MIN_CONFIDENCE:
            reasons.append(f"confidence<{START_GATE_MIN_CONFIDENCE:.0f}")

    if obj_name == "SCOOP" and visible and confidence >= START_GATE_MIN_CONFIDENCE:
        dist = brick.get("dist")
        angle = abs(brick.get("angle", 0.0))
        offset = abs(brick.get("offset_x", 0.0))

        corridor = world.get_scoop_corridor_limits(dist) if dist else None
        scoop_metrics = _resolve_start_gate_metrics(learned_rules, "SCOOP", fallback_obj="ALIGN")

        max_angle = None
        max_offset = None
        if corridor:
            max_angle = corridor.get("max_angle")
            max_offset = corridor.get("max_offset_x")
        else:
            max_angle = (scoop_metrics.get("angle_abs") or {}).get("max", world.align_tol_angle)
            max_offset = (scoop_metrics.get("offset_abs") or {}).get("max", world.align_tol_offset)

        dist_min = world.align_tol_dist_min
        dist_max = (scoop_metrics.get("dist") or {}).get("max", world.align_tol_dist_max)

        if dist is None or dist <= 0:
            reasons.append("distance unknown")
        else:
            if dist_min is not None and dist < dist_min:
                reasons.append(f"dist<{dist_min:.0f}mm")
            if dist_max is not None and dist > dist_max:
                reasons.append(f"dist>{dist_max:.0f}mm")
        if max_angle is not None and angle > max_angle:
            reasons.append(f"angle>{max_angle:.1f}deg")
        if max_offset is not None and offset > max_offset:
            reasons.append(f"offset>{max_offset:.1f}mm")

    if obj_name == "LIFT":
        if not brick.get("seated"):
            reasons.append("brick not seated")

    if obj_name == "PLACE":
        if world.wall_origin is None:
            reasons.append("wall origin unset")

    return {"ok": not reasons, "reasons": reasons}

def scoop_blind_window_active(world):
    if not world:
        return False
    blind = world.learned_rules.get("SCOOP", {}).get("blind", {})
    time_gate = blind.get("time_s") or {}
    if time_gate.get("samples", 0) < 1:
        return False
    max_time = time_gate.get("max")
    if max_time is None or world.last_visible_time is None:
        return False
    return (time.time() - world.last_visible_time) <= max_time

def scoop_commit_speed(world):
    if not world:
        return SCOOP_APPROACH_SPEED
    commit = world.learned_rules.get("SCOOP", {}).get("commit", {})
    if commit.get("speed") is not None:
        return max(0.0, min(1.0, commit["speed"]))
    blind = world.learned_rules.get("SCOOP", {}).get("blind", {})
    blind_speed = (blind.get("speed") or {}).get("mu")
    if blind_speed is not None:
        return max(0.0, min(1.0, blind_speed))
    return SCOOP_APPROACH_SPEED

def evaluate_phase_gates(world, step, elapsed_s, learned_rules):
    obj_name = step.value if isinstance(step, StepState) else step
    gates = learned_rules.get(obj_name, {}).get("gates", {})
    success_metrics = gates.get("success", {}).get("metrics", {})
    failure_metrics = gates.get("failure", {}).get("metrics", {})
    temporal = gates.get("temporal", {})
    metric_list = STEP_METRICS.get(obj_name, ())
    now = time.time()
    fail_time = temporal.get("fail_time_s")
    fail_time_gate = None
    if fail_time and fail_time.get("samples", 0) >= GATE_MIN_SAMPLES:
        fail_time_gate = fail_time.get("max")
        if fail_time_gate is not None:
            fail_time_gate = max(fail_time_gate, FAIL_TIME_OVERRIDE_S)
    if fail_time_gate is None and obj_name in ("FIND", "SCOOP"):
        fail_time_gate = FAIL_TIME_OVERRIDE_S

    brick = world.brick or {}
    brick_visible = bool(brick.get("visible"))
    values = {
        "angle_abs": abs(brick.get("angle", 0.0)),
        "offset_abs": abs(brick.get("offset_x", 0.0)),
        "dist": brick.get("dist", 0.0),
        "confidence": brick.get("confidence", 0.0),
        "visible": 1.0 if brick_visible else 0.0,
        "lift_height": getattr(world, "lift_height", 0.0),
    }

    success_ok = True
    correction = False
    failure = False
    reasons = []

    for metric in metric_list:
        if obj_name == "SCOOP" and metric == "visible":
            continue
        if metric == "visible" and obj_name == "FIND" and not brick_visible:
            success_ok = False
            correction = True
            if fail_time_gate is not None:
                if elapsed_s >= fail_time_gate:
                    failure = True
                    reasons.append("visible gate (time-to-fail)")
                elif world.last_visible_time and (now - world.last_visible_time) >= fail_time_gate:
                    failure = True
                    reasons.append("lost visibility gate")
            continue
        if metric in VISIBILITY_REQUIRED_METRICS and not brick_visible:
            success_ok = False
            continue
        success_stats = success_metrics.get(metric)
        if not success_stats or success_stats.get("samples", 0) < 1:
            success_ok = False
            continue
        if metric == "angle_abs":
            success_stats = dict(success_stats)
            if success_stats.get("max") is not None:
                success_stats["max"] *= ANGLE_TOLERANCE_MULTIPLIER
        status = metric_status(
            values.get(metric, 0.0),
            success_stats,
            failure_metrics.get(metric),
            METRIC_DIRECTIONS.get(metric),
        )
        if status == "fail":
            if metric == "dist" and obj_name in ("FIND", "SCOOP"):
                correction = True
                success_ok = False
            elif obj_name in ("FIND", "SCOOP") and fail_time_gate is not None and elapsed_s < fail_time_gate:
                correction = True
                success_ok = False
            else:
                failure = True
                success_ok = False
                reasons.append(f"{metric} gate")
        elif status == "correct":
            correction = True
            success_ok = False
        elif status != "success":
            success_ok = False

    min_success_time = None
    success_time = temporal.get("success_time_s")
    if success_time and success_time.get("samples", 0) >= GATE_MIN_SAMPLES:
        min_success_time = max(0.0, success_time.get("min", 0.0))
        max_success_time = success_time.get("max")
        if max_success_time is not None and elapsed_s > max_success_time:
            correction = True

    if fail_time_gate is not None and elapsed_s > fail_time_gate:
        failure = True
        reasons.append("time-to-fail gate")

    return {
        "success_ok": success_ok,
        "correction": correction,
        "fail": failure,
        "reason": "; ".join(reasons) if reasons else None,
        "min_success_time": min_success_time,
    }

def should_commit_scoop(world):
    if not world:
        return False
    commit = world.learned_rules.get("SCOOP", {}).get("commit", {})
    commit_time = commit.get("time_s")
    commit_dist = commit.get("max_dist")
    if not commit_time or not commit_dist:
        return False
    if world.last_align_time is None or world.last_align_dist is None:
        return False
    if world.brick.get("visible"):
        return False
    if time.time() - world.last_align_time > commit_time:
        return False
    if world.last_align_dist > commit_dist:
        return False
    return True

def compute_target_velocity(brick_visible, angle, offset_x, dist, step_state, world=None, correction=False):
    """
    Compute target velocity vector based on current perception.
    Returns: {'linear': forward_speed, 'angular': turn_speed}
    Range: -1.0 to 1.0 for both
    """
    if brick_visible:
        tol_ang = 5.0
        tol_off = 12.0
        if step_state == StepState.SCOOP and world:
            corridor = world.get_scoop_corridor_limits(dist)
            if corridor:
                tol_off = corridor.get("max_offset_x", tol_off)
                tol_ang = corridor.get("max_angle", tol_ang)
        aligned = abs(angle) <= tol_ang and abs(offset_x) <= tol_off
        if step_state == StepState.SCOOP:
            linear = SCOOP_APPROACH_SPEED if aligned else SCOOP_ALIGN_SPEED
        else:
            linear = 0.15  # Slow forward motion
        
        # Angular velocity proportional to angle error
        angular = 0.0
        if abs(angle) > 5.0:
            # Turn to align (proportional control)
            angular = max(-0.4, min(0.4, angle / 90.0))
        elif abs(offset_x) > 12.0:
            # Adjust for offset
            angular = max(-0.2, min(0.2, offset_x / 100.0))
        
        if correction:
            linear *= 0.7
            angular *= 1.2
        linear = max(0.0, min(1.0, linear))
        angular = max(-0.6, min(0.6, angular))
        return {'linear': linear, 'angular': angular * TURN_SIGN}
    else:
        # No brick - slow scanning turn unless we are committing the scoop
        if step_state == StepState.SCOOP and (should_commit_scoop(world) or scoop_blind_window_active(world)):
            commit_speed = scoop_commit_speed(world)
            return {'linear': commit_speed, 'angular': 0.0}
        if step_state == StepState.SCOOP:
            return {'linear': 0.0, 'angular': 0.25 * TURN_SIGN}  # Slower scan during scoop
        return {'linear': 0.0, 'angular': 0.3 * TURN_SIGN}  # Turn to scan

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
        return ('l' if TURN_SIGN > 0 else 'r'), 0.1

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
    
    print("Continuous velocity control @ 20Hz")
    warmup_start = time.time()
    while time.time() - warmup_start < 2.0:
        with app_state.lock:
            if app_state.current_frame is not None: break
        time.sleep(0.1)

    print(f"\n{'='*60}")
    print(f"CONTINUOUS VELOCITY CONTROL - Smooth flow toward target")
    print(f"{'='*60}")
    
    steps = [StepState.FIND, StepState.SCOOP, StepState.LIFT, StepState.PLACE]
    step_index = 0
    with app_state.lock:
        app_state.active_step = steps[step_index].value
        app_state.status_msg = f"Step: {steps[step_index].value}"
        app_state.world.step_state = steps[step_index]
    
    app_state.logger.log_keyframe("JOB_START")
    app_state.logger.log_keyframe("OBJ_START", app_state.world.step_state.value)
    
    # Control parameters
    CONTROL_HZ = 20  # 20 Hz control loop
    CONTROL_DT = 1.0 / CONTROL_HZ
    
    # State variables
    current_velocity = {'linear': 0.0, 'angular': 0.1 * TURN_SIGN}  # Start with slow turn
    run_start = time.time()
    step_start = run_start
    cycle_count = 0
    last_log_time = run_start
    
    # Failure tracking
    failure_reason = None
    
    print(f"  └─ Running at {CONTROL_HZ}Hz | Min speed: 10% | Velocity smoothing: EMA")
    print("  └─ Failure detection: Learned gates + temporal envelopes")
    
    # MAIN CONTROL LOOP - Fixed frequency
    failure_step = app_state.world.step_state.value
    while app_state.running:
        cycle_start = time.time()
        
        if not app_state.running:
            break
        
        # 1. GET TELEMETRY (non-blocking read from shared state)
        with app_state.lock:
            current_step = app_state.world.step_state
            current_step_label = current_step.value
            elapsed = time.time() - step_start
            gate_eval = evaluate_phase_gates(app_state.world, current_step, elapsed, app_state.world.learned_rules)

            # Check completion
            step_done = False
            if current_step in (StepState.LIFT, StepState.PLACE):
                action_duration = LIFT_ACTION_DURATION if current_step == StepState.LIFT else PLACE_ACTION_DURATION
                if elapsed >= action_duration:
                    step_done = True
            elif step_complete(app_state.world):
                step_done = True

            if step_done:
                print(f"\n✓ {current_step_label} Step Complete! (after {cycle_count} cycles)")
                app_state.logger.log_keyframe("OBJ_SUCCESS", current_step_label)
                app_state.robot.stop()
                if step_index < len(steps) - 1:
                    next_obj = steps[step_index + 1]
                    start_eval = evaluate_start_gates(app_state.world, next_obj, app_state.world.learned_rules)
                    if not start_eval["ok"]:
                        reason = _format_start_gate_reasons(start_eval["reasons"])
                        failure_reason = f"start gate blocked: {reason}"
                        failure_step = next_obj.value
                        print(f"Start gate blocked for {step_display_label(next_obj)}: {reason}")
                        break
                    step_index += 1
                    app_state.active_step = next_obj.value
                    app_state.status_msg = f"Step: {next_obj.value}"
                    app_state.world.step_state = next_obj
                    if next_obj == StepState.SCOOP:
                        app_state.world.brick["seated"] = False
                        app_state.world.verification_stage = "IDLE"
                        app_state.world.verify_dist_mm = 0.0
                        app_state.world.verify_turn_deg = 0.0
                        app_state.world.verify_vision_hits = 0
                    elif next_obj == StepState.PLACE:
                        app_state.world.brick["seated"] = True
                    app_state.logger.log_keyframe("OBJ_START", next_obj.value)
                    step_start = time.time()
                    current_velocity = {'linear': 0.0, 'angular': 0.1 * TURN_SIGN}
                    continue
                break
            
            # Get current perception
            brick_visible = app_state.world.brick['visible']
            angle = app_state.world.brick['angle']
            offset_x = app_state.world.brick['offset_x']
            dist = app_state.world.brick['dist']
            verification_cmd = get_scoop_verification_command(app_state.world)
        correction_mode = gate_eval.get("correction", False)
        if gate_eval.get("fail"):
            readable_reason = humanize_failure_reason(gate_eval.get("reason"))
            print(f"Failure detected for {step_display_label(current_step)}: {readable_reason} (elapsed {elapsed:.1f}s)")
            for detail in format_failure_details(app_state.world, current_step, gate_eval, app_state.world.learned_rules, elapsed):
                print(f"  - {detail}")
            failure_reason = readable_reason
            failure_step = current_step_label
            break
        
        if current_step in (StepState.LIFT, StepState.PLACE):
            cmd = 'u' if current_step == StepState.LIFT else 'd'
            speed = LIFT_ACTION_SPEED if current_step == StepState.LIFT else PLACE_ACTION_SPEED
            app_state.robot.send_command(cmd, speed)
            evt = MotionEvent(cmd_to_motion_type(cmd), int(speed * 255), int(CONTROL_DT * 1000))
            with app_state.lock:
                app_state.world.update_from_motion(evt)
                action_label = "LIFTING" if current_step == StepState.LIFT else "LOWERING"
                app_state.status_msg = f"{current_step_label}: {action_label}"
            app_state.logger.log_event(evt, current_step_label)
        elif verification_cmd:
            cmd, speed, reason = verification_cmd
            actual_speed = GEAR_1_SPEED * speed
            app_state.robot.send_command(cmd, actual_speed)
            evt = MotionEvent(cmd_to_motion_type(cmd), int(speed * 255), int(CONTROL_DT * 1000))
            with app_state.lock:
                app_state.world.update_from_motion(evt)
                app_state.status_msg = f"SCOOP {reason}"
            app_state.logger.log_event(evt, current_step_label)
        else:
            # 2. COMPUTE TARGET VELOCITY VECTOR
            target_velocity = compute_target_velocity(brick_visible, angle, offset_x, dist, current_step, app_state.world, correction=correction_mode)
            
            # 3. SMOOTH VELOCITY (blend with previous)
            smoothed_velocity = smooth_velocity(current_velocity, target_velocity, alpha=0.3)
            
            # 4. CONVERT TO COMMAND
            cmd, speed = velocity_to_command(smoothed_velocity)
            
            # 5. SEND COMMAND (non-blocking)
            actual_speed = GEAR_1_SPEED * speed
            app_state.robot.send_command(cmd, actual_speed)
            evt = MotionEvent(cmd_to_motion_type(cmd), int(speed * 255), int(CONTROL_DT * 1000))
            with app_state.lock:
                app_state.world.update_from_motion(evt)
                app_state.status_msg = "ALIGNING" if brick_visible else "SCANNING"
            app_state.logger.log_event(evt, current_step_label)
        
        # Update state
        if current_step not in (StepState.LIFT, StepState.PLACE) and not verification_cmd:
            current_velocity = smoothed_velocity
        cycle_count += 1
        
        # Periodic logging (every 2 seconds)
        if time.time() - last_log_time > 2.0:
            if current_step in (StepState.LIFT, StepState.PLACE):
                status = "LIFTING" if current_step == StepState.LIFT else "LOWERING"
                vel_str = f"cmd={cmd}@{speed:.2f}"
            elif verification_cmd:
                status = "VERIFYING"
                vel_str = f"cmd={cmd}@{speed:.2f}"
            else:
                status = "ALIGNING" if brick_visible else "SCANNING"
                vel_str = f"v=({smoothed_velocity['linear']:.2f}, {smoothed_velocity['angular']:.2f})"
            print(f"  {status} | {cmd}@{speed:.2f} | {vel_str} | {cycle_count} cycles")
            last_log_time = time.time()
        
        # 6. MAINTAIN FIXED FREQUENCY (non-blocking sleep)
        cycle_elapsed = time.time() - cycle_start
        sleep_time = CONTROL_DT - cycle_elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    total_time = time.time() - run_start
    
    # Clean shutdown with failure logging
    app_state.robot.stop()
    
    if failure_reason:
        print(f"\n✗ {failure_step} Step FAILED: {failure_reason}")
        app_state.logger.log_keyframe("FAIL_START", failure_step)
        app_state.logger.log_keyframe("FAIL_END", failure_step)
    
    app_state.logger.log_keyframe("JOB_SUCCESS")  # Job completes even if step failed
    
    avg_hz = cycle_count / total_time if total_time > 0 else 0
    status_icon = "✗" if failure_reason else "✓"
    print(f"\n{status_icon} Run complete: {cycle_count} cycles in {total_time:.1f}s (avg {avg_hz:.1f}Hz)")
    app_state.logger.close()
    app_state.running = False

def main_autoplay(session_name, scenarios=None):
    if scenarios is None:
        scenarios = [("SUCCESS", 60)]
    scenario_names = " -> ".join(name for name, _ in scenarios)
    print("\nTraining mode")
    print(f"  Sequence: {scenario_names}\n")
    
    learned_rules = aggregate_heuristics()
    demo_attempts = None
    demo_fail = None
    demo_recover = None
    demo_success_by_obj = {}
    demo_recover_by_obj = {}
    needs_fail = any(name == "FAIL" for name, _ in scenarios)
    needs_success = any(name not in ("FAIL", "RECOVER") for name, _ in scenarios)
    needs_recover = any(name == "RECOVER" for name, _ in scenarios) or needs_success
    if SMART_REPLAY and session_name and (needs_fail or needs_recover or needs_success):
        _, _, demo_attempts = load_demo(session_name, include_attempts=True)
        if needs_fail:
            demo_fail = pick_best_attempt(demo_attempts, "FAIL")
        if needs_recover:
            demo_recover = pick_best_attempt(demo_attempts, "RECOVER")
        if (needs_success or needs_recover) and demo_attempts:
            for obj in (StepState.FIND, StepState.SCOOP, StepState.LIFT, StepState.PLACE):
                demo_success_by_obj[obj.value] = pick_best_attempt_for_step(
                    demo_attempts,
                    "SUCCESS",
                    obj.value
                )
                demo_recover_by_obj[obj.value] = pick_best_attempt_for_step(
                    demo_attempts,
                    "RECOVER",
                    obj.value
                )
    
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

    def run_recover_for_step(obj_enum):
        print(f"  → Recovery for {obj_enum.value} (timeout={RECOVER_TIMEOUT}s)")
        demo_recover_seq = demo_recover_by_obj.get(obj_enum.value) or demo_recover
        if demo_recover_seq:
            success = replay_demo_attempt(
                demo_recover_seq,
                obj_enum.value,
                "RECOVERY",
                "RECOVER_START",
                "RECOVER_END"
            )
            with app_state.lock:
                app_state.world.attempt_status = "NORMAL"
            return success

        app_state.logger.log_keyframe("RECOVER_START", obj_enum.value)
        with app_state.lock:
            app_state.world.step_state = obj_enum
            app_state.active_step = obj_enum.value
            app_state.status_msg = f"Step: {obj_enum.value} (RECOVERY)"
            app_state.world.attempt_status = "RECOVERY"
            if obj_enum == StepState.SCOOP:
                app_state.world.brick["seated"] = False
                app_state.world.verification_stage = "IDLE"
                app_state.world.verify_dist_mm = 0.0
                app_state.world.verify_turn_deg = 0.0
                app_state.world.verify_vision_hits = 0
            elif obj_enum == StepState.PLACE:
                app_state.world.brick["seated"] = True

        if obj_enum in (StepState.LIFT, StepState.PLACE):
            action_cmd = 'u' if obj_enum == StepState.LIFT else 'd'
            action_speed = LIFT_ACTION_SPEED if obj_enum == StepState.LIFT else PLACE_ACTION_SPEED
            action_duration = LIFT_ACTION_DURATION if obj_enum == StepState.LIFT else PLACE_ACTION_DURATION
            action_start = time.time()

            while time.time() - action_start < action_duration:
                app_state.robot.send_command(action_cmd, action_speed)
                evt = MotionEvent(cmd_to_motion_type(action_cmd), int(action_speed * 255), 50)
                with app_state.lock:
                    app_state.world.update_from_motion(evt)
                    action_label = "LIFTING" if obj_enum == StepState.LIFT else "LOWERING"
                    app_state.status_msg = f"{obj_enum.value}: {action_label}"
                app_state.logger.log_event(evt, obj_enum.value)
                time.sleep(0.05)

            app_state.robot.stop()
            app_state.logger.log_keyframe("OBJ_SUCCESS", obj_enum.value)
            app_state.logger.log_keyframe("RECOVER_END", obj_enum.value)
            with app_state.lock:
                app_state.world.attempt_status = "NORMAL"
            return True

        current_velocity = {'linear': 0.0, 'angular': 0.1 * TURN_SIGN}
        recover_start = time.time()
        cycles = 0
        while time.time() - recover_start < RECOVER_TIMEOUT:
            with app_state.lock:
                if step_complete(app_state.world):
                    app_state.logger.log_keyframe("OBJ_SUCCESS", obj_enum.value)
                    app_state.logger.log_keyframe("RECOVER_END", obj_enum.value)
                    with app_state.lock:
                        app_state.world.attempt_status = "NORMAL"
                    return True

                brick_visible = app_state.world.brick['visible']
                angle = app_state.world.brick['angle']
                offset_x = app_state.world.brick['offset_x']
                dist = app_state.world.brick['dist']
                step_state = app_state.world.step_state
                verification_cmd = get_scoop_verification_command(app_state.world)

            if verification_cmd:
                cmd, speed, reason = verification_cmd
                app_state.robot.send_command(cmd, GEAR_1_SPEED * speed)
                evt = MotionEvent(cmd_to_motion_type(cmd), int(speed * 255), 50)
                with app_state.lock:
                    app_state.world.update_from_motion(evt)
                    app_state.status_msg = f"SCOOP {reason}"
                app_state.logger.log_event(evt, obj_enum.value)
            else:
                target_velocity = compute_target_velocity(brick_visible, angle, offset_x, dist, step_state, app_state.world)
                smoothed_velocity = smooth_velocity(current_velocity, target_velocity, alpha=0.3)
                cmd, speed = velocity_to_command(smoothed_velocity)
                app_state.robot.send_command(cmd, GEAR_1_SPEED * speed)
                evt = MotionEvent(cmd_to_motion_type(cmd), int(speed * 255), 50)
                with app_state.lock:
                    app_state.world.update_from_motion(evt)
                    app_state.status_msg = "ALIGNING" if brick_visible else "SCANNING"
                app_state.logger.log_event(evt, obj_enum.value)
                current_velocity = smoothed_velocity

            cycles += 1
            time.sleep(0.05)

        app_state.robot.stop()
        print(f"Recovery timed out for {step_display_label(obj_enum)} after {RECOVER_TIMEOUT:.1f}s")
        app_state.logger.log_keyframe("FAIL_START", obj_enum.value)
        app_state.logger.log_keyframe("FAIL_END", obj_enum.value)
        with app_state.lock:
            app_state.world.attempt_status = "NORMAL"
        return False

    # Run training scenarios
    total_scenarios = len(scenarios)
    for idx, (name, timeout) in enumerate(scenarios, 1):
        scenario_type = "recover" if name == "RECOVER" else ("fail" if name == "FAIL" else "normal")
        min_success_time = 1.0
        min_success_cycles = 5
        use_demo_fail = scenario_type == "fail" and demo_fail
        use_demo_recover = scenario_type == "recover" and demo_recover
        
        if scenario_type == "fail":
            scenario_label = "Attempting a failed FIND"
        elif scenario_type == "recover":
            scenario_label = "Attempting a recovery FIND"
        else:
            scenario_label = "Attempting a successful FIND"

        print(f"{'='*70}")
        print(f"{scenario_label} (scenario {idx}/{total_scenarios})")
        print(f"{'='*70}\n")
        
        # Log start
        if scenario_type == "recover":
            if not use_demo_recover:
                app_state.logger.log_keyframe("RECOVER_START", "FIND")
        elif scenario_type == "fail":
            app_state.logger.log_keyframe("OBJ_START", "FIND")
        
        with app_state.lock:
            app_state.world.reset_mission()
            app_state.world.step_state = StepState.FIND
            app_state.active_step = StepState.FIND.value
            app_state.status_msg = f"Step: {StepState.FIND.value}"
            app_state.world.attempt_status = "NORMAL"
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
            run_start = time.time()
            scenario_deadline = run_start + timeout
            cycle_count = 0
            success_flag = True
            failed_step = None
            failed_reason = None
            failure_logged = False

            if scenario_type == "normal":
                steps = [StepState.FIND, StepState.SCOOP, StepState.LIFT, StepState.PLACE]
                for obj in steps:
                    attempts = 0
                    step_success = False
                    last_failure_reason = None
                    obj_label = "CARRY" if obj == StepState.LIFT else obj.value
                    while attempts < MAX_STEP_ATTEMPTS and not step_success:
                        attempts += 1
                        with app_state.lock:
                            app_state.world.step_state = obj
                            app_state.active_step = obj.value
                            app_state.status_msg = f"Step: {obj_label} (attempt {attempts}/{MAX_STEP_ATTEMPTS})"
                            if obj == StepState.SCOOP:
                                app_state.world.brick["seated"] = False
                                app_state.world.verification_stage = "IDLE"
                                app_state.world.verify_dist_mm = 0.0
                                app_state.world.verify_turn_deg = 0.0
                                app_state.world.verify_vision_hits = 0
                            elif obj == StepState.PLACE:
                                app_state.world.brick["seated"] = True

                        print(f"{obj_label} (attempt {attempts}/{MAX_STEP_ATTEMPTS})")
                        start_eval = evaluate_start_gates(app_state.world, obj, app_state.world.learned_rules)
                        if not start_eval["ok"]:
                            reason = _format_start_gate_reasons(start_eval["reasons"])
                            print(f"Start gate blocked for {step_display_label(obj)}: {reason}")
                            app_state.logger.log_keyframe("FAIL_START", obj.value)
                            app_state.logger.log_keyframe("FAIL_END", obj.value)
                            failure_logged = True
                            last_failure_reason = f"start gate: {reason}"
                            run_recover_for_step(obj)
                            continue
                        app_state.logger.log_keyframe("OBJ_START", obj.value)
                        current_velocity = {'linear': 0.0, 'angular': 0.1 * TURN_SIGN}
                        obj_start = time.time()
                        obj_cycles = 0
                        demo_success = demo_success_by_obj.get(obj.value)

                        while True:
                            if demo_success and obj in (StepState.LIFT, StepState.PLACE):
                                print(f"  → Replaying SUCCESS demo for {obj.value}")
                                if replay_demo_attempt(demo_success, obj.value, "NORMAL", "SUCCESS_START", "SUCCESS_END"):
                                    app_state.logger.log_keyframe("OBJ_SUCCESS", obj.value)
                                    step_success = True
                                break

                            if obj in (StepState.LIFT, StepState.PLACE):
                                action_cmd = 'u' if obj == StepState.LIFT else 'd'
                                action_speed = LIFT_ACTION_SPEED if obj == StepState.LIFT else PLACE_ACTION_SPEED
                                action_duration = LIFT_ACTION_DURATION if obj == StepState.LIFT else PLACE_ACTION_DURATION
                                action_start = time.time()

                                while time.time() - action_start < action_duration:
                                    app_state.robot.send_command(action_cmd, action_speed)
                                    evt = MotionEvent(cmd_to_motion_type(action_cmd), int(action_speed * 255), 50)
                                    with app_state.lock:
                                        app_state.world.update_from_motion(evt)
                                        action_label = "LIFTING" if obj == StepState.LIFT else "LOWERING"
                                        app_state.status_msg = f"{obj.value}: {action_label}"
                                    app_state.logger.log_event(evt, obj.value)
                                    time.sleep(0.05)

                                app_state.robot.stop()
                                if time.time() - action_start >= action_duration:
                                    app_state.logger.log_keyframe("OBJ_SUCCESS", obj.value)
                                    step_success = True
                                break

                            with app_state.lock:
                                elapsed = time.time() - obj_start
                                gate_eval = evaluate_phase_gates(app_state.world, obj, elapsed, app_state.world.learned_rules)
                                if step_complete(app_state.world):
                                    print(f"  ✓ {obj.value} Step complete! ({obj_cycles} cycles)\n")
                                    app_state.logger.log_keyframe("OBJ_SUCCESS", obj.value)
                                    step_success = True
                                    break

                                brick_visible = app_state.world.brick['visible']
                                angle = app_state.world.brick['angle']
                                offset_x = app_state.world.brick['offset_x']
                                dist = app_state.world.brick['dist']
                                step_state = app_state.world.step_state
                                verification_cmd = get_scoop_verification_command(app_state.world)

                            correction_mode = gate_eval.get("correction", False)
                            if gate_eval.get("fail"):
                                readable_reason = humanize_failure_reason(gate_eval.get("reason"))
                                print(f"Failure detected for {step_display_label(obj)}: {readable_reason} (attempt {attempts}/{MAX_STEP_ATTEMPTS}, elapsed {elapsed:.1f}s)")
                                for detail in format_failure_details(app_state.world, obj, gate_eval, app_state.world.learned_rules, elapsed):
                                    print(f"  - {detail}")
                                last_failure_reason = readable_reason
                                break

                            if verification_cmd:
                                cmd, speed, reason = verification_cmd
                                app_state.robot.send_command(cmd, GEAR_1_SPEED * speed)
                                evt = MotionEvent(cmd_to_motion_type(cmd), int(speed * 255), 50)
                                with app_state.lock:
                                    app_state.world.update_from_motion(evt)
                                    app_state.status_msg = f"SCOOP {reason}"
                                app_state.logger.log_event(evt, obj.value)
                            else:
                                target_velocity = compute_target_velocity(brick_visible, angle, offset_x, dist, step_state, app_state.world, correction=correction_mode)
                                smoothed_velocity = smooth_velocity(current_velocity, target_velocity, alpha=0.3)
                                cmd, speed = velocity_to_command(smoothed_velocity)
                                app_state.robot.send_command(cmd, GEAR_1_SPEED * speed)
                                evt = MotionEvent(cmd_to_motion_type(cmd), int(speed * 255), 50)
                                with app_state.lock:
                                    app_state.world.update_from_motion(evt)
                                    app_state.status_msg = "ALIGNING" if brick_visible else "SCANNING"
                                app_state.logger.log_event(evt, obj.value)
                                current_velocity = smoothed_velocity

                            obj_cycles += 1
                            cycle_count += 1
                            time.sleep(0.05)  # 20Hz

                        app_state.robot.stop()

                        if step_success:
                            break

                        app_state.logger.log_keyframe("FAIL_START", obj.value)
                        app_state.logger.log_keyframe("FAIL_END", obj.value)
                        failure_logged = True
                        if not last_failure_reason:
                            last_failure_reason = "Failure condition met"
                        run_recover_for_step(obj)

                    if not step_success:
                        success_flag = False
                        failed_step = obj.value
                        failed_reason = f"Exceeded {MAX_STEP_ATTEMPTS} attempts"
                        if last_failure_reason:
                            failed_reason += f" (last: {last_failure_reason})"
                        break
            else:
                current_velocity = {'linear': 0.0, 'angular': 0.1 * TURN_SIGN}
                success_flag = False
                while time.time() < scenario_deadline:
                    with app_state.lock:
                        elapsed = time.time() - run_start
                        gate_eval = evaluate_phase_gates(app_state.world, app_state.world.step_state, elapsed, app_state.world.learned_rules)
                        if step_complete(app_state.world):
                            print(f"  ✓ Step complete! ({cycle_count} cycles)\n")
                            app_state.logger.log_keyframe("OBJ_SUCCESS", "FIND")
                            success_flag = True
                            break

                        brick_visible = app_state.world.brick['visible']
                        angle = app_state.world.brick['angle']
                        offset_x = app_state.world.brick['offset_x']
                        dist = app_state.world.brick['dist']
                        step_state = app_state.world.step_state
                        verification_cmd = get_scoop_verification_command(app_state.world)

                    correction_mode = gate_eval.get("correction", False)
                    if gate_eval.get("fail"):
                        readable_reason = humanize_failure_reason(gate_eval.get("reason"))
                        print(f"Failure detected for {step_display_label(app_state.world.step_state)}: {readable_reason} (elapsed {elapsed:.1f}s)")
                        for detail in format_failure_details(app_state.world, app_state.world.step_state, gate_eval, app_state.world.learned_rules, elapsed):
                            print(f"  - {detail}")
                        break
                    if verification_cmd:
                        cmd, speed, reason = verification_cmd
                        app_state.robot.send_command(cmd, GEAR_1_SPEED * speed)
                        evt = MotionEvent(cmd_to_motion_type(cmd), int(speed * 255), 50)
                        with app_state.lock:
                            app_state.world.update_from_motion(evt)
                            app_state.status_msg = f"SCOOP {reason}"
                        app_state.logger.log_event(evt, step_state.value)
                    else:
                        target_velocity = compute_target_velocity(brick_visible, angle, offset_x, dist, step_state, app_state.world, correction=correction_mode)
                        smoothed_velocity = smooth_velocity(current_velocity, target_velocity, alpha=0.3)
                        cmd, speed = velocity_to_command(smoothed_velocity)
                        app_state.robot.send_command(cmd, GEAR_1_SPEED * speed)
                        evt = MotionEvent(cmd_to_motion_type(cmd), int(speed * 255), 50)
                        with app_state.lock:
                            app_state.world.update_from_motion(evt)
                            app_state.status_msg = "ALIGNING" if brick_visible else "SCANNING"
                        app_state.logger.log_event(evt, step_state.value)
                        current_velocity = smoothed_velocity

                    cycle_count += 1
                    time.sleep(0.05)  # 20Hz

                app_state.robot.stop()

            total_time = time.time() - run_start

            # Log result
            if not success_flag:
                if not failed_reason:
                    failed_reason = "Failure conditions not met"
                print(f"  ✗ Failed: {failed_reason}\n")
                if not failure_logged:
                    fail_obj = failed_step or "FIND"
                    app_state.logger.log_keyframe("FAIL_START", fail_obj)
                    app_state.logger.log_keyframe("FAIL_END", fail_obj)
            elif scenario_type == "recover":
                app_state.logger.log_keyframe("RECOVER_END", "FIND")
        
        # Pause between scenarios
        if idx < total_scenarios:
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
