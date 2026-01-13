import argparse
import sys
import threading
import time
import tty
import termios
from pathlib import Path

import autostack
from robot_control import Robot
from helper_demo_log_utils import prune_log_file
from helper_stream_server import StreamServer, format_stream_url
from train_brick_vision import BrickDetector
from telemetry_robot import (
    WorldModel,
    TelemetryLogger,
    MotionEvent,
    ObjectiveState,
    draw_telemetry_overlay,
    objective_sequence,
)

class _LineStartTracker:
    def __init__(self):
        self.at_line_start = True

class LineStartWriter:
    def __init__(self, wrapped, tracker):
        self.wrapped = wrapped
        self.tracker = tracker

    def write(self, data):
        if not data:
            return 0
        if isinstance(data, bytes):
            data = data.decode(errors="replace")
        data = data.replace("\r\n", "\n").replace("\r", "\n")
        if not self.tracker.at_line_start and not data.startswith("\n"):
            data = "\n" + data
        lines = data.splitlines(True)
        line_start = self.tracker.at_line_start
        out = []
        for line in lines:
            if line_start:
                line = line.lstrip(" \t")
            out.append(line)
            line_start = line.endswith("\n")
        output = "".join(out)
        n = self.wrapped.write(output)
        self.tracker.at_line_start = output.endswith("\n")
        return n

    def flush(self):
        return self.wrapped.flush()

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

_line_start_tracker = _LineStartTracker()
sys.stdout = LineStartWriter(sys.stdout, _line_start_tracker)
sys.stderr = LineStartWriter(sys.stderr, _line_start_tracker)
 
# --- CONFIG ---
LOG_RATE_HZ = 10
GEAR_1_SPEED = 0.32  # 4x faster per user request
GEAR_9_SPEED = 1.0   # 100% capacity
HEARTBEAT_TIMEOUT = 0.3 # Stop if no key for 0.3s
STREAM_HOST = "127.0.0.1"
STREAM_PORT = 5000
STREAM_FPS = 10
STREAM_JPEG_QUALITY = 85
DEMOS_DIR = Path(__file__).resolve().parent / "demos"
DEMO_OBJECTIVES = objective_sequence()

OBJECTIVE_CODES = {str(idx + 1): obj for idx, obj in enumerate(DEMO_OBJECTIVES)}

OBJECTIVE_NAMES = {obj.value.lower(): obj for obj in DEMO_OBJECTIVES}
OBJECTIVE_NAMES["wall"] = ObjectiveState.FIND_WALL
OBJECTIVE_NAMES["find"] = ObjectiveState.FIND_BRICK
OBJECTIVE_NAMES["align"] = ObjectiveState.ALIGN_BRICK
OBJECTIVE_NAMES["carry"] = ObjectiveState.FIND_WALL2
OBJECTIVE_NAMES["wall2"] = ObjectiveState.FIND_WALL2
OBJECTIVE_NAMES["position"] = ObjectiveState.POSITION_BRICK

ATTEMPT_CODES = {
    "f": "FAIL",
    "s": "SUCCESS",
    "r": "RECOVER",
}

ATTEMPT_NAMES = {
    "fail": "FAIL",
    "failure": "FAIL",
    "success": "SUCCESS",
    "recover": "RECOVER",
    "recovery": "RECOVER",
}

ATTEMPT_MARKERS = {
    "FAIL": ("FAIL_START", "FAIL_END"),
    "RECOVER": ("RECOVER_START", "RECOVER_END"),
    "SUCCESS": ("SUCCESS_START", "SUCCESS_END"),
}

ATTEMPT_STATUS = {
    "FAIL": "FAIL",
    "RECOVER": "RECOVERY",
    "SUCCESS": "NORMAL",
}

def objective_label(obj_enum):
    return obj_enum.value

def log_line(message):
    print(str(message).strip(), flush=True)

def open_new_log(app_state):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = app_state.demos_dir / f"kbd_{timestamp}.json"
    log_line(f"[SESSION] Recording Keyboard Demo to: {log_path}")
    app_state.logger = TelemetryLogger(log_path)
    app_state.logger.enabled = False  # Wait for first attempt marker
    app_state.logger_closed = False
    app_state.log_path = log_path
    app_state.world.run_id = f"run_{timestamp}"
    app_state.world.attempt_id = 1

def ensure_log_open(app_state):
    if app_state.logger is None or app_state.logger_closed:
        open_new_log(app_state)

def close_log(app_state, marker="JOB_END"):
    if app_state.logger is None or app_state.logger_closed:
        return
    app_state.logger.log_keyframe(marker)
    app_state.logger.enabled = False
    app_state.logger.close()
    app_state.logger_closed = True
    if app_state.log_path:
        prune_log_file(app_state.log_path, delete_if_empty=True)

def build_stream_provider(app_state):
    def _provider():
        with app_state.lock:
            frame = app_state.current_frame
            if frame is None:
                return None
            return frame
    return _provider

def prompt_for_stage(stage, obj_label):
    if stage == "FAIL":
        return f"Press f to start failed {obj_label} demo."
    if stage == "RECOVER":
        return f"Press f to start recovery {obj_label} demo."
    if stage == "SUCCESS":
        return f"Press f to start clean {obj_label} demo."
    return None

def resolve_objective_token(token):
    if not token:
        return None
    key = token.strip().lower()
    if key in OBJECTIVE_CODES:
        return OBJECTIVE_CODES[key]
    if key in OBJECTIVE_NAMES:
        return OBJECTIVE_NAMES[key]
    for name, obj in OBJECTIVE_NAMES.items():
        if name.startswith(key):
            return obj
    return None

def resolve_attempt_token(token):
    if not token:
        return None
    key = token.strip().lower()
    if key in ATTEMPT_CODES:
        return ATTEMPT_CODES[key]
    if key in ATTEMPT_NAMES:
        return ATTEMPT_NAMES[key]
    for name, attempt in ATTEMPT_NAMES.items():
        if name.startswith(key):
            return attempt
    return None

def parse_text_command(text):
    if not text:
        return None, None, None, "Empty command."
    tokens = text.strip().lower().split()
    auto_mode = False
    if tokens and tokens[0] in ("auto", "robot", "bot", "run"):
        auto_mode = True
        tokens = tokens[1:]
    if not tokens:
        return auto_mode, None, None, "Missing objective/attempt."
    if len(tokens) == 1 and len(tokens[0]) >= 2:
        token = tokens[0]
        if token[0].isdigit():
            digits = []
            for ch in token:
                if ch.isdigit():
                    digits.append(ch)
                else:
                    break
            obj_token = "".join(digits)
            attempt_token = token[len(obj_token):]
            if not attempt_token:
                return auto_mode, None, None, "Missing attempt."
        else:
            obj_token = token[0]
            attempt_token = token[1:]
    elif len(tokens) >= 2:
        obj_token = tokens[0]
        attempt_token = tokens[1]
    else:
        return auto_mode, None, None, "Missing attempt."
    obj_enum = resolve_objective_token(obj_token)
    attempt = resolve_attempt_token(attempt_token)
    if not obj_enum or not attempt:
        return auto_mode, obj_enum, attempt, "Unknown objective or attempt."
    return auto_mode, obj_enum, attempt, None

def print_command_help(app_state=None):
    log_line("[CMD] Enter command mode with ':'")
    codes = ", ".join(
        f"{idx + 1}={obj.value.lower()}" for idx, obj in enumerate(DEMO_OBJECTIVES)
    )
    log_line(f"[CMD] Objective codes: {codes}")
    log_line("[CMD] Attempt codes: f=fail, s=success, r=recover")
    log_line("[CMD] Example: :4f (scoop fail)")
    log_line("[CMD] Auto-run: :auto 4f")
    log_line("[CMD] Auto-run: :auto 4f")
    log_line("[CMD] Camera (ranges): :cam exp [-10..0], :cam hue [0..50], :cam sat/val [0..255]")
    log_line("[CMD] Save: :cam save (Required to persist changes!)")
    
    if app_state and app_state.vision:
        v = app_state.vision
        exp = getattr(v, 'cam_exposure', 'N/A')
        c = getattr(v, 'brick_color_config', {})
        h_m = c.get('hue', 'N/A')
        s_m = c.get('sat', 'N/A')
        v_m = c.get('val', 'N/A')
        log_line(f"[STATE] Current: Exp={exp}, Hue={h_m}, Sat={s_m}, Val={v_m}")

    log_line("[CMD] End attempt: repeat the command or use :end")

def command_mode_exit_messages(app_state):
    if app_state.active_attempt:
        return ["[MODE] Manual mode (logging active)."]
    return ["[MODE] Manual mode and not logging."]

def handle_command_line(app_state, cmd):
    cmd_lower = (cmd or "").strip().lower()
    messages = []
    do_help = False
    exit_mode = False
    ended_info = None
    should_close = False

    if cmd_lower in ("", ":"):
        exit_mode = True
        return exit_mode, do_help, messages, ended_info, should_close
    if cmd_lower in ("q", "quit", "exit"):
        with app_state.lock:
            app_state.running = False
        messages.append("Stopping manual recording...")
        exit_mode = True
        return exit_mode, do_help, messages, ended_info, should_close

    with app_state.lock:
        if cmd_lower in ("help", "h", "?"):
            do_help = True
        elif cmd_lower in ("status", "state"):
            obj = app_state.world.objective_state
            attempt = app_state.active_attempt or "NONE"
            messages.append(f"[STATE] Objective={objective_label(obj)} Attempt={attempt}")
        elif cmd_lower in ("end", "stop", "done"):
            ok, msg, obj_enum, attempt_type, should_close = end_attempt(app_state)
            messages.append(msg)
            if ok:
                ended_info = (obj_enum, attempt_type)
                ended_info = (obj_enum, attempt_type)
        elif cmd_lower.startswith("cam ") or cmd_lower.startswith(":cam "):
             # Cam Tuning
             parts = cmd_lower.split()
             # Skip ':cam' or 'cam'
             if parts[0] in (":cam", "cam"): parts = parts[1:]
             
             if not parts:
                 messages.append("[CAM] Usage: :cam [exp|hue|sat|val] [value] OR :cam save")
             else:
                 op = parts[0]
                 val = int(parts[1]) if len(parts) > 1 and parts[1].lstrip('-').isdigit() else None
                 
                 if op == "save":
                     if app_state.vision.save_settings():
                         messages.append("[CAM] Settings Saved to world_model_brick.json!")
                     else:
                         messages.append("[CAM] Save Failed.")
                 elif val is None:
                     messages.append(f"[CAM] Missing value for {op}.")
                 elif op in ("exp", "exposure"):
                     app_state.vision.update_settings(exposure=val)
                     messages.append(f"[CAM] Exposure set to {val}")
                 elif op in ("hue", "h"):
                     app_state.vision.update_settings(hue_margin=val)
                     messages.append(f"[CAM] Hue Margin set to {val}")
                 elif op in ("sat", "s"):
                     app_state.vision.update_settings(sat_margin=val)
                     messages.append(f"[CAM] Saturation Margin set to {val}")
                 elif op in ("val", "v"):
                     app_state.vision.update_settings(val_margin=val)
                     messages.append(f"[CAM] Value Margin set to {val}")
                 else:
                     messages.append(f"[CAM] Unknown property: {op}")
        else:
            auto_mode, obj_enum, attempt_type, err = parse_text_command(cmd)
            if err:
                messages.append(f"[CMD] {err}")
            elif auto_mode:
                if app_state.autostack_active or app_state.autostack_request:
                    messages.append("[AUTO] Autostack already running.")
                elif app_state.objective_open and app_state.open_objective != obj_enum:
                    messages.append(f"[OBJ] Finish {objective_label(app_state.open_objective)} before switching objectives.")
                else:
                    app_state.autostack_request = (obj_enum, attempt_type)
                    messages.append(f"[AUTO] Queued {objective_label(obj_enum)} {attempt_type}.")
                    exit_mode = True
            else:
                if app_state.autostack_active or app_state.autostack_request:
                    messages.append("[AUTO] Autostack pending; wait to issue manual commands.")
                else:
                    ok, msg, ended, should_close = handle_attempt_command(app_state, obj_enum, attempt_type)
                    messages.append(msg)
                    if ended:
                        ended_info = ended
                    if ok and app_state.active_attempt:
                        messages.append("[CMD] Press ':' when finished recording.")
                        exit_mode = True

    return exit_mode, do_help, messages, ended_info, should_close


def prompt_keep_discard(obj_enum, attempt_type):
    obj_label = objective_label(obj_enum)
    prompt = f"[KEEP] Keep {attempt_type} attempt for {obj_label}? [y/N]: "
    while True:
        try:
            answer = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if answer in ("y", "yes"):
            return True
        if answer in ("", "n", "no", "d", "discard"):
            return False
        print("Please enter y or n.")


def record_attempt_discard(app_state, obj_enum, attempt_type):
    if app_state.logger_closed:
        return
    marker = f"{attempt_type}_DISCARD"
    prev_enabled = app_state.logger.enabled
    app_state.logger.log_keyframe(marker, objective_label(obj_enum))
    app_state.logger.enabled = prev_enabled

def start_attempt(app_state, obj_enum, attempt_type):
    obj_label = objective_label(obj_enum)
    if app_state.objective_open and app_state.open_objective != obj_enum:
        return False, f"[OBJ] Finish {objective_label(app_state.open_objective)} before switching objectives."
    ensure_log_open(app_state)
    if not app_state.objective_open:
        app_state.objective_open = True
        app_state.open_objective = obj_enum
    app_state.world.objective_state = obj_enum
    marker = ATTEMPT_MARKERS[attempt_type][0]
    app_state.logger.log_keyframe(marker, obj_label)
    app_state.active_attempt = attempt_type
    app_state.world.attempt_status = ATTEMPT_STATUS[attempt_type]
    app_state.world.recording_active = True
    return True, f"[OBJ] {obj_label} {attempt_type} started."

def end_attempt(app_state, complete_objective=True):
    if not app_state.active_attempt:
        return False, "[OBJ] No active attempt.", None, None, False
    obj_enum = app_state.world.objective_state
    obj_label = objective_label(obj_enum)
    attempt_type = app_state.active_attempt
    marker = ATTEMPT_MARKERS[attempt_type][1]
    app_state.logger.log_keyframe(marker, obj_label)
    should_close = False

    if attempt_type == "SUCCESS":
        if complete_objective:
            app_state.objective_open = False
            app_state.open_objective = None
            current_obj = app_state.world.objective_state
            app_state.world.reset_mission()
            app_state.world.objective_state = current_obj
            should_close = True
        app_state.world.recording_active = False
    else:
        app_state.world.recording_active = False

    app_state.world.attempt_status = "NORMAL"
    app_state.active_attempt = None
    app_state.logger.enabled = False
    return True, f"[OBJ] {obj_label} attempt ended.", obj_enum, attempt_type, should_close

def handle_attempt_command(app_state, obj_enum, attempt_type):
    obj_label = objective_label(obj_enum)
    if app_state.objective_open and app_state.open_objective != obj_enum:
        return False, f"[OBJ] Finish {objective_label(app_state.open_objective)} before switching objectives.", None, False

    if app_state.active_attempt == attempt_type:
        ok, msg, ended_obj, ended_attempt, should_close = end_attempt(app_state)
        ended_info = (ended_obj, ended_attempt) if ok else None
        return ok, msg, ended_info, should_close

    if app_state.active_attempt:
        ok, msg, ended_obj, ended_attempt, should_close = end_attempt(app_state)
        ended_info = (ended_obj, ended_attempt) if ok else None
        ended_close = should_close
    else:
        ended_info = None
        ended_close = False

    ok, msg = start_attempt(app_state, obj_enum, attempt_type)
    return ok, msg, ended_info, ended_close

class AppState:
    def __init__(self):
        self.running = True
        self.active_command = None
        self.active_speed = 0.0
        self.last_key_time = 0
        self.turn_speed_multiplier = 1.0
        
        # Job Status
        self.job_success = False
        self.job_success_timer = 0
        self.job_start = False
        self.job_start_timer = 0
        self.job_abort = False
        self.job_abort_timer = 0
        
        # Job Status
        
        self.lock = threading.Lock()
        self.current_frame = None
        self.objective_open = False
        self.open_objective = None
        self.active_attempt = None
        self.autostack_request = None
        self.autostack_active = False
        self.pending_review = None
        
        # Session Setup
        self.demos_dir = DEMOS_DIR
        self.demos_dir.mkdir(parents=True, exist_ok=True)
        self.world = WorldModel()
        self.logger = None
        self.logger_closed = True
        self.log_path = None
        open_new_log(self)
        
        # ID Init
        self.world.objective_state = DEMO_OBJECTIVES[0]

        self.vision = None
        self.robot = None
        
        self.config_mtime = 0
        self.last_config_check = 0

def getch():
    """Reads a single character from stdin in raw mode."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def keyboard_thread(app_state):
    while app_state.running:
        pending_review = None
        with app_state.lock:
            if app_state.pending_review:
                pending_review = app_state.pending_review
                app_state.pending_review = None
        if pending_review:
            obj_enum, attempt_type, should_close = pending_review
            keep = prompt_keep_discard(obj_enum, attempt_type)
            if not keep:
                with app_state.lock:
                    record_attempt_discard(app_state, obj_enum, attempt_type)
                log_line(f"[KEEP] Discarded {attempt_type} attempt for {objective_label(obj_enum)}.")
            else:
                log_line(f"[KEEP] Keeping {attempt_type} attempt for {objective_label(obj_enum)}.")
            if should_close:
                with app_state.lock:
                    close_log(app_state, marker="JOB_END")

        ch = getch()
        if not ch:
            continue
        ch_lower = ch.lower()

        messages = []
        if ch_lower == 'q':
            with app_state.lock:
                app_state.last_key_time = time.time()
                app_state.running = False
            messages.append("Stopping manual recording...")
        elif ch_lower == ':':
            with app_state.lock:
                app_state.last_key_time = time.time()
                app_state.active_command = None
                app_state.active_speed = 0.0
            log_line("[CMD] Enter command (ex: 4f, auto 4f, help). Use ':' or blank to exit.")
            
            # Force enable ECHO and ICANON so input() is visible
            fd_term = sys.stdin.fileno()
            attr_term = termios.tcgetattr(fd_term)
            attr_term[3] |= termios.ECHO | termios.ICANON
            termios.tcsetattr(fd_term, termios.TCSANOW, attr_term)
            
            while app_state.running:
                try:
                    cmd = input("[CMD] > ")
                except (EOFError, KeyboardInterrupt):
                    cmd = ""
                exit_mode, do_help, messages, ended_info, should_close = handle_command_line(app_state, cmd)
                if do_help:
                    print_command_help(app_state)
                for msg in messages:
                    log_line(msg)
                if ended_info:
                    obj_enum, attempt_type = ended_info
                    keep = prompt_keep_discard(obj_enum, attempt_type)
                    if not keep:
                        with app_state.lock:
                            record_attempt_discard(app_state, obj_enum, attempt_type)
                        log_line(f"[KEEP] Discarded {attempt_type} attempt for {objective_label(obj_enum)}.")
                    else:
                        log_line(f"[KEEP] Keeping {attempt_type} attempt for {objective_label(obj_enum)}.")
                    if should_close:
                        with app_state.lock:
                            close_log(app_state, marker="JOB_END")
                if exit_mode:
                    for msg in command_mode_exit_messages(app_state):
                        log_line(msg)
                    break
        else:
            with app_state.lock:
                app_state.last_key_time = time.time()
                if app_state.autostack_active or app_state.autostack_request:
                    if ch_lower in ('w', 's', 'a', 'd', 'z', 'c', 'p', 'l'):
                        messages.append("[AUTO] Autostack active; manual controls paused.")
                else:
                    # MOVEMENT (Heartbeat triggers)
                    if ch_lower == 'w':
                        app_state.active_command = 'f'
                    elif ch_lower == 's':
                        app_state.active_command = 'b'
                    elif ch_lower == 'a':
                        app_state.active_command = 'l'
                        app_state.turn_speed_multiplier = 1.0
                    elif ch_lower == 'd':
                        app_state.active_command = 'r'
                        app_state.turn_speed_multiplier = 0.5
                    elif ch_lower == 'z':
                        app_state.active_command = 'l'
                        app_state.turn_speed_multiplier = 2.0
                    elif ch_lower == 'c':
                        app_state.active_command = 'r'
                        app_state.turn_speed_multiplier = 2.0
                    elif ch_lower == 'p':
                        app_state.active_command = 'u'
                    elif ch_lower == 'l':
                        app_state.active_command = 'd'
                    elif ch_lower in ('h', '?'):
                        print_command_help(app_state)

        for msg in messages:
            log_line(msg)

        if not app_state.running:
            break

def run_auto_attempt(app_state, obj_enum, attempt_type):
    with app_state.lock:
        ok, msg = start_attempt(app_state, obj_enum, attempt_type)
    log_line(msg)
    if not ok:
        return False

    def frame_callback(frame):
        if frame is None:
            return
        frame_copy = frame.copy()
        draw_telemetry_overlay(frame_copy, app_state.world, show_prompt=False)
        with app_state.lock:
            app_state.current_frame = frame_copy

    ok, info = autostack.run_demo_attempt(
        objective_label(obj_enum),
        attempt_type,
        robot=app_state.robot,
        vision=app_state.vision,
        world=app_state.world,
        telemetry_logger=app_state.logger,
        frame_callback=frame_callback,
        log_rate_hz=LOG_RATE_HZ,
        stop_flag=lambda: not app_state.running,
    )
    log_line(info)
    complete_obj = True
    if attempt_type == "SUCCESS":
        complete_obj = ok
    with app_state.lock:
        _, _, ended_obj, ended_attempt, should_close = end_attempt(app_state, complete_objective=complete_obj)
    if ended_obj and ended_attempt:
        with app_state.lock:
            app_state.pending_review = (ended_obj, ended_attempt, should_close)
    return ok

def control_loop(app_state):
    app_state.robot = Robot()
    # speed_optimize=False so we get the debug markers drawn on the frame
    app_state.vision = BrickDetector(debug=True, speed_optimize=False)
    print_command_help(app_state)
    
    dt = 1.0 / LOG_RATE_HZ
    was_moving = False
    
    while app_state.running:
        loop_start = time.time()

        request = None
        with app_state.lock:
            if app_state.autostack_request and not app_state.autostack_active:
                request = app_state.autostack_request
                app_state.autostack_request = None
                app_state.autostack_active = True
                app_state.active_command = None
                app_state.active_speed = 0.0

        if request:
            obj_enum, attempt_type = request
            app_state.robot.stop()
            run_auto_attempt(app_state, obj_enum, attempt_type)
            with app_state.lock:
                app_state.autostack_active = False
            was_moving = False
            continue
        
        # 1a. Config Hot-Reload Check (Every 1s)
        if time.time() - app_state.last_config_check > 1.0:
            app_state.last_config_check = time.time()
            try:
                # Use absolute path relative to this script
                p = Path(__file__).parent / "world_model_brick.json"
                if p.exists():
                     mtime = p.stat().st_mtime
                     if mtime > app_state.config_mtime:
                         if app_state.config_mtime > 0: # Skip first check
                             log_line("[CONFIG] Change detected. Reloading...")
                             if app_state.vision.reload_from_file():
                                 log_line("[CONFIG] Reloaded successfully.")
                         app_state.config_mtime = mtime
            except Exception:
                pass

        # 1. Heartbeat Check
        with app_state.lock:
            if time.time() - app_state.last_key_time > HEARTBEAT_TIMEOUT:
                app_state.active_command = None
                app_state.active_speed = 0.0
            
            # Fixed speed based on Gear 1
            gear_speed = GEAR_1_SPEED
            
            cmd = app_state.active_command
            if cmd:
                speed = gear_speed
                if cmd in ('l', 'r'):
                    speed = min(1.0, speed * app_state.turn_speed_multiplier)
                if cmd in ('u', 'd'):
                    speed = min(1.0, speed * 4.0)
                app_state.active_speed = speed 
            else:
                speed = 0.0
                app_state.active_speed = 0.0
            
        if cmd and speed > 0:
            app_state.robot.send_command(cmd, speed)
            was_moving = True
        elif was_moving:
            app_state.robot.stop()
            was_moving = False
            
        # 2. Vision
        found, angle, dist, offset_x, conf, cam_h, brick_above, brick_below = app_state.vision.read()
        
        # 3. Telemetry Update
        app_state.world.update_vision(found, dist, angle, conf, offset_x, cam_h, brick_above, brick_below)
        
        # 4. Update Web Stream Frame
        if app_state.vision.current_frame is not None:
            # Create a copy and draw our rich HUD
            frame = app_state.vision.current_frame.copy()
            
            with app_state.lock:
                draw_telemetry_overlay(frame, app_state.world, show_prompt=False)
                app_state.current_frame = frame
        
        # Track Motion
        if cmd and speed > 0:
            atype = "unknown"
            if cmd == 'f': atype = "forward"
            elif cmd == 'b': atype = "backward"
            elif cmd == 'l': atype = "left_turn"
            elif cmd == 'r': atype = "right_turn"
            elif cmd == 'u': atype = "mast_up"
            elif cmd == 'd': atype = "mast_down"
            
            pwr = int(speed * 255)
            evt = MotionEvent(atype, pwr, int(dt*1000))
            app_state.world.update_from_motion(evt)
            if app_state.active_attempt:
                app_state.logger.log_event(evt, app_state.world.objective_state.value)
            
        # 5. Save Log (Image saving removed)
        with app_state.lock:
            if app_state.active_attempt:
                app_state.logger.log_state(app_state.world)
        
        # 6. Rate Limiting
        elapsed = time.time() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", dest="stream", action="store_true",
                        help="Enable livestreaming")
    parser.add_argument("--no-stream", dest="stream", action="store_false",
                        help="Disable livestreaming")
    parser.set_defaults(stream=None)
    args = parser.parse_args()

    if args.stream is None:
        choice = input("Enable livestream? [Y/n]: ").strip().lower()
        args.stream = choice not in ("n", "no")

    state = AppState()
    
    # Keyboard thread
    kb_t = threading.Thread(target=keyboard_thread, args=(state,), daemon=True)
    kb_t.start()
    
    # Web Stream thread (optional)
    if args.stream:
        stream_server = StreamServer(
            build_stream_provider(state),
            host=STREAM_HOST,
            port=STREAM_PORT,
            fps=STREAM_FPS,
            jpeg_quality=STREAM_JPEG_QUALITY,
            title="Robot Leia - Keyboard Training",
            header="Robot Leia - Keyboard Training",
            footer="Use the terminal for controls. Keep this window open to see the live feed.",
            img_width=800,
            sharpen=True,
        )
        stream_server.start()
        log_line(f"[VISION] Stream started at {format_stream_url(STREAM_HOST, STREAM_PORT)}")
    else:
        log_line("[VISION] Stream disabled")

    log_line("[CTRL] W/S drive, A/D turn slow, Z/C turn fast, P/L mast, F action, ':' command, Q quit")
    
    try:
        control_loop(state)
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        if state.robot: state.robot.close()
        if state.vision: state.vision.close()
        close_log(state, marker="JOB_ABORT")
        log_line("Shutdown complete.")
