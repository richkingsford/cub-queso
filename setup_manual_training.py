import argparse
import json
import sys
import threading
import time
import tty
import termios
import statistics
from pathlib import Path

from helper_robot_control import Robot
from helper_demo_log_utils import load_demo_logs, normalize_objective_label, prune_log_file
from helper_gate_utils import load_process_objectives
from helper_stream_server import StreamServer, format_stream_url
from helper_vision_aruco import ArucoBrickVision
from helper_manual_config import load_manual_training_config
from telemetry_robot import (
    WorldModel,
    TelemetryLogger,
    MotionEvent,
    ObjectiveState,
    draw_telemetry_overlay,
    objective_sequence,
)
from helper_autobuild import (
    collect_segments,
    CONTROL_DT,
    format_gate_lines,
    load_process_model,
    replay_segment,
    refresh_autobuild_config,
    select_demo_segment,
    update_world_from_vision,
    update_process_model_from_demos,
)
import telemetry_brick

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
_MANUAL_CONFIG = load_manual_training_config()
LOG_RATE_HZ = float(_MANUAL_CONFIG.get("log_rate_hz", 10))
GEAR_1_SPEED = float(_MANUAL_CONFIG.get("gear_1_speed", 0.32))
GEAR_9_SPEED = float(_MANUAL_CONFIG.get("gear_9_speed", 1.0))
HEARTBEAT_TIMEOUT = float(_MANUAL_CONFIG.get("heartbeat_timeout", 0.3))
STREAM_HOST = _MANUAL_CONFIG.get("stream_host", "127.0.0.1")
STREAM_PORT = int(_MANUAL_CONFIG.get("stream_port", 5000))
STREAM_FPS = int(_MANUAL_CONFIG.get("stream_fps", 10))
STREAM_JPEG_QUALITY = int(_MANUAL_CONFIG.get("stream_jpeg_quality", 85))
DEMOS_DIR = Path(__file__).resolve().parent / "demos"
PROCESS_MODEL_FILE = Path(__file__).resolve().parent / "world_model_process.json"
DEMO_OBJECTIVES = objective_sequence()

MM_METRICS = {
    "xAxis_offset_abs",
    "dist",
    "distance",
    "lift_height",
}

BRICK_STUDY_FRAMES = 4
BRICK_STUDY_SPLIT_DIFF_MM = 20.0
BRICK_STUDY_SPLIT_DIFF_DEG = 8.0
BRICK_STUDY_OUTLIER_MM = 12.0
BRICK_STUDY_OUTLIER_DEG = 6.0
BRICK_STUDY_STD_LOW_MM = 4.0
BRICK_STUDY_STD_LOW_DEG = 2.0
BRICK_STUDY_STD_HIGH_MM = 8.0
BRICK_STUDY_STD_HIGH_DEG = 4.0

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
    "n": "NOMINAL",
}

ATTEMPT_NAMES = {
    "fail": "FAIL",
    "failure": "FAIL",
    "success": "SUCCESS",
    "recover": "RECOVER",
    "recovery": "RECOVER",
    "nominal": "NOMINAL",
    "nom": "NOMINAL",
}

ATTEMPT_MARKERS = {
    "FAIL": ("FAIL_START", "FAIL_END"),
    "RECOVER": ("RECOVER_START", "RECOVER_END"),
    "SUCCESS": ("SUCCESS_START", "SUCCESS_END"),
    "NOMINAL": ("NOMINAL_START", "NOMINAL_END"),
}

ATTEMPT_STATUS = {
    "FAIL": "FAIL",
    "RECOVER": "RECOVERY",
    "SUCCESS": "NORMAL",
    "NOMINAL": "NOMINAL",
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

def close_log(app_state, marker=None):
    if app_state.logger is None or app_state.logger_closed:
        return
    if marker:
        app_state.logger.log_keyframe(marker)
    app_state.logger.enabled = False
    app_state.logger.close()
    app_state.logger_closed = True
    if app_state.log_path:
        prune_log_file(app_state.log_path, delete_if_empty=True)

def trash_current_session(app_state):
    log_path = app_state.log_path
    app_state.active_attempt = None
    app_state.objective_open = False
    app_state.open_objective = None
    app_state.world.recording_active = False
    app_state.world.attempt_status = "NORMAL"
    if app_state.logger is not None and not app_state.logger_closed:
        app_state.logger.enabled = False
        app_state.logger.close()
    app_state.logger_closed = True
    app_state.log_path = None
    if log_path and log_path.exists():
        try:
            log_path.unlink()
        except OSError:
            return log_path, False
    return log_path, True

def run_auto_objective(app_state, obj_enum):
    logs = load_demo_logs(app_state.demos_dir)
    update_process_model_from_demos(logs, PROCESS_MODEL_FILE)
    refresh_autobuild_config(PROCESS_MODEL_FILE)
    if not logs:
        log_line("[AUTO] No demo logs found for auto mode.")
        return False

    segments_by_obj, _ = collect_segments(logs)
    model = load_process_model(PROCESS_MODEL_FILE)
    process_rules = model.get("objectives") if isinstance(model, dict) else {}
    if not isinstance(process_rules, dict):
        process_rules = {}

    app_state.world.process_rules = process_rules
    app_state.world.rules = process_rules

    objective_key = normalize_objective_label(obj_enum.value)
    app_state.world.objective_state = obj_enum

    cfg = process_rules.get(objective_key, {}) if process_rules else {}
    nominal_only = bool(cfg.get("nominalDemosOnly"))
    segment, seg_type = select_demo_segment(segments_by_obj, objective_key, nominal_only)
    if not segment:
        log_line(f"[AUTO] No demo segment found for {objective_key}.")
        return False

    quiet_align = objective_key == "ALIGN_BRICK"
    if not quiet_align:
        start_desc, success_desc = format_gate_lines(cfg)
        log_line(f"[AUTO] {objective_key} demo={seg_type} start gates: {start_desc}")
        log_line(f"[AUTO] {objective_key} demo={seg_type} success gates: {success_desc}")
    if app_state.robot:
        app_state.robot.stop()
    observer = make_auto_observer(app_state) if quiet_align else None
    confirm_callback = None
    ok, reason = replay_segment(
        segment,
        objective_key,
        app_state.robot,
        app_state.vision,
        app_state.world,
        observer=observer,
        analysis_pause_s=0.0,
        confirm_callback=confirm_callback,
        align_silent=quiet_align,
    )
    if ok:
        log_line(f"[AUTO] {objective_key} success ({reason}).")
    else:
        if not quiet_align:
            log_line(f"[AUTO] {objective_key} failed ({reason}).")
    return ok


def update_brick_analytics(app_state):
    objectives = app_state.world.process_rules or load_process_objectives()
    analytics = telemetry_brick.compute_brick_analytics(
        app_state.world,
        objectives,
        app_state.world.learned_rules,
        "ALIGN_BRICK",
        duration_s=CONTROL_DT,
    )
    app_state.gate_status = analytics.get("gate_status") or []
    app_state.gate_progress = analytics.get("gate_progress") or []
    app_state.brick_highlight_metric = analytics.get("highlight_metric")
    suggestion = analytics.get("suggestion")
    if suggestion:
        app_state.objective_suggestions = [("ALIGN_BRICK", suggestion)]
    else:
        app_state.objective_suggestions = []

def refresh_brick_telemetry(app_state):
    update_brick_analytics(app_state)
    update_stream_frame(app_state)






def _ordinal_list(indices):
    names = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
    parts = [names.get(idx, f"{idx}th") for idx in indices]
    if not parts:
        return "none", 0
    if len(parts) == 1:
        return parts[0], 1
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}", 2
    return ", ".join(parts[:-1]) + f", and {parts[-1]}", len(parts)


def _variance_label(std_dist, std_offset, std_angle):
    if std_dist <= BRICK_STUDY_STD_LOW_MM and std_offset <= BRICK_STUDY_STD_LOW_MM and std_angle <= BRICK_STUDY_STD_LOW_DEG:
        return "low variance"
    if std_dist >= BRICK_STUDY_STD_HIGH_MM or std_offset >= BRICK_STUDY_STD_HIGH_MM or std_angle >= BRICK_STUDY_STD_HIGH_DEG:
        return "high variance"
    return "med variance"


def _average_frames(frames):
    def mean(values):
        return sum(values) / len(values) if values else 0.0

    def majority(values):
        return sum(1 for v in values if v) >= (len(values) / 2.0)

    return {
        "found": majority([f["found"] for f in frames]),
        "dist": mean([f["dist"] for f in frames]),
        "angle": mean([f["angle"] for f in frames]),
        "offset_x": mean([f["offset_x"] for f in frames]),
        "conf": mean([f["conf"] for f in frames]),
        "cam_h": mean([f["cam_h"] for f in frames]),
        "brick_above": majority([f["brick_above"] for f in frames]),
        "brick_below": majority([f["brick_below"] for f in frames]),
    }


def evaluate_brick_frames(frames):
    if len(frames) < BRICK_STUDY_FRAMES:
        return None

    first_two = frames[:2]
    last_two = frames[2:]
    avg_first = _average_frames(first_two)
    avg_last = _average_frames(last_two)
    if (
        abs(avg_first["dist"] - avg_last["dist"]) > BRICK_STUDY_SPLIT_DIFF_MM
        or abs(avg_first["offset_x"] - avg_last["offset_x"]) > BRICK_STUDY_SPLIT_DIFF_MM
        or abs(avg_first["angle"] - avg_last["angle"]) > BRICK_STUDY_SPLIT_DIFF_DEG
    ):
        return {
            "confidence": 0,
            "message": "0% confidence (frames 1 and 2 were highly different than frames 3 and 4)",
            "average": None,
            "reset": True,
        }

    med_dist = statistics.median([f["dist"] for f in frames])
    med_offset = statistics.median([f["offset_x"] for f in frames])
    med_angle = statistics.median([f["angle"] for f in frames])
    keep = []
    discarded = []
    for idx, frame in enumerate(frames, start=1):
        if not frame["found"]:
            discarded.append(idx)
            continue
        if (
            abs(frame["dist"] - med_dist) > BRICK_STUDY_OUTLIER_MM
            or abs(frame["offset_x"] - med_offset) > BRICK_STUDY_OUTLIER_MM
            or abs(frame["angle"] - med_angle) > BRICK_STUDY_OUTLIER_DEG
        ):
            discarded.append(idx)
        else:
            keep.append(frame)

    if len(keep) < 3:
        return {
            "confidence": 0,
            "message": "0% confidence (variance too high on all frames)",
            "average": None,
            "reset": True,
        }

    std_dist = statistics.pstdev([f["dist"] for f in keep]) if len(keep) > 1 else BRICK_STUDY_STD_HIGH_MM
    std_offset = statistics.pstdev([f["offset_x"] for f in keep]) if len(keep) > 1 else BRICK_STUDY_STD_HIGH_MM
    std_angle = statistics.pstdev([f["angle"] for f in keep]) if len(keep) > 1 else BRICK_STUDY_STD_HIGH_DEG
    variance_label = _variance_label(std_dist, std_offset, std_angle)
    if variance_label == "high variance":
        return {
            "confidence": 0,
            "message": "0% confidence (variance too high on all frames)",
            "average": None,
            "reset": True,
        }

    confidence = int(round(100 * len(keep) / BRICK_STUDY_FRAMES))
    discarded_label, discarded_count = _ordinal_list(discarded)
    if discarded_count == 0:
        discarded_phrase = "discarded none"
    elif discarded_count == 1:
        discarded_phrase = f"discarded the {discarded_label} frame"
    else:
        discarded_phrase = f"discarded the {discarded_label} frames"
    message = (
        f"{confidence}% confidence ({len(keep)}/{BRICK_STUDY_FRAMES} frames had {variance_label}; "
        f"{discarded_phrase})"
    )
    return {
        "confidence": confidence,
        "message": message,
        "average": _average_frames(keep),
        "reset": True,
    }


def update_stream_frame(app_state):
    if app_state.vision.current_frame is None:
        return
    frame = app_state.vision.current_frame.copy()
    with app_state.lock:
        objective_suggestions = []
        if app_state.objective_suggestions:
            objective_suggestions.extend(app_state.objective_suggestions)
        draw_telemetry_overlay(
            frame,
            app_state.world,
            show_prompt=False,
            gate_status=app_state.gate_status,
            gate_progress=app_state.gate_progress,
            objective_suggestions=objective_suggestions,
            highlight_metric=app_state.brick_highlight_metric,
            loop_id=getattr(app_state.world, "loop_id", None),
        )
        app_state.current_frame = frame


def make_auto_observer(app_state):
    def _observer(stage, world, vision, cmd, speed, reason):
        refresh_brick_telemetry(app_state)
    return _observer


def make_auto_confirm(app_state):
    def _confirm(world, vision):
        with app_state.lock:
            app_state.auto_confirm_event.clear()
            app_state.auto_confirm_needed = True
            last_enter_time = app_state.last_enter_time
            start_enter_time = app_state.last_enter_time
        if time.time() - last_enter_time <= 0.75:
            with app_state.lock:
                app_state.last_enter_time = 0.0
                app_state.auto_confirm_needed = False
            return True
        log_line("[AUTO] Press Enter to execute suggested action.")
        while app_state.running:
            if app_state.auto_confirm_event.is_set():
                return True
            with app_state.lock:
                recent_enter = app_state.last_enter_time
            if recent_enter > start_enter_time and time.time() - recent_enter <= 1.0:
                with app_state.lock:
                    app_state.last_enter_time = 0.0
                    app_state.auto_confirm_needed = False
                return True
            update_world_from_vision(world, vision, log=False)
            refresh_brick_telemetry(app_state)
            time.sleep(0.05)
        return False
    return _confirm

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
    log_line("[CMD] Attempt codes: f=fail, s=success, r=recover, n=nominal")
    log_line("[CMD] Example: :4s (scoop success), :4n (scoop nominal)")
    log_line("[CMD] Auto mode: press 'p' then an objective code.")
    log_line("[CMD] Trash current session log: press '1' during recording.")
    log_line("[CMD] Auto-run: use 'p' + objective code.")
    log_line("[CMD] End attempt: press ':' to finish and return to the command prompt.")

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

    with app_state.lock:
        if cmd_lower in ("", ":"):
            if app_state.active_attempt:
                ok, msg, obj_enum, attempt_type, should_close = end_attempt(app_state)
                messages.append(msg)
                if ok:
                    ended_info = (obj_enum, attempt_type)
            exit_mode = True
            return exit_mode, do_help, messages, ended_info
        if cmd_lower in ("q", "quit", "exit"):
            app_state.running = False
            messages.append("Stopping manual recording...")
            exit_mode = True
            return exit_mode, do_help, messages, ended_info
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
        else:
            auto_mode, obj_enum, attempt_type, err = parse_text_command(cmd)
            if err:
                messages.append(f"[CMD] {err}")
            elif auto_mode:
                messages.append("[AUTO] Auto-run is disabled. Use manual commands.")
                exit_mode = True
            else:
                ok, msg, ended, should_close_val = handle_attempt_command(app_state, obj_enum, attempt_type)
                messages.append(msg)
                should_close = should_close_val # Propagate
                if ok:
                    # Return to driving mode immediately on start/stop
                    exit_mode = True
                    if app_state.active_attempt:
                        messages.append("[CMD] Press ':' to finish and return to the command prompt.")
                if ended:
                    ended_info = ended

    return exit_mode, do_help, messages, ended_info


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
        app_state.world.recording_active = False
    else:
        # For FAIL/RECOVER, we keep it open by default to allow retry,
        # UNLESS we are explicitly closing the objective.
        if complete_objective:
            app_state.objective_open = False
            app_state.open_objective = None
        app_state.world.recording_active = False

    app_state.world.attempt_status = "NORMAL"
    app_state.active_attempt = None
    app_state.logger.enabled = False
    return True, f"[OBJ] {obj_label} {attempt_type} finished.", obj_enum, attempt_type, False

def handle_attempt_command(app_state, obj_enum, attempt_type):
    obj_label = objective_label(obj_enum)
    ended_info = None
    ended_close = False

    # 1. If ANY recording is active, end it first.
    if app_state.active_attempt:
        ok, msg, ended_obj, ended_attempt, should_close = end_attempt(app_state, complete_objective=(app_state.active_attempt == attempt_type))
        if ok:
            ended_info = (ended_obj, ended_attempt)
            ended_close = should_close
            log_line(msg)

    # 2. Check Objective Constraints for the NEW attempt
    if app_state.objective_open and app_state.open_objective != obj_enum:
        return False, f"[OBJ] Finish {objective_label(app_state.open_objective)} before switching objectives.", ended_info, ended_close

    # 3. If they were just toggling OFF the same thing, we are done.
    if ended_info and ended_info[0] == obj_enum and ended_info[1] == attempt_type:
        return True, f"[OBJ] {obj_label} finished.", ended_info, ended_close

    # 4. Start the new attempt
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

        self.auto_prompt = False
        self.auto_request = None
        self.auto_running = False
        self.auto_confirm_needed = False
        self.auto_confirm_event = threading.Event()
        self.last_enter_time = 0.0

        self.gate_status = []
        self.gate_progress = []
        self.objective_suggestions = []
        self.brick_highlight_metric = None
        self.brick_frame_buffer = []

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
        ch = getch()
        if not ch:
            continue
        ch_lower = ch.lower()
        with app_state.lock:
            auto_prompt = app_state.auto_prompt
            auto_confirm_needed = app_state.auto_confirm_needed

        messages = []
        if ch == 'Q':
            with app_state.lock:
                app_state.last_key_time = time.time()
                app_state.running = False
            messages.append("Stopping manual recording...")
        elif ch in ('\n', '\r'):
            with app_state.lock:
                app_state.last_key_time = time.time()
                app_state.last_enter_time = time.time()
                if auto_confirm_needed:
                    app_state.auto_confirm_needed = False
                    app_state.auto_confirm_event.set()
            if auto_confirm_needed:
                messages.append("[AUTO] Action confirmed.")
        elif ch_lower == '1':
            with app_state.lock:
                app_state.last_key_time = time.time()
                trashed_path, ok = trash_current_session(app_state)
            if trashed_path:
                if ok:
                    messages.append(f"[SESSION] Trashed log: {trashed_path}")
                else:
                    messages.append(f"[SESSION] Failed to delete log: {trashed_path}")
            else:
                messages.append("[SESSION] No active log to trash.")
        elif auto_prompt:
            with app_state.lock:
                app_state.last_key_time = time.time()
                if ch_lower == 'p':
                    app_state.auto_prompt = False
                    messages.append("[AUTO] Auto mode cancelled.")
                else:
                    obj_enum = resolve_objective_token(ch_lower)
                    if obj_enum:
                        app_state.auto_prompt = False
                        app_state.auto_request = obj_enum
                        app_state.active_command = None
                        app_state.active_speed = 0.0
                        messages.append(f"[AUTO] Queued {objective_label(obj_enum)}.")
                    else:
                        messages.append("[AUTO] Unknown objective code.")
        elif ch_lower == ':':
            with app_state.lock:
                app_state.last_key_time = time.time()
                end_msg = None
                if app_state.active_attempt:
                    ok, msg, _, _, _ = end_attempt(app_state, complete_objective=True)
                    end_msg = msg

                app_state.active_command = None
                app_state.active_speed = 0.0
            if end_msg:
                log_line(end_msg)
            log_line("[CMD] Enter command (ex: 4s, 4f, help). Use ':' or blank to exit.")
            
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
                exit_mode, do_help, messages, ended_info = handle_command_line(app_state, cmd)
                if do_help:
                    print_command_help(app_state)
                for msg in messages:
                    log_line(msg)
                if ended_info:
                    pass
                if exit_mode:
                    for msg in command_mode_exit_messages(app_state):
                        log_line(msg)
                    # Flush and CLEAR messages to prevent double-printing at bottom of thread
                    messages = []
                    try:
                        termios.tcflush(sys.stdin, termios.TCIFLUSH)
                    except:
                        pass
                    break
        elif ch_lower == 'p':
            with app_state.lock:
                app_state.last_key_time = time.time()
                app_state.auto_prompt = True
                app_state.active_command = None
                app_state.active_speed = 0.0
            messages.append("[AUTO] Select an objective code to run autonomously (press 'p' again to cancel).")
        else:
            with app_state.lock:
                app_state.last_key_time = time.time()
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
                    app_state.turn_speed_multiplier = 1.0
                elif ch_lower == 'q':
                    app_state.active_command = 'l'
                    app_state.turn_speed_multiplier = 1.0 / 3.0
                elif ch_lower == 'e':
                    app_state.active_command = 'r'
                    app_state.turn_speed_multiplier = 1.0 / 3.0
                elif ch_lower == 'z':
                    app_state.active_command = 'l'
                    app_state.turn_speed_multiplier = 2.0
                elif ch_lower == 'c':
                    app_state.active_command = 'r'
                    app_state.turn_speed_multiplier = 2.0
                elif ch_lower == 'u':
                    app_state.active_command = 'u'
                elif ch_lower == 'l':
                    app_state.active_command = 'd'
                elif ch_lower in ('h', '?'):
                    print_command_help(app_state)

        for msg in messages:
            log_line(msg)

        if not app_state.running:
            break

def control_loop(app_state):
    app_state.robot = Robot()
    # speed_optimize=False so we get the debug markers drawn on the frame
    app_state.vision = ArucoBrickVision(debug=True)
    print_command_help(app_state)
    
    dt = 1.0 / LOG_RATE_HZ
    was_moving = False
    
    while app_state.running:
        loop_start = time.time()
        auto_obj = None
        with app_state.lock:
            if app_state.auto_request and not app_state.auto_running:
                auto_obj = app_state.auto_request
                app_state.auto_request = None
                app_state.auto_running = True
                app_state.active_command = None
                app_state.active_speed = 0.0

        if auto_obj:
            log_line(f"[AUTO] Starting {objective_label(auto_obj)}...")
            run_auto_objective(app_state, auto_obj)
            with app_state.lock:
                app_state.auto_running = False
                app_state.active_command = None
                app_state.active_speed = 0.0
            continue

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
        study = None
        with app_state.lock:
            app_state.brick_frame_buffer.append({
                "found": bool(found),
                "dist": float(dist),
                "angle": float(angle),
                "offset_x": float(offset_x),
                "conf": float(conf),
                "cam_h": float(cam_h),
                "brick_above": bool(brick_above),
                "brick_below": bool(brick_below),
            })
            if len(app_state.brick_frame_buffer) > BRICK_STUDY_FRAMES:
                app_state.brick_frame_buffer.pop(0)

            study = evaluate_brick_frames(app_state.brick_frame_buffer)
            if study and study.get("reset"):
                app_state.brick_frame_buffer = []
        if study and study.get("average"):
            avg = study["average"]
            app_state.world.update_vision(
                avg["found"],
                avg["dist"],
                avg["angle"],
                avg["conf"],
                avg["offset_x"],
                avg["cam_h"],
                avg["brick_above"],
                avg["brick_below"],
            )
        refresh_brick_telemetry(app_state)
        
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
    parser.set_defaults(stream=True)
    args = parser.parse_args()

    logs = load_demo_logs(DEMOS_DIR)
    if logs:
        update_process_model_from_demos(logs, PROCESS_MODEL_FILE)

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

    log_line("[CTRL] W/S drive, A/D turn, q/e slow turns, z/c fast turns, U/L mast, F action, ':' command, p auto, 1 trash log, Q quit")
    
    try:
        control_loop(state)
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        if state.robot: state.robot.close()
        if state.vision: state.vision.close()
        close_log(state, marker=None)
        log_line("Shutdown complete.")
