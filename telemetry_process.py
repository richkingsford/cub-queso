#!/usr/bin/env python3
import argparse
import json
import collections
import math
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path

from helper_demo_log_utils import extract_attempt_segments, load_demo_logs, normalize_step_label
from helper_manual_config import load_manual_training_config
from helper_vision_aruco import ArucoBrickVision
from helper_robot_control import Robot
from telemetry_robot import MotionEvent, StepState, WorldModel, draw_telemetry_overlay
import telemetry_brick
import telemetry_robot as telemetry_robot_module
import telemetry_wall
import helper_learning

USE_LEARNED_POLICY = False
GLOBAL_POLICY = None


DEMO_DIR = Path(__file__).resolve().parent / "demos"
PROCESS_MODEL_FILE = Path(__file__).resolve().parent / "world_model_process.json"

_MANUAL_CONFIG = load_manual_training_config()
STREAM_HOST = _MANUAL_CONFIG.get("stream_host", "127.0.0.1")
STREAM_PORT = int(_MANUAL_CONFIG.get("stream_port", 5000))
STREAM_FPS = int(_MANUAL_CONFIG.get("stream_fps", 10))
STREAM_JPEG_QUALITY = int(_MANUAL_CONFIG.get("stream_jpeg_quality", 85))

DEFAULT_AUTOBUILD_CONFIG = {
    "control_hz": 20.0,
    "gate_stability_frames": 5,
    "success_consecutive_frames": 3,
    "success_majority_window": 5,
    "success_majority_required": 3,
    "success_visible_false_grace_s": 0.0,
    "success_confidence_min": 0.85,
    "confidence_log_min": 0.5,
    "success_confirmation_s": 0.35,
    "success_confirmation_frames": 4,
    "success_confirmation_start_confidence": 0.5,
    "failure_tighten_low_pct": 0.1,
    "failure_tighten_high_pct": 0.9,
    "start_gate_timeout_s": 8.0,
    "success_settle_s": 1.0,
    "success_tail_window_s": 0.5,
    "success_tail_count": 4,
    "max_phase_attempts": 1,
    "smooth_step_s": 1.0,
        "fail_pause_s": 3.0,
    "find_brick_slow_factor": 4.0,
    "visibility_lost_hold_s": 0.5,
    "learned_policy_confidence_threshold": 0.4,
    "success_frames_by_step": {"FIND_BRICK": 3},
    "alignment_metrics": ["angle_abs", "xAxis_offset_abs", "dist"],
}

DEFAULT_SPEED_SCORE = telemetry_robot_module.SPEED_SCORE_DEFAULT
MAX_SPEED_SCORE = telemetry_robot_module.SPEED_SCORE_MAX

MM_METRICS = {
    "xAxis_offset_abs",
    "dist",
    "distance",
    "lift_height",
}
MIN_MM_TOL = 1.5



def _as_float(value, fallback):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _as_int(value, fallback):
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def default_speed_for_cmd(cmd, score=DEFAULT_SPEED_SCORE):
    return telemetry_robot_module.manual_speed_for_cmd(cmd, score)


def max_speed_for_cmd(cmd, score=MAX_SPEED_SCORE):
    return telemetry_robot_module.manual_speed_for_cmd(cmd, score)


def apply_autobuild_config(cfg):
    global CONTROL_HZ
    global CONTROL_DT
    global GATE_STABILITY_FRAMES
    global SUCCESS_CONSECUTIVE_FRAMES
    global SUCCESS_MAJORITY_WINDOW
    global SUCCESS_MAJORITY_REQUIRED
    global SUCCESS_VISIBLE_FALSE_GRACE_S
    global SUCCESS_CONFIDENCE_MIN
    global CONFIDENCE_LOG_MIN
    global SUCCESS_CONFIRMATION_S
    global SUCCESS_CONFIRMATION_FRAMES
    global SUCCESS_CONFIRMATION_START_CONFIDENCE
    global FAILURE_TIGHTEN_LOW_PCT
    global FAILURE_TIGHTEN_HIGH_PCT
    global START_GATE_TIMEOUT_S
    global SUCCESS_SETTLE_S
    global SUCCESS_TAIL_WINDOW_S
    global SUCCESS_TAIL_COUNT
    global MAX_PHASE_ATTEMPTS
    global SMOOTH_STEP_S
    global FAIL_PAUSE_S
    global FIND_BRICK_SLOW_FACTOR
    global VISIBILITY_LOST_HOLD_S
    global LEARNED_POLICY_CONFIDENCE_THRESHOLD
    global SUCCESS_FRAMES_BY_STEP
    global ALIGNMENT_METRICS

    CONTROL_HZ = _as_float(cfg.get("control_hz"), DEFAULT_AUTOBUILD_CONFIG["control_hz"])
    CONTROL_DT = 1.0 / max(CONTROL_HZ, 1e-6)
    GATE_STABILITY_FRAMES = _as_int(cfg.get("gate_stability_frames"), DEFAULT_AUTOBUILD_CONFIG["gate_stability_frames"])
    SUCCESS_CONSECUTIVE_FRAMES = _as_int(
        cfg.get("success_consecutive_frames"),
        DEFAULT_AUTOBUILD_CONFIG["success_consecutive_frames"],
    )
    SUCCESS_MAJORITY_WINDOW = _as_int(
        cfg.get("success_majority_window"),
        DEFAULT_AUTOBUILD_CONFIG["success_majority_window"],
    )
    SUCCESS_MAJORITY_REQUIRED = _as_int(
        cfg.get("success_majority_required"),
        DEFAULT_AUTOBUILD_CONFIG["success_majority_required"],
    )
    SUCCESS_VISIBLE_FALSE_GRACE_S = _as_float(
        cfg.get("success_visible_false_grace_s"),
        DEFAULT_AUTOBUILD_CONFIG["success_visible_false_grace_s"],
    )
    SUCCESS_CONFIDENCE_MIN = _as_float(
        cfg.get("success_confidence_min"),
        DEFAULT_AUTOBUILD_CONFIG["success_confidence_min"],
    )
    CONFIDENCE_LOG_MIN = _as_float(
        cfg.get("confidence_log_min"),
        DEFAULT_AUTOBUILD_CONFIG["confidence_log_min"],
    )
    SUCCESS_CONFIRMATION_S = _as_float(
        cfg.get("success_confirmation_s"),
        DEFAULT_AUTOBUILD_CONFIG["success_confirmation_s"],
    )
    SUCCESS_CONFIRMATION_FRAMES = _as_int(
        cfg.get("success_confirmation_frames"),
        DEFAULT_AUTOBUILD_CONFIG["success_confirmation_frames"],
    )
    SUCCESS_CONFIRMATION_START_CONFIDENCE = _as_float(
        cfg.get("success_confirmation_start_confidence"),
        DEFAULT_AUTOBUILD_CONFIG["success_confirmation_start_confidence"],
    )
    FAILURE_TIGHTEN_LOW_PCT = _as_float(
        cfg.get("failure_tighten_low_pct"),
        DEFAULT_AUTOBUILD_CONFIG["failure_tighten_low_pct"],
    )
    FAILURE_TIGHTEN_HIGH_PCT = _as_float(
        cfg.get("failure_tighten_high_pct"),
        DEFAULT_AUTOBUILD_CONFIG["failure_tighten_high_pct"],
    )
    START_GATE_TIMEOUT_S = _as_float(
        cfg.get("start_gate_timeout_s"),
        DEFAULT_AUTOBUILD_CONFIG["start_gate_timeout_s"],
    )
    SUCCESS_SETTLE_S = _as_float(
        cfg.get("success_settle_s"),
        DEFAULT_AUTOBUILD_CONFIG["success_settle_s"],
    )
    SUCCESS_TAIL_WINDOW_S = _as_float(
        cfg.get("success_tail_window_s"),
        DEFAULT_AUTOBUILD_CONFIG["success_tail_window_s"],
    )
    SUCCESS_TAIL_COUNT = _as_int(
        cfg.get("success_tail_count"),
        DEFAULT_AUTOBUILD_CONFIG["success_tail_count"],
    )
    MAX_PHASE_ATTEMPTS = _as_int(
        cfg.get("max_phase_attempts"),
        DEFAULT_AUTOBUILD_CONFIG["max_phase_attempts"],
    )
    SMOOTH_STEP_S = _as_float(
        cfg.get("smooth_step_s"),
        DEFAULT_AUTOBUILD_CONFIG["smooth_step_s"],
    )
    FAIL_PAUSE_S = _as_float(
        cfg.get("fail_pause_s"),
        DEFAULT_AUTOBUILD_CONFIG["fail_pause_s"],
    )
    FIND_BRICK_SLOW_FACTOR = _as_float(
        cfg.get("find_brick_slow_factor"),
        DEFAULT_AUTOBUILD_CONFIG["find_brick_slow_factor"],
    )
    VISIBILITY_LOST_HOLD_S = _as_float(
        cfg.get("visibility_lost_hold_s"),
        DEFAULT_AUTOBUILD_CONFIG["visibility_lost_hold_s"],
    )
    LEARNED_POLICY_CONFIDENCE_THRESHOLD = _as_float(
        cfg.get("learned_policy_confidence_threshold"),
        DEFAULT_AUTOBUILD_CONFIG["learned_policy_confidence_threshold"],
    )
    frames_cfg = cfg.get("success_frames_by_step")
    if not isinstance(frames_cfg, dict):
        frames_cfg = DEFAULT_AUTOBUILD_CONFIG["success_frames_by_step"]
    SUCCESS_FRAMES_BY_STEP = dict(frames_cfg)
    metrics_cfg = cfg.get("alignment_metrics")
    if not isinstance(metrics_cfg, (list, tuple, set)):
        metrics_cfg = DEFAULT_AUTOBUILD_CONFIG["alignment_metrics"]
    ALIGNMENT_METRICS = set(metrics_cfg)


apply_autobuild_config(DEFAULT_AUTOBUILD_CONFIG)


COLOR_RESET = "\033[0m"
COLOR_GREEN = "\033[32m"
COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_RED = "\033[31m"
COLOR_WHITE = "\033[37m"
COLOR_GRAY = "\033[90m"


ACTION_CMD_MAP = {
    "forward": "f",
    "backward": "b",
    "left_turn": "l",
    "right_turn": "r",
    "mast_up": "u",
    "mast_down": "d",
}

ACTION_CMD_DESC = {
    "f": "moving forward",
    "b": "moving backward",
    "l": "turning left",
    "r": "turning right",
    "u": "lifting mast",
    "d": "lowering mast",
}


def cmd_to_motion_type(cmd):
    return {
        "f": "forward",
        "b": "backward",
        "l": "left_turn",
        "r": "right_turn",
        "u": "mast_up",
        "d": "mast_down",
    }.get(cmd, "wait")


def format_headline(headline, color, details=""):
    if details is None:
        details = ""
    return f"{color}{headline}{COLOR_RESET}{details}"


@dataclass
class MotionStep:
    cmd: str
    speed: float
    duration_s: float
    label: str


@dataclass
class SuccessGateTracker:
    consecutive_required: int
    consecutive: int = 0
    window: list = field(default_factory=list)

    def update(self, success_ok):
        if success_ok:
            self.consecutive += 1
        else:
            self.consecutive = 0
        self.window.append(bool(success_ok))
        if len(self.window) > SUCCESS_MAJORITY_WINDOW:
            self.window.pop(0)
        if self.consecutive >= self.consecutive_required:
            return True
        if len(self.window) == SUCCESS_MAJORITY_WINDOW and sum(self.window) >= SUCCESS_MAJORITY_REQUIRED:
            return True
        return False


def select_tail_states(states, window_s=SUCCESS_TAIL_WINDOW_S, max_count=SUCCESS_TAIL_COUNT):
    if not states:
        return []
    if max_count is not None and max_count > 0:
        states = states[-max_count:]
    last_ts = None
    for state in reversed(states):
        ts = state.get("timestamp")
        if ts is not None:
            last_ts = ts
            break
    if last_ts is None:
        return states
    cutoff = last_ts - window_s
    tail = [state for state in states if state.get("timestamp") is not None and state.get("timestamp") >= cutoff]
    return tail or states


def failure_based_scale(success_count, fail_count):
    return (success_count + 1.0) / (fail_count + 1.0)


def derive_success_gate_scales(segments_by_obj, step_rules=None):
    scales = {}
    for obj, segs in segments_by_obj.items():
        success_count = len(segs.get("SUCCESS", []))
        fail_count = len(segs.get("FAIL", []))
        if success_count <= 0 and fail_count <= 0:
            continue
        scales[obj] = failure_based_scale(success_count, fail_count)
    if isinstance(step_rules, dict):
        for obj, rules in step_rules.items():
            if not isinstance(rules, dict):
                continue
            scale = rules.get("success_gate_scale")
            if scale is None:
                continue
            try:
                scale_val = float(scale)
            except (TypeError, ValueError):
                continue
            if scale_val <= 0:
                continue
            obj_key = normalize_step_label(obj)
            if not obj_key:
                continue
            current = scales.get(obj_key, 1.0)
            scales[obj_key] = current * scale_val
    return scales


def success_frames_required(step):
    obj_key = normalize_step_label(step)
    required = SUCCESS_FRAMES_BY_STEP.get(obj_key, GATE_STABILITY_FRAMES)
    return min(required, SUCCESS_CONSECUTIVE_FRAMES)


def round_value(value, decimals=2):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(value, decimals)
    return value


def _fmt_gate_value(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def format_gate_metrics(gates):
    if not gates:
        return "none"
    parts = []
    for metric, stats in gates.items():
        if not isinstance(stats, dict):
            continue
        if metric == "visible":
            if "min" in stats:
                parts.append(f"{metric}={_fmt_gate_value(stats.get('min'))}")
            elif "max" in stats:
                parts.append(f"{metric}={_fmt_gate_value(stats.get('max'))}")
            continue
        target = stats.get("target")
        tol = stats.get("tol")
        if target is not None and tol is not None:
            parts.append(f"{metric}~{_fmt_gate_value(target)}+/-{_fmt_gate_value(tol)}")
            continue
        mu = stats.get("mu")
        sigma = stats.get("sigma")
        if mu is not None and sigma is not None:
            parts.append(f"{metric}~{_fmt_gate_value(mu)}+/-{_fmt_gate_value(sigma)}")
            continue
        min_val = stats.get("min")
        max_val = stats.get("max")
        if min_val is not None and max_val is not None:
            parts.append(f"{_fmt_gate_value(min_val)}<={metric}<={_fmt_gate_value(max_val)}")
        elif min_val is not None:
            parts.append(f"{metric}>={_fmt_gate_value(min_val)}")
        elif max_val is not None:
            parts.append(f"{metric}<={_fmt_gate_value(max_val)}")
    return ", ".join(parts) if parts else "none"


def format_gate_lines(cfg):
    return (
        format_gate_metrics(cfg.get("start_gates") or {}),
        format_gate_metrics(cfg.get("success_gates") or {}),
    )


def format_success_details(world, step):
    obj_key = normalize_step_label(step)
    rules = world.process_rules.get(obj_key, {}) if world.process_rules else {}
    gates = rules.get("success_gates") or {}
    if not gates:
        return "[SUCCESS DETAILS] no success gates"

    brick = world.brick or {}
    parts = []
    for metric, stats in gates.items():
        actual = None
        if metric == "visible":
            actual = bool(brick.get("visible"))
        elif metric == "angle_abs":
            actual = abs(brick.get("angle", 0.0))
        elif metric == "xAxis_offset_abs":
            x_axis = brick.get("x_axis")
            if x_axis is None:
                x_axis = brick.get("offset_x", 0.0)
            actual = x_axis
        elif metric == "dist":
            actual = brick.get("dist")
        elif metric == "confidence":
            actual = brick.get("confidence")
        elif metric == "lift_height":
            actual = world.lift_height

        actual_str = _fmt_gate_value(actual) if actual is not None else "n/a"
        gate_bits = []
        if isinstance(stats, dict):
            if "min" in stats:
                gate_bits.append(f"min={_fmt_gate_value(stats.get('min'))}")
            if "max" in stats:
                gate_bits.append(f"max={_fmt_gate_value(stats.get('max'))}")
            if "target" in stats:
                gate_bits.append(f"target={_fmt_gate_value(stats.get('target'))}")
            if "tol" in stats:
                gate_bits.append(f"tol={_fmt_gate_value(stats.get('tol'))}")
            if "mu" in stats:
                gate_bits.append(f"mu={_fmt_gate_value(stats.get('mu'))}")
            if "sigma" in stats:
                gate_bits.append(f"sigma={_fmt_gate_value(stats.get('sigma'))}")

        if metric == "visible":
            last_seen = getattr(world, "last_visible_time", None)
            if last_seen is not None:
                age_s = time.time() - last_seen
                gate_bits.append(f"last_seen={age_s:.2f}s")

        gate_desc = " ".join(gate_bits) if gate_bits else "gate"
        parts.append(f"{metric}={actual_str} ({gate_desc})")

    return "[SUCCESS DETAILS] " + "; ".join(parts)


def format_brick_state_line(world):
    brick = world.brick or {}
    visible = bool(brick.get("visible"))
    parts = [f"visible={'true' if visible else 'false'}"]
    if visible:
        dist = brick.get("dist")
        angle = brick.get("angle")
        offset = brick.get("x_axis")
        if offset is None:
            offset = brick.get("offset_x")
        conf = brick.get("confidence")
        above = brick.get("brickAbove")
        below = brick.get("brickBelow")
        if dist is not None:
            parts.append(f"dist={dist:.1f}mm")
        if angle is not None:
            parts.append(f"angle={angle:.2f}deg")
        if offset is not None:
            parts.append(f"x_axis={offset:.2f}mm")
        if conf is not None:
            parts.append(f"conf={conf:.0f}")
        if above is not None:
            parts.append(f"above={str(bool(above)).lower()}")
        if below is not None:
            parts.append(f"below={str(bool(below)).lower()}")
    else:
        last_seen = getattr(world, "last_visible_time", None)
        if last_seen is not None:
            parts.append(f"last_seen={time.time() - last_seen:.2f}s")
    return "[BRICK] " + " ".join(parts)


def format_action_line(step, target_visible, reason=None):
    action = ACTION_CMD_DESC.get(step.cmd, "moving")
    power = f"{step.speed:.2f}"
    if target_visible is None:
        suffix = ""
    else:
        if target_visible:
            suffix = "until brick becomes visible"
        else:
            suffix = "until brick is no longer visible"
    reason = str(reason).strip() if reason else ""
    # Make reason gray
    reason_suffix = f" {COLOR_GRAY}({reason}){COLOR_RESET}" if reason else ""
    return f"[ACT] {action} at {power} power {suffix}{reason_suffix}"


def format_control_action_line(cmd, speed, reason=None):
    if not cmd:
        reason_str = str(reason).strip() if reason else ""
        suffix = f" ({reason_str})" if reason_str else ""
        return f"[ACT] holding position{suffix}"

    # Action Description
    action_map = {
        "f": "move forward",
        "b": "move backward",
        "l": "turn left",
        "r": "turn right",
        "u": "lift mast",
        "d": "lower mast",
    }
    action = action_map.get(cmd, "move")
    
    # Power Level
    level = "Major" if speed >= 0.24 else "Minor"
    
    # Parse Reason
    reason_str = str(reason) if reason else ""
    parts = reason_str.split("|")
    gap_info = parts[0] if parts else ""
    delta_info = parts[1] if len(parts) > 1 else ""
    
    speed_str = f"{speed:.2f}p"
    # Format Line 1
    if ":" in gap_info:
        # e.g. "angle:19.18deg" -> "close the 19.18deg gap"
        _, gap_val = gap_info.split(":", 1)
        line1 = f"[ACT] {level} {action} {speed_str} to close the {gap_val} gap."
    else:
        line1 = f"[ACT] {level} {action} {speed_str} ({reason_str})"
        
    # Format Line 2 (Progress)
    line2 = ""
    if delta_info and ":" in delta_info:
        # e.g. "closer:2.13deg" -> "🟢 The last act got us 2.13deg closer to perfect alignment."
        direction, delta_val = delta_info.split(":", 1)
        emoji = "🟢" if direction == "closer" else "🔴"
        word = "closer to" if direction == "closer" else "further from"
        line2 = f"\n{emoji} The last act got us {delta_val} {word} perfect alignment."
        
    return f"{line1}{line2}"


def record_action_display(world, step, cmd, speed, speed_score=None):
    if world is None or not cmd:
        return
    obj_key = normalize_step_label(step)
    world._last_action_obj = obj_key
    world._last_action_time = time.time()
    world._last_action_cmd = cmd
    world._last_action_speed = speed
    world._last_action_score = speed_score
    score_suffix = f" {int(speed_score)}%" if speed_score is not None else ""
    world._last_action_display = f"{cmd.upper()} {speed:.2f}p{score_suffix}"


def send_robot_command(robot, world, step, cmd, speed, speed_score=None):
    if robot is None or cmd is None:
        return
    quantized_speed, quantized_score = telemetry_robot_module.quantize_speed(
        cmd,
        speed=speed,
        score=speed_score,
    )
    if quantized_score is None:
        quantized_score = speed_score
    record_action_display(world, step, cmd, quantized_speed, speed_score=quantized_score)
    robot.send_command(cmd, quantized_speed)


def step_uses_alignment_control(step, process_rules):
    obj_key = normalize_step_label(step)
    rules = (process_rules or {}).get(obj_key, {})
    controller = rules.get("controller")
    if controller == "replay":
        return False
    if controller == "align":
        return True
    success_gates = rules.get("success_gates") or {}
    return any(metric in success_gates for metric in ALIGNMENT_METRICS)


def derive_action_speeds(steps, fallback_score=DEFAULT_SPEED_SCORE):
    def avg(cmds):
        values = [step.speed for step in steps if step.cmd in cmds]
        if not values:
            return None
        return sum(values) / len(values)

    turn = avg({"l", "r"})
    forward = avg({"f"})
    backward = avg({"b"})
    if backward is None:
        backward = forward

    def pick(value, cmd):
        if value is None:
            value = default_speed_for_cmd(cmd, fallback_score)
        cap = max_speed_for_cmd(cmd)
        return min(value, cap)

    turn = pick(turn, "l")
    forward = pick(forward, "f")
    backward = pick(backward, "b")
    return {
        "turn": turn,
        "scan": turn,
        "forward": forward,
        "backward": backward,
    }


def alignment_command(world, step, gate_bounds, speeds, preview=False):
    analytics = telemetry_brick.compute_alignment_analytics(
        world,
        world.process_rules or {},
        world.learned_rules or {},
        step,
        duration_s=CONTROL_DT,
    )
    cmd = analytics.get("cmd")
    speed = analytics.get("speed") or 0.0
    reason = analytics.get("worst_metric") or "align"
    return cmd, speed, reason


def adjust_speed_for_find_brick(world, step, speed):
    if speed is None or speed <= 0:
        return speed
    obj_key = normalize_step_label(step)
    if obj_key != "FIND_BRICK":
        return speed
    brick = world.brick or {}
    if not brick.get("visible"):
        return speed
    return speed / FIND_BRICK_SLOW_FACTOR


def success_visible_target(world, step):
    step_key = normalize_step_label(step)
    rules = world.process_rules.get(step_key, {}) if world.process_rules else {}
    visible_gate = (rules.get("success_gates") or {}).get("visible", {})
    if "min" in visible_gate:
        return bool(visible_gate.get("min"))
    if "max" in visible_gate:
        return bool(visible_gate.get("max"))
    return None




def step_elapsed_s(world):
    start_time = getattr(world, "_step_start_time", None)
    if start_time is None:
        return None
    return max(0.0, time.time() - start_time)


def step_confidence(success_ok, world, step):
    if not success_ok:
        if getattr(world, "_success_start_time", None) is not None:
            world._success_start_time = None
        world._success_confirm_frames = 0
        world._success_confirm_progress = None
        world._success_confirm_logged = False
        return 0.0
    now = time.time()
    confirm_frames = getattr(world, "_success_confirm_frames", 0) + 1
    world._success_confirm_frames = confirm_frames
    if getattr(world, "_success_start_time", None) is None:
        world._success_start_time = now
    start_conf = max(0.0, min(SUCCESS_CONFIRMATION_START_CONFIDENCE, SUCCESS_CONFIDENCE_MIN))
    if SUCCESS_CONFIRMATION_FRAMES <= 1:
        confirm_progress = 1.0
    else:
        confirm_progress = (confirm_frames - 1) / (SUCCESS_CONFIRMATION_FRAMES - 1)
        confirm_progress = max(0.0, min(1.0, confirm_progress))
    if confirm_progress < 1.0:
        confirm_conf = start_conf + (SUCCESS_CONFIDENCE_MIN - start_conf) * confirm_progress
    else:
        confirm_conf = SUCCESS_CONFIDENCE_MIN
    world._success_confirm_progress = confirm_progress

    combined_conf = confirm_conf
    if confirm_progress is not None and combined_conf < start_conf:
        combined_conf = start_conf
    return combined_conf


def confidence_suspect(success_ok, confidence):
    if not success_ok or confidence is None:
        return False
    return confidence < SUCCESS_CONFIDENCE_MIN


def apply_confidence_speed(speed, success_ok, confidence, world=None):
    if speed is None:
        return speed
    return speed


def apply_pursuit_speed(speed):
    if speed is None:
        return speed
    return min(max(0.0, speed), 1.0)


def log_confidence(world, confidence, step):
    if isinstance(step, MotionStep):
        return
    if confidence is None:
        return
    if normalize_step_label(step) == "EXIT_WALL":
        return
    confirm_progress = getattr(world, "_success_confirm_progress", None)
    if confirm_progress is not None and confirm_progress < 1.0:
        pct = int(round(confidence * 100))
        print(format_headline(f"[CONFIRM] {step} {pct}% confidence", COLOR_WHITE))
        return
    if confirm_progress is not None and not getattr(world, "_success_confirm_logged", False):
        pct = int(round(confidence * 100))
        print(format_headline(f"[CONFIRM] {step} {pct}% confidence", COLOR_WHITE))
        world._success_confirm_logged = True
        return
    if confidence < CONFIDENCE_LOG_MIN or confidence >= SUCCESS_CONFIDENCE_MIN:
        return
    pct = int(round(confidence * 100))
    print(format_headline(f"[CONF] {step} {pct}% confidence", COLOR_WHITE))


def pause_after_fail(robot):
    if robot:
        robot.stop()
    time.sleep(FAIL_PAUSE_S)


def load_process_model(path=PROCESS_MODEL_FILE):
    if not path.exists():
        return {"steps": {}}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"steps": {}}


def refresh_autobuild_config(path=PROCESS_MODEL_FILE):
    cfg = dict(DEFAULT_AUTOBUILD_CONFIG)
    model = load_process_model(path)
    model_cfg = model.get("autobuild") if isinstance(model, dict) else None
    if isinstance(model_cfg, dict):
        cfg.update(model_cfg)
    apply_autobuild_config(cfg)
    return cfg


def write_process_model(model, path=PROCESS_MODEL_FILE):
    path.write_text(json.dumps(model, indent=4))


def step_metrics_map():
    metrics = {}
    for obj, items in telemetry_brick.METRICS_BY_STEP.items():
        metrics.setdefault(obj, []).extend([m for m in items if m not in metrics.get(obj, [])])
    for obj, items in telemetry_robot_module.METRICS_BY_STEP.items():
        metrics.setdefault(obj, []).extend([m for m in items if m not in metrics.get(obj, [])])
    return metrics


def success_gate_metrics_for_step(metrics, step, step_rules=None):
    obj_key = normalize_step_label(step)
    rules = (step_rules or {}).get(obj_key, {})
    allowed = rules.get("success_metrics")
    if isinstance(allowed, list):
        return [metric for metric in metrics if metric in allowed]
    success_gates = rules.get("success_gates")
    if isinstance(success_gates, dict) and success_gates:
        return [metric for metric in metrics if metric in success_gates]
    return metrics


def metric_value_from_state(state, metric):
    brick = state.get("brick") or {}
    if metric == "angle_abs":
        val = brick.get("angle")
        return abs(val) if val is not None else None
    if metric == "xAxis_offset_abs":
        val = brick.get("x_axis")
        if val is None:
            val = brick.get("offset_x")
        return float(val) if val is not None else None
    if metric == "dist":
        val = brick.get("dist")
        if val is None or val <= 0:
            return None
        return float(val)
    if metric == "visible":
        return bool(brick.get("visible"))
    if metric == "confidence":
        return brick.get("confidence")
    if metric == "lift_height":
        return state.get("lift_height")
    return None


def metric_direction(metric, step):
    if metric in telemetry_brick.METRIC_DIRECTIONS:
        return telemetry_brick.metric_direction_for_step(metric, step)
    return telemetry_robot_module.METRIC_DIRECTIONS.get(metric)


def collect_metric_values(segments, metrics):
    metric_values = {metric: [] for metric in metrics}
    for seg in segments:
        for state in select_tail_states(seg.get("states") or []):
            for metric in metrics:
                value = metric_value_from_state(state, metric)
                if value is None:
                    continue
                metric_values[metric].append(value)
    return metric_values


def collect_segments(logs):
    segments_by_obj = {}
    attempt_types = {}
    for _, data in logs:
        for seg in extract_attempt_segments(data):
            obj = normalize_step_label(seg.get("step"))
            seg_type = seg.get("type")
            if not obj or not seg_type:
                continue
            attempt_types.setdefault(obj, set()).add(seg_type)
            segments_by_obj.setdefault(obj, {}).setdefault(seg_type, []).append(seg)
    return segments_by_obj, attempt_types


def derive_start_gates(success_segments):
    start_gates = {}
    for obj, segs in success_segments.items():
        total = 0
        visible_count = 0
        for seg in segs:
            states = seg.get("states") or []
            if not states:
                continue
            brick = states[0].get("brick") or {}
            total += 1
            if brick.get("visible"):
                visible_count += 1
        if total <= 0:
            continue
        visible_ratio = visible_count / total
        if visible_ratio >= 0.5:
            start_gates[obj] = {"visible": {"min": True}}
        elif visible_ratio <= 0.5:
            start_gates[obj] = {"visible": {"min": False}}
    return start_gates


def derive_success_gates(success_segments, scale_by_step=None, step_rules=None):
    metrics_by_obj = step_metrics_map()
    scale_by_step = scale_by_step or {}

    success_gates = {}
    for obj, segs in success_segments.items():
        metrics = success_gate_metrics_for_step(
            metrics_by_obj.get(obj, []),
            obj,
            step_rules,
        )
        if not metrics:
            continue
        metric_values = collect_metric_values(segs, metrics)

        visible_gate = None
        if metric_values.get("visible"):
            visible_ratio = sum(1 for v in metric_values["visible"] if v) / len(metric_values["visible"])
            visible_gate = {"min": visible_ratio >= 0.5}
            if visible_gate["min"] is False:
                success_gates[obj] = {"visible": visible_gate}
                continue

        obj_gates = {}
        if visible_gate is not None:
            obj_gates["visible"] = visible_gate
        for metric, values in metric_values.items():
            if not values or metric == "visible":
                continue

            p10 = telemetry_brick.percentile(values, 0.1)
            p50 = telemetry_brick.percentile(values, 0.5)
            p90 = telemetry_brick.percentile(values, 0.9)
            if p50 is None:
                continue
            tol = None
            if p10 is not None and p90 is not None:
                tol = max(abs(p50 - p10), abs(p90 - p50))
            scale = scale_by_step.get(obj, 1.0)
            if tol is not None:
                tol *= scale
            if metric in MM_METRICS:
                if tol is None:
                    tol = MIN_MM_TOL
                else:
                    tol = max(tol, MIN_MM_TOL)

            stats = {"target": round_value(p50)}
            if tol is not None:
                stats["tol"] = round_value(tol)
            obj_gates[metric] = stats

        if obj_gates:
            success_gates[obj] = obj_gates
    return success_gates


def derive_success_durations(success_segments):
    durations = {}
    for obj, segs in success_segments.items():
        values = []
        for seg in segs:
            start = seg.get("start")
            end = seg.get("end")
            if start is None or end is None:
                timestamps = [
                    state.get("timestamp")
                    for state in (seg.get("states") or [])
                    if state.get("timestamp") is not None
                ]
                if not timestamps:
                    timestamps = [
                        evt.get("timestamp")
                        for evt in (seg.get("events") or [])
                        if evt.get("timestamp") is not None
                    ]
                if timestamps:
                    start = min(timestamps)
                    end = max(timestamps)
            if start is None or end is None:
                continue
            duration = end - start
            if duration is None or duration <= 0:
                continue
            values.append(duration)
        if not values:
            continue
        p10 = telemetry_brick.percentile(values, 0.1)
        p50 = telemetry_brick.percentile(values, 0.5)
        p90 = telemetry_brick.percentile(values, 0.9)
        if p50 is None:
            continue
        tol = None
        if p10 is not None and p90 is not None:
            tol = max(abs(p50 - p10), abs(p90 - p50))
        stats = {"target": round_value(p50)}
        if tol is not None:
            stats["tol"] = round_value(tol)
        durations[obj] = stats
    return durations


def refine_success_gates_with_failures(success_gates, fail_segments, step_rules=None):
    if not success_gates or not fail_segments:
        return success_gates
    metrics_by_obj = step_metrics_map()
    for obj, gates in success_gates.items():
        if not isinstance(gates, dict):
            continue
        # Tighten success tolerances when failure samples overlap the current success window.
        metrics = success_gate_metrics_for_step(
            metrics_by_obj.get(obj, []),
            obj,
            step_rules,
        )
        if not metrics:
            continue
        fail_values = collect_metric_values(fail_segments.get(obj, []), metrics)
        for metric, stats in gates.items():
            if not isinstance(stats, dict):
                continue
            if metric == "visible":
                continue
            values = fail_values.get(metric)
            if not values:
                continue

            direction = metric_direction(metric, obj)
            if direction not in ("low", "high"):
                continue

            target = stats.get("target")
            tol = stats.get("tol")
            if target is not None and tol is not None:
                if direction == "low":
                    failure_floor = telemetry_brick.percentile(values, FAILURE_TIGHTEN_LOW_PCT)
                    if failure_floor is None:
                        continue
                    success_max = target + tol
                    if failure_floor <= target or failure_floor >= success_max:
                        continue
                    stats["tol"] = round_value(failure_floor - target)
                else:
                    failure_ceiling = telemetry_brick.percentile(values, FAILURE_TIGHTEN_HIGH_PCT)
                    if failure_ceiling is None:
                        continue
                    success_min = target - tol
                    if failure_ceiling >= target or failure_ceiling <= success_min:
                        continue
                    stats["tol"] = round_value(target - failure_ceiling)
                continue

            if direction == "low":
                failure_floor = telemetry_brick.percentile(values, FAILURE_TIGHTEN_LOW_PCT)
                if failure_floor is None:
                    continue
                max_val = stats.get("max")
                if max_val is None or failure_floor >= max_val:
                    continue
                stats["max"] = round_value(failure_floor)
            else:
                failure_ceiling = telemetry_brick.percentile(values, FAILURE_TIGHTEN_HIGH_PCT)
                if failure_ceiling is None:
                    continue
                min_val = stats.get("min")
                if min_val is None or failure_ceiling <= min_val:
                    continue
                stats["min"] = round_value(failure_ceiling)
    return success_gates


def derive_movement_learnings(segments_by_obj):
    learnings = {}
    for obj, segs_by_type in segments_by_obj.items():
        segments = []
        segments.extend(segs_by_type.get("SUCCESS") or [])
        segments.extend(segs_by_type.get("NOMINAL") or [])
        if not segments:
            continue

        aggregate_by_cmd = {}
        for seg in segments:
            steps = merge_motion_steps(build_motion_sequence(seg.get("events") or []))
            if not steps:
                continue
            per_cmd = {}
            for step in steps:
                entry = per_cmd.setdefault(step.cmd, {"duration": 0.0, "speed_sum": 0.0})
                entry["speed_sum"] += step.speed
            for cmd, entry in per_cmd.items():
                if entry["duration"] <= 0:
                    continue
                avg_speed = entry["speed_sum"] / entry["duration"]
                rollup = aggregate_by_cmd.setdefault(cmd, {"durations": [], "speeds": []})
                rollup["durations"].append(entry["duration"])
                rollup["speeds"].append(avg_speed)

        if not aggregate_by_cmd:
            continue

        cmd_stats = {}
        for cmd, values in aggregate_by_cmd.items():
            duration_avg = sum(values["durations"]) / len(values["durations"])
            speed_avg = sum(values["speeds"]) / len(values["speeds"])
            cmd_stats[cmd] = {
                "speed": round_value(speed_avg),
            }

        dominant_cmd, stats = max(
            cmd_stats.items(),
            )
        learnings[obj] = {"cmd": dominant_cmd, **stats}
    return learnings


def update_process_model_from_demos(logs, path=PROCESS_MODEL_FILE):
    model = load_process_model(path)
    steps = model.get("steps")
    if not isinstance(steps, dict):
        steps = {}
        model["steps"] = steps

    segments_by_obj, attempt_types = collect_segments(logs)
    success_segments = {
        obj: segs.get("SUCCESS", [])
        for obj, segs in segments_by_obj.items()
        if segs.get("SUCCESS")
    }
    fail_segments = {
        obj: segs.get("FAIL", [])
        for obj, segs in segments_by_obj.items()
        if segs.get("FAIL")
    }

    start_gates = derive_start_gates(success_segments)
    wall_step_rules = telemetry_wall.load_wall_step_rules()
    step_rules = {}
    if isinstance(wall_step_rules, dict):
        step_rules.update(wall_step_rules)
    if isinstance(steps, dict):
        step_rules.update(steps)
    success_gate_scales = derive_success_gate_scales(segments_by_obj, step_rules)

    # Train Policy if requested
    if USE_LEARNED_POLICY:
        global GLOBAL_POLICY
        print(format_headline("[LEARNING REPLAY] Training Policy from Demos...", COLOR_GREEN))
        GLOBAL_POLICY = helper_learning.BehavioralCloningPolicy()
        GLOBAL_POLICY.train(segments_by_obj)

    success_gates = derive_success_gates(
        success_segments,
        success_gate_scales,
        step_rules,
    )
    success_durations = derive_success_durations(success_segments)
    success_gates = refine_success_gates_with_failures(
        success_gates,
        fail_segments,
        step_rules,
    )
    for obj, stats in []:
        if obj not in success_gates:
            continue

    movement_learnings = derive_movement_learnings(segments_by_obj)

    all_steps = set(steps.keys())
    all_steps.update(start_gates.keys())
    all_steps.update(success_gates.keys())
    all_steps.update(movement_learnings.keys())
    all_steps.update(attempt_types.keys())

    for obj in all_steps:
        cfg = steps.setdefault(obj, {})

        if obj in start_gates:
            cfg["start_gates"] = start_gates[obj]
        else:
            cfg.pop("start_gates", None)

        if obj in success_gates:
            cfg["success_gates"] = success_gates[obj]
        else:
            cfg.pop("success_gates", None)
        cfg.pop("fail_gates", None)
        visible_gate = cfg.get("success_gates", {}).get("visible", {})
        if visible_gate.get("min") is False:
            cfg["success_gates"] = {"visible": {"min": False}}

        if obj in movement_learnings:
            cfg["movement_learnings"] = movement_learnings[obj]
        else:
            cfg.pop("movement_learnings", None)

        obj_types = attempt_types.get(obj, set())
        if obj_types == {"NOMINAL"}:
            cfg["nominalDemosOnly"] = True
        else:
            cfg.pop("nominalDemosOnly", None)

        for key in ("start_gates", "success_gates"):
            if key in cfg and not cfg[key]:
                cfg.pop(key)

    ordered = {}
    for name in steps.keys():
        ordered[name] = steps[name]
    for name in sorted(all_steps):
        if name not in ordered:
            ordered[name] = steps.get(name, {})
    model["steps"] = ordered

    write_process_model(model, path)
    return model


def build_motion_sequence(events):
    steps = []
    for evt in events:
        if evt.get("type") == "action":
            cmd_name = evt.get("command")
            power = evt.get("power", 0)
            duration_ms = evt.get("duration_ms", 0)
        elif evt.get("type") == "event":
            payload = evt.get("event") or {}
            cmd_name = payload.get("type")
            power = payload.get("power", 0)
            duration_ms = payload.get("duration_ms", 0)
        else:
            continue

        cmd = ACTION_CMD_MAP.get(cmd_name)
        duration_s = (duration_ms or 0) / 1000.0
        if not cmd or duration_s <= 0:
            continue
        speed = max(0.0, min(1.0, float(power or 0) / 255.0))
        if speed <= 0:
            continue
        steps.append(MotionStep(cmd, speed, duration_s, cmd_name))
    return steps


def merge_motion_steps(steps, speed_tol=0.02):
    if not steps:
        return []
    merged = [steps[0]]
    for step in steps[1:]:
        last = merged[-1]
        if step.cmd == last.cmd and abs(step.speed - last.speed) <= speed_tol:
            last.duration_s += step.duration_s
        else:
            merged.append(step)
    return merged


def smooth_motion_steps(steps, speed_score=DEFAULT_SPEED_SCORE, step_s=SMOOTH_STEP_S):
    if not steps:
        return []
    smooth = []
    for step in steps:
        total_duration = step.duration_s
        chunks = max(1, int(math.ceil(total_duration / step_s)))
        for idx in range(chunks):
            duration = min(step_s, total_duration - (idx * step_s))
            if duration <= 0:
                continue
            base_speed = default_speed_for_cmd(step.cmd, speed_score)
            capped_speed = min(base_speed, max_speed_for_cmd(step.cmd))
            smooth.append(MotionStep(step.cmd, capped_speed, duration, step.label))
    return smooth


def select_demo_segment(segments_by_obj, step, nominal_only):
    obj_key = normalize_step_label(step)
    if not obj_key:
        return None, None

    prefer = ["NOMINAL", "SUCCESS"] if nominal_only else ["SUCCESS", "NOMINAL"]
    candidates = []
    chosen_type = None
    for attempt_type in prefer:
        candidates = segments_by_obj.get(obj_key, {}).get(attempt_type, [])
        if candidates:
            chosen_type = attempt_type
            break

    if not candidates:
        return None, None

    def score(seg):
        events = seg.get("events") or []
        duration = 0.0
        if seg.get("start") is not None and seg.get("end") is not None:
            duration = seg["end"] - seg["start"]
        return (len(events), duration)

    candidates.sort(key=score, reverse=True)
    return candidates[0], chosen_type


def update_world_from_vision(world, vision, log=True):
    now = time.time()
    sec = int(now)
    ms = int((now - sec) * 1000)
    cam_times = getattr(world, "_camera_frame_times", [])
    cam_times.append((sec, ms))
    if len(cam_times) > 3:
        cam_times.pop(0)
    world._camera_frame_times = cam_times
    ms_values = [t[1] for t in cam_times]
    world._camera_dupe_ms = len(ms_values) != len(set(ms_values))
    stamp = (sec, ms)

    found, angle, dist, offset_x, conf, cam_h, brick_above, brick_below = vision.read()
    if dist == -1:
        found = False
        angle = 0.0
        dist = 0.0
        offset_x = 0.0
        conf = 0.0
        cam_h = 0.0
        brick_above = False
        brick_below = False
    frame = {
        "found": bool(found),
        "dist": float(dist),
        "angle": float(angle),
        "offset_x": float(offset_x),
        "conf": float(conf),
        "cam_h": float(cam_h),
        "brick_above": bool(brick_above),
        "brick_below": bool(brick_below),
        "stamp": stamp,
    }
    buffer = getattr(world, "_brick_frame_buffer", None)
    if buffer is None:
        buffer = []
        setattr(world, "_brick_frame_buffer", buffer)
    last_stamp = buffer[-1]["stamp"] if buffer else None
    if stamp != last_stamp:
        buffer.append(frame)
    if len(buffer) > telemetry_brick.BRICK_SMOOTH_FRAMES:
        buffer.pop(0)
    world._camera_unique_frames = len(buffer)
    avg = None
    obs_note = None
    if len(buffer) >= telemetry_brick.BRICK_SMOOTH_FRAMES:
        avg, should_reset, obs_note = telemetry_brick._filtered_brick_frame_average(buffer)
        if should_reset and avg is None:
            buffer.clear()
    if obs_note:
        world._last_obs_note = obs_note
    if avg is not None:
        world._brick_inconsistent = False
        world.update_vision(
            avg["found"],
            avg["dist"],
            avg["angle"],
            avg["conf"],
            avg["offset_x"],
            avg["cam_h"],
            avg["brick_above"],
            avg["brick_below"],
        )
        world._frame_id = getattr(world, "_frame_id", 0) + 1
        if log:
            current_obj = getattr(world, "step_state", None)
            current_name = getattr(current_obj, "value", None)
            if normalize_step_label(current_name) == "EXIT_WALL":
                pass
            elif not getattr(world, "suppress_brick_state_log", False):
                print(format_headline(format_brick_state_line(world), COLOR_WHITE))
            if getattr(world, "log_brick_frames", False):
                frame_note = obs_note or "ok"
                raw_frames = []
                for idx, fr in enumerate(buffer[-telemetry_brick.BRICK_SMOOTH_FRAMES:]):
                    raw_frames.append(
                        f"{idx+1}:v={int(fr['found'])} d={fr['dist']:.1f} a={fr['angle']:.1f} "
                        f"x={fr['offset_x']:.1f} c={fr['conf']:.1f}"
                    )
                avg_line = (
                    f"avg:v={int(avg['found'])} d={avg['dist']:.1f} a={avg['angle']:.1f} "
                    f"x={avg['offset_x']:.1f} c={avg['conf']:.1f}"
                )
                print(
                    format_headline(
                        "[BRICK] frames [" + " | ".join(raw_frames) + f"] -> {avg_line} ({frame_note})",
                        COLOR_WHITE,
                    )
                )
    elif len(buffer) >= telemetry_brick.BRICK_SMOOTH_FRAMES:
        # Mark inconsistent snapshot: force confidence to 0 and visible false until we stabilize.
        world._brick_inconsistent = True
        world.brick["visible"] = False
        world.brick["confidence"] = 0.0
        world.brick["dist"] = 0.0
        world.brick["angle"] = 0.0
        world.brick["offset_x"] = 0.0
        world.brick["x_axis"] = 0.0
        world.brick["brickAbove"] = False
        world.brick["brickBelow"] = False
    else:
        world._brick_inconsistent = True
        world.brick["visible"] = False
        world.brick["confidence"] = 0.0
        world.brick["dist"] = 0.0
        world.brick["angle"] = 0.0
        world.brick["offset_x"] = 0.0
        world.brick["x_axis"] = 0.0
        world.brick["brickAbove"] = False
        world.brick["brickBelow"] = False
    update_stream_frame(world, vision)


def _gate_value_line(metric, value, stats):
    if value is None:
        val_str = "n/a"
    elif isinstance(value, bool):
        val_str = "true" if value else "false"
    else:
        val_str = f"{value:.2f}"
    gate_desc = ""
    if isinstance(stats, dict):
        target = stats.get("target")
        tol = stats.get("tol")
        min_val = stats.get("min")
        max_val = stats.get("max")
        # Round all values to 2 decimal places
        if target is not None:
            target = round(float(target), 2)
        if tol is not None:
            tol = round(float(tol), 2)
        if min_val is not None:
            min_val = round(float(min_val), 2)
        if max_val is not None:
            max_val = round(float(max_val), 2)
        if target is not None and tol is not None:
            gate_desc = f"target {target:.2f}±{tol:.2f}"
        elif min_val is not None and max_val is not None:
            gate_desc = f"{min_val:.2f}-{max_val:.2f}"
        elif min_val is not None:
            gate_desc = f">={min_val:.2f}"
        elif max_val is not None:
            gate_desc = f"<={max_val:.2f}"
    suffix = f" ({gate_desc})" if gate_desc else ""
    return f"{metric}: {val_str}{suffix}"


def compute_stream_gate_summary(world, step, active=True):
    if not active:
        return [], None
    obj_name = normalize_step_label(step)
    if not obj_name:
        obj_name = str(step)
    analytics = telemetry_brick.compute_brick_analytics(
        world,
        world.process_rules or {},
        world.learned_rules or {},
        obj_name,
        duration_s=CONTROL_DT,
    )
    pct = None
    for name, val in analytics.get("gate_progress") or []:
        if normalize_step_label(name) == obj_name:
            pct = val
            break
    if pct is None:
        success_check = telemetry_brick.evaluate_success_gates(world, obj_name, world.learned_rules or {}, world.process_rules)
        pct = 100.0 if success_check.ok else 0.0
    pct_display = int(max(0.0, min(100.0, pct)))
    success_gates = (world.process_rules or {}).get(obj_name, {}).get("success_gates") or {}
    metric_lines = []
    for metric, stats in success_gates.items():
        if metric == "lift_height":
            value = world.lift_height
        else:
            value = telemetry_brick.metric_value(world.brick or {}, metric)
        if metric == 'visible':
            metric_lines.append(f"visible: {'true' if bool(value) else 'false'}")
        else:
            metric_lines.append(_gate_value_line(metric, value, stats))
    if getattr(world, "_brick_inconsistent", False):
        metric_lines.append(("confidence: 0.00 (inconsistent)", (0, 0, 255)))
    suggestion = analytics.get("suggestion") or "HOLD"
    last_display = getattr(world, "_last_action_display", None)
    last_obj = getattr(world, "_last_action_obj", None)
    last_time = getattr(world, "_last_action_time", None)
    use_last = False
    if last_display and last_obj == obj_name:
        if last_time is None or (time.time() - last_time) <= 1.5:
            suggestion = last_display
            use_last = True
    cmd = analytics.get("cmd")
    speed = analytics.get("speed") if cmd else None
    if cmd and speed is not None and not use_last:
        success_ok, confidence = evaluate_gate_status(world, obj_name)
        speed = apply_pursuit_speed(speed)
        speed = apply_confidence_speed(speed, success_ok, confidence, world)
        speed_score = analytics.get("speed_score")
        speed, speed_score = telemetry_robot_module.quantize_speed(
            cmd,
            speed=speed,
            score=speed_score,
        )
        score_suffix = f" {int(speed_score)}%" if speed_score is not None else ""
        suggestion = f"{cmd.upper()} {speed:.2f}p{score_suffix}"
    if suggestion == "HOLD":
        scan_dir = (world.process_rules or {}).get(obj_name, {}).get("scan_direction")
        if scan_dir in ("l", "r"):
            scan_speed = default_speed_for_cmd(scan_dir)
            suggestion = f"{scan_dir.upper()} {scan_speed:.2f}p {int(DEFAULT_SPEED_SCORE)}%"
    summary = [f"STEP: {obj_name}", f"SUCCESS: {pct_display}%"]
    summary.extend(metric_lines)
    summary.append(f"ACT: {suggestion}")
    return summary, analytics


def update_stream_frame(world, vision):
    stream_state = getattr(world, "_stream_state", None)
    if stream_state is None or vision.current_frame is None:
        return
    gate_summary, analytics = compute_stream_gate_summary(
        world,
        world.step_state.value,
        active=True,
    )
    world._gate_summary = gate_summary
    world._highlight_metric = (analytics or {}).get("highlight_metric")
    frame = vision.current_frame.copy()
    draw_telemetry_overlay(
        frame,
        world,
        show_prompt=False,
        gate_status=None,
        gate_progress=None,
        step_suggestions=None,
        highlight_metric=getattr(world, "_highlight_metric", None),
        loop_id=getattr(world, "loop_id", None),
        gate_summary=getattr(world, "_gate_summary", []),
    )
    with stream_state["lock"]:
        stream_state["frame"] = frame


def refresh_world_after_action(world, vision, log=True, attempts=8):
    start_frame_id = getattr(world, "_frame_id", 0)
    last_frame = None
    for _ in range(attempts):
        update_world_from_vision(world, vision, log=log)
        frame_id = getattr(world, "_frame_id", 0)
        if log and getattr(world, "log_fresh_frames", False):
            print(
                format_headline(
                    f"[FRAME] action refresh frame_id={frame_id} (start={start_frame_id})",
                    COLOR_WHITE,
                )
            )
        if world.brick:
            last_frame = {
                "dist": world.brick.get("dist"),
                "angle": world.brick.get("angle"),
                "x_axis": world.brick.get("x_axis"),
                "frame_id": frame_id,
            }
        # Prefer a new observed frame over fixed sleeps.
        if frame_id > start_frame_id:
            if log and getattr(world, "log_fresh_frames", False):
                print(
                    format_headline(
                        f"[FRAME] fresh image observed (frame_id {frame_id} > {start_frame_id})",
                        COLOR_GREEN,
                    )
                )
            return last_frame
        time.sleep(CONTROL_DT * 0.5)
    return last_frame


def evaluate_gate_status(world, step):
    process_rules = world.process_rules or {}
    telemetry_rules = {}
    visibility_grace_s = None
    if success_visible_target(world, step) is False:
        visibility_grace_s = SUCCESS_VISIBLE_FALSE_GRACE_S
    brick_success = telemetry_brick.evaluate_success_gates(
        world,
        step,
        telemetry_rules,
        process_rules,
        visibility_grace_s=visibility_grace_s,
    )
    wall_success = telemetry_wall.evaluate_success_gates(world, step, world.wall_envelope)
    robot_success = telemetry_robot_module.evaluate_success_gates(world, step, telemetry_rules, process_rules)

    success_ok = brick_success.ok and wall_success.ok and robot_success.ok
    confidence = step_confidence(success_ok, world, step)
    return success_ok, confidence


def wait_for_start_gates(
    world,
    vision,
    step,
    robot=None,
    cmd=None,
    speed=None,
    log=True,
    observer=None,
):
    start_time = time.time()
    stable = 0
    success_tracker = SuccessGateTracker(success_frames_required(step))
    success_seen = False
    last_cmd = None
    last_speed = None
    if robot:
        robot.stop()
    while time.time() - start_time < START_GATE_TIMEOUT_S:
        update_world_from_vision(world, vision, log=log)
        if observer:
            observer("frame", world, vision, None, None, None)
        success_ok, confidence = evaluate_gate_status(world, step)
        if log:
            log_confidence(world, confidence, step)
        if success_ok:
            if robot:
                robot.stop()
            return "success"
        confidence_ok = success_ok and confidence >= SUCCESS_CONFIDENCE_MIN
        success_met = success_tracker.update(confidence_ok)
        if success_ok and not success_seen and robot:
            robot.stop()
            success_seen = True
        if success_met:
            if robot:
                robot.stop()
            return "success"
        success_seen = False

        brick_check = telemetry_brick.evaluate_start_gates(world, step, {}, world.process_rules)
        wall_check = telemetry_wall.evaluate_start_gates(world, step, world.wall_envelope)
        robot_check = telemetry_robot_module.evaluate_start_gates(world, step, {}, world.process_rules)
        if brick_check.ok and wall_check.ok and robot_check.ok:
            stable += 1
            if stable >= GATE_STABILITY_FRAMES:
                if log:
                    print(format_headline(f"[START] {step} start gates met", COLOR_GREEN))
                return "start"
        else:
            stable = 0
            if robot and cmd and speed:
                if log and (cmd != last_cmd or speed != last_speed):
                    print(format_headline(format_control_action_line(cmd, speed, "start_gate"), COLOR_WHITE))
                    last_cmd = cmd
                    last_speed = speed
                send_robot_command(robot, world, step, cmd, speed)
                evt = MotionEvent(
                    cmd_to_motion_type(cmd),
                    int(speed * 255),
                    int(CONTROL_DT * 1000),
                )
                world.update_from_motion(evt)
            elif log and (cmd != last_cmd or speed != last_speed):
                hold_reason = "start_gate: missing cmd/speed"
                print(format_headline(format_control_action_line(None, 0.0, hold_reason), COLOR_WHITE))
                last_cmd = cmd
                last_speed = speed
        time.sleep(CONTROL_DT)
    return "timeout"


def run_alignment_segment(
    segment,
    step,
    robot,
    vision,
    world,
    steps,
    raw_steps,
    observer=None,
    analysis_pause_s=0.0,
    confirm_callback=None,
    align_silent=False,
):
    action_speeds = derive_action_speeds(raw_steps)
    gate_bounds = telemetry_brick.success_gate_bounds(
        world.process_rules or {},
        world.learned_rules or {},
        step,
    )
    step_key = normalize_step_label(step)
    rules = (world.process_rules or {}).get(step_key, {})
    movement = rules.get("movement_learnings") if isinstance(rules, dict) else None
    scan_cmd = telemetry_robot_module.resolve_scan_direction(world.process_rules, step)
    start_cmd = scan_cmd
    start_speed = action_speeds["scan"]
    if isinstance(movement, dict):
        learned_cmd = movement.get("cmd")
        learned_speed = movement.get("speed")
        if learned_cmd in ("f", "b", "l", "r"):
            start_cmd = learned_cmd
            if learned_speed is not None:
                start_speed = max(0.0, min(1.0, float(learned_speed)))
    start_status = wait_for_start_gates(
        world,
        vision,
        step,
        robot=robot,
        cmd=start_cmd,
        speed=start_speed,
        log=not align_silent,
        observer=observer,
    )
    if start_status == "success":
        if robot:
            robot.stop()
        if not align_silent:
            print(format_headline(f"[SUCCESS] {step} criteria met", COLOR_GREEN))
            print(format_headline(format_success_details(world, step), COLOR_WHITE))
        return True, "success gate"
    if start_status != "start":
        pause_after_fail(robot)
        return False, "start gates not met"

    if telemetry_brick.success_gates_visible_only(world.process_rules or {}, step):
        world._visible_speed_cycle = 0

    success_tracker = SuccessGateTracker(success_frames_required(step))
    last_cmd = None
    last_reason = None
    last_speed = None
    last_action_frame = None
    loop_id = 0

    while True:
        loop_id += 1
        world.loop_id = loop_id
        update_world_from_vision(world, vision, log=not align_silent)
        obs_note = getattr(world, "_last_obs_note", None)
        if obs_note:
            print(format_headline(f"[OBS] {obs_note}", COLOR_WHITE))
            world._last_obs_note = None
        # pause1 removed per request
        if observer:
            observer("frame", world, vision, None, None, None)
        success_ok, confidence = evaluate_gate_status(world, step)
        if not align_silent:
            log_confidence(world, confidence, step)
        if success_ok:
            if robot:
                robot.stop()
            if not align_silent:
                print(format_headline(f"[SUCCESS] {step} criteria met", COLOR_GREEN))
                print(format_headline(format_success_details(world, step), COLOR_WHITE))
            return True, "success gate"
        if success_ok:
            if robot:
                robot.stop()
            if observer:
                observer("action", world, vision, None, 0.0, "success gate")
            time.sleep(CONTROL_DT)
            continue

        analytics = telemetry_brick.compute_alignment_analytics(
            world,
            world.process_rules or {},
            world.learned_rules or {},
            step,
            duration_s=CONTROL_DT,
        )
        cmd = analytics.get("cmd")
        speed = analytics.get("speed") or 0.0
        cmd_reason = analytics.get("worst_metric") or "align"
        speed_score = analytics.get("speed_score")
        speed = apply_pursuit_speed(speed)
        speed = apply_confidence_speed(speed, success_ok, confidence, world)
        if cmd != last_cmd or cmd_reason != last_reason or speed != last_speed:
            if not align_silent:
                brick_success = telemetry_brick.evaluate_success_gates(
                    world,
                    step,
                    {},
                    world.process_rules or {},
                )
                wall_success = telemetry_wall.evaluate_success_gates(world, step, world.wall_envelope)
                robot_success = telemetry_robot_module.evaluate_success_gates(world, step, {}, world.process_rules or {})
                reasons = []
                reasons.extend(brick_success.reasons or [])
                reasons.extend(wall_success.reasons or [])
                reasons.extend(robot_success.reasons or [])
                if reasons:
                    print(format_headline(f"[ALIGN] Not within success gates: {', '.join(reasons)}", COLOR_WHITE))
                else:
                    print(format_headline("[ALIGN] Not within success gates: no reasons reported", COLOR_WHITE))
                reason = f"{cmd_reason} {int(speed_score)}%" if speed_score is not None else cmd_reason
                world._last_action_line = format_control_action_line(cmd, speed, reason)
            if not align_silent:
                pass
            if observer:
                observer("analysis", world, vision, cmd, speed, cmd_reason)
            if analysis_pause_s:
                time.sleep(analysis_pause_s)
            last_cmd = cmd
            last_reason = cmd_reason
            last_speed = speed

        if cmd:
            pre_frame_id = getattr(world, "_frame_id", 0)
            if confirm_callback:
                if not confirm_callback(world, vision):
                    return False, "confirm cancelled"
            send_robot_command(robot, world, step, cmd, speed, speed_score=analytics.get("speed_score"))
            evt = MotionEvent(
                cmd_to_motion_type(cmd),
                int(speed * 255),
                int(CONTROL_DT * 1000),
            )
            world.update_from_motion(evt)
            if telemetry_brick.success_gates_visible_only(world.process_rules or {}, step):
                world._visible_speed_cycle = int(getattr(world, "_visible_speed_cycle", 0)) + 1
            if confirm_callback and robot:
                robot.stop()
            last_action_frame = refresh_world_after_action(world, vision, log=not align_silent)
            post_frame_id = getattr(world, "_frame_id", 0)
            if post_frame_id <= pre_frame_id:
                print(format_headline("[WARN] No new frame observed after action", COLOR_RED))
        else:
            if robot:
                robot.stop()
        if observer:
            observer("action", world, vision, cmd, speed, cmd_reason)
        time.sleep(CONTROL_DT)
        if last_action_frame:
            current_frame = {
                "dist": world.brick.get("dist"),
                "angle": world.brick.get("angle"),
                "x_axis": world.brick.get("x_axis"),
            }
            if current_frame == last_action_frame:
                print(format_headline("[ERROR] Frame unchanged before/after action", COLOR_RED))

    if robot:
        robot.stop()
    if not align_silent:
        loops = getattr(world, "loop_id", 0)
        print(
            format_headline(
                f"[FAIL] Gave up after {loops} loops.",
                COLOR_RED,
            )
        )
    settle_deadline = time.time() + SUCCESS_SETTLE_S
    settle_tracker = SuccessGateTracker(success_frames_required(step))
    loop_id = 0
    while time.time() < settle_deadline:
        loop_id += 1
        world.loop_id = loop_id
        update_world_from_vision(world, vision, log=not align_silent)
        obs_note = getattr(world, "_last_obs_note", None)
        if obs_note:
            print(format_headline(f"[OBS] {obs_note}", COLOR_WHITE))
            world._last_obs_note = None
        if observer:
            observer("frame", world, vision, None, None, None)
        success_ok, confidence = evaluate_gate_status(world, step)
        if not align_silent:
            log_confidence(world, confidence, step)
        if success_ok:
            if not align_silent:
                print(format_headline(f"[SUCCESS] {step} criteria met", COLOR_GREEN))
                print(format_headline(format_success_details(world, step), COLOR_WHITE))
            return True, "success gate"
        if success_ok:
            if robot:
                robot.stop()
            time.sleep(CONTROL_DT)
            continue

        analytics = telemetry_brick.compute_alignment_analytics(
            world,
            world.process_rules or {},
            world.learned_rules or {},
            step,
            duration_s=CONTROL_DT,
        )
        cmd = analytics.get("cmd")
        speed = analytics.get("speed") or 0.0
        cmd_reason = analytics.get("worst_metric") or "align"
        speed_score = analytics.get("speed_score")
        speed = apply_pursuit_speed(speed)
        speed = apply_confidence_speed(speed, success_ok, confidence, world)
        if cmd != last_cmd or cmd_reason != last_reason or speed != last_speed:
            if not align_silent:
                brick_success = telemetry_brick.evaluate_success_gates(
                    world,
                    step,
                    {},
                    world.process_rules or {},
                )
                wall_success = telemetry_wall.evaluate_success_gates(world, step, world.wall_envelope)
                robot_success = telemetry_robot_module.evaluate_success_gates(world, step, {}, world.process_rules or {})
                reasons = []
                reasons.extend(brick_success.reasons or [])
                reasons.extend(wall_success.reasons or [])
                reasons.extend(robot_success.reasons or [])
                if reasons:
                    print(format_headline(f"[ALIGN] Not within success gates: {', '.join(reasons)}", COLOR_WHITE))
                else:
                    print(format_headline("[ALIGN] Not within success gates: no reasons reported", COLOR_WHITE))
                reason = f"{cmd_reason} {int(speed_score)}%" if speed_score is not None else cmd_reason
                world._last_action_line = format_control_action_line(cmd, speed, reason)
            last_cmd = cmd
            last_reason = cmd_reason
            last_speed = speed

        if cmd:
            if confirm_callback:
                if not confirm_callback(world, vision):
                    return False, "confirm cancelled"
            send_robot_command(robot, world, step, cmd, speed, speed_score=analytics.get("speed_score"))
            evt = MotionEvent(
                cmd_to_motion_type(cmd),
                int(speed * 255),
                int(CONTROL_DT * 1000),
            )
            world.update_from_motion(evt)
            if telemetry_brick.success_gates_visible_only(world.process_rules or {}, step):
                world._visible_speed_cycle = int(getattr(world, "_visible_speed_cycle", 0)) + 1
            if confirm_callback and robot:
                robot.stop()
        time.sleep(CONTROL_DT)

    pause_after_fail(robot)
    return False, "success gate not reached"


def replay_segment(
    segment,
    step,
    robot,
    vision,
    world,
    observer=None,
    analysis_pause_s=0.0,
    confirm_callback=None,
    align_silent=False,
):
    events = segment.get("events") or []
    raw_steps = merge_motion_steps(build_motion_sequence(events))
    steps = smooth_motion_steps(raw_steps)
    if step_uses_alignment_control(step, world.process_rules):
        return run_alignment_segment(
            segment,
            step,
            robot,
            vision,
            world,
            steps,
            raw_steps,
            observer=observer,
            analysis_pause_s=analysis_pause_s,
            confirm_callback=confirm_callback,
            align_silent=align_silent,
        )
    if not steps:
        return False, "no motion steps"

    default_step = steps[0]
    target_visible = success_visible_target(world, step)
    start_status = wait_for_start_gates(
        world,
        vision,
        step,
        robot=robot,
        cmd=default_step.cmd,
        speed=default_step.speed,
    )
    if start_status == "success":
        if robot:
            robot.stop()
        print(format_headline(f"[SUCCESS] {step} criteria met 🎉", COLOR_GREEN))
        print(format_headline(format_success_details(world, step), COLOR_WHITE))
        time.sleep(3.0)
        return True, "success gate"
    if start_status != "start":
        pause_after_fail(robot)
        return False, "start gates not met"

    allow_early_exit = True

    success_tracker = SuccessGateTracker(success_frames_required(step))
    last_action = None
    last_cmd = default_step.cmd
    last_speed_base = default_step.speed

    for step in steps:
        if step.label != last_action:
            update_world_from_vision(world, vision)
            if observer:
                observer("frame", world, vision, None, None, None)
            world._last_action_line = format_control_action_line(
                step.cmd,
                step.speed,
                "replay",
            )
            last_action = step.label
        last_cmd = step.cmd
        last_speed_base = step.speed
        step_start = time.time()
        while time.time() - step_start < step.duration_s:
            update_world_from_vision(world, vision)
            if observer:
                observer("frame", world, vision, None, None, None)
            success_ok, confidence = evaluate_gate_status(world, step)
            confidence_ok = success_ok and confidence >= SUCCESS_CONFIDENCE_MIN
            log_confidence(world, confidence, step)

            if allow_early_exit:
                success_met = success_tracker.update(confidence_ok)
                if success_met:
                    robot.stop()
                    print(format_headline(f"[SUCCESS] {step} criteria met 🎉", COLOR_GREEN))
                    print(format_headline(format_success_details(world, step), COLOR_WHITE))
                    return True, "success gate"

            active_speed = adjust_speed_for_find_brick(world, step, step.speed)
            analytics = telemetry_brick.compute_alignment_analytics(
                world,
                world.process_rules or {},
                world.learned_rules or {},
                step,
                duration_s=CONTROL_DT,
            )
            gate_speed = analytics.get("speed")
            speed_score = analytics.get("speed_score")
            if gate_speed:
                active_speed = gate_speed
            if speed_score is not None:
                world._last_action_line = format_control_action_line(
                    step.cmd,
                    active_speed,
                    f"replay {int(speed_score)}%",
                )
            active_speed = apply_pursuit_speed(active_speed)
            active_speed = apply_confidence_speed(active_speed, success_ok, confidence, world)
            send_robot_command(robot, world, step, step.cmd, active_speed, speed_score=analytics.get("speed_score"))
            evt = MotionEvent(
                cmd_to_motion_type(step.cmd),
                int(active_speed * 255),
                int(CONTROL_DT * 1000),
            )
            world.update_from_motion(evt)
            if observer:
                observer("action", world, vision, step.cmd, active_speed, "replay")
            time.sleep(CONTROL_DT)

    robot.stop()
    settle_deadline = time.time() + SUCCESS_SETTLE_S
    settle_tracker = SuccessGateTracker(success_frames_required(step))
    while time.time() < settle_deadline:
        update_world_from_vision(world, vision)
        if observer:
            observer("frame", world, vision, None, None, None)
        success_ok, confidence = evaluate_gate_status(world, step)
        confidence_ok = success_ok and confidence >= SUCCESS_CONFIDENCE_MIN
        log_confidence(world, confidence, step)
        success_met = settle_tracker.update(confidence_ok)
        if success_met:
            print(format_headline(f"[SUCCESS] {step} criteria met 🎉", COLOR_GREEN))
            print(format_headline(format_success_details(world, step), COLOR_WHITE))
            return True, "success gate"
        if last_cmd:
            active_speed = adjust_speed_for_find_brick(world, step, last_speed_base)
            analytics = telemetry_brick.compute_alignment_analytics(
                world,
                world.process_rules or {},
                world.learned_rules or {},
                step,
                duration_s=CONTROL_DT,
            )
            gate_speed = analytics.get("speed")
            speed_score = analytics.get("speed_score")
            if gate_speed:
                active_speed = gate_speed
            if speed_score is not None:
                world._last_action_line = format_control_action_line(
                    last_cmd,
                    active_speed,
                    f"replay {int(speed_score)}%",
                )
            active_speed = apply_pursuit_speed(active_speed)
            active_speed = apply_confidence_speed(active_speed, success_ok, confidence, world)
            send_robot_command(robot, world, step, last_cmd, active_speed, speed_score=analytics.get("speed_score"))
            evt = MotionEvent(
                cmd_to_motion_type(last_cmd),
                int(active_speed * 255),
                int(CONTROL_DT * 1000),
            )
            world.update_from_motion(evt)
            if observer:
                observer("action", world, vision, last_cmd, active_speed, "replay")
        time.sleep(CONTROL_DT)

    if True:
        tail_tracker = SuccessGateTracker(success_frames_required(step))
        while True:
            update_world_from_vision(world, vision)
            if observer:
                observer("frame", world, vision, None, None, None)
            success_ok, confidence = evaluate_gate_status(world, step)
            confidence_ok = success_ok and confidence >= SUCCESS_CONFIDENCE_MIN
            log_confidence(world, confidence, step)
            success_met = tail_tracker.update(confidence_ok)
            if success_met:
                if robot:
                    robot.stop()
                print(format_headline(f"[SUCCESS] {step} criteria met 🎉", COLOR_GREEN))
                print(format_headline(format_success_details(world, step), COLOR_WHITE))
                return True, "success gate"
            if last_cmd:
                active_speed = adjust_speed_for_find_brick(world, step, last_speed_base)
                analytics = telemetry_brick.compute_alignment_analytics(
                    world,
                    world.process_rules or {},
                    world.learned_rules or {},
                    step,
                    duration_s=CONTROL_DT,
                )
                gate_speed = analytics.get("speed")
                speed_score = analytics.get("speed_score")
                if gate_speed:
                    active_speed = gate_speed
                if speed_score is not None:
                    world._last_action_line = format_control_action_line(
                        last_cmd,
                        active_speed,
                        f"replay {int(speed_score)}%",
                    )
                active_speed = apply_pursuit_speed(active_speed)
                active_speed = apply_confidence_speed(active_speed, success_ok, confidence, world)
                send_robot_command(robot, world, step, last_cmd, active_speed, speed_score=analytics.get("speed_score"))
                evt = MotionEvent(
                    cmd_to_motion_type(last_cmd),
                    int(active_speed * 255),
                    int(CONTROL_DT * 1000),
                )
                world.update_from_motion(evt)
                if observer:
                    observer("action", world, vision, last_cmd, active_speed, "replay")
            time.sleep(CONTROL_DT)

    pause_after_fail(robot)
    return False, "success gate not reached"
