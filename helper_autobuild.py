#!/usr/bin/env python3
import argparse
import json
import collections
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from helper_demo_log_utils import extract_attempt_segments, load_demo_logs, normalize_objective_label
from helper_vision_aruco import ArucoBrickVision
from helper_robot_control import Robot
from telemetry_robot import MotionEvent, ObjectiveState, WorldModel
import telemetry_brick
import telemetry_robot as telemetry_robot_module
import telemetry_wall
import helper_learning

USE_LEARNED_POLICY = False
GLOBAL_POLICY = None


DEMO_DIR = Path(__file__).resolve().parent / "demos"
PROCESS_MODEL_FILE = Path(__file__).resolve().parent / "world_model_process.json"

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
    "pursuit_speed": 0.32,
    "success_confirmation_slow_speed": 0.0,
    "suspect_speed": 0.6,
    "max_objective_duration_s": 20.0,
    "failure_tighten_low_pct": 0.1,
    "failure_tighten_high_pct": 0.9,
    "start_gate_timeout_s": 8.0,
    "success_settle_s": 1.0,
    "success_tail_window_s": 0.5,
    "success_tail_count": 4,
    "max_phase_attempts": 1,
    "max_speed": 0.5,
    "smooth_speed": 0.32,
    "smooth_step_s": 1.0,
    "duration_scale": 3.0,
    "fail_pause_s": 3.0,
    "find_brick_slow_factor": 4.0,
    "min_align_speed": 0.2,
    "max_align_speed": 0.28,
    "micro_align_speed": 0.21,
    "micro_align_offset_mm": 10.0,
    "micro_align_angle_deg": 5.0,
    "visibility_lost_hold_s": 0.5,
    "learned_policy_confidence_threshold": 0.4,
    "success_frames_by_objective": {"FIND_BRICK": 3},
    "alignment_metrics": ["angle_abs", "xAxis_offset_abs", "dist"],
}

MM_METRICS = {
    "xAxis_offset_abs",
    "dist",
    "distance",
    "lift_height",
}
MIN_MM_TOL = 1.5

BRICK_SMOOTH_FRAMES = 4
BRICK_SMOOTH_SPLIT_MM = 20.0
BRICK_SMOOTH_SPLIT_DEG = 8.0
BRICK_SMOOTH_OUTLIER_MM = 12.0
BRICK_SMOOTH_OUTLIER_DEG = 6.0


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
    global PURSUIT_SPEED
    global SUCCESS_CONFIRMATION_SLOW_SPEED
    global SUSPECT_SPEED
    global MAX_OBJECTIVE_DURATION_S
    global FAILURE_TIGHTEN_LOW_PCT
    global FAILURE_TIGHTEN_HIGH_PCT
    global START_GATE_TIMEOUT_S
    global SUCCESS_SETTLE_S
    global SUCCESS_TAIL_WINDOW_S
    global SUCCESS_TAIL_COUNT
    global MAX_PHASE_ATTEMPTS
    global MAX_SPEED
    global SMOOTH_SPEED
    global SMOOTH_STEP_S
    global DURATION_SCALE
    global FAIL_PAUSE_S
    global FIND_BRICK_SLOW_FACTOR
    global MIN_ALIGN_SPEED
    global MAX_ALIGN_SPEED
    global MICRO_ALIGN_SPEED
    global MICRO_ALIGN_OFFSET_MM
    global MICRO_ALIGN_ANGLE_DEG
    global VISIBILITY_LOST_HOLD_S
    global LEARNED_POLICY_CONFIDENCE_THRESHOLD
    global SUCCESS_FRAMES_BY_OBJECTIVE
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
    PURSUIT_SPEED = _as_float(
        cfg.get("pursuit_speed"),
        DEFAULT_AUTOBUILD_CONFIG["pursuit_speed"],
    )
    SUCCESS_CONFIRMATION_SLOW_SPEED = _as_float(
        cfg.get("success_confirmation_slow_speed"),
        DEFAULT_AUTOBUILD_CONFIG["success_confirmation_slow_speed"],
    )
    SUSPECT_SPEED = _as_float(
        cfg.get("suspect_speed"),
        DEFAULT_AUTOBUILD_CONFIG["suspect_speed"],
    )
    MAX_OBJECTIVE_DURATION_S = _as_float(
        cfg.get("max_objective_duration_s"),
        DEFAULT_AUTOBUILD_CONFIG["max_objective_duration_s"],
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
    MAX_SPEED = _as_float(
        cfg.get("max_speed"),
        DEFAULT_AUTOBUILD_CONFIG["max_speed"],
    )
    SMOOTH_SPEED = _as_float(
        cfg.get("smooth_speed"),
        DEFAULT_AUTOBUILD_CONFIG["smooth_speed"],
    )
    SMOOTH_STEP_S = _as_float(
        cfg.get("smooth_step_s"),
        DEFAULT_AUTOBUILD_CONFIG["smooth_step_s"],
    )
    DURATION_SCALE = _as_float(
        cfg.get("duration_scale"),
        DEFAULT_AUTOBUILD_CONFIG["duration_scale"],
    )
    FAIL_PAUSE_S = _as_float(
        cfg.get("fail_pause_s"),
        DEFAULT_AUTOBUILD_CONFIG["fail_pause_s"],
    )
    FIND_BRICK_SLOW_FACTOR = _as_float(
        cfg.get("find_brick_slow_factor"),
        DEFAULT_AUTOBUILD_CONFIG["find_brick_slow_factor"],
    )
    MIN_ALIGN_SPEED = _as_float(
        cfg.get("min_align_speed"),
        DEFAULT_AUTOBUILD_CONFIG["min_align_speed"],
    )
    MAX_ALIGN_SPEED = _as_float(
        cfg.get("max_align_speed"),
        DEFAULT_AUTOBUILD_CONFIG["max_align_speed"],
    )
    MICRO_ALIGN_SPEED = _as_float(
        cfg.get("micro_align_speed"),
        DEFAULT_AUTOBUILD_CONFIG["micro_align_speed"],
    )
    MICRO_ALIGN_OFFSET_MM = _as_float(
        cfg.get("micro_align_offset_mm"),
        DEFAULT_AUTOBUILD_CONFIG["micro_align_offset_mm"],
    )
    MICRO_ALIGN_ANGLE_DEG = _as_float(
        cfg.get("micro_align_angle_deg"),
        DEFAULT_AUTOBUILD_CONFIG["micro_align_angle_deg"],
    )
    VISIBILITY_LOST_HOLD_S = _as_float(
        cfg.get("visibility_lost_hold_s"),
        DEFAULT_AUTOBUILD_CONFIG["visibility_lost_hold_s"],
    )
    LEARNED_POLICY_CONFIDENCE_THRESHOLD = _as_float(
        cfg.get("learned_policy_confidence_threshold"),
        DEFAULT_AUTOBUILD_CONFIG["learned_policy_confidence_threshold"],
    )
    frames_cfg = cfg.get("success_frames_by_objective")
    if not isinstance(frames_cfg, dict):
        frames_cfg = DEFAULT_AUTOBUILD_CONFIG["success_frames_by_objective"]
    SUCCESS_FRAMES_BY_OBJECTIVE = dict(frames_cfg)
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


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    pct = max(0.0, min(1.0, pct))
    idx = int(round(pct * (len(values) - 1)))
    return values[idx]


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


def derive_success_gate_scales(segments_by_obj, objective_rules=None):
    scales = {}
    for obj, segs in segments_by_obj.items():
        success_count = len(segs.get("SUCCESS", []))
        fail_count = len(segs.get("FAIL", []))
        if success_count <= 0 and fail_count <= 0:
            continue
        scales[obj] = failure_based_scale(success_count, fail_count)
    if isinstance(objective_rules, dict):
        for obj, rules in objective_rules.items():
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
            obj_key = normalize_objective_label(obj)
            if not obj_key:
                continue
            current = scales.get(obj_key, 1.0)
            scales[obj_key] = current * scale_val
    return scales


def success_frames_required(objective):
    obj_key = normalize_objective_label(objective)
    required = SUCCESS_FRAMES_BY_OBJECTIVE.get(obj_key, GATE_STABILITY_FRAMES)
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


def format_success_details(world, objective):
    obj_key = normalize_objective_label(objective)
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
    duration = f"{step.duration_s:.2f}"
    if target_visible is None:
        suffix = f"for {duration}s"
    else:
        if target_visible:
            stop_clause = "until brick becomes visible"
        else:
            stop_clause = "until brick is no longer visible"
        suffix = f"for {duration}s or {stop_clause}"
    reason = str(reason).strip() if reason else ""
    # Make reason gray
    reason_suffix = f" {COLOR_GRAY}({reason}){COLOR_RESET}" if reason else ""
    return f"[ACT] {action} at {power} power {suffix}{reason_suffix}"


def format_control_action_line(cmd, speed, reason=None):
    if not cmd:
        return f"[ACT] holding position"

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
    
    # Format Line 1
    if ":" in gap_info:
        # e.g. "angle:19.18deg" -> "close the 19.18deg gap"
        _, gap_val = gap_info.split(":", 1)
        line1 = f"[ACT] {level} {action} to close the {gap_val} gap."
    else:
        line1 = f"[ACT] {level} {action} ({reason_str})"
        
    # Format Line 2 (Progress)
    line2 = ""
    if delta_info and ":" in delta_info:
        # e.g. "closer:2.13deg" -> "🟢 The last act got us 2.13deg closer to perfect alignment."
        direction, delta_val = delta_info.split(":", 1)
        emoji = "🟢" if direction == "closer" else "🔴"
        word = "closer to" if direction == "closer" else "further from"
        line2 = f"\n{emoji} The last act got us {delta_val} {word} perfect alignment."
        
    return f"{line1}{line2}"


def objective_uses_alignment_control(objective, process_rules):
    obj_key = normalize_objective_label(objective)
    rules = (process_rules or {}).get(obj_key, {})
    controller = rules.get("controller")
    if controller == "replay":
        return False
    if controller == "align":
        return True
    success_gates = rules.get("success_gates") or {}
    return any(metric in success_gates for metric in ALIGNMENT_METRICS)


def derive_action_speeds(steps, fallback=SMOOTH_SPEED):
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

    def pick(value):
        if value is None:
            value = fallback
        return min(value, MAX_SPEED, PURSUIT_SPEED)

    turn = pick(turn)
    forward = pick(forward)
    backward = pick(backward)
    return {
        "turn": turn,
        "scan": turn,
        "forward": forward,
        "backward": backward,
    }


def alignment_command(world, objective, gate_bounds, speeds, preview=False):
    analytics = telemetry_brick.compute_alignment_analytics(
        world,
        world.process_rules or {},
        world.learned_rules or {},
        objective,
        duration_s=CONTROL_DT,
    )
    cmd = analytics.get("cmd")
    speed = analytics.get("speed") or 0.0
    reason = analytics.get("worst_metric") or "align"
    return cmd, speed, reason


def adjust_speed_for_find_brick(world, objective, speed):
    if speed is None or speed <= 0:
        return speed
    obj_key = normalize_objective_label(objective)
    if obj_key != "FIND_BRICK":
        return speed
    brick = world.brick or {}
    if not brick.get("visible"):
        return speed
    return speed / FIND_BRICK_SLOW_FACTOR


def success_visible_target(world, objective):
    objective_key = normalize_objective_label(objective)
    rules = world.process_rules.get(objective_key, {}) if world.process_rules else {}
    visible_gate = (rules.get("success_gates") or {}).get("visible", {})
    if "min" in visible_gate:
        return bool(visible_gate.get("min"))
    if "max" in visible_gate:
        return bool(visible_gate.get("max"))
    return None


def objective_duration_stats(process_rules, objective):
    objective_key = normalize_objective_label(objective)
    rules = (process_rules or {}).get(objective_key, {})
    stats = (rules.get("success_gates") or {}).get("duration_s")
    if isinstance(stats, dict):
        return stats
    return None


def objective_elapsed_s(world):
    start_time = getattr(world, "_objective_start_time", None)
    if start_time is None:
        return None
    return max(0.0, time.time() - start_time)


def objective_confidence(success_ok, world, objective):
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

    duration_conf = 1.0
    stats = objective_duration_stats(world.process_rules or {}, objective)
    if stats:
        target = stats.get("target")
        if target is not None and target > 0:
            elapsed = objective_elapsed_s(world)
            if elapsed is not None:
                duration_conf = max(0.0, min(1.0, elapsed / target))

    combined_conf = max(confirm_conf, duration_conf)
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
    capped_speed = speed
    if world is not None:
        confirm_progress = getattr(world, "_success_confirm_progress", None)
        if confirm_progress is not None:
            capped_speed = min(capped_speed, SUCCESS_CONFIRMATION_SLOW_SPEED)
    if confidence_suspect(success_ok, confidence):
        capped_speed = min(capped_speed, SUSPECT_SPEED)
    return capped_speed


def apply_pursuit_speed(speed):
    if speed is None:
        return speed
    return min(speed, PURSUIT_SPEED)


def log_confidence(world, confidence, objective):
    if confidence is None:
        return
    confirm_progress = getattr(world, "_success_confirm_progress", None)
    if confirm_progress is not None and confirm_progress < 1.0:
        pct = int(round(confidence * 100))
        print(format_headline(f"[CONFIRM] {objective} {pct}% confidence", COLOR_WHITE))
        return
    if confirm_progress is not None and not getattr(world, "_success_confirm_logged", False):
        pct = int(round(confidence * 100))
        print(format_headline(f"[CONFIRM] {objective} {pct}% confidence", COLOR_WHITE))
        world._success_confirm_logged = True
        return
    if confidence < CONFIDENCE_LOG_MIN or confidence >= SUCCESS_CONFIDENCE_MIN:
        return
    pct = int(round(confidence * 100))
    print(format_headline(f"[CONF] {objective} {pct}% confidence", COLOR_WHITE))


def pause_after_fail(robot):
    if robot:
        robot.stop()
    time.sleep(FAIL_PAUSE_S)


def load_process_model(path=PROCESS_MODEL_FILE):
    if not path.exists():
        return {"objectives": {}}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"objectives": {}}


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


def objective_metrics_map():
    metrics = {}
    for obj, items in telemetry_brick.METRICS_BY_OBJECTIVE.items():
        metrics.setdefault(obj, []).extend([m for m in items if m not in metrics.get(obj, [])])
    for obj, items in telemetry_robot_module.METRICS_BY_OBJECTIVE.items():
        metrics.setdefault(obj, []).extend([m for m in items if m not in metrics.get(obj, [])])
    return metrics


def success_gate_metrics_for_objective(metrics, objective, objective_rules=None):
    obj_key = normalize_objective_label(objective)
    rules = (objective_rules or {}).get(obj_key, {})
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


def metric_direction(metric, objective):
    if metric in telemetry_brick.METRIC_DIRECTIONS:
        return telemetry_brick.metric_direction_for_objective(metric, objective)
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
            obj = normalize_objective_label(seg.get("objective"))
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
    return start_gates


def derive_success_gates(success_segments, scale_by_objective=None, objective_rules=None):
    metrics_by_obj = objective_metrics_map()
    scale_by_objective = scale_by_objective or {}

    success_gates = {}
    for obj, segs in success_segments.items():
        metrics = success_gate_metrics_for_objective(
            metrics_by_obj.get(obj, []),
            obj,
            objective_rules,
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

            p10 = percentile(values, 0.1)
            p50 = percentile(values, 0.5)
            p90 = percentile(values, 0.9)
            if p50 is None:
                continue
            tol = None
            if p10 is not None and p90 is not None:
                tol = max(abs(p50 - p10), abs(p90 - p50))
            scale = scale_by_objective.get(obj, 1.0)
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
        p10 = percentile(values, 0.1)
        p50 = percentile(values, 0.5)
        p90 = percentile(values, 0.9)
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


def refine_success_gates_with_failures(success_gates, fail_segments, objective_rules=None):
    if not success_gates or not fail_segments:
        return success_gates
    metrics_by_obj = objective_metrics_map()
    for obj, gates in success_gates.items():
        if not isinstance(gates, dict):
            continue
        # Tighten success tolerances when failure samples overlap the current success window.
        metrics = success_gate_metrics_for_objective(
            metrics_by_obj.get(obj, []),
            obj,
            objective_rules,
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
                    failure_floor = percentile(values, FAILURE_TIGHTEN_LOW_PCT)
                    if failure_floor is None:
                        continue
                    success_max = target + tol
                    if failure_floor <= target or failure_floor >= success_max:
                        continue
                    stats["tol"] = round_value(failure_floor - target)
                else:
                    failure_ceiling = percentile(values, FAILURE_TIGHTEN_HIGH_PCT)
                    if failure_ceiling is None:
                        continue
                    success_min = target - tol
                    if failure_ceiling >= target or failure_ceiling <= success_min:
                        continue
                    stats["tol"] = round_value(target - failure_ceiling)
                continue

            if direction == "low":
                failure_floor = percentile(values, FAILURE_TIGHTEN_LOW_PCT)
                if failure_floor is None:
                    continue
                max_val = stats.get("max")
                if max_val is None or failure_floor >= max_val:
                    continue
                stats["max"] = round_value(failure_floor)
            else:
                failure_ceiling = percentile(values, FAILURE_TIGHTEN_HIGH_PCT)
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
                entry["duration"] += step.duration_s
                entry["speed_sum"] += step.speed * step.duration_s
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
                "duration_s": round_value(duration_avg),
                "speed": round_value(speed_avg),
            }

        dominant_cmd, stats = max(
            cmd_stats.items(),
            key=lambda item: item[1].get("duration_s") or 0.0,
        )
        learnings[obj] = {"cmd": dominant_cmd, **stats}
    return learnings


def update_process_model_from_demos(logs, path=PROCESS_MODEL_FILE):
    model = load_process_model(path)
    objectives = model.get("objectives")
    if not isinstance(objectives, dict):
        objectives = {}
        model["objectives"] = objectives

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
    wall_objective_rules = telemetry_wall.load_wall_objective_rules()
    objective_rules = {}
    if isinstance(wall_objective_rules, dict):
        objective_rules.update(wall_objective_rules)
    if isinstance(objectives, dict):
        objective_rules.update(objectives)
    success_gate_scales = derive_success_gate_scales(segments_by_obj, objective_rules)

    # Train Policy if requested
    if USE_LEARNED_POLICY:
        global GLOBAL_POLICY
        print(format_headline("[LEARNING REPLAY] Training Policy from Demos...", COLOR_GREEN))
        GLOBAL_POLICY = helper_learning.BehavioralCloningPolicy()
        GLOBAL_POLICY.train(segments_by_obj)

    success_gates = derive_success_gates(
        success_segments,
        success_gate_scales,
        objective_rules,
    )
    success_durations = derive_success_durations(success_segments)
    success_gates = refine_success_gates_with_failures(
        success_gates,
        fail_segments,
        objective_rules,
    )
    for obj, stats in success_durations.items():
        if obj not in success_gates:
            continue
        if isinstance(success_gates.get(obj), dict):
            success_gates[obj]["duration_s"] = stats

    movement_learnings = derive_movement_learnings(segments_by_obj)

    all_objectives = set(objectives.keys())
    all_objectives.update(start_gates.keys())
    all_objectives.update(success_gates.keys())
    all_objectives.update(movement_learnings.keys())
    all_objectives.update(attempt_types.keys())

    for obj in all_objectives:
        cfg = objectives.setdefault(obj, {})

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
            duration_stats = cfg.get("success_gates", {}).get("duration_s")
            cfg["success_gates"] = {"visible": {"min": False}}
            if duration_stats:
                cfg["success_gates"]["duration_s"] = duration_stats

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
    for name in objectives.keys():
        ordered[name] = objectives[name]
    for name in sorted(all_objectives):
        if name not in ordered:
            ordered[name] = objectives.get(name, {})
    model["objectives"] = ordered

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


def _average_brick_frames(frames):
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


def _filtered_brick_frame_average(frames):
    if len(frames) < BRICK_SMOOTH_FRAMES:
        return None, False
    first_two = frames[:2]
    last_two = frames[2:]
    avg_first = _average_brick_frames(first_two)
    avg_last = _average_brick_frames(last_two)
    if (
        abs(avg_first["dist"] - avg_last["dist"]) > BRICK_SMOOTH_SPLIT_MM
        or abs(avg_first["offset_x"] - avg_last["offset_x"]) > BRICK_SMOOTH_SPLIT_MM
        or abs(avg_first["angle"] - avg_last["angle"]) > BRICK_SMOOTH_SPLIT_DEG
    ):
        return None, True

    dist_vals = [f["dist"] for f in frames]
    offset_vals = [f["offset_x"] for f in frames]
    angle_vals = [f["angle"] for f in frames]
    med_dist = percentile(dist_vals, 0.5)
    med_offset = percentile(offset_vals, 0.5)
    med_angle = percentile(angle_vals, 0.5)

    keep = []
    for frame in frames:
        if not frame["found"]:
            continue
        if (
            abs(frame["dist"] - med_dist) > BRICK_SMOOTH_OUTLIER_MM
            or abs(frame["offset_x"] - med_offset) > BRICK_SMOOTH_OUTLIER_MM
            or abs(frame["angle"] - med_angle) > BRICK_SMOOTH_OUTLIER_DEG
        ):
            continue
        keep.append(frame)

    if len(keep) < 3:
        return None, True
    return _average_brick_frames(keep), True


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


def smooth_motion_steps(
    steps,
    speed=SMOOTH_SPEED,
    step_s=SMOOTH_STEP_S,
    max_speed=PURSUIT_SPEED,
    duration_scale=DURATION_SCALE,
):
    if not steps:
        return []
    smooth = []
    speed = min(speed, max_speed)
    for step in steps:
        total_duration = step.duration_s * duration_scale
        chunks = max(1, int(math.ceil(total_duration / step_s)))
        for idx in range(chunks):
            duration = min(step_s, total_duration - (idx * step_s))
            if duration <= 0:
                continue
            smooth.append(MotionStep(step.cmd, speed, duration, step.label))
    return smooth


def select_demo_segment(segments_by_obj, objective, nominal_only):
    obj_key = normalize_objective_label(objective)
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
    }
    buffer = getattr(world, "_brick_frame_buffer", None)
    if buffer is None:
        buffer = []
        setattr(world, "_brick_frame_buffer", buffer)
    buffer.append(frame)
    if len(buffer) > BRICK_SMOOTH_FRAMES:
        buffer.pop(0)
    avg, should_reset = _filtered_brick_frame_average(buffer)
    if should_reset:
        buffer.clear()
    if avg is not None:
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
        if log:
            print(format_headline(format_brick_state_line(world), COLOR_WHITE))


def evaluate_gate_status(world, objective):
    process_rules = world.process_rules or {}
    telemetry_rules = {}
    visibility_grace_s = None
    if success_visible_target(world, objective) is False:
        visibility_grace_s = SUCCESS_VISIBLE_FALSE_GRACE_S
    brick_success = telemetry_brick.evaluate_success_gates(
        world,
        objective,
        telemetry_rules,
        process_rules,
        visibility_grace_s=visibility_grace_s,
    )
    wall_success = telemetry_wall.evaluate_success_gates(world, objective, world.wall_envelope)
    robot_success = telemetry_robot_module.evaluate_success_gates(world, objective, telemetry_rules, process_rules)

    success_ok = brick_success.ok and wall_success.ok and robot_success.ok
    confidence = objective_confidence(success_ok, world, objective)
    return success_ok, confidence


def wait_for_start_gates(
    world,
    vision,
    objective,
    robot=None,
    cmd=None,
    speed=None,
    log=True,
    observer=None,
):
    start_time = time.time()
    stable = 0
    success_tracker = SuccessGateTracker(success_frames_required(objective))
    success_seen = False
    if robot:
        robot.stop()
    while time.time() - start_time < START_GATE_TIMEOUT_S:
        update_world_from_vision(world, vision, log=log)
        if observer:
            observer("frame", world, vision, None, None, None)
        success_ok, confidence = evaluate_gate_status(world, objective)
        if log:
            log_confidence(world, confidence, objective)
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

        brick_check = telemetry_brick.evaluate_start_gates(world, objective, {}, world.process_rules)
        wall_check = telemetry_wall.evaluate_start_gates(world, objective, world.wall_envelope)
        robot_check = telemetry_robot_module.evaluate_start_gates(world, objective, {}, world.process_rules)
        if brick_check.ok and wall_check.ok and robot_check.ok:
            stable += 1
            if stable >= GATE_STABILITY_FRAMES:
                if log:
                    print(format_headline(f"[START] {objective} start gates met", COLOR_GREEN))
                return "start"
        else:
            stable = 0
            if robot and cmd and speed:
                robot.send_command(cmd, speed)
                evt = MotionEvent(
                    cmd_to_motion_type(cmd),
                    int(speed * 255),
                    int(CONTROL_DT * 1000),
                )
                world.update_from_motion(evt)
        time.sleep(CONTROL_DT)
    return "timeout"


def run_alignment_segment(
    segment,
    objective,
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
        objective,
    )
    scan_cmd = telemetry_robot_module.resolve_scan_direction(world.process_rules, objective)
    start_status = wait_for_start_gates(
        world,
        vision,
        objective,
        robot=robot,
        cmd=scan_cmd,
        speed=action_speeds["scan"],
        log=not align_silent,
        observer=observer,
    )
    if start_status == "success":
        if robot:
            robot.stop()
        if not align_silent:
            print(format_headline(f"[SUCCESS] {objective} criteria met", COLOR_GREEN))
            print(format_headline(format_success_details(world, objective), COLOR_WHITE))
        return True, "success gate"
    if start_status != "start":
        pause_after_fail(robot)
        return False, "start gates not met"

    objective_deadline = time.time() + MAX_OBJECTIVE_DURATION_S
    success_tracker = SuccessGateTracker(success_frames_required(objective))
    last_cmd = None
    last_reason = None
    last_speed = None

    while time.time() < objective_deadline:
        update_world_from_vision(world, vision, log=not align_silent)
        if observer:
            observer("frame", world, vision, None, None, None)
        success_ok, confidence = evaluate_gate_status(world, objective)
        if not align_silent:
            log_confidence(world, confidence, objective)
        if success_ok:
            if robot:
                robot.stop()
            if not align_silent:
                print(format_headline(f"[SUCCESS] {objective} criteria met", COLOR_GREEN))
                print(format_headline(format_success_details(world, objective), COLOR_WHITE))
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
            objective,
            duration_s=CONTROL_DT,
        )
        cmd = analytics.get("cmd")
        speed = analytics.get("speed") or 0.0
        cmd_reason = analytics.get("worst_metric") or "align"
        speed = apply_pursuit_speed(speed)
        speed = apply_confidence_speed(speed, success_ok, confidence, world)
        if objective == "ALIGN_BRICK":
            speed = telemetry_brick.ALIGN_FIXED_SPEED
        if cmd != last_cmd or cmd_reason != last_reason or speed != last_speed:
            if not align_silent:
                brick_success = telemetry_brick.evaluate_success_gates(
                    world,
                    objective,
                    {},
                    world.process_rules or {},
                )
                wall_success = telemetry_wall.evaluate_success_gates(world, objective, world.wall_envelope)
                robot_success = telemetry_robot_module.evaluate_success_gates(world, objective, {}, world.process_rules or {})
                reasons = []
                reasons.extend(brick_success.reasons or [])
                reasons.extend(wall_success.reasons or [])
                reasons.extend(robot_success.reasons or [])
                if reasons:
                    print(format_headline(f"[ALIGN] Not within success gates: {', '.join(reasons)}", COLOR_WHITE))
                else:
                    print(format_headline("[ALIGN] Not within success gates: no reasons reported", COLOR_WHITE))
                print(format_headline(format_control_action_line(cmd, speed, cmd_reason), COLOR_WHITE))
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
            if confirm_callback:
                if not confirm_callback(world, vision):
                    return False, "confirm cancelled"
            robot.send_command(cmd, speed)
            evt = MotionEvent(
                cmd_to_motion_type(cmd),
                int(speed * 255),
                int(CONTROL_DT * 1000),
            )
            world.update_from_motion(evt)
            if confirm_callback and robot:
                robot.stop()
        else:
            if robot:
                robot.stop()
        if observer:
            observer("action", world, vision, cmd, speed, cmd_reason)
        time.sleep(CONTROL_DT)

    if robot:
        robot.stop()
    settle_deadline = min(objective_deadline, time.time() + SUCCESS_SETTLE_S)
    settle_tracker = SuccessGateTracker(success_frames_required(objective))
    while time.time() < settle_deadline:
        update_world_from_vision(world, vision, log=not align_silent)
        if observer:
            observer("frame", world, vision, None, None, None)
        success_ok, confidence = evaluate_gate_status(world, objective)
        if not align_silent:
            log_confidence(world, confidence, objective)
        if success_ok:
            if not align_silent:
                print(format_headline(f"[SUCCESS] {objective} criteria met", COLOR_GREEN))
                print(format_headline(format_success_details(world, objective), COLOR_WHITE))
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
            objective,
            duration_s=CONTROL_DT,
        )
        cmd = analytics.get("cmd")
        speed = analytics.get("speed") or 0.0
        cmd_reason = analytics.get("worst_metric") or "align"
        speed = apply_pursuit_speed(speed)
        speed = apply_confidence_speed(speed, success_ok, confidence, world)
        if objective == "ALIGN_BRICK":
            speed = telemetry_brick.ALIGN_FIXED_SPEED
        if cmd != last_cmd or cmd_reason != last_reason or speed != last_speed:
            if not align_silent:
                brick_success = telemetry_brick.evaluate_success_gates(
                    world,
                    objective,
                    {},
                    world.process_rules or {},
                )
                wall_success = telemetry_wall.evaluate_success_gates(world, objective, world.wall_envelope)
                robot_success = telemetry_robot_module.evaluate_success_gates(world, objective, {}, world.process_rules or {})
                reasons = []
                reasons.extend(brick_success.reasons or [])
                reasons.extend(wall_success.reasons or [])
                reasons.extend(robot_success.reasons or [])
                if reasons:
                    print(format_headline(f"[ALIGN] Not within success gates: {', '.join(reasons)}", COLOR_WHITE))
                else:
                    print(format_headline("[ALIGN] Not within success gates: no reasons reported", COLOR_WHITE))
                print(format_headline(format_control_action_line(cmd, speed, cmd_reason), COLOR_WHITE))
            last_cmd = cmd
            last_reason = cmd_reason
            last_speed = speed

        if cmd:
            if confirm_callback:
                if not confirm_callback(world, vision):
                    return False, "confirm cancelled"
            robot.send_command(cmd, speed)
            evt = MotionEvent(
                cmd_to_motion_type(cmd),
                int(speed * 255),
                int(CONTROL_DT * 1000),
            )
            world.update_from_motion(evt)
            if confirm_callback and robot:
                robot.stop()
        time.sleep(CONTROL_DT)

    pause_after_fail(robot)
    return False, "success gate not reached"


def replay_segment(
    segment,
    objective,
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
    if objective_uses_alignment_control(objective, world.process_rules):
        return run_alignment_segment(
            segment,
            objective,
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
    target_visible = success_visible_target(world, objective)
    start_status = wait_for_start_gates(
        world,
        vision,
        objective,
        robot=robot,
        cmd=default_step.cmd,
        speed=default_step.speed,
    )
    if start_status == "success":
        if robot:
            robot.stop()
        print(format_headline(f"[SUCCESS] {objective} criteria met 🎉", COLOR_GREEN))
        print(format_headline(format_success_details(world, objective), COLOR_WHITE))
        time.sleep(3.0)
        return True, "success gate"
    if start_status != "start":
        pause_after_fail(robot)
        return False, "start gates not met"

    allow_early_exit = True

    objective_deadline = time.time() + MAX_OBJECTIVE_DURATION_S
    success_tracker = SuccessGateTracker(success_frames_required(objective))
    last_action = None
    last_cmd = default_step.cmd
    last_speed_base = default_step.speed

    for step in steps:
        if step.label != last_action:
            update_world_from_vision(world, vision)
            print(
                format_headline(
                    format_action_line(step, target_visible, reason="replaying demo action"),
                    COLOR_WHITE,
                )
            )
            last_action = step.label
        last_cmd = step.cmd
        last_speed_base = step.speed
        step_start = time.time()
        while time.time() - step_start < step.duration_s:
            if time.time() >= objective_deadline:
                break
            update_world_from_vision(world, vision)
            success_ok, confidence = evaluate_gate_status(world, objective)
            confidence_ok = success_ok and confidence >= SUCCESS_CONFIDENCE_MIN
            log_confidence(world, confidence, objective)

            if allow_early_exit:
                success_met = success_tracker.update(confidence_ok)
                if success_met:
                    robot.stop()
                    print(format_headline(f"[SUCCESS] {objective} criteria met 🎉", COLOR_GREEN))
                    print(format_headline(format_success_details(world, objective), COLOR_WHITE))
                    return True, "success gate"

            active_speed = adjust_speed_for_find_brick(world, objective, step.speed)
            active_speed = apply_pursuit_speed(active_speed)
            active_speed = apply_confidence_speed(active_speed, success_ok, confidence, world)
            robot.send_command(step.cmd, active_speed)
            evt = MotionEvent(
                cmd_to_motion_type(step.cmd),
                int(active_speed * 255),
                int(CONTROL_DT * 1000),
            )
            world.update_from_motion(evt)
            time.sleep(CONTROL_DT)
        if time.time() >= objective_deadline:
            break

    robot.stop()
    settle_deadline = min(objective_deadline, time.time() + SUCCESS_SETTLE_S)
    settle_tracker = SuccessGateTracker(success_frames_required(objective))
    while time.time() < settle_deadline:
        update_world_from_vision(world, vision)
        success_ok, confidence = evaluate_gate_status(world, objective)
        confidence_ok = success_ok and confidence >= SUCCESS_CONFIDENCE_MIN
        log_confidence(world, confidence, objective)
        success_met = settle_tracker.update(confidence_ok)
        if success_met:
            print(format_headline(f"[SUCCESS] {objective} criteria met 🎉", COLOR_GREEN))
            print(format_headline(format_success_details(world, objective), COLOR_WHITE))
            return True, "success gate"
        if last_cmd:
            active_speed = adjust_speed_for_find_brick(world, objective, last_speed_base)
            active_speed = apply_pursuit_speed(active_speed)
            active_speed = apply_confidence_speed(active_speed, success_ok, confidence, world)
            robot.send_command(last_cmd, active_speed)
            evt = MotionEvent(
                cmd_to_motion_type(last_cmd),
                int(active_speed * 255),
                int(CONTROL_DT * 1000),
            )
            world.update_from_motion(evt)
        time.sleep(CONTROL_DT)

    if time.time() < objective_deadline:
        tail_tracker = SuccessGateTracker(success_frames_required(objective))
        while time.time() < objective_deadline:
            update_world_from_vision(world, vision)
            success_ok, confidence = evaluate_gate_status(world, objective)
            confidence_ok = success_ok and confidence >= SUCCESS_CONFIDENCE_MIN
            log_confidence(world, confidence, objective)
            success_met = tail_tracker.update(confidence_ok)
            if success_met:
                if robot:
                    robot.stop()
                print(format_headline(f"[SUCCESS] {objective} criteria met 🎉", COLOR_GREEN))
                print(format_headline(format_success_details(world, objective), COLOR_WHITE))
                return True, "success gate"
            if last_cmd:
                active_speed = adjust_speed_for_find_brick(world, objective, last_speed_base)
                active_speed = apply_pursuit_speed(active_speed)
                active_speed = apply_confidence_speed(active_speed, success_ok, confidence, world)
                robot.send_command(last_cmd, active_speed)
                evt = MotionEvent(
                    cmd_to_motion_type(last_cmd),
                    int(active_speed * 255),
                    int(CONTROL_DT * 1000),
                )
                world.update_from_motion(evt)
            time.sleep(CONTROL_DT)

    pause_after_fail(robot)
    return False, "success gate not reached"


def run_autobuild(session_name=None):
    logs = load_demo_logs(DEMO_DIR, session_name)
    update_process_model_from_demos(logs, PROCESS_MODEL_FILE)
    refresh_autobuild_config(PROCESS_MODEL_FILE)
    if not logs:
        print("[AUTO] No demo logs found after update.")
        return

    segments_by_obj, _ = collect_segments(logs)
    model = load_process_model(PROCESS_MODEL_FILE)
    objectives = list((model.get("objectives") or {}).keys())
    if not objectives:
        print("[AUTO] No objectives defined in process model.")
        return

    robot = Robot()
    vision = ArucoBrickVision(debug=False)
    world = WorldModel()

    try:
        for obj_name in objectives:
            normalized = normalize_objective_label(obj_name)
            if normalized in ObjectiveState.__members__:
                world.objective_state = ObjectiveState[normalized]
            else:
                world.objective_state = ObjectiveState.FIND_BRICK

            cfg = world.process_rules.get(normalized, {}) if world.process_rules else {}
            nominal_only = bool(cfg.get("nominalDemosOnly"))
            segment, seg_type = select_demo_segment(segments_by_obj, normalized, nominal_only)
            if not segment:
                print(format_headline(f"[FAIL] {normalized}: no demo segment found", COLOR_RED))
                return

            attempts = 0
            last_reason = None
            while attempts < MAX_PHASE_ATTEMPTS:
                header = (
                    f"Attempting {normalized} "
                    f"(attempt {attempts + 1}/{MAX_PHASE_ATTEMPTS}; demo {seg_type})"
                )
                print(format_headline(header, COLOR_GREEN))
                start_desc, success_desc = format_gate_lines(cfg)
                print(f"  start gates: {start_desc}")
                print(f"  success gates: {success_desc}")
                ok, reason = replay_segment(segment, normalized, robot, vision, world)
                if ok:
                    break
                attempts += 1
                last_reason = reason or "unknown"
                print(format_headline(f"[RETRY] {normalized} ({last_reason})", COLOR_RED))
            if attempts >= MAX_PHASE_ATTEMPTS:
                reason_suffix = f" ({last_reason})" if last_reason else ""
                print(
                    format_headline(
                        f"[FAIL] {normalized} failed after {attempts} attempts{reason_suffix}",
                        COLOR_RED,
                    )
                )
                return

        print("[JOB] SUCCESS")
    finally:
        robot.stop()
        vision.close()


def main():
    parser = argparse.ArgumentParser(description="Robot Leia Autobuild")
    parser.add_argument("--session", help="demo session file or folder", default=None)
    parser.add_argument("--learn", help="enable learning from demonstration policy", action="store_true")
    args = parser.parse_args()

    global USE_LEARNED_POLICY
    if args.learn:
        USE_LEARNED_POLICY = True

    run_autobuild(session_name=args.session)
    print("\n" * 5, end="")


if __name__ == "__main__":
    main()
