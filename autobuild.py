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

CONTROL_HZ = 20.0
CONTROL_DT = 1.0 / CONTROL_HZ
GATE_STABILITY_FRAMES = 5
SUCCESS_CONSECUTIVE_FRAMES = 3
SUCCESS_MAJORITY_WINDOW = 5
SUCCESS_MAJORITY_REQUIRED = 3
SUCCESS_VISIBLE_FALSE_GRACE_S = 0.0
SUCCESS_CONFIDENCE_MIN = 0.85
CONFIDENCE_LOG_MIN = 0.5
SUCCESS_CONFIRMATION_S = 0.35
SUCCESS_CONFIRMATION_FRAMES = 4
SUCCESS_CONFIRMATION_START_CONFIDENCE = 0.5
PURSUIT_SPEED = 0.32
SUCCESS_CONFIRMATION_SLOW_SPEED = 0.0
SUSPECT_SPEED = 0.6
MAX_OBJECTIVE_DURATION_S = 20.0
FAILURE_TIGHTEN_LOW_PCT = 0.1
FAILURE_TIGHTEN_HIGH_PCT = 0.9
START_GATE_TIMEOUT_S = 8.0
SUCCESS_SETTLE_S = 1.0
SUCCESS_TAIL_WINDOW_S = 0.5
MAX_PHASE_ATTEMPTS = 1
MAX_SPEED = 0.5
SMOOTH_SPEED = PURSUIT_SPEED
SMOOTH_STEP_S = 1.0
DURATION_SCALE = 3.0
FAIL_PAUSE_S = 3.0
FIND_BRICK_SLOW_FACTOR = 4.0
MIN_ALIGN_SPEED = 0.2
MAX_ALIGN_SPEED = 0.28
MICRO_ALIGN_SPEED = 0.21
MICRO_ALIGN_OFFSET_MM = 10.0
MICRO_ALIGN_ANGLE_DEG = 5.0
VISIBILITY_LOST_HOLD_S = 0.5
LEARNED_POLICY_CONFIDENCE_THRESHOLD = 0.4

SUCCESS_FRAMES_BY_OBJECTIVE = {
    "FIND_BRICK": 3,
}

ALIGNMENT_METRICS = {"angle_abs", "offset_abs", "dist"}


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


def select_tail_states(states, window_s=SUCCESS_TAIL_WINDOW_S):
    if not states:
        return []
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
        elif metric == "offset_abs":
            actual = abs(brick.get("offset_x", 0.0))
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
        offset = brick.get("offset_x")
        conf = brick.get("confidence")
        above = brick.get("brickAbove")
        below = brick.get("brickBelow")
        if dist is not None:
            parts.append(f"dist={dist:.1f}mm")
        if angle is not None:
            parts.append(f"angle={angle:.2f}deg")
        if offset is not None:
            parts.append(f"offset_x={offset:.2f}mm")
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
    action = ACTION_CMD_DESC.get(cmd, "moving") if cmd else "holding position"
    reason = str(reason).strip() if reason else ""
    reason_suffix = f" {COLOR_GRAY}({reason}){COLOR_RESET}" if reason else ""
    if cmd:
        return f"[ACT] {action} at {speed:.2f} power{reason_suffix}"
    return f"[ACT] {action}{reason_suffix}"


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


def recent_turn_preference(world, max_age_s):
    last_seen = getattr(world, "last_visible_time", None)
    if last_seen is None or (time.time() - last_seen) > max_age_s:
        return None
    angle = getattr(world, "last_seen_angle", None)
    if angle is not None and abs(angle) > 1e-6:
        return "l" if angle > 0 else "r"
    offset = getattr(world, "last_seen_offset_x", None)
    if offset is not None and abs(offset) > 1e-6:
        return "l" if offset > 0 else "r"
    return None


def alignment_command(world, objective, gate_bounds, speeds):
    # Backtrack Queue Management
    backtrack_queue = getattr(world, "backtrack_queue", None)
    if backtrack_queue is None:
        backtrack_queue = collections.deque()
        setattr(world, "backtrack_queue", backtrack_queue)

    brick = world.brick or {}
    if not brick.get("visible"):
        # Hold still for a moment to confirm loss
        last_seen = getattr(world, "last_visible_time", None)
        if last_seen is not None and (time.time() - last_seen) < VISIBILITY_LOST_HOLD_S:
            return None, 0.0, "waiting for brick visibility"

        # 1. If we have a queue, execute it
        if backtrack_queue:
            cmd, speed = backtrack_queue.popleft()
            return cmd, speed, f"backtracking: replaying history ({len(backtrack_queue)} steps remaining)"

        # 2. If no queue, generate one from history (Last 5 seconds)
        # Only do this ONCE when we first realized we're lost and strictly need to recover
        # To avoid constant re-generation, check if we just finished a queue? 
        # Actually, if the queue is empty, we check history. If history is empty, we default to scan.
        # But we need to be careful not to infinite loop. 
        # Let's say we only generate if we haven't backtracked recently?
        # Or simpler: if we are lost, we generate the queue once.
        # We need a flag "actions_inverted".
        
        # Heuristic: If we moved > 20mm forward recently, trigger the rewind.
        if world.get_recent_net_forward_mm(window_s=5.0) > 10.0:  # Lower threshold for sensitivity
             now = time.time()
             # Get events from last 5s
             # History is oldest->newest. We want newest->oldest (reversed)
             for evt in reversed(world.action_history):
                 if (now - evt.timestamp) > 5.0:
                     break
                 
                 # Invert Logic
                 inv_cmd = None
                 if evt.action_type == "forward": inv_cmd = "b"
                 elif evt.action_type == "backward": inv_cmd = "f"
                 elif evt.action_type == "left_turn": inv_cmd = "r"
                 elif evt.action_type == "right_turn": inv_cmd = "l"
                 
                 if inv_cmd:
                     # Speed estimation: power / 255.0
                     speed = evt.power / 255.0
                     # Add to queue
                     backtrack_queue.append((inv_cmd, speed))
             
             if backtrack_queue:
                 cmd, speed = backtrack_queue.popleft()
                 return cmd, speed, f"backtracking: starting replay of {len(backtrack_queue)+1} steps"

        scan_cmd = recent_turn_preference(world, telemetry_brick.VISIBILITY_LOST_GRACE_S)
        if scan_cmd is None:
            scan_cmd = telemetry_robot_module.resolve_scan_direction(world.process_rules, objective)
        return scan_cmd, speeds["scan"], "scanning for brick visibility"
    
    # If visible, clear any stale backtrack queue
    if backtrack_queue:
        backtrack_queue.clear()

    # Calculate ratios to find the most egregious error
    offset_max = (gate_bounds.get("offset_abs") or {}).get("max")
    offset_x = brick.get("offset_x") or 0.0
    offset_ratio = 0.0
    if offset_max is not None:
        if offset_max > 0:
            offset_ratio = abs(offset_x) / offset_max
        elif abs(offset_x) > 0:
            offset_ratio = float('inf')

    angle_max = (gate_bounds.get("angle_abs") or {}).get("max")
    angle = brick.get("angle") or 0.0
    angle_ratio = 0.0
    if angle_max is not None:
        if angle_max > 0:
            angle_ratio = abs(angle) / angle_max
        elif abs(angle) > 0:
            angle_ratio = float('inf')

    # If both are valid (ratio <= 1.0), check distance
    if offset_ratio <= 1.0 and angle_ratio <= 1.0:
        dist = brick.get("dist")
        dist_bounds = gate_bounds.get("dist") or {}
        dist_min = dist_bounds.get("min")
        dist_max = dist_bounds.get("max")
        
        # Emoji Logic for Distance
        last_dist = getattr(world, "last_align_dist", None)
        dist_emoji = ""
        if last_dist is not None and dist is not None:
             # For distance, "progress" depends on context, but generally closer to target is better?
             # Actually, simpler: did the absolute error decrease?
             # Error could be (dist - dist_max) or (dist_min - dist). 
             # Let's just track raw distance change relative to action.
             # If we moved 'f' (closer), dist should decrease.
             pass

        if dist is not None:
            if dist_max is not None and dist > dist_max:
                # We need to move forward (closer) -> Distance should decrease
                if last_dist is not None:
                     if dist < last_dist - 0.5: dist_emoji = " 🟢"
                     elif dist > last_dist + 0.5: dist_emoji = " 🔴"
                setattr(world, "last_align_dist", dist)
                return "f", speeds["forward"], f"closing distance long by {abs(dist - dist_max):.1f}mm{dist_emoji}"
            
            if dist_min is not None and dist < dist_min:
                # We need to move backward (further) -> Distance should increase
                if last_dist is not None:
                     if dist > last_dist + 0.5: dist_emoji = " 🟢"
                     elif dist < last_dist - 0.5: dist_emoji = " 🔴"
                setattr(world, "last_align_dist", dist)
                return "b", speeds["backward"], f"closing distance short by {abs(dist_min - dist):.1f}mm{dist_emoji}"
        
        setattr(world, "last_align_dist", dist)
        return None, 0.0, "within success gates"

    # Policy-Based Learning Check
    policy_reason = None
    if USE_LEARNED_POLICY and GLOBAL_POLICY:
        l_cmd, l_speed, l_conf = GLOBAL_POLICY.query(objective, world)
        if l_cmd:
            if l_conf >= LEARNED_POLICY_CONFIDENCE_THRESHOLD:
                 l_dir = ACTION_CMD_DESC.get(l_cmd, "moving")
                 return l_cmd, l_speed, f"learned behavior | {l_conf*100:.0f}% conf"
            else:
                 policy_reason = f"hardcoded rule | policy conf {l_conf*100:.0f}% < {LEARNED_POLICY_CONFIDENCE_THRESHOLD*100:.0f}%"
        else:
            policy_reason = "hardcoded rule | no demo match"
    
    # Default policy reason if not using learning
    if not policy_reason:
        policy_reason = "hardcoded rule"

    # Determine Dynamic Speed
    max_ratio = max(offset_ratio, angle_ratio)
    
    # Progress Emoji Logic
    last_ratio = getattr(world, "last_align_ratio", None)
    ratio_emoji = ""
    if last_ratio is not None:
        if max_ratio < last_ratio - 0.01:
            ratio_emoji = " 🟢" # Improved
        elif max_ratio > last_ratio + 0.01:
            ratio_emoji = " 🔴" # Regressed
    setattr(world, "last_align_ratio", max_ratio)

    # Map ratio 1.0 -> MIN_SPEED, ratio 3.0 -> MAX_SPEED
    speed_factor = max(0.0, min(1.0, (max_ratio - 1.0) / 2.0))
    dynamic_speed = MIN_ALIGN_SPEED + (MAX_ALIGN_SPEED - MIN_ALIGN_SPEED) * speed_factor

    # Prioritize the most egregious error
    if offset_ratio >= angle_ratio:
        offset_dir = "right" if offset_x > 0 else "left"
        cmd = "r" if offset_x > 0 else "l"
        
        # Micro-Correction: If within 10mm, slow down
        if abs(offset_x) < MICRO_ALIGN_OFFSET_MM:
             dynamic_speed = min(dynamic_speed, MICRO_ALIGN_SPEED)

        return cmd, dynamic_speed, f"{policy_reason}: closing {offset_dir} offset of {abs(offset_x):.2f}mm (ratio {offset_ratio:.2f}){ratio_emoji}"
    else:
        angle_dir = "right" if angle > 0 else "left"
        cmd = "r" if angle > 0 else "l"

        # Micro-Correction: If within 5deg, slow down
        if abs(angle) < MICRO_ALIGN_ANGLE_DEG:
            dynamic_speed = min(dynamic_speed, MICRO_ALIGN_SPEED)

        return cmd, dynamic_speed, f"{policy_reason}: closing {angle_dir} angle of {abs(angle):.2f}deg (ratio {angle_ratio:.2f}){ratio_emoji}"


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
    if metric == "offset_abs":
        val = brick.get("offset_x")
        return abs(val) if val is not None else None
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


def update_world_from_vision(world, vision):
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
    world.update_vision(found, dist, angle, conf, offset_x, cam_h, brick_above, brick_below)
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


def wait_for_start_gates(world, vision, objective, robot=None, cmd=None, speed=None):
    start_time = time.time()
    stable = 0
    success_tracker = SuccessGateTracker(success_frames_required(objective))
    success_seen = False
    if robot:
        robot.stop()
    while time.time() - start_time < START_GATE_TIMEOUT_S:
        update_world_from_vision(world, vision)
        success_ok, confidence = evaluate_gate_status(world, objective)
        log_confidence(world, confidence, objective)
        confidence_ok = success_ok and confidence >= SUCCESS_CONFIDENCE_MIN
        success_met = success_tracker.update(confidence_ok)
        if success_ok and not success_seen and robot:
            robot.stop()
            success_seen = True
        if not success_ok:
            success_seen = False
        if success_met:
            if robot:
                robot.stop()
            return "success"
        if success_ok:
            time.sleep(CONTROL_DT)
            continue

        brick_check = telemetry_brick.evaluate_start_gates(world, objective, {}, world.process_rules)
        wall_check = telemetry_wall.evaluate_start_gates(world, objective, world.wall_envelope)
        robot_check = telemetry_robot_module.evaluate_start_gates(world, objective, {}, world.process_rules)
        if brick_check.ok and wall_check.ok and robot_check.ok:
            stable += 1
            if stable >= GATE_STABILITY_FRAMES:
                print(format_headline(f"[START] {objective} start gates met", COLOR_GREEN))
                return "start"
        else:
            stable = 0
        time.sleep(CONTROL_DT)
    return "timeout"


def run_alignment_segment(segment, objective, robot, vision, world, steps, raw_steps):
    if not steps:
        return False, "no motion steps"

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
    )
    if start_status == "success":
        if robot:
            robot.stop()
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
        update_world_from_vision(world, vision)
        success_ok, confidence = evaluate_gate_status(world, objective)
        log_confidence(world, confidence, objective)
        confidence_ok = success_ok and confidence >= SUCCESS_CONFIDENCE_MIN

        success_met = success_tracker.update(confidence_ok)
        if success_met:
            if robot:
                robot.stop()
            print(format_headline(f"[SUCCESS] {objective} criteria met", COLOR_GREEN))
            print(format_headline(format_success_details(world, objective), COLOR_WHITE))
            return True, "success gate"

        cmd, speed, cmd_reason = alignment_command(world, objective, gate_bounds, action_speeds)
        speed = apply_pursuit_speed(speed)
        speed = apply_confidence_speed(speed, success_ok, confidence, world)
        if cmd != last_cmd or cmd_reason != last_reason or speed != last_speed:
            print(format_headline(format_control_action_line(cmd, speed, cmd_reason), COLOR_WHITE))
            last_cmd = cmd
            last_reason = cmd_reason
            last_speed = speed

        if cmd:
            robot.send_command(cmd, speed)
            evt = MotionEvent(
                cmd_to_motion_type(cmd),
                int(speed * 255),
                int(CONTROL_DT * 1000),
            )
            world.update_from_motion(evt)
        else:
            if robot:
                robot.stop()
        time.sleep(CONTROL_DT)

    if robot:
        robot.stop()
    settle_deadline = min(objective_deadline, time.time() + SUCCESS_SETTLE_S)
    settle_tracker = SuccessGateTracker(success_frames_required(objective))
    while time.time() < settle_deadline:
        update_world_from_vision(world, vision)
        success_ok, confidence = evaluate_gate_status(world, objective)
        log_confidence(world, confidence, objective)
        confidence_ok = success_ok and confidence >= SUCCESS_CONFIDENCE_MIN
        success_met = settle_tracker.update(confidence_ok)
        if success_met:
            print(format_headline(f"[SUCCESS] {objective} criteria met", COLOR_GREEN))
            print(format_headline(format_success_details(world, objective), COLOR_WHITE))
            return True, "success gate"

        cmd, speed, cmd_reason = alignment_command(world, objective, gate_bounds, action_speeds)
        speed = apply_pursuit_speed(speed)
        speed = apply_confidence_speed(speed, success_ok, confidence, world)
        if cmd != last_cmd or cmd_reason != last_reason or speed != last_speed:
            print(format_headline(format_control_action_line(cmd, speed, cmd_reason), COLOR_WHITE))
            last_cmd = cmd
            last_reason = cmd_reason
            last_speed = speed

        if cmd:
            robot.send_command(cmd, speed)
            evt = MotionEvent(
                cmd_to_motion_type(cmd),
                int(speed * 255),
                int(CONTROL_DT * 1000),
            )
            world.update_from_motion(evt)
        time.sleep(CONTROL_DT)

    pause_after_fail(robot)
    return False, "success gate not reached"


def replay_segment(segment, objective, robot, vision, world):
    events = segment.get("events") or []
    raw_steps = merge_motion_steps(build_motion_sequence(events))
    steps = smooth_motion_steps(raw_steps)
    if not steps:
        return False, "no motion steps"
    if objective_uses_alignment_control(objective, world.process_rules):
        return run_alignment_segment(segment, objective, robot, vision, world, steps, raw_steps)

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
