#!/usr/bin/env python3
import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from helper_demo_log_utils import extract_attempt_segments, load_demo_logs, normalize_objective_label
from helper_vision_aruco import ArucoBrickVision
from robot_control import Robot
from telemetry_robot import MotionEvent, ObjectiveState, WorldModel
import telemetry_brick
import telemetry_robot as telemetry_robot_module
import telemetry_wall


DEMO_DIR = Path(__file__).resolve().parent / "demos"
PROCESS_MODEL_FILE = Path(__file__).resolve().parent / "world_model_process.json"

CONTROL_HZ = 20.0
CONTROL_DT = 1.0 / CONTROL_HZ
GATE_STABILITY_FRAMES = 5
SUCCESS_CONSECUTIVE_FRAMES = 3
SUCCESS_MAJORITY_WINDOW = 5
SUCCESS_MAJORITY_REQUIRED = 3
SUCCESS_VISIBLE_FALSE_GRACE_S = 0.0
START_GATE_TIMEOUT_S = 8.0
SUCCESS_SETTLE_S = 1.0
SUCCESS_TAIL_WINDOW_S = 0.5
MAX_PHASE_ATTEMPTS = 1
MAX_SPEED = 0.5
SMOOTH_SPEED = 0.3
SMOOTH_STEP_S = 1.0
DURATION_SCALE = 3.0
FAIL_PAUSE_S = 3.0
FIND_BRICK_SLOW_FACTOR = 4.0

SUCCESS_FRAMES_BY_OBJECTIVE = {
    "FIND_BRICK": 3,
}

ALIGNMENT_METRICS = {"angle_abs", "offset_abs", "dist"}


COLOR_RESET = "\033[0m"
COLOR_GREEN = "\033[32m"
COLOR_RED = "\033[31m"
COLOR_WHITE = "\033[37m"


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
        format_gate_metrics(cfg.get("fail_gates") or {}),
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


def format_action_line(step, target_visible):
    action = ACTION_CMD_DESC.get(step.cmd, "moving")
    power = f"{step.speed:.2f}"
    duration = f"{step.duration_s:.2f}"
    if target_visible is None:
        suffix = f"for {duration}s"
    else:
        vis_txt = "true" if target_visible else "false"
        suffix = f"for {duration}s or until brickVisible={vis_txt}"
    return f"[ACT] {action} at {power} power {suffix}"


def format_control_action_line(cmd, speed, reason):
    action = ACTION_CMD_DESC.get(cmd, "moving") if cmd else "holding position"
    if cmd:
        return f"[ACT] {action} at {speed:.2f} power ({reason})"
    return f"[ACT] {action} ({reason})"


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
        return min(value, MAX_SPEED)

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
    brick = world.brick or {}
    if not brick.get("visible"):
        scan_cmd = recent_turn_preference(world, telemetry_brick.VISIBILITY_LOST_GRACE_S)
        if scan_cmd is None:
            scan_cmd = telemetry_robot_module.resolve_scan_direction(world.process_rules, objective)
        return scan_cmd, speeds["scan"], "scan"

    angle_max = (gate_bounds.get("angle_abs") or {}).get("max")
    angle = brick.get("angle") or 0.0
    if angle_max is not None and abs(angle) > angle_max:
        cmd = "l" if angle > 0 else "r"
        return cmd, speeds["turn"], "angle"

    offset_max = (gate_bounds.get("offset_abs") or {}).get("max")
    offset_x = brick.get("offset_x") or 0.0
    if offset_max is not None and abs(offset_x) > offset_max:
        cmd = "l" if offset_x > 0 else "r"
        return cmd, speeds["turn"], "offset"

    dist = brick.get("dist")
    dist_bounds = gate_bounds.get("dist") or {}
    dist_min = dist_bounds.get("min")
    dist_max = dist_bounds.get("max")
    if dist is not None:
        if dist_max is not None and dist > dist_max:
            return "f", speeds["forward"], "dist"
        if dist_min is not None and dist < dist_min:
            return "b", speeds["backward"], "dist"

    return None, 0.0, "hold"


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
        metric_values = {metric: [] for metric in metrics}
        for seg in segs:
            for state in select_tail_states(seg.get("states") or []):
                for metric in metrics:
                    value = metric_value_from_state(state, metric)
                    if value is None:
                        continue
                    metric_values[metric].append(value)

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

    start_gates = derive_start_gates(success_segments)
    wall_objective_rules = telemetry_wall.load_wall_objective_rules()
    objective_rules = {}
    if isinstance(wall_objective_rules, dict):
        objective_rules.update(wall_objective_rules)
    if isinstance(objectives, dict):
        objective_rules.update(objectives)
    success_gate_scales = derive_success_gate_scales(segments_by_obj, objective_rules)
    success_gates = derive_success_gates(
        success_segments,
        success_gate_scales,
        objective_rules,
    )

    all_objectives = set(objectives.keys())
    all_objectives.update(start_gates.keys())
    all_objectives.update(success_gates.keys())
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
        visible_gate = cfg.get("success_gates", {}).get("visible", {})
        if visible_gate.get("min") is False:
            cfg["success_gates"] = {"visible": {"min": False}}


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
    max_speed=MAX_SPEED,
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

    brick_fail = telemetry_brick.evaluate_failure_gates(world, objective, telemetry_rules, process_rules)
    wall_fail = telemetry_wall.evaluate_failure_gates(world, objective, world.wall_envelope)
    robot_fail = telemetry_robot_module.evaluate_failure_gates(world, objective, telemetry_rules, process_rules)

    success_ok = brick_success.ok and wall_success.ok and robot_success.ok
    reasons = []
    if not brick_fail.ok:
        reasons.extend(brick_fail.reasons)
    if not wall_fail.ok:
        reasons.extend(wall_fail.reasons)
    if not robot_fail.ok:
        reasons.extend(robot_fail.reasons)
    fail = bool(reasons)
    return success_ok, fail, "; ".join(reasons)


def wait_for_start_gates(world, vision, objective, robot=None, cmd=None, speed=None):
    start_time = time.time()
    stable = 0
    success_tracker = SuccessGateTracker(success_frames_required(objective))
    while time.time() - start_time < START_GATE_TIMEOUT_S:
        update_world_from_vision(world, vision)
        success_ok, _, _ = evaluate_gate_status(world, objective)
        success_met = success_tracker.update(success_ok)
        if success_ok and success_tracker.consecutive == 1 and robot:
            robot.stop()
        if success_met:
            if robot:
                robot.stop()
            return "success"
        if success_ok:
            time.sleep(CONTROL_DT)
            continue

        if robot and cmd and speed:
            active_speed = adjust_speed_for_find_brick(world, objective, speed)
            robot.send_command(cmd, active_speed)
            evt = MotionEvent(
                cmd_to_motion_type(cmd),
                int(active_speed * 255),
                int(CONTROL_DT * 1000),
            )
            world.update_from_motion(evt)
        brick_check = telemetry_brick.evaluate_start_gates(world, objective, {}, world.process_rules)
        wall_check = telemetry_wall.evaluate_start_gates(world, objective, world.wall_envelope)
        robot_check = telemetry_robot_module.evaluate_start_gates(world, objective, {}, world.process_rules)
        if brick_check.ok and wall_check.ok and robot_check.ok:
            stable += 1
            if stable >= GATE_STABILITY_FRAMES:
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
        time.sleep(3.0)
        return True, "success gate"
    if start_status != "start":
        pause_after_fail(robot)
        return False, "start gates not met"

    duration_s = sum(step.duration_s for step in steps)
    if duration_s <= 0:
        duration_s = START_GATE_TIMEOUT_S

    success_tracker = SuccessGateTracker(success_frames_required(objective))
    fail_frames = 0
    last_cmd = None
    last_reason = None
    start_time = time.time()

    while time.time() - start_time < duration_s:
        update_world_from_vision(world, vision)
        success_ok, fail, reason = evaluate_gate_status(world, objective)

        if fail:
            fail_frames += 1
            if fail_frames >= GATE_STABILITY_FRAMES:
                if robot:
                    robot.stop()
                pause_after_fail(robot)
                return False, reason or "fail gate"
        else:
            fail_frames = 0

        success_met = success_tracker.update(success_ok)
        if success_ok:
            if success_tracker.consecutive == 1 and robot:
                robot.stop()
            if success_met:
                if robot:
                    robot.stop()
                print(format_headline(f"[SUCCESS] {objective} criteria met", COLOR_GREEN))
                print(format_headline(format_success_details(world, objective), COLOR_WHITE))
                time.sleep(3.0)
                return True, "success gate"
            time.sleep(CONTROL_DT)
            continue

        cmd, speed, cmd_reason = alignment_command(world, objective, gate_bounds, action_speeds)
        if cmd != last_cmd or cmd_reason != last_reason:
            print(format_headline(format_control_action_line(cmd, speed, cmd_reason), COLOR_WHITE))
            last_cmd = cmd
            last_reason = cmd_reason

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
    settle_start = time.time()
    settle_tracker = SuccessGateTracker(success_frames_required(objective))
    while time.time() - settle_start < SUCCESS_SETTLE_S:
        update_world_from_vision(world, vision)
        success_ok, fail, reason = evaluate_gate_status(world, objective)
        if fail:
            pause_after_fail(robot)
            return False, reason or "fail gate"
        success_met = settle_tracker.update(success_ok)
        if success_met:
            print(format_headline(f"[SUCCESS] {objective} criteria met", COLOR_GREEN))
            print(format_headline(format_success_details(world, objective), COLOR_WHITE))
            time.sleep(3.0)
            return True, "success gate"

        cmd, speed, cmd_reason = alignment_command(world, objective, gate_bounds, action_speeds)
        if cmd != last_cmd or cmd_reason != last_reason:
            print(format_headline(format_control_action_line(cmd, speed, cmd_reason), COLOR_WHITE))
            last_cmd = cmd
            last_reason = cmd_reason

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

    success_tracker = SuccessGateTracker(success_frames_required(objective))
    fail_frames = 0
    last_action = None
    last_cmd = default_step.cmd
    last_speed_base = default_step.speed

    for step in steps:
        if step.label != last_action:
            update_world_from_vision(world, vision)
            print(format_headline(format_action_line(step, target_visible), COLOR_WHITE))
            last_action = step.label
        last_cmd = step.cmd
        last_speed_base = step.speed
        step_start = time.time()
        while time.time() - step_start < step.duration_s:
            update_world_from_vision(world, vision)
            success_ok, fail, reason = evaluate_gate_status(world, objective)

            if allow_early_exit:
                if fail:
                    fail_frames += 1
                    if fail_frames >= GATE_STABILITY_FRAMES:
                        robot.stop()
                        pause_after_fail(robot)
                        return False, reason or "fail gate"
                else:
                    fail_frames = 0

                success_met = success_tracker.update(success_ok)
                if success_ok and success_tracker.consecutive == 1:
                    robot.stop()
                if success_met:
                    robot.stop()
                    print(format_headline(f"[SUCCESS] {objective} criteria met 🎉", COLOR_GREEN))
                    print(format_headline(format_success_details(world, objective), COLOR_WHITE))
                    time.sleep(3.0)
                    return True, "success gate"
                if success_ok:
                    time.sleep(CONTROL_DT)
                    continue

            active_speed = adjust_speed_for_find_brick(world, objective, step.speed)
            robot.send_command(step.cmd, active_speed)
            evt = MotionEvent(
                cmd_to_motion_type(step.cmd),
                int(active_speed * 255),
                int(CONTROL_DT * 1000),
            )
            world.update_from_motion(evt)
            time.sleep(CONTROL_DT)

    robot.stop()
    settle_start = time.time()
    settle_tracker = SuccessGateTracker(success_frames_required(objective))
    while time.time() - settle_start < SUCCESS_SETTLE_S:
        update_world_from_vision(world, vision)
        success_ok, fail, reason = evaluate_gate_status(world, objective)
        if fail and allow_early_exit:
            pause_after_fail(robot)
            return False, reason or "fail gate"
        success_met = settle_tracker.update(success_ok)
        if success_met:
            print(format_headline(f"[SUCCESS] {objective} criteria met 🎉", COLOR_GREEN))
            print(format_headline(format_success_details(world, objective), COLOR_WHITE))
            time.sleep(3.0)
            return True, "success gate"
        if last_cmd:
            active_speed = adjust_speed_for_find_brick(world, objective, last_speed_base)
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
                start_desc, success_desc, fail_desc = format_gate_lines(cfg)
                print(f"  start gates: {start_desc}")
                print(f"  success gates: {success_desc}")
                print(f"  fail gates: {fail_desc}")
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
    args = parser.parse_args()

    run_autobuild(session_name=args.session)
    print("\n" * 5, end="")


if __name__ == "__main__":
    main()
