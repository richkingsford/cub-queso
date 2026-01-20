#!/usr/bin/env python3
import argparse
import json
import math
import time
from dataclasses import dataclass
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
START_GATE_TIMEOUT_S = 8.0
SUCCESS_SETTLE_S = 1.0
MAX_PHASE_ATTEMPTS = 1
MAX_SPEED = 0.5
SMOOTH_SPEED = 0.3
SMOOTH_STEP_S = 1.0
DURATION_SCALE = 3.0
FAIL_PAUSE_S = 3.0
GATE_RANGE_SCALE = 5.0


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


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    pct = max(0.0, min(1.0, pct))
    idx = int(round(pct * (len(values) - 1)))
    return values[idx]


def expand_gate_range(min_val, max_val, scale=GATE_RANGE_SCALE):
    if min_val is None or max_val is None:
        return min_val, max_val
    center = (min_val + max_val) / 2.0
    half = (max_val - min_val) / 2.0
    half *= scale
    return center - half, center + half


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


def format_state_line(world):
    brick = world.brick or {}
    visible = "true" if brick.get("visible") else "false"
    return f"[STATE] brick visible={visible}"


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


def derive_success_gates(success_segments):
    metrics_by_obj = objective_metrics_map()
    directions = {}
    directions.update(telemetry_brick.METRIC_DIRECTIONS)
    directions.update(telemetry_robot_module.METRIC_DIRECTIONS)

    success_gates = {}
    for obj, segs in success_segments.items():
        metrics = metrics_by_obj.get(obj, [])
        if not metrics:
            continue
        metric_values = {metric: [] for metric in metrics}
        for seg in segs:
            for state in seg.get("states") or []:
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
            p90 = percentile(values, 0.9)
            stats = {}
            direction = directions.get(metric)
            if direction in ("low", "high", "band") or direction is None:
                min_val = p10
                max_val = p90
                if min_val is not None and max_val is not None:
                    min_val, max_val = expand_gate_range(min_val, max_val)
                if min_val is not None:
                    stats["min"] = round_value(min_val)
                if max_val is not None:
                    stats["max"] = round_value(max_val)
            if stats:
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
    success_gates = derive_success_gates(success_segments)

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


def evaluate_gate_status(world, objective):
    process_rules = world.process_rules or {}
    telemetry_rules = {}
    brick_success = telemetry_brick.evaluate_success_gates(world, objective, telemetry_rules, process_rules)
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
    success_frames = 0
    while time.time() - start_time < START_GATE_TIMEOUT_S:
        update_world_from_vision(world, vision)
        success_ok, _, _ = evaluate_gate_status(world, objective)
        if success_ok:
            success_frames += 1
            if success_frames == 1 and robot:
                robot.stop()
            if success_frames >= GATE_STABILITY_FRAMES:
                return "success"
            time.sleep(CONTROL_DT)
            continue
        success_frames = 0

        if robot and cmd and speed:
            robot.send_command(cmd, speed)
            evt = MotionEvent(
                cmd_to_motion_type(cmd),
                int(speed * 255),
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


def replay_segment(segment, objective, robot, vision, world):
    events = segment.get("events") or []
    steps = smooth_motion_steps(merge_motion_steps(build_motion_sequence(events)))
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
        time.sleep(3.0)
        return True, "success gate"
    if start_status != "start":
        pause_after_fail(robot)
        return False, "start gates not met"

    allow_early_exit = True

    success_frames = 0
    fail_frames = 0
    last_action = None
    last_cmd = default_step.cmd
    last_speed = default_step.speed

    for step in steps:
        if step.label != last_action:
            update_world_from_vision(world, vision)
            print(format_headline(format_state_line(world), COLOR_WHITE))
            print(format_headline(format_action_line(step, target_visible), COLOR_WHITE))
            last_action = step.label
        last_cmd = step.cmd
        last_speed = step.speed
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

                if success_ok:
                    success_frames += 1
                    if success_frames == 1:
                        robot.stop()
                    if success_frames >= GATE_STABILITY_FRAMES:
                        robot.stop()
                        print(format_headline(f"[SUCCESS] {objective} criteria met 🎉", COLOR_GREEN))
                        time.sleep(3.0)
                        return True, "success gate"
                    time.sleep(CONTROL_DT)
                    continue
                else:
                    success_frames = 0

            robot.send_command(step.cmd, step.speed)
            evt = MotionEvent(
                cmd_to_motion_type(step.cmd),
                int(step.speed * 255),
                int(CONTROL_DT * 1000),
            )
            world.update_from_motion(evt)
            time.sleep(CONTROL_DT)

    robot.stop()
    settle_start = time.time()
    while time.time() - settle_start < SUCCESS_SETTLE_S:
        update_world_from_vision(world, vision)
        success_ok, fail, reason = evaluate_gate_status(world, objective)
        if fail and allow_early_exit:
            pause_after_fail(robot)
            return False, reason or "fail gate"
        if success_ok:
            print(format_headline(f"[SUCCESS] {objective} criteria met 🎉", COLOR_GREEN))
            time.sleep(3.0)
            return True, "success gate"
        if last_cmd:
            robot.send_command(last_cmd, last_speed)
            evt = MotionEvent(
                cmd_to_motion_type(last_cmd),
                int(last_speed * 255),
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
