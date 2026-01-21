import argparse
import json
import math
import os
import statistics
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from helper_demo_log_utils import extract_attempt_segments, load_demo_logs, normalize_objective_label
from helper_robot_control import Robot
import telemetry_brick
import telemetry_robot as telemetry_robot_module
import telemetry_wall
from telemetry_robot import WorldModel, MotionEvent, ObjectiveState
from helper_stream_server import StreamServer
from helper_vision_aruco import ArucoBrickVision

# ANSI colors for terminal highlights.
COLOR_RESET = "\033[0m"
COLOR_GREEN = "\033[32m"
COLOR_RED = "\033[31m"

def format_headline(headline, color, details=""):
    if details is None:
        details = ""
    return f"{color}{headline}{COLOR_RESET}{details}"

def log_action(phase, label, cmd=None, speed=None, duration=None):
    line = f"[ACT] {phase}: {label}"
    details = []
    if cmd is not None:
        details.append(str(cmd))
    if speed is not None:
        details.append(f"{speed:.2f}")
    if duration is not None:
        details.append(f"{duration:.2f}s")
    if details:
        line += " (" + " ".join(details) + ")"
    print(line)

WORLD_MODEL_PROCESS_FILE = Path(__file__).resolve().parent / "world_model_process.json"

def load_process_sequence(model_path=WORLD_MODEL_PROCESS_FILE):
    if not model_path.exists():
        return list(PHASE_SEQUENCE)
    try:
        with open(model_path, "r") as f:
            model = json.load(f)
    except (OSError, json.JSONDecodeError):
        return list(PHASE_SEQUENCE)
    objectives = model.get("objectives")
    if not isinstance(objectives, dict) or not objectives:
        return list(PHASE_SEQUENCE)
    sequence = []
    seen = set()
    for name in objectives.keys():
        normalized = normalize_objective_label(name)
        if not normalized or normalized in seen:
            continue
        sequence.append(normalized)
        seen.add(normalized)
    return sequence or list(PHASE_SEQUENCE)

# --- CONFIG ---
DEMO_DIR = Path(__file__).resolve().parent / "demos"
WEB_PORT = 5000
STREAM_HOST = "127.0.0.1"
STREAM_FPS = 10
STREAM_JPEG_QUALITY = 90
CONTROL_HZ = 20.0
CONTROL_DT = 1.0 / CONTROL_HZ
STATE_LOG_INTERVAL = 0.5

SCAN_SPEED = 0.3
ALIGN_FORWARD_SPEED = 0.15
ALIGN_TURN_SPEED = 0.25

SCOOP_DRIVE_SPEED = 0.25
SCOOP_DRIVE_TIME = 1.2
SCOOP_LIFT_SPEED = 0.5
SCOOP_LIFT_TIME = 0.8

LIFT_SPEED = 0.6
LIFT_TIME = 2.0
FIND_WALL2_SPEED = 0.0
FIND_WALL2_TIME = 0.0
POSITION_BRICK_SPEED = 0.0
POSITION_BRICK_TIME = 0.0
PLACE_SPEED = 0.6
PLACE_TIME = 2.0
RETREAT_SPEED = 0.2
RETREAT_TIME = 1.0

BLIND_WINDOW_FALLBACK_S = 2.0
COMMIT_WINDOW_FALLBACK_S = 1.0

RECOVERY_TURN_SPEED = 0.25
RECOVERY_BACK_SPEED = 0.2
RECOVERY_STEP_S = 0.6
RECOVERY_MAX_S = 6.0
MAX_PHASE_ATTEMPTS = 5

GATE_MIN_SAMPLES = 5
GATE_STABILITY_FRAMES = 5
BRICK_HEIGHT_MIN_MM = 5.0
BRICK_HEIGHT_MAX_MM = 150.0
BRICK_HEIGHT_MAX_ADJUST_FRACTION = 0.7
BRICK_HEIGHT_MIN_DURATION_S = 0.2

METRIC_DIRECTIONS = {
    "angle_abs": "low",
    "offset_abs": "low",
    "dist": "low",
    "visible": "high",
}

PHASE_METRICS = {
    "EXIT_WALL": ("angle_abs", "offset_abs", "dist", "visible"),
    "FIND_BRICK": ("angle_abs", "offset_abs", "dist", "visible"),
    "ALIGN_BRICK": ("angle_abs", "offset_abs", "dist", "visible"),
    "SCOOP": ("angle_abs", "offset_abs", "dist", "visible"),
    "POSITION_BRICK": ("angle_abs", "offset_abs", "dist", "visible"),
}

PHASE_SEQUENCE = [
    "FIND_WALL",
    "EXIT_WALL",
    "FIND_BRICK",
    "ALIGN_BRICK",
    "SCOOP",
    "LIFT",
    "FIND_WALL2",
    "POSITION_BRICK",
    "PLACE",
    "RETREAT",
]

PERCEPTION_PHASES = {"FIND_BRICK", "ALIGN_BRICK"}
START_GATE_BRICK_PHASES = {"ALIGN_BRICK", "SCOOP"}

DEFAULT_GATE_TEMPLATES = {
    "FIND_BRICK": {
        "success": {
            "angle_abs": {"max": 25.0, "samples": 1},
            "offset_abs": {"max": 40.0, "samples": 1},
            "dist": {"max": 600.0, "samples": 1},
            "visible": {"min": 1.0, "samples": 1},
        },
        "failure": {},
        "temporal": {},
    },
    "ALIGN_BRICK": {
        "success": {
            "angle_abs": {"max": 12.0, "samples": 1},
            "offset_abs": {"max": 20.0, "samples": 1},
            "dist": {"max": 400.0, "samples": 1},
            "visible": {"min": 1.0, "samples": 1},
        },
        "failure": {},
        "temporal": {},
    },
}


class Phase(Enum):
    FIND_WALL = "FIND_WALL"
    EXIT_WALL = "EXIT_WALL"
    FIND_BRICK = "FIND_BRICK"
    ALIGN_BRICK = "ALIGN_BRICK"
    SCOOP = "SCOOP"
    LIFT = "LIFT"
    FIND_WALL2 = "FIND_WALL2"
    POSITION_BRICK = "POSITION_BRICK"
    PLACE = "PLACE"
    RETREAT = "RETREAT"


@dataclass
class GateResult:
    success: bool
    correction: bool
    fail: bool
    reason: str = ""


@dataclass
class PhaseResult:
    success: bool
    reason: str
    elapsed: float
    start_gate_failed: bool = False


@dataclass
class MotionPrimitive:
    cmd: str
    speed: float
    duration_s: float
    label: str


class PhaseLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.path, "w")
        self._fp.write("[\n")
        self._first = True

    def _write(self, payload):
        if not self._first:
            self._fp.write(",\n")
        self._first = False
        self._fp.write(json.dumps(payload))
        self._fp.flush()

    def log_phase(self, phase, marker, world, status=None, reason=None):
        payload = {
            "type": "phase",
            "marker": marker,
            "phase": phase,
            "timestamp": round(time.time(), 3),
            "world": world.to_dict(),
        }
        if status is not None:
            payload["status"] = status
        if reason:
            payload["reason"] = reason
        self._write(payload)

    def log_state(self, phase, world):
        payload = {
            "type": "state",
            "phase": phase,
            "timestamp": round(time.time(), 3),
            "world": world.to_dict(),
        }
        self._write(payload)

    def close(self):
        if self._fp:
            self._fp.write("\n]\n")
            self._fp.close()
            self._fp = None


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(round(pct * (len(values) - 1)))))
    return values[k]


def build_stat_band(values, clamp_min=None, clamp_max=None):
    if not values:
        return None
    mu = statistics.mean(values)
    sigma = statistics.pstdev(values) if len(values) > 1 else 0.0
    min_val = mu - sigma
    max_val = mu + sigma
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


ACTION_CMD_MAP = {
    "forward": "f",
    "backward": "b",
    "left_turn": "l",
    "right_turn": "r",
    "mast_up": "u",
    "mast_down": "d",
}

ATTEMPT_TYPES = {"SUCCESS", "FAIL", "RECOVER", "NOMINAL"}


def _normalize_attempt_type(attempt_type):
    if attempt_type is None:
        return None
    key = str(attempt_type).strip().upper()
    if key in ("RECOVERY", "RECOVER"):
        return "RECOVER"
    if key in ATTEMPT_TYPES:
        return key
    return None


def _normalize_objective_label(objective):
    if objective is None:
        return None
    if isinstance(objective, ObjectiveState):
        return objective.value
    return normalize_objective_label(objective)


def _objective_state_from_label(label):
    if not label:
        return None
    label = normalize_objective_label(label)
    try:
        return ObjectiveState(label)
    except ValueError:
        return None


def select_demo_attempt_segment(logs, objective, attempt_type):
    target_obj = _normalize_objective_label(objective)
    target_type = _normalize_attempt_type(attempt_type)
    if not target_obj or not target_type:
        return None, None

    def collect_matches(allow_objective_span):
        found = []
        for path, data in logs:
            for seg in extract_attempt_segments(data):
                seg_obj = normalize_objective_label(seg.get("objective"))
                if seg_obj != target_obj:
                    continue
                if seg.get("type") != target_type:
                    continue
                if not allow_objective_span and seg.get("source") == "objective":
                    continue
                events = seg.get("events") or []
                event_count = len(events)
                duration = 0.0
                if seg.get("start") is not None and seg.get("end") is not None:
                    duration = seg["end"] - seg["start"]
                found.append((event_count, duration, seg, path))
        return found

    matches = collect_matches(allow_objective_span=False)
    if not matches and target_type == "SUCCESS":
        matches = collect_matches(allow_objective_span=True)

    if not matches:
        return None, None

    matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best = matches[0]
    return best[2], best[3]


def build_motion_sequence(events):
    sequence = []
    for evt in events:
        if evt.get("type") == "action":
            cmd_name = evt.get("command")
            power = evt.get("power", 0)
            duration_ms = evt.get("duration_ms", 0)
        elif evt.get("type") == "event":
            event = evt.get("event") or {}
            cmd_name = event.get("type")
            power = event.get("power", 0)
            duration_ms = event.get("duration_ms", 0)
        else:
            continue
        cmd = ACTION_CMD_MAP.get(cmd_name)
        duration_s = (duration_ms or 0) / 1000.0
        if not cmd or duration_s <= 0:
            continue
        speed = max(0.0, min(1.0, float(power or 0) / 255.0))
        if speed <= 0:
            continue
        sequence.append(MotionPrimitive(cmd, speed, duration_s, cmd_name))
    return sequence


def merge_motion_sequence(sequence, speed_tol=0.02):
    if not sequence:
        return []
    merged = [sequence[0]]
    for step in sequence[1:]:
        last = merged[-1]
        if step.cmd == last.cmd and abs(step.speed - last.speed) <= speed_tol:
            last.duration_s += step.duration_s
        else:
            merged.append(step)
    return merged


def replay_motion_sequence(sequence, objective, robot, vision, world, telemetry_logger=None,
                           stream_state=None, frame_callback=None, log_rate_hz=10, stop_flag=None):
    if not sequence:
        return False

    dt = 1.0 / max(1.0, log_rate_hz)
    for primitive in sequence:
        elapsed = 0.0
        while elapsed < primitive.duration_s:
            if stop_flag and stop_flag():
                robot.stop()
                return False
            step = min(dt, primitive.duration_s - elapsed)
            update_world_model(world, vision, stream_state=stream_state, frame_callback=frame_callback)
            robot.send_command(primitive.cmd, primitive.speed)
            evt = MotionEvent(_cmd_to_motion_type(primitive.cmd), int(primitive.speed * 255), int(step * 1000))
            world.update_from_motion(evt)
            if telemetry_logger:
                telemetry_logger.log_event(evt, objective)
                telemetry_logger.log_state(world)
            time.sleep(step)
            elapsed += step

    robot.stop()
    return True


def run_demo_attempt(objective, attempt_type, session_name=None, robot=None, vision=None, world=None,
                     telemetry_logger=None, stream_state=None, frame_callback=None,
                     log_rate_hz=10, stop_flag=None):
    target_obj = _normalize_objective_label(objective)
    target_type = _normalize_attempt_type(attempt_type)
    if not target_obj or not target_type:
        return False, f"Unknown objective or attempt: {objective} / {attempt_type}"

    logs = load_demo_logs(DEMO_DIR, session_name)
    segment, source = select_demo_attempt_segment(logs, target_obj, target_type)
    if not segment:
        return False, f"No demo attempts for {target_obj} ({target_type})."

    sequence = merge_motion_sequence(build_motion_sequence(segment.get("events") or []))
    if not sequence:
        return False, f"No motion events in demo attempt for {target_obj} ({target_type})."

    created_robot = False
    created_vision = False
    if robot is None:
        robot = Robot()
        created_robot = True
    if vision is None:
        vision = ArucoBrickVision(debug=False)
        created_vision = True
    if world is None:
        world = WorldModel()

    obj_state = _objective_state_from_label(target_obj)
    if obj_state:
        world.objective_state = obj_state

    try:
        ok = replay_motion_sequence(
            sequence,
            target_obj,
            robot,
            vision,
            world,
            telemetry_logger=telemetry_logger,
            stream_state=stream_state,
            frame_callback=frame_callback,
            log_rate_hz=log_rate_hz,
            stop_flag=stop_flag,
        )
    finally:
        robot.stop()
        if created_robot:
            robot.close()
        if created_vision:
            vision.close()

    return ok, f"Replayed {source}"


def summarize_demo_stats(logs):
    if not logs:
        print("[STATS] No demo logs loaded.")
        return
    total_entries = sum(len(data) for _, data in logs)
    total_segments = 0
    counts = {}
    durations = {}

    for _, data in logs:
        segments = extract_attempt_segments(data)
        total_segments += len(segments)
        for seg in segments:
            obj = normalize_objective_label(seg.get("objective")) or "UNKNOWN"
            seg_type = seg.get("type") or "UNKNOWN"
            counts.setdefault(obj, {}).setdefault(seg_type, 0)
            counts[obj][seg_type] += 1
            if seg.get("start") is not None and seg.get("end") is not None:
                durations.setdefault(obj, {}).setdefault(seg_type, []).append(seg["end"] - seg["start"])

    print(f"[STATS] Loaded {len(logs)} demo log(s), {total_entries} entries, {total_segments} segments.")
    for obj in sorted(counts):
        parts = []
        for seg_type in sorted(counts[obj]):
            count = counts[obj][seg_type]
            seg_durations = durations.get(obj, {}).get(seg_type, [])
            if seg_durations:
                avg = statistics.mean(seg_durations)
                parts.append(f"{seg_type}: {count} (avg {avg:.2f}s)")
            else:
                parts.append(f"{seg_type}: {count}")
        print(f"[STATS] {obj}: " + "; ".join(parts))


def _metric_value(brick, metric):
    visible = bool(brick.get("visible"))
    if metric == "visible":
        return 1.0 if visible else 0.0
    if not visible:
        return None
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
    return None


def extract_metrics_from_states(states, metrics):
    collected = {m: [] for m in metrics}
    for state in states:
        brick = state.get("brick") or {}
        for metric in metrics:
            val = _metric_value(brick, metric)
            if val is None:
                continue
            collected[metric].append(val)
    return collected


def average_forward_speed(events, start_time, end_time):
    if start_time is None or end_time is None or end_time <= start_time:
        return None
    total = 0.0
    weighted = 0.0
    for evt in events:
        if evt.get("type") == "action":
            cmd = evt.get("command")
            power = evt.get("power", 0)
            duration = (evt.get("duration_ms") or 0) / 1000.0
        else:
            cmd = (evt.get("event") or {}).get("type")
            power = (evt.get("event") or {}).get("power", 0)
            duration = ((evt.get("event") or {}).get("duration_ms") or 0) / 1000.0
        if cmd != "forward" or duration <= 0:
            continue
        evt_start = evt.get("timestamp")
        if evt_start is None:
            continue
        evt_end = evt_start + duration
        overlap = max(0.0, min(evt_end, end_time) - max(evt_start, start_time))
        if overlap <= 0:
            continue
        speed = (power or 0) / 255.0
        weighted += speed * overlap
        total += overlap
    if total <= 0:
        return None
    return weighted / total


def collect_attempt_segments(logs, attempt_type):
    target_type = _normalize_attempt_type(attempt_type)
    if not target_type:
        return []
    segments = []
    for _, data in logs:
        for seg in extract_attempt_segments(data):
            if seg.get("type") == target_type:
                segments.append(seg)
    return segments


def collect_attempt_types_by_objective(logs):
    attempt_types = {}
    for _, data in logs:
        for seg in extract_attempt_segments(data):
            obj = normalize_objective_label(seg.get("objective"))
            if not obj:
                continue
            seg_type = _normalize_attempt_type(seg.get("type"))
            if not seg_type:
                continue
            attempt_types.setdefault(obj, set()).add(seg_type)
    return attempt_types


def learn_scan_preferences(logs, attempt_types=("SUCCESS", "NOMINAL")):
    prefs = {}
    for _, data in logs:
        for seg in extract_attempt_segments(data):
            if seg.get("type") not in attempt_types:
                continue
            obj = normalize_objective_label(seg.get("objective"))
            if not obj:
                continue
            summary = _segment_motion_summary(seg.get("events") or [])
            if not summary:
                continue
            left = summary.get("l", {}).get("duration", 0.0)
            right = summary.get("r", {}).get("duration", 0.0)
            if left <= 0 and right <= 0:
                continue
            totals = prefs.setdefault(obj, {"l": 0.0, "r": 0.0})
            totals["l"] += left
            totals["r"] += right

    resolved = {}
    for obj, totals in prefs.items():
        left = totals.get("l", 0.0)
        right = totals.get("r", 0.0)
        if left == right:
            continue
        resolved[obj] = "l" if left > right else "r"
    return resolved


def _segment_motion_summary(events):
    sequence = build_motion_sequence(events or [])
    if not sequence:
        return {}
    per_cmd = {}
    for primitive in sequence:
        entry = per_cmd.setdefault(primitive.cmd, {"duration": 0.0, "speed_weighted": 0.0})
        entry["duration"] += primitive.duration_s
        entry["speed_weighted"] += primitive.speed * primitive.duration_s
    summary = {}
    for cmd, entry in per_cmd.items():
        duration = entry["duration"]
        if duration <= 0:
            continue
        summary[cmd] = {
            "duration": duration,
            "speed": entry["speed_weighted"] / duration,
        }
    return summary


def learn_gates_from_logs(logs):
    gate_acc = {"success": {}, "failure": {}, "success_times": {}, "failure_times": {}}
    success_segments = []

    for _, data in logs:
        segments = extract_attempt_segments(data)
        for seg in segments:
            obj = normalize_objective_label(seg.get("objective"))
            seg_type = seg.get("type")
            if not obj or not seg_type:
                continue
            duration = None
            if seg.get("start") is not None and seg.get("end") is not None:
                duration = seg["end"] - seg["start"]
            metrics = extract_metrics_from_states(seg.get("states", []), PHASE_METRICS.get(obj, ()))
            if seg_type == "SUCCESS":
                if duration is not None:
                    gate_acc["success_times"].setdefault(obj, []).append(duration)
                for metric, values in metrics.items():
                    if values:
                        gate_acc["success"].setdefault(obj, {}).setdefault(metric, []).extend(values)
                success_segments.append(seg)
            elif seg_type == "FAIL":
                if duration is not None:
                    gate_acc["failure_times"].setdefault(obj, []).append(duration)
                for metric, values in metrics.items():
                    if values:
                        gate_acc["failure"].setdefault(obj, {}).setdefault(metric, []).extend(values)

    learned = {}
    for obj, metrics in PHASE_METRICS.items():
        success_metrics = {}
        failure_metrics = {}
        for metric in metrics:
            success_vals = gate_acc["success"].get(obj, {}).get(metric, [])
            failure_vals = gate_acc["failure"].get(obj, {}).get(metric, [])
            clamp_min = 0.0
            clamp_max = 1.0 if metric == "visible" else None
            success_stats = build_stat_band(success_vals, clamp_min=clamp_min, clamp_max=clamp_max)
            failure_stats = build_stat_band(failure_vals, clamp_min=clamp_min, clamp_max=clamp_max)
            if success_stats:
                success_metrics[metric] = success_stats
            if failure_stats:
                failure_metrics[metric] = failure_stats

        success_time = build_stat_band(gate_acc["success_times"].get(obj, []), clamp_min=0.0)
        failure_time = build_stat_band(gate_acc["failure_times"].get(obj, []), clamp_min=0.0)
        if success_metrics or failure_metrics or success_time or failure_time:
            learned[obj] = {
                "success": success_metrics,
                "failure": failure_metrics,
                "temporal": {
                    "success_time": success_time,
                    "fail_time": failure_time,
                },
            }

    return learned, success_segments


def _telemetry_rules_from_gates(learned_gates):
    telemetry_rules = {}
    for obj, gate in learned_gates.items():
        if obj == "profiles" or not isinstance(gate, dict):
            continue
        success_metrics = gate.get("success", {})
        failure_metrics = gate.get("failure", {})
        if not success_metrics and not failure_metrics:
            continue
        telemetry_rules[obj] = {
            "gates": {
                "success": {"metrics": success_metrics},
                "failure": {"metrics": failure_metrics},
            }
        }
    for obj, template in DEFAULT_GATE_TEMPLATES.items():
        if obj in telemetry_rules:
            continue
        success_metrics = template.get("success", {})
        failure_metrics = template.get("failure", {})
        if not success_metrics and not failure_metrics:
            continue
        telemetry_rules[obj] = {
            "gates": {
                "success": {"metrics": success_metrics},
                "failure": {"metrics": failure_metrics},
            }
        }
    return telemetry_rules


def check_gates(phase, world, gates, elapsed):
    phase_name = phase.value if isinstance(phase, Phase) else phase
    telemetry_rules = world.learned_rules or {}
    process_rules = world.process_rules or {}

    brick_success = telemetry_brick.evaluate_success_gates(world, phase_name, telemetry_rules, process_rules)
    brick_fail = telemetry_brick.evaluate_failure_gates(world, phase_name, telemetry_rules, process_rules)
    wall_success = telemetry_wall.evaluate_success_gates(world, phase_name, world.wall_envelope)
    wall_fail = telemetry_wall.evaluate_failure_gates(world, phase_name, world.wall_envelope)
    robot_success = telemetry_robot_module.evaluate_success_gates(world, phase_name, telemetry_rules, process_rules)
    robot_fail = telemetry_robot_module.evaluate_failure_gates(world, phase_name, telemetry_rules, process_rules)

    success_ok = brick_success.ok and wall_success.ok and robot_success.ok
    reasons = []
    if not brick_fail.ok:
        reasons.extend(brick_fail.reasons)
    if not wall_fail.ok:
        reasons.extend(wall_fail.reasons)
    if not robot_fail.ok:
        reasons.extend(robot_fail.reasons)
    fail = bool(reasons)

    temporal = gates.get("temporal", {})
    success_time = temporal.get("success_time")
    if success_time and success_time.get("samples", 0) >= GATE_MIN_SAMPLES:
        min_time = success_time.get("min", 0.0)
        if elapsed < min_time:
            success_ok = False

    fail_time = temporal.get("fail_time")
    if fail_time and fail_time.get("samples", 0) >= GATE_MIN_SAMPLES:
        max_time = fail_time.get("max")
        if max_time is not None and elapsed >= max_time:
            fail = True
            reasons.append("time-to-fail")

    correction = not success_ok and not fail
    return GateResult(success=success_ok, correction=correction, fail=fail, reason="; ".join(reasons))


def _format_start_gate_reasons(reasons):
    return "; ".join(reasons) if reasons else ""


def evaluate_start_gates(phase, world, learned_gates):
    phase_name = phase.value if isinstance(phase, Phase) else phase
    telemetry_rules = world.learned_rules or _telemetry_rules_from_gates(learned_gates)
    process_rules = world.process_rules or {}

    brick_check = telemetry_brick.evaluate_start_gates(world, phase_name, telemetry_rules, process_rules)
    wall_check = telemetry_wall.evaluate_start_gates(world, phase_name, world.wall_envelope)
    robot_check = telemetry_robot_module.evaluate_start_gates(world, phase_name, telemetry_rules, process_rules)

    reasons = brick_check.reasons + wall_check.reasons + robot_check.reasons
    ok = brick_check.ok and wall_check.ok and robot_check.ok
    return {"ok": ok, "reasons": reasons}


def resolve_phase_gates(phase, learned):
    if phase in learned:
        return learned[phase]
    if phase == "ALIGN_BRICK":
        if "ALIGN_BRICK" in learned:
            return learned["ALIGN_BRICK"]
        if "ALIGN" in learned:
            return learned["ALIGN"]
        if "SCOOP" in learned:
            return learned["SCOOP"]
        if "FIND_BRICK" in learned:
            return learned["FIND_BRICK"]
        if "FIND" in learned:
            return learned["FIND"]
    return DEFAULT_GATE_TEMPLATES.get(phase, {})


def resolve_runtime_gates(phase, learned, process_rules):
    gates = resolve_phase_gates(phase, learned)
    obj_name = normalize_objective_label(phase)
    process_cfg = (process_rules or {}).get(obj_name) or {}
    if not isinstance(process_cfg, dict):
        return gates
    success = process_cfg.get("success_gates") or {}
    failure = process_cfg.get("fail_gates") or {}
    if success or failure:
        return {
            "success": success or gates.get("success", {}),
            "failure": failure or gates.get("failure", {}),
            "temporal": gates.get("temporal", {}),
        }
    return gates


def _fmt_gate_value(value):
    if value is None:
        return None
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def format_gate_summary(phase, gates):
    metrics = PHASE_METRICS.get(phase, ())
    if not metrics:
        return "success gate: none; fail gate: none; time gate: none"

    success_metrics = gates.get("success", {})
    failure_metrics = gates.get("failure", {})
    temporal = gates.get("temporal", {})

    success_parts = []
    fail_parts = []
    for metric in metrics:
        direction = METRIC_DIRECTIONS.get(metric)
        success_stats = success_metrics.get(metric) or {}
        failure_stats = failure_metrics.get(metric) or {}

        if direction == "low":
            success_max = success_stats.get("max")
            fail_max = failure_stats.get("max")
            if success_max is not None:
                success_parts.append(f"{metric}<={_fmt_gate_value(success_max)}")
            if fail_max is not None:
                fail_parts.append(f"{metric}>={_fmt_gate_value(fail_max)}")
        elif direction == "high":
            success_min = success_stats.get("min")
            fail_max = failure_stats.get("max")
            if success_min is not None:
                success_parts.append(f"{metric}>={_fmt_gate_value(success_min)}")
            if fail_max is not None:
                fail_parts.append(f"{metric}<={_fmt_gate_value(fail_max)}")

    success_desc = ", ".join(success_parts) if success_parts else "none"
    fail_desc = ", ".join(fail_parts) if fail_parts else "none"

    time_parts = []
    success_time = temporal.get("success_time")
    if success_time and success_time.get("samples", 0) >= GATE_MIN_SAMPLES:
        min_time = success_time.get("min")
        if min_time is not None:
            time_parts.append(f"success>={_fmt_gate_value(min_time)}s")
    fail_time = temporal.get("fail_time")
    if fail_time and fail_time.get("samples", 0) >= GATE_MIN_SAMPLES:
        max_time = fail_time.get("max")
        if max_time is not None:
            time_parts.append(f"fail<={_fmt_gate_value(max_time)}s")

    time_desc = ", ".join(time_parts) if time_parts else "none"
    return f"success gate: {success_desc}; fail gate: {fail_desc}; time gate: {time_desc}"


def _objective_process_config(phase_name, process_rules):
    obj_name = normalize_objective_label(phase_name)
    cfg = (process_rules or {}).get(obj_name)
    if isinstance(cfg, dict):
        return cfg
    return {}


def _format_gate_metric(metric, stats, direction, kind):
    if not stats or not isinstance(stats, dict):
        return None
    mu = stats.get("mu")
    sigma = stats.get("sigma")
    if mu is not None and sigma is not None:
        return f"{metric}~{_fmt_gate_value(mu)}+/-{_fmt_gate_value(sigma)}"
    min_val = stats.get("min")
    max_val = stats.get("max")
    if metric == "visible":
        if isinstance(min_val, bool):
            return f"{metric}={'true' if min_val else 'false'}"
        if isinstance(max_val, bool):
            return f"{metric}={'true' if max_val else 'false'}"
    if min_val is not None and max_val is not None:
        return f"{_fmt_gate_value(min_val)}<={metric}<={_fmt_gate_value(max_val)}"
    if min_val is not None:
        return f"{metric}>={_fmt_gate_value(min_val)}"
    if max_val is not None:
        if kind == "fail":
            if direction == "low":
                return f"{metric}>={_fmt_gate_value(max_val)}"
            if direction == "high":
                return f"{metric}<={_fmt_gate_value(max_val)}"
        return f"{metric}<={_fmt_gate_value(max_val)}"
    return None


def format_gate_metrics(phase_name, metrics, kind):
    if not metrics:
        return "none"
    metric_order = list(PHASE_METRICS.get(phase_name, ()))
    for metric in metrics:
        if metric not in metric_order:
            metric_order.append(metric)
    parts = []
    for metric in metric_order:
        if metric not in metrics:
            continue
        stats = metrics.get(metric) or {}
        entry = _format_gate_metric(metric, stats, METRIC_DIRECTIONS.get(metric), kind)
        if entry:
            parts.append(entry)
    return ", ".join(parts) if parts else "none"


def format_process_gate_lines(phase_name, process_rules):
    process_cfg = _objective_process_config(phase_name, process_rules)
    success_metrics = process_cfg.get("success_gates") or {}
    failure_metrics = process_cfg.get("fail_gates") or {}
    return (
        format_gate_metrics(phase_name, success_metrics, "success"),
        format_gate_metrics(phase_name, failure_metrics, "fail"),
    )


def _metric_snapshot_value(world, metric):
    brick = world.brick or {}
    if metric == "confidence":
        return brick.get("confidence")
    if metric == "lift_height":
        return getattr(world, "lift_height", None)
    return _metric_value(brick, metric)


def _format_metric_snapshot(metrics, world):
    parts = []
    for metric in metrics:
        val = _metric_snapshot_value(world, metric)
        if val is None:
            continue
        if metric == "visible":
            parts.append(f"{metric}={1 if val else 0}")
        elif metric in ("angle_abs", "offset_abs"):
            parts.append(f"{metric}={val:.1f}")
        elif metric in ("dist", "lift_height"):
            parts.append(f"{metric}={val:.1f}mm")
        elif metric == "confidence":
            parts.append(f"{metric}={val:.0f}")
        elif isinstance(val, (int, float)):
            parts.append(f"{metric}={val:.1f}")
        else:
            parts.append(f"{metric}={val}")
    return ", ".join(parts)


def format_start_gate_details(phase_name, world):
    metrics = []
    if phase_name in START_GATE_BRICK_PHASES:
        metrics = list(PHASE_METRICS.get(phase_name, ()))
        if metrics and "confidence" not in metrics:
            metrics.append("confidence")
    parts = []
    metric_detail = _format_metric_snapshot(metrics, world)
    if metric_detail:
        parts.append(metric_detail)
    wall = getattr(world, "wall", None) or {}
    if wall.get("origin") is not None and wall.get("valid", False):
        parts.append("wall=ok")
    if parts:
        return ", ".join(parts)
    return "ready"


def format_success_gate_details(phase_name, world, gates):
    metrics = list((gates or {}).get("success", {}).keys())
    if not metrics:
        metrics = list(PHASE_METRICS.get(phase_name, ()))
    detail = _format_metric_snapshot(metrics, world)
    return detail or "ok"


def _normalize_scan_direction(value):
    if value is None:
        return None
    key = str(value).strip().lower()
    if key in ("l", "left"):
        return "l"
    if key in ("r", "right"):
        return "r"
    return None


def _lookup_scan_pref(process_rules, phase_name):
    if not process_rules:
        return None
    obj_name = normalize_objective_label(phase_name)
    cfg = process_rules.get(obj_name)
    if isinstance(cfg, dict):
        scan_pref = _normalize_scan_direction(cfg.get("scan_direction"))
        if scan_pref:
            return scan_pref
    for key, cfg in process_rules.items():
        if normalize_objective_label(key) != obj_name:
            continue
        if not isinstance(cfg, dict):
            continue
        scan_pref = _normalize_scan_direction(cfg.get("scan_direction"))
        if scan_pref:
            return scan_pref
    return None


def resolve_scan_direction(world, phase):
    phase_name = phase.value if isinstance(phase, Phase) else str(phase)
    scan_pref = _lookup_scan_pref(getattr(world, "process_rules", None), phase_name)
    if scan_pref:
        return scan_pref
    if normalize_objective_label(phase_name) == "FIND_WALL":
        return "r"
    return "l"


def compute_alignment_command(world, gates, phase=None):
    brick = world.brick or {}
    if not brick.get("visible"):
        scan_direction = resolve_scan_direction(world, phase)
        return scan_direction, SCAN_SPEED, "scan"

    angle = brick.get("angle") or 0.0
    offset_x = brick.get("offset_x") or 0.0
    success_metrics = gates.get("success", {})
    angle_tol = success_metrics.get("angle_abs", {}).get("max", 5.0)
    offset_tol = success_metrics.get("offset_abs", {}).get("max", 12.0)

    if abs(angle) > angle_tol:
        cmd = "l" if angle > 0 else "r"
        return cmd, ALIGN_TURN_SPEED, "angle"
    if abs(offset_x) > offset_tol:
        cmd = "l" if offset_x > 0 else "r"
        return cmd, ALIGN_TURN_SPEED * 0.8, "offset"

    return "f", ALIGN_FORWARD_SPEED, "center"


def update_world_model(world, vision, stream_state=None, frame_callback=None):
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
    # Signature: update_vision(self, found, dist, angle, conf, ...)
    world.update_vision(found, dist, angle, conf, offset_x, cam_h, brick_above, brick_below)
    if stream_state is not None:
        with stream_state["lock"]:
            stream_state["frame"] = vision.current_frame
    if frame_callback is not None:
        frame_callback(vision.current_frame)


def build_stream_provider(stream_state):
    def _provider():
        with stream_state["lock"]:
            frame = stream_state.get("frame")
            if frame is None:
                return None
            return frame
    return _provider


def apply_motion(robot, world, cmd, speed, duration_s):
    robot.send_command(cmd, speed)
    evt = MotionEvent(_cmd_to_motion_type(cmd), int(speed * 255), int(duration_s * 1000))
    world.update_from_motion(evt)


def _cmd_to_motion_type(cmd):
    return {
        "f": "forward",
        "b": "backward",
        "l": "left_turn",
        "r": "right_turn",
        "u": "mast_up",
        "d": "mast_down",
    }.get(cmd, "wait")


def recover_phase(phase, world, robot, vision, stream_state, logger, learned_gates):
    start = time.time()
    logger.log_phase(phase, "RECOVER_START", world)
    gates = resolve_runtime_gates(phase, learned_gates, world.process_rules)
    step = 0
    good_frames = 0
    while time.time() - start < RECOVERY_MAX_S:
        update_world_model(world, vision, stream_state=stream_state)
        gate_eval = check_gates(phase, world, gates, time.time() - start)
        if not gate_eval.fail and (gate_eval.success or gate_eval.correction):
            good_frames += 1
            if good_frames >= GATE_STABILITY_FRAMES:
                logger.log_phase(phase, "RECOVER_END", world, status="success")
                return True
        else:
            good_frames = 0

        cmd = "l" if step % 2 == 0 else "b"
        speed = RECOVERY_TURN_SPEED if cmd == "l" else RECOVERY_BACK_SPEED
        step_start = time.time()
        while time.time() - step_start < RECOVERY_STEP_S:
            update_world_model(world, vision, stream_state=stream_state)
            apply_motion(robot, world, cmd, speed, CONTROL_DT)
            time.sleep(CONTROL_DT)
        step += 1

    logger.log_phase(phase, "RECOVER_END", world, status="fail")
    return False


def build_commit_profile(success_segments, success_gates):
    if not success_segments or not success_gates:
        return {}
    time_vals = []
    dist_vals = []
    speed_vals = []

    for seg in success_segments:
        obj = normalize_objective_label(seg.get("objective"))
        if obj not in ("SCOOP", "ALIGN_BRICK", "ALIGN"):
            continue
        last_aligned = None
        for state in seg.get("states", []):
            brick = state.get("brick") or {}
            if not brick.get("visible"):
                continue
            if _within_success(brick, success_gates):
                last_aligned = state
        if last_aligned and seg.get("end"):
            dt = seg["end"] - last_aligned.get("timestamp", seg["end"])
            if dt >= 0:
                time_vals.append(dt)
            dist = brick_dist(last_aligned)
            if dist is not None:
                dist_vals.append(dist)
            speed = average_forward_speed(seg.get("events", []), last_aligned.get("timestamp"), seg.get("end"))
            if speed is not None:
                speed_vals.append(speed)

    return {
        "time_s": percentile(time_vals, 0.9) if time_vals else None,
        "max_dist": percentile(dist_vals, 0.9) if dist_vals else None,
        "speed": percentile(speed_vals, 0.9) if speed_vals else None,
    }


def brick_dist(state):
    brick = state.get("brick") or {}
    dist = brick.get("dist")
    if dist is None or dist <= 0:
        return None
    return float(dist)


def _within_success(brick, success_gates):
    for metric, stats in success_gates.items():
        val = _metric_value(brick, metric)
        if val is None:
            return False
        direction = METRIC_DIRECTIONS.get(metric)
        if direction == "low" and stats.get("max") is not None and val > stats.get("max"):
            return False
        if direction == "high" and stats.get("min") is not None and val < stats.get("min"):
            return False
    return True


def learn_blind_window(success_segments):
    times = []
    for seg in success_segments:
        obj = normalize_objective_label(seg.get("objective"))
        if obj != "SCOOP":
            continue
        last_visible = None
        for state in seg.get("states", []):
            brick = state.get("brick") or {}
            if brick.get("visible"):
                last_visible = state.get("timestamp")
        if last_visible and seg.get("end") and seg.get("end") >= last_visible:
            times.append(seg["end"] - last_visible)
    return percentile(times, 0.9) if times else None


def format_motion_plan(plan):
    if not plan:
        return "none"
    parts = []
    for primitive in plan:
        parts.append(
            f"{primitive.label}({primitive.cmd} {primitive.speed:.2f} for {primitive.duration_s:.2f}s)"
        )
    return ", ".join(parts)

def _brick_height_adjust_s(world, speed):
    if world is None:
        return 0.0
    height = getattr(world, "height_mm", None)
    if height is None or height < BRICK_HEIGHT_MIN_MM:
        return 0.0
    height = min(height, BRICK_HEIGHT_MAX_MM)
    mm_per_s = world.lift_mm_per_sec * max(0.1, speed)
    return height / mm_per_s


def _adjust_plan_for_brick_height(plan, world):
    if not plan or world is None:
        return plan
    for primitive in plan:
        if primitive.cmd not in ("u", "d"):
            continue
        adjust_s = _brick_height_adjust_s(world, primitive.speed)
        if adjust_s <= 0:
            continue
        max_adjust = primitive.duration_s * BRICK_HEIGHT_MAX_ADJUST_FRACTION
        new_duration = primitive.duration_s - min(adjust_s, max_adjust)
        primitive.duration_s = max(BRICK_HEIGHT_MIN_DURATION_S, new_duration)
    return plan


def motion_plan_for_phase(phase, learned_profiles, world=None):
    if phase == "SCOOP":
        profile = learned_profiles.get("SCOOP", {})
        plan = [
            MotionPrimitive("f", profile.get("speed", SCOOP_DRIVE_SPEED), profile.get("drive_time", SCOOP_DRIVE_TIME), "SCOOP_DRIVE"),
            MotionPrimitive("u", profile.get("lift_speed", SCOOP_LIFT_SPEED), profile.get("lift_time", SCOOP_LIFT_TIME), "SCOOP_LIFT"),
        ]
        return _adjust_plan_for_brick_height(plan, world)
    if phase == "LIFT":
        profile = learned_profiles.get("LIFT", {})
        plan = [MotionPrimitive("u", profile.get("speed", LIFT_SPEED), profile.get("duration", LIFT_TIME), "LIFT")]
        return _adjust_plan_for_brick_height(plan, world)
    if phase == "FIND_WALL2":
        profile = learned_profiles.get("FIND_WALL2", {}) or learned_profiles.get("CARRY", {})
        return [MotionPrimitive("f", profile.get("speed", FIND_WALL2_SPEED), profile.get("duration", FIND_WALL2_TIME), "FIND_WALL2")]
    if phase == "POSITION_BRICK":
        profile = learned_profiles.get("POSITION_BRICK", {})
        return [MotionPrimitive("f", profile.get("speed", POSITION_BRICK_SPEED), profile.get("duration", POSITION_BRICK_TIME), "POSITION_BRICK")]
    if phase == "PLACE":
        profile = learned_profiles.get("PLACE", {})
        plan = [MotionPrimitive("d", profile.get("speed", PLACE_SPEED), profile.get("duration", PLACE_TIME), "PLACE")]
        return _adjust_plan_for_brick_height(plan, world)
    if phase == "RETREAT":
        profile = learned_profiles.get("RETREAT", {})
        return [MotionPrimitive("b", profile.get("speed", RETREAT_SPEED), profile.get("duration", RETREAT_TIME), "RETREAT")]
    return []


def _learn_nominal_profiles(nominal_segments):
    if not nominal_segments:
        return {}

    def mean_or_none(values):
        if not values:
            return None
        return statistics.mean(values)

    per_obj = {}
    for seg in nominal_segments:
        obj = normalize_objective_label(seg.get("objective"))
        if not obj:
            continue
        summary = _segment_motion_summary(seg.get("events") or [])
        if not summary:
            continue
        obj_stats = per_obj.setdefault(obj, {})
        for cmd, stats in summary.items():
            cmd_stats = obj_stats.setdefault(cmd, {"durations": [], "speeds": []})
            cmd_stats["durations"].append(stats["duration"])
            cmd_stats["speeds"].append(stats["speed"])

    profiles = {}
    for obj, cmd_stats in per_obj.items():
        profile = {}
        if obj == "SCOOP":
            if "f" in cmd_stats:
                speed = mean_or_none(cmd_stats["f"]["speeds"])
                duration = mean_or_none(cmd_stats["f"]["durations"])
                if speed is not None:
                    profile["speed"] = speed
                if duration is not None:
                    profile["drive_time"] = duration
            if "u" in cmd_stats:
                speed = mean_or_none(cmd_stats["u"]["speeds"])
                duration = mean_or_none(cmd_stats["u"]["durations"])
                if speed is not None:
                    profile["lift_speed"] = speed
                if duration is not None:
                    profile["lift_time"] = duration
        elif obj in ("FIND_WALL2", "POSITION_BRICK"):
            if "f" in cmd_stats:
                speed = mean_or_none(cmd_stats["f"]["speeds"])
                duration = mean_or_none(cmd_stats["f"]["durations"])
                if speed is not None:
                    profile["speed"] = speed
                if duration is not None:
                    profile["duration"] = duration
        elif obj == "LIFT":
            if "u" in cmd_stats:
                speed = mean_or_none(cmd_stats["u"]["speeds"])
                duration = mean_or_none(cmd_stats["u"]["durations"])
                if speed is not None:
                    profile["speed"] = speed
                if duration is not None:
                    profile["duration"] = duration
        elif obj == "PLACE":
            if "d" in cmd_stats:
                speed = mean_or_none(cmd_stats["d"]["speeds"])
                duration = mean_or_none(cmd_stats["d"]["durations"])
                if speed is not None:
                    profile["speed"] = speed
                if duration is not None:
                    profile["duration"] = duration
        elif obj == "RETREAT":
            if "b" in cmd_stats:
                speed = mean_or_none(cmd_stats["b"]["speeds"])
                duration = mean_or_none(cmd_stats["b"]["durations"])
                if speed is not None:
                    profile["speed"] = speed
                if duration is not None:
                    profile["duration"] = duration
        if profile:
            profiles[obj] = profile

    return profiles


def learn_motion_profiles(success_segments, commit_profile, nominal_segments=None):
    profiles = {}
    duration_acc = {}

    for seg in success_segments:
        obj = normalize_objective_label(seg.get("objective"))
        if not obj or seg.get("start") is None or seg.get("end") is None:
            continue
        duration = seg["end"] - seg["start"]
        if duration <= 0:
            continue
        duration_acc.setdefault(obj, []).append(duration)

    for obj, durations in duration_acc.items():
        profiles.setdefault(obj, {})
        profiles[obj]["duration"] = percentile(durations, 0.9) or durations[-1]

    if commit_profile:
        scoop_profile = profiles.setdefault("SCOOP", {})
        if commit_profile.get("time_s") and "drive_time" not in scoop_profile:
            scoop_profile["drive_time"] = commit_profile["time_s"]
        if commit_profile.get("speed") and "speed" not in scoop_profile:
            scoop_profile["speed"] = commit_profile["speed"]

    nominal_profiles = _learn_nominal_profiles(nominal_segments)
    for obj, profile in nominal_profiles.items():
        profiles.setdefault(obj, {}).update(profile)

    return profiles


def map_phase_to_objective(phase):
    phase_name = phase.value if isinstance(phase, Phase) else str(phase)
    if phase_name == "FIND_WALL":
        return ObjectiveState.FIND_WALL
    if phase_name == "EXIT_WALL":
        return ObjectiveState.EXIT_WALL
    if phase_name == "FIND_BRICK":
        return ObjectiveState.FIND_BRICK
    if phase_name == "ALIGN_BRICK":
        return ObjectiveState.SCOOP
    if phase_name == "SCOOP":
        return ObjectiveState.SCOOP
    if phase_name == "LIFT":
        return ObjectiveState.LIFT
    if phase_name == "PLACE":
        return ObjectiveState.PLACE
    if phase_name == "FIND_WALL2":
        return ObjectiveState.LIFT
    if phase_name == "POSITION_BRICK":
        return ObjectiveState.LIFT
    if phase_name == "RETREAT":
        return ObjectiveState.PLACE
    try:
        return ObjectiveState(phase_name)
    except ValueError:
        return ObjectiveState.FIND_BRICK


def _frame_label(count):
    return "frame" if count == 1 else "frames"


def confirm_start_gate(phase_name, world, vision, stream_state, learned_gates, required_frames=GATE_STABILITY_FRAMES):
    required = max(1, int(required_frames))
    details = ""
    for idx in range(required):
        update_world_model(world, vision, stream_state=stream_state)
        start_eval = evaluate_start_gates(phase_name, world, learned_gates)
        if not start_eval["ok"]:
            reason = _format_start_gate_reasons(start_eval["reasons"])
            return False, reason
        details = format_start_gate_details(phase_name, world)
        if idx < required - 1:
            time.sleep(CONTROL_DT)
    return True, details


def run_phase_dynamic(phase, world, robot, vision, stream_state, logger, learned_gates):
    phase_name = phase.value
    start_ok, start_info = confirm_start_gate(phase_name, world, vision, stream_state, learned_gates)
    if not start_ok:
        reason = start_info
        details = f": {reason}" if reason else ""
        print(format_headline(f"[START] {phase_name} start criteria failed", COLOR_RED, details))
        return PhaseResult(False, f"start gate: {reason}", 0.0, start_gate_failed=True)
    start_details = start_info
    frame_label = _frame_label(GATE_STABILITY_FRAMES)
    detail_suffix = f"; {start_details}" if start_details else ""
    print(format_headline(
        f"[START] {phase_name} start criteria stable ({GATE_STABILITY_FRAMES} {frame_label}{detail_suffix})",
        COLOR_GREEN,
    ))
    gates = resolve_runtime_gates(phase_name, learned_gates, world.process_rules)
    start = time.time()
    logger.log_phase(phase_name, "PHASE_START", world)
    last_state_log = 0.0
    
    # Track gate events for terminal highlights
    failure_announced = False
    success_frames = 0
    last_action = None
    last_cmd = None

    while True:
        loop_start = time.time()
        update_world_model(world, vision, stream_state=stream_state)
        elapsed = loop_start - start
        gate_eval = check_gates(phase_name, world, gates, elapsed)
        
        # Highlight first time we meet failure criteria
        if gate_eval.fail and not failure_announced:
            details = f" ({gate_eval.reason})" if gate_eval.reason else ""
            print(format_headline(f"--- {phase_name}: FAILURE criteria met", COLOR_RED, details))
            failure_announced = True

        if gate_eval.fail:
            logger.log_phase(phase_name, "PHASE_END", world, status="fail", reason=gate_eval.reason)
            return PhaseResult(False, gate_eval.reason or "gate", elapsed)
        if gate_eval.success:
            success_frames += 1
            if success_frames >= GATE_STABILITY_FRAMES:
                success_details = format_success_gate_details(phase_name, world, gates)
                frame_label = _frame_label(GATE_STABILITY_FRAMES)
                detail_suffix = f"; {success_details}" if success_details else ""
                print(format_headline(
                    f"--- {phase_name}: SUCCESS criteria stable ({GATE_STABILITY_FRAMES} {frame_label}{detail_suffix})",
                    COLOR_GREEN,
                ))
                logger.log_phase(phase_name, "PHASE_END", world, status="success")
                return PhaseResult(True, "", elapsed)
        else:
            success_frames = 0

        cmd, speed, action = compute_alignment_command(world, gates, phase_name)
        if action != last_action or cmd != last_cmd:
            log_action(phase_name, action, cmd=cmd, speed=speed)
            last_action = action
            last_cmd = cmd
        apply_motion(robot, world, cmd, speed, CONTROL_DT)

        if time.time() - last_state_log >= STATE_LOG_INTERVAL:
            logger.log_state(phase_name, world)
            last_state_log = time.time()

        sleep_time = CONTROL_DT - (time.time() - loop_start)
        if sleep_time > 0:
            time.sleep(sleep_time)


def run_phase_motion(phase, world, robot, vision, stream_state, logger, learned_gates, blind_window_s=None):
    phase_name = phase.value
    start_ok, start_info = confirm_start_gate(phase_name, world, vision, stream_state, learned_gates)
    if not start_ok:
        reason = start_info
        details = f": {reason}" if reason else ""
        print(format_headline(f"[START] {phase_name} start criteria failed", COLOR_RED, details))
        return PhaseResult(False, f"start gate: {reason}", 0.0, start_gate_failed=True)
    start_details = start_info
    frame_label = _frame_label(GATE_STABILITY_FRAMES)
    detail_suffix = f"; {start_details}" if start_details else ""
    print(format_headline(
        f"[START] {phase_name} start criteria stable ({GATE_STABILITY_FRAMES} {frame_label}{detail_suffix})",
        COLOR_GREEN,
    ))
    plan = motion_plan_for_phase(phase_name, learned_gates.get("profiles", {}), world)
    if not plan:
        logger.log_phase(phase_name, "PHASE_START", world)
        logger.log_phase(phase_name, "PHASE_END", world, status="success")
        return PhaseResult(True, "", 0.0)

    logger.log_phase(phase_name, "PHASE_START", world)
    start = time.time()
    last_state_log = 0.0

    for primitive in plan:
        log_action(phase_name, primitive.label, cmd=primitive.cmd, speed=primitive.speed, duration=primitive.duration_s)
        step_start = time.time()
        while time.time() - step_start < primitive.duration_s:
            update_world_model(world, vision, stream_state=stream_state)
            if phase_name == "SCOOP" and blind_window_s is not None:
                if world.last_visible_time is not None:
                    if time.time() - world.last_visible_time > blind_window_s:
                        logger.log_phase(phase_name, "PHASE_END", world, status="fail", reason="blind-window")
                        return PhaseResult(False, "blind-window", time.time() - start)
            apply_motion(robot, world, primitive.cmd, primitive.speed, CONTROL_DT)
            if time.time() - last_state_log >= STATE_LOG_INTERVAL:
                logger.log_state(phase_name, world)
                last_state_log = time.time()
            time.sleep(CONTROL_DT)

    logger.log_phase(phase_name, "PHASE_END", world, status="success")
    return PhaseResult(True, "", time.time() - start)


def save_gates_to_process_model(learned_gates, success_segments=None, filepath=None, attempt_types_by_objective=None, scan_prefs_by_objective=None):
    """Save learned gates to world_model_process.json for persistence and reuse."""
    if filepath is None:
        filepath = Path(__file__).parent / "world_model_process.json"
    
    # Load existing model or create new structure
    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                model = json.load(f)
        except:
            model = {"objectives": {}}
    else:
        model = {"objectives": {}}
    
    # Helpers to round values and coerce visibility gates to boolean
    def round_value(value):
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return round(value, 2)
        return value

    def visible_bool_from_stats(stats):
        if not stats:
            return None
        for key in ("mu", "min", "max"):
            if key in stats and stats[key] is not None:
                return bool(stats[key] >= 0.5)
        return None

    def normalize_gate_dict(gates, is_failure=False):
        normalized = {}
        for metric, stats in (gates or {}).items():
            if metric == "visible":
                visible_required = visible_bool_from_stats(stats)
                if visible_required is None:
                    continue
                if is_failure:
                    if visible_required is False:
                        normalized[metric] = {"max": False}
                else:
                    normalized[metric] = {"min": visible_required}
                continue
            normalized[metric] = {}
            for key, value in (stats or {}).items():
                normalized[metric][key] = round_value(value)
        return normalized

    def prune_metrics_when_invisible(gates):
        if not gates:
            return gates
        visible_gate = gates.get("visible")
        if not isinstance(visible_gate, dict):
            return gates
        if visible_gate.get("min") is False or visible_gate.get("max") is False:
            return {"visible": visible_gate}
        return gates
    
    # Derive start_gates from success segments
    start_conditions = {}
    if success_segments:
        for seg in success_segments:
            obj = normalize_objective_label(seg.get("objective"))
            if not obj:
                continue
            states = seg.get("states", [])
            if states:
                # Get the first state to understand start conditions
                first_state = states[0]
                brick = first_state.get("brick", {})
                if obj not in start_conditions:
                    start_conditions[obj] = {"visible_count": 0, "total": 0}
                start_conditions[obj]["total"] += 1
                if brick.get("visible"):
                    start_conditions[obj]["visible_count"] += 1

    derived_start_gates = {}
    for obj_name, cond in start_conditions.items():
        total = cond.get("total", 0)
        if total <= 0:
            continue
        visible_ratio = cond.get("visible_count", 0) / total
        if visible_ratio >= 0.5:
            derived_start_gates[obj_name] = {"visible": {"min": True}}
    if attempt_types_by_objective is None:
        attempt_types_by_objective = {}
    if scan_prefs_by_objective is None:
        scan_prefs_by_objective = {}

    objective_order = list(model["objectives"].keys()) if model["objectives"] else list(PHASE_SEQUENCE)
    
    # Update each objective with learned gates
    objectives = set(model["objectives"].keys())
    objectives.update(obj for obj in learned_gates.keys() if obj != "profiles")
    objectives.update(derived_start_gates.keys())
    for obj_name in objectives:
        gates = learned_gates.get(obj_name, {})
        
        if obj_name not in model["objectives"]:
            model["objectives"][obj_name] = {}
        
        obj_config = model["objectives"][obj_name]
        
        # Convert learned gates to process model format
        success_metrics = gates.get("success", {}) if gates else {}
        failure_metrics = gates.get("failure", {}) if gates else {}
        
        # Save success gates (using max for upper bounds)
        if success_metrics:
            obj_config["success_gates"] = {}
            for metric, stats in success_metrics.items():
                if metric == "visible":
                    visible_required = visible_bool_from_stats(stats)
                    if visible_required is None:
                        continue
                    obj_config["success_gates"][metric] = {"min": visible_required}
                    continue
                obj_config["success_gates"][metric] = {}
                if "max" in stats:
                    obj_config["success_gates"][metric]["max"] = round_value(stats["max"])
                if "min" in stats:
                    obj_config["success_gates"][metric]["min"] = round_value(stats["min"])
        
        # Save failure gates
        if failure_metrics:
            obj_config["fail_gates"] = {}
            for metric, stats in failure_metrics.items():
                if metric == "visible":
                    visible_failure = visible_bool_from_stats(stats)
                    if visible_failure is False:
                        obj_config["fail_gates"][metric] = {"max": False}
                    continue
                if "mu" in stats and "sigma" in stats:
                    # Save mu and sigma for pattern matching
                    obj_config["fail_gates"][metric] = {
                        "mu": round_value(stats["mu"]),
                        "sigma": round_value(stats["sigma"])
                    }
                elif "max" in stats:
                    obj_config["fail_gates"][metric] = {"max": round_value(stats["max"])}
        
        # Derive and save start_gates
        obj_config["start_gates"] = derived_start_gates.get(obj_name, {})

        # Normalize any existing gate values (rounding + visibility booleans)
        obj_config["success_gates"] = normalize_gate_dict(obj_config.get("success_gates", {}))
        obj_config["fail_gates"] = normalize_gate_dict(obj_config.get("fail_gates", {}), is_failure=True)
        obj_config["start_gates"] = normalize_gate_dict(obj_config.get("start_gates", {}))

        obj_config["success_gates"] = prune_metrics_when_invisible(obj_config.get("success_gates", {}))
        obj_config["fail_gates"] = prune_metrics_when_invisible(obj_config.get("fail_gates", {}))
        obj_config["start_gates"] = prune_metrics_when_invisible(obj_config.get("start_gates", {}))

        if obj_name in attempt_types_by_objective:
            attempt_types = attempt_types_by_objective[obj_name]
            nominal_only = attempt_types == {"NOMINAL"}
            if nominal_only:
                obj_config["nominalDemosOnly"] = True
            else:
                obj_config.pop("nominalDemosOnly", None)
        if scan_prefs_by_objective is not None and obj_name in scan_prefs_by_objective:
            scan_dir = scan_prefs_by_objective.get(obj_name)
            if scan_dir in ("l", "r"):
                obj_config["scan_direction"] = scan_dir

        for key in ("start_gates", "success_gates", "fail_gates"):
            if key in obj_config and not obj_config[key]:
                obj_config.pop(key)

    ordered_objectives = {}
    for name in objective_order:
        if name in model["objectives"]:
            ordered_objectives[name] = model["objectives"][name]
    for name in sorted(model["objectives"]):
        if name not in ordered_objectives:
            ordered_objectives[name] = model["objectives"][name]
    model["objectives"] = ordered_objectives
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(model, f, indent=4)
    
    print(f"[GATES] Saved learned gates to {filepath}")


def run_autostack(session_name=None, stream=True):
    logs = load_demo_logs(DEMO_DIR, session_name)
    summarize_demo_stats(logs)
    learned_gates, success_segments = learn_gates_from_logs(logs)
    nominal_segments = collect_attempt_segments(logs, "NOMINAL")
    attempt_types_by_objective = collect_attempt_types_by_objective(logs)
    scan_prefs_by_objective = learn_scan_preferences(logs)
    
    # Save learned gates to world_model_process.json for persistence
    save_gates_to_process_model(
        learned_gates,
        success_segments,
        attempt_types_by_objective=attempt_types_by_objective,
        scan_prefs_by_objective=scan_prefs_by_objective,
    )
    
    align_gates = resolve_phase_gates("ALIGN_BRICK", learned_gates)
    commit_profile = build_commit_profile(success_segments, align_gates.get("success", {}))
    blind_window = learn_blind_window(success_segments) or BLIND_WINDOW_FALLBACK_S

    motion_profiles = learn_motion_profiles(success_segments, commit_profile, nominal_segments=nominal_segments)
    learned_gates["profiles"] = motion_profiles
    telemetry_rules = _telemetry_rules_from_gates(learned_gates)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = DEMO_DIR / f"autostack_{timestamp}.json"
    logger = PhaseLogger(log_path)
    print(f"[SESSION] Recording to {log_path}")

    stream_state = {"frame": None, "lock": threading.Lock()}
    if stream:
        StreamServer(
            build_stream_provider(stream_state),
            host=STREAM_HOST,
            port=WEB_PORT,
            fps=STREAM_FPS,
            jpeg_quality=STREAM_JPEG_QUALITY,
            title="Robot Leia - Autostack",
            header="Robot Leia - Autostack",
            sharpen=True,
        ).start()

    robot = Robot()
    vision = ArucoBrickVision(debug=False)
    world = WorldModel()
    world.learned_rules = telemetry_rules

    try:
        process_sequence = load_process_sequence()
        dynamic_phases = {"FIND_WALL", "EXIT_WALL", "FIND_BRICK", "ALIGN_BRICK"}
        for idx, phase_name in enumerate(process_sequence):
            try:
                phase = Phase(phase_name)
            except ValueError:
                phase = phase_name
            world.objective_state = map_phase_to_objective(phase)
            attempts = 0
            while attempts < MAX_PHASE_ATTEMPTS:
                success_desc, fail_desc = format_process_gate_lines(phase_name, world.process_rules)
                if phase_name in dynamic_phases:
                    print(f"[PHASE] Attempting {phase_name}")
                    print(f"  attempt: {attempts + 1}/{MAX_PHASE_ATTEMPTS}")
                    print(f"  success gate: {success_desc}")
                    print(f"  fail gate: {fail_desc}")
                    result = run_phase_dynamic(phase, world, robot, vision, stream_state, logger, learned_gates)
                else:
                    plan = motion_plan_for_phase(phase_name, learned_gates.get("profiles", {}), world)
                    plan_summary = format_motion_plan(plan)
                    print(f"[PHASE] Attempting {phase_name} (motion: {plan_summary})")
                    print(f"  attempt: {attempts + 1}/{MAX_PHASE_ATTEMPTS}")
                    print(f"  success gate: {success_desc}")
                    print(f"  fail gate: {fail_desc}")
                    result = run_phase_motion(phase, world, robot, vision, stream_state, logger, learned_gates, blind_window_s=blind_window)

                if result.start_gate_failed:
                    return
                if result.success:
                    break
                if phase_name in ("FIND_BRICK", "ALIGN_BRICK"):
                    recovered = recover_phase(phase_name, world, robot, vision, stream_state, logger, learned_gates)
                    if recovered:
                        attempts += 1
                        continue
                attempts += 1
                if attempts >= MAX_PHASE_ATTEMPTS:
                    details = f" after {attempts} attempts"
                    print(format_headline(f"[FAIL] {phase_name} failed", COLOR_RED, details))
                    return

        print(format_headline("[JOB] SUCCESS", COLOR_GREEN))
    finally:
        robot.stop()
        vision.close()
        logger.close()


def main():
    parser = argparse.ArgumentParser(description="Robot Leia Autostack")
    parser.add_argument("--session", help="demo session file or folder", default=None)
    parser.add_argument("--no-stream", action="store_true", help="disable livestream")
    args = parser.parse_args()

    run_autostack(session_name=args.session, stream=not args.no_stream)


if __name__ == "__main__":
    main()
