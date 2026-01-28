import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
from helper_demo_log_utils import extract_attempt_segments, load_demo_logs
from telemetry_robot import (
    manual_speed_for_cmd,
    SPEED_SCORE_DEFAULT,
    SPEED_SCORE_MIN,
    SPEED_SCORE_MAX,
    normalize_speed_score,
    ALIGN_MIN_SPEED,
    ALIGN_MAX_SPEED,
    ALIGN_MICRO_SPEED,
    ALIGN_FIXED_SPEED,
    ALIGN_SPEED_MIN_POWER,
    ALIGN_SPEED_SLOW,
    ALIGN_SPEED_NORMAL,
    ALIGN_SPEED_FAST,
    ALIGN_SPEED_SLOW_MM,
    ALIGN_SPEED_FAST_MM,
    ALIGN_MICRO_OFFSET_MM,
    ALIGN_MICRO_ANGLE_DEG,
    COMBINED_MOVEMENTS,
)
START_GATE_MIN_CONFIDENCE = 25.0
ALIGN_CONFIDENCE_MIN = 25.0
COMBINED_MOVEMENT_THRESHOLD_MM = 5.0  # Use combined movements when lateral offset > 5mm
COMBINED_MOVEMENT_FAST_THRESHOLD_MM = 20.0  # Use fast combined movements when distance > 20mm
VISIBILITY_LOST_GRACE_S = 0.5
BRICK_SMOOTH_FRAMES = 3
BRICK_SMOOTH_OUTLIER_MM = 12.0
BRICK_SMOOTH_OUTLIER_DEG = 6.0
VISIBLE_FALSE_GRACE_S_BY_STEP = {
    "EXIT_WALL": 1.0,
}

MANUAL_TURN_SPEED = manual_speed_for_cmd("l", SPEED_SCORE_DEFAULT)

STEP_ALIASES = {
    "FIND": "FIND_BRICK",
    "ALIGN": "ALIGN_BRICK",
    "CARRY": "FIND_WALL2",
}

METRICS_BY_STEP = {
    "FIND_WALL": ("angle_abs", "xAxis_offset_abs", "dist", "visible"),
    "EXIT_WALL": ("angle_abs", "xAxis_offset_abs", "dist", "visible"),
    "FIND_BRICK": ("angle_abs", "xAxis_offset_abs", "dist", "visible"),
    "ALIGN_BRICK": ("angle_abs", "xAxis_offset_abs", "dist", "visible"),
    "SCOOP": ("angle_abs", "xAxis_offset_abs", "dist", "visible"),
    "POSITION_BRICK": ("angle_abs", "xAxis_offset_abs", "dist", "visible"),
}

METRIC_DIRECTIONS = {
    "angle_abs": "low",
    "xAxis_offset_abs": "low",
    "dist": "low",
    "visible": "high",
    "confidence": "high",
}

VISIBILITY_REQUIRED_METRICS = {"angle_abs", "xAxis_offset_abs", "dist"}


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


def _filtered_brick_frame_average(frames, min_frames=BRICK_SMOOTH_FRAMES):
    if len(frames) < min_frames:
        return None, False, None

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

    if len(keep) < min_frames:
        return None, True, f"{len(keep)}/{min_frames} tight variance frames (inconsistent)"
    return _average_brick_frames(keep), True, None


def _score_from_mm(mm_off, slow_mm, fast_mm):
    if mm_off is None:
        return SPEED_SCORE_DEFAULT
    if mm_off <= slow_mm:
        return SPEED_SCORE_MIN
    if mm_off >= fast_mm:
        return SPEED_SCORE_MAX
    return SPEED_SCORE_DEFAULT


def smoothed_brick_snapshot(world):
    brick = world.brick or {}
    buffer = getattr(world, "_brick_frame_buffer", None)
    if not buffer:
        return brick
    avg, _, _ = _filtered_brick_frame_average(buffer)
    if avg is None:
        return brick
    return {
        "visible": bool(avg.get("found")),
        "dist": float(avg.get("dist", 0.0)),
        "angle": float(avg.get("angle", 0.0)),
        "confidence": float(avg.get("conf", 0.0)),
        "offset_x": float(avg.get("offset_x", 0.0)),
        "x_axis": float(avg.get("offset_x", 0.0)),
        "brickAbove": bool(avg.get("brick_above")),
        "brickBelow": bool(avg.get("brick_below")),
    }


@dataclass
class GateCheck:
    ok: bool
    reasons: List[str] = field(default_factory=list)

    def reason_str(self):
        return "; ".join(self.reasons) if self.reasons else ""


def combine_gate_checks(*checks):
    ok = True
    reasons = []
    for check in checks:
        if check is None:
            continue
        if not check.ok:
            ok = False
            reasons.extend(check.reasons)
    return GateCheck(ok=ok, reasons=reasons)


def _step_name(step):
    if hasattr(step, "value"):
        name = step.value
    else:
        name = str(step)
    key = name.strip().upper()
    return STEP_ALIASES.get(key, key)


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    pct = max(0.0, min(1.0, pct))
    idx = int(round(pct * (len(values) - 1)))
    return values[idx]


def metric_direction_for_step(metric, step):
    direction = METRIC_DIRECTIONS.get(metric)
    obj_name = _step_name(step)
    if obj_name == "FIND_BRICK" and metric == "dist":
        return None
    return direction


def metric_value(brick, metric):
    if metric == "angle_abs":
        return abs(brick.get("angle", 0.0))
    if metric == "xAxis_offset_abs":
        return brick.get("x_axis", brick.get("offset_x", 0.0))
    if metric == "dist":
        return brick.get("dist", 0.0)
    if metric == "visible":
        return 1.0 if brick.get("visible") else 0.0
    if metric == "confidence":
        return brick.get("confidence", 0.0)
    return None


def _effective_visible(world, visible, grace_s=VISIBILITY_LOST_GRACE_S):
    if visible:
        return True
    last_seen = getattr(world, "last_visible_time", None)
    if last_seen is None:
        return False
    return (time.time() - last_seen) <= grace_s


def metric_status(value, success_stats, failure_stats, direction):
    if success_stats is None or direction is None:
        return "unknown"
    if direction == "low":
        success_max = success_stats.get("max")
        failure_max = failure_stats.get("max") if failure_stats else None
        if success_max is not None and value <= success_max:
            return "success"
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


def _target_tol_ok(value, stats, direction):
    target = stats.get("target") if isinstance(stats, dict) else None
    tol = stats.get("tol") if isinstance(stats, dict) else None
    if target is None or tol is None:
        return None
    if direction == "high":
        return value >= (target - tol)
    if direction == "low":
        return value <= (target + tol)
    return abs(value - target) <= tol


def success_metric_bounds(stats, direction):
    if not isinstance(stats, dict) or direction is None:
        return None, None
    target = stats.get("target")
    tol = stats.get("tol")
    if target is not None and tol is not None:
        if direction == "low":
            return None, target + tol
        if direction == "high":
            return target - tol, None
        return target - tol, target + tol
    return stats.get("min"), stats.get("max")


def compute_brick_world_xy(world, dist, angle_deg):
    heading = math.radians(world.theta + angle_deg)
    return (
        world.x + (dist * math.cos(heading)),
        world.y + (dist * math.sin(heading)),
    )


def get_scoop_corridor_limits(world, dist):
    corridor = world.learned_rules.get("SCOOP", {}).get("corridor")
    if not corridor or dist is None:
        return None
    for row in corridor:
        dist_min = row.get("dist_min", 0)
        dist_max = row.get("dist_max", 0)
        if dist_min <= dist < dist_max:
            return row
    if dist < corridor[0].get("dist_min", 0):
        return corridor[0]
    return corridor[-1]


def build_envelope(process_rules, learned_rules, step):
    obj_name = _step_name(step)
    process = (process_rules or {}).get(obj_name, {})
    learned = learned_rules.get(obj_name, {}) if learned_rules else {}
    learned_gates = learned.get("gates", {})
    success = process.get("success_gates") or learned_gates.get("success", {}).get("metrics", {})
    failure = process.get("fail_gates") or learned_gates.get("failure", {}).get("metrics", {})
    return {"success": success, "failure": failure}


def success_gate_bounds(process_rules, learned_rules, step):
    envelope = build_envelope(process_rules or {}, learned_rules or {}, step)
    success_metrics = envelope.get("success") or {}
    bounds = {}
    for metric, stats in success_metrics.items():
        direction = metric_direction_for_step(metric, step)
        min_val, max_val = success_metric_bounds(stats, direction)
        bounds[metric] = {"min": min_val, "max": max_val}
    return bounds


def success_gates_visible_only(process_rules, step):
    obj_name = _step_name(step)
    success_metrics = (process_rules or {}).get(obj_name, {}).get("success_gates") or {}
    if not success_metrics:
        return False
    keys = {key for key in success_metrics.keys() if key is not None}
    return keys == {"visible"}


def stream_status_lines(world, step, process_rules=None, learned_rules=None):
    obj_name = _step_name(step)
    analytics = compute_brick_analytics(
        world,
        process_rules or {},
        learned_rules or {},
        obj_name,
        duration_s=getattr(world, "_last_dt", 0.05),
    )
    gate_progress = analytics.get("gate_progress") or []
    pct = None
    for name, val in gate_progress:
        if _step_name(name) == obj_name:
            pct = val
            break
    if pct is None:
        success_check = evaluate_success_gates(world, obj_name, learned_rules or {}, process_rules or {})
        pct = 100.0 if success_check.ok else 0.0
    pct_display = int(max(0.0, min(100.0, pct)))
    suggestion = getattr(world, "_last_action_line", None) or analytics.get("suggestion") or "HOLD"
    return [
        f"OBJ: {obj_name}",
        f"SUCCESS: {pct_display}%",
        f"ACT: {suggestion}",
    ]


def compute_alignment_analytics(world, process_rules, learned_rules, step, duration_s=0.05):
    obj_name = _step_name(step)
    success_metrics = (process_rules or {}).get(obj_name, {}).get("success_gates") or {}
    brick = world.brick or {}
    visible = bool(brick.get("visible"))
    x_axis = brick.get("x_axis")
    if x_axis is None:
        x_axis = 0.0
    x_axis = float(x_axis)
    angle = float(brick.get("angle", 0.0) or 0.0)
    dist = float(brick.get("dist", 0.0) or 0.0)

    metrics = {
        "xAxis_offset_abs": x_axis,
        "angle_abs": abs(angle),
        "dist": dist,
    }
    signed_values = {
        "xAxis_offset_abs": x_axis,
        "angle_abs": angle,
        "dist": dist,
    }
    progress_values = []
    offsets = {}
    ratios = {}
    mm_errors = {}
    def fallback_stats(metric):
        if metric == "xAxis_offset_abs":
            return {"max": float(getattr(world, "align_tol_offset", 12.0))}
        if metric == "angle_abs":
            return {"max": float(getattr(world, "align_tol_angle", 5.0))}
        if metric == "dist":
            return {
                "min": float(getattr(world, "align_tol_dist_min", 30.0)),
                "max": float(getattr(world, "align_tol_dist_max", 500.0)),
            }
        return {}
    def metric_within_gate(stats, value):
        target = stats.get("target")
        tol = stats.get("tol")
        if target is not None and tol is not None:
            return abs(value - target) <= tol
        min_val = stats.get("min")
        max_val = stats.get("max")
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True

    for metric, value in metrics.items():
        stats = success_metrics.get(metric) or fallback_stats(metric)
        direction = metric_direction_for_step(metric, obj_name)
        if direction is None:
            continue
        target = stats.get("target")
        tol = stats.get("tol")
        min_val = stats.get("min")
        max_val = stats.get("max")
        error = 0.0
        signed_error = 0.0
        progress = None

        if target is not None and tol is not None:
            signed_error = value - target
            error = max(0.0, abs(signed_error) - tol)
            if tol > 0:
                distance = abs(signed_error)
                if distance <= tol:
                    progress = 1.0
                else:
                    progress = max(0.0, 1.0 - (distance - tol) / tol)
                ratios[metric] = error / max(float(tol), 1e-3)
            else:
                progress = 1.0 if signed_error == 0 else 0.0
                ratios[metric] = 1.0 if signed_error != 0 else 0.0
        else:
            if min_val is not None and value < min_val:
                signed_error = value - min_val
                error = min_val - value
            if max_val is not None and value > max_val:
                signed_error = value - max_val
                error = max(value - max_val, error)
            if min_val is not None and max_val is not None:
                if min_val <= value <= max_val:
                    progress = 1.0
                else:
                    span = max(1e-3, max_val - min_val)
                    if value < min_val:
                        progress = max(0.0, 1.0 - (min_val - value) / span)
                    else:
                        progress = max(0.0, 1.0 - (value - max_val) / span)
            elif min_val is not None:
                progress = 1.0 if value >= min_val else max(0.0, value / max(min_val, 1e-3))
            elif max_val is not None:
                progress = 1.0 if value <= max_val else max(0.0, 1.0 - (value - max_val) / max(max_val, 1e-3))
            scale = max(abs(max_val or min_val or 1.0), 1e-3)
            ratios[metric] = error / scale

        if progress is not None:
            progress_values.append(progress)
        if metric in ("xAxis_offset_abs", "dist"):
            mm_errors[metric] = max(0.0, float(error))
        if metric == "xAxis_offset_abs":
            offsets["x_axis"] = signed_error
        elif metric == "angle_abs":
            offsets["angle"] = signed_error
        elif metric == "dist":
            offsets["dist"] = signed_error

    progress = sum(progress_values) / len(progress_values) if progress_values else None
    x_axis_stats = success_metrics.get("xAxis_offset_abs") or fallback_stats("xAxis_offset_abs")
    x_axis_ok = metric_within_gate(x_axis_stats, x_axis)
    worst_metric = max(ratios, key=lambda m: ratios[m], default=None)
    worst_ratio = ratios.get(worst_metric, 0.0) if worst_metric else 0.0

    if success_metrics:
        all_success = True
        for metric, stats in success_metrics.items():
            if metric == "visible":
                continue
            if not isinstance(stats, dict):
                continue
            direction = metric_direction_for_step(metric, obj_name)
            if metric == "angle_abs":
                value = abs(angle)
            elif metric == "xAxis_offset_abs":
                value = x_axis
            elif metric == "dist":
                value = dist
            else:
                continue
            target = stats.get("target")
            tol = stats.get("tol")
            if target is not None and tol is not None:
                ok = abs(value - target) <= tol
            else:
                ok = _target_tol_ok(value, stats, direction)
            if ok is False:
                all_success = False
                break
        if all_success:
            return {
                "progress": 1.0 if progress is None else progress,
                "worst_metric": None,
                "cmd": None,
                "speed": 0.0,
                "duration_s": duration_s,
                "x_axis": x_axis,
                "angle": angle,
                "dist": dist,
                "offsets": offsets,
            }

    if worst_metric is None or worst_ratio <= 0.0:
        fallback_metric = None
        fallback_value = 0.0
        for metric, value in offsets.items():
            if abs(value) > abs(fallback_value):
                fallback_metric = metric
                fallback_value = value
        if fallback_metric and abs(fallback_value) > 0.0:
            worst_metric = {
                "x_axis": "xAxis_offset_abs",
                "angle": "angle_abs",
                "dist": "dist",
            }.get(fallback_metric)
            worst_ratio = 1.0
        else:
            if not x_axis_ok:
                worst_metric = "xAxis_offset_abs"
                worst_ratio = max(ratios.get("xAxis_offset_abs", 1.0), 1.0)
            else:
                return {
                    "progress": progress,
                    "worst_metric": None,
                    "cmd": None,
                    "speed": 0.0,
                    "duration_s": duration_s,
                    "x_axis": x_axis,
                    "angle": angle,
                    "dist": dist,
                    "offsets": offsets,
                }

    if worst_metric == "dist" and not x_axis_ok:
        worst_metric = "xAxis_offset_abs"
        worst_ratio = max(worst_ratio, ratios.get("xAxis_offset_abs", 1.0), 1.0)

    min_speed = ALIGN_MIN_SPEED
    max_speed = ALIGN_MAX_SPEED
    micro_speed = ALIGN_MICRO_SPEED
    micro_offset_mm = ALIGN_MICRO_OFFSET_MM
    micro_angle_deg = ALIGN_MICRO_ANGLE_DEG

    speed_factor = max(0.0, min(1.0, (worst_ratio - 1.0) / 2.0))
    speed = min_speed + (max_speed - min_speed) * speed_factor

    cmd = None
    if worst_metric == "dist":
        dist_stats = success_metrics.get("dist") or fallback_stats("dist")
        dist_min = dist_stats.get("min")
        dist_max = dist_stats.get("max")
        target = dist_stats.get("target")
        tol = dist_stats.get("tol")
        if target is not None and tol is not None:
            if dist > target + tol:
                cmd = "f"
            elif dist < target - tol:
                cmd = "b"
        if cmd is None:
            if dist_max is not None and dist > dist_max:
                cmd = "f"
            elif dist_min is not None and dist < dist_min:
                cmd = "b"
        
        # Check if we should use combined movement (diagonal correction)
        # Use combined movement when we need both distance AND lateral correction
        if cmd is not None and not x_axis_ok:
            x_offset = signed_values.get("xAxis_offset_abs", 0.0)
            # Only use combined movements if lateral offset is significant
            if abs(x_offset) > COMBINED_MOVEMENT_THRESHOLD_MM:
                # Determine speed variant
                suffix = "_fast" if dist > COMBINED_MOVEMENT_FAST_THRESHOLD_MM else "_slow"
                
                # Forward and left -- only if far enough away
                if cmd == "f" and x_offset < 0 and dist > 70.0:
                    cmd = f"fl{suffix}"
                # Forward and right -- only if far enough away
                elif cmd == "f" and x_offset > 0 and dist > 70.0:
                    cmd = f"fr{suffix}"
                # Backward and left
                elif cmd == "b" and x_offset < 0:
                    cmd = f"bl{suffix}"
                # Backward and right
                elif cmd == "b" and x_offset > 0:
                    cmd = f"br{suffix}"
    elif worst_metric == "xAxis_offset_abs":
        signed = signed_values.get("xAxis_offset_abs", 0.0)
        stats = success_metrics.get("xAxis_offset_abs") or fallback_stats("xAxis_offset_abs")
        target = stats.get("target")
        tol = stats.get("tol")
        signed_error = signed - target if target is not None and tol is not None else signed
        if signed_error == 0:
            cmd = None
        else:
            cmd = "r" if signed_error > 0 else "l"
        if abs(signed_error) < micro_offset_mm:
            speed = min(speed, micro_speed)
        
        # Check if we should use combined movement when lateral is worst metric
        # Use combined movement when we need both lateral AND distance correction
        if cmd is not None:
            dist_offset = offsets.get("dist", 0.0)
            dist_stats = success_metrics.get("dist") or fallback_stats("dist")
            dist_min = dist_stats.get("min")
            dist_max = dist_stats.get("max")
            target_dist = dist_stats.get("target")
            tol_dist = dist_stats.get("tol")
            
            # Determine if distance needs correction
            needs_forward = False
            needs_backward = False
            if target_dist is not None and tol_dist is not None:
                if dist > target_dist + tol_dist:
                    needs_forward = True
                elif dist < target_dist - tol_dist:
                    needs_backward = True
            else:
                if dist_max is not None and dist > dist_max:
                    needs_forward = True
                elif dist_min is not None and dist < dist_min:
                    needs_backward = True
            
            # Use combined movements if distance offset is significant
            if abs(dist_offset) > COMBINED_MOVEMENT_THRESHOLD_MM:
                # Determine speed variant
                suffix = "_fast" if dist > COMBINED_MOVEMENT_FAST_THRESHOLD_MM else "_slow"
                
                # Left and forward -- only if far enough away
                if cmd == "l" and needs_forward and dist > 70.0:
                    cmd = f"fl{suffix}"
                # Right and forward -- only if far enough away
                elif cmd == "r" and needs_forward and dist > 70.0:
                    cmd = f"fr{suffix}"
                # Left and backward
                elif cmd == "l" and needs_backward:
                    cmd = f"bl{suffix}"
                # Right and backward
                elif cmd == "r" and needs_backward:
                    cmd = f"br{suffix}"
    elif worst_metric == "angle_abs":
        signed = signed_values.get("angle_abs", 0.0)
        stats = success_metrics.get("angle_abs") or fallback_stats("angle_abs")
        target = stats.get("target")
        tol = stats.get("tol")
        mag = abs(signed)
        signed_error = mag - target if target is not None and tol is not None else signed
        if signed_error >= 0:
            cmd = "r" if signed > 0 else "l"
        else:
            cmd = "l" if signed > 0 else "r"
        if abs(signed) < micro_angle_deg:
            speed = min(speed, micro_speed)

    visible_only = success_gates_visible_only(process_rules, obj_name)
    if visible_only:
        speed_score = SPEED_SCORE_DEFAULT
        speed_score = normalize_speed_score(speed_score)
    else:
        mm_off = None
        if cmd in ("l", "r"):
            mm_off = mm_errors.get("xAxis_offset_abs")
        elif cmd in ("f", "b"):
            mm_off = mm_errors.get("dist")
        if (mm_off is None or mm_off <= 0.0) and worst_metric == "angle_abs":
            mm_off = None
        slow_mm = ALIGN_SPEED_SLOW_MM
        fast_mm = ALIGN_SPEED_FAST_MM
        if obj_name != "ALIGN_BRICK":
            slow_mm = ALIGN_SPEED_SLOW_MM / 4.0
            fast_mm = ALIGN_SPEED_FAST_MM / 4.0
        speed_score = _score_from_mm(mm_off, slow_mm, fast_mm)
        speed_score = normalize_speed_score(speed_score)

    if not visible:
        speed_score = SPEED_SCORE_DEFAULT
        speed_score = normalize_speed_score(speed_score)

    # Override for combined movements
    if cmd in COMBINED_MOVEMENTS:
        speed_score = COMBINED_MOVEMENTS[cmd].get("score", speed_score)

    if cmd:
        speed = manual_speed_for_cmd(cmd, speed_score)
    else:
        speed = 0.0

    return {
        "progress": progress,
        "worst_metric": worst_metric,
        "cmd": cmd,
        "speed": speed,
        "speed_score": speed_score,
        "duration_s": duration_s,
        "x_axis": x_axis,
        "angle": angle,
        "dist": dist,
        "offsets": offsets,
    }


def compute_brick_analytics(world, process_rules, learned_rules, step, duration_s=0.05):
    process_rules = process_rules or {}
    align = compute_alignment_analytics(world, process_rules, learned_rules, step, duration_s=duration_s)
    gate_status = []
    gate_progress = []
    suggestion = None

    for obj_name, data in process_rules.items():
        if not isinstance(data, dict):
            continue
        success_gates = data.get("success_gates") or {}
        if not success_gates:
            continue
        progress_values = []
        ok = True
        for metric, stats in success_gates.items():
            direction = metric_direction_for_step(metric, obj_name)
            value = metric_value(world.brick or {}, metric)
            if value is None or direction is None or not isinstance(stats, dict):
                ok = False
                continue
            if isinstance(value, bool):
                min_val = stats.get("min")
                max_val = stats.get("max")
                if min_val is not None:
                    ok = ok and (value is bool(min_val))
                    progress_values.append(1.0 if value is bool(min_val) else 0.0)
                    continue
                if max_val is not None:
                    ok = ok and (value is bool(max_val))
                    progress_values.append(1.0 if value is bool(max_val) else 0.0)
                    continue
                ok = False
                continue

            target = stats.get("target")
            tol = stats.get("tol")
            if target is not None and tol is not None:
                err = abs(value - target)
                ok = ok and (err <= tol)
                if tol > 0:
                    if err <= tol:
                        progress_values.append(1.0)
                    else:
                        progress_values.append(max(0.0, 1.0 - (err - tol) / tol))
                else:
                    progress_values.append(1.0 if err == 0 else 0.0)
                continue

            min_val = stats.get("min")
            max_val = stats.get("max")
            if min_val is not None and value < min_val:
                ok = False
            if max_val is not None and value > max_val:
                ok = False
            if min_val is not None and max_val is not None:
                if min_val <= value <= max_val:
                    progress_values.append(1.0)
                else:
                    span = max(1e-3, max_val - min_val)
                    if value < min_val:
                        progress_values.append(max(0.0, 1.0 - (min_val - value) / span))
                    else:
                        progress_values.append(max(0.0, 1.0 - (value - max_val) / span))
            elif min_val is not None:
                progress_values.append(1.0 if value >= min_val else max(0.0, value / max(min_val, 1e-3)))
            elif max_val is not None:
                progress_values.append(1.0 if value <= max_val else max(0.0, 1.0 - (value - max_val) / max(max_val, 1e-3)))

        if obj_name == _step_name(step) and align.get("progress") is not None:
            gate_progress.append((obj_name, align.get("progress") * 100))
        elif progress_values:
            gate_progress.append((obj_name, sum(progress_values) / len(progress_values) * 100))
        if ok and progress_values:
            gate_status.append(obj_name)

    cmd = align.get("cmd")
    if cmd is not None:
        speed = align.get("speed") or 0.0
        speed_score = align.get("speed_score")
        suffix = f" {int(speed_score)}%" if speed_score is not None else ""
        suggestion = f"{cmd.upper()} {speed:.2f}p{suffix}"

    return {
        "gate_status": gate_status,
        "gate_progress": gate_progress,
        "suggestion": suggestion,
        "align": align,
        "highlight_metric": align.get("worst_metric"),
    }


def update_from_motion(world, event, delta):
    return


def update_from_vision(world, found, dist, angle, conf, offset_x=0, cam_h=0, brick_above=False, brick_below=False):
    world.brick["visible"] = bool(found)
    world.brick["dist"] = float(dist)
    world.brick["angle"] = float(angle)
    world.brick["confidence"] = float(conf)
    world.brick["offset_x"] = float(offset_x)
    world.brick["x_axis"] = float(offset_x)
    world.brick["brickAbove"] = bool(brick_above)
    world.brick["brickBelow"] = bool(brick_below)
    if found:
        world.last_visible_time = time.time()
        world.last_seen_angle = float(angle)
        world.last_seen_offset_x = float(offset_x)
        world.last_seen_x_axis = float(offset_x)
        world.last_seen_dist = float(dist)
        world.last_seen_confidence = float(conf)

    if world.step_state.value == "SCOOP":
        world.scoop_forward_preferred = True
        world.scoop_desired_offset_x = 0.0
        world.scoop_lateral_drift = world.brick["offset_x"] - world.scoop_desired_offset_x
    else:
        world.scoop_forward_preferred = False
        world.scoop_lateral_drift = 0.0

    brick_height = None
    if found and conf > 50 and cam_h > 0:
        if world.camera_height_anchor is None:
            world.camera_height_anchor = cam_h
        brick_height = max(0.0, world.camera_height_anchor - cam_h)
        world.height_mm = brick_height
    else:
        world.height_mm = None

    if found and conf >= ALIGN_CONFIDENCE_MIN:
        align_rules = world.learned_rules.get("ALIGN_BRICK", {}) or world.learned_rules.get("ALIGN", {})
        tol_off = align_rules.get("max_offset_x", world.align_tol_offset)
        tol_ang = align_rules.get("max_angle", world.align_tol_angle)

        if world.step_state.value == "SCOOP":
            corridor = get_scoop_corridor_limits(world, dist)
            if corridor:
                tol_off = corridor.get("max_offset_x", tol_off)
                tol_ang = corridor.get("max_angle", tol_ang)

        tol_off *= 1.1
        tol_ang *= 1.1
        if world.step_state.value == "SCOOP":
            tol_off *= world.scoop_success_offset_factor

        angle_ok = abs(angle) <= tol_ang
        offset_ok = abs(offset_x) <= tol_off
        dist_ok = world.align_tol_dist_min <= dist <= world.align_tol_dist_max

        if angle_ok and offset_ok and dist_ok:
            world.stability_count += 1
        else:
            world.stability_count = 0
    else:
        world.stability_count = 0

    return brick_height


def evaluate_start_gates(world, step, learned_rules, process_rules=None):
    obj_name = _step_name(step)
    reasons = []
    brick = smoothed_brick_snapshot(world)
    visible = bool(brick.get("visible"))
    confidence = brick.get("confidence", 0.0) or 0.0

    if obj_name in ("ALIGN_BRICK", "SCOOP", "POSITION_BRICK"):
        if not visible:
            reasons.append("brick not visible")
        elif confidence < START_GATE_MIN_CONFIDENCE:
            reasons.append(f"confidence<{START_GATE_MIN_CONFIDENCE:.0f}")

    if obj_name == "SCOOP" and visible and confidence >= START_GATE_MIN_CONFIDENCE:
        dist = brick.get("dist")
        angle = abs(brick.get("angle", 0.0))
        offset = abs(brick.get("offset_x", 0.0))

        corridor = get_scoop_corridor_limits(world, dist) if dist else None
        envelope = build_envelope(process_rules or {}, learned_rules or {}, "SCOOP")
        scoop_metrics = envelope.get("success", {})

        max_angle = None
        max_offset = None
        if corridor:
            max_angle = corridor.get("max_angle")
            max_offset = corridor.get("max_offset_x")
        else:
            max_angle = (scoop_metrics.get("angle_abs") or {}).get("max", world.align_tol_angle)
            max_offset = (scoop_metrics.get("xAxis_offset_abs") or {}).get("max", world.align_tol_offset)

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

    start_metrics = (process_rules or {}).get(obj_name, {}).get("start_gates") or {}
    if start_metrics:
        for metric, stats in start_metrics.items():
            if metric in ("angle_abs", "xAxis_offset_abs", "dist", "confidence") and not visible:
                reasons.append("brick not visible")
                continue
            min_val = stats.get("min")
            max_val = stats.get("max")
            if metric == "visible":
                if isinstance(min_val, bool):
                    if bool(visible) != min_val:
                        reasons.append("visible gate")
                elif isinstance(max_val, bool):
                    if bool(visible) != max_val:
                        reasons.append("visible gate")
                else:
                    if (1.0 if visible else 0.0) < (min_val or 0.0):
                        reasons.append("visible gate")
                continue
            value = metric_value(brick, metric)
            if value is None:
                continue
            if min_val is not None and value < min_val:
                reasons.append(f"{metric}<{min_val}")
            if max_val is not None and value > max_val:
                reasons.append(f"{metric}>{max_val}")

    return GateCheck(ok=not reasons, reasons=reasons)


def evaluate_success_gates(world, step, learned_rules, process_rules=None, visibility_grace_s=None):
    obj_name = _step_name(step)
    if obj_name not in METRICS_BY_STEP:
        return GateCheck(ok=True)

    envelope = build_envelope(process_rules or {}, learned_rules or {}, obj_name)
    success_metrics = envelope.get("success") or {}
    if not success_metrics:
        return GateCheck(ok=False, reasons=["no success envelope"])

    brick = smoothed_brick_snapshot(world)
    visible = bool(brick.get("visible"))
    visible_gate = success_metrics.get("visible") or {}
    if visibility_grace_s is None:
        visible_grace_s = VISIBILITY_LOST_GRACE_S
        if isinstance(visible_gate, dict) and visible_gate.get("min") is False:
            visible_grace_s = VISIBLE_FALSE_GRACE_S_BY_STEP.get(
                obj_name,
                VISIBILITY_LOST_GRACE_S,
            )
    else:
        visible_grace_s = visibility_grace_s
    effective_visible = _effective_visible(world, visible, grace_s=visible_grace_s)
    reasons = []
    
    # FIND_BRICK: Ensure we're finding a loose brick, not one already on the wall
    if obj_name == "FIND_BRICK":
        if brick.get("brickBelow"):
            reasons.append("brick already stacked")
            return GateCheck(ok=False, reasons=reasons)
    
    for metric, stats in success_metrics.items():
        direction = metric_direction_for_step(metric, obj_name)
        if metric in ("angle_abs", "xAxis_offset_abs", "dist", "confidence") and not visible:
            reasons.append("brick not visible")
            continue

        if metric == "angle_abs":
            angle_val = abs(brick.get("angle", 0.0))
            ok = _target_tol_ok(angle_val, stats, direction)
            if ok is False:
                reasons.append("angle_abs gate")
            elif ok is None and angle_val > stats.get("max", 0.0):
                reasons.append("angle_abs gate")
        elif metric == "xAxis_offset_abs":
            offset_val = brick.get("x_axis", brick.get("offset_x", 0.0))
            target = stats.get("target") if isinstance(stats, dict) else None
            tol = stats.get("tol") if isinstance(stats, dict) else None
            if target is not None and tol is not None:
                ok = abs(offset_val - target) <= tol
            else:
                ok = _target_tol_ok(offset_val, stats, direction)
            if ok is False:
                reasons.append("xAxis_offset_abs gate")
            elif ok is None and offset_val > stats.get("max", 0.0):
                reasons.append("xAxis_offset_abs gate")
        elif metric == "dist":
            dist_val = brick.get("dist", 0.0)
            ok = _target_tol_ok(dist_val, stats, direction)
            if ok is False:
                reasons.append("dist gate")
            elif ok is None and dist_val > stats.get("max", 0.0):
                reasons.append("dist gate")
        elif metric == "confidence":
            conf_val = brick.get("confidence", 0.0)
            ok = _target_tol_ok(conf_val, stats, direction)
            if ok is False:
                reasons.append("confidence gate")
            elif ok is None and conf_val < stats.get("min", 0.0):
                reasons.append("confidence gate")
        elif metric == "visible":
            min_val = stats.get("min")
            max_val = stats.get("max")
            if isinstance(min_val, bool):
                if bool(effective_visible) != min_val:
                    reasons.append("visible gate")
            elif isinstance(max_val, bool):
                if bool(effective_visible) != max_val:
                    reasons.append("visible gate")
            else:
                if (1.0 if effective_visible else 0.0) < stats.get("min", 0.0):
                    reasons.append("visible gate")
    return GateCheck(ok=not reasons, reasons=reasons)


def evaluate_failure_gates(world, step, learned_rules, process_rules=None):
    obj_name = _step_name(step)
    if obj_name not in METRICS_BY_STEP:
        return GateCheck(ok=True)
    envelope = build_envelope(process_rules or {}, learned_rules or {}, obj_name)
    failure_metrics = envelope.get("failure") or {}
    if not failure_metrics:
        return GateCheck(ok=True)
    
    brick = smoothed_brick_snapshot(world)
    visible = bool(brick.get("visible"))
    effective_visible = _effective_visible(world, visible)
    reasons = []
    
    # Check if current state matches known failure patterns
    for metric, stats in failure_metrics.items():
        if metric in ("angle_abs", "xAxis_offset_abs", "dist") and not visible:
            # Can't match failure pattern if brick isn't visible
            continue
            
        if metric == "visible":
            value = 1.0 if effective_visible else 0.0
        else:
            value = metric_value(brick, metric)
        if value is None:
            continue
        
        direction = metric_direction_for_step(metric, obj_name)
        
        # If we have learned failure mu/sigma, check if we're in the failure zone
        if "mu" in stats and "sigma" in stats:
            mu = stats.get("mu")
            sigma = stats.get("sigma", 0.0)
            
            # Direction-aware pattern matching
            if direction == "low":
                # For "low" metrics (angle, offset, dist), being HIGH is bad
                # Only trigger if we're near or above the failure pattern
                if value >= mu - sigma:
                    reasons.append(f"{metric} matches failure pattern ({value:.1f} ≈ {mu:.1f})")
            elif direction == "high":
                # For "high" metrics (visible, confidence), being LOW is bad
                # Only trigger if we're near or below the failure pattern
                if value <= mu + sigma:
                    reasons.append(f"{metric} matches failure pattern ({value:.1f} ≈ {mu:.1f})")
            else:
                # For other metrics, use simple range check
                if abs(value - mu) <= sigma:
                    reasons.append(f"{metric} matches failure pattern ({value:.1f} ≈ {mu:.1f})")
        else:
            # Fallback to min/max range check
            status = metric_status(value, {}, stats, METRIC_DIRECTIONS.get(metric))
            if status == "fail":
                reasons.append(f"{metric} gate")
    
    return GateCheck(ok=not reasons, reasons=reasons)

"""
Brick alignment telemetry helpers and correction suggestions.
"""


DEFAULT_DEMOS_DIR = Path(__file__).resolve().parent / "demos"


def offset_side_label(offset_x):
    if offset_x is None:
        return ""
    if offset_x > 0:
        return "right"
    if offset_x < 0:
        return "left"
    return "center"


def offset_marker_direction(offset_x):
    side = offset_side_label(offset_x)
    if side == "left":
        return "left of the marker"
    if side == "right":
        return "right of the marker"
    return ""


def offset_gap_phrase(offset_x):
    side = offset_side_label(offset_x)
    if side == "right":
        return "between the right side of the robot and the aruco marker"
    if side == "left":
        return "between the left side of the robot and the aruco marker"
    return "between the robot and the aruco marker"


def offset_cmd_from_offset_x(offset_x):
    if offset_x is None:
        return None
    if offset_x > 0:
        return "r"
    if offset_x < 0:
        return "l"
    return None


def distance_marker_direction(dist, gates):
    if dist is None:
        return ""
    stats = (gates or {}).get("dist") or {}
    target = stats.get("target")
    tol = stats.get("tol")
    min_val = stats.get("min")
    max_val = stats.get("max")
    if target is not None and tol is not None:
        if dist > target + tol:
            return "in front of the marker"
        if dist < target - tol:
            return "behind the marker"
        return ""
    if max_val is not None and dist > max_val:
        return "in front of the marker"
    if min_val is not None and dist < min_val:
        return "behind the marker"
    return ""


def worst_offset_direction(metric, measurement, gates):
    if not measurement:
        return ""
    if metric == "xAxis_offset_abs":
        x_axis = measurement.get("x_axis")
        if x_axis is None:
            x_axis = measurement.get("offset_x")
        return offset_marker_direction(x_axis)
    if metric == "dist":
        return distance_marker_direction(measurement.get("dist"), gates)
    return ""


def gap_direction_from_cmd(axis, cmd):
    if axis == "angle":
        return "to the right" if cmd == "l" else "to the left"
    if axis == "offset":
        return ""
    if axis == "distance":
        return "in front" if cmd == "f" else "behind"
    return ""


def distance_correction_cmd(measurement, gates):
    if not measurement:
        return None
    dist = measurement.get("dist")
    if dist is None:
        return None
    stats = (gates or {}).get("dist") or {}
    target = stats.get("target")
    tol = stats.get("tol")
    min_val = stats.get("min")
    max_val = stats.get("max")
    if target is not None and tol is not None:
        if dist > target + tol:
            return "f"
        if dist < target - tol:
            return "b"
        return None
    if max_val is not None and dist > max_val:
        return "f"
    if min_val is not None and dist < min_val:
        return "b"
    return None


def distance_gap_value(dist, gates):
    if dist is None:
        return None
    stats = (gates or {}).get("dist") or {}
    target = stats.get("target")
    tol = stats.get("tol")
    min_val = stats.get("min")
    max_val = stats.get("max")
    if target is not None and tol is not None:
        return abs(dist - target)
    if max_val is not None and dist > max_val:
        return dist - max_val
    if min_val is not None and dist < min_val:
        return min_val - dist
    return None


def offset_correction_cmd(measurement, gates):
    if not measurement:
        return None
    offset = measurement.get("offset_x")
    if offset is None:
        return None
    stats = (gates or {}).get("xAxis_offset_abs") or {}
    target = stats.get("target")
    tol = stats.get("tol")
    min_val = stats.get("min")
    max_val = stats.get("max")
    abs_offset = abs(offset)
    if target is not None and tol is not None:
        if abs_offset > target + tol:
            return "r" if offset > 0 else "l"
        if abs_offset < target - tol:
            return "l" if offset > 0 else "r"
        return None
    if max_val is not None and abs_offset > max_val:
        return "r" if offset > 0 else "l"
    if min_val is not None and abs_offset < min_val:
        return "l" if offset > 0 else "r"
    return None


def offset_gap_value(offset, gates):
    if offset is None:
        return None
    stats = (gates or {}).get("xAxis_offset_abs") or {}
    target = stats.get("target")
    tol = stats.get("tol")
    min_val = stats.get("min")
    max_val = stats.get("max")
    abs_offset = abs(offset)
    if target is not None and tol is not None:
        return abs(abs_offset - target)
    if max_val is not None and abs_offset > max_val:
        return abs_offset - max_val
    if min_val is not None and abs_offset < min_val:
        return min_val - abs_offset
    return None


def suggested_minor_correction(brick, success_gates):
    if not brick or not brick.get("visible"):
        return None
    cmd = offset_correction_cmd(brick, success_gates)
    if cmd:
        return "turn right" if cmd == "r" else "turn left"
    cmd = distance_correction_cmd(brick, success_gates)
    if cmd:
        return "forward" if cmd == "f" else "backward"
    return None


@dataclass
class BrickAlignmentState:
    dist: float
    offset: float
    angle: float
    visible: bool

    @classmethod
    def from_brick(cls, brick: Optional[dict]) -> "BrickAlignmentState":
        if not brick:
            return cls(0.0, 0.0, 0.0, False)
        dist = brick.get("dist")
        offset = brick.get("offset_x")
        angle = brick.get("angle")
        return cls(
            dist=float(dist) if dist is not None else 0.0,
            offset=float(offset) if offset is not None else 0.0,
            angle=float(angle) if angle is not None else 0.0,
            visible=bool(brick.get("visible")),
        )


@dataclass
class BrickAdjustment:
    mode: str
    distance_delta: float
    offset_delta: float
    angle_delta: float
    confidence: float = 0.0


class AlignmentEnvelope:
    def __init__(self, max_samples: int = 2048, neighbors: int = 6):
        self.max_samples = max_samples
        self.neighbors = max(1, neighbors)
        self.samples: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []

    def _normalize(self, state: BrickAlignmentState) -> Tuple[float, float, float]:
        dist = max(0.0, min(500.0, state.dist)) / 500.0
        offset = max(-200.0, min(200.0, state.offset)) / 200.0
        angle = max(-180.0, min(180.0, state.angle)) / 180.0
        return dist, offset, angle

    def record_transition(self, previous: BrickAlignmentState, current: BrickAlignmentState) -> None:
        if not (previous.visible and current.visible):
            return
        delta_dist = current.dist - previous.dist
        delta_offset = current.offset - previous.offset
        delta_angle = current.angle - previous.angle
        if (
            abs(delta_dist) < 0.3
            and abs(delta_offset) < 0.3
            and abs(delta_angle) < 0.25
        ):
            return
        features = self._normalize(previous)
        delta = (delta_dist, delta_offset, delta_angle)
        self.samples.append((features, delta))
        if len(self.samples) > self.max_samples:
            self.samples.pop(0)

    def _distance(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def predict(
        self, state: BrickAlignmentState
    ) -> Optional[Tuple[float, float, float, float]]:
        if not self.samples or not state.visible:
            return None
        query = self._normalize(state)
        scored = []
        for features, delta in self.samples:
            dist = self._distance(query, features)
            scored.append((dist, delta))
        scored.sort(key=lambda pair: pair[0])
        top = scored[: min(self.neighbors, len(scored))]
        total_weight = 0.0
        weighted_dist = 0.0
        weighted_offset = 0.0
        weighted_angle = 0.0
        for dist, delta in top:
            weight = 1.0 / (dist + 1e-3)
            total_weight += weight
            weighted_dist += delta[0] * weight
            weighted_offset += delta[1] * weight
            weighted_angle += delta[2] * weight
        if total_weight == 0.0:
            return None
        confidence = min(1.0, len(top) / self.neighbors)
        return (
            weighted_dist / total_weight,
            weighted_offset / total_weight,
            weighted_angle / total_weight,
            confidence,
        )

    def learn_from_demos(self, demos_dir: Optional[Path] = None, session: Optional[str] = None) -> None:
        demos_dir = Path(demos_dir) if demos_dir else DEFAULT_DEMOS_DIR
        if not demos_dir.exists():
            return
        logs = load_demo_logs(demos_dir, session)
        for _, rows in logs:
            segments = extract_attempt_segments(rows)
            for seg in segments:
                states = seg.get("states") or []
                sorted_states = sorted(states, key=lambda row: row.get("timestamp", 0.0))
                for prev, curr in zip(sorted_states, sorted_states[1:]):
                    prev_state = BrickAlignmentState.from_brick(prev.get("brick"))
                    curr_state = BrickAlignmentState.from_brick(curr.get("brick"))
                    self.record_transition(prev_state, curr_state)


class BrickAlignmentController:
    APPROACH_DISTANCE_THRESHOLD = 70.0
    APPROACH_OFFSET_THRESHOLD = 40.0
    APPROACH_FALLBACK_DISTANCE_GAIN = 0.3
    APPROACH_FALLBACK_OFFSET_GAIN = 0.35
    APPROACH_FALLBACK_ANGLE_GAIN = 0.6
    MICRO_DISTANCE_GAIN = 0.4
    MICRO_OFFSET_GAIN = 0.45

    def __init__(self, demos_dir: Optional[Path] = None):
        self.demos_dir = Path(demos_dir) if demos_dir else DEFAULT_DEMOS_DIR
        self.envelope = AlignmentEnvelope()
        self._last_state: Optional[BrickAlignmentState] = None
        self.envelope.learn_from_demos(self.demos_dir)

    def _choose_mode(self, state: BrickAlignmentState) -> str:
        if not state.visible:
            return "unknown"
        if state.dist > self.APPROACH_DISTANCE_THRESHOLD or abs(state.offset) > self.APPROACH_OFFSET_THRESHOLD:
            return "approach"
        return "micro"

    def _register_telemetry(self, state: BrickAlignmentState) -> None:
        if self._last_state:
            self.envelope.record_transition(self._last_state, state)
        self._last_state = state

    def next_adjustment(self, brick: Optional[dict]) -> Optional[BrickAdjustment]:
        state = BrickAlignmentState.from_brick(brick)
        if not state.visible:
            self._register_telemetry(state)
            return None
        self._register_telemetry(state)
        mode = self._choose_mode(state)
        if mode == "approach":
            return self._approach_adjustment(state)
        return self._micro_adjustment(state)

    def _approach_adjustment(self, state: BrickAlignmentState) -> BrickAdjustment:
        prediction = self.envelope.predict(state)
        if prediction:
            dist_delta, offset_delta, angle_delta, confidence = prediction
        else:
            dist_delta = -state.dist * self.APPROACH_FALLBACK_DISTANCE_GAIN
            offset_delta = -state.offset * self.APPROACH_FALLBACK_OFFSET_GAIN
            angle_delta = -state.angle * self.APPROACH_FALLBACK_ANGLE_GAIN
            confidence = 0.0
        if abs(angle_delta) < 1e-3:
            angle_delta = -state.angle * self.APPROACH_FALLBACK_ANGLE_GAIN
        return BrickAdjustment(
            mode="approach",
            distance_delta=dist_delta,
            offset_delta=offset_delta,
            angle_delta=angle_delta,
            confidence=confidence,
        )

    def _micro_adjustment(self, state: BrickAlignmentState) -> BrickAdjustment:
        return BrickAdjustment(
            mode="micro",
            distance_delta=-state.dist * self.MICRO_DISTANCE_GAIN,
            offset_delta=-state.offset * self.MICRO_OFFSET_GAIN,
            angle_delta=0.0,
            confidence=1.0,
        )
