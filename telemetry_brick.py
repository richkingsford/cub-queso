import math
import time
from dataclasses import dataclass, field
from typing import List

START_GATE_MIN_CONFIDENCE = 25.0
ALIGN_CONFIDENCE_MIN = 25.0
VISIBILITY_LOST_GRACE_S = 0.5
ALIGN_MIN_SPEED = 0.2
ALIGN_MAX_SPEED = 0.28
ALIGN_MICRO_SPEED = 0.21
ALIGN_FIXED_SPEED = 0.19
ALIGN_SPEED_MIN_POWER = 0.288
ALIGN_SPEED_SLOW = ALIGN_SPEED_MIN_POWER
ALIGN_SPEED_NORMAL = 0.28
ALIGN_SPEED_FAST = 0.392
ALIGN_SPEED_SLOW_MM = 16.0
ALIGN_SPEED_FAST_MM = 36.0
ALIGN_MICRO_OFFSET_MM = 10.0
ALIGN_MICRO_ANGLE_DEG = 5.0
VISIBLE_FALSE_GRACE_S_BY_OBJECTIVE = {
    "EXIT_WALL": 1.0,
}

OBJECTIVE_ALIASES = {
    "FIND": "FIND_BRICK",
    "ALIGN": "ALIGN_BRICK",
    "CARRY": "FIND_WALL2",
}

METRICS_BY_OBJECTIVE = {
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


def _objective_name(objective):
    if hasattr(objective, "value"):
        name = objective.value
    else:
        name = str(objective)
    key = name.strip().upper()
    return OBJECTIVE_ALIASES.get(key, key)


def metric_direction_for_objective(metric, objective):
    direction = METRIC_DIRECTIONS.get(metric)
    obj_name = _objective_name(objective)
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


def build_envelope(process_rules, learned_rules, objective):
    obj_name = _objective_name(objective)
    process = (process_rules or {}).get(obj_name, {})
    learned = learned_rules.get(obj_name, {}) if learned_rules else {}
    learned_gates = learned.get("gates", {})
    success = process.get("success_gates") or learned_gates.get("success", {}).get("metrics", {})
    failure = process.get("fail_gates") or learned_gates.get("failure", {}).get("metrics", {})
    return {"success": success, "failure": failure}


def success_gate_bounds(process_rules, learned_rules, objective):
    envelope = build_envelope(process_rules or {}, learned_rules or {}, objective)
    success_metrics = envelope.get("success") or {}
    bounds = {}
    for metric, stats in success_metrics.items():
        direction = metric_direction_for_objective(metric, objective)
        min_val, max_val = success_metric_bounds(stats, direction)
        bounds[metric] = {"min": min_val, "max": max_val}
    return bounds


def compute_alignment_analytics(world, process_rules, learned_rules, objective, duration_s=0.05):
    obj_name = _objective_name(objective)
    success_metrics = (process_rules or {}).get(obj_name, {}).get("success_gates") or {}
    brick = world.brick or {}
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
        direction = metric_direction_for_objective(metric, obj_name)
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
            direction = metric_direction_for_objective(metric, obj_name)
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

    speed_label = None
    if obj_name == "ALIGN_BRICK":
        mm_off = mm_errors.get("xAxis_offset_abs")
        if (mm_off is None or mm_off <= 0.0) and worst_metric == "angle_abs":
            mm_off = None
        speed_label = "n"
        speed = ALIGN_SPEED_NORMAL
        if mm_off is not None:
            if mm_off <= ALIGN_SPEED_SLOW_MM:
                speed_label = "s"
                speed = ALIGN_SPEED_SLOW
            elif mm_off >= ALIGN_SPEED_FAST_MM:
                speed_label = "f"
                speed = ALIGN_SPEED_FAST
        if cmd in ("f", "b"):
            speed = min(1.0, speed * 2.0)

    return {
        "progress": progress,
        "worst_metric": worst_metric,
        "cmd": cmd,
        "speed": speed,
        "speed_label": speed_label,
        "duration_s": duration_s,
        "x_axis": x_axis,
        "angle": angle,
        "dist": dist,
        "offsets": offsets,
    }


def compute_brick_analytics(world, process_rules, learned_rules, objective, duration_s=0.05):
    process_rules = process_rules or {}
    align = compute_alignment_analytics(world, process_rules, learned_rules, objective, duration_s=duration_s)
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
            direction = metric_direction_for_objective(metric, obj_name)
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

        if obj_name == _objective_name(objective) and align.get("progress") is not None:
            gate_progress.append((obj_name, align.get("progress") * 100))
        elif progress_values:
            gate_progress.append((obj_name, sum(progress_values) / len(progress_values) * 100))
        if ok and progress_values:
            gate_status.append(obj_name)

    cmd = align.get("cmd")
    if cmd is not None:
        speed = align.get("speed") or 0.0
        speed_label = align.get("speed_label")
        suffix = f" {speed_label}" if speed_label else ""
        suggestion = f"{cmd.upper()} {duration_s:.2f}s {speed:.2f}p{suffix}"

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

    if world.objective_state.value == "SCOOP":
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

        if world.objective_state.value == "SCOOP":
            corridor = get_scoop_corridor_limits(world, dist)
            if corridor:
                tol_off = corridor.get("max_offset_x", tol_off)
                tol_ang = corridor.get("max_angle", tol_ang)

        tol_off *= 1.1
        tol_ang *= 1.1
        if world.objective_state.value == "SCOOP":
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


def evaluate_start_gates(world, objective, learned_rules, process_rules=None):
    obj_name = _objective_name(objective)
    reasons = []
    brick = world.brick or {}
    visible = bool(brick.get("visible"))
    confidence = brick.get("confidence", 0.0) or 0.0

    if obj_name in ("ALIGN_BRICK", "SCOOP"):
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


def evaluate_success_gates(world, objective, learned_rules, process_rules=None, visibility_grace_s=None):
    obj_name = _objective_name(objective)
    if obj_name not in METRICS_BY_OBJECTIVE:
        return GateCheck(ok=True)

    envelope = build_envelope(process_rules or {}, learned_rules or {}, obj_name)
    success_metrics = envelope.get("success") or {}
    if not success_metrics:
        return GateCheck(ok=False, reasons=["no success envelope"])

    brick = world.brick or {}
    visible = bool(brick.get("visible"))
    visible_gate = success_metrics.get("visible") or {}
    if visibility_grace_s is None:
        visible_grace_s = VISIBILITY_LOST_GRACE_S
        if isinstance(visible_gate, dict) and visible_gate.get("min") is False:
            visible_grace_s = VISIBLE_FALSE_GRACE_S_BY_OBJECTIVE.get(
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
        direction = metric_direction_for_objective(metric, obj_name)
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


def evaluate_failure_gates(world, objective, learned_rules, process_rules=None):
    obj_name = _objective_name(objective)
    if obj_name not in METRICS_BY_OBJECTIVE:
        return GateCheck(ok=True)
    envelope = build_envelope(process_rules or {}, learned_rules or {}, obj_name)
    failure_metrics = envelope.get("failure") or {}
    if not failure_metrics:
        return GateCheck(ok=True)
    
    brick = world.brick or {}
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
        
        direction = metric_direction_for_objective(metric, obj_name)
        
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
