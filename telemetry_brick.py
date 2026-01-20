import math
import time
from dataclasses import dataclass, field
from typing import List

START_GATE_MIN_CONFIDENCE = 25.0
ALIGN_CONFIDENCE_MIN = 25.0
VISIBILITY_LOST_GRACE_S = 0.5
VISIBLE_FALSE_GRACE_S_BY_OBJECTIVE = {
    "EXIT_WALL": 1.0,
}

OBJECTIVE_ALIASES = {
    "FIND": "FIND_BRICK",
    "ALIGN": "ALIGN_BRICK",
    "CARRY": "FIND_WALL2",
}

METRICS_BY_OBJECTIVE = {
    "FIND_WALL": ("angle_abs", "offset_abs", "dist", "visible"),
    "EXIT_WALL": ("angle_abs", "offset_abs", "dist", "visible"),
    "FIND_BRICK": ("angle_abs", "offset_abs", "dist", "visible"),
    "ALIGN_BRICK": ("angle_abs", "offset_abs", "dist", "visible"),
    "SCOOP": ("angle_abs", "offset_abs", "dist", "visible"),
    "POSITION_BRICK": ("angle_abs", "offset_abs", "dist", "visible"),
}

METRIC_DIRECTIONS = {
    "angle_abs": "low",
    "offset_abs": "low",
    "dist": "low",
    "visible": "high",
    "confidence": "high",
}

VISIBILITY_REQUIRED_METRICS = {"angle_abs", "offset_abs", "dist"}


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
    if metric == "offset_abs":
        return abs(brick.get("offset_x", 0.0))
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


def update_from_motion(world, event, delta):
    return


def update_from_vision(world, found, dist, angle, conf, offset_x=0, cam_h=0, brick_above=False, brick_below=False):
    world.brick["visible"] = bool(found)
    world.brick["dist"] = float(dist)
    world.brick["angle"] = float(angle)
    world.brick["confidence"] = float(conf)
    world.brick["offset_x"] = float(offset_x)
    world.brick["brickAbove"] = bool(brick_above)
    world.brick["brickBelow"] = bool(brick_below)
    if found:
        world.last_visible_time = time.time()
        world.last_seen_angle = float(angle)
        world.last_seen_offset_x = float(offset_x)
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
            max_offset = (scoop_metrics.get("offset_abs") or {}).get("max", world.align_tol_offset)

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
            if metric in ("angle_abs", "offset_abs", "dist", "confidence") and not visible:
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
        if metric in ("angle_abs", "offset_abs", "dist", "confidence") and not visible:
            reasons.append("brick not visible")
            continue

        if metric == "angle_abs":
            angle_val = abs(brick.get("angle", 0.0))
            ok = _target_tol_ok(angle_val, stats, direction)
            if ok is False:
                reasons.append("angle_abs gate")
            elif ok is None and angle_val > stats.get("max", 0.0):
                reasons.append("angle_abs gate")
        elif metric == "offset_abs":
            offset_val = abs(brick.get("offset_x", 0.0))
            ok = _target_tol_ok(offset_val, stats, direction)
            if ok is False:
                reasons.append("offset_abs gate")
            elif ok is None and offset_val > stats.get("max", 0.0):
                reasons.append("offset_abs gate")
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
        if metric in ("angle_abs", "offset_abs", "dist") and not visible:
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
