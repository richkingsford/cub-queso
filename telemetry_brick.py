import math
import time

from telemetry_envelope import GateCheck

START_GATE_MIN_CONFIDENCE = 25.0
ALIGN_CONFIDENCE_MIN = 25.0
VISION_HIT_CONFIDENCE = 40.0

OBJECTIVE_ALIASES = {
    "FIND": "FIND_BRICK",
    "ALIGN": "ALIGN_BRICK",
    "CARRY": "FIND_WALL2",
}

LOCK_JUMP_ANGLE_DEG = 8.0
LOCK_JUMP_OFFSET_MM = 20.0
LOCK_JUMP_DIST_MM = 80.0
LOCK_CONFIDENCE_ALPHA = 0.35

METRICS_BY_OBJECTIVE = {
    "FIND_WALL": ("angle_abs", "offset_abs", "dist", "visible"),
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


def _objective_name(objective):
    if hasattr(objective, "value"):
        name = objective.value
    else:
        name = str(objective)
    key = name.strip().upper()
    return OBJECTIVE_ALIASES.get(key, key)


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


def _update_lock_confidence(world, found, dist, angle, offset_x, conf):
    if not found:
        world._lock_last = None
        world._lock_confidence = 0.0
        world.brick["lock_confidence"] = 0.0
        return

    raw_conf = max(0.0, min(100.0, float(conf)))
    prev = getattr(world, "_lock_last", None)
    stability = 1.0
    if prev:
        angle_delta = abs(angle - prev.get("angle", angle))
        offset_delta = abs(offset_x - prev.get("offset_x", offset_x))
        dist_delta = abs(dist - prev.get("dist", dist))
        angle_score = max(0.0, 1.0 - (angle_delta / LOCK_JUMP_ANGLE_DEG))
        offset_score = max(0.0, 1.0 - (offset_delta / LOCK_JUMP_OFFSET_MM))
        dist_score = max(0.0, 1.0 - (dist_delta / LOCK_JUMP_DIST_MM))
        stability = min(angle_score, offset_score, dist_score)

    inst_conf = raw_conf * stability
    prev_conf = getattr(world, "_lock_confidence", raw_conf)
    smoothed = (LOCK_CONFIDENCE_ALPHA * inst_conf) + ((1.0 - LOCK_CONFIDENCE_ALPHA) * prev_conf)
    world._lock_confidence = smoothed
    world.brick["lock_confidence"] = smoothed
    world._lock_last = {
        "angle": float(angle),
        "offset_x": float(offset_x),
        "dist": float(dist),
        "timestamp": time.time(),
    }


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


def update_from_motion(world, event, delta):
    if event.action_type == "backward":
        if world.verification_stage == "BACK":
            world.verify_dist_mm += delta.dist_mm
            if world.verify_dist_mm >= 100:
                world.verification_stage = "LEFT"
    elif event.action_type == "left_turn":
        if world.verification_stage == "LEFT":
            world.verify_turn_deg += delta.rot_deg
            if world.verify_turn_deg >= 20:
                world.verification_stage = "RIGHT"
                world.verify_turn_deg = 0
    elif event.action_type == "right_turn":
        if world.verification_stage == "RIGHT":
            world.verify_turn_deg += delta.rot_deg
            if world.verify_turn_deg >= 20:
                seen = world.verify_vision_hits > 3
                if world.objective_state.value == "SCOOP":
                    world.brick["seated"] = not seen
                elif world.objective_state.value == "PLACE":
                    if seen:
                        world.brick["seated"] = False
                        world.brick["held"] = False
                world.verification_stage = "IDLE"
                world.verify_dist_mm = 0.0
                world.verify_turn_deg = 0.0
                world.verify_vision_hits = 0


def update_from_vision(world, found, dist, angle, conf, offset_x=0, cam_h=0, brick_above=False, brick_below=False):
    world.brick["visible"] = bool(found)
    world.brick["dist"] = float(dist)
    world.brick["angle"] = float(angle)
    world.brick["confidence"] = float(conf)
    world.brick["offset_x"] = float(offset_x)
    world.brick["brickAbove"] = bool(brick_above)
    world.brick["brickBelow"] = bool(brick_below)
    _update_lock_confidence(world, found, dist, angle, offset_x, conf)
    if found:
        world.last_visible_time = time.time()

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
        world.brick["height_mm"] = brick_height
    else:
        world.brick["height_mm"] = None

    world.brick["perfect_align"] = False
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

        commit_off = tol_off
        if world.objective_state.value == "SCOOP":
            commit_off *= world.scoop_commit_offset_factor

        angle_ok = abs(angle) <= tol_ang
        offset_ok = abs(offset_x) <= tol_off
        commit_offset_ok = abs(offset_x) <= commit_off
        dist_ok = world.align_tol_dist_min <= dist <= world.align_tol_dist_max

        if angle_ok and offset_ok and dist_ok:
            world.stability_count += 1
            if world.stability_count >= world.stability_threshold:
                world.brick["perfect_align"] = True
        else:
            world.stability_count = 0

        world.last_dist = dist
        if world.brick["perfect_align"] or (world.objective_state.value == "SCOOP" and angle_ok and commit_offset_ok and dist_ok):
            world.last_align_time = time.time()
            world.last_align_dist = dist
    else:
        commit = world.learned_rules.get("SCOOP", {}).get("commit", {})
        commit_dist = commit.get("max_dist", 100)
        commit_time = commit.get("time_s")
        aligned_dist = world.last_align_dist if world.last_align_dist is not None else world.last_dist
        aligned_recent = world.last_align_time is not None
        if aligned_recent and commit_time is not None:
            aligned_recent = (time.time() - world.last_align_time) <= commit_time
        if aligned_recent and aligned_dist < commit_dist and world.objective_state.value == "SCOOP":
            world.brick["seated"] = True
            if world.verification_stage == "IDLE":
                world.verification_stage = "BACK"
        world.stability_count = 0

    if found and conf > VISION_HIT_CONFIDENCE:
        if world.verification_stage in ["BACK", "LEFT", "RIGHT"]:
            world.verify_vision_hits += 1

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

    if obj_name == "LIFT":
        if not brick.get("seated"):
            reasons.append("brick not seated")

    return GateCheck(ok=not reasons, reasons=reasons)


def evaluate_success_gates(world, objective, learned_rules, process_rules=None):
    obj_name = _objective_name(objective)
    if obj_name == "SCOOP":
        ok = bool(world.brick.get("seated")) and world.verification_stage == "IDLE"
        return GateCheck(ok=ok, reasons=[] if ok else ["brick not seated"])

    if obj_name not in METRICS_BY_OBJECTIVE:
        return GateCheck(ok=True)

    envelope = build_envelope(process_rules or {}, learned_rules or {}, obj_name)
    success_metrics = envelope.get("success") or {}
    if not success_metrics:
        return GateCheck(ok=False, reasons=["no success envelope"])

    brick = world.brick or {}
    visible = bool(brick.get("visible"))
    reasons = []
    for metric, stats in success_metrics.items():
        if metric in ("angle_abs", "offset_abs", "dist", "confidence") and not visible:
            reasons.append("brick not visible")
            continue
        if metric == "angle_abs":
            if abs(brick.get("angle", 0.0)) > stats.get("max", 0.0):
                reasons.append("angle_abs gate")
        elif metric == "offset_abs":
            if abs(brick.get("offset_x", 0.0)) > stats.get("max", 0.0):
                reasons.append("offset_abs gate")
        elif metric == "dist":
            if brick.get("dist", 0.0) > stats.get("max", 0.0):
                reasons.append("dist gate")
        elif metric == "confidence":
            if brick.get("confidence", 0.0) < stats.get("min", 0.0):
                reasons.append("confidence gate")
        elif metric == "visible":
            if (1.0 if visible else 0.0) < stats.get("min", 0.0):
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
    reasons = []
    for metric, stats in failure_metrics.items():
        value = metric_value(brick, metric)
        status = metric_status(value, {}, stats, METRIC_DIRECTIONS.get(metric))
        if status == "fail":
            reasons.append(f"{metric} gate")
    return GateCheck(ok=not reasons, reasons=reasons)
