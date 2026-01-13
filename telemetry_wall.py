import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from telemetry_envelope import GateCheck

WALL_MODEL_FILE = Path(__file__).parent / "world_model_wall.json"

OBJECTIVE_ALIASES = {
    "FIND": "FIND_BRICK",
    "ALIGN": "ALIGN_BRICK",
    "CARRY": "FIND_WALL2",
}

@dataclass
class WallEnvelope:
    angle_deg: float
    min_confidence: float
    max_origin_drift_mm: float
    max_angle_drift_deg: float
    place_offset_mm: float
    allow_auto_origin: bool
    lock_objective: str
    origin: Optional[dict]


def load_wall_model(path=WALL_MODEL_FILE):
    defaults = {
        "wall": {
            "x": None,
            "y": None,
            "angle_deg": 0.0,
            "immutable": True,
            "allow_auto_origin": True,
            "lock_objective": "FIND_WALL",
            "min_confidence": 80.0,
            "max_origin_drift_mm": 75.0,
            "max_angle_drift_deg": 10.0,
            "place_offset_mm": 25.0,
        }
    }
    if not path.exists():
        return defaults
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return defaults
    if "wall" not in data:
        data["wall"] = {}
    merged = defaults["wall"].copy()
    merged.update(data.get("wall", {}))
    return {"wall": merged}


def build_envelope(model):
    wall = (model or {}).get("wall", {})
    origin = None
    if wall.get("x") is not None and wall.get("y") is not None:
        origin = {
            "x": float(wall.get("x")),
            "y": float(wall.get("y")),
            "theta": float(wall.get("angle_deg", 0.0)),
        }
    return WallEnvelope(
        angle_deg=float(wall.get("angle_deg", 0.0)),
        min_confidence=float(wall.get("min_confidence", 80.0)),
        max_origin_drift_mm=float(wall.get("max_origin_drift_mm", 75.0)),
        max_angle_drift_deg=float(wall.get("max_angle_drift_deg", 10.0)),
        place_offset_mm=float(wall.get("place_offset_mm", 25.0)),
        allow_auto_origin=bool(wall.get("allow_auto_origin", True)),
        lock_objective=str(wall.get("lock_objective", "FIND_WALL")),
        origin=origin,
    )


def init_wall_state(envelope: WallEnvelope):
    origin = envelope.origin
    return {
        "origin": origin,
        "angle_deg": envelope.angle_deg,
        "valid": origin is not None,
        "immutable": origin is not None,
        "source": "MODEL" if origin is not None else None,
        "contradiction_reason": None,
        "last_seen": None,
    }


def _objective_name(objective):
    if hasattr(objective, "value"):
        name = objective.value
    else:
        name = str(objective)
    key = name.strip().upper()
    return OBJECTIVE_ALIASES.get(key, key)


def _needs_wall_origin(obj_name):
    if obj_name in ("PLACE", "POSITION_BRICK"):
        return True
    return "ALIGN" in obj_name and "WALL" in obj_name


def _wall_origin_distance(a, b):
    if not a or not b:
        return None
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    return math.hypot(dx, dy)


def compute_wall_origin(world, dist, angle_deg, envelope: WallEnvelope):
    heading = math.radians(world.theta + angle_deg)
    return {
        "x": world.x + (dist * math.cos(heading)),
        "y": world.y + (dist * math.sin(heading)),
        "theta": envelope.angle_deg,
    }


def update_from_vision(world, found, dist, angle_deg, conf, envelope: WallEnvelope):
    wall = world.wall
    if not found or conf < envelope.min_confidence:
        return

    candidate = compute_wall_origin(world, dist, angle_deg, envelope)

    if wall["origin"] is None:
        obj_name = _objective_name(world.objective_state)
        if envelope.allow_auto_origin or obj_name == envelope.lock_objective:
            wall["origin"] = candidate
            wall["angle_deg"] = envelope.angle_deg
            wall["valid"] = True
            wall["immutable"] = True
            wall["source"] = obj_name
            wall["last_seen"] = time.time()
        return

    wall["last_seen"] = time.time()
    if not wall["valid"]:
        return

    drift_mm = _wall_origin_distance(candidate, wall["origin"])
    if drift_mm is not None and drift_mm > envelope.max_origin_drift_mm:
        wall["valid"] = False
        wall["contradiction_reason"] = f"origin drift {drift_mm:.1f}mm"


def evaluate_start_gates(world, objective, envelope: WallEnvelope):
    obj_name = _objective_name(objective)
    if not _needs_wall_origin(obj_name):
        return GateCheck(ok=True)
    wall = world.wall
    reasons = []
    if wall.get("origin") is None:
        reasons.append("wall origin unset")
    if not wall.get("valid", False):
        reasons.append(wall.get("contradiction_reason") or "wall invalid")
    return GateCheck(ok=not reasons, reasons=reasons)


def evaluate_failure_gates(world, objective, envelope: WallEnvelope):
    obj_name = _objective_name(objective)
    if not _needs_wall_origin(obj_name):
        return GateCheck(ok=True)
    wall = world.wall
    if wall.get("origin") is None:
        return GateCheck(ok=False, reasons=["wall origin unset"])
    if not wall.get("valid", False):
        return GateCheck(ok=False, reasons=[wall.get("contradiction_reason") or "wall invalid"])
    return GateCheck(ok=True)


def evaluate_success_gates(world, objective, envelope: WallEnvelope):
    obj_name = _objective_name(objective)
    if not _needs_wall_origin(obj_name):
        return GateCheck(ok=True)
    wall = world.wall
    if wall.get("origin") is None:
        return GateCheck(ok=False, reasons=["wall origin unset"])
    if not wall.get("valid", False):
        return GateCheck(ok=False, reasons=[wall.get("contradiction_reason") or "wall invalid"])

    brick = world.brick or {}
    if not brick.get("visible"):
        return GateCheck(ok=True)

    dist = brick.get("dist")
    angle = brick.get("angle", 0.0)
    if dist is None:
        return GateCheck(ok=True)

    brick_x, brick_y = world.compute_brick_world_xy(dist, angle)
    wall_x = wall["origin"]["x"]
    wall_y = wall["origin"]["y"]
    wall_angle = math.radians(wall.get("angle_deg", envelope.angle_deg))

    dx = brick_x - wall_x
    dy = brick_y - wall_y
    perp_mm = abs(-math.sin(wall_angle) * dx + math.cos(wall_angle) * dy)
    if perp_mm > envelope.place_offset_mm:
        return GateCheck(ok=False, reasons=[f"wall offset {perp_mm:.1f}mm"])
    return GateCheck(ok=True)
