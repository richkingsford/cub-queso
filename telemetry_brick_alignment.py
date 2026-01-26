"""
Brick alignment telemetry helpers and correction suggestions.
"""
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from helper_demo_log_utils import extract_attempt_segments, load_demo_logs

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
