import json
from pathlib import Path

PROCESS_MODEL_FILE = Path(__file__).resolve().parent / "world_model_process.json"


def load_process_objectives(path=PROCESS_MODEL_FILE):
    if not Path(path).exists():
        return {}
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data.get("objectives") or {}


def metric_value_from_measurement(measurement, metric):
    if not measurement:
        return None
    if metric == "visible":
        return bool(measurement.get("visible"))
    if metric == "angle_abs":
        value = measurement.get("angle")
        return abs(value) if value is not None else None
    if metric == "xAxis_offset_abs":
        value = measurement.get("x_axis")
        if value is None:
            value = measurement.get("offset_x")
        return abs(value) if value is not None else None
    if metric == "dist":
        value = measurement.get("dist")
        return float(value) if value is not None else None
    if metric == "angle":
        return measurement.get("angle")
    if metric == "x_axis":
        value = measurement.get("x_axis")
        if value is None:
            value = measurement.get("offset_x")
        return value
    if metric == "distance":
        return measurement.get("dist")
    return None


def metric_error(value, stats):
    if value is None or not isinstance(stats, dict):
        return None
    if isinstance(value, bool):
        min_val = stats.get("min")
        max_val = stats.get("max")
        if min_val is not None:
            return 0.0 if value is bool(min_val) else 1.0
        if max_val is not None:
            return 0.0 if value is bool(max_val) else 1.0
        return None
    target = stats.get("target")
    tol = stats.get("tol")
    if target is not None and tol is not None:
        return max(0.0, abs(value - target) - tol)
    min_val = stats.get("min")
    max_val = stats.get("max")
    if min_val is not None and value < min_val:
        return min_val - value
    if max_val is not None and value > max_val:
        return value - max_val
    return 0.0


def metric_progress(value, stats):
    if value is None or not isinstance(stats, dict):
        return None
    if isinstance(value, bool):
        err = metric_error(value, stats)
        return 1.0 if err == 0.0 else 0.0
    target = stats.get("target")
    tol = stats.get("tol")
    if target is not None and tol is not None:
        if tol <= 0:
            return 1.0 if value == target else 0.0
        distance = abs(value - target)
        if distance <= tol:
            return 1.0
        return max(0.0, 1.0 - (distance - tol) / tol)
    min_val = stats.get("min")
    max_val = stats.get("max")
    if min_val is not None and max_val is not None:
        if min_val <= value <= max_val:
            return 1.0
        span = max(1e-3, max_val - min_val)
        if value < min_val:
            return max(0.0, 1.0 - (min_val - value) / span)
        return max(0.0, 1.0 - (value - max_val) / span)
    if min_val is not None:
        return 1.0 if value >= min_val else max(0.0, value / max(min_val, 1e-3))
    if max_val is not None:
        return 1.0 if value <= max_val else max(0.0, 1.0 - (value - max_val) / max(max_val, 1e-3))
    return None


def objective_progress(measurement, success_gates):
    if not success_gates:
        return None
    progress_values = []
    for metric, stats in success_gates.items():
        value = metric_value_from_measurement(measurement, metric)
        prog = metric_progress(value, stats)
        if prog is not None:
            progress_values.append(prog)
    if not progress_values:
        return None
    return sum(progress_values) / len(progress_values)


def gate_satisfied(measurement, gates):
    if not gates:
        return False
    saw_value = False
    for metric, stats in gates.items():
        value = metric_value_from_measurement(measurement, metric)
        if value is None:
            continue
        saw_value = True
        err = metric_error(value, stats)
        if err is None or err > 0:
            return False
    return saw_value


def satisfied_objectives(measurement, objectives):
    satisfied = []
    for objective_name, data in objectives.items():
        success_gates = (data or {}).get("success_gates") or {}
        if gate_satisfied(measurement, success_gates):
            satisfied.append(objective_name)
    return satisfied
