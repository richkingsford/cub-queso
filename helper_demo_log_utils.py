import json
from pathlib import Path

DEMO_LOG_FILENAMES = ("a_log.json", "log.json")
OBJECTIVE_ALIASES = {
    "FIND": "FIND_BRICK",
    "ALIGN": "ALIGN_BRICK",
    "CARRY": "FIND_WALL2",
}
ATTEMPT_START_MARKERS = {
    "SUCCESS_START": "SUCCESS",
    "FAIL_START": "FAIL",
    "RECOVER_START": "RECOVER",
}
ATTEMPT_END_MARKERS = {
    "SUCCESS_END": "SUCCESS",
    "FAIL_END": "FAIL",
    "RECOVER_END": "RECOVER",
}
ALWAYS_KEEP_KEYFRAMES = {
    "JOB_START",
    "JOB_END",
    "JOB_ABORT",
    "JOB_SUCCESS",
}


def normalize_objective_label(label):
    if label is None:
        return None
    if hasattr(label, "value"):
        label = label.value
    key = str(label).strip().upper()
    return OBJECTIVE_ALIASES.get(key, key)


def resolve_session_log(session_path):
    path = Path(session_path)
    if path.is_file():
        return path
    for name in DEMO_LOG_FILENAMES:
        candidate = path / name
        if candidate.exists():
            return candidate
    return None


def find_demo_files(demos_dir, session_name=None):
    demos_dir = Path(demos_dir)
    if session_name:
        path = Path(session_name)
        if not path.is_absolute():
            path = demos_dir / session_name
        if path.is_dir():
            candidates = [path / name for name in DEMO_LOG_FILENAMES]
            existing = [p for p in candidates if p.exists()]
            if existing:
                return existing
            return sorted([p for p in path.glob("*.json") if p.is_file()])
        if path.exists():
            return [path]
        if not path.suffix:
            path = path.with_suffix(".json")
            if path.exists():
                return [path]
        return []

    if not demos_dir.exists():
        return []
    files = sorted([p for p in demos_dir.glob("*.json") if p.is_file()])
    for entry in sorted(demos_dir.iterdir()):
        if not entry.is_dir():
            continue
        for name in DEMO_LOG_FILENAMES:
            candidate = entry / name
            if candidate.exists():
                files.append(candidate)
    return files


def read_demo_log(path, strict=False):
    path = Path(path)
    try:
        raw = path.read_text()
    except OSError:
        if strict:
            raise
        return []

    if not raw.strip():
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or line in ("[", "]"):
                continue
            if line.endswith(","):
                line = line[:-1]
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if strict:
                    raise
                continue
            if isinstance(row, dict):
                data.append(row)
        return data

    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def load_demo_logs(demos_dir, session_name=None):
    logs = []
    for path in find_demo_files(demos_dir, session_name):
        rows = read_demo_log(path)
        if not any(row.get("type") == "keyframe" for row in rows):
            continue
        logs.append((path, rows))
    return logs


def extract_attempt_segments(log_data):
    start_markers = ATTEMPT_START_MARKERS
    end_markers = ATTEMPT_END_MARKERS
    discard_markers = {
        "SUCCESS_DISCARD": "SUCCESS",
        "FAIL_DISCARD": "FAIL",
        "RECOVER_DISCARD": "RECOVER",
    }
    segments = []
    active = {}
    current_obj = None
    obj_span = None

    for entry in log_data:
        if entry.get("type") == "keyframe":
            if entry.get("objective"):
                current_obj = normalize_objective_label(entry.get("objective"))
            marker = entry.get("marker")
            ts = entry.get("timestamp")
            if marker == "OBJ_START":
                obj_span = {
                    "type": "SUCCESS",
                    "source": "objective",
                    "objective": current_obj,
                    "start": ts,
                    "states": [],
                    "events": [],
                }
            if marker in start_markers:
                seg_type = start_markers[marker]
                active[seg_type] = {
                    "type": seg_type,
                    "source": "attempt",
                    "objective": current_obj,
                    "start": ts,
                    "states": [],
                    "events": [],
                }
            if marker in end_markers:
                seg_type = end_markers[marker]
                seg = active.pop(seg_type, None)
                if seg:
                    seg["end"] = ts
                    segments.append(seg)
            if marker in discard_markers:
                seg_type = discard_markers[marker]
                target_obj = normalize_objective_label(entry.get("objective") or current_obj)
                to_mark = 2 if seg_type == "SUCCESS" else 1
                for seg in reversed(segments):
                    if to_mark <= 0:
                        break
                    if seg.get("discarded"):
                        continue
                    if seg.get("objective") != target_obj:
                        continue
                    if seg.get("type") != seg_type:
                        continue
                    seg["discarded"] = True
                    to_mark -= 1
            if marker == "OBJ_SUCCESS" and obj_span:
                obj_span["end"] = ts
                segments.append(obj_span)
                obj_span = None
            continue

        if entry.get("type") == "state":
            for seg in active.values():
                seg["states"].append(entry)
            if obj_span is not None:
                obj_span["states"].append(entry)
        elif entry.get("type") in ("action", "event"):
            for seg in active.values():
                seg["events"].append(entry)
            if obj_span is not None:
                obj_span["events"].append(entry)

    return [seg for seg in segments if not seg.get("discarded")]


def write_demo_log(path, entries):
    path = Path(path)
    with open(path, "w") as f:
        f.write("[\n")
        for idx, entry in enumerate(entries):
            if idx:
                f.write(",\n")
            json.dump(entry, f)
        f.write("\n]\n")


def prune_unmatched_blocks(entries):
    active = {}
    keep_indices = set()
    valid_indices = set()
    valid_blocks = 0

    for idx, entry in enumerate(entries):
        entry_type = entry.get("type")
        marker = entry.get("marker") if entry_type == "keyframe" else None
        is_start = marker in ATTEMPT_START_MARKERS
        is_end = marker in ATTEMPT_END_MARKERS
        always_keep = marker in ALWAYS_KEEP_KEYFRAMES
        unmatched_end = False

        if is_start:
            seg_type = ATTEMPT_START_MARKERS[marker]
            active[seg_type] = []

        if is_end:
            seg_type = ATTEMPT_END_MARKERS[marker]
            seg_indices = active.pop(seg_type, None)
            if seg_indices is not None:
                seg_indices.append(idx)
                valid_indices.update(seg_indices)
                valid_blocks += 1
            else:
                unmatched_end = True

        for seg_indices in active.values():
            seg_indices.append(idx)

        if always_keep:
            keep_indices.add(idx)
        elif not active and not is_start and not unmatched_end:
            keep_indices.add(idx)

    keep_indices.update(valid_indices)
    pruned = [entry for idx, entry in enumerate(entries) if idx in keep_indices]
    return pruned, valid_blocks


def prune_log_file(path, delete_if_empty=False):
    path = Path(path)
    entries = read_demo_log(path)
    if not entries:
        if delete_if_empty and path.exists():
            path.unlink()
        return 0

    pruned, valid_blocks = prune_unmatched_blocks(entries)
    if delete_if_empty and valid_blocks == 0:
        if path.exists():
            path.unlink()
        return 0

    write_demo_log(path, pruned)
    return valid_blocks
