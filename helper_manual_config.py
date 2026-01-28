import json
from pathlib import Path

PROCESS_MODEL_FILE = Path(__file__).resolve().parent / "world_model_process.json"

DEFAULT_MANUAL_TRAINING_CONFIG = {
    "log_rate_hz": 10,
    "command_rate_hz": 30,
    "gear_1_speed": 0.32,
    "gear_9_speed": 1.0,
    "heartbeat_timeout": 0.3,
    "stream_host": "127.0.0.1",
    "stream_port": 5000,
    "stream_fps": 10,
    "stream_jpeg_quality": 85,
}


def load_manual_training_config(path=PROCESS_MODEL_FILE):
    cfg = dict(DEFAULT_MANUAL_TRAINING_CONFIG)
    try:
        model = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return cfg
    section = model.get("manual_training") if isinstance(model, dict) else None
    if isinstance(section, dict):
        cfg.update(section)
    return cfg
