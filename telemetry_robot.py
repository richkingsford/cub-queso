"""
# telemetry_robot.py
-----------------
Handles the World Model and Logging for Robot Leia.
"""
import json
import math
import os
import threading
import time
import collections
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Speed/PWM tuning (single source of truth)
MIN_PWM = 36
MAX_PWM = 255
MIN_TURN_POWER = 0.064

ALIGN_MIN_SPEED = 0.2
ALIGN_MAX_SPEED = 0.28
ALIGN_MICRO_SPEED = 0.21
ALIGN_FIXED_SPEED = 0.19
ALIGN_SPEED_MIN_POWER = 0.288
ALIGN_SPEED_SLOW = ALIGN_SPEED_MIN_POWER
ALIGN_SPEED_NORMAL = 0.28
ALIGN_SPEED_FAST = 0.392
ALIGN_SPEED_SLOW_MM = 8.0
ALIGN_SPEED_FAST_MM = 18.0
ALIGN_MICRO_OFFSET_MM = 10.0
ALIGN_MICRO_ANGLE_DEG = 5.0

SPEED_SCORE_MIN = 1
SPEED_SCORE_DEFAULT = 50
SPEED_SCORE_MAX = 100
SPEED_SCORE_LEVELS = (SPEED_SCORE_MIN, SPEED_SCORE_DEFAULT, SPEED_SCORE_MAX)

ROBOT_MODEL_FILE = Path(__file__).resolve().parent / "world_model_robot.json"
DEFAULT_SPEED_MODEL = {
    "hotkey_speed_scores": {
        "w": {"cmd": "f", "score": 50},
        "s": {"cmd": "b", "score": 50},
        "r": {"cmd": "f", "score": 1},
        "f": {"cmd": "b", "score": 1},
        "t": {"cmd": "f", "score": 100},
        "g": {"cmd": "b", "score": 100},
        "q": {"cmd": "l", "score": 1},
        "a": {"cmd": "l", "score": 50},
        "z": {"cmd": "l", "score": 100},
        "e": {"cmd": "r", "score": 1},
        "d": {"cmd": "r", "score": 50},
        "c": {"cmd": "r", "score": 100},
        "u": {"cmd": "u", "score": 50},
        "l": {"cmd": "d", "score": 50},
    },
    "score_power_pwm": {
        "1": {"power": 0.064, "pwm": 50},
        "50": {"power": 0.5, "pwm": 145},
        "100": {"power": 1.0, "pwm": 255},
    },
}

def _brick_module():
    import telemetry_brick
    return telemetry_brick


def _wall_module():
    import telemetry_wall
    return telemetry_wall
    

def _load_speed_model(path=None):
    if path is None:
        path = ROBOT_MODEL_FILE
    
    print(f"[SYSTEM] Loading speed model from {path}...")
    model = DEFAULT_SPEED_MODEL
    if path.exists():
        try:
            text = path.read_text()
            data = json.loads(text)
            if isinstance(data, dict):
                model = data
                print(f"[SYSTEM] Loaded {len(model.get('combined_movements', {}))} combined movements.")
            else:
                print(f"[ERROR] JSON root is not a dict: {type(data)}")
        except (OSError, json.JSONDecodeError) as e:
            print(f"[ERROR] Failed to load speed model: {e}")
            model = DEFAULT_SPEED_MODEL
    else:
        print(f"[WARNING] Speed model file not found: {path}")


def _closest_score(score, levels, default=SPEED_SCORE_DEFAULT):
    try:
        score = float(score)
    except (TypeError, ValueError):
        return int(default)
    closest = None
    for candidate in levels:
        if closest is None or abs(candidate - score) < abs(closest - score):
            closest = candidate
    return int(closest if closest is not None else default)


def normalize_speed_score(score, default=SPEED_SCORE_DEFAULT):
    levels = SPEED_SCORE_LEVELS or (SPEED_SCORE_MIN, SPEED_SCORE_DEFAULT, SPEED_SCORE_MAX)
    return _closest_score(score, levels, default=default)


def _coerce_score_power_pwm(raw, fallback):
    if not isinstance(raw, dict):
        raw = fallback
    cleaned = {}
    for key, value in raw.items():
        try:
            score_key = int(float(key))
        except (TypeError, ValueError):
            continue
        if not isinstance(value, dict):
            continue
        power = value.get("power")
        pwm = value.get("pwm")
        try:
            power = float(power)
            pwm = int(pwm)
        except (TypeError, ValueError):
            continue
        cleaned[score_key] = {"power": power, "pwm": pwm}
    return cleaned


def _coerce_hotkeys(raw, fallback, score_levels):
    if not isinstance(raw, dict):
        raw = fallback
    cleaned = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        cmd = value.get("cmd")
        if not cmd:
            continue
        score = _closest_score(value.get("score"), score_levels, default=SPEED_SCORE_DEFAULT)
        cleaned[str(key)] = {"cmd": str(cmd), "score": score}
    return cleaned


def _load_speed_model(path):
    model = DEFAULT_SPEED_MODEL
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                model = data
        except (OSError, json.JSONDecodeError):
            model = DEFAULT_SPEED_MODEL
    score_map = _coerce_score_power_pwm(model.get("score_power_pwm"), DEFAULT_SPEED_MODEL["score_power_pwm"])
    if not score_map:
        score_map = _coerce_score_power_pwm(DEFAULT_SPEED_MODEL["score_power_pwm"], {})
    levels = tuple(sorted(score_map.keys()))
    if not levels:
        levels = (SPEED_SCORE_MIN, SPEED_SCORE_DEFAULT, SPEED_SCORE_MAX)
        score_map = _coerce_score_power_pwm(DEFAULT_SPEED_MODEL["score_power_pwm"], {})
    hotkeys = _coerce_hotkeys(model.get("hotkey_speed_scores"), DEFAULT_SPEED_MODEL["hotkey_speed_scores"], levels)
    if not hotkeys:
        hotkeys = _coerce_hotkeys(DEFAULT_SPEED_MODEL["hotkey_speed_scores"], {}, levels)
    return hotkeys, score_map, levels, model.get("combined_movements", {})


HOTKEY_SPEED_SCORES, SCORE_POWER_PWM, SPEED_SCORE_LEVELS, COMBINED_MOVEMENTS = _load_speed_model(ROBOT_MODEL_FILE)


def speed_power_pwm_for_cmd(cmd, score):
    score = normalize_speed_score(score)
    entry = SCORE_POWER_PWM.get(score, {})
    power = entry.get("power", 0.0)
    pwm = entry.get("pwm", 0)
    return power, pwm, score


def quantize_speed(cmd, speed=None, score=None):
    if score is not None:
        power, _, score_used = speed_power_pwm_for_cmd(cmd, score)
        return power, score_used
    if speed is None:
        return 0.0, None
    candidates = []
    for entry_score, entry in SCORE_POWER_PWM.items():
        power = entry.get("power")
        if power is None:
            continue
        candidates.append((abs(power - speed), entry_score, power))
    if not candidates:
        return 0.0, None
    candidates.sort(key=lambda item: item[0])
    _, score_used, power = candidates[0]
    return power, int(score_used)


def manual_speed_for_cmd(cmd, score):
    power, _, _ = speed_power_pwm_for_cmd(cmd, score)
    return power


def manual_key_action(key):
    entry = HOTKEY_SPEED_SCORES.get(key)
    if not entry:
        return None
    return entry["cmd"], entry["score"]


def _step_name(step):
    return _brick_module()._step_name(step)


def _build_envelope(process_rules, learned_rules, step):
    return _brick_module().build_envelope(process_rules, learned_rules, step)

METRICS_BY_STEP = {
    "LIFT": ("lift_height",),
    "PLACE": ("lift_height",),
}

METRIC_DIRECTIONS = {
    "lift_height": "band",
}


def resolve_scan_direction(process_rules, step, fallback="l"):
    obj_name = _step_name(step)
    rules = (process_rules or {}).get(obj_name, {})
    scan_direction = rules.get("scan_direction")
    if scan_direction in ("l", "r"):
        return scan_direction
    return fallback


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


@dataclass
class MotionDelta:
    dist_mm: float = 0.0
    rot_deg: float = 0.0
    lift_mm: float = 0.0




def evaluate_start_gates(world, step, learned_rules, process_rules=None):
    GateCheck = _brick_module().GateCheck
    return GateCheck(ok=True)


def evaluate_success_gates(world, step, learned_rules, process_rules=None):
    GateCheck = _brick_module().GateCheck
    obj_name = _step_name(step)
    if obj_name not in METRICS_BY_STEP:
        return GateCheck(ok=True)
    envelope = _build_envelope(process_rules or {}, learned_rules or {}, step)
    success_metrics = envelope.get("success") or {}
    if not success_metrics:
        return GateCheck(ok=False, reasons=["no lift success envelope"])
    stats = success_metrics.get("lift_height") or {}
    lift = world.lift_height
    ok = _target_tol_ok(lift, stats, METRIC_DIRECTIONS.get("lift_height"))
    if ok is False:
        return GateCheck(ok=False, reasons=["lift gate"])
    if ok is None:
        min_val = stats.get("min")
        max_val = stats.get("max")
        if min_val is not None and lift < min_val:
            return GateCheck(ok=False, reasons=[f"lift<{min_val:.1f}mm"])
        if max_val is not None and lift > max_val:
            return GateCheck(ok=False, reasons=[f"lift>{max_val:.1f}mm"])
    return GateCheck(ok=True)


def evaluate_failure_gates(world, step, learned_rules, process_rules=None):
    GateCheck = _brick_module().GateCheck
    obj_name = _step_name(step)
    if obj_name not in METRICS_BY_STEP:
        return GateCheck(ok=True)
    envelope = _build_envelope(process_rules or {}, learned_rules or {}, step)
    failure_metrics = envelope.get("failure") or {}
    stats = failure_metrics.get("lift_height")
    if not stats:
        return GateCheck(ok=True)
    lift = world.lift_height
    min_val = stats.get("min")
    max_val = stats.get("max")
    reasons = []
    if min_val is not None and lift < min_val:
        reasons.append(f"lift<{min_val:.1f}mm")
    if max_val is not None and lift > max_val:
        reasons.append(f"lift>{max_val:.1f}mm")
    return GateCheck(ok=not reasons, reasons=reasons)


def update_from_motion(world, event):
    dt = event.duration_ms / 1000.0
    power_ratio = event.power / 255.0
    dist_pulse = 0.0
    rot_pulse = 0.0
    lift_pulse = 0.0

    if event.action_type == "forward":
        dist_pulse = world.mm_per_sec_full_speed * power_ratio * dt
        rad = math.radians(world.theta)
        world.x += dist_pulse * math.cos(rad)
        world.y += dist_pulse * math.sin(rad)
    elif event.action_type == "backward":
        dist_pulse = world.mm_per_sec_full_speed * power_ratio * dt
        rad = math.radians(world.theta)
        world.x -= dist_pulse * math.cos(rad)
        world.y -= dist_pulse * math.sin(rad)
    elif event.action_type == "left_turn":
        rot_pulse = world.deg_per_sec_full_speed * power_ratio * dt
        world.theta += rot_pulse
    elif event.action_type == "right_turn":
        rot_pulse = world.deg_per_sec_full_speed * power_ratio * dt
        world.theta -= rot_pulse
    elif event.action_type == "mast_up":
        lift_pulse = world.lift_mm_per_sec * power_ratio * dt
        world.lift_height += lift_pulse
    elif event.action_type == "mast_down":
        lift_pulse = world.lift_mm_per_sec * power_ratio * dt
        world.lift_height -= lift_pulse
        if world.lift_height < 0:
            world.lift_height = 0

    return MotionDelta(dist_mm=dist_pulse, rot_deg=rot_pulse, lift_mm=lift_pulse)


def update_lift_from_vision(world, cam_h, brick_height, conf):
    if cam_h <= 0 or conf < 50:
        return
    brick_height = brick_height or 0.0
    if world.lift_height_anchor is None:
        world.lift_height_anchor = cam_h - world.lift_height + brick_height

    vis_lift = cam_h - world.lift_height_anchor + brick_height
    world.lift_height = (0.9 * world.lift_height) + (0.1 * vis_lift)

class StepState(Enum):
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

class MotionEvent:
    def __init__(self, action_type, power, duration_ms):
        self.action_type = action_type
        self.power = int(power) if power is not None else 0
        self.duration_ms = duration_ms
        self.timestamp = time.time()
        if self.action_type in ("left_turn", "right_turn") and 0 < self.power < MIN_TURN_POWER_PWM:
            self.power = MIN_TURN_POWER_PWM

    def to_dict(self):
        return {
            "type": self.action_type,
            "power": self.power,
            "duration_ms": self.duration_ms,
            "timestamp": round(self.timestamp, 3)
        }

MOTION_EVENT_TYPES = {
    "forward",
    "backward",
    "left_turn",
    "right_turn",
    "mast_up",
    "mast_down"
}

WORLD_MODEL_PROCESS_FILE = Path(__file__).parent / "world_model_process.json"
WORLD_MODEL_BRICK_FILE = Path(__file__).parent / "world_model_brick.json"
WORLD_MODEL_MOTION_FILE = Path(__file__).parent / "world_model_motion.json"

DEFAULT_MM_PER_SEC_FULL_SPEED = 200.0
DEFAULT_DEG_PER_SEC_FULL_SPEED = 90.0
DEFAULT_LIFT_MM_PER_SEC = 23.5
DEFAULT_MOTION_TICK_MS = 100.0
MIN_TURN_POWER_PWM = int(math.ceil(MIN_TURN_POWER * 255))


def _coerce_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_motion_calibration(path=WORLD_MODEL_MOTION_FILE):
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    motion = (data.get("calibration") or {}).get("motion") or {}
    return motion if isinstance(motion, dict) else {}


def motion_speeds_from_calibration(motion):
    if not isinstance(motion, dict):
        motion = {}

    mm_per_sec = _coerce_float(motion.get("mm_per_sec_full_speed"))
    deg_per_sec = _coerce_float(motion.get("deg_per_sec_full_speed"))
    lift_per_sec = _coerce_float(motion.get("mm_per_sec_mast"))

    tick_ms = _coerce_float(
        motion.get("tick_ms")
        or motion.get("command_duration_ms")
        or motion.get("cmd_duration_ms")
    )
    if tick_ms is None or tick_ms <= 0:
        tick_ms = DEFAULT_MOTION_TICK_MS
    tick_s = tick_ms / 1000.0

    if mm_per_sec is None:
        mm_per_tick = _coerce_float(motion.get("mm_per_tick"))
        if mm_per_tick is not None:
            mm_per_sec = mm_per_tick / tick_s
    if deg_per_sec is None:
        deg_per_tick = _coerce_float(motion.get("deg_per_tick"))
        if deg_per_tick is not None:
            deg_per_sec = deg_per_tick / tick_s
    if lift_per_sec is None:
        mm_per_tick_mast = _coerce_float(motion.get("mm_per_tick_mast"))
        if mm_per_tick_mast is not None:
            lift_per_sec = mm_per_tick_mast / tick_s

    return mm_per_sec, deg_per_sec, lift_per_sec

def _load_process_step_names():
    if not WORLD_MODEL_PROCESS_FILE.exists():
        return []
    try:
        with open(WORLD_MODEL_PROCESS_FILE, 'r') as f:
            model = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    steps = model.get("steps", {})
    if isinstance(steps, dict):
        return list(steps.keys())
    return []

def step_sequence():
    names = _load_process_step_names()
    if names:
        sequence = []
        seen = set()
        for name in names:
            normalized = _step_name(name)
            if normalized in StepState.__members__:
                obj = StepState[normalized]
                if obj not in seen:
                    sequence.append(obj)
                    seen.add(obj)
        if sequence:
            return sequence
    return list(StepState)

class WorldModel:
    def __init__(self):
        # Load Process Rules
        self.process_rules = {}
        if WORLD_MODEL_PROCESS_FILE.exists():
            try:
                with open(WORLD_MODEL_PROCESS_FILE, 'r') as f:
                    self.process_rules = json.load(f).get("steps", {})
            except: pass
        self.rules = self.process_rules
            
        self.learned_rules = {} # Rules derived from demo analysis
            
        # Robot Pose (Dead Reckoning)
        self.x = 0.0 # mm
        self.y = 0.0 # mm
        self.theta = 0.0 # degrees

        # Wall Model
        wall_module = _wall_module()
        self.wall_model = wall_module.load_wall_model()
        self.wall_envelope = wall_module.build_envelope(self.wall_model)
        self.wall = wall_module.init_wall_state(self.wall_envelope)

        # Brick Data
        self.brick = {
            "visible": False,
            "id": None,
            "dist": 0,
            "angle": 0,
            "offset_x": 0,
            "x_axis": 0,
            "confidence": 0,
            "held": False,
            "brickAbove": False,
            "brickBelow": False
        }

        # Forklift
        self.lift_height = 0.0 # mm (estimated)
        self.camera_height_anchor = None
        self.height_mm = None

        # Step
        self._step_state = None
        self._step_start_time = 0
        self._success_start_time = None
        self.step_state = StepState.FIND_BRICK
        self.attempt_status = "NORMAL" # NORMAL, FAIL, RECOVERY
        self.run_id = "unset"
        self.attempt_id = 0
        self.recording_active = False # For HUD prompt logic (Idle vs Success phase)
        
        # Alignment & Stability
        self.align_tol_angle = 5.0    # +/- Degrees
        self.align_tol_offset = 12.0  # +/- mm
        self.align_tol_dist_min = 30.0 # mm (Too close)
        self.align_tol_dist_max = 500.0 # mm (Too far)
        self.scoop_success_offset_factor = 1.2
        self.stability_count = 0
        self.stability_threshold = 10  # 10 frames @ 20Hz = 0.5 seconds
        
        self.last_visible_time = None
        self.scoop_desired_offset_x = 0.0
        self.scoop_lateral_drift = 0.0
        self.scoop_forward_preferred = False
        self.last_seen_angle = None
        self.last_seen_offset_x = None
        self.last_seen_dist = None
        self.last_seen_confidence = None
        
        self.last_image_file = None
        
        # Internal physics constants for dead reckoning (Calibration needed!)
        self.mm_per_sec_full_speed = DEFAULT_MM_PER_SEC_FULL_SPEED
        self.deg_per_sec_full_speed = DEFAULT_DEG_PER_SEC_FULL_SPEED
        self.lift_mm_per_sec = DEFAULT_LIFT_MM_PER_SEC
        motion = load_motion_calibration()
        mm_per_sec, deg_per_sec, lift_per_sec = motion_speeds_from_calibration(motion)
        if mm_per_sec is not None:
            self.mm_per_sec_full_speed = mm_per_sec
        if deg_per_sec is not None:
            self.deg_per_sec_full_speed = deg_per_sec
        if lift_per_sec is not None:
            self.lift_mm_per_sec = lift_per_sec
        self.lift_height_anchor = None # The Vision height at Mast=0mm
        
        self.action_history = collections.deque(maxlen=100)

    @property
    def step_state(self):
        return self._step_state

    @step_state.setter
    def step_state(self, value):
        if self._step_state == value:
            return
        self._step_state = value
        self._step_start_time = time.time()
        self._success_start_time = None
        self.last_visible_time = None
        # print(f"[WORLD] Step changed to {value}, timer reset.", flush=True)

    @property
    def wall_origin(self):
        return self.wall.get("origin")

    @wall_origin.setter
    def wall_origin(self, value):
        self.wall["origin"] = value
        self.wall["valid"] = value is not None

    def update_from_motion(self, event):
        """
        Updates pose based on motion events (Dead Reckoning).
        """
        delta = update_from_motion(self, event)
        brick_module = _brick_module()
        wall_module = _wall_module()
        brick_module.update_from_motion(self, event, delta)
        wall_module.update_from_motion(self, delta, self.wall_envelope)
        self.action_history.append(event)

    def get_recent_net_forward_mm(self, window_s=5.0):
        """
        Calculates net forward distance (Forward - Backward) in the last window_s seconds.
        """
        now = time.time()
        cutoff = now - window_s
        net_dist = 0.0
        
        for event in reversed(self.action_history):
            if event.timestamp < cutoff:
                break
                
            dist = 0.0
            dt = event.duration_ms / 1000.0
            power_ratio = event.power / 255.0
            
            if event.action_type == "forward":
                dist = self.mm_per_sec_full_speed * power_ratio * dt
                net_dist += dist
            elif event.action_type == "backward":
                dist = self.mm_per_sec_full_speed * power_ratio * dt
                net_dist -= dist
                
        return net_dist

    def update_vision(self, found, dist, angle, conf, offset_x=0, cam_h=0, brick_above=False, brick_below=False):
        brick_module = _brick_module()
        wall_module = _wall_module()
        brick_height = brick_module.update_from_vision(
            self,
            found,
            dist,
            angle,
            conf,
            offset_x,
            cam_h,
            brick_above,
            brick_below,
        )
        update_lift_from_vision(self, cam_h, brick_height, conf)
        wall_module.update_from_vision(self, found, dist, angle, conf, self.wall_envelope)

    def get_scoop_corridor_limits(self, dist):
        brick_module = _brick_module()
        return brick_module.get_scoop_corridor_limits(self, dist)

    def compute_brick_world_xy(self, dist, angle_deg):
        brick_module = _brick_module()
        return brick_module.compute_brick_world_xy(self, dist, angle_deg)

    def is_aligned(self):
        """Returns True if metrics have been stable and centered."""
        return self.stability_count >= self.stability_threshold

    def check_step_complete(self):
        """Checks if success criteria are met using learned rules from demos."""
        wall_module = _wall_module()
        wall_check = wall_module.evaluate_success_gates(self, self.step_state, self.wall_envelope)
        if not wall_check.ok:
            return False
        obj_name = self.step_state.value

        gates = self.learned_rules.get(obj_name, {}).get("gates", {})
        success_metrics = gates.get("success", {}).get("metrics", {})
        if success_metrics:
            brick = self.brick or {}
            brick_visible = bool(brick.get("visible"))
            for metric, stats in success_metrics.items():
                if metric in ("angle_abs", "xAxis_offset_abs", "dist", "confidence") and not brick_visible:
                    return False
                if metric == "angle_abs":
                    if abs(brick.get("angle", 0.0)) > stats.get("max", 0.0):
                        return False
                elif metric == "xAxis_offset_abs":
                    if abs(brick.get("offset_x", 0.0)) > stats.get("max", 0.0):
                        return False
                elif metric == "dist":
                    if brick.get("dist", 0.0) > stats.get("max", 0.0):
                        return False
                elif metric == "confidence":
                    if brick.get("confidence", 0.0) < stats.get("min", 0.0):
                        return False
                elif metric == "visible":
                    if (1.0 if brick_visible else 0.0) < stats.get("min", 0.0):
                        return False
                elif metric == "lift_height":
                    lift = self.lift_height
                    if lift < stats.get("min", lift) or lift > stats.get("max", lift):
                        return False
            return True

        learned = self.learned_rules.get(obj_name, {})
        if not learned:
            return False

        target_vis = learned.get("final_visibility", True)
        if self.brick["visible"] != target_vis:
            return False

        if target_vis:
            max_x = learned.get("max_offset_x", 0)
            if abs(self.brick["offset_x"]) > max_x:
                return False
            max_ang = learned.get("max_angle", 0)
            if abs(self.brick["angle"]) > max_ang:
                return False

        return True

    def next_step(self):
        """Cycles through steps in the process order."""
        sequence = step_sequence()
        if not sequence:
            sequence = list(StepState)
        try:
            curr_idx = sequence.index(self.step_state)
        except ValueError:
            sequence = list(StepState)
            curr_idx = sequence.index(self.step_state)
        next_idx = (curr_idx + 1) % len(sequence)
        self.step_state = sequence[next_idx]
        if next_idx == 0:
            self.brick["held"] = False
        return self.step_state.value

    def get_next_step_label(self):
        """Returns the string label of the next step in sequence."""
        sequence = step_sequence()
        if not sequence:
            sequence = list(StepState)
        labels = [o.value for o in sequence]
        try:
            curr_idx = labels.index(self.step_state.value)
        except ValueError:
            labels = [o.value for o in StepState]
            curr_idx = labels.index(self.step_state.value)
        next_idx = (curr_idx + 1) % len(labels)
        return labels[next_idx]

    def reset_mission(self):
        """Resets the step state and all mission-specific flags."""
        self.step_state = StepState.FIND_BRICK
        self.brick["held"] = False
        self.stability_count = 0
        self.last_visible_time = None
        return self.step_state.value

    def to_dict(self):
        # Format Brick Data
        brick_fmt = self.brick.copy()
        if self.step_state == StepState.FIND_BRICK:
            brick_fmt['dist'] = None
            brick_fmt['angle'] = None
            brick_fmt['offset_x'] = None
            brick_fmt['x_axis'] = None
            brick_fmt['confidence'] = None
            brick_fmt['brickAbove'] = None
            brick_fmt['brickBelow'] = None
        elif brick_fmt.get("visible"):
            if brick_fmt.get("dist") is not None:
                brick_fmt['dist'] = round(brick_fmt['dist'], 2)
            if brick_fmt.get("angle") is not None:
                brick_fmt['angle'] = round(brick_fmt['angle'], 3)
            if brick_fmt.get("offset_x") is not None:
                brick_fmt['offset_x'] = round(brick_fmt['offset_x'], 2)
            if brick_fmt.get("x_axis") is not None:
                brick_fmt['x_axis'] = round(brick_fmt['x_axis'], 2)
            if brick_fmt.get("confidence") is not None:
                brick_fmt['confidence'] = int(brick_fmt['confidence'])
        else:
            brick_fmt['dist'] = None
            brick_fmt['angle'] = None
            brick_fmt['offset_x'] = None
            brick_fmt['x_axis'] = None
            brick_fmt['confidence'] = None

        # Format Wall Origin
        wall_fmt = None
        if self.wall.get("origin"):
            wall_fmt = {
                'x': round(self.wall["origin"]['x'], 2),
                'y': round(self.wall["origin"]['y'], 2),
                'theta': round(self.wall["origin"]['theta'], 3)
            }
        wall_state = {
            "origin": wall_fmt,
            "angle_deg": round(self.wall.get("angle_deg", 0.0), 3),
            "valid": bool(self.wall.get("valid", False)),
            "source": self.wall.get("source"),
            "contradiction": self.wall.get("contradiction_reason"),
        }

        return {
            "type": "state",
            "timestamp": round(time.time(), 3),
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "robot_pose": {
                "x": round(self.x, 2), 
                "y": round(self.y, 2), 
                "theta": round(self.theta, 3),
                "height_mm": None if self.height_mm is None else round(self.height_mm, 2)
            },
            "wall_origin": wall_fmt,
            "wall": wall_state,
            "brick": brick_fmt,
            "lift_height": round(self.lift_height, 2)
        }

class TelemetryLogger:
    def __init__(self, filename="leia_log.json"):
        self.filename = filename
        self.lock = threading.Lock()
        self.enabled = False # Don't log state until first keyframe
        # Clear old log
        with open(self.filename, 'w') as f:
            f.write("[\n") # Start JSON array
        self.first_entry = True

    def log_state(self, world_model: WorldModel):
        if not self.enabled:
            return
        data = world_model.to_dict()
        self._write_row(data)

    def log_keyframe(self, marker, step=None, timestamp=None):
        self.enabled = True # Start recording state once we have a semantic marker
        if timestamp is None:
            timestamp = time.time()
        
        data = {
            "type": "keyframe",
            "timestamp": round(timestamp, 3),
            "marker": marker
        }
        if step:
            data["step"] = step
            
        self._write_row(data)

    def _write_row(self, data):
        with self.lock:
            with open(self.filename, 'a') as f:
                if not self.first_entry:
                    f.write(",\n")
                json.dump(data, f)
                self.first_entry = False

    def log_event(self, event: MotionEvent, step=None):
        semantic_events = ['FAIL', 'RECOVERY_START', 'STEP_SUCCESS', 'JOB_SUCCESS', 'JOB_START']
        if event.action_type in semantic_events:
            self.log_keyframe(event.action_type, step, event.timestamp)
            return

        if not self.enabled:
            return

        if event.action_type not in MOTION_EVENT_TYPES:
            return

        data = {
            "type": "action",
            "timestamp": round(event.timestamp, 3),
            "command": event.action_type,
            "power": int(event.power),
            "duration_ms": int(event.duration_ms)
        }

        self._write_row(data)

    def close(self):
        """
        Consolidated close method that handles JSON array termination.
        Robustly handles crashes by searching backward for the last valid '}'.
        """
        with self.lock:
            if not os.path.exists(self.filename):
                return
                
            try:
                with open(self.filename, 'rb+') as f:
                    f.seek(0, os.SEEK_END)
                    pos = f.tell()
                    
                    found_last_brace = False
                    # Search backwards for the last '}'
                    while pos > 0:
                        pos -= 1
                        f.seek(pos)
                        char = f.read(1)
                        if char == b'}':
                            # Found the end of a valid JSON object.
                            # Keep this row, truncate after it.
                            f.seek(pos + 1)
                            f.truncate()
                            found_last_brace = True
                            break
                        elif char == b'[': 
                            # Empty array case
                            f.seek(pos + 1)
                            f.truncate()
                            break
                    
                    # Ensure any trailing garbage (like a loose comma) is gone
                    # We already truncated at '}', so we are good.
                    
                    # Add final closing bracket
                    f.seek(0, os.SEEK_END)
                    if found_last_brace:
                        f.write(b"\n]\n")
                    else:
                        # If list was totally empty or malformed
                        f.write(b"]\n")
                        
                print(f"[LOGGER] Log closed and sanitized: {self.filename}")
            except Exception as e:
                print(f"[LOGGER] Error closing log: {e}")

    def _print_terminal(self, data):
        p = data.get('robot_pose', {'x':0, 'y':0, 'theta':0})
        b = data.get('brick', {})
        wall = "SET" if data.get('wall_origin') else "UNSET"
        print(f"{'='*40}")
        print(f"TIME: {data.get('timestamp', 0):.2f}s")
        if 'step' in data:
            print(f"STEP: {data['step']}")
        print(f"WALL: {wall}")
        print(f"{'-'*40}")
        print(f"POSE:")
        print(f"  X: {p['x']:.2f} mm")
        print(f"  Y: {p['y']:.2f} mm")
        print(f"  Heading: {p['theta']:.2f}°")
        print(f"  Lift: {data.get('lift_height', 0):.2f} mm")
        print(f"{'-'*40}")
        print(f"BRICK:")
        print(f"  Visible: {b.get('visible', False)}")
        if b.get('visible'):
            print(f"  Distance: {b.get('dist', 0):.2f} mm")
            print(f"  Angle: {b.get('angle', 0):.2f}°")
            print(f"  Offset: {b.get('offset_x', 0):.2f} mm")
            print(f"  Confidence: {b.get('confidence', 0):.2f}%")
        print(f"{'-'*40}")
        
        print(f"{'='*40}")

# --- SHARED VISUALIZATION ---
import cv2

def draw_telemetry_overlay(
    frame,
    wm: WorldModel,
    extra_messages=None,
    reminders=None,
    gear=None,
    show_prompt=True,
    gate_status=None,
    gate_progress=None,
    step_suggestions=None,
    highlight_metric=None,
    loop_id=None,
    header_lines=None,
    gate_summary=None,
):
    """
    Simplified HUD renderer.
    - Merged step/checklist/status into single-line prompt.
    - Controls are logged in terminal, not shown on the overlay.
    - Optional gear label is handled separately.
    """
    h, w = frame.shape[:2]
    
    # --- COLORS (BGR) ---
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    WHITE = (255, 255, 255)
    ORANGE = (0, 165, 255)
    YELLOW = (0, 255, 255)
    
    # 0. Center Alignment Line
    cal_offset = 0
    if WORLD_MODEL_BRICK_FILE.exists():
        try:
            with open(WORLD_MODEL_BRICK_FILE, 'r') as f:
                cal_offset = json.load(f).get('calibration', {}).get('camera_center_offset_px', 0)
        except: pass
    cv2.line(frame, (int(w//2 + cal_offset), 0), (int(w//2 + cal_offset), h), (60, 60, 60), 1)

    # 1. Background Panel (Left Side)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (220, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # 2. Text Setup
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.38
    thickness = 1 # No bolding, as it gets fuzzy
    x_base = 12
    y_cur = 25
    line_h = 20
    
    def put_line(txt, c=WHITE, s=scale, th=thickness, thickness=None):
        nonlocal y_cur
        if thickness is not None:
            th = thickness
        cv2.putText(frame, txt, (x_base, y_cur), font, s, c, th)
        y_cur += line_h

    # 3. MERGED STATE & PROMPT - REMOVED per user request
    y_cur += 5

    # 3b. Header lines (Step/Success/Suggested act)
    if header_lines:
        for line in header_lines:
            put_line(str(line), WHITE, 0.45, 1)
        y_cur += 5

    # 4. Reminders
    if reminders:
        put_line("--- REMINDERS ---", WHITE, 0.35, 1)
        if isinstance(reminders, list):
            for msg in reminders:
                put_line(str(msg), WHITE, 0.35, 1)
        else:
            put_line(str(reminders), WHITE, 0.35, 1)
        y_cur += 5

    # 4b. Success Gates (summary or current step only)
    if gate_summary is not None:
        put_line("--- SUCCESS GATES ---", WHITE, 0.35, 1)
        if gate_summary:
            for line in gate_summary:
                if isinstance(line, tuple):
                    text, color = line
                    put_line(str(text), color, 0.35, 1)
                else:
                    put_line(str(line), WHITE, 0.35, 1)
        else:
            put_line("(idle)", WHITE, 0.35, 1)
        y_cur += 5
    elif gate_progress is not None:
        if loop_id is not None:
            put_line(f"LOOP ID: {loop_id}", WHITE, 0.35, 1)
            y_cur += 3
        put_line("--- SUCCESS GATES ---", WHITE, 0.35, 1)
        current_obj = wm.step_state.value if wm.step_state else None
        match = None
        if gate_progress and current_obj:
            for name, pct in gate_progress:
                if str(name) == str(current_obj):
                    match = (name, pct)
                    break
        if match:
            name, pct = match
            pct_display = int(max(0.0, min(100.0, pct)))
            put_line(f"{name}: {pct_display}%", WHITE, 0.35, 1)
            if step_suggestions:
                for obj_name, suggestion in step_suggestions:
                    if str(obj_name) == str(name):
                        sug_color = WHITE
                        trend_map = getattr(wm, "_align_metrics_trend", {})
                        if suggestion.startswith(("L ", "R ")):
                            trend_val = trend_map.get("x_axis")
                        elif suggestion.startswith(("F ", "B ")):
                            trend_val = trend_map.get("dist")
                        else:
                            trend_val = 0
                        if trend_val == 1:
                            sug_color = GREEN
                        elif trend_val == -1:
                            sug_color = RED
                        put_line(f"  {suggestion}", sug_color, 0.35, 1)
        else:
            put_line("(none)", WHITE, 0.35, 1)
        y_cur += 5

    # 5. Position Info
    put_line("--- BRICK[0] TELEMETRY ---", WHITE, 0.35, 1)
    visible_now = bool(wm.brick.get("visible"))
    x_axis = wm.brick.get("x_axis", wm.brick.get("offset_x", 0.0))
    obj_rules = (wm.process_rules or {}).get("ALIGN_BRICK", {}) if wm.process_rules else {}
    success_gates = (obj_rules or {}).get("success_gates") or {}
    x_gate = success_gates.get("xAxis_offset_abs") or {}
    angle_gate = success_gates.get("angle_abs") or {}
    dist_gate = success_gates.get("dist") or {}
    def _target_tol_str(stats, fmt):
        target = stats.get("target")
        tol = stats.get("tol")
        if isinstance(target, (int, float)) and isinstance(tol, (int, float)):
            return f" ({fmt(target)}+/-{fmt(tol)})"
        min_val = stats.get("min")
        max_val = stats.get("max")
        if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
            return f" ({fmt(min_val)}-{fmt(max_val)})"
        return ""
    def _gate_line(stats, fmt, label, current_val, signed=False):
        target = stats.get("target")
        tol = stats.get("tol")
        if isinstance(target, (int, float)) and isinstance(tol, (int, float)):
            off_val = current_val - target if signed else abs(current_val - target)
            return f"  {label} {fmt(target)} +/- {fmt(tol)} | {fmt(off_val)} off"
        min_val = stats.get("min")
        max_val = stats.get("max")
        if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
            if current_val < min_val:
                off_val = min_val - current_val
            elif current_val > max_val:
                off_val = current_val - max_val
            else:
                off_val = 0.0
            return f"  {label} {fmt(min_val)}-{fmt(max_val)} | {fmt(off_val)} off"
        return None
    x_prefix = "* " if highlight_metric == "xAxis_offset_abs" else ""
    angle_prefix = "* " if highlight_metric == "angle_abs" else ""
    dist_prefix = "* " if highlight_metric == "dist" else ""
    if visible_now:
        put_line(f"{x_prefix}X-AXIS: {x_axis:.1f} mm", WHITE, 0.38, 1)
    else:
        put_line(f"{x_prefix}X-AXIS: -", WHITE, 0.38, 1)
    x_gate_line = _gate_line(x_gate, lambda v: f"{v:.1f}", "TARGET", x_axis, signed=True)
    if x_gate_line:
        x_trend = getattr(wm, "_align_metrics_trend", {}).get("x_axis")
        x_gate_color = GREEN if x_trend == 1 else RED if x_trend == -1 else WHITE
        put_line(x_gate_line, x_gate_color, 0.35, 1)
    if visible_now:
        put_line(f"{angle_prefix}ANGLE:  {wm.brick['angle']:.1f} deg", WHITE, 0.38, 1)
    else:
        put_line(f"{angle_prefix}ANGLE:  -", WHITE, 0.38, 1)
    angle_gate_line = _gate_line(angle_gate, lambda v: f"{v:.1f}", "TARGET", wm.brick["angle"])
    if angle_gate_line:
        put_line(angle_gate_line, WHITE, 0.35, 1)
    if visible_now:
        put_line(f"{dist_prefix}DIST:   {wm.brick['dist']:.0f} mm", WHITE, 0.38, 1)
    else:
        put_line(f"{dist_prefix}DIST:   -", WHITE, 0.38, 1)
    dist_gate_line = _gate_line(dist_gate, lambda v: f"{v:.1f}", "TARGET", wm.brick["dist"])
    if dist_gate_line:
        dist_trend = getattr(wm, "_align_metrics_trend", {}).get("dist")
        dist_gate_color = GREEN if dist_trend == 1 else RED if dist_trend == -1 else WHITE
        put_line(dist_gate_line, dist_gate_color, 0.35, 1)
    brick_conf = wm.brick.get("confidence")
    if brick_conf is None:
        brick_conf = 0.0
    if visible_now:
        put_line(f"CONF:   {brick_conf:.0f}%", WHITE, 0.38, 1)
        above_txt = "YES" if wm.brick.get("brickAbove") else "NO"
        below_txt = "YES" if wm.brick.get("brickBelow") else "NO"
    else:
        put_line("CONF:   -", WHITE, 0.38, 1)
        above_txt = "-"
        below_txt = "-"
    put_line(f"BRICK ABOVE: {above_txt}", WHITE, 0.38, 1)
    put_line(f"BRICK_BELOW: {below_txt}", WHITE, 0.38, 1)
    
    y_cur += 5
    put_line("--- LEIA TELEMETRY ---", WHITE, 0.35, 1)
    put_line(f"X:      {wm.x:.1f} mm", (200, 200, 255), 0.38, 1)
    put_line(f"Y:      {wm.y:.1f} mm", (200, 200, 255), 0.38, 1)
    put_line(f"THETA:  {wm.theta:.1f} deg", (200, 200, 255), 0.38, 1)
    put_line(f"LIFT:   {wm.lift_height:.0f} mm", (200, 200, 255), 0.38, 1)
    cam_times = getattr(wm, "_camera_frame_times", [])
    if cam_times:
        cam_str = " ".join(f"{sec % 60:02d}:{ms:03d}" for sec, ms in cam_times[-3:])
        unique_frames = getattr(wm, "_camera_unique_frames", None)
        if isinstance(unique_frames, int):
            cam_str = f"{cam_str} u={unique_frames}/3"
        cam_color = RED if getattr(wm, "_camera_dupe_ms", False) else WHITE
        put_line(f"CAMERA: {cam_str}", cam_color, 0.38, 1)

    # 6. Vision Info
    y_cur += 12
    if not wm.brick['visible']:
        put_line("VISION: SEARCHING", (0, 0, 255), 0.38, 1)
    
    y_cur += 8 # Spacer

    # 8. Extra Messages (Banners -> Moved to Sidebar)
    if extra_messages:
        y_cur = h - 20
        for msg in extra_messages:
             put_line(f"! {msg}", YELLOW, 0.4, 2)

    # 9. GEAR Display
    if gear:
        cv2.putText(frame, f"GEAR: {gear}", (x_base, h - 35), font, 0.4, WHITE, 2)
