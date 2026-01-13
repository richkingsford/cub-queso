#!/usr/bin/env python3
import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from robot_control import Robot

ACTION_SEQUENCE = [
    {"label": "FORWARD", "cmd": "f", "measure": "mm"},
    {"label": "BACKWARD", "cmd": "b", "measure": "mm"},
    {"label": "LEFT", "cmd": "l", "measure": "deg"},
    {"label": "RIGHT", "cmd": "r", "measure": "deg"},
    {"label": "LIFT_UP", "cmd": "u", "measure": "height"},
    {"label": "LIFT_DOWN", "cmd": "d", "measure": "height"},
]

POWER_LEVELS = [0.2, 0.4, 0.6, 0.8]
DURATIONS = [1.0, 2.0]

SAMPLE_FIELDS = [
    "action",
    "power",
    "duration",
    "ticks",
    "real_mm",
    "real_deg",
    "real_height",
]

WORLD_MODEL_MOTION = Path(__file__).resolve().parent / "world_model_motion.json"

@dataclass
class CalibrationSample:
    action: str
    power: float
    duration: float
    ticks: int | None
    real_mm: float | None
    real_deg: float | None
    real_height: float | None


def prompt_text(prompt):
    try:
        return input(prompt)
    except EOFError:
        return ""


def prompt_number(prompt, allow_blank=True, number_type=float):
    while True:
        raw = prompt_text(prompt).strip()
        if not raw and allow_blank:
            return None
        try:
            return number_type(raw)
        except ValueError:
            print("Please enter a number.")


def clear_serial_buffer(robot):
    ser = getattr(robot, "ser", None)
    if ser is None:
        return
    try:
        ser.reset_input_buffer()
    except Exception:
        pass


def read_encoder_ticks(robot, timeout_s=1.0):
    ser = getattr(robot, "ser", None)
    if ser is None:
        return None
    try:
        ser.timeout = 0.1
    except Exception:
        pass
    end_time = time.time() + timeout_s
    last_ticks = None
    while time.time() < end_time:
        try:
            if not getattr(ser, "in_waiting", 0):
                time.sleep(0.02)
                continue
            line = ser.readline().decode("utf-8", "ignore").strip()
        except Exception:
            break
        if not line:
            continue
        matches = re.findall(r"-?\d+", line)
        if matches:
            last_ticks = int(matches[-1])
    return last_ticks


def run_motor(robot, cmd, speed, duration_s):
    interval = max(0.05, robot.CMD_DURATION / 1000.0)
    start = time.time()
    while time.time() - start < duration_s:
        robot.send_command(cmd, speed)
        time.sleep(interval)
    robot.stop()


def write_samples_csv(path, samples):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        for sample in samples:
            writer.writerow({
                "action": sample.action,
                "power": f"{sample.power:.2f}",
                "duration": f"{sample.duration:.2f}",
                "ticks": "" if sample.ticks is None else sample.ticks,
                "real_mm": "" if sample.real_mm is None else sample.real_mm,
                "real_deg": "" if sample.real_deg is None else sample.real_deg,
                "real_height": "" if sample.real_height is None else sample.real_height,
            })


def write_samples_jsonl(path, samples):
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample.__dict__) + "\n")


def _aggregate_ratio(samples, actions, field_name):
    num = 0.0
    den = 0.0
    for sample in samples:
        if sample.action not in actions:
            continue
        value = getattr(sample, field_name)
        if value is None or sample.ticks is None or sample.ticks == 0:
            continue
        num += abs(value)
        den += abs(sample.ticks)
    if den <= 0:
        return None, 0
    return num / den, den


def save_motion_calibration(path, mm_per_tick, deg_per_tick, mast_mm_per_tick, samples):
    payload = {
        "calibration": {
            "motion": {
                "mm_per_tick": mm_per_tick,
                "deg_per_tick": deg_per_tick,
                "mm_per_tick_mast": mast_mm_per_tick,
                "source": "setup_calibrate_motion.py",
                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                "samples": len(samples),
            }
        }
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Leia motion calibration (terminal guided)")
    parser.add_argument("--format", choices=("csv", "jsonl"), default="csv")
    parser.add_argument("--output", default=None, help="output path for samples")
    parser.add_argument("--world-model", default=str(WORLD_MODEL_MOTION), help="calibration output JSON")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = args.output
    if output_path is None:
        suffix = "csv" if args.format == "csv" else "jsonl"
        output_path = Path(__file__).resolve().parent / f"calibration_samples_{timestamp}.{suffix}"
    else:
        output_path = Path(output_path)

    print("\nLeia Motion Calibration")
    print("-----------------------")
    print("This routine will run each action at 20%, 40%, 60%, 80% for 1s and 2s.")
    print("After each run, we will capture encoder ticks and ask for real-world movement.")
    print("Press Enter to begin, or 'q' to quit.")
    if prompt_text("> ").strip().lower() == "q":
        return

    robot = Robot()
    samples = []

    try:
        for action in ACTION_SEQUENCE:
            label = action["label"]
            cmd = action["cmd"]
            measure = action["measure"]
            for power in POWER_LEVELS:
                for duration in DURATIONS:
                    print("\n---")
                    print(f"Action: {label} | Power: {int(power * 100)}% | Duration: {duration:.0f}s")
                    print("Make sure the robot is clear. Press Enter to run, 's' to skip, or 'q' to quit.")
                    choice = prompt_text("> ").strip().lower()
                    if choice == "q":
                        raise KeyboardInterrupt
                    if choice == "s":
                        continue

                    clear_serial_buffer(robot)
                    run_motor(robot, cmd, power, duration)

                    ticks = read_encoder_ticks(robot)
                    if ticks is None:
                        ticks = prompt_number("Enter encoder ticks (blank if unavailable): ", allow_blank=True, number_type=int)

                    tick_display = "N/A" if ticks is None else str(ticks)
                    print(f"Encoder ticks: {tick_display}")

                    real_mm = None
                    real_deg = None
                    real_height = None
                    if measure == "mm":
                        real_mm = prompt_number("Enter mm moved (use negative if needed): ")
                    elif measure == "deg":
                        real_deg = prompt_number("Enter degrees turned (use negative if needed): ")
                    elif measure == "height":
                        real_height = prompt_number("Enter mast lift mm (use negative if needed): ")

                    sample = CalibrationSample(
                        action=label,
                        power=power,
                        duration=duration,
                        ticks=ticks,
                        real_mm=real_mm,
                        real_deg=real_deg,
                        real_height=real_height,
                    )
                    samples.append(sample)

    except KeyboardInterrupt:
        print("\nCalibration interrupted.")
    finally:
        robot.stop()
        robot.close()

    if not samples:
        print("No samples captured.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "csv":
        write_samples_csv(output_path, samples)
    else:
        write_samples_jsonl(output_path, samples)

    mm_per_tick, _ = _aggregate_ratio(samples, {"FORWARD", "BACKWARD"}, "real_mm")
    deg_per_tick, _ = _aggregate_ratio(samples, {"LEFT", "RIGHT"}, "real_deg")
    mast_mm_per_tick, _ = _aggregate_ratio(samples, {"LIFT_UP", "LIFT_DOWN"}, "real_height")

    print("\nCalibration Summary")
    print("-------------------")
    print(f"mm_per_tick: {mm_per_tick}")
    print(f"deg_per_tick: {deg_per_tick}")
    print(f"mm_per_tick_mast: {mast_mm_per_tick}")

    world_model_path = Path(args.world_model)
    save_motion_calibration(world_model_path, mm_per_tick, deg_per_tick, mast_mm_per_tick, samples)
    print(f"Saved samples to {output_path}")
    print(f"Saved calibration to {world_model_path}")


if __name__ == "__main__":
    main()
