#!/usr/bin/env python3
import argparse
import json
import sys
import termios
import tty
import time
from pathlib import Path

from helper_robot_control import Robot
from telemetry_robot import ROBOT_MODEL_FILE, DEFAULT_SPEED_MODEL, MIN_PWM, MAX_PWM


KEY_ORDER = [
    "w", "s", "r", "f", "t", "g",
    "a", "d", "q", "e", "z", "c",
    "u", "l",
]


def _get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            seq = sys.stdin.read(2)
            return ch + seq
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _load_model(path):
    model = DEFAULT_SPEED_MODEL
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                model = data
        except (OSError, json.JSONDecodeError):
            model = DEFAULT_SPEED_MODEL
    hotkeys = model.get("hotkey_speed_scores") or {}
    score_map = model.get("score_power_pwm") or {}
    if not hotkeys:
        hotkeys = DEFAULT_SPEED_MODEL["hotkey_speed_scores"]
    if not score_map:
        score_map = DEFAULT_SPEED_MODEL["score_power_pwm"]
    model["hotkey_speed_scores"] = dict(hotkeys)
    model["score_power_pwm"] = dict(score_map)
    return model


def _save_model(path, model):
    path.write_text(json.dumps(model, indent=2))


def _score_entry(score_map, score):
    entry = score_map.get(str(int(score)))
    if not isinstance(entry, dict):
        return None
    return entry


def _set_score_entry(score_map, score, power, pwm):
    score_map[str(int(score))] = {"power": float(power), "pwm": int(pwm)}


def _power_to_pwm(power):
    if power <= 0:
        return 0
    pwm = int(MIN_PWM + (MAX_PWM - MIN_PWM) * power)
    return min(pwm, MAX_PWM)


def _run_demo(robot, cmd, power, duration_s):
    interval = max(0.05, robot.CMD_DURATION / 1000.0)
    start = time.time()
    while time.time() - start < duration_s:
        robot.send_command(cmd, power)
        time.sleep(interval)
    robot.stop()


def _ordered_hotkeys(hotkeys):
    ordered = []
    seen = set()
    for key in KEY_ORDER:
        if key in hotkeys:
            ordered.append(key)
            seen.add(key)
    for key in sorted(hotkeys.keys()):
        if key not in seen:
            ordered.append(key)
    return ordered


def main():
    parser = argparse.ArgumentParser(description="Leia speed calibration (terminal guided)")
    parser.add_argument("--model", default=str(ROBOT_MODEL_FILE))
    parser.add_argument("--step", type=float, default=0.01, help="power adjustment per arrow key")
    parser.add_argument("--duration", type=float, default=0.75, help="demo duration in seconds")
    args = parser.parse_args()

    model_path = Path(args.model)
    model = _load_model(model_path)
    hotkeys = model["hotkey_speed_scores"]
    score_map = model["score_power_pwm"]
    key_list = _ordered_hotkeys(hotkeys)
    if not key_list:
        print("No hotkeys found in the model.")
        return

    print("\nLeia Speed Calibration")
    print("----------------------")
    print("n = next hotkey | d = demo act | ↑/↓ = adjust speed | q = quit")
    print("Note: adjustments change the score's speed for all hotkeys with that score.")
    print("")

    robot = Robot()
    idx = 0
    try:
        while True:
            key = key_list[idx]
            entry = hotkeys.get(key, {})
            cmd = entry.get("cmd", "")
            score = entry.get("score", 0)
            score_entry = _score_entry(score_map, score)
            if not score_entry:
                score_entry = {"power": 0.5, "pwm": 128}
                _set_score_entry(score_map, score, score_entry["power"], score_entry["pwm"])
                _save_model(model_path, model)

            power = float(score_entry.get("power", 0.0))
            pwm = int(score_entry.get("pwm", 0))
            print(f"[{idx + 1}/{len(key_list)}] Key {key.upper()} -> cmd '{cmd}' | score {score}% | power {power:.3f} | pwm {pwm}")
            print("Press a key: ", end="", flush=True)
            ch = _get_key()
            print("")
            if not ch:
                continue
            if ch in ("q", "Q"):
                break
            if ch in ("n", "N"):
                idx = (idx + 1) % len(key_list)
                continue
            if ch in ("d", "D"):
                _run_demo(robot, cmd, power, args.duration)
                continue
            if ch == "\x1b[A":  # up arrow
                power = min(1.0, power + args.step)
                pwm = _power_to_pwm(power)
                _set_score_entry(score_map, score, power, pwm)
                _save_model(model_path, model)
                print(f"Updated score {score}% -> power {power:.3f}, pwm {pwm}")
                continue
            if ch == "\x1b[B":  # down arrow
                power = max(0.0, power - args.step)
                pwm = _power_to_pwm(power)
                _set_score_entry(score_map, score, power, pwm)
                _save_model(model_path, model)
                print(f"Updated score {score}% -> power {power:.3f}, pwm {pwm}")
                continue
    finally:
        try:
            robot.stop()
        except Exception:
            pass
        try:
            robot.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
