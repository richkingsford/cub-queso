"""Manual camera sweep utility matching the keyboard-driven servo test.

Hold the left/right arrow keys to nudge the servo speed, press "s" to stop,
"q" to quit, and the script captures a JPEG every second into the photos
folder.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import keyboard  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    keyboard = None

from robot_controller import ServoController

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_STEP_US = 50  # microsecond increment per arrow tap (matches test_arduino)


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def _clear_directory(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return

    for entry in path.iterdir():
        if entry.is_file() or entry.is_symlink():
            entry.unlink(missing_ok=True)
        elif entry.is_dir():
            shutil.rmtree(entry)


def _save_frame(frame, dest_dir: Path, counter: int) -> Optional[Path]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = dest_dir / f"frame_{timestamp}_{counter:04d}.jpg"
    if cv2.imwrite(str(filename), frame):
        return filename
    return None


def _pulse_to_drive(pulse: int, servo: ServoController) -> float:
    stop = servo.stop_us
    if pulse == stop:
        return 0.0
    if pulse > stop:
        span = max(1, servo.forward_us - stop)
        return min(1.0, (pulse - stop) / span)
    span = max(1, stop - servo.reverse_us)
    return -min(1.0, (stop - pulse) / span)


def run_loop(
    camera_index: int,
    output_dir: Path,
    interval: float,
    *,
    min_pulse: int,
    max_pulse: int,
    step_us: int,
) -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required but not installed. Run 'pip install opencv-python'.")
    if keyboard is None:
        raise RuntimeError("The 'keyboard' package is required. Install it with 'pip install keyboard'.")

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open camera {camera_index}. Check the index or USB connection.")

    servo = ServoController(reverse_us=min_pulse, forward_us=max_pulse)
    print("[servo] Centering at neutral (stop) before manual control...")
    servo.drive(0.0)
    time.sleep(0.5)
    pulse_us = servo.stop_us
    current_drive = _pulse_to_drive(pulse_us, servo)

    last_capture = 0.0
    frame_counter = 0

    print("Manual camera sweep running:")
    print(f"  Pulse range: {min_pulse}-{max_pulse} µs, step {step_us} µs")
    print("  Left arrow  -> decrease pulse (spin reverse)")
    print("  Right arrow -> increase pulse (spin forward)")
    print("  S key       -> stop / neutral")
    print("  Q key       -> quit")
    print()

    try:
        while True:
            now = time.monotonic()
            changed = False

            if keyboard.is_pressed("left"):
                new_pulse = max(min_pulse, pulse_us - step_us)
                if new_pulse != pulse_us:
                    pulse_us = new_pulse
                    changed = True
                    print(f"[input] Pulse -> {pulse_us}")
            elif keyboard.is_pressed("right"):
                new_pulse = min(max_pulse, pulse_us + step_us)
                if new_pulse != pulse_us:
                    pulse_us = new_pulse
                    changed = True
                    print(f"[input] Pulse -> {pulse_us}")
            elif keyboard.is_pressed("s"):
                if pulse_us != servo.stop_us:
                    pulse_us = servo.stop_us
                    changed = True
                    print("[input] Stop (neutral)")
            elif keyboard.is_pressed("q"):
                print("[input] Exit requested")
                break

            if changed:
                drive = _pulse_to_drive(pulse_us, servo)
                if abs(drive - current_drive) > 1e-4:
                    servo.drive(drive)
                    current_drive = drive

            if now - last_capture >= interval:
                ok, frame = capture.read()
                if not ok or frame is None:
                    print("[camera] Warning: failed to read frame; retrying...")
                    time.sleep(0.1)
                    continue

                last_capture = now
                saved = _save_frame(frame, output_dir, frame_counter)
                frame_counter += 1
                if saved:
                    print(f"[camera] Saved {saved.name}")
                else:
                    print("[camera] Failed to save frame.")

            time.sleep(0.01)
    finally:
        capture.release()
        try:
            servo.drive(0.0)
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual camera sweep controller")
    parser.add_argument("--camera", type=int, default=1, help="Camera index to open (default: 1).")
    parser.add_argument(
        "--photos-dir",
        default=str(BASE_DIR / "photos"),
        help="Directory where captured frames are stored (default: repo ./photos).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between captures (default: 1.0).",
    )
    parser.add_argument(
        "--min-pulse",
        type=int,
        default=1000,
        help="Lower PWM bound (microseconds) for reverse (default: 1000).",
    )
    parser.add_argument(
        "--max-pulse",
        type=int,
        default=2000,
        help="Upper PWM bound (microseconds) for forward (default: 2000).",
    )
    parser.add_argument(
        "--step-us",
        type=int,
        default=DEFAULT_STEP_US,
        help="PWM increment in microseconds per arrow press (default: 50).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.min_pulse >= args.max_pulse:
        print("[error] --min-pulse must be less than --max-pulse", file=sys.stderr)
        sys.exit(1)

    if not (args.min_pulse < 1500 < args.max_pulse):
        print("[warn] Expected stop pulse (1500) to lie between min and max pulses.")

    output_dir = _resolve_path(args.photos_dir)
    _clear_directory(output_dir)

    try:
        run_loop(
            args.camera,
            output_dir,
            args.interval,
            min_pulse=args.min_pulse,
            max_pulse=args.max_pulse,
            step_us=max(1, args.step_us),
        )
    except Exception as exc:  # pragma: no cover - CLI diagnostics
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
