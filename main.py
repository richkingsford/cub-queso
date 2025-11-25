"""Entry point for the dot-seeking robot demo."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List

from robot_controller import ColorObservation, DotJobOrchestrator, ServoController

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency for diagnostics
    cv2 = None


@dataclass
class _SimulatedClock:
    value: float = 0.0

    def __call__(self) -> float:
        return self.value


class _TestServo:
    """Servo stub used by lightweight startup tests."""

    def __init__(self) -> None:
        self.movements = []
        self.pauses = []
        self.jiggle_count = 0

    def step(self, direction: int) -> float:  # type: ignore[override]
        self.movements.append(direction)
        return 0.0

    def pause(self, seconds: float) -> None:
        self.pauses.append(seconds)

    def jiggle(self, spread: float = 5.0, repetitions: int = 2, pause: float = 0.1) -> None:
        self.jiggle_count += 1


def demo_observations() -> List[ColorObservation]:
    """Provide a deterministic observation timeline for demonstration."""

    return [
        ColorObservation(0.0, None),
        ColorObservation(0.5, "green"),
        ColorObservation(2.6, "green"),  # triggers start
        ColorObservation(3.2, "purple"),
        ColorObservation(4.3, "purple"),  # triggers purple hold
        ColorObservation(5.0, "red"),
        ColorObservation(6.2, "red"),  # triggers red hold
        ColorObservation(7.1, "dark blue"),
        ColorObservation(8.3, "dark blue"),  # triggers blue hold
        ColorObservation(9.6, "pink"),
        ColorObservation(11.7, "pink"),  # triggers end
    ]


def live_color_stream() -> Iterable[ColorObservation]:
    """Yield observations entered manually through stdin."""

    print("Enter the detected color name (blank to exit):")
    while True:
        color = input("> ").strip().lower()
        if not color:
            break
        yield ColorObservation(time.monotonic(), color)


def _run_unit_tests() -> None:
    """Execute quick smoke tests every time the script starts."""

    def _assert(condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)

    clock = _SimulatedClock()
    servo = _TestServo()
    orchestrator = DotJobOrchestrator(servo, logger=lambda _: None, now=clock)

    timeline = [
        ColorObservation(0.0, "green"),
        ColorObservation(2.1, "green"),  # start
        ColorObservation(3.0, "purple"),
        ColorObservation(4.1, "purple"),  # purple hit
        ColorObservation(5.0, "red"),
        ColorObservation(6.1, "red"),  # red hit
        ColorObservation(7.0, "dark blue"),
        ColorObservation(8.1, "dark blue"),  # blue hit
        ColorObservation(9.0, "pink"),
        ColorObservation(11.1, "pink"),  # end
    ]

    for obs in timeline:
        clock.value = obs.timestamp
        orchestrator.process_observation(obs)

    _assert(orchestrator.state == "done", "Orchestrator should finish after end signal.")
    _assert(orchestrator.current_target is None, "All targets should be completed.")
    _assert(servo.jiggle_count == 3, "Each target should trigger a jiggle.")
    _assert(len(servo.pauses) >= 3, "Servo pauses should be recorded for target holds.")

    print("[startup] Smoke tests passed.")


def _check_camera(index: int = 1) -> None:
    """Best-effort check that the desired camera is reachable."""

    if cv2 is None:
        print("[startup] OpenCV not installed; skipping camera check.")
        return

    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        print(f"[startup] Warning: unable to open camera {index}. Check USB connections or permissions.")
        capture.release()
        return

    ok, frame = capture.read()
    capture.release()
    if ok and frame is not None:
        height, width = frame.shape[:2]
        print(f"[startup] Camera {index} reachable. Frame size: {width}x{height}.")
    else:
        print(f"[startup] Warning: camera {index} opened but no frame was read.")


def run_demo() -> None:
    clock = _SimulatedClock()
    servo = ServoController()
    orchestrator = DotJobOrchestrator(servo, now=clock)

    for observation in demo_observations():
        clock.value = observation.timestamp
        orchestrator.process_observation(observation)


def run_live() -> None:
    servo = ServoController()
    orchestrator = DotJobOrchestrator(servo)
    orchestrator.run(live_color_stream())


def main() -> None:
    try:
        _run_unit_tests()
    except AssertionError as exc:
        print(f"[startup] Unit tests failed: {exc}", file=sys.stderr)
        sys.exit(1)

    _check_camera(index=1)

    parser = argparse.ArgumentParser(description="Run the dot job orchestrator")
    parser.add_argument(
        "--mode",
        choices=["demo", "live"],
        default="demo",
        help="Run a prerecorded demo or interactively type observed colors.",
    )
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo()
    else:
        run_live()


if __name__ == "__main__":
    main()
