"""Servo control helpers for the dot job.

The :class:`ServoController` wraps the low-level servo interface so the job
orchestrator can issue movements without worrying about hardware specifics.
The controller keeps the current angle in memory and clamps values between
`min_angle` and `max_angle` before sending them to the attached transport.

The default transport simply logs intended angles, which keeps the module
hardware-agnostic for environments where an Arduino is not attached. To drive
an Arduino Uno with a micro servo on pin 5, provide a transport callable that
accepts the numeric angle and writes it to your servo library of choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable


Transport = Callable[[float], None]


def _noop_transport(angle: float) -> None:
    """Fallback transport used in non-hardware environments."""
    print(f"[servo] target angle -> {angle:.1f}°")


@dataclass
class ServoController:
    """Simple servo wrapper for a single micro servo on pin 5.

    Parameters
    ----------
    min_angle:
        Lower bound for servo rotation.
    max_angle:
        Upper bound for servo rotation.
    step_degrees:
        Default delta used when stepping left or right.
    transport:
        Callable that receives the target angle. Replace this with your
        hardware-specific writer (for example, a pyfirmata call).
    """

    min_angle: float = 0.0
    max_angle: float = 180.0
    step_degrees: float = 5.0
    transport: Transport = field(default_factory=lambda: _noop_transport)

    def __post_init__(self) -> None:
        self._angle = 90.0
        self.transport(self._angle)

    @property
    def angle(self) -> float:
        return self._angle

    def move_to(self, angle: float) -> float:
        """Move to an absolute angle, clamped to the configured range."""
        clamped = max(self.min_angle, min(self.max_angle, angle))
        self._angle = clamped
        self.transport(clamped)
        return clamped

    def step(self, direction: int) -> float:
        """Incrementally move the servo in a direction.

        A positive direction pans right, a negative direction pans left.
        """

        if direction == 0:
            return self._angle
        delta = self.step_degrees if direction > 0 else -self.step_degrees
        return self.move_to(self._angle + delta)

    def jiggle(self, spread: float = 5.0, repetitions: int = 2, pause: float = 0.1) -> None:
        """Perform a small wiggle to draw attention to the current dot."""
        base = self._angle
        for _ in range(repetitions):
            self.move_to(base + spread)
            time.sleep(pause)
            self.move_to(base - spread)
            time.sleep(pause)
        self.move_to(base)

    def pause(self, seconds: float) -> None:
        """Hold position for the requested duration."""
        time.sleep(seconds)
