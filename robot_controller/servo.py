"""Servo control helpers for the dot job, using a continuous servo via Arduino.

This implementation assumes:
- An Arduino Uno is connected over USB (default COM6).
- A continuous rotation micro servo is wired to pin 5 on the Arduino.
- The Arduino sketch listens on Serial and receives PWM pulse widths (microseconds)
  as newline-terminated integers, e.g. "1500\n", and writes them to the servo.

The ServoController API is kept compatible with the original design so that
DotJobOrchestrator can continue to call:

    servo.step(direction)
    servo.jiggle(...)
    servo.pause(...)

Internally we treat the "angle" as a virtual value and translate movement
requests into short nudges of the continuous servo (forward or reverse) by
sending appropriate PWM values to the Arduino.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Optional

import serial  # requires `pip install pyserial`


@dataclass
class ServoController:
    """Servo wrapper for a single continuous micro servo on pin 5 (via Arduino).

    Parameters
    ----------
    port:
        Serial port for the Arduino (e.g. "COM6" on Windows).
    baudrate:
        Serial baud rate; must match the Arduino sketch.
    stop_us:
        Pulse width (microseconds) that stops the continuous servo.
    forward_us:
        Pulse width that spins the servo in the "forward" direction.
    reverse_us:
        Pulse width that spins the servo in the "reverse" direction.
    step_degrees:
        Virtual step size for internal angle bookkeeping. This is only used
        to decide whether to nudge forward or backward when `step()` is called.
    nudge_seconds:
        How long each nudge should last before the servo is stopped again.
    """

    port: str = "COM6"
    baudrate: int = 115200

    stop_us: int = 1500
    forward_us: int = 1700
    reverse_us: int = 1300

    step_degrees: float = 5.0
    nudge_seconds: float = 0.15

    # internal state
    _serial: Optional[serial.Serial] = field(init=False, default=None)
    _angle: float = field(init=False, default=90.0)
    _current_pwm: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        # Open serial connection to the Arduino
        self._serial = serial.Serial(self.port, self.baudrate, timeout=1)
        # Give the Arduino a moment to reset
        time.sleep(2.0)

        # Initialize the continuous servo in a "stopped" state
        self._send_pulse(self.stop_us)
        self._angle = 90.0
        self._current_pwm = self.stop_us
        print(f"[servo] Connected to Arduino on {self.port} at {self.baudrate} baud.")

    # ---- Public API expected by DotJobOrchestrator -------------------------

    @property
    def angle(self) -> float:
        """Virtual angle (for compatibility with the original design)."""
        return self._angle

    def move_to(self, angle: float) -> float:
        """Move towards a virtual absolute angle.

        Since this is a continuous servo, there is no true "angle".
        We instead:
        - Compare the new target to the current virtual angle.
        - Nudge the servo forward or backward briefly.
        - Update the stored virtual angle.
        """
        # Clamp to a plausible range just to keep things sane
        clamped = max(0.0, min(180.0, angle))
        delta = clamped - self._angle
        self._angle = clamped

        if delta > 0:
            # Target is to the "right" -> nudge forward
            self._nudge(self.forward_us)
        elif delta < 0:
            # Target is to the "left" -> nudge reverse
            self._nudge(self.reverse_us)
        else:
            # No change -> stop
            self._send_pulse(self.stop_us)

        return self._angle

    def step(self, direction: int) -> float:
        """Incrementally move the servo in a direction.

        A positive direction pans right, a negative direction pans left.
        direction == 0 stops the servo.
        """
        if direction == 0:
            self._send_pulse(self.stop_us)
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

    def drive(self, direction: float) -> None:
        """Continuously drive the servo until another call changes direction."""

        direction = max(-1.0, min(1.0, direction))
        if abs(direction) < 1e-3:
            pwm = self.stop_us
        elif direction > 0:
            pwm = int(self.stop_us + (self.forward_us - self.stop_us) * direction)
        else:
            pwm = int(self.stop_us + (self.reverse_us - self.stop_us) * (-direction))

        if pwm != self._current_pwm:
            self._send_pulse(pwm)
            self._current_pwm = pwm

    def pause(self, seconds: float) -> None:
        """Hold position for the requested duration."""
        # For a continuous servo this just means "stay stopped" for a while.
        self._send_pulse(self.stop_us)
        time.sleep(seconds)

    # ---- Internal helpers --------------------------------------------------

    def _send_pulse(self, microseconds: int) -> None:
        """Send a PWM pulse width (in microseconds) to the Arduino."""
        if not self._serial:
            raise RuntimeError("Serial connection not initialized.")

        msg = f"{microseconds}\n"
        self._serial.write(msg.encode("ascii"))
        self._serial.flush()
        self._current_pwm = microseconds
        # Optional debug:
        # print(f"[servo] PWM -> {microseconds} µs")

    def _nudge(self, pwm: int) -> None:
        """Briefly move in one direction, then stop."""
        self._send_pulse(pwm)
        time.sleep(self.nudge_seconds)
        self._send_pulse(self.stop_us)
