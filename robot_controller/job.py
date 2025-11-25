"""Job orchestration for the dot-finding task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional
import time

from .servo import ServoController


@dataclass
class ColorObservation:
    """Represents the color currently centered in the camera."""

    timestamp: float
    color: Optional[str]


class _HoldTracker:
    """Helper that fires when a color stays visible long enough."""

    def __init__(self, target_color: str, threshold_seconds: float) -> None:
        self.target_color = target_color
        self.threshold_seconds = threshold_seconds
        self._start: Optional[float] = None
        self._armed = True

    def update(self, observation: ColorObservation) -> bool:
        if observation.color != self.target_color:
            self._start = None
            self._armed = True
            return False

        if not self._armed:
            return False

        if self._start is None:
            self._start = observation.timestamp
            return False

        if observation.timestamp - self._start >= self.threshold_seconds:
            self._armed = False
            return True

        return False

    def reset(self) -> None:
        self._start = None
        self._armed = True


class DotJobOrchestrator:
    """Coordinates the dot-seeking job using a servo and color observations."""

    def __init__(
        self,
        servo: ServoController,
        *,
        logger: Callable[[str], None] = print,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.servo = servo
        self.logger = logger
        self.now = now
        self.targets = ["purple", "red", "dark blue"]
        self.current_target_index = 0
        self.state: str = "idle"
        self._current_direction = 1
        self._last_pan_direction = 1
        self._last_color: Optional[str] = None
        self._target_hold: Optional[_HoldTracker] = None
        self._start_hold = _HoldTracker("green", 2.0)
        self._end_hold = _HoldTracker("pink", 2.0)

    def _log(self, message: str) -> None:
        timestamp = self.now()
        self.logger(f"[{timestamp:0.3f}] {message}")

    def _log_color_if_changed(self, observation: ColorObservation) -> None:
        if observation.color != self._last_color and observation.color:
            self._last_color = observation.color
            self._log(f"Camera centered on the {observation.color} dot")

    @property
    def current_target(self) -> Optional[str]:
        if self.current_target_index >= len(self.targets):
            return None
        return self.targets[self.current_target_index]

    def _ensure_target_tracker(self) -> _HoldTracker:
        if self._target_hold is None:
            self._target_hold = _HoldTracker(self.targets[self.current_target_index], 1.0)
        return self._target_hold

    def _pan_for_target(self) -> None:
        next_direction = -self._current_direction
        self.servo.step(self._current_direction)
        if self._current_direction != self._last_pan_direction:
            self._log(
                f"Panning {'right' if self._current_direction > 0 else 'left'} looking for the {self.current_target} dot"
            )
        self._last_pan_direction = self._current_direction
        self._current_direction = next_direction

    def _handle_target_hit(self, target_color: str) -> None:
        self._log(f"Locked on the {target_color} dot; holding for 1 second")
        self.servo.pause(1.0)
        self._log(f"Jiggling on the {target_color} dot")
        self.servo.jiggle()
        self.servo.pause(1.0)
        self.current_target_index += 1
        self._target_hold = None

    def process_observation(self, observation: ColorObservation) -> None:
        """Consume a single camera observation and drive the job state machine."""

        self._log_color_if_changed(observation)

        if self.state == "idle":
            if self._start_hold.update(observation):
                self.state = "running"
                self._target_hold = _HoldTracker(self.targets[0], 1.0)
                self._log("Start job: green dot held for 2 seconds")
            return

        if self.state == "done":
            return

        if self._end_hold.update(observation):
            self.state = "done"
            self._log("End job: pink dot held for 2 seconds")
            return

        target = self.current_target
        if target is None:
            # All targets complete; wait for end signal.
            return

        hold_tracker = self._ensure_target_tracker()
        if hold_tracker.update(observation):
            self._handle_target_hit(target)
            return

        self._pan_for_target()

    def run(self, observations: Iterable[ColorObservation]) -> None:
        """Drive the orchestrator until the job completes.

        The generator typically comes from a color detector. A simulation can
        also feed pre-recorded values for offline testing.
        """

        for observation in observations:
            self.process_observation(observation)
            if self.state == "done":
                break
