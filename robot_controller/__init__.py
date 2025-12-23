"""Robot control helpers for dot-seeking jobs."""

from .job import DotJobOrchestrator, ColorObservation
from .servo import ServoController
from .vision import annotate_image, annotate_photo_directory, detect_color_dots

__all__ = [
	"DotJobOrchestrator",
	"ColorObservation",
	"ServoController",
	"detect_color_dots",
	"annotate_image",
	"annotate_photo_directory",
]
