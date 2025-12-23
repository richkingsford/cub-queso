"""Color-based dot detection and annotation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

# Hue range is 0-180 in OpenCV's HSV representation.
ColorRange = Tuple[Tuple[int, int, int], Tuple[int, int, int]]

# Multiple entries per color allow us to span wrap-around hues (e.g., red).
DEFAULT_COLOR_RANGES: Dict[str, Sequence[ColorRange]] = {
    "green": [((40, 70, 50), (85, 255, 255))],
    "purple": [((125, 50, 40), (155, 255, 255))],
    "red": [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "dark blue": [((100, 80, 40), (125, 255, 255))],
    "pink": [((150, 80, 100), (175, 255, 255))],
}

COLOR_TO_BGR = {
    "green": (0, 255, 0),
    "purple": (255, 0, 255),
    "red": (0, 0, 255),
    "dark blue": (255, 0, 0),
    "pink": (255, 192, 203),
}


@dataclass
class DetectedDot:
    color: str
    center: Tuple[int, int]
    radius: float
    confidence: float

    @property
    def label(self) -> str:
        pct = int(self.confidence * 100)
        return f"{self.color} ({pct}%)"


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for image annotation but is not installed.")


def detect_color_dots(
    image: np.ndarray,
    *,
    color_ranges: Dict[str, Sequence[ColorRange]] | None = None,
    min_area: float = 200.0,
    min_radius: float = 5.0,
) -> List[DetectedDot]:
    """Return detected colored dots within an image."""

    _require_cv2()

    if color_ranges is None:
        color_ranges = DEFAULT_COLOR_RANGES

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    detections: List[DetectedDot] = []

    for color_name, ranges in color_ranges.items():
        mask = None
        for lower, upper in ranges:
            current_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = current_mask if mask is None else cv2.bitwise_or(mask, current_mask)

        if mask is None:
            continue

        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < min_radius:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            circle_area = np.pi * radius * radius
            confidence = float(min(1.0, area / circle_area))
            detections.append(DetectedDot(color=color_name, center=(cx, cy), radius=radius, confidence=confidence))

    return detections


def annotate_image(image: np.ndarray, detections: Iterable[DetectedDot]) -> np.ndarray:
    """Draw circles and labels for the provided detections."""

    _require_cv2()
    annotated = image.copy()

    for detection in detections:
        color_bgr = COLOR_TO_BGR.get(detection.color, (255, 255, 255))
        center = (int(detection.center[0]), int(detection.center[1]))
        radius = int(max(1, round(detection.radius)))
        cv2.circle(annotated, center, radius, color_bgr, 2)
        text = detection.label
        text_origin = (center[0] - radius, max(20, center[1] - radius - 10))
        cv2.putText(
            annotated,
            text,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color_bgr,
            2,
            cv2.LINE_AA,
        )

    return annotated


def annotate_photo_directory(
    input_dir: str | Path,
    output_dir: str | Path = "images",
    *,
    overwrite: bool = True,
    color_ranges: Dict[str, Sequence[ColorRange]] | None = None,
) -> List[Path]:
    """Annotate every photo within a directory and write them to output_dir."""

    _require_cv2()

    src = Path(input_dir)
    if not src.exists():
        raise FileNotFoundError(f"Input directory not found: {src}")

    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    image_paths = [p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    written: List[Path] = []

    for image_path in sorted(image_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[annotate] Skipping {image_path.name}: unable to read file.")
            continue

        detections = detect_color_dots(image, color_ranges=color_ranges)
        annotated = annotate_image(image, detections)

        out_path = dest / image_path.name
        if not overwrite and out_path.exists():
            out_path = dest / f"{image_path.stem}_annotated{image_path.suffix}"

        success = cv2.imwrite(str(out_path), annotated)
        if not success:
            print(f"[annotate] Failed to write {out_path}")
            continue

        written.append(out_path)
        dot_count = len(detections)
        print(f"[annotate] {image_path.name} -> {out_path.name} ({dot_count} dot{'s' if dot_count != 1 else ''})")

    if not written:
        print("[annotate] No images were written. Check the input directory and file extensions.")

    return written
