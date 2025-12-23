"""Quick camera test: capture a single frame and save it to disk."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None


def capture_frame(camera_index: int, output_path: Path) -> Path:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is not installed. Install opencv-python to run this test.")

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open camera {camera_index}. Check the index or USB connection.")

    ok, frame = capture.read()
    capture.release()

    if not ok or frame is None:
        raise RuntimeError("Camera opened but failed to read a frame.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Failed to write image to {output_path}.")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Test that the camera is reachable and capture one frame.")
    parser.add_argument("--camera", type=int, default=1, help="Camera index to open (default: 1).")
    parser.add_argument(
        "--output",
        default="camera_test.jpg",
        help="Filename for the captured frame (default: camera_test.jpg).",
    )
    args = parser.parse_args()

    output_path = Path(__file__).resolve().parent / args.output

    try:
        saved_path = capture_frame(args.camera, output_path)
    except Exception as exc:  # pragma: no cover - CLI diagnostic
        print(f"[camera-test] Failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[camera-test] Saved frame from camera {args.camera} to {saved_path}")


if __name__ == "__main__":
    main()
