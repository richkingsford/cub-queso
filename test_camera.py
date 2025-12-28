"""Quick USB camera test: capture a single frame and save it to disk."""

from __future__ import annotations

import argparse
import sys
import glob
import time
from pathlib import Path

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None


def list_video_devices():
    """Helper to list available video devices on Linux."""
    devices = glob.glob("/dev/video*")
    print(f"[camera-test] Found video devices: {', '.join(devices)}")
    return devices


def capture_frame(camera_index: int, output_path: Path) -> Path:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is not installed. Install opencv-python to run this test.")

    print(f"[camera-test] Attempting to open camera index: {camera_index}...")
    capture = cv2.VideoCapture(camera_index)
    
    # 1. FORCE COMPATIBLE RESOLUTION
    # This helps avoid errors if the camera defaults to an unsupported massive resolution.
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open camera {camera_index}. Check the index or USB connection.")

    # 2. WARM UP CAMERA
    # USB cams often need a few frames to adjust exposure/white balance.
    print("[camera-test] Camera opened. Warming up (skipping 10 frames)...")
    for _ in range(10):
        capture.read()

    # 3. CAPTURE THE REAL FRAME
    ok, frame = capture.read()
    capture.release()

    if not ok or frame is None:
        raise RuntimeError("Camera opened but failed to read a valid frame after warmup.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Failed to write image to {output_path}.")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Test that the camera is reachable and capture one frame.")
    # Default changed to -1 to trigger auto-detection logic below if needed
    parser.add_argument("--camera", type=int, default=0, help="Camera index to open (default: 0).")
    parser.add_argument(
        "--output",
        default="camera_test.jpg",
        help="Filename for the captured frame (default: camera_test.jpg).",
    )
    args = parser.parse_args()

    # Print available devices to help debug
    list_video_devices()

    output_path = Path(__file__).resolve().parent / args.output

    try:
        saved_path = capture_frame(args.camera, output_path)
    except Exception as exc:  # pragma: no cover - CLI diagnostic
        print(f"[camera-test] Failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[camera-test] SUCCESS! Saved frame from camera {args.camera} to {saved_path}")


if __name__ == "__main__":
    main()