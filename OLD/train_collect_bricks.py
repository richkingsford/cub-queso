"""
Data Collection Tool v4: Quick 5s Capture
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None


def collect_training_data(camera_index: int, output_dir: Path, duration: int, angle_label: str) -> int:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is not installed.")

    # Open the camera
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open camera {camera_index}. Try changing the index to 0 or 1.")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Get Ready! Recording '{angle_label}' in 2 seconds... ---")
    
    # Short 2-second countdown
    for i in range(2, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    print(f"--- GO! Recording for {duration} seconds. Shift the brick around! ---")

    start_time = time.time()
    count = 0
    
    while (time.time() - start_time) < duration:
        ok, frame = capture.read()
        if not ok or frame is None:
            continue

        # File naming: angle_timestamp_counter.jpg
        # Example: 10deg_16788822_0001.jpg
        timestamp = int(time.time() * 1000) 
        filename = f"{angle_label}deg_{timestamp}_{count:04d}.jpg"
        save_path = output_dir / filename
        
        cv2.imwrite(str(save_path), frame)
        count += 1
        
        cv2.imshow(f'Recording {angle_label}...', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture dataset images.")
    # UPDATED DEFAULT: Camera 0 (Try 1 if this is your webcam)
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    # UPDATED DEFAULT: 5 seconds
    parser.add_argument("--seconds", type=int, default=5, help="Duration per angle.")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent 
    output_path = base_path / "photos" / "angled_bricks_dataset"

    print("--- Brick Collector (Desk Mode) ---")
    print(f"Camera Index: {args.camera}")
    print(f"Duration: {args.seconds} seconds per angle")
    print(f"Saving to: {output_path}")
    print("Press Ctrl+C to quit.\n")

    try:
        while True:
            user_input = input("Enter angle (0, 22.5, 45...) or 'q': ").strip()
            
            if user_input.lower() == 'q':
                break
            
            # Sanitize input (2.5 -> 2pt5)
            clean_label = user_input.replace('.', 'pt')
            
            try:
                total = collect_training_data(args.camera, output_path, args.seconds, clean_label)
                print(f"--> Done. Captured {total} images for {user_input} degrees.\n")
            except RuntimeError as e:
                print(f"Error: {e}")
                print("Tip: Run 'python collect_bricks_angled_v4.py --camera 1' to try the other camera.")
                break

    except KeyboardInterrupt:
        print("\nSee you later!")
        sys.exit(0)

if __name__ == "__main__":
    main()