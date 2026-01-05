"""
setup_check_camera.py
---------------------
Unified Camera Setup and Testing Utility.
1. Finds attached video devices (`/dev/video*`).
2. Tests opening them and verifies resolution capabilities.
3. Saves debug snapshots to `~/leia/debug_snaps`.
4. Tests write permissions and folder creation.
"""

import cv2
import glob
import os
import sys
import time
import argparse

# --- CONFIG ---
SAVE_FOLDER = os.path.join(os.getcwd(), "debug_captures")

def find_cameras():
    """Finds all /dev/video* devices."""
    devices = glob.glob("/dev/video*")
    devices.sort()
    return devices

def test_camera(device_path_or_idx):
    """
    Tests a specific camera device.
    Args:
        device_path_or_idx: e.g., "/dev/video0" or 0
    """
    print(f"\n--- Testing Camera: {device_path_or_idx} ---")
    
    # Resolve index
    if isinstance(device_path_or_idx, str):
        try:
            device_str = device_path_or_idx.replace("/dev/video", "")
            idx = int(device_str)
        except ValueError:
            print(f"[SKIP] Could not parse index from {device_path_or_idx}")
            return
    else:
        idx = int(device_path_or_idx)

    cap = cv2.VideoCapture(idx)
    
    # 1. Force 640x480 (Safe Mode)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print(f"[FAIL] Could not open camera index {idx}")
        return

    # 2. Warm Up
    print("[INFO] Warming up sensor (10 frames)...")
    for _ in range(10): 
        cap.read()
        
    # 3. Capture
    ret, frame = cap.read()
    
    # 4. Diagnostics
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Resolution: {width}x{height} @ {fps} FPS")
    
    cap.release()

    if ret and frame is not None:
        # Save Frame
        filename = f"cam_{idx}_test.jpg"
        full_path = os.path.join(SAVE_FOLDER, filename)
        
        cv2.putText(frame, f"CAM {idx}: {width}x{height}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        success = cv2.imwrite(full_path, frame)
        if success:
            print(f"[SUCCESS] Saved snapshot to: {full_path}")
        else:
            print(f"[FAIL] cv2.imwrite failed for {full_path}")
    else:
        print("[FAIL] Captured frame was empty.")

def check_filesystem():
    """Verifies output directory exists and is writable."""
    print(f"\n--- Checking Filesystem ---")
    print(f"Target: {SAVE_FOLDER}")
    
    if not os.path.exists(SAVE_FOLDER):
        try:
            os.makedirs(SAVE_FOLDER)
            print(f"[OK] Created directory.")
        except OSError as e:
            print(f"[CRITICAL] Failed to create directory: {e}")
            sys.exit(1)
    
    # Test Write
    test_file = os.path.join(SAVE_FOLDER, ".write_test")
    try:
        with open(test_file, "w") as f: f.write("ok")
        os.remove(test_file)
        print(f"[OK] Directory is writable.")
    except Exception as e:
        print(f"[CRITICAL] Directory NOT writable: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Leia Camera Setup Utility")
    parser.add_argument("--camera", type=int, help="Specific camera index to test")
    args = parser.parse_args()

    check_filesystem()

    if args.camera is not None:
        test_camera(args.camera)
    else:
        print(f"\n--- Scanning for Devices ---")
        devices = find_cameras()
        if not devices:
            print("[WARN] No /dev/video* devices found.")
        else:
            print(f"Found {len(devices)} devices: {', '.join(devices)}")
            for dev in devices:
                test_camera(dev)

    print("\n--- Done ---")

if __name__ == "__main__":
    main()
