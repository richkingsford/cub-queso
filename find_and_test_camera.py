"""
Find and Test Camera (Saves to ~/leia)
--------------------------------------
1. Scans /dev/video* to find all attached cameras.
2. Attempts to open each one.
3. Warms up the sensor.
4. Saves a snapshot explicitly to /home/richkingsford/leia/
"""

import cv2
import glob
import os

# CONFIGURATION
SAVE_FOLDER = "/home/richkingsford/leia"

def find_cameras():
    # Find all devices matching /dev/video*
    devices = glob.glob("/dev/video*")
    devices.sort()
    return devices

def test_device(device_path):
    print(f"\n--- Testing Device: {device_path} ---")
    
    # Extract the index number (e.g., /dev/video0 -> 0)
    try:
        index = int(device_path.replace("/dev/video", ""))
    except ValueError:
        print(f"Skipping {device_path}: Could not determine index.")
        return

    cap = cv2.VideoCapture(index)
    
    # FORCE 640x480 (Safe Mode)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"FAILED: Could not open {device_path}")
        return

    # WARM UP
    print("Warming up camera (10 frames)...")
    for _ in range(10):
        cap.read()

    # CAPTURE
    ret, frame = cap.read()
    cap.release()

    if ret and frame is not None:
        # Create folder if it doesn't exist
        if not os.path.exists(SAVE_FOLDER):
            os.makedirs(SAVE_FOLDER)
            print(f"Created directory: {SAVE_FOLDER}")

        # Save the file
        filename = os.path.join(SAVE_FOLDER, f"camera_test_video{index}.jpg")
        cv2.imwrite(filename, frame)
        print(f"SUCCESS: Image saved to {filename}")
    else:
        print("FAILED: Camera opened but returned an empty frame.")

def main():
    print(f"Scanning for cameras (Target folder: {SAVE_FOLDER})...")
    devices = find_cameras()
    
    if not devices:
        print("No /dev/video devices found! Is the camera plugged in?")
        return

    print(f"Found {len(devices)} device(s): {', '.join(devices)}")
    
    for device in devices:
        test_device(device)

if __name__ == "__main__":
    main()