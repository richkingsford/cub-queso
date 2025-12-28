"""
VISION TUNER
------------
1. Loads the latest snapshot from 'debug_snaps'.
2. Prints the HSV color of the center pixel (Sampling the brick).
3. Generates 'analysis_mask.jpg' (The Black & White filter view).
4. Generates 'analysis_contours.jpg' (What shapes it found).
"""
import cv2
import numpy as np
import os
import glob

# --- CONFIG ---
SNAP_FOLDER = "/home/richkingsford/leia/debug_snaps"

def main():
    # 1. Find the latest snapshot
    list_of_files = glob.glob(os.path.join(SNAP_FOLDER, '*.jpg'))
    if not list_of_files:
        print("No snapshots found! Run lay_bricks.py first.")
        return
    
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Analyzing Image: {latest_file}")
    
    frame = cv2.imread(latest_file)
    h, w = frame.shape[:2]

    # 2. Sample Center Pixel (Assume you pointed the camera at the brick)
    #    This tells us the EXACT color of the brick in your room.
    center_y, center_x = h // 2, w // 2
    pixel_bgr = frame[center_y, center_x]
    pixel_hsv = cv2.cvtColor(np.uint8([[pixel_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    
    print(f"\n--- COLOR PROBE (Center Pixel) ---")
    print(f"HSV Value:  [{pixel_hsv[0]}, {pixel_hsv[1]}, {pixel_hsv[2]}]")
    print(f"  > Hue (Color): {pixel_hsv[0]} (0-180)")
    print(f"  > Sat (Richness): {pixel_hsv[1]} (0-255)")
    print(f"  > Val (Bright): {pixel_hsv[2]} (0-255)")
    print("----------------------------------\n")

    # 3. Apply the RED Filter (The logic from brick_vision.py)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Red Range 1 (0-10)
    mask1 = cv2.inRange(hsv, np.array([0, 100, 50]), np.array([10, 255, 255]))
    # Red Range 2 (170-180)
    mask2 = cv2.inRange(hsv, np.array([170, 100, 50]), np.array([180, 255, 255]))
    
    # Combine
    mask = mask1 + mask2
    
    # Save the Mask
    cv2.imwrite("analysis_mask.jpg", mask)
    print(f"Saved 'analysis_mask.jpg' -> Check this! Is the brick white and floor black?")

    # 4. Find Contours (Debug Shape Detection)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours.")
    
    debug_img = frame.copy()
    cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > 1000:
            print(f"  > Contour {i}: Area={int(area)} (Large enough)")
        else:
            print(f"  > Contour {i}: Area={int(area)} (Too small - noise)")

    cv2.imwrite("analysis_contours.jpg", debug_img)
    print("Saved 'analysis_contours.jpg' -> Shows what shapes were detected.")

if __name__ == "__main__":
    main()