"""
Lay Bricks V98 - "Smart Debug"
------------------------------
1. SMART SNAPS: Saves a debug photo every 3.0 seconds.
2. FILTERED: ONLY saves the photo if a brick is DETECTED. (No empty table shots).
3. CONTINUOUS: Runs forever.
"""
import time
import sys
import os
from brick_vision import BrickDetector

# --- CONFIG ---
SMART_DEBUG_MODE = True    # Set to True to save occasional verified snaps
SNAPSHOT_INTERVAL = 3.0    # Seconds between debug saves (if brick found)
DEBUG_FOLDER = "/home/richkingsford/leia/debug_snaps"

def main():
    print("--- LEIA: BRICK LAYING STARTED (V98) ---")
    
    # Ensure debug folder exists
    if not os.path.exists(DEBUG_FOLDER):
        try: os.makedirs(DEBUG_FOLDER)
        except: pass

    vision = BrickDetector(debug=True, save_folder=DEBUG_FOLDER)
    print(f"Vision Online. Smart Debug: {SMART_DEBUG_MODE}")
    
    last_save = time.time()
    snap_count = 1

    try:
        while True:
            found, angle, dist = vision.read()

            if found:
                print(f"--> TARGET ACQUIRED:  Dist: {int(dist)}mm  |  Angle: {int(angle)}°")
                
                # --- SMART SNAPSHOT LOGIC ---
                # Only save if: Debug is ON, Interval passed, AND Brick is FOUND.
                if SMART_DEBUG_MODE and (time.time() - last_save > SNAPSHOT_INTERVAL):
                    last_save = time.time()
                    snap_name = f"smart_snap_{snap_count:04d}.jpg"
                    snap_path = os.path.join(DEBUG_FOLDER, snap_name)
                    
                    if vision.save_frame(snap_path):
                        print(f"   [DEBUG] Valid Brick Saved: {snap_name}")
                        snap_count += 1
            else:
                print(f"\r... scanning ...", end="")
            
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping robot...")
    finally:
        if 'vision' in locals() and hasattr(vision, 'close'):
            vision.close()
        print("System shutdown.")

if __name__ == "__main__":
    main()