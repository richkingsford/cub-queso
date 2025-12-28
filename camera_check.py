"""
Camera Config Tuner (V90 - "Paranoid Mode")
-------------------------------------------
1. Checks if the folder exists and is writable.
2. Verifies immediately if the file was created on disk.
3. Prints ABSOLUTE paths for everything.
"""
import cv2
import time
import os
import sys

# --- TARGET FOLDER ---
# We use the absolute path to be 100% sure where files land.
SAVE_FOLDER = "/home/richkingsford/leia/debug_snaps"

print(f"\n--- DIAGNOSTIC START ---")
print(f"User: {os.getlogin()}")
print(f"CWD:  {os.getcwd()}")
print(f"Target: {SAVE_FOLDER}")

# 1. Force Create Directory
if not os.path.exists(SAVE_FOLDER):
    try:
        os.makedirs(SAVE_FOLDER)
        print(f"[OK] Created directory: {SAVE_FOLDER}")
    except OSError as e:
        print(f"[CRITICAL] Failed to create directory: {e}")
        sys.exit(1)
else:
    print(f"[OK] Directory exists.")

# 2. Test Write Permissions
test_file = os.path.join(SAVE_FOLDER, "write_test.txt")
try:
    with open(test_file, "w") as f: f.write("ok")
    os.remove(test_file)
    print(f"[OK] Directory is writable.")
except Exception as e:
    print(f"[CRITICAL] Directory is NOT writable: {e}")
    sys.exit(1)

CONFIGS = [
    {"name": "1_Default",       "exp": -1,   "bri": -1,   "sat": -1},
    {"name": "2_Low_Exposure",  "exp": -6,   "bri": 100,  "sat": -1}, 
    {"name": "3_High_Sat",      "exp": -4,   "bri": 110,  "sat": 200}, 
    {"name": "4_High_Bright",   "exp": -2,   "bri": 150,  "sat": 150}, 
    {"name": "5_Low_Bright",    "exp": -7,   "bri": 80,   "sat": 150}, 
]

def set_camera_props(cap, conf):
    if conf["name"] != "1_Default":
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # Manual
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) # Auto

    if conf["exp"] != -1: cap.set(cv2.CAP_PROP_EXPOSURE, conf["exp"])
    if conf["bri"] != -1: cap.set(cv2.CAP_PROP_BRIGHTNESS, conf["bri"])
    if conf["sat"] != -1: cap.set(cv2.CAP_PROP_SATURATION, conf["sat"])
    time.sleep(1.5)

def main():
    print(f"--- STARTING CAMERA CAPTURE ---")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[CRITICAL] Could not open camera.")
        sys.exit(1)

    # Set Resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    for conf in CONFIGS:
        print(f"\nTesting Profile: {conf['name']}...")
        set_camera_props(cap, conf)
        
        # Flush
        for _ in range(5): cap.read()
        
        ret, frame = cap.read()
        if not ret:
            print("  [ERROR] Camera returned no frame (ret=False).")
            continue

        # Label
        label = f"Config: {conf['name']}"
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # SAVE
        filename = f"config_{conf['name']}.jpg"
        full_path = os.path.join(SAVE_FOLDER, filename)
        
        success = cv2.imwrite(full_path, frame)
        
        if success:
            if os.path.exists(full_path):
                print(f"  -> [SAVED] {full_path}")
            else:
                print(f"  -> [WEIRD] imwrite said True, but file not found at {full_path}")
        else:
            print(f"  -> [FAIL] cv2.imwrite returned False for {full_path}")

    cap.release()
    print("\n--- DONE ---")

if __name__ == "__main__":
    main()