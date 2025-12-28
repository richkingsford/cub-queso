"""
lay_bricks.py
-------------
The "Manager". It sets up the hardware and assigns the current task.
"""
from brick_vision import BrickDetector
# Assuming your hardware driver is saved here:
from robot_control import Robot 
import maneuvers
import time
import shutil
import os
import threading
import cv2
from flask import Flask, Response

# --- FLASK CONFIG ---
app = Flask(__name__)
outputFrame = None
lock = threading.Lock()

def start_server():
    # Run Flask in a separate thread
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

def generate():
    # Stream the outputFrame as MJPEG
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')

@app.route("/")
def index():
    # Simple HTML page to view the stream
    return "<html><body><h1>Robot Vision</h1><img src='/video_feed'></body></html>"

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

def main():
    global outputFrame, lock
    
    # 1. Initialize Hardware
    print("--- INITIALIZING ROBOT ---")
    
    # SYSTEM MODE:
    # True  = Save Debug Screenshots (slower)
    # False = Production Mode (faster, no saving)
    DEBUG_MODE = True
    
    robot = Robot()
    
    # Start Web Server
    print("[SYSTEM] Starting Web Server on port 5000...")
    t = threading.Thread(target=start_server, daemon=True)
    t.start()
    
    if DEBUG_MODE:
        print("[SYSTEM] DEBUG MODE ON - Clearing old captures...")
        if os.path.exists("debug_captures"):
            shutil.rmtree("debug_captures")
        os.makedirs("debug_captures")
        
        print("[SYSTEM] Saving Vision Captures to 'debug_captures'...")
        vision = BrickDetector(debug=True, save_folder="debug_captures")
    else:
        print("[SYSTEM] PRODUCTION MODE - Max Speed")
        vision = BrickDetector(debug=False, save_folder=None)

    print("--- STARTING APPROACH ---")
    print("Press Ctrl+C to abort.")

    try:
        while True:
            # We delegate the actual movement logic to our new file
            maneuvers.crawl_forward_if_aligned(robot, vision)
            
            # --- WEB STREAM UPDATE ---
            if vision.current_frame is not None:
                with lock:
                    outputFrame = vision.current_frame.copy()
            
            # A tiny sleep keeps the loop running smoothly without
            # overwhelming the CPU (approx 60-100Hz update rate)
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n--- EMERGENCY STOP ---")
        robot.stop()
        vision.close()

if __name__ == "__main__":
    main()