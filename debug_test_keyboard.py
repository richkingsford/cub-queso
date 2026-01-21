"""
debug_test_keyboard.py
----------------------
A simple script to "wiggle" the robot using terminal keys.
No recording, no vision. Just raw movement tests.

Controls:
  W/S: Forward/Backward
  A/D: Left/Right
  P/L: Mast Up/Down
  Q: Quit
"""
import sys
import threading
import time
import tty
import termios
import cv2
import numpy as np
from flask import Flask, Response
from helper_robot_control import Robot
from helper_brick_vision import BrickDetector
from telemetry_robot import WorldModel, MotionEvent, draw_telemetry_overlay

# --- CONFIG ---
GEAR_1_SPEED = 0.32
HEARTBEAT_TIMEOUT = 0.3 

class TestState:
    def __init__(self):
        self.running = True
        self.active_command = None
        self.active_speed = 0.0
        self.last_key_time = 0
        self.current_frame = None
        self.world = WorldModel()
        self.lock = threading.Lock()

app = Flask(__name__)

def generate_frames(state):
    while state.running:
        with state.lock:
            if state.current_frame is None:
                frame_to_send = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame_to_send, "NO SIGNAL", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                frame_to_send = state.current_frame.copy()
        
        (flag, encodedImage) = cv2.imencode(".jpg", frame_to_send)
        if not flag: continue
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')

@app.route("/")
def index():
    return "<html><body style='background:#111; color:#eee; text-align:center;'><h1>Robot Eyes (DEBUG)</h1><img src='/video_feed'></body></html>"

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(app_state_ref), mimetype="multipart/x-mixed-replace; boundary=frame")

def vision_thread(state):
    detector = BrickDetector(debug=True, speed_optimize=False)
    while state.running:
        found, angle, dist, offset_x, conf, h, brick_above, brick_below = detector.read()
        
        with state.lock:
            state.world.update_vision(found, dist, angle, conf, offset_x, h, brick_above, brick_below)
            if detector.current_frame is not None:
                frame = detector.current_frame.copy()
                reminders = ["W/A/S/D to move", "Q to quit"]
                draw_telemetry_overlay(frame, state.world, reminders=reminders)
                state.current_frame = frame
        time.sleep(0.05)

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def keyboard_thread(state):
    print("\n[WIGGLE TEST] Ready.")
    print("Hold W/A/S/D or P/L to move. Release to stop.")
    print("Press 'q' to quit.")
    
    while state.running:
        ch = getch().lower()
        with state.lock:
            state.last_key_time = time.time()
            if ch == 'q':
                state.running = False
                break
            
            if ch == 'w':
                state.active_command = 'b' # INVERTED
            elif ch == 's':
                state.active_command = 'f' # INVERTED
            elif ch == 'a':
                state.active_command = 'l'
            elif ch == 'd':
                state.active_command = 'r'
            elif ch == 'p':
                state.active_command = 'u'
            elif ch == 'l':
                state.active_command = 'd'
            
            # MOVEMENT

def main():
    state = TestState()
    robot = Robot()
    
    # Start vision and web server
    global app_state_ref
    app_state_ref = state
    
    threading.Thread(target=vision_thread, args=(state,), daemon=True).start()
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, 
                                          debug=False, use_reloader=False), 
                     daemon=True).start()
    
    print("\n[VISION] Stream started at http://<robot-ip>:5000")
    
    # Start keyboard thread
    kb_t = threading.Thread(target=keyboard_thread, args=(state,), daemon=True)
    kb_t.start()
    
    was_moving = False
    dt = 0.05 # 20Hz update
    
    try:
        while state.running:
            loop_start = time.time()
            
            with state.lock:
                # Heartbeat check
                if time.time() - state.last_key_time > HEARTBEAT_TIMEOUT:
                    state.active_command = None
                    state.active_speed = 0.0
                
                # Fixed speed based on Gear 1
                speed = GEAR_1_SPEED
                
                cmd = state.active_command
            
            if cmd and speed > 0:
                robot.send_command(cmd, speed)
                was_moving = True
                
                # TRACK MOTION for Telemetry
                atype = "unknown"
                if cmd == 'f': atype = "forward"
                elif cmd == 'b': atype = "backward"
                elif cmd == 'l': atype = "left_turn"
                elif cmd == 'r': atype = "right_turn"
                elif cmd == 'u': atype = "mast_up"
                elif cmd == 'd': atype = "mast_down"
                
                pwr = int(speed * 255)
                # Use current dt for duration
                evt = MotionEvent(atype, pwr, int(dt * 1000))
                with state.lock:
                    state.world.update_from_motion(evt)
            elif was_moving:
                robot.stop()
                was_moving = False
            
            # Rate Limiting
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
            
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        robot.stop()
        robot.close()
        print("\nWiggle Test Stopped.")

if __name__ == "__main__":
    main()
