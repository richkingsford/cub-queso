"""
# train_demo_keyboard.py
-----------------------
Keyboard-based version of the demo recorder.
Runs on the Jetson. 
Uses W/A/S/D/P/L for slow, precise control.
NON-STICKY: Robot only moves while keys are held/tapped.
"""
import sys
import threading
import time
import os
import tty
import termios
import cv2
from flask import Flask, Response

from robot_control import Robot
from train_brick_vision import BrickDetector
from robot_leia_telemetry import WorldModel, TelemetryLogger, MotionEvent, ObjectiveState, draw_telemetry_overlay
 
# --- CONFIG ---
LOG_RATE_HZ = 10
WEB_PORT = 5000
GEAR_1_SPEED = 0.32  # 4x faster per user request
GEAR_9_SPEED = 1.0   # 100% capacity
HEARTBEAT_TIMEOUT = 0.3 # Stop if no key for 0.3s

# --- FLASK ---
flask_app = Flask(__name__)
# Global-ish for generator
_stream_state = None 

def generate_frames():
    while True:
        if _stream_state is None:
            time.sleep(0.1)
            continue
            
        with _stream_state.lock:
            if _stream_state.current_frame is None:
                frame_to_send = None
            else:
                frame_to_send = _stream_state.current_frame.copy()
        
        if frame_to_send is None:
            time.sleep(0.05)
            continue

        (flag, encodedImage) = cv2.imencode(".jpg", frame_to_send)
        if not flag:
            continue
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')

@flask_app.route("/")
def index():
    return "<html><body style='background:#111; color:#eee; font-family:sans-serif; text-align:center;'><h1>Robot Eyes (Keyboard Mode)</h1><img src='/video_feed' style='border:2px solid #555; border-radius:10px;'></body></html>"

@flask_app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

class AppState:
    def __init__(self):
        self.running = True
        self.active_command = None
        self.active_speed = 0.0
        self.last_key_time = 0
        
        # Job Status
        self.job_success = False
        self.job_success_timer = 0
        self.job_start = False
        self.job_start_timer = 0
        self.job_abort = False
        self.job_abort_timer = 0
        
        # Speed Gears
        self.current_gear = 1 # 1-9
        
        self.lock = threading.Lock()
        
        # Session Setup
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(os.getcwd(), "demos", f"kbd_{timestamp}")
        if not os.path.exists(self.session_dir):
            os.makedirs(self.session_dir)
            
        print(f"[SESSION] Recording Keyboard Demo to: {self.session_dir}")
        print("CONTROLS (Hold or tap rapidly):")
        print("  W/S: Forward/Backward          |  A/D: Turn Left/Right")
        print("  P/L: Lift Up/Down")
        print("  Y: Start Job           |  K: Success (End)")
        print("  J: Abort Job           |  X: Cycle Objective")
        print("  1-9: Change Speed Gear")
        print("  Q: Quit")
        
        # Telemetry
        self.world = WorldModel()
        log_path = os.path.join(self.session_dir, "a_log.json")
        self.logger = TelemetryLogger(log_path)
        self.vision = None
        self.robot = None
        
        # Video
        self.current_frame = None

def getch():
    """Reads a single character from stdin in raw mode."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def keyboard_thread(app_state):
    while app_state.running:
        ch = getch().lower()
        
        with app_state.lock:
            app_state.last_key_time = time.time()
            if ch == 'q':
                app_state.running = False
                break
            
            # MOVEMENT (Heartbeat triggers)
            if ch == 'w':
                app_state.active_command = 'f'
            elif ch == 's':
                app_state.active_command = 'b'
            elif ch == 'a':
                app_state.active_command = 'l'
            elif ch == 'd':
                app_state.active_command = 'r'
            elif ch == 'p':
                app_state.active_command = 'u'
            elif ch == 'l':
                app_state.active_command = 'd'
            
            # JOB CONTROL (Single Taps)
            # MISSION STATE CONTROL
            elif ch == 'r':
                app_state.world.reset_mission()
                app_state.job_start = True
                app_state.job_start_timer = time.time()
                app_state.job_success = False
                app_state.job_abort = False
                evt = MotionEvent("JOB_START", 0, 0)
                app_state.world.update_from_motion(evt)
                print(f"\n[EVENT] Objective: FIND (New Job Started)")
            elif ch == 't':
                app_state.world.objective_state = ObjectiveState.ALIGN
                print(f"\n[EVENT] Objective: ALIGN")
            elif ch == 'y': 
                app_state.world.objective_state = ObjectiveState.SCOOP
                print(f"\n[EVENT] Objective: SCOOP")
            elif ch == 'u':
                app_state.world.objective_state = ObjectiveState.LIFT
                print(f"\n[EVENT] Objective: LIFT")
            elif ch == 'i':
                app_state.world.objective_state = ObjectiveState.PLACE
                print(f"\n[EVENT] Objective: PLACE")
            
            # JOB CONTROL
            elif ch == 'g': # 'G' for GO / START
                app_state.job_start = True
                app_state.job_start_timer = time.time()
                app_state.world.reset_mission()
                evt = MotionEvent("JOB_START", 0, 0)
                app_state.world.update_from_motion(evt)
                print("\n[EVENT] JOB STARTED")
            elif ch == 'k':
                app_state.job_success = True
                app_state.job_success_timer = time.time()
                evt = MotionEvent("JOB_SUCCESS", 0, 0)
                app_state.world.update_from_motion(evt)
                print("\n[EVENT] JOB SUCCESS")
            elif ch == 'j':
                app_state.job_abort = True
                app_state.job_abort_timer = time.time()
                app_state.world.reset_mission()
                evt = MotionEvent("JOB_ABORT", 0, 0)
                app_state.world.update_from_motion(evt)
                print("\n[EVENT] JOB ABORTED")
            
            # GEARS (1-9)
            elif ch in '123456789':
                app_state.current_gear = int(ch)
                print(f"\n[GEAR] Switched to Gear {app_state.current_gear}")

def control_loop(app_state):
    app_state.robot = Robot()
    app_state.vision = BrickDetector(debug=True, speed_optimize=False) # Changed to False for better visualization
    
    dt = 1.0 / LOG_RATE_HZ
    was_moving = False
    
    while app_state.running:
        loop_start = time.time()
        
        # 1. Heartbeat Check
        with app_state.lock:
            if time.time() - app_state.last_key_time > HEARTBEAT_TIMEOUT:
                app_state.active_command = None
                app_state.active_speed = 0.0
            
            # Dynamic Speed Calculation based on Gear
            # Gear 1 = 0.32, Gear 9 = 1.0
            gear_ratio = (app_state.current_gear - 1) / 8.0
            gear_speed = 0.32 + gear_ratio * (1.0 - 0.32)
            
            cmd = app_state.active_command
            if cmd:
                # Assign speed based on gear
                speed = gear_speed
                if cmd in ('u', 'd'):
                    speed = min(1.0, speed * 4.0)
                app_state.active_speed = speed # Update state for telemetry
            else:
                speed = 0.0
                app_state.active_speed = 0.0
            
        if cmd and speed > 0:
            app_state.robot.send_command(cmd, speed)
            was_moving = True
        elif was_moving:
            app_state.robot.stop()
            was_moving = False
            
        # 2. Vision
        found, angle, dist, offset_x, max_y, conf = app_state.vision.read()
        
        # 3. Telemetry Update
        app_state.world.update_vision(found, dist, angle, conf, offset_x, max_y)
        
        # Track Motion
        if cmd and speed > 0:
            atype = "unknown"
            if cmd == 'f': atype = "forward"
            elif cmd == 'b': atype = "backward"
            elif cmd == 'l': atype = "left_turn"
            elif cmd == 'r': atype = "right_turn"
            elif cmd == 'u': atype = "mast_up"
            elif cmd == 'd': atype = "mast_down"
            
            pwr = int(speed * 255)
            evt = MotionEvent(atype, pwr, int(dt*1000))
            app_state.world.update_from_motion(evt)
            
        # 4. Save Raw Frame & Log
        if app_state.vision.raw_frame is not None:
            ts_ns = time.time_ns()
            img_name = f"frame_{ts_ns}.jpg"
            img_path = os.path.join(app_state.session_dir, img_name)
            cv2.imwrite(img_path, app_state.vision.raw_frame)
            app_state.world.last_image_file = img_name
        
        # Log state with lock to ensure objective_state is read atomically
        with app_state.lock:
            app_state.logger.log_state(app_state.world)
        
        # 5. Draw HUD for Web Stream
        # Use standardized telemetry overlay
        if app_state.vision.current_frame is not None:
            view_frame = app_state.vision.current_frame.copy()
            
            messages = []
            if app_state.job_start and time.time() - app_state.job_start_timer < 3.0:
                messages.append("JOB STARTED")
            if app_state.job_success and time.time() - app_state.job_success_timer < 5.0:
                messages.append("JOB COMPLETE!")
            if app_state.job_abort and time.time() - app_state.job_abort_timer < 3.0:
                messages.append("JOB ABORTED")
                
            reminders = [
                f"W: Drive | P/L: Mast | 1-9: Gears ({app_state.current_gear})",
                "R: New Job / Find"
            ]
                
            draw_telemetry_overlay(view_frame, app_state.world, messages, reminders, gear=app_state.current_gear)
            
            with app_state.lock:
                app_state.current_frame = view_frame
        
        # 6. Rate Limiting
        elapsed = time.time() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)


if __name__ == "__main__":
    state = AppState()
    _stream_state = state
    
    # Web thread
    t_web = threading.Thread(target=lambda: flask_app.run(host="0.0.0.0", port=WEB_PORT, debug=False, use_reloader=False), daemon=True)
    t_web.start()
    
    # Keyboard thread
    kb_t = threading.Thread(target=keyboard_thread, args=(state,), daemon=True)
    kb_t.start()
    
    try:
        control_loop(state)
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        if state.robot: state.robot.close()
        if state.vision: state.vision.close()
        state.logger.close()
        print("\nShutdown complete.")
