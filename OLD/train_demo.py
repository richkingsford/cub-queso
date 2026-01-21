"""
# train_demo.py
--------------
Runs on the Robot.
Acts as a TCP Server for the Windows Xbox Client.
integrates Vision, Telemetry, and Motor Control.
Records events and handles Job Success confirmation.
"""
import socket
import threading
import time
import sys
import enum
import cv2
from pathlib import Path
from flask import Flask, Response

from helper_robot_control import Robot
from helper_brick_vision import BrickDetector
from robot_leia_telemetry import WorldModel, TelemetryLogger, MotionEvent, ObjectiveState, draw_telemetry_overlay

# --- CONFIG ---
HOST_IP = '0.0.0.0'
TCP_PORT = 65432
WEB_PORT = 5000
LOG_RATE_HZ = 10
CMD_TIMEOUT = 0.2 # If no command for 0.2s, stop motors
DEMOS_DIR = Path(__file__).resolve().parent / "demos"

def format_stream_url(port):
    local_url = f"http://localhost:{port}"
    ip = None
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        ip = None
    if ip and not ip.startswith("127."):
        return f"http://{ip}:{port} (local {local_url})"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
        if ip and not ip.startswith("127."):
            return f"http://{ip}:{port} (local {local_url})"
    except OSError:
        pass
    return local_url

# --- FLASK ---
flask_app = Flask(__name__)

def generate_frames():
    while True:
        if app_state is None:
            time.sleep(0.1)
            continue
            
        with app_state.lock:
            if app_state.current_frame is None:
                frame_to_send = None
            else:
                frame_to_send = app_state.current_frame.copy()
        
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
    return "<html><body style='background:#111; color:#eee; font-family:sans-serif; text-align:center;'><h1>Robot Eyes (Xbox Mode)</h1><img src='/video_feed' style='border:2px solid #555; border-radius:10px;'></body></html>"

@flask_app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# --- SHARED APP STATE ---
class AppState:
    def __init__(self):
        self.running = True
        self.active_command = None # 'f', 'b', 'l', 'r', 'u', 'd', or None
        self.active_speed = 0.0
        self.last_cmd_time = 0
        
        # Job Status
        self.job_success = False
        self.job_success_timer = 0
        self.job_start = False
        self.job_start_timer = 0
        self.job_abort = False
        self.job_abort_timer = 0
        
        self.lock = threading.Lock()
        
        # Session Setup
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = DEMOS_DIR / timestamp
        self.session_dir.mkdir(parents=True, exist_ok=True)
            
        print(f"[SESSION] Recording to: {self.session_dir}")
        
        # Telemetry
        self.world = WorldModel()
        log_path = self.session_dir / "a_log.json"
        self.logger = TelemetryLogger(log_path)
        self.vision = None
        self.robot = None
        
        # Video
        self.current_frame = None

app_state = AppState()

# ... (rest of file)

def handle_command(cmd_str):
    """Parses strings from Windows Client: 'f 100 50', 'BTN_A', etc."""
    parts = cmd_str.split()
    if not parts: return
    
    cmd = parts[0]
    app_state.last_cmd_time = time.time()
    
    # BUTTONS
    if cmd == "BTN_Y":
        # Job Start
        app_state.job_start = True
        app_state.job_start_timer = time.time()
        
        # Reset Objective
        app_state.world.reset_mission()
        
        # Log Event
        evt = MotionEvent("JOB_START", 0, 0)
        app_state.world.update_from_motion(evt)
        print("[DEMO] JOB STARTED!")
        return

    if cmd == "BTN_A":
        # Toggle Job Success
        app_state.job_success = True
        app_state.job_success_timer = time.time()
        # Log Special Event
        evt = MotionEvent("JOB_SUCCESS", 0, 0)
        app_state.world.update_from_motion(evt) # Stores last event
        print("[DEMO] JOB SUCCESS CONFIRMED!")
        return

    if cmd == "BTN_B":
        # Job Abort
        app_state.job_abort = True
        app_state.job_abort_timer = time.time()
        
        # Reset Objective
        app_state.world.reset_mission()
        
        # Log Event
        evt = MotionEvent("JOB_ABORT", 0, 0)
        app_state.world.update_from_motion(evt)
        print("[DEMO] JOB ABORTED!")
        return
        
    if cmd == "BTN_X":
        # Cycle Objective
        new_state = app_state.world.next_objective()
        print(f"[DEMO] Objective Set: {new_state}")
        return

    # MOVEMENT
    # Format: <char> <speed> <dur>
    # Note: Client sends "f 200 50", we just need 'f' and speed ratio.
    if len(parts) < 2: return
    
    app_state.active_command = cmd
    
    try:
        val = int(parts[1])
        app_state.active_speed = val / 255.0
    except: 
        app_state.active_speed = 0.0

# --- MAIN CONTROL LOOP ---
def control_loop():
    print("[SYSTEM] Starting Control Loop...")
    
    # Init Hardware
    app_state.robot = Robot()
    app_state.vision = BrickDetector(debug=True, save_folder=None)
    
    dt = 1.0 / LOG_RATE_HZ
    was_moving = False
    
    while app_state.running:
        loop_start = time.time()
        
        # 1. Safety Timeout
        if time.time() - app_state.last_cmd_time > CMD_TIMEOUT:
            app_state.active_command = None
            app_state.active_speed = 0.0
            
        # 2. Apply Inputs
        if app_state.active_command and app_state.active_speed > 0:
            app_state.robot.send_command(app_state.active_command, app_state.active_speed)
            was_moving = True
        elif was_moving:
            # Only send stop ONCE when we transition to zero speed
            app_state.robot.stop()
            was_moving = False
        else:
            # Already stopped, do not spam 'f 0'
            pass
        
        # 3. Vision
        found, angle, dist, offset_x, conf, cam_h = app_state.vision.read()
        view_frame = app_state.vision.current_frame
        
        # 4. Telemetry Update
        app_state.world.update_vision(found, dist, angle, conf, offset_x, cam_h)
        
        # Track Motion
        if app_state.active_command and app_state.active_speed > 0.05:
            # Map char to name
            c = app_state.active_command
            atype = "unknown"
            if c == 'f': atype = "forward"
            elif c == 'b': atype = "backward"
            elif c == 'l': atype = "left_turn"
            elif c == 'r': atype = "right_turn"
            elif c == 'u': atype = "mast_up"
            elif c == 'd': atype = "mast_down"
            
            pwr = int(app_state.active_speed * 255)
            evt = MotionEvent(atype, pwr, int(dt*1000))
            app_state.world.update_from_motion(evt)
            app_state.logger.log_event(evt, app_state.world.objective_state.value)

        # 5. Log
        app_state.logger.log_state(app_state.world)
        
        # 6. Draw HUD
        messages = []
        if app_state.job_start and time.time() - app_state.job_start_timer < 3.0:
            messages.append("JOB STARTED")
        if app_state.job_success and time.time() - app_state.job_success_timer < 5.0:
            messages.append("JOB COMPLETE!")
        if app_state.job_abort and time.time() - app_state.job_abort_timer < 3.0:
            messages.append("JOB ABORTED")

        draw_telemetry_overlay(view_frame, app_state.world, messages)
        with app_state.lock:
            app_state.current_frame = view_frame.copy()
            
        # Rate Limiting
        elapsed = time.time() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)


def main():
    # 1. TCP Server Thread
    t_tcp = threading.Thread(target=tcp_server_thread, daemon=True)
    t_tcp.start()
    
    # 2. Web Server Thread
    t_web = threading.Thread(target=lambda: flask_app.run(host="0.0.0.0", port=WEB_PORT, debug=False, use_reloader=False), daemon=True)
    t_web.start()
    print(f"[VISION] Stream started at {format_stream_url(WEB_PORT)}")
    
    # 3. Main Control Loop (Blocking)
    try:
        control_loop()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        app_state.running = False
        if app_state.robot: app_state.robot.close()
        if app_state.vision: app_state.vision.close()
        app_state.logger.close()
        print("\nShutdown complete.")

if __name__ == "__main__":
    main()
