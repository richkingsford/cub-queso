import argparse
import sys
import threading
import time
import tty
import termios
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, Response

from robot_control import Robot
from train_brick_vision import BrickDetector
from robot_leia_telemetry import WorldModel, TelemetryLogger, MotionEvent, ObjectiveState, draw_telemetry_overlay
 
# --- CONFIG ---
LOG_RATE_HZ = 10
GEAR_1_SPEED = 0.32  # 4x faster per user request
GEAR_9_SPEED = 1.0   # 100% capacity
HEARTBEAT_TIMEOUT = 0.3 # Stop if no key for 0.3s
DEMOS_DIR = Path(__file__).resolve().parent / "demos"
DEMO_OBJECTIVES = [ObjectiveState.FIND, ObjectiveState.SCOOP]

def objective_label(obj_enum):
    if obj_enum == ObjectiveState.SCOOP:
        return "PICK"
    return obj_enum.value

def log_line(message):
    sys.stdout.write(f"{str(message).strip()}\n")
    sys.stdout.flush()

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
        
        # Job Status
        
        self.lock = threading.Lock()
        self.current_frame = None
        self.internal_step = "START_ERROR" # START_ERROR, START_RECOVERY, START_SUCCESS, FINISH_OBJ
        
        # Session Setup
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.demos_dir = DEMOS_DIR
        self.demos_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.demos_dir / f"kbd_{timestamp}.json"
        log_line(f"[SESSION] Recording Keyboard Demo to: {log_path}")
        log_line("W/S: Forward/Backward | A/D: Turn Left/Right")
        log_line("P/L: Lift Up/Down")
        log_line("F: Next action (Fail -> Recover -> Success; FIND then PICK)")
        log_line("Q: Quit")
        log_line("[DEBUG] Scope restricted to FIND and PICK objectives.")
        log_line("NEXT: Press 'f' to begin FAIL for FIND")
        
        # Telemetry
        self.world = WorldModel()
        self.logger = TelemetryLogger(log_path)
        self.logger.log_keyframe("JOB_START")
        self.logger.enabled = False # Wait for OBJ_START
        
        # ID Init
        self.world.run_id = f"run_{timestamp}"
        self.world.attempt_id = 1
        self.objective_index = 0
        self.world.objective_state = DEMO_OBJECTIVES[self.objective_index]

        self.vision = None
        self.robot = None

# --- WEB STREAMING ---
web_app = Flask(__name__)
app_state_ref = None

def generate_frames(app_state):
    while app_state.running:
        with app_state.lock:
            if app_state.current_frame is None:
                # Placeholder if no frame yet
                frame_to_send = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame_to_send, "WAITING FOR CAMERA...", (120, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            else:
                frame_to_send = app_state.current_frame.copy()
        
        (flag, encodedImage) = cv2.imencode(".jpg", frame_to_send)
        if not flag:
            continue
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')

@web_app.route("/")
def index():
    return """
    <html>
        <head>
            <title>Robot Leia - Keyboard Training</title>
            <style>
                body { background: #1a1a1a; color: #eee; font-family: sans-serif; text-align: center; margin-top: 50px; }
                .stream-container { display: inline-block; border: 5px solid #333; border-radius: 8px; overflow: hidden; }
                h1 { color: #f0ad4e; }
            </style>
        </head>
        <body>
            <h1>Robot Leia - Keyboard Training</h1>
            <div class="stream-container">
                <img src="/video_feed" width="800">
            </div>
            <p>Use the terminal for controls. Keep this window open to see the live feed.</p>
        </body>
    </html>
    """

@web_app.route("/video_feed")
def video_feed():
    return Response(generate_frames(app_state_ref), 
                    mimetype="multipart/x-mixed-replace; boundary=frame")

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

        messages = []
        with app_state.lock:
            app_state.last_key_time = time.time()
            if ch == 'q':
                app_state.running = False
                messages.append("Stopping manual recording...")
            else:
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
                
                # JOB CONTROL (Single Taps - Refactored to 'f')
                elif ch == 'f':
                    obj_enum = app_state.world.objective_state
                    obj = obj_enum.value
                    obj_label = objective_label(obj_enum)
                    
                    if app_state.internal_step == "START_ERROR":
                        # --- Start Objective and Mark FAIL Start ---
                        app_state.world.recording_active = True
                        app_state.logger.log_keyframe("OBJ_START", obj)
                        app_state.logger.log_keyframe("FAIL_START", obj)
                        
                        app_state.world.attempt_status = "FAIL"
                        evt = MotionEvent("FAIL", 0, 0)
                        app_state.world.update_from_motion(evt)
                        messages.append(f"[{obj_label}] -> FAIL_START (Mistake Started)")
                        messages.append("NEXT: Press 'f' to begin RECOVERY")
                        app_state.internal_step = "START_RECOVERY"
                        
                    elif app_state.internal_step == "START_RECOVERY":
                        # --- End FAIL, Start RECOVER ---
                        app_state.logger.log_keyframe("FAIL_END", obj)
                        app_state.logger.log_keyframe("RECOVER_START", obj)
                        
                        app_state.world.attempt_status = "RECOVERY"
                        evt = MotionEvent("RECOVERY_START", 0, 0)
                        app_state.world.update_from_motion(evt)
                        messages.append(f"[{obj_label}] -> RECOVER_START (Recovery Started)")
                        messages.append("NEXT: Press 'f' to begin SUCCESS")
                        app_state.internal_step = "START_SUCCESS"
                        
                    elif app_state.internal_step == "START_SUCCESS":
                        # --- End RECOVER, Start SUCCESS Demo ---
                        app_state.logger.log_keyframe("RECOVER_END", obj)
                        app_state.logger.log_keyframe("SUCCESS_START", obj)
                        
                        app_state.world.attempt_status = "NORMAL"
                        messages.append(f"[{obj_label}] -> SUCCESS_START (Clean Run Started)")
                        messages.append("NEXT: Press 'f' when SUCCESS demo is complete")
                        app_state.internal_step = "FINISH_OBJ"
                    
                    elif app_state.internal_step == "FINISH_OBJ":
                        # --- End SUCCESS, Wrap Objective ---
                        app_state.logger.log_keyframe("SUCCESS_END", obj)
                        app_state.logger.log_keyframe("OBJ_SUCCESS", obj)
                        
                        # Pause high-freq state logging until next objective starts
                        app_state.logger.enabled = False
                        app_state.world.recording_active = False
                        
                        app_state.world.attempt_status = "NORMAL"
                        
                        if app_state.objective_index < len(DEMO_OBJECTIVES) - 1:
                            # Move to Next Objective in Demo Scope
                            evt = MotionEvent("OBJECTIVE_SUCCESS", 0, 0)
                            app_state.world.update_from_motion(evt)
                            app_state.objective_index += 1
                            next_obj = DEMO_OBJECTIVES[app_state.objective_index]
                            app_state.world.reset_mission()
                            app_state.world.objective_state = next_obj
                            messages.append(f"[{obj_label}] complete. Reset robot for {objective_label(next_obj)}.")
                            messages.append(f"NEXT: Press 'f' to begin FAIL for {objective_label(next_obj)}")
                        else:
                            # Job Complete after last objective
                            app_state.job_success = True
                            app_state.job_success_timer = time.time()
                            app_state.logger.log_keyframe("JOB_SUCCESS")
                            
                            evt = MotionEvent("JOB_SUCCESS", 0, 0)
                            app_state.world.update_from_motion(evt)
                            messages.append("[JOB] SUCCESS - Reset robot for next run.")
                            
                            # Reset for Next Job
                            app_state.world.reset_mission()
                            app_state.world.attempt_id += 1 
                            app_state.logger.log_keyframe("JOB_START")
                            app_state.logger.enabled = False # Wait for OBJ_START
                            
                            app_state.job_start = True
                            app_state.job_start_timer = time.time()
                            evt_start = MotionEvent("JOB_START", 0, 0)
                            app_state.world.update_from_motion(evt_start)
                            app_state.objective_index = 0
                            app_state.world.objective_state = DEMO_OBJECTIVES[app_state.objective_index]
                            messages.append("NEXT: Press 'f' to begin FAIL for FIND")
                        
                        app_state.internal_step = "START_ERROR"

        for msg in messages:
            log_line(msg)

        if not app_state.running:
            break

def control_loop(app_state):
    app_state.robot = Robot()
    # speed_optimize=False so we get the debug markers drawn on the frame
    app_state.vision = BrickDetector(debug=True, speed_optimize=False)
    
    dt = 1.0 / LOG_RATE_HZ
    was_moving = False
    
    while app_state.running:
        loop_start = time.time()
        
        # 1. Heartbeat Check
        with app_state.lock:
            if time.time() - app_state.last_key_time > HEARTBEAT_TIMEOUT:
                app_state.active_command = None
                app_state.active_speed = 0.0
            
            # Fixed speed based on Gear 1
            gear_speed = GEAR_1_SPEED
            
            cmd = app_state.active_command
            if cmd:
                speed = gear_speed
                if cmd in ('u', 'd'):
                    speed = min(1.0, speed * 4.0)
                app_state.active_speed = speed 
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
        found, angle, dist, offset_x, conf, cam_h = app_state.vision.read()
        
        # 3. Telemetry Update
        app_state.world.update_vision(found, dist, angle, conf, offset_x, cam_h)
        
        # 4. Update Web Stream Frame
        if app_state.vision.current_frame is not None:
            # Create a copy and draw our rich HUD
            frame = app_state.vision.current_frame.copy()
            
            with app_state.lock:
                draw_telemetry_overlay(frame, app_state.world, show_prompt=False)
                app_state.current_frame = frame
        
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
            app_state.logger.log_event(evt, app_state.world.objective_state.value)
            
        # 5. Save Log (Image saving removed)
        with app_state.lock:
            app_state.logger.log_state(app_state.world)
        
        # 6. Rate Limiting
        elapsed = time.time() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true", help="Enable livestreaming")
    args = parser.parse_args()

    state = AppState()
    
    # Keyboard thread
    kb_t = threading.Thread(target=keyboard_thread, args=(state,), daemon=True)
    kb_t.start()
    
    # Web Stream thread (optional)
    if args.stream:
        app_state_ref = state
        stream_t = threading.Thread(
            target=lambda: web_app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False),
            daemon=True
        )
        stream_t.start()
        log_line("[VISION] Stream started at http://localhost:5000")
    else:
        log_line("[VISION] Stream disabled (use --stream to enable)")
    
    try:
        control_loop(state)
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        if state.robot: state.robot.close()
        if state.vision: state.vision.close()
        state.logger.close()
        log_line("Shutdown complete.")
