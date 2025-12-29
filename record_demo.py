"""
record_demo.py
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
import os
from flask import Flask, Response

from robot_control import Robot
from brick_vision import BrickDetector
from leia_telemetry import WorldModel, TelemetryLogger, MotionEvent, ObjectiveState, draw_telemetry_overlay

# --- CONFIG ---
HOST_IP = '0.0.0.0'
TCP_PORT = 65432
WEB_PORT = 5000
LOG_RATE_HZ = 10
CMD_TIMEOUT = 0.2 # If no command for 0.2s, stop motors

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
        self.session_dir = os.path.join(os.getcwd(), "demos", timestamp)
        if not os.path.exists(self.session_dir):
            os.makedirs(self.session_dir)
            
        print(f"[SESSION] Recording to: {self.session_dir}")
        
        # Telemetry
        self.world = WorldModel()
        log_path = os.path.join(self.session_dir, "log.json")
        self.logger = TelemetryLogger(log_path)
        self.vision = None
        self.robot = None
        
        # Video
        self.current_frame = None

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
        app_state.world.objective_state = ObjectiveState.FIND
        
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
        app_state.world.objective_state = ObjectiveState.FIND
        
        # Log Event
        evt = MotionEvent("JOB_ABORT", 0, 0)
        app_state.world.update_from_motion(evt)
        print("[DEMO] JOB ABORTED!")
        return
        
    if cmd == "BTN_X":

        # Cycle Objective
        # Simple cycle logic
        states = list(ObjectiveState)
        curr_idx = states.index(app_state.world.objective_state)
        next_idx = (curr_idx + 1) % len(states)
        app_state.world.objective_state = states[next_idx]
        print(f"[DEMO] Objective Set: {app_state.world.objective_state.name}")
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
        found, angle, dist, offset_x = app_state.vision.read()
        view_frame = app_state.vision.current_frame
        
        # 4. Telemetry Update
        conf = 100 if found else 0
        app_state.world.update_vision(found, dist, angle, conf)
        
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

        # 5. Log
        app_state.logger.log_state(app_state.world)
        
        # 6. Draw HUD
        draw_hud(view_frame, app_state.world)
        with app_state.lock:
            app_state.current_frame = view_frame.copy()
            
        # Rate Limiting
        elapsed = time.time() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

def draw_hud(frame, wm):
    h, w = frame.shape[:2]
    
    # Reuse simple overlay logic
    overlay = frame.copy()
    cv2.rectangle(overlay, (w-250, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    x_base = w - 240
    y_cur = 30
    
    cv2.putText(frame, f"OBJ: {wm.objective_state.value}", (x_base, y_cur), font, 0.5, (255, 200, 0), 1)
    y_cur += 25
    
    if wm.brick['visible']:
        cv2.putText(frame, f"DIST: {wm.brick['dist']:.0f}", (x_base, y_cur), font, 0.5, (0, 255, 0), 1)
    else:
        cv2.putText(frame, "SEARCHING", (x_base, y_cur), font, 0.5, (0, 0, 255), 1)
    y_cur += 25
    
    if wm.last_event:
        cv2.putText(frame, f"ACT: {wm.last_event.action_type}", (x_base, y_cur), font, 0.5, (255, 100, 255), 1)
        
    # START BANNER
    if app_state.job_start:
        if time.time() - app_state.job_start_timer < 3.0:
            cv2.putText(frame, "JOB STARTED", (w//2 - 150, h//2 - 50), font, 1.5, (0, 255, 255), 3)
        else:
            app_state.job_start = False

    # SUCCESS BANNER
    if app_state.job_success:
        # Show for 5 seconds
        if time.time() - app_state.job_success_timer < 5.0:
            cv2.putText(frame, "JOB COMPLETE!", (w//2 - 150, h//2), font, 1.5, (0, 255, 0), 3)
        else:
            app_state.job_success = False

def main():
    # 1. TCP Server Thread
    t_tcp = threading.Thread(target=tcp_server_thread, daemon=True)
    t_tcp.start()
    
    # 2. Web Server Thread
    t_web = threading.Thread(target=lambda: flask_app.run(host="0.0.0.0", port=WEB_PORT, debug=False, use_reloader=False), daemon=True)
    t_web.start()
    
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

if __name__ == "__main__":
    main()
