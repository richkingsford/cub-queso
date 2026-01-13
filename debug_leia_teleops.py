"""
# debug_leia_teleops.py
---------------
The Brain of Leia for Teleoperation Demos.
Combines:
1. Robot Hardware Interface (Motors)
2. Vision System (Brick Detection)
3. State Machine (Teleop vs Auto)
4. Web Visualization (Flask)
"""
import time
import threading
import cv2
import sys
import enum
from flask import Flask, Response

# Import our capabilities
from robot_control import Robot
from train_brick_vision import BrickDetector
from telemetry_robot import WorldModel, TelemetryLogger, MotionEvent, ObjectiveState, draw_telemetry_overlay

# --- CONFIG ---
WEB_PORT = 5000
LOG_RATE_HZ = 10

# --- STATE MACHINE ---
class RobotState(enum.Enum):
    IDLE = 0
    TELEOP = 1
    AUTONOMOUS = 2

class TeleopManager:
    def __init__(self):
        # 1. Hardware
        print("[SYSTEM] Initializing Robot Hardware...")
        self.robot = Robot()
        
        # 2. Vision
        print("[SYSTEM] Initializing Vision...")
        self.vision = BrickDetector(debug=True, save_folder=None)
        
        # 3. Telemetry & World Model
        self.world = WorldModel()
        self.logger = TelemetryLogger()
        
        # 4. State
        self.state = RobotState.IDLE
        self.running = True
        
        # 5. Teleop Input
        self.input_left = 0.0
        self.input_right = 0.0
        self.input_lift = 0.0
        
        # 6. Shared Data for Web Stream
        self.current_frame = None
        self.lock = threading.Lock()

    def update_loop(self):
        """Main control loop running at ~10Hz (Log Rate)"""
        print("[SYSTEM] Starting Control Loop...")
        dt = 1.0 / LOG_RATE_HZ
        
        while self.running:
            loop_start = time.time()
            
            # A. Read Vision
            found, angle, dist, offset_x, conf, cam_h, brick_above, brick_below = self.vision.read()
            view_frame = self.vision.current_frame
            
            # Update World Model with Vision
            self.world.update_vision(found, dist, angle, conf, offset_x, cam_h, brick_above, brick_below)
            
            # B. Robot Control & Motion Tracking
            if self.state == RobotState.IDLE:
                self.robot.stop()
                
            elif self.state == RobotState.TELEOP:
                # Apply Inputs
                self.robot.set_left_motor(self.input_left)
                self.robot.set_right_motor(self.input_right)
                self.robot.set_lift_motor(self.input_lift)
                
                # Generate Motion Events for Dead Reckoning
                # We assume the command lasts for 'dt' seconds (or use CMD_DURATION from robot)
                # Ideally, we infer motion from the COMMAND we just sent.
                # Simplification: Convert continuous input to discrete "events" for the logger?
                # The user asked for "Motion Events: each action logged".
                # If we are holding a joystick forward, that's a continuous stream of events?
                # Or one long event?
                # User said: "duration_ms".
                # So we can log "Steps".
                duration_ms = int(dt * 1000)
                
                # Determine Action Type (Simplified logic)
                action = None
                pwr = 0
                
                if abs(self.input_left) > 0.1 and abs(self.input_right) > 0.1:
                    # Both moving
                    if self.input_left > 0 and self.input_right > 0: action = "forward"
                    elif self.input_left < 0 and self.input_right < 0: action = "backward"
                    elif self.input_left < 0 and self.input_right > 0: action = "left_turn" # Tank turn
                    elif self.input_left > 0 and self.input_right < 0: action = "right_turn"
                    pwr = int(max(abs(self.input_left), abs(self.input_right)) * 255)
                elif abs(self.input_lift) > 0.1:
                    action = "mast_up" if self.input_lift > 0 else "mast_down"
                    pwr = int(abs(self.input_lift) * 255)
                
                if action:
                    event = MotionEvent(action, pwr, duration_ms)
                    self.world.update_from_motion(event)
                    self.logger.log_event(event, self.world.objective_state.value)
                
            elif self.state == RobotState.AUTONOMOUS:
                pass

            # C. Log Telemetry
            self.logger.log_state(self.world)

            # D. Update Video Stream with Telemetry Overlay
            # Pass Mode as extra message/context if we want (or maybe just rely on logs)
            # The shared renderer doesn't print "MODE: TELEOP", but we can pass it as a message or update the renderer to take arbitrary lines?
            # The user wants "high confidence... similar output formats".
            # The shared renderer output is cleaner.
            # Let's pass the Mode as an extra message at the top just to be safe.
            draw_telemetry_overlay(view_frame, self.world, [f"MODE: {self.state.name}"])
            
            with self.lock:
                self.current_frame = view_frame.copy()

            # Sleep to maintain rate
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    def shutdown(self):
        self.running = False
        self.robot.close()
        self.vision.close()
        self.logger.close()

# --- WEB SERVER ---
app = Flask(__name__)
manager = None

def generate_frames():
    while True:
        if manager and manager.current_frame is not None:
            with manager.lock:
                frame = manager.current_frame
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            if not flag: continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                  bytearray(encodedImage) + b'\r\n')
        else:
            time.sleep(0.1)

@app.route("/")
def index():
    return "<html><body><h1>Leia Teleops</h1><img src='/video_feed'></body></html>"

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# --- INPUT SIMULATION (Simple CLI for Demo) ---
# In a real system, you'd use a gamepad library or web buttons.
# Here we just run the loop.

def main():
    global manager
    manager = TeleopManager()
    
    # Start Control Loop Thread
    t_control = threading.Thread(target=manager.update_loop, daemon=True)
    t_control.start()
    
    # Start Web Server Thread
    t_server = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=WEB_PORT, debug=False, use_reloader=False), daemon=True)
    t_server.start()
    
    print("\n--- LEIA TELEOPS READY ---")
    print(f"Stream: http://localhost:{WEB_PORT}")
    print("Press Ctrl+C to exit.")
    
    # Simulating some user input for demonstration
    # (In reality, this would be updated by joystick events)
    try:
        manager.state = RobotState.TELEOP
        while True:
            # Demo: Just sit there for now, or maybe wiggle motors
            # manager.input_left = 0.5 # Example
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        manager.shutdown()

if __name__ == "__main__":
    main()
