#!/usr/bin/env python3
"""
reset.py - Simple script to turn the robot left 45 degrees
"""
import time
from robot_control import Robot

# Constants
TURN_ANGLE = 999  # degrees
DEG_PER_SEC_FULL_SPEED = 140.0  # from world model
SPEED = 1.0  # full speed

def main():
    print(f"[RESET] Turning robot right {TURN_ANGLE} degrees...")
    
    # Calculate duration needed for 45 degree turn
    duration = TURN_ANGLE / DEG_PER_SEC_FULL_SPEED
    
    # Initialize robot
    robot = Robot()
    
    try:
        # Turn right - smooth continuous motion without sleep
        start_time = time.time()
        while time.time() - start_time < duration:
            robot.send_command('r', SPEED)
        
        # Stop
        robot.stop()
        print(f"[RESET] Turn complete! ({duration:.2f}s)")
        
    except KeyboardInterrupt:
        robot.stop()
        print("\n[RESET] Interrupted by user")
    except Exception as e:
        robot.stop()
        print(f"[RESET] Error: {e}")

if __name__ == "__main__":
    main()
