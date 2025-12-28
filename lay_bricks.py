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

def main():
    # 1. Initialize Hardware
    print("--- INITIALIZING ROBOT ---")
    robot = Robot()
    vision = BrickDetector(debug=True)

    print("--- STARTING APPROACH ---")
    print("Press Ctrl+C to abort.")

    try:
        while True:
            # We delegate the actual movement logic to our new file
            maneuvers.crawl_forward_if_aligned(robot, vision)
            
            # A tiny sleep keeps the loop running smoothly without
            # overwhelming the CPU (approx 60-100Hz update rate)
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n--- EMERGENCY STOP ---")
        robot.stop()
        vision.close()

if __name__ == "__main__":
    main()