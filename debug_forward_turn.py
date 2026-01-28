#!/usr/bin/env python3
"""
debug_forward_turn.py
---------------------
Experiments with combined forward/backward and turning movements.
This tests a new movement pattern where the robot moves forward/backward
while simultaneously turning left/right.
"""
import time
import threading
from helper_robot_control import Robot


class CombinedMovement:
    """Handles combined forward/backward + turn movements."""
    
    def __init__(self, robot):
        self.robot = robot
        self.running = False
        self.thread = None
    
    def _combined_movement_thread(self, drive_cmd, turn_cmd, drive_speed, turn_speed, duration_s):
        """Thread that sends alternating drive and turn commands."""
        start_time = time.time()
        while self.running and (time.time() - start_time) < duration_s:
            # Send drive command
            self.robot.send_command(drive_cmd, drive_speed)
            time.sleep(0.05)  # Small delay between commands
            
            # Send turn command
            self.robot.send_command(turn_cmd, turn_speed)
            time.sleep(0.05)  # Small delay before next cycle
    
    def move_and_turn(self, drive_cmd, turn_cmd, drive_speed=0.5, turn_speed=0.3, duration_s=1.0):
        """
        Execute combined movement (forward/backward + turn).
        
        Args:
            drive_cmd: 'f' for forward, 'b' for backward
            turn_cmd: 'l' for left, 'r' for right
            drive_speed: Speed for forward/backward (0.0 to 1.0)
            turn_speed: Speed for turning (0.0 to 1.0)
            duration_s: Duration of the movement in seconds
        """
        print(f"[MOVEMENT] {drive_cmd.upper()} + {turn_cmd.upper()} for {duration_s}s (drive={drive_speed:.2f}, turn={turn_speed:.2f})")
        
        self.running = True
        self.thread = threading.Thread(
            target=self._combined_movement_thread,
            args=(drive_cmd, turn_cmd, drive_speed, turn_speed, duration_s)
        )
        self.thread.start()
        self.thread.join()  # Wait for movement to complete
        self.running = False
        
        # Stop the robot after movement
        self.robot.stop()
        print(f"[MOVEMENT] Completed {drive_cmd.upper()} + {turn_cmd.upper()}")


def main():
    """Main experiment sequence."""
    print("[DEBUG] Starting combined forward/turn experiments...")
    
    # Initialize robot
    robot = Robot()
    mover = CombinedMovement(robot)
    
    try:
        # 1. Forward + Left turn for 1 second
        print("\n=== Test 1: Forward + Turn Left ===")
        mover.move_and_turn('f', 'l', drive_speed=0.5, turn_speed=0.3, duration_s=1.0)
        time.sleep(0.5)  # Brief pause between movements
        
        # 2. Forward + Right turn for 1 second
        print("\n=== Test 2: Forward + Turn Right ===")
        mover.move_and_turn('f', 'r', drive_speed=0.5, turn_speed=0.3, duration_s=1.0)
        time.sleep(0.5)
        
        # === FAST TESTS (Pivot: Drive=Turn=0.3) ===
        print("\n=== Test 1: Forward + Turn Left (Pivot) ===")
        mover.move_and_turn('f', 'l', drive_speed=0.3, turn_speed=0.3)
        time.sleep(1)

        print("\n=== Test 2: Forward + Turn Right (Pivot) ===")
        mover.move_and_turn('f', 'r', drive_speed=0.3, turn_speed=0.3)
        time.sleep(1)

        print("\n=== Test 3: Backward + Turn Left (Pivot) ===")
        mover.move_and_turn('b', 'l', drive_speed=0.3, turn_speed=0.3)
        time.sleep(1)

        print("\n=== Test 4: Backward + Turn Right (Pivot) ===")
        mover.move_and_turn('b', 'r', drive_speed=0.3, turn_speed=0.3)
        time.sleep(1)

        # === SLOW TESTS (Pivot: Drive=Turn=0.22) ===
        print("\n=== Test 5: Forward + Turn Left (Slow Pivot) ===")
        mover.move_and_turn('f', 'l', drive_speed=0.22, turn_speed=0.22)
        time.sleep(1)

        print("\n=== Test 6: Forward + Turn Right (Slow Pivot) ===")
        mover.move_and_turn('f', 'r', drive_speed=0.22, turn_speed=0.22)
        time.sleep(1)

        print("\n=== Test 7: Backward + Turn Left (Slow Pivot) ===")
        mover.move_and_turn('b', 'l', drive_speed=0.22, turn_speed=0.22)
        time.sleep(1)

        print("\n=== Test 8: Backward + Turn Right (Slow Pivot) ===")
        mover.move_and_turn('b', 'r', drive_speed=0.22, turn_speed=0.22)
        time.sleep(1)
        
        print("\n[DEBUG] All tests completed successfully!")
        
    except KeyboardInterrupt:
        print("\n[DEBUG] Interrupted by user")
    finally:
        # Ensure robot stops
        robot.stop()
        robot.close()
        print("[DEBUG] Robot stopped and connection closed")


if __name__ == "__main__":
    main()
