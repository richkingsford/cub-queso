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
from robot_control import Robot

# --- CONFIG ---
GEAR_1_SPEED = 0.32 
GEAR_9_SPEED = 1.0
HEARTBEAT_TIMEOUT = 0.3 

class TestState:
    def __init__(self):
        self.running = True
        self.active_command = None
        self.active_speed = 0.0
        self.current_gear = 1
        self.last_key_time = 0
        self.lock = threading.Lock()

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
            
            # GEARS
            elif ch in '123456789':
                state.current_gear = int(ch)
                print(f"\n[GEAR] Gear {state.current_gear}")

def main():
    state = TestState()
    robot = Robot()
    
    # Start keyboard thread
    kb_t = threading.Thread(target=keyboard_thread, args=(state,), daemon=True)
    kb_t.start()
    
    was_moving = False
    
    try:
        while state.running:
            with state.lock:
                # Heartbeat check
                if time.time() - state.last_key_time > HEARTBEAT_TIMEOUT:
                    state.active_command = None
                    state.active_speed = 0.0
                
                # Gear Speed
                gear_ratio = (state.current_gear - 1) / 8.0
                gear_speed = GEAR_1_SPEED + gear_ratio * (GEAR_9_SPEED - GEAR_1_SPEED)
                
                cmd = state.active_command
                if cmd:
                    speed = gear_speed
                else:
                    speed = 0.0
            
            if cmd and speed > 0:
                robot.send_command(cmd, speed)
                was_moving = True
            elif was_moving:
                robot.stop()
                was_moving = False
            
            time.sleep(0.05) # 20Hz update
            
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        robot.stop()
        robot.close()
        print("\nWiggle Test Stopped.")

if __name__ == "__main__":
    main()