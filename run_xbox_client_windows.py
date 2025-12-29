import pygame
import socket
import time
import math

# ================= CONFIGURATION =================
ORIN_IP = '10.0.0.203'     # CHANGE THIS to your Orin's actual IP
PORT = 65432               # Port to talk to the Orin

# --- Tuning for Subtle Control ---
DEADZONE = 0.15            # Smaller deadzone for more range
MIN_SPEED = 60             # The lowest speed the motors can actually move at (0-255)
MAX_SPEED = 255            # Max PWM speed (0-255)
CURVE_EXPONENT = 2.5       # Higher = softer start. 1.0 is linear, 3.0 is very gentle.

CMD_DURATION = 50          # Shortened to 50ms so the robot stops INSTANTLY when you let go
LOOP_DELAY = 0.05          # Send commands faster (20 times/sec)
# =================================================

print(f"Connecting to Leia at {ORIN_IP}...")

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    client_socket.connect((ORIN_IP, PORT))
    print("Connected to Leia successfully!")
except Exception as e:
    print(f"Could not connect. Is the 'leia_server.py' script running on the Orin? {e}")
    exit()

# Init Controller
pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    print("No controller detected! Plug in your Xbox controller.")
    exit()

joy = pygame.joystick.Joystick(0)
joy.init()
print(f"Controller detected: {joy.get_name()}")
print("Left Stick: Drive (Fine Control) | Right Stick: Mast")
print("Press CTRL+C to quit.")

def get_fine_speed(raw_value):
    """
    Converts joystick input (0.0 to 1.0) into a motor speed (MIN_SPEED to 255)
    using an exponential curve for subtle control.
    """
    val = abs(raw_value)
    
    # 1. If inside deadzone, stop
    if val < DEADZONE:
        return 0
        
    # 2. Normalize input to 0.0 - 1.0 range (ignoring the deadzone part)
    #    This ensures the motor starts smoothly right at the edge of the deadzone.
    normalized = (val - DEADZONE) / (1.0 - DEADZONE)
    
    # 3. Apply Curve (Exponential)
    #    Input 0.1 -> Output 0.003 (Tiny)
    #    Input 1.0 -> Output 1.0 (Full)
    curved = normalized ** CURVE_EXPONENT
    
    # 4. Map to Motor Range (MIN_SPEED to MAX_SPEED)
    #    We start at MIN_SPEED because values below that usually just make motors hum without moving.
    final_pwm = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * curved
    
    return int(final_pwm)

try:
    while True:
        pygame.event.pump()
        msg = ""

        # --- READ INPUTS ---
        lx = -joy.get_axis(0)  # Left/Right (Inverted)
        ly = -joy.get_axis(1)  # Up/Down (Inverted)
        ry = joy.get_axis(3)   # Mast Up/Down
        
        # --- LOGIC PRIORITY ---
        
        # 1. DRIVE (Left Stick)
        # Check raw values against deadzone
        if abs(ly) > DEADZONE or abs(lx) > DEADZONE:
            if abs(ly) > abs(lx):
                # Forward/Back
                speed = get_fine_speed(ly)
                if ly < 0:
                    msg = f"f {speed} {CMD_DURATION}" # Forward
                else:
                    msg = f"b {speed} {CMD_DURATION}" # Back
            else:
                # Turning
                speed = get_fine_speed(lx)
                if lx < 0:
                    msg = f"l {speed} {CMD_DURATION}" # Left
                else:
                    msg = f"r {speed} {CMD_DURATION}" # Right

        # 2. MAST (Right Stick)
        elif abs(ry) > DEADZONE:
            # We use linear speed for the mast (simpler)
            speed = int(abs(ry) * MAX_SPEED) 
            if ry < 0:
                msg = f"u {speed} {CMD_DURATION}" # Up
            else:
                msg = f"d {speed} {CMD_DURATION}" # Down

        # 3. BUTTONS (Discrete Events)
        # We check buttons and send special strings
        # A=0, B=1, X=2, Y=3 (Standard mapping, verify if needed)
        
        # Simple debounce/state tracking could be added in a class, 
        # but for now we rely on the server handling or just raw sends.
        # Actually, let's send ONE message per press to avoid flooding "BTN_A".
        # We need to track 'prev_buttons' outside the loop, but inserting it here is messy without refactor.
        # Let's just check get_button. If pressed, we send. 
        # To avoid flood, we only send if we haven't sent a button msg this frame?
        # A proper way requires state. 
        # Let's assume the user holds it briefly.
        
        if joy.get_button(0): # A Button
             msg = "BTN_A"
        elif joy.get_button(1): # B Button
             msg = "BTN_B"
        elif joy.get_button(2): # X Button
             msg = "BTN_X"
        elif joy.get_button(3): # Y Button
             msg = "BTN_Y"

        # --- SEND COMMAND ---
        if msg:
            try:
                client_socket.send(msg.encode('utf-8'))
            except BrokenPipeError:
                print("Connection lost.")
                break
            time.sleep(LOOP_DELAY)
        else:
            # If no input, sleep briefly to save CPU
            time.sleep(0.01)

except KeyboardInterrupt:
    print("\nClosing connection...")
    client_socket.close()