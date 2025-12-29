"""
setup_check_arduino.py
----------------------
Unified Arduino Setup and Testing Utility.
1. Scans for connected Serial/COM ports.
2. Allows manual servo testing via Keyboard input.
"""

import serial
import serial.tools.list_ports
import time
import sys

try:
    import keyboard
except ImportError:
    print("This script requires the 'keyboard' library.")
    print("pip install keyboard")
    # We continue, but test_servo will fail if used.
    keyboard = None

# --- CONSTANTS ---
BAUD_RATE = 115200
MIN_PULSE = 1000
MAX_PULSE = 2000
STOP_PULSE = 1500
STEP = 50

def list_ports():
    print("\n--- Scanning for Serial Ports ---")
    ports = serial.tools.list_ports.comports()
    
    if not ports:
        print("[WARN] No devices found!")
        return []
    
    found = []
    for port, desc, hwid in ports:
        print(f"FOUND: {port}")
        print(f"       Desc: {desc}")
        print(f"       ID:   {hwid}")
        print("-" * 30)
        found.append(port)
    return found

def test_servo(port):
    if not keyboard:
        print("[ERROR] 'keyboard' library not installed. Cannot run interactive test.")
        return

    print(f"\n--- Starting Servo Test on {port} ---")
    print(f"Baud: {BAUD_RATE}")
    print("Controls: LEFT/RIGHT Arrows to pulse. 'S' to stop. 'Q' to quit.")
    
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        time.sleep(2) # Wait for Arduino Reset
    except serial.SerialException as e:
        print(f"[FAIL] Could not open port: {e}")
        return

    pulse = STOP_PULSE
    ser.write(f"{pulse}\n".encode())
    
    try:
        while True:
            if keyboard.is_pressed("left"):
                pulse = max(MIN_PULSE, pulse - STEP)
            elif keyboard.is_pressed("right"):
                pulse = min(MAX_PULSE, pulse + STEP)
            elif keyboard.is_pressed("s"):
                pulse = STOP_PULSE
            elif keyboard.is_pressed("q"):
                print("Quitting...")
                break
            
            # Send command
            cmd = f"{pulse}\n"
            ser.write(cmd.encode())
            print(f"\rPulse: {pulse}   ", end="", flush=True)
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("\nStopping Servo...")
        ser.write(f"{STOP_PULSE}\n".encode())
        ser.close()

def main():
    ports = list_ports()
    
    if not ports:
        return

    print("\nDo you want to run the Servo Test? (y/n)")
    # Simple input check (blocking)
    choice = input("> ").strip().lower()
    if choice == 'y':
        if len(ports) == 1:
            target = ports[0]
        else:
            print("Enter port name (e.g. /dev/ttyUSB0 or COM3):")
            target = input("> ").strip()
        
        test_servo(target)

if __name__ == "__main__":
    main()
