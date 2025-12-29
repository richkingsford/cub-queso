"""
manual_test.py
--------------
A raw diagnostic tool to test Motor Power and Serial Protocol.
Bypasses all vision and autonomy logic.

Controls:
  'w' = Forward (High Power - 200 PWM)
  's' = Stop    (0 PWM)
  'r' = Reverse (High Power - 200 PWM)
  
  OR type a custom string (e.g., "100,100") to send raw data.
"""
import serial
import time
import sys

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/ttyCH341USB0' 
BAUD_RATE = 115200 

def main():
    print("--- MOTOR DIAGNOSTIC TOOL ---")
    print(f"Connecting to {SERIAL_PORT}...")
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # Wait for Arduino Reset
        print("CONNECTED.")
    except Exception as e:
        print(f"ERROR: {e}")
        return

    print("\nCOMMANDS:")
    print("  [w] Forward (PWM 200)")
    print("  [s] STOP")
    print("  [r] Reverse (PWM -200)")
    print("  ...or type a raw string like '100,100' or '<150,150>' to test protocol.")
    print("  [q] Quit")
    print("-" * 30)

    try:
        while True:
            user_input = input("CMD > ").strip()
            
            if user_input.lower() == 'q':
                break
            
            command_to_send = ""

            # --- PRESETS ---
            if user_input.lower() == 'w':
                command_to_send = "200,200\n"
                print(f"  -> Sending HIGH POWER Forward: {repr(command_to_send)}")
            
            elif user_input.lower() == 's':
                command_to_send = "0,0\n"
                print(f"  -> Sending STOP: {repr(command_to_send)}")
            
            elif user_input.lower() == 'r':
                command_to_send = "-200,-200\n"
                print(f"  -> Sending REVERSE: {repr(command_to_send)}")
                
            # --- RAW INPUT (For Debugging Protocol) ---
            else:
                # If you type "100,100", we append newline just in case
                command_to_send = user_input + "\n"
                print(f"  -> Sending RAW: {repr(command_to_send)}")

            # SEND TO ARDUINO
            ser.write(command_to_send.encode('utf-8'))
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        ser.write("0,0\n".encode('utf-8'))
        ser.close()

if __name__ == "__main__":
    main()