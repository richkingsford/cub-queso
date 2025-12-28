"""
robot_control.py
----------------
The Translator.
Converts autonomous decisions into the specific command strings 
that your Arduino firmware expects (e.g., "f 200 50").
"""
import serial
import time
import sys

class Robot:
    def __init__(self):
        self.SERIAL_PORT = '/dev/ttyCH341USB0' 
        self.BAUD_RATE = 115200
        self.ser = None
        
        # --- PHYSICAL CONSTANTS (Copied from your Xbox Config) ---
        self.MIN_PWM = 60   # Motors won't turn below this
        self.MAX_PWM = 255
        self.CMD_DURATION = 100 # ms (Keep it running slightly longer for smooth auto-drive)
        
        self.connect()

    def connect(self):
        try:
            print(f"[ROBOT] Connecting to Arduino on {self.SERIAL_PORT}...")
            self.ser = serial.Serial(self.SERIAL_PORT, self.BAUD_RATE, timeout=1)
            time.sleep(2) 
            self.ser.reset_input_buffer()
            print("[ROBOT] Connected.")
        except Exception as e:
            print(f"[ROBOT] ERROR: {e}")
            sys.exit(1)

    def _send(self, command_str):
        """Internal helper to write the string to Serial"""
        if self.ser:
            try:
                # The Arduino expects bytes
                self.ser.write(command_str.encode('utf-8'))
            except Exception as e:
                print(f"[ROBOT] Write Error: {e}")

    def drive(self, speed):
        """
        Moves the robot Forward or Backward.
        speed: Float between -1.0 (Full Reverse) and 1.0 (Full Forward)
        """
        # 1. Deadzone check
        if abs(speed) < 0.05:
            # Stop command (Forward at 0 speed)
            self._send(f"f 0 {self.CMD_DURATION}\n") 
            return

        # 2. Convert 0.0-1.0 to MIN_PWM-255
        abs_speed = abs(speed)
        pwm = int(self.MIN_PWM + (self.MAX_PWM - self.MIN_PWM) * abs_speed)
        pwm = min(pwm, 255) # Cap at 255

        # 3. Choose Direction
        direction = 'f' if speed > 0 else 'b'

        # 4. Send Command (e.g., "f 150 100")
        cmd = f"{direction} {pwm} {self.CMD_DURATION}\n"
        self._send(cmd)

    def stop(self):
        self._send(f"f 0 {self.CMD_DURATION}\n")

    def close(self):
        self.stop()
        if self.ser:
            self.ser.close()