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

    def send_command(self, cmd_char, speed):
        """
        Sends a high-level command: {char} {pwm} {duration}
        cmd_char: f, b, l, r, u, d
        speed: 0.0 to 1.0
        """
        # 1. Deadzone check (implied stop)
        if abs(speed) < 0.05:
            # For safety, sending 0 speed usually stops that action
            self._send(f"{cmd_char} 0 {self.CMD_DURATION}\n")
            return

        # 2. Scale to PWM
        pwm = int(self.MIN_PWM + (self.MAX_PWM - self.MIN_PWM) * abs(speed))
        pwm = min(pwm, 255)

        # 3. Hardware Inversion Correction
        # The robot wiring is currently swapped: 'f' moves backward, 'b' moves forward.
        # We fix this here so all high-level scripts (autolay, recording, etc) can use logical commands.
        real_hw_cmd = cmd_char
        if cmd_char == 'f': real_hw_cmd = 'b'
        elif cmd_char == 'b': real_hw_cmd = 'f'

        # 4. Send
        self._send(f"{real_hw_cmd} {pwm} {self.CMD_DURATION}\n")

    def drive(self, speed):
        """Wrapper for BACKWARD COMPATIBILITY with maneuvers.py"""
        if speed > 0:
            self.send_command('f', speed)
        elif speed < 0:
            self.send_command('b', abs(speed))
        else:
            self.stop()

    def spin(self, speed):
        """Wrapper: speed>0 -> Right, speed<0 -> Left"""
        if speed > 0:
            self.send_command('r', speed)
        elif speed < 0:
            self.send_command('l', abs(speed))
        else:
            self.stop() # Or just stop drive?
            
    def set_lift_motor(self, speed):
        """Wrapper: speed>0 -> Up, speed<0 -> Down"""
        if speed > 0:
            self.send_command('u', speed)
        elif speed < 0:
            self.send_command('d', abs(speed))
        else:
            self.send_command('u', 0)

    def stop(self):
        # Stop everything. 'f 0' usually stops the base?
        # Let's send a stop for drive and lift to be sure.
        self._send(f"f 0 {self.CMD_DURATION}\n")
        self._send(f"u 0 {self.CMD_DURATION}\n")

    def close(self):
        self.stop()
        if self.ser:
            self.ser.close()