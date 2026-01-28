"""
helper_robot_control.py
----------------
The Translator.
Converts autonomous decisions into the specific command strings 
that your Arduino firmware expects (e.g., "f 200 50").
"""
import serial
import time
import sys
import threading

from telemetry_robot import MIN_PWM, MAX_PWM, MIN_TURN_POWER, COMBINED_MOVEMENTS

class CombinedMover:
    """
    Handles combined movements (Forward/Backward + Turn) in a background thread.
    This ensures a precise 50ms/50ms rhythm independent of the main program loop.
    """
    def __init__(self, robot):
        self.robot = robot
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.params = None
        
    def start(self, d_cmd, t_cmd, d_spd, t_spd):
        """Update parameters and ensure thread is running."""
        with self.lock:
            # Check if params changed to avoid restarting thread unnecessarily
            new_params = (d_cmd, t_cmd, d_spd, t_spd)
            if self.params == new_params and self.running:
                return
                
            self.params = new_params
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self._loop, daemon=True)
                self.thread.start()
    
    def stop(self):
        """Stop the background movement thread."""
        with self.lock:
            if not self.running:
                return
            self.running = False
            self.params = None
        # We don't join here to keep the main loop fast; thread will exit on next cycle
        
    def _loop(self):
        """Background loop toggling drive and turn commands."""
        while True:
            with self.lock:
                if not self.running or not self.params:
                    self.running = False
                    break
                d_cmd, t_cmd, d_spd, t_spd = self.params
            
            # 1. Drive Phase (50ms)
            # We call robot.send_command, which handles inversion/normalization.
            # It will NOT recurse back here because d_cmd ('f'/'b') is not a combined key.
            self.robot.send_command(d_cmd, d_spd)
            time.sleep(0.05)
            
            # Check exit
            with self.lock:
                if not self.running: break

            # 2. Turn Phase (50ms)
            self.robot.send_command(t_cmd, t_spd)
            time.sleep(0.05)


class Robot:
    def __init__(self):
        self.SERIAL_PORT = '/dev/ttyCH341USB0' 
        self.BAUD_RATE = 115200
        self.ser = None
        
        # --- PHYSICAL CONSTANTS (Single source: telemetry_robot) ---
        self.MIN_PWM = MIN_PWM
        self.MAX_PWM = MAX_PWM
        self.MIN_TURN_POWER = MIN_TURN_POWER
        self.CMD_DURATION = 100 # ms (Keep it running slightly longer for smooth auto-drive)
        
        # Background mover for combined diagonal actions
        self.mover = CombinedMover(self)
        
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

    def normalize_speed(self, cmd_char, speed):
        speed = abs(speed)
        if cmd_char in ('l', 'r') and 0.0 < speed < self.MIN_TURN_POWER:
            speed = self.MIN_TURN_POWER
        if speed < 0.05:
            return 0.0, 0
        pwm = int(self.MIN_PWM + (self.MAX_PWM - self.MIN_PWM) * speed)
        pwm = min(pwm, 255)
        return speed, pwm

    def send_command(self, cmd_char, speed):
        """
        Sends a high-level command: {char} {pwm} {duration}
        cmd_char: f, b, l, r, u, d, or combined movements (fl_slow, fl_fast, etc)
        speed: 0.0 to 1.0 (ignored for combined movements as they have internal speeds)
        """
        # 1. Check for combined movement commands from configuration
        if cmd_char in COMBINED_MOVEMENTS:
            config = COMBINED_MOVEMENTS[cmd_char]
            drive_cmd = config.get("drive_cmd")
            turn_cmd = config.get("turn_cmd")
            drive_speed = config.get("drive_speed", 0.0)
            turn_speed = config.get("turn_speed", 0.0)
            
            if drive_cmd and turn_cmd:
                # Hand off to background thread for precise timing
                self.mover.start(drive_cmd, turn_cmd, drive_speed, turn_speed)
                return

        # 2. For normal commands, ensure background mover is stopped
        # (Only stop if we are sending a conflicting move command)
        # If we send a STOP (speed=0), we definitely stop the mover.
        # If we send a regular move, we stop the mover to take control.
        self.mover.stop()
        
        speed, pwm = self.normalize_speed(cmd_char, speed)
        if speed <= 0.0:
            # For safety, sending 0 speed usually stops that action
            self._send(f"{cmd_char} 0 {self.CMD_DURATION}\n")
            return

        # 3. Hardware Inversion Correction
        # The robot wiring is currently swapped:
        # - 'f' moves backward, 'b' moves forward
        # - 'l' turns right, 'r' turns left
        # We fix this here so all high-level scripts (autolay, recording, etc) can use logical commands.
        real_hw_cmd = cmd_char
        if cmd_char == 'f':
            real_hw_cmd = 'b'
        elif cmd_char == 'b':
            real_hw_cmd = 'f'
        elif cmd_char == 'l':
            real_hw_cmd = 'r'
        elif cmd_char == 'r':
            real_hw_cmd = 'l'

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
        # Stop background mover
        self.mover.stop()
        
        # Stop everything. 'f 0' usually stops the base?
        # Let's send a stop for drive and lift to be sure.
        self._send(f"f 0 {self.CMD_DURATION}\n")
        self._send(f"u 0 {self.CMD_DURATION}\n")

    def close(self):
        self.stop()
        if self.ser:
            self.ser.close()
