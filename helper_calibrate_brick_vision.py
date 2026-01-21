
import cv2
import time
import json
import threading
import sys
import tty
import termios
from pathlib import Path

# Fix relative imports
sys.path.append(str(Path(__file__).parent))

from helper_brick_vision import BrickDetector
from helper_stream_server import StreamServer, format_stream_url

WORLD_MODEL_BRICK_FILE = Path(__file__).parent / "world_model_brick.json"

class CalibrationApp:
    def __init__(self):
        self.vision = BrickDetector(debug=True, speed_optimize=False)
        self.current_frame = None
        self.lock = threading.Lock()
        self.running = True
        
        # Load initial config
        self.config = self._load_config()
        self.hue = self.config['color'].get('hue', 180)
        self.sat = self.config['color'].get('sat', 60)
        self.val = self.config['color'].get('val', 100)
        self.hex_code = self.config['color'].get('hex', '#7B8284')
        
        # Start Vision Loop
        self.vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self.vision_thread.start()
        
        # Start Stream Server
        self.server = StreamServer(self._get_frame, port=5000)
        self.server.start()
        print(f"[VISION] Stream started at {format_stream_url('127.0.0.1', 5000)}")

    def _load_config(self):
        if WORLD_MODEL_BRICK_FILE.exists():
            with open(WORLD_MODEL_BRICK_FILE, 'r') as f:
                data = json.load(f)
                return data['brick']
        return {'color': {}}

    def _save_config(self):
        if WORLD_MODEL_BRICK_FILE.exists():
            with open(WORLD_MODEL_BRICK_FILE, 'r') as f:
                data = json.load(f)
        else:
            data = {'brick': {'color': {}}}
            
        data['brick']['color']['hue'] = self.hue
        data['brick']['color']['sat'] = self.sat
        data['brick']['color']['val'] = self.val
        data['brick']['color']['hex'] = self.hex_code
        
        with open(WORLD_MODEL_BRICK_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print("[CONFIG] Saved to world_model_brick.json")

    def _get_frame(self):
        with self.lock:
            return self.current_frame

    def _vision_loop(self):
        while self.running:
            # Apply current calibration to vision system (Mocking update_settings)
            # Since we removed the runtime update method methods, we have to reload from file or inject manually.
            # But wait, I removed the update_settings method from BrickDetector!
            # I need to manually update the internal config of the detector instance.
            
            try:
                # Update HSV ranges using local config + current tuning values
                self.vision.hsv_ranges = self.vision.hex_to_hsv_ranges(
                    self.hex_code, 
                    self.hue, self.sat, self.val
                )
            except Exception as e:
                print(f"Error in vision loop: {e}")

            self.vision.read()
            frame = self.vision.current_frame
            if frame is not None:
                # Draw Info
                overlay = frame.copy()
                text = f"HUE: {self.hue}  SAT: {self.sat}  VAL: {self.val}  HEX: {self.hex_code}"
                cv2.rectangle(overlay, (0,0), (640, 40), (0,0,0), -1)
                cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Draw Color Dot
                try:
                    h = self.hex_code.lstrip('#')
                    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) # R,G,B
                    bgr = (rgb[2], rgb[1], rgb[0])
                    cv2.circle(overlay, (600, 20), 12, bgr, -1)
                    cv2.circle(overlay, (600, 20), 13, (255,255,255), 2)
                except: pass

                with self.lock:
                    self.current_frame = overlay
            time.sleep(0.01)

    def display_hex(self, h):
        return h

    def run(self):
        # Restore Terminal Settings (Echo + Canonical)
        fd = sys.stdin.fileno()
        attr = termios.tcgetattr(fd)
        attr[3] |= termios.ECHO | termios.ICANON
        termios.tcsetattr(fd, termios.TCSANOW, attr)

        print("\n--- VISION CALIBRATION TOOL ---")
        print("Adjust the settings to see the brick clearly.")
        print("Goals:")
        print("1. Brick should be visible (White in mask).")
        print("2. Floor should be ignored (Black in mask).")
        print("-------------------------------")
        
        while self.running:
            print(f"\nCurrent Settings: Hex={self.hex_code}, Hue={self.hue}, Sat={self.sat}, Val={self.val}")
            print("1. Adjust Color Match (Hue)  [+/- 10]")
            print("2. Adjust Saturation  (Sat)  [+/- 10]")
            print("3. Adjust Brightness  (Val)  [+/- 10]")
            print("4. Change Hex Code")
            print("5. Save & M. Exit")
            print("Q. Quit (No Save)")
            
            try:
                choice = input("Select Option (1-5): ").strip().lower()
            except KeyboardInterrupt:
                break
                
            if choice == '1':
                val = input(f"Enter new Hue (Current {self.hue}): ").strip()
                if val.isdigit(): self.hue = int(val)
            elif choice == '2':
                val = input(f"Enter new Sat (Current {self.sat}): ").strip()
                if val.isdigit(): self.sat = int(val)
            elif choice == '3':
                val = input(f"Enter new Val (Current {self.val}): ").strip()
                if val.isdigit(): self.val = int(val)
            elif choice == '4':
                code = input(f"Enter new Hex (Current {self.hex_code}): ").strip()
                if code.startswith('#') and len(code) == 7:
                    self.hex_code = code
                else:
                    print("Invalid Hex! Must be format #RRGGBB")
            elif choice == '5':
                self._save_config()
                self.running = False
            elif choice == 'q':
                self.running = False

        print("Exiting...")
        self.server.stop()
        sys.exit(0)

if __name__ == "__main__":
    app = CalibrationApp()
    app.run()
