import cv2
import numpy as np
import threading
import time
import sys
import tty
import termios
from pathlib import Path

# Setup path
sys.path.append(str(Path(__file__).parent))

from helper_vision_leia import LeiaVision
from helper_stream_server import StreamServer, format_stream_url

class LeiaApp:
    def __init__(self):
        self.vision = LeiaVision(debug=True)
        self.lock = threading.Lock()
        self.current_frame = None
        self.running = True

        # Camera Setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Init Vision Camera Matrix
        self.vision.init_camera(640, 480)

        # Start Thread
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

        # Start Stream
        self.server = StreamServer(self._get_frame, port=5000)
        self.server.start()
        print(f"[LEIA] Vision System Started at {format_stream_url('127.0.0.1', 5000)}")

    def _get_frame(self):
        with self.lock:
            return self.current_frame

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
                
            # Run Vision
            pose = self.vision.process(frame)
            
            # Get Debug View (Structure) or Original
            display = self.vision.debug_frame if self.vision.debug_frame is not None else frame
            
            # Overlay Text
            status_color = (0, 255, 255) if pose.found else (0, 0, 255) # Cyan for found, Red for lost
            text = f"FOUND: {pose.found} | Conf: {pose.confidence:.2f}"
            cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            if pose.found:
                xyz = f"XYZ: {pose.position[0]:.1f}, {pose.position[1]:.1f}, {pose.position[2]:.1f}"
                cv2.putText(display, xyz, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2) # Magenta
            
            # Draw Model Overlay
            self.draw_model_overlay(display)
            
            with self.lock:
                self.current_frame = display
            
            time.sleep(0.01)

    def run(self):
        # Restore Terminal
        fd = sys.stdin.fileno()
        attr = termios.tcgetattr(fd)
        attr[3] |= termios.ECHO | termios.ICANON
        termios.tcsetattr(fd, termios.TCSANOW, attr)
        
        print("\n--- LEIA VISION (Geometric) ---")
        print("Viewing the world through structure/edges.")
        print("Press Enter to quit.")
        
        try:
            input()
        except:
            pass
            
        self.running = False
        self.server.stop()
        self.cap.release()

    def draw_model_overlay(self, frame):
        # Draw 64x48mm brick model in bottom left
        # Scale: 2px per mm -> 128x96px
        scale = 2.0
        h, w = frame.shape[:2]
        pad = 20
        
        # Origin (Bottom Left of screen)
        ox = pad + 64 # Center of brick
        oy = h - pad  # Bottom of brick in mm space
        
        # Color: Cyan
        c = (255, 255, 0)
        
        # Draw Brick Face (Polygon from world_model_brick.json)
        # Coordinates: (-32,0), (-32,38), (-12,38), (-12,48), (12,48), (12,38), (32,38), (32,0), (16,0), (16,8), (-16,8), (-16,0)
        face_poly = [
            (-32, 0), (-32, 38), (-12, 38), (-12, 48), (12, 48), (12, 38), 
            (32, 38), (32, 0), (16, 0), (16, 8), (-16, 8), (-16, 0)
        ]
        
        poly_pts = []
        for nx, ny in face_poly:
            px = int(ox + (nx * scale))
            py = int(oy - (ny * scale))
            poly_pts.append([px, py])
            
        pts = np.array(poly_pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, c, 2)
        
        # Draw Face Lines (Vertical)
        # x=-4 and x=4 from y=0 to y=48
        face_lines = [
            [(-4, 0), (-4, 48)],
            [(4, 0), (4, 48)]
        ]
        for p1, p2 in face_lines:
            pt1 = (int(ox + (p1[0] * scale)), int(oy - (p1[1] * scale)))
            pt2 = (int(ox + (p2[0] * scale)), int(oy - (p2[1] * scale)))
            cv2.line(frame, pt1, pt2, c, 1)

        # Draw Notch Points (Relative to Center)
        # BL(-16,0), TL(-16,8), TR(16,8), BR(16,0)
        notches = [(-16, 0), (-16, 8), (16, 8), (16, 0)]
        for nx, ny in notches:
            px = int(ox + (nx * scale))
            py = int(oy - (ny * scale)) # Y is up in mm, down in pixels
            cv2.circle(frame, (px, py), 4, (0, 0, 255), -1) # Red dots
            
        cv2.putText(frame, "MODEL", (int(ox - 32*scale), int(oy - 50*scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

if __name__ == "__main__":
    app = LeiaApp()
    app.run()
