import cv2
import numpy as np
import threading
import time
import sys
import os
import logging
from pathlib import Path

# Setup path
sys.path.append(str(Path(__file__).parent))

from helper_vision_aruco import ArucoBrickVision
from helper_stream_server import StreamServer, format_stream_url

# Suppress Flask/Werkzeug logs
logging.getLogger('werkzeug').setLevel(logging.ERROR)

class ArucoApp:
    def __init__(self):
        self.vision = ArucoBrickVision(debug=False) # Simple mode
        self.lock = threading.Lock()
        self.current_frame = None
        self.running = True

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.vision.init_camera(640, 480)

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

        self.server = StreamServer(self._get_frame, port=5000)
        self.server.start()
        print(f"[ARUCO] Vision Started: {format_stream_url('127.0.0.1', 5000)}")

    def _get_frame(self):
        with self.lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
                
            pose = self.vision.process(frame)
            display = frame.copy()
            
            if pose.found:
                # 1. Clean Teal Marker
                teal = (238, 238, 175) # BGR for Teal/Cyan-ish
                pts = pose.corners.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(display, [pts], True, (255, 255, 0), 2) # Cyan/Teal
                
                # 2. Simple Info
                dist = pose.position[2]
                angle = pose.orientation[1]
                cv2.putText(display, f"Dist: {dist:.0f}mm | Angle: {angle:.1f}deg", (15, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                stack_info = []
                if pose.brickAbove: stack_info.append("ABOVE")
                if pose.brickBelow: stack_info.append("BELOW")
                if stack_info:
                    msg = "STACK: " + " + ".join(stack_info)
                    cv2.putText(display, msg, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            with self.lock:
                self.current_frame = display
            time.sleep(0.01)

    def run(self):
        print("ArUco Vision active. Press Enter to quit.")
        try: input()
        except: pass
        self.running = False
        self.server.stop()
        self.cap.release()

if __name__ == "__main__":
    ArucoApp().run()
