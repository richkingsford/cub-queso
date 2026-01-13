import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np
from flask import Flask, Response

DEFAULT_STREAM_HOST = "127.0.0.1"
DEFAULT_STREAM_FPS = 10
DEFAULT_JPEG_QUALITY = 90


def format_stream_url(host, port):
    return f"http://{host}:{port}"


class StreamServer:
    def __init__(
        self,
        frame_provider: Callable[[], Optional[np.ndarray]],
        host: str = DEFAULT_STREAM_HOST,
        port: int = 5000,
        fps: int = DEFAULT_STREAM_FPS,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        title: str = "Robot Leia",
        header: Optional[str] = None,
        footer: Optional[str] = None,
        img_width: Optional[int] = None,
        sharpen: bool = False,
    ):
        self.frame_provider = frame_provider
        self.host = host
        self.port = port
        self.fps = max(1, int(fps))
        self.jpeg_quality = int(jpeg_quality)
        self.title = title
        self.header = header or title
        self.footer = footer
        self.img_width = img_width
        self.sharpen = sharpen
        self._stop = threading.Event()
        self._thread = None

        self.app = Flask(__name__)

        @self.app.route("/")
        def index():
            return self._index_html()

        @self.app.route("/video_feed")
        def video_feed():
            return Response(
                self._generate_frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

    def start(self):
        self._thread = threading.Thread(
            target=self.app.run,
            kwargs={
                "host": self.host,
                "port": self.port,
                "debug": False,
                "use_reloader": False,
            },
            daemon=True,
        )
        self._thread.start()
        return self._thread

    def stop(self):
        self._stop.set()

    def _index_html(self):
        width_attr = f' width="{self.img_width}"' if self.img_width else ""
        footer_html = f"<p>{self.footer}</p>" if self.footer else ""
        return (
            "<html><head><title>"
            f"{self.title}"
            "</title>"
            "<style>"
            "body{background:#1a1a1a;color:#eee;font-family:sans-serif;"
            "text-align:center;margin-top:40px;}"
            ".stream{display:inline-block;border:4px solid #333;border-radius:8px;"
            "overflow:hidden;box-shadow:0 6px 20px rgba(0,0,0,0.35);}"
            "img{image-rendering:auto;}"
            "h1{color:#f0ad4e;}"
            "</style>"
            "</head><body>"
            f"<h1>{self.header}</h1>"
            f"<div class='stream'><img src='/video_feed'{width_attr}></div>"
            f"{footer_html}</body></html>"
        )

    def _generate_frames(self):
        frame_interval = 1.0 / self.fps
        last_sent = 0.0
        while not self._stop.is_set():
            now = time.time()
            elapsed = now - last_sent
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

            try:
                frame = self.frame_provider()
            except Exception:
                frame = None

            if frame is None:
                frame = self._placeholder_frame()
            else:
                frame = frame.copy()

            if self.sharpen:
                frame = self._apply_sharpen(frame)

            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not ok:
                continue
            last_sent = time.time()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + bytearray(encoded)
                + b"\r\n"
            )

    @staticmethod
    def _placeholder_frame():
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "WAITING FOR CAMERA...",
            (120, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        return frame

    @staticmethod
    def _apply_sharpen(frame):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(frame, -1, kernel)
