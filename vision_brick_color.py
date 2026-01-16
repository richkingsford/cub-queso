"""
Brick Vision (Color + Notch)
----------------------------
Detects a painted brick face using color, finds the bottom notch,
and derives the brick angle from the 4 notch points.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from helper_stream_server import StreamServer, format_stream_url


WORLD_MODEL_BRICK_FILE = Path(__file__).parent / "world_model_brick.json"
DEFAULT_HEX_COLOR = "#081524"

H_MARGIN = 12
S_MARGIN = 110
V_MARGIN = 90

MIN_AREA_RATIO = 0.0015
MIN_AREA_PIXELS = 350.0
MAX_AREA_RATIO = 0.75
MIN_ASPECT = 1.0
MAX_ASPECT = 6.0
MASK_OPEN_ITERS = 1
MASK_CLOSE_ITERS = 2

NOTCH_DARK_PERCENTILE = 15.0
NOTCH_MIN_AREA_RATIO = 0.004
NOTCH_MAX_AREA_RATIO = 0.1
NOTCH_BOTTOM_MIN_RATIO = 0.55

STREAM_HOST = "127.0.0.1"
STREAM_PORT = 5000
STREAM_FPS = 10
STREAM_JPEG_QUALITY = 85


DEFAULT_NOTCH_POINTS = {
    "bottom_left": (-16.0, 0.0, 0.0),
    "top_left": (-16.0, 8.0, 0.0),
    "top_right": (16.0, 8.0, 0.0),
    "bottom_right": (16.0, 0.0, 0.0),
}


@dataclass
class BrickDetection:
    found: bool
    angle_deg: float
    center: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    confidence: float
    brickAbove: bool
    brickBelow: bool
    notch_points: Optional[np.ndarray]


class BrickColorDetector:
    def __init__(
        self,
        debug: bool = True,
        model_path: Optional[Path] = None,
        color_hex: Optional[str] = None,
        h_margin: int = H_MARGIN,
        s_margin: int = S_MARGIN,
        v_margin: int = V_MARGIN,
    ):
        self.debug = debug
        self.headless = not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        self.model_path = model_path or WORLD_MODEL_BRICK_FILE
        model = self._load_model(self.model_path)
        self.notch_points_3d = self._load_notch_points(model)
        self.color_hex = color_hex or self._load_color_hex(model) or DEFAULT_HEX_COLOR
        self.hsv_ranges = self._hex_to_hsv_ranges(self.color_hex, h_margin, s_margin, v_margin)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1), dtype="double")
        self.debug_frame = None

    def _load_model(self, model_path: Path) -> Optional[dict]:
        if model_path.exists():
            try:
                with open(model_path, "r") as f:
                    return json.load(f)
            except (OSError, ValueError, TypeError):
                return None
        return None

    def _load_color_hex(self, model: Optional[dict]) -> Optional[str]:
        if not model:
            return None
        color = model.get("brick", {}).get("color", {})
        hex_value = color.get("hex")
        if isinstance(hex_value, str) and hex_value.strip():
            return hex_value.strip()
        return None

    def _load_notch_points(self, model: Optional[dict]) -> np.ndarray:
        points = DEFAULT_NOTCH_POINTS.copy()
        if model:
            notch = model.get("brick", {}).get("notch", {})
            for p in notch.get("points_3d", []):
                label = p.get("label")
                if label in points:
                    points[label] = (float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0)))
        order = ["top_left", "top_right", "bottom_right", "bottom_left"]
        return np.array([points[label] for label in order], dtype="double")

    def _hex_to_hsv_ranges(self, hex_str: str, h_margin: int, s_margin: int, v_margin: int):
        hex_str = hex_str.lstrip("#")
        if len(hex_str) != 6:
            raise ValueError(f"Invalid hex color: {hex_str}")
        r, g, b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
        pixel = np.uint8([[[b, g, r]]])
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        s_low = max(0, s - s_margin)
        s_high = min(255, s + s_margin)
        v_low = max(0, v - v_margin)
        v_high = min(255, v + v_margin)
        if h - h_margin < 0:
            return [
                (np.array([0, s_low, v_low]), np.array([h + h_margin, s_high, v_high])),
                (np.array([180 + (h - h_margin), s_low, v_low]), np.array([180, s_high, v_high])),
            ]
        if h + h_margin > 180:
            return [
                (np.array([0, s_low, v_low]), np.array([(h + h_margin) - 180, s_high, v_high])),
                (np.array([h - h_margin, s_low, v_low]), np.array([180, s_high, v_high])),
            ]
        return [(np.array([h - h_margin, s_low, v_low]), np.array([h + h_margin, s_high, v_high]))]

    def _init_camera(self, width: int, height: int) -> None:
        focal = width
        center = (width / 2.0, height / 2.0)
        self.camera_matrix = np.array(
            [[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]],
            dtype="double",
        )

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle >= 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle

    @staticmethod
    def _order_box_points(points: np.ndarray) -> np.ndarray:
        pts = np.array(points, dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        ordered = np.zeros((4, 2), dtype="float32")
        ordered[0] = pts[np.argmin(s)]  # TL
        ordered[2] = pts[np.argmax(s)]  # BR
        ordered[1] = pts[np.argmin(diff)]  # TR
        ordered[3] = pts[np.argmax(diff)]  # BL
        return ordered

    def _build_mask(self, hsv: np.ndarray) -> np.ndarray:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.hsv_ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        kernel = np.ones((3, 3), np.uint8)
        if MASK_OPEN_ITERS > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MASK_OPEN_ITERS)
        if MASK_CLOSE_ITERS > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MASK_CLOSE_ITERS)
        return mask

    def _find_notch_points(
        self, hsv: np.ndarray, mask: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Tuple[Optional[np.ndarray], float, Optional[np.ndarray]]:
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return None, 0.0, None
        roi_mask = mask[y : y + h, x : x + w]
        if roi_mask.size == 0:
            return None, 0.0, None
        brick_pixels = roi_mask > 0
        if np.count_nonzero(brick_pixels) < 100:
            return None, 0.0, None
        v = hsv[y : y + h, x : x + w, 2]
        v_vals = v[brick_pixels]
        if v_vals.size == 0:
            return None, 0.0, None
        threshold = float(np.percentile(v_vals, NOTCH_DARK_PERCENTILE))
        dark = (v <= threshold) & brick_pixels
        dark = dark.astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0, dark
        best = None
        best_score = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            area_ratio = area / float(w * h)
            if area_ratio < NOTCH_MIN_AREA_RATIO or area_ratio > NOTCH_MAX_AREA_RATIO:
                continue
            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
            if cy < h * NOTCH_BOTTOM_MIN_RATIO:
                continue
            center_score = 1.0 - min(1.0, abs(cx - w / 2.0) / max(1.0, w / 2.0))
            bottom_score = min(1.0, (cy - h * NOTCH_BOTTOM_MIN_RATIO) / max(1.0, h * (1.0 - NOTCH_BOTTOM_MIN_RATIO)))
            score = (area_ratio * 2.0) + (center_score * 0.6) + (bottom_score * 0.4)
            if score > best_score:
                best_score = score
                best = cnt
        if best is None:
            return None, 0.0, dark
        rect = cv2.minAreaRect(best)
        box = cv2.boxPoints(rect)
        ordered = self._order_box_points(box)
        ordered[:, 0] += x
        ordered[:, 1] += y
        confidence = max(0.0, min(1.0, best_score))
        return ordered, confidence, dark

    def _solve_angle(self, image_points: np.ndarray) -> Optional[float]:
        if self.camera_matrix is None:
            return None
        flags = cv2.SOLVEPNP_IPPE if hasattr(cv2, "SOLVEPNP_IPPE") else cv2.SOLVEPNP_ITERATIVE
        success, rvec, _tvec = cv2.solvePnP(
            self.notch_points_3d, image_points.astype("double"), self.camera_matrix, self.dist_coeffs, flags=flags
        )
        if not success:
            return None
        rmat, _ = cv2.Rodrigues(rvec)
        brick_x_in_cam = rmat[:, 0]
        angle = -math.degrees(math.atan2(brick_x_in_cam[2], brick_x_in_cam[0]))
        return self._normalize_angle(angle)

    def _stack_flags(self, selected: dict, candidates: List[dict]) -> Tuple[bool, bool]:
        if not selected:
            return False, False
        sel_x, sel_y = selected["center"]
        _, _, w, h = selected["bbox"]
        x_tol = max(10.0, w * 0.6)
        y_tol = max(10.0, h * 0.35)
        above = False
        below = False
        for cand in candidates:
            if cand is selected:
                continue
            cx, cy = cand["center"]
            if abs(cx - sel_x) > x_tol:
                continue
            if cy < sel_y - y_tol:
                above = True
            elif cy > sel_y + y_tol:
                below = True
        return above, below

    def process(self, frame: np.ndarray) -> BrickDetection:
        if self.camera_matrix is None:
            self._init_camera(frame.shape[1], frame.shape[0])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self._build_mask(hsv)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = frame.shape[0] * frame.shape[1]
        min_area = max(MIN_AREA_PIXELS, frame_area * MIN_AREA_RATIO)
        max_area = frame_area * MAX_AREA_RATIO

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), _ = rect
            if w <= 0 or h <= 0:
                continue
            aspect = max(w, h) / max(1.0, min(w, h))
            if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
                continue
            bbox = cv2.boundingRect(cnt)
            notch_points, notch_conf, dark_mask = self._find_notch_points(hsv, mask, bbox)
            if notch_points is None:
                continue
            angle = self._solve_angle(notch_points)
            if angle is None:
                continue
            candidates.append(
                {
                    "center": (float(cx), float(cy)),
                    "bbox": bbox,
                    "angle": angle,
                    "confidence": notch_conf,
                    "contour": cnt,
                    "notch_points": notch_points,
                    "dark_mask": dark_mask,
                }
            )

        selected = None
        if candidates:
            center_x = frame.shape[1] / 2.0
            center_y = frame.shape[0] / 2.0
            selected = min(
                candidates,
                key=lambda c: (c["center"][0] - center_x) ** 2 + (c["center"][1] - center_y) ** 2,
            )

        brick_above, brick_below = self._stack_flags(selected, candidates)
        if selected:
            detection = BrickDetection(
                found=True,
                angle_deg=selected["angle"],
                center=selected["center"],
                bbox=selected["bbox"],
                confidence=selected["confidence"],
                brickAbove=brick_above,
                brickBelow=brick_below,
                notch_points=selected["notch_points"],
            )
        else:
            detection = BrickDetection(
                found=False,
                angle_deg=0.0,
                center=(0.0, 0.0),
                bbox=(0, 0, 0, 0),
                confidence=0.0,
                brickAbove=False,
                brickBelow=False,
                notch_points=None,
            )

        if self.debug:
            self.debug_frame = self._draw_debug(frame, mask, candidates, selected, detection)
        return detection

    def _draw_debug(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        candidates: List[dict],
        selected: Optional[dict],
        detection: BrickDetection,
    ) -> np.ndarray:
        display = frame.copy()
        for cand in candidates:
            color = (0, 255, 0) if cand is selected else (0, 165, 255)
            cv2.drawContours(display, [cand["contour"]], -1, color, 2)
            bx, by, bw, bh = cand["bbox"]
            cv2.rectangle(display, (bx, by), (bx + bw, by + bh), color, 1)
            for p in cand["notch_points"]:
                cv2.circle(display, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)
            cv2.putText(
                display,
                f"{cand['angle']:.1f}deg",
                (bx, by + bh + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        if detection.found:
            status = f"angle={detection.angle_deg:.1f}  above={detection.brickAbove}  below={detection.brickBelow}"
        else:
            status = "no brick"
        cv2.putText(display, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        if display.shape[0] >= 120 and display.shape[1] >= 160:
            mask_small = cv2.resize(mask, (160, 120))
            mask_small = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            display[0:120, display.shape[1] - 160 : display.shape[1]] = mask_small
        return display


def _run_camera(
    camera_index: int,
    debug: bool,
    stream: bool,
    stream_host: str,
    stream_port: int,
    stream_fps: int,
    jpeg_quality: int,
    color_hex: Optional[str],
    h_margin: int,
    s_margin: int,
    v_margin: int,
) -> None:
    detector = BrickColorDetector(
        debug=debug, color_hex=color_hex, h_margin=h_margin, s_margin=s_margin, v_margin=v_margin
    )
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Camera {camera_index} not found.")
    stream_server = None
    frame_lock = threading.Lock()
    latest_frame = {"frame": None}
    if stream:
        def frame_provider():
            with frame_lock:
                if latest_frame["frame"] is None:
                    return None
                return latest_frame["frame"].copy()
        stream_server = StreamServer(
            frame_provider,
            host=stream_host,
            port=stream_port,
            fps=stream_fps,
            jpeg_quality=jpeg_quality,
            title="Brick Color Vision",
            header="Brick Color Vision",
            footer="Press q to quit.",
        )
        stream_server.start()
        print(f"[STREAM] {format_stream_url(stream_host, stream_port)}")
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detection = detector.process(frame)
        if stream_server:
            output_frame = detector.debug_frame if detector.debug and detector.debug_frame is not None else frame
            with frame_lock:
                latest_frame["frame"] = output_frame.copy()
        if debug and not detector.headless:
            cv2.imshow("Brick Color Vision", detector.debug_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            if detection.found:
                print(
                    f"found angle={detection.angle_deg:.1f} above={detection.brickAbove} below={detection.brickBelow}",
                    flush=True,
                )
    cap.release()
    if stream_server:
        stream_server.stop()
    if debug and not detector.headless:
        cv2.destroyAllWindows()


def _run_image(path: Path, debug: bool, color_hex: Optional[str], h_margin: int, s_margin: int, v_margin: int) -> None:
    detector = BrickColorDetector(
        debug=debug, color_hex=color_hex, h_margin=h_margin, s_margin=s_margin, v_margin=v_margin
    )
    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"Unable to read image: {path}")
    detection = detector.process(frame)
    print(detection)
    if debug and not detector.headless:
        cv2.imshow("Brick Color Vision", detector.debug_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Color-based brick detection using notch points.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for live detection.")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image to process.")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug overlays/window.")
    parser.add_argument("--stream", action="store_true", help="Serve a live MJPEG stream.")
    parser.add_argument("--no-stream", action="store_true", help="Disable the live MJPEG stream.")
    parser.add_argument("--stream-host", type=str, default=STREAM_HOST, help="Stream host.")
    parser.add_argument("--stream-port", type=int, default=STREAM_PORT, help="Stream port.")
    parser.add_argument("--stream-fps", type=int, default=STREAM_FPS, help="Stream FPS.")
    parser.add_argument("--stream-jpeg", type=int, default=STREAM_JPEG_QUALITY, help="Stream JPEG quality.")
    parser.add_argument("--hex", type=str, default=None, help="Override hex color (e.g., #081524).")
    parser.add_argument("--h-margin", type=int, default=H_MARGIN, help="HSV hue margin.")
    parser.add_argument("--s-margin", type=int, default=S_MARGIN, help="HSV saturation margin.")
    parser.add_argument("--v-margin", type=int, default=V_MARGIN, help="HSV value margin.")
    args = parser.parse_args()

    debug = not args.no_debug
    stream = args.stream
    if args.no_stream:
        stream = False
    elif not args.stream and not args.image:
        stream = True

    if args.image:
        _run_image(Path(args.image), debug, args.hex, args.h_margin, args.s_margin, args.v_margin)
    else:
        _run_camera(
            args.camera,
            debug,
            stream,
            args.stream_host,
            args.stream_port,
            args.stream_fps,
            args.stream_jpeg,
            args.hex,
            args.h_margin,
            args.s_margin,
            args.v_margin,
        )


if __name__ == "__main__":
    main()
