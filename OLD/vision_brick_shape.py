"""
Brick Vision (Shape-Only)
-------------------------
Detects the brick face by contour geometry (no color),
estimates angle from the outline, and flags stacked bricks
above/below the centermost detection.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from helper_stream_server import StreamServer, format_stream_url


WORLD_MODEL_BRICK_FILE = Path(__file__).parent / "world_model_brick.json"

MIN_AREA_RATIO = 0.001
MIN_AREA_PIXELS = 400.0
MAX_AREA_RATIO = 0.95
MATCH_THRESHOLD = 2.0  # Safe with strong CLAHE+Median preprocessing
MIN_ASPECT = 1.1
MAX_ASPECT = 4.5
CANNY_SIGMA = 0.33
EDGE_CLOSE_ITERS = 4
EDGE_DILATE_ITERS = 2
TAB_BAND_RATIO = 0.2
TAB_CENTER_RATIO = 0.12
HOUGH_THRESHOLD = 25
HOUGH_MIN_LINE_RATIO = 0.3
HOUGH_MAX_GAP_RATIO = 0.35
LINE_VERTICAL_TOL_DEG = 70.0
LINE_VERTICAL_KERNEL = 7
CENTER_LINE_OFFSET_TOL = 0.1
CENTER_LINE_SEPARATION_TOL = 0.1
CENTER_LINE_SIDE_MARGIN_RATIO = 0.15
LINE_SCORE_SINGLE = 0.55
LINE_SCORE_PAIR = 0.9
LINE_CLAHE_CLIP = 2.0
LINE_CLAHE_TILE = 8
LINE_SOBEL_KSIZE = 3
LINE_SOBEL_PERCENTILE = 85.0
LINE_MASK_CLOSE_ITERS = 2
LINE_MASK_OPEN_ITERS = 1
LINE_MASK_MIN_THRESHOLD = 8.0
STREAM_HOST = "127.0.0.1"
STREAM_PORT = 5000
STREAM_FPS = 10
STREAM_JPEG_QUALITY = 85
TEMPLATE_PREVIEW_W = 160
TEMPLATE_PREVIEW_H = 120
TEMPLATE_PREVIEW_MARGIN = 6
TOP_MASK_RATIO = 0.4

DEFAULT_FACE_POLYGON = [
    {"x": -32, "y": 0},
    {"x": -32, "y": 38},
    {"x": -12, "y": 38},
    {"x": -12, "y": 48},
    {"x": 12, "y": 48},
    {"x": 12, "y": 38},
    {"x": 32, "y": 38},
    {"x": 32, "y": 0},
    {"x": 16, "y": 0},
    {"x": 16, "y": 8},
    {"x": -16, "y": 8},
    {"x": -16, "y": 0},
]


@dataclass
class BrickDetection:
    found: bool
    angle_deg: float
    center: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    match_score: float
    confidence: float
    brickAbove: bool
    brickBelow: bool


class BrickShapeDetector:
    def __init__(self, debug: bool = True, model_path: Optional[Path] = None):
        self.debug = debug
        self.headless = not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        self.model_path = model_path or WORLD_MODEL_BRICK_FILE
        model = self._load_model(self.model_path)
        self.brick_width_mm, self.brick_height_mm = self._load_dimensions(model)
        self.template_contour = self._load_template_contour(model)
        self.center_line_offsets, self.center_line_sep_ratio = self._load_center_lines(model)
        (
            self.template_outline,
            self.template_left,
            self.template_right,
            self.template_bounds,
            self.template_line_xs,
        ) = self._load_template_polygons(model)
        self._line_clahe = cv2.createCLAHE(
            clipLimit=LINE_CLAHE_CLIP, tileGridSize=(LINE_CLAHE_TILE, LINE_CLAHE_TILE)
        )
        self.debug_frame = None

    def _load_dimensions(self, model: Optional[dict]) -> Tuple[float, float]:
        width = 64.0
        height = 48.0
        if model:
            dims = model.get("brick", {}).get("dimensions", {})
            width = float(dims.get("width", width)) or width
            height = float(dims.get("height", height)) or height
        if width <= 0:
            width = 64.0
        if height <= 0:
            height = 48.0
        return width, height

    def _load_model(self, model_path: Path) -> Optional[dict]:
        if model_path.exists():
            try:
                with open(model_path, "r") as f:
                    return json.load(f)
            except (OSError, ValueError, TypeError):
                return None
        return None

    def _load_template_contour(self, model: Optional[dict]) -> np.ndarray:
        face_poly = None
        if model:
            face_poly = model.get("brick", {}).get("facePolygon")
        if not face_poly:
            face_poly = DEFAULT_FACE_POLYGON
        points = np.array([[p["x"], p["y"]] for p in face_poly], dtype=np.float32)
        return points.reshape((-1, 1, 2))

    def _load_center_lines(self, model: Optional[dict]) -> Tuple[List[float], float]:
        offsets = []
        width_mm = 64.0
        if model:
            dims = model.get("brick", {}).get("dimensions", {})
            width_mm = float(dims.get("width", width_mm)) or width_mm
            face_lines = model.get("brick", {}).get("faceLines", [])
            xs = []
            for line in face_lines:
                p1 = line.get("p1", {})
                p2 = line.get("p2", {})
                if "x" in p1:
                    xs.append(float(p1["x"]))
                elif "x" in p2:
                    xs.append(float(p2["x"]))
            xs = sorted(set(xs))
            if xs:
                offsets = [x / width_mm for x in xs]
        if not offsets:
            offsets = [-0.0625, 0.0625]
        offsets = sorted(offsets)
        if len(offsets) >= 2:
            sep_ratio = abs(offsets[-1] - offsets[0])
        else:
            sep_ratio = 0.125
        return offsets, sep_ratio

    def _clip_polygon_vertical(
        self, points: List[Tuple[float, float]], x_clip: float, keep_left: bool
    ) -> List[Tuple[float, float]]:
        if not points:
            return []
        def inside(pt):
            return pt[0] <= x_clip if keep_left else pt[0] >= x_clip

        def intersect(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            if x2 == x1:
                return (x_clip, y1)
            t = (x_clip - x1) / (x2 - x1)
            t = max(0.0, min(1.0, t))
            y = y1 + t * (y2 - y1)
            return (x_clip, y)

        output = []
        prev = points[-1]
        prev_in = inside(prev)
        for curr in points:
            curr_in = inside(curr)
            if curr_in:
                if not prev_in:
                    output.append(intersect(prev, curr))
                output.append(curr)
            elif prev_in:
                output.append(intersect(prev, curr))
            prev = curr
            prev_in = curr_in
        return output

    def _load_template_polygons(
        self, model: Optional[dict]
    ) -> Tuple[
        List[Tuple[float, float]],
        List[Tuple[float, float]],
        List[Tuple[float, float]],
        Tuple[float, float, float, float],
        List[float],
    ]:
        face_poly = None
        if model:
            face_poly = model.get("brick", {}).get("facePolygon")
        if not face_poly:
            face_poly = DEFAULT_FACE_POLYGON
        outline = [(float(p["x"]), float(p["y"])) for p in face_poly]
        xs = [p[0] for p in outline]
        ys = [p[1] for p in outline]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        width_mm = max_x - min_x if max_x > min_x else 64.0
        line_xs = []
        if model:
            face_lines = model.get("brick", {}).get("faceLines", [])
            for line in face_lines:
                p1 = line.get("p1", {})
                p2 = line.get("p2", {})
                if "x" in p1:
                    line_xs.append(float(p1["x"]))
                if "x" in p2:
                    line_xs.append(float(p2["x"]))
        line_xs = sorted(set(line_xs))
        left_x = None
        right_x = None
        if len(line_xs) >= 2:
            negatives = [x for x in line_xs if x < 0.0]
            positives = [x for x in line_xs if x > 0.0]
            if negatives and positives:
                left_x = max(negatives)
                right_x = min(positives)
            else:
                left_x = line_xs[0]
                right_x = line_xs[-1]
        else:
            offsets = self.center_line_offsets or [-0.0625, 0.0625]
            if len(offsets) >= 2:
                left_x = min(offsets) * width_mm
                right_x = max(offsets) * width_mm
        if left_x is None or right_x is None:
            left_x = -4.0
            right_x = 4.0
        left_poly = self._clip_polygon_vertical(outline, left_x, keep_left=True)
        right_poly = self._clip_polygon_vertical(outline, right_x, keep_left=False)
        return outline, left_poly, right_poly, (min_x, min_y, max_x, max_y), [left_x, right_x]

    def _auto_canny(self, gray: np.ndarray) -> np.ndarray:
        v = np.median(gray)
        lower = int(max(0, (1.0 - CANNY_SIGMA) * v))
        upper = int(min(255, (1.0 + CANNY_SIGMA) * v))
        return cv2.Canny(gray, lower, upper)

    def _edge_mask(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Scenario 2: Gaussian 5 + Auto Canny
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = self._auto_canny(blur)
        
        # Morphology
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=EDGE_CLOSE_ITERS)
        edges = cv2.dilate(edges, kernel, iterations=EDGE_DILATE_ITERS)
        return edges

    def _line_mask(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = self._line_clahe.apply(gray)
        blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
        sobelx = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=LINE_SOBEL_KSIZE)
        absx = cv2.convertScaleAbs(sobelx)
        threshold = float(np.percentile(absx, LINE_SOBEL_PERCENTILE))
        threshold = max(LINE_MASK_MIN_THRESHOLD, threshold)
        _, mask = cv2.threshold(absx, threshold, 255, cv2.THRESH_BINARY)
        vert_kernel = np.ones((LINE_VERTICAL_KERNEL, 1), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, vert_kernel, iterations=LINE_MASK_CLOSE_ITERS)
        if LINE_MASK_OPEN_ITERS > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=LINE_MASK_OPEN_ITERS)
        return mask

    def _rect_angle(self, rect) -> float:
        (_, _), (w, h), angle = rect
        if w < h:
            angle += 90.0
        if angle >= 180.0:
            angle -= 180.0
        if angle < -180.0:
            angle += 180.0
        return angle

    def _count_segments(self, occupied: np.ndarray) -> int:
        count = 0
        active = False
        for val in occupied:
            if val and not active:
                count += 1
                active = True
            elif not val and active:
                active = False
        return count

    def _contour_mask(self, cnt: np.ndarray, center: Tuple[float, float], angle_deg: float) -> Optional[np.ndarray]:
        pts = cnt.reshape(-1, 2).astype(np.float32)
        pts -= np.array(center, dtype=np.float32)
        theta = math.radians(-angle_deg)
        rot = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
            dtype=np.float32,
        )
        rot_pts = pts @ rot.T
        min_xy = rot_pts.min(axis=0)
        max_xy = rot_pts.max(axis=0)
        width = int(math.ceil(max_xy[0] - min_xy[0])) + 2
        height = int(math.ceil(max_xy[1] - min_xy[1])) + 2
        if width <= 2 or height <= 2:
            return None
        shifted = rot_pts - min_xy + 1
        poly = np.round(shifted).astype(np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        return mask

    def _infer_tab_flip(self, cnt: np.ndarray, center: Tuple[float, float], angle_deg: float) -> Optional[int]:
        mask = self._contour_mask(cnt, center, angle_deg)
        if mask is None:
            return None
        height, width = mask.shape
        band = max(2, int(round(height * TAB_BAND_RATIO)))
        if band * 2 >= height:
            return None
        top = mask[:band, :]
        bottom = mask[-band:, :]
        top_proj = np.any(top > 0, axis=0)
        bottom_proj = np.any(bottom > 0, axis=0)
        top_segments = self._count_segments(top_proj)
        bottom_segments = self._count_segments(bottom_proj)
        center_idx = width // 2
        span = max(2, int(round(width * TAB_CENTER_RATIO)))
        left = max(0, center_idx - span)
        right = min(width, center_idx + span)
        top_center_on = np.any(top_proj[left:right])
        bottom_center_on = np.any(bottom_proj[left:right])
        if top_center_on and not bottom_center_on:
            return 0
        if bottom_center_on and not top_center_on:
            return 180
        if bottom_segments > top_segments:
            return 0
        if top_segments > bottom_segments:
            return 180
        return None

    def _normalize_angle(self, angle: float) -> float:
        while angle >= 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle

    def _normalize_line_angle(self, angle: float) -> float:
        if angle < 0.0:
            angle += 180.0
        if angle >= 180.0:
            angle -= 180.0
        return angle

    def _detect_center_lines(
        self, line_mask: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Tuple[
        Optional[float],
        bool,
        float,
        List[Tuple[Tuple[int, int], Tuple[int, int]]],
        Optional[Tuple[int, int, int, int]],
        Optional[Tuple[Tuple[float, float], Tuple[float, float], float]],
    ]:
        bx, by, bw, bh = bbox
        if bw <= 0 or bh <= 0:
            return None, False, 0.0, [], None, None
        x1 = max(0, bx)
        y1 = max(0, by)
        x2 = min(line_mask.shape[1], bx + bw)
        y2 = min(line_mask.shape[0], by + bh)
        roi = line_mask[y1:y2, x1:x2].copy()
        if roi.size == 0:
            return None, False, 0.0, [], None, None
        side_margin = int(round(roi.shape[1] * CENTER_LINE_SIDE_MARGIN_RATIO))
        if side_margin * 2 < roi.shape[1]:
            roi[:, :side_margin] = 0
            roi[:, -side_margin:] = 0
        vert_kernel = np.ones((LINE_VERTICAL_KERNEL, 1), np.uint8)
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, vert_kernel, iterations=1)
        min_len = max(8, int(round(roi.shape[0] * HOUGH_MIN_LINE_RATIO)))
        max_gap = max(3, int(round(roi.shape[0] * HOUGH_MAX_GAP_RATIO)))
        lines = cv2.HoughLinesP(
            roi, 1, np.pi / 180, HOUGH_THRESHOLD, minLineLength=min_len, maxLineGap=max_gap
        )
        if lines is None:
            return None, False, 0.0, [], None, None
        expected_offsets = self.center_line_offsets
        matches = {exp: None for exp in expected_offsets}
        line_segments = []
        for line in lines:
            lx1, ly1, lx2, ly2 = line[0]
            dx = lx2 - lx1
            dy = ly2 - ly1
            length = math.hypot(dx, dy)
            if length < min_len:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            angle = self._normalize_line_angle(angle)
            if abs(angle - 90.0) > LINE_VERTICAL_TOL_DEG:
                continue
            mid_x = (lx1 + lx2) / 2.0
            mid_y = (ly1 + ly2) / 2.0
            abs_mid_x = x1 + mid_x
            abs_mid_y = y1 + mid_y
            offset_ratio = (mid_x - (roi.shape[1] / 2.0)) / max(1.0, roi.shape[1])
            for exp in expected_offsets:
                diff = abs(offset_ratio - exp)
                if diff <= CENTER_LINE_OFFSET_TOL:
                    prev = matches.get(exp)
                    if prev is None or diff < prev["diff"]:
                        matches[exp] = {
                            "angle": angle,
                            "length": length,
                            "offset_ratio": offset_ratio,
                            "diff": diff,
                            "mid": (abs_mid_x, abs_mid_y),
                            "segment": (
                                (x1 + lx1, y1 + ly1),
                                (x1 + lx2, y1 + ly2),
                            ),
                        }
        matched = [m for m in matches.values() if m is not None]
        if matched:
            for m in matched:
                line_segments.append(m["segment"])
        line_pair = False
        line_bbox = None
        line_rect = None
        if len(matched) >= 2:
            matched_sorted = sorted(matched, key=lambda m: m["offset_ratio"])
            left = matched_sorted[0]
            right = matched_sorted[-1]
            if left["offset_ratio"] < 0.0 and right["offset_ratio"] > 0.0:
                actual_sep = abs(right["offset_ratio"] - left["offset_ratio"])
                if abs(actual_sep - self.center_line_sep_ratio) <= CENTER_LINE_SEPARATION_TOL:
                    line_pair = True
        if not matched:
            return None, False, 0.0, [], None, None
        weights = [m["length"] for m in matched]
        angles = [m["angle"] for m in matched]
        angle_avg = sum(a * w for a, w in zip(angles, weights)) / max(1.0, sum(weights))
        brick_angle = self._normalize_angle(angle_avg - 90.0)
        if line_pair:
            matched_sorted = sorted(matched, key=lambda m: m["offset_ratio"])
            left = matched_sorted[0]
            right = matched_sorted[-1]
            sep_px = abs(right["mid"][0] - left["mid"][0])
            if sep_px > 0.0 and self.center_line_sep_ratio > 0.0:
                width_px = sep_px / self.center_line_sep_ratio
                height_px = width_px * (self.brick_height_mm / self.brick_width_mm)
                center_x = (left["mid"][0] + right["mid"][0]) / 2.0
                center_y = sum(m["mid"][1] * m["length"] for m in matched) / max(1.0, sum(weights))
                if width_px > 1.0 and height_px > 1.0:
                    line_rect = ((center_x, center_y), (width_px, height_px), brick_angle)
                    box = cv2.boxPoints(line_rect)
                    bbox = cv2.boundingRect(np.int32(box))
                    frame_h, frame_w = line_mask.shape
                    bx2, by2, bw2, bh2 = bbox
                    bx2 = max(0, bx2)
                    by2 = max(0, by2)
                    bw2 = min(frame_w - bx2, bw2)
                    bh2 = min(frame_h - by2, bh2)
                    if bw2 > 0 and bh2 > 0:
                        line_bbox = (bx2, by2, bw2, bh2)
        line_score = LINE_SCORE_PAIR if line_pair else LINE_SCORE_SINGLE
        return brick_angle, line_pair, line_score, line_segments, line_bbox, line_rect

    def _evaluate_candidates(
        self, contours: List[np.ndarray], edges: np.ndarray, line_mask: np.ndarray, frame_shape: Tuple[int, int]
    ) -> List[dict]:
        frame_h, frame_w = frame_shape
        frame_area = frame_h * frame_w
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
            
            # Rejection based on vertical position (ignore background)
            if cy < frame_h * TOP_MASK_RATIO:
                continue

            aspect = max(w, h) / max(1.0, min(w, h))
            if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
            match_score = cv2.matchShapes(self.template_contour, cnt, cv2.CONTOURS_MATCH_I1, 0.0)
            
            bbox = cv2.boundingRect(cnt)
            (
                line_angle,
                line_pair,
                line_score,
                line_segments,
                line_bbox,
                line_rect,
            ) = self._detect_center_lines(line_mask, bbox)
            if match_score > MATCH_THRESHOLD and (not line_pair and line_angle is None):
                continue
            
            axis_angle = self._rect_angle(rect)
            angle = axis_angle
            if line_angle is not None:
                angle = line_angle
            if line_rect is not None:
                (cx, cy), _, _ = line_rect
                bbox = line_bbox or bbox
            flip = self._infer_tab_flip(cnt, (cx, cy), angle)
            if flip:
                angle = self._normalize_angle(angle + flip)
            confidence = max(0.0, min(1.0, 1.0 - (match_score / MATCH_THRESHOLD)))
            if line_score > 0.0:
                confidence = max(confidence, line_score)
            candidates.append(
                {
                    "center": (float(cx), float(cy)),
                    "angle_deg": float(angle),
                    "axis_angle": float(axis_angle),
                    "match_score": float(match_score),
                    "confidence": float(confidence),
                    "contour": cnt,
                    "bbox": bbox,
                    "line_angle": line_angle,
                    "line_pair": line_pair,
                    "line_score": line_score,
                    "line_segments": line_segments,
                    "line_rect": line_rect,
                }
            )
        return candidates

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
        edges = self._edge_mask(frame)
        line_mask = self._line_mask(frame)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = self._evaluate_candidates(contours, edges, line_mask, frame.shape[:2])

        selected = None
        if candidates:
            center_x = frame.shape[1] / 2.0
            center_y = frame.shape[0] / 2.0
            # IMPROVEMENT: Weigh distance by Area to favor large brick contours 
            # over small background noise (common with cement/wood textures)
            # We also give a massive bonus to objects with a line pair.
            selected = min(
                candidates,
                key=lambda c: (
                    ((c["center"][0] - center_x) ** 2 + (c["center"][1] - center_y) ** 2)
                    / (max(1.0, cv2.contourArea(c["contour"])) ** 0.5)
                    * (0.1 if c["line_pair"] else 1.0)
                ),
            )

        brick_above, brick_below = self._stack_flags(selected, candidates)
        if selected:
            detection = BrickDetection(
                found=True,
                angle_deg=selected["angle_deg"],
                center=selected["center"],
                bbox=selected["bbox"],
                match_score=selected["match_score"],
                confidence=selected["confidence"],
                brickAbove=brick_above,
                brickBelow=brick_below,
            )
        else:
            detection = BrickDetection(
                found=False,
                angle_deg=0.0,
                center=(0.0, 0.0),
                bbox=(0, 0, 0, 0),
                match_score=1.0,
                confidence=0.0,
                brickAbove=False,
                brickBelow=False,
            )

        if self.debug:
            self.debug_frame = self._draw_debug(frame, edges, candidates, selected, detection)
        return detection

    def _draw_debug(
        self,
        frame: np.ndarray,
        edges: np.ndarray,
        candidates: List[dict],
        selected: Optional[dict],
        detection: BrickDetection,
    ) -> np.ndarray:
        display = frame.copy()
        for cand in candidates:
            color = (0, 255, 0) if cand is selected else (0, 165, 255)
            cv2.drawContours(display, [cand["contour"]], -1, color, 2)
            cx, cy = cand["center"]
            cv2.circle(display, (int(cx), int(cy)), 4, color, -1)
            bx, by, bw, bh = cand["bbox"]
            cv2.rectangle(display, (bx, by), (bx + bw, by + bh), color, 1)
            cv2.putText(
                display,
                f"{cand['match_score']:.3f}",
                (bx, max(12, by - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )
            if cand["line_segments"]:
                for seg in cand["line_segments"]:
                    cv2.line(display, seg[0], seg[1], (255, 0, 255), 2)
            if cand["line_angle"] is not None:
                cv2.putText(
                    display,
                    f"L {cand['line_angle']:.1f}",
                    (bx, by + bh + 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )
            if cand["line_rect"] is not None:
                box = cv2.boxPoints(cand["line_rect"])
                cv2.polylines(display, [np.int32(box)], True, (255, 0, 255), 2)
            cv2.putText(
                display,
                f"{cand['angle_deg']:.1f}deg",
                (bx, by + bh + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )

        if detection.found:
            status = f"angle={detection.angle_deg:.1f}  above={detection.brickAbove}  below={detection.brickBelow}"
        else:
            status = "no brick"
        
        cv2.putText(display, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        if display.shape[0] >= 120 and display.shape[1] >= 160:
            edge_small = cv2.resize(edges, (160, 120))
            edge_small = cv2.cvtColor(edge_small, cv2.COLOR_GRAY2BGR)
            display[0:120, display.shape[1] - 160 : display.shape[1]] = edge_small
        self._draw_template_preview(display)
        return display

    def _draw_template_preview(self, display: np.ndarray) -> None:
        if not self.template_outline:
            return
        preview = np.zeros((TEMPLATE_PREVIEW_H, TEMPLATE_PREVIEW_W, 3), dtype=np.uint8)
        preview[:] = (20, 20, 20)
        min_x, min_y, max_x, max_y = self.template_bounds
        width = max_x - min_x
        height = max_y - min_y
        if width <= 0 or height <= 0:
            return
        margin = TEMPLATE_PREVIEW_MARGIN
        scale_x = (TEMPLATE_PREVIEW_W - 2 * margin) / width
        scale_y = (TEMPLATE_PREVIEW_H - 2 * margin) / height
        scale = min(scale_x, scale_y)
        if scale <= 0:
            return

        def transform(points: List[Tuple[float, float]]) -> Optional[np.ndarray]:
            if len(points) < 3:
                return None
            arr = np.array(points, dtype=np.float32)
            xs = (arr[:, 0] - min_x) * scale + margin
            ys = (max_y - arr[:, 1]) * scale + margin
            pts = np.stack([xs, ys], axis=1).astype(np.int32)
            return pts.reshape((-1, 1, 2))

        left_pts = transform(self.template_left)
        right_pts = transform(self.template_right)
        outline_pts = transform(self.template_outline)

        if left_pts is not None:
            cv2.fillPoly(preview, [left_pts], (70, 120, 255))
        if right_pts is not None:
            cv2.fillPoly(preview, [right_pts], (120, 220, 120))
        if outline_pts is not None:
            cv2.polylines(preview, [outline_pts], True, (200, 200, 200), 1)

        for x in self.template_line_xs:
            lx = int(round((x - min_x) * scale + margin))
            cv2.line(preview, (lx, margin), (lx, TEMPLATE_PREVIEW_H - margin), (255, 0, 255), 1)

        cv2.putText(preview, "L", (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 120, 255), 1)
        cv2.putText(preview, "R", (TEMPLATE_PREVIEW_W - 14, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 220, 120), 1)

        edge_w = 160
        edge_h = 120
        if display.shape[0] >= edge_h + TEMPLATE_PREVIEW_H and display.shape[1] >= TEMPLATE_PREVIEW_W:
            x0 = display.shape[1] - TEMPLATE_PREVIEW_W
            y0 = edge_h
        elif display.shape[1] >= TEMPLATE_PREVIEW_W:
            x0 = display.shape[1] - TEMPLATE_PREVIEW_W
            y0 = max(0, display.shape[0] - TEMPLATE_PREVIEW_H)
        else:
            x0 = 0
            y0 = 0
        display[y0 : y0 + TEMPLATE_PREVIEW_H, x0 : x0 + TEMPLATE_PREVIEW_W] = preview


def _run_camera(
    camera_index: int,
    debug: bool,
    stream: bool,
    stream_host: str,
    stream_port: int,
    stream_fps: int,
    jpeg_quality: int,
) -> None:
    detector = BrickShapeDetector(debug=debug)
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
            title="Brick Shape Vision",
            header="Brick Shape Vision",
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
            cv2.imshow("Brick Shape Vision", detector.debug_frame)
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


def _run_image(path: Path, debug: bool) -> None:
    detector = BrickShapeDetector(debug=debug)
    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"Unable to read image: {path}")
    detection = detector.process(frame)
    print(detection)
    if debug and not detector.headless:
        cv2.imshow("Brick Shape Vision", detector.debug_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Shape-only brick detection (angle + above/below).")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for live detection.")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image to process.")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug overlays/window.")
    parser.add_argument("--stream", action="store_true", help="Serve a live MJPEG stream.")
    parser.add_argument("--no-stream", action="store_true", help="Disable the live MJPEG stream.")
    parser.add_argument("--stream-host", type=str, default=STREAM_HOST, help="Stream host.")
    parser.add_argument("--stream-port", type=int, default=STREAM_PORT, help="Stream port.")
    parser.add_argument("--stream-fps", type=int, default=STREAM_FPS, help="Stream FPS.")
    parser.add_argument("--stream-jpeg", type=int, default=STREAM_JPEG_QUALITY, help="Stream JPEG quality.")
    args = parser.parse_args()

    debug = not args.no_debug
    stream = args.stream
    if args.no_stream:
        stream = False
    elif not args.stream and not args.image:
        stream = True
    if args.image:
        _run_image(Path(args.image), debug)
    else:
        _run_camera(
            args.camera,
            debug,
            stream,
            args.stream_host,
            args.stream_port,
            args.stream_fps,
            args.stream_jpeg,
        )


if __name__ == "__main__":
    main()
