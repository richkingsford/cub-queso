"""
Brick Vision V106 - "The Fingerprint Validator"
-----------------------------------------------
1. SHAPE FINGERPRINTING:
   - Compares the geometry of the Detected Contour (Green) vs the Detected Notch (Orange).
   - Calculates a Confidence Score based on expected physical ratios (Width, Height, Alignment).
2. REJECTION LOGIC:
   - If Confidence < 60%, the detection is marked as "REJECTED" (Red Box).
   - The robot receives "found=False", keeping it still during bad detections (like hands).
3. DEBUGGING:
   - Prints detailed Confidence metrics to the console for tuning.
"""
import cv2
import numpy as np
import json
import math
import sys
import os
import time
from collections import deque
from pathlib import Path

# --- CONFIG ---
CAMERA_INDEX = 0
WORLD_MODEL_BRICK_FILE = Path(__file__).parent / "world_model_brick.json"
MIN_AREA_THRESHOLD = 1000 
SHADOW_CUT_THRESHOLD = 80
FACE_MATCH_MAX = 0.35
COLOR_MARGIN_SCALE = 0.7
NOTCH_CONFIDENCE_THRESHOLD = 75.0
BLOB_CONFIDENCE_THRESHOLD = 85.0
NOTCH_VERTICAL_TOL_RATIO = 0.12
NOTCH_HORIZONTAL_TOL_RATIO = 0.12
NOTCH_DARK_PERCENTILE = 12.0
NOTCH_MIN_AREA_RATIO = 0.002
NOTCH_MAX_AREA_RATIO = 0.08
NOTCH_ROW_TOLERANCE_RATIO = 0.08
NOTCH_ROW_MIN_REL = 0.2
NOTCH_ROW_MAX_REL = 0.8
NOTCH_ROW_MIN_SPAN_RATIO = 0.3

class BrickDetector:
    def __init__(self, debug=True, save_folder=None, speed_optimize=False):
        self.debug = debug
        self.speed_optimize = speed_optimize
        # Check for Display
        if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
             self.headless = True
        else:
             self.headless = True # Default to True for safety unless explicitly disabled elsewhere, 
                                  # but code below suggests we want non-headless if debug=True?
                                  # The original code had `self.headless = True` hardcoded on line 33!
                                  # Wait, line 33 was: `self.headless = True`
                                  # And line 61: `except: self.headless = True`
                                  # Let's make it dynamic based on Config + Environment.
        
        # Override if debug is requested but we have a display
        if self.debug and (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            self.headless = False
            
        # Camera State
        self.fail_count = 0
        
        self.save_folder = save_folder
        self.current_frame = None
        self.raw_frame = None
        
        if self.save_folder and not os.path.exists(self.save_folder):
            try: os.makedirs(self.save_folder)
            except: pass

        # Try to find camera (Indices 0-3) using Default Backend
        self.cap = None
        for idx in range(4):
            print(f"[VISION] Attempting to open Camera {idx}...")
            temp_cap = cv2.VideoCapture(idx)
            if temp_cap.isOpened():
                print(f"[VISION] Success! Connected to Camera {idx}")
                print(f"[VISION] Backend: {temp_cap.getBackendName()}")
                self.cap = temp_cap
                break
            temp_cap.release()
            
        if self.cap is None:
            print("[VISION] FATAL: Could not find ANY camera (0-3).")
            # We don't exit here, we let the first read() fail so the main loop handles it
            self.cap = cv2.VideoCapture(0) # Dummy
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) # Force Auto Exposure
        
        # Warm Up
        print("[VISION] Warming up (5 frames)...")
        for i in range(5): 
            ret, _ = self.cap.read()
            # Silent warmup
            
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4,1))
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.hsv_ranges = []
        # Default fallback
        # Default fallback
        self.search_margin_mm = 6.0
        self.camera_center_offset_px = 0.0
        self.brick_width_mm = 64.0
        self.brick_height_mm = 48.0
        self.brick_depth_mm = 20.0
        self.face_template_contour = None
        self.face_template_hull = None
        self.face_polygon_centered = None
        self.face_polygon_size_mm = None
        self.stack_history = deque(maxlen=6)
        self.object_points, self.template_contour = self.load_world_model()
        
        self.debug_search_zones = []

        if self.debug and not self.speed_optimize and not self.headless:
            try: cv2.namedWindow("Brick Vision Debug")
            except: 
                print("[VISION] Warning: Could not create window. Running headless.")
                self.headless = True

    def draw_text_with_bg(self, img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, text_color=(255, 255, 255), thickness=1, bg_color=(0, 0, 0)):
        x, y = pos
        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(img, (x, y - h - 5), (x + w, y + 5), bg_color, -1)
        cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

    def hex_to_hsv_ranges(self, hex_str, h_margin, s_margin, v_margin):
        scale = COLOR_MARGIN_SCALE
        h_margin = max(1, int(h_margin * scale))
        s_margin = max(1, int(s_margin * scale))
        v_margin = max(1, int(v_margin * scale))
        hex_str = hex_str.lstrip('#')
        r, g, b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
        pixel = np.uint8([[[b, g, r]]])
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        l_bound = np.array([max(0, h - h_margin), max(0, s - s_margin), max(0, v - v_margin)])
        u_bound = np.array([min(180, h + h_margin), min(255, s + s_margin), min(255, v + v_margin)])
        return [(l_bound, u_bound)]



    def load_world_model(self):
        pts_3d = np.zeros((4,3), dtype="double")
        if WORLD_MODEL_BRICK_FILE.exists():
            try:
                with open(WORLD_MODEL_BRICK_FILE, 'r') as f:
                    model = json.load(f)
                
                # Load Calibration
                cal = model.get('calibration', {})
                self.camera_center_offset_px = cal.get('camera_center_offset_px', 0.0)

                c = model['brick']['color']
                self.hsv_ranges = self.hex_to_hsv_ranges(
                    c.get('hex', '#AE363E'), c.get('hue_margin', 15),
                    c.get('sat_margin', 100), c.get('val_margin', 100)
                )
                
                # Load Search Margin
                notch = model['brick'].get('notch', {})
                self.notch_search_margin_mm = notch.get('search_margin_mm', 7.0)
                
                dims = model['brick'].get('dimensions', {})
                if dims:
                    self.brick_width_mm = float(dims.get('width', self.brick_width_mm))
                    self.brick_height_mm = float(dims.get('height', self.brick_height_mm))
                    self.brick_depth_mm = float(dims.get('depth', self.brick_depth_mm))

                face_poly = model['brick'].get('facePolygon', [])
                if face_poly:
                    face_pts = np.array(
                        [[p['x'], p['y']] for p in face_poly if 'x' in p and 'y' in p],
                        dtype="float32",
                    )
                    if face_pts.size:
                        min_x = float(np.min(face_pts[:, 0]))
                        max_x = float(np.max(face_pts[:, 0]))
                        min_y = float(np.min(face_pts[:, 1]))
                        max_y = float(np.max(face_pts[:, 1]))
                        center_x = (min_x + max_x) / 2.0
                        center_y = (min_y + max_y) / 2.0
                        centered = np.column_stack(
                            (face_pts[:, 0] - center_x, center_y - face_pts[:, 1])
                        ).astype("float32")
                        self.face_polygon_centered = centered
                        self.face_polygon_size_mm = (max_x - min_x, max_y - min_y)

                        contour = face_pts.reshape((-1, 1, 2))
                        self.face_template_contour = contour
                        self.face_template_hull = cv2.convexHull(contour)
                
                notch_pts = model['brick']['notch']['points_3d']
                self.notch_points_3d = notch_pts # Store for 2D projection logic
                p_map = {p['label']: [p['x'], p['y'], p['z']] for p in notch_pts}
                pts_3d = np.array([
                    p_map['bottom_left'], p_map['top_left'],
                    p_map['top_right'], p_map['bottom_right']
                ], dtype="double")
            except Exception as e:
                print(f"[VISION] Failed to load world model: {e}")
                self.hsv_ranges = []
                self.camera_center_offset_px = 0.0
                # Keep other defaults as set in __init__
        return pts_3d, None

    def init_camera_matrix(self, w, h):
        focal_length = w 
        center = (w / 2, h / 2)
        self.camera_matrix = np.array([[focal_length, 0, center[0]],[0, focal_length, center[1]],[0, 0, 1]], dtype="double")

    def find_vertical_boundary(self, mask, x, y_start, y_end, direction):
        if x < 0 or x >= mask.shape[1]: return None
        y1 = max(0, min(y_start, y_end))
        y2 = min(mask.shape[0], max(y_start, y_end))
        col = mask[y1:y2, x]
        if direction == -1: # Scanning UP
            scan_col = col[::-1] 
            hits = np.where(scan_col == 255)[0]
            if len(hits) == 0: return None
            return (y2 - 1) - hits[0]
        else: # Scanning DOWN
            hits = np.where(col == 0)[0]
            if len(hits) == 0: return y2 
            return y1 + hits[0]

    def calculate_fingerprint_confidence(self, contour, notch_points):
        """
        Calculates confidence scores based on how well the detected geometry
        matches a standard brick profile.
        Returns (notch_score, blob_score, reasons)
        """
        notch_score = 100.0
        blob_score = 100.0
        reasons = []

        # --- 1. BLOB ANALYSIS (Contour only) ---
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        # Use world model proportions to reject non-brick blobs.
        if self.brick_height_mm > 0:
            face_ratio = self.brick_width_mm / self.brick_height_mm
        else:
            face_ratio = 1.3
        if self.brick_depth_mm > 0:
            top_ratio = self.brick_width_mm / self.brick_depth_mm
        else:
            top_ratio = 1.0
        min_ratio = min(face_ratio, top_ratio) * 0.6
        max_ratio = max(face_ratio, top_ratio) * 1.4
        if not (min_ratio <= aspect_ratio <= max_ratio):
            blob_score -= 40
            reasons.append(f"Bad Blob Aspect: {aspect_ratio:.2f}")
        
        area = cv2.contourArea(contour)
        if area < MIN_AREA_THRESHOLD * 1.5:
            blob_score -= 20
            reasons.append("Blob Small")

        if self.face_template_hull is not None:
            match = cv2.matchShapes(cv2.convexHull(contour), self.face_template_hull, cv2.CONTOURS_MATCH_I1, 0.0)
            if match > FACE_MATCH_MAX:
                blob_score = 0.0
                reasons.append(f"Face Mismatch: {match:.2f}")

        if notch_points is None:
            return 0.0, max(0.0, blob_score), reasons

        # --- 2. NOTCH ANALYSIS (Fingerprinting) ---
        # Points: [BL, TL, TR, BR]
        notch_w = np.linalg.norm(notch_points[3] - notch_points[0])
        notch_h = np.linalg.norm(notch_points[1] - notch_points[0]) 
        notch_center_x = (notch_points[0][0] + notch_points[3][0]) / 2.0
        notch_bottom_y = (notch_points[0][1] + notch_points[3][1]) / 2.0

        # --- CHECK A: WIDTH RATIO ---
        width_ratio = notch_w / float(w)
        expected_width_ratio = 0.45 
        diff_w = abs(width_ratio - expected_width_ratio)
        if diff_w > 0.2: 
            notch_score -= (diff_w - 0.2) * 100
            reasons.append(f"Bad Width Ratio: {width_ratio:.2f}")

        # --- CHECK B: HEIGHT RATIO ---
        height_ratio = notch_h / float(h)
        # Lenient height check for steeper camera angles (we see the brick top)
        if height_ratio < 0.05: 
            notch_score -= 60 
            reasons.append(f"Contour Too Tall: {height_ratio:.2f}")
        elif height_ratio > 0.60:
            notch_score -= 40
            reasons.append(f"Contour Too Short: {height_ratio:.2f}")

        # --- CHECK C: BOTTOM ALIGNMENT ---
        cnt_bottom = y + h
        dist_from_bottom = abs(cnt_bottom - notch_bottom_y)
        if dist_from_bottom > (h * 0.12): 
            notch_score -= 40
            reasons.append(f"Notch Floating")

        # --- CHECK D: NOTCH VERTICAL SIDES & HORIZONTAL EDGES ---
        if notch_w > 0:
            left_dx = abs(notch_points[1][0] - notch_points[0][0])
            right_dx = abs(notch_points[2][0] - notch_points[3][0])
            top_dy = abs(notch_points[1][1] - notch_points[2][1])
            bottom_dy = abs(notch_points[0][1] - notch_points[3][1])
            if left_dx > (notch_w * NOTCH_VERTICAL_TOL_RATIO) or right_dx > (notch_w * NOTCH_VERTICAL_TOL_RATIO):
                notch_score -= 50
                reasons.append("Notch Sides Not Vertical")
            if top_dy > (notch_h * NOTCH_HORIZONTAL_TOL_RATIO) or bottom_dy > (notch_h * NOTCH_HORIZONTAL_TOL_RATIO):
                notch_score -= 35
                reasons.append("Notch Edges Not Level")

        for p in notch_points:
            if cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), False) < 0:
                notch_score = 0.0
                reasons.append("Notch Off Face")
                break

        return max(0.0, notch_score), max(0.0, blob_score), reasons

    def detect_notch_row(self, v_channel, contour):
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            return None

        fill = np.zeros(v_channel.shape, dtype=np.uint8)
        cv2.drawContours(fill, [contour], -1, 255, -1)
        roi_v = v_channel[y:y+h, x:x+w]
        roi_fill = fill[y:y+h, x:x+w]
        if roi_v.size == 0 or not np.any(roi_fill):
            return None

        vals = roi_v[roi_fill > 0]
        if vals.size < 30:
            return None

        dark_thresh = np.percentile(vals, NOTCH_DARK_PERCENTILE)
        dark_mask = (roi_v <= dark_thresh).astype(np.uint8) * 255
        dark_mask = cv2.bitwise_and(dark_mask, dark_mask, mask=roi_fill)
        kernel = np.ones((3, 3), np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        notch_cnts, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not notch_cnts:
            return None

        min_area = w * h * NOTCH_MIN_AREA_RATIO
        max_area = w * h * NOTCH_MAX_AREA_RATIO
        notch_centers = []
        for cnt in notch_cnts:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            if rh <= 0:
                continue
            aspect = rw / float(rh)
            if aspect < 0.5 or aspect > 2.0:
                continue
            notch_centers.append((rx + (rw / 2.0), ry + (rh / 2.0)))

        if len(notch_centers) < 2:
            return None

        tolerance = h * NOTCH_ROW_TOLERANCE_RATIO
        best_group = None
        for base in notch_centers:
            group = [p for p in notch_centers if abs(p[1] - base[1]) <= tolerance]
            if len(group) >= 2:
                xs = [p[0] for p in group]
                if (max(xs) - min(xs)) >= (w * NOTCH_ROW_MIN_SPAN_RATIO):
                    best_group = group
                    break

        if not best_group:
            return None

        avg_y = sum(p[1] for p in best_group) / len(best_group)
        notch_y = y + avg_y
        rel = (notch_y - y) / float(h)
        if rel < NOTCH_ROW_MIN_REL or rel > NOTCH_ROW_MAX_REL:
            return None
        return notch_y

    def project_face_polygon(self, rect):
        if self.face_polygon_centered is None or self.face_polygon_size_mm is None:
            return None
        (cx, cy), (w_rect, h_rect), angle = rect
        if w_rect <= 0 or h_rect <= 0:
            return None
        if w_rect < h_rect:
            w_rect, h_rect = h_rect, w_rect
            angle += 90.0
        face_w_mm, face_h_mm = self.face_polygon_size_mm
        if face_w_mm <= 0 or face_h_mm <= 0:
            return None

        scale = w_rect / face_w_mm
        theta = math.radians(angle)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        pts = []
        for x_mm, y_mm in self.face_polygon_centered:
            x_px = (x_mm * scale * cos_t) - (y_mm * scale * sin_t) + cx
            y_px = (x_mm * scale * sin_t) + (y_mm * scale * cos_t) + cy
            pts.append([int(round(x_px)), int(round(y_px))])
        return np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

    def stabilize_stack_state(self, above, below, found, confidence):
        if not found or confidence < 25.0:
            self.stack_history.clear()
            return False, False

        self.stack_history.append((above, below))
        if not self.stack_history:
            return above, below

        above_votes = sum(1 for a, _ in self.stack_history if a)
        below_votes = sum(1 for _, b in self.stack_history if b)
        threshold = max(2, (len(self.stack_history) // 2) + 1)
        return above_votes >= threshold, below_votes >= threshold

    def get_notch_from_contour(self, mask, contour):
        # 1. Fit Rotated Rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box) # Box is 4 corners, not ordered
        
        # 2. Identify Long Axis (Width) vs Short Axis (Height)
        # minAreaRect returns (center), (width, height), angle
        # Width/Height order depends on rotation, so we normalize.
        dim1, dim2 = rect[1]
        if dim1 > dim2:
            w_box, h_box = dim1, dim2
            angle_corr = rect[2]
        else:
            w_box, h_box = dim2, dim1
            angle_corr = rect[2] + 90
            
        # 3. Estimate Scale (Pixels per mm)
        # Using the dominant dimension as the Width (64mm)
        mm_per_pixel = w_box / 64.0
        margin_px = int(self.notch_search_margin_mm * mm_per_pixel) # Use notch-specific margin
        
        # 4. Find the "Bottom Left" corner of the face in 2D space
        # We need a vector system:
        # Origin = Bottom-Left of brick face
        # U-Vector = Along the width (Right)
        # V-Vector = Along the height (Up)
        
        # Sort points by Y to find the bottom-most points
        # Then sort those by X to find Bottom-Left vs Bottom-Right
        box_pts = sorted(box, key=lambda p: p[1]) # Sort by Y
        # The bottom 2 points are the last 2 in the list
        bottom_pts = sorted(box_pts[-2:], key=lambda p: p[0]) 
        top_pts = sorted(box_pts[:2], key=lambda p: p[0])
        
        p_bl = np.array(bottom_pts[0], dtype='float32')
        p_br = np.array(bottom_pts[1], dtype='float32')
        p_tl = np.array(top_pts[0], dtype='float32') # Approx top-left
        
        # Calculate Unit Vectors
        vec_right = p_br - p_bl
        len_right = np.linalg.norm(vec_right)
        if len_right < 1.0: return None, [] # Degenerate
        u_vec = vec_right / len_right
        
        vec_up = p_tl - p_bl
        len_up = np.linalg.norm(vec_up)
        if len_up < 1.0: return None, []
        v_vec = vec_up / len_up
        
        # 5. Project 3D Notch Points to 2D
        found_points = []
        projected_zones = []
        
        # Notch 3D (relative to center of bottom edge of face)
        # X: -16 to 16, Y: 0 to 8 (mm)
        # Brick origin in our vector system is Bottom-Left (X=-32)
        
        for pt in self.notch_points_3d:
            # Shift X from center-relative to BL-relative
            # pt['x'] is -16 (left notch side) or +16 (right notch side)
            # relative to center. 
            # So dist_from_left_edge_mm = pt['x'] + 32.0
            x_mm = pt['x'] + 32.0 
            y_mm = pt['y']        # Height up from bottom edge

            # Convert to pixels
            tx = x_mm * mm_per_pixel
            ty = y_mm * mm_per_pixel
            
            # Project using our vectors
            # P_proj = P_bl + (tx * u_vec) + (ty * v_vec)
            # Note: v_vec points "Up" in image space (towards top of screen) 
            # if we defined it that way. Actually p_tl has lower Y than p_bl.
            # So vec_up is numerically negative in Y. That's correct for "Up".
            
            projected_pt = p_bl + (tx * u_vec) + (ty * v_vec)
            px, py = int(projected_pt[0]), int(projected_pt[1])
            
            # Create search zone around this projected point
            # Since the zone is square, we don't need to rotate the zone itself,
            # just center it on the rotated projection.
            x1 = max(0, px - margin_px)
            x2 = min(mask.shape[1], px + margin_px)
            y1 = max(0, py - margin_px)
            y2 = min(mask.shape[0], py + margin_px)
            
            projected_zones.append((x1, y1, x2-x1, y2-y1))
            
            # Search for best contour point
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            best_p = None
            best_dist = 99999
            
            for p_wrap in approx:
                p = p_wrap[0]
                if x1 <= p[0] <= x2 and y1 <= p[1] <= y2:
                    dist = np.linalg.norm(np.array([px, py]) - p)
                    if dist < best_dist:
                        best_dist = dist
                        best_p = p
            
            if best_p is not None:
                found_points.append(best_p)
            else:
                return None, projected_zones
        
        if len(found_points) == 4:
            return np.array(found_points, dtype="double"), projected_zones
        return None, projected_zones

    def process_frame(self, frame):
        if self.camera_matrix is None: self.init_camera_matrix(frame.shape[1], frame.shape[0])
        self.debug_search_zones = []
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = self.clahe.apply(v)
        hsv_enhanced = cv2.merge([h, s, v])

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lower, upper) in self.hsv_ranges:
            mask += cv2.inRange(hsv_enhanced, lower, upper)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        found, final_angle, final_dist, final_offset_x, confidence, cam_height = False, 0.0, 0.0, 0.0, 0.0, 0.0
        brick_above = False
        brick_below = False
        
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_AREA_THRESHOLD: continue
            
            x, y, w, h_box = cv2.boundingRect(cnt)
            roi_h = int(h_box * 0.35)
            roi_y = y + h_box - roi_h
            if roi_y > 0 and roi_h > 0:
                roi_v = v[roi_y:roi_y+roi_h, x:x+w]
                _, roi_bright_mask = cv2.threshold(roi_v, SHADOW_CUT_THRESHOLD, 255, cv2.THRESH_BINARY)
                current_roi_mask = mask[roi_y:roi_y+roi_h, x:x+w]
                cleaned_roi = cv2.bitwise_and(current_roi_mask, roi_bright_mask)
                mask[roi_y:roi_y+roi_h, x:x+w] = cleaned_roi

        clean_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        def estimate_blob_metrics(rect):
            (cx, cy), (w_rect, h_rect), angle = rect
            if w_rect < h_rect:
                w_estimate = h_rect
                final_angle = angle + 90
            else:
                w_estimate = w_rect
                final_angle = angle

            if final_angle > 90:
                final_angle -= 180
            if final_angle < -90:
                final_angle += 180

            focal = self.camera_matrix[0,0]
            if w_estimate <= 0:
                return final_angle, 0.0, 0.0, 0.0

            final_dist = (64.0 * focal) / w_estimate
            img_w = frame.shape[1]
            cal_cx = (img_w / 2.0) + self.camera_center_offset_px
            final_offset_x = (cx - cal_cx) * (final_dist / focal)
            cam_height = 0.0
            return final_angle, final_dist, final_offset_x, cam_height

        candidates = []
        for cnt in clean_contours:
            if cv2.contourArea(cnt) < MIN_AREA_THRESHOLD:
                continue

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            bbox = cv2.boundingRect(cnt)

            image_points, _ = self.get_notch_from_contour(mask, cnt)
            notch_conf, blob_conf, reasons = self.calculate_fingerprint_confidence(cnt, image_points)

            is_notch_lock = (image_points is not None and notch_conf >= NOTCH_CONFIDENCE_THRESHOLD)
            is_blob_lock = (not is_notch_lock and blob_conf >= BLOB_CONFIDENCE_THRESHOLD)
            if not (is_notch_lock or is_blob_lock):
                continue

            (cx, cy) = rect[0]
            candidate_conf = notch_conf if is_notch_lock else blob_conf
            candidate_angle = 0.0
            candidate_dist = 0.0
            candidate_offset_x = 0.0
            candidate_cam_h = 0.0

            if is_notch_lock:
                success, rvec, tvec = cv2.solvePnP(
                    self.object_points, image_points, self.camera_matrix,
                    self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE
                )
                if success:
                    candidate_dist = np.linalg.norm(tvec)
                    R, _ = cv2.Rodrigues(rvec)
                    brick_x_in_cam = R[:, 0]
                    candidate_angle = -math.degrees(math.atan2(brick_x_in_cam[2], brick_x_in_cam[0]))

                    cam_pos_world = -np.matrix(R).T @ np.matrix(tvec)
                    candidate_cam_h = float(cam_pos_world[1, 0])

                    focal = self.camera_matrix[0,0]
                    mm_offset_fix = (self.camera_center_offset_px * candidate_dist) / focal
                    candidate_offset_x = tvec[0][0] - mm_offset_fix
                else:
                    candidate_angle, candidate_dist, candidate_offset_x, candidate_cam_h = estimate_blob_metrics(rect)
            else:
                candidate_angle, candidate_dist, candidate_offset_x, candidate_cam_h = estimate_blob_metrics(rect)

            candidates.append({
                "center": (float(cx), float(cy)),
                "angle": candidate_angle,
                "dist": candidate_dist,
                "offset_x": candidate_offset_x,
                "confidence": float(candidate_conf),
                "cam_height": candidate_cam_h,
                "contour": cnt,
                "bbox": bbox,
                "box": box,
                "is_notch_lock": is_notch_lock,
                "image_points": image_points,
            })

        if candidates:
            img_w = frame.shape[1]
            img_h = frame.shape[0]
            center_x = (img_w / 2.0) + self.camera_center_offset_px
            center_y = img_h / 2.0
            selected = min(
                candidates,
                key=lambda c: (c["center"][0] - center_x) ** 2 + (c["center"][1] - center_y) ** 2
            )
            sel_y = selected["center"][1]
            brick_above = any(c is not selected and c["center"][1] < sel_y for c in candidates)
            brick_below = any(c is not selected and c["center"][1] > sel_y for c in candidates)
            if not brick_above and not brick_below:
                notch_y = self.detect_notch_row(v, selected["contour"])
                if notch_y is not None:
                    center_y = selected["center"][1]
                    tol = selected["bbox"][3] * 0.08
                    if abs(notch_y - center_y) <= tol:
                        brick_above = True
                        brick_below = True
                    elif notch_y < center_y:
                        brick_above = True
                    else:
                        brick_below = True

            if not self.speed_optimize:
                palette = [
                    (0, 255, 0),
                    (0, 180, 255),
                    (255, 0, 255),
                    (255, 128, 0),
                    (0, 255, 128),
                ]
                for idx, cand in enumerate(candidates):
                    color = palette[idx % len(palette)]
                    thickness = 2
                    label = "NOTCH LOCK" if cand["is_notch_lock"] else "BLOB LOCK"
                    if cand is selected:
                        color = (0, 255, 255)
                        thickness = 4
                        label = f"{label} (SELECTED)"
                    bx, by, bw, bh = cand["bbox"]
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, thickness)
                    self.draw_text_with_bg(
                        frame,
                        f"{label} {int(cand['confidence'])}%",
                        (bx, by - 10),
                        text_color=color,
                    )
                    if cand["image_points"] is not None:
                        for p in cand["image_points"]:
                            cv2.circle(frame, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)

            found = True
            confidence = selected["confidence"]
            final_angle = selected["angle"]
            final_dist = selected["dist"]
            final_offset_x = selected["offset_x"]
            cam_height = selected["cam_height"]
            brick_above, brick_below = self.stabilize_stack_state(brick_above, brick_below, found, confidence)

        overlay_w, overlay_h = 160, 120
        mask_small = cv2.resize(mask, (overlay_w, overlay_h))
        mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(mask_color, (0,0), (overlay_w-1, overlay_h-1), (0,255,0), 1)
        frame[0:overlay_h, frame.shape[1]-overlay_w:frame.shape[1]] = mask_color

        return found, final_angle, final_dist, final_offset_x, frame, confidence, cam_height, brick_above, brick_below

    def read(self):
        ret, frame = self.cap.read()
        if not ret: 
            # SIGNAL HARDWARE FAILURE with dist = -1
            return False, 0, -1, 0, 0, 0, False, False
        
        # Store RAW frame for ML training (clean, no text)
        self.raw_frame = frame.copy()
        
        found, final_angle, final_dist, final_offset_x, display_frame, confidence, cam_height, brick_above, brick_below = self.process_frame(frame)
        self.current_frame = display_frame
        
        # SAVE SCREENSHOT IF ENABLED
        if self.save_folder is not None:
             timestamp = int(time.time() * 1000)
             filename = os.path.join(self.save_folder, f"frame_{timestamp}.jpg")
             cv2.imwrite(filename, display_frame)
        
        if self.debug and not self.headless and not self.speed_optimize:
            cv2.imshow("Brick Vision Debug", display_frame)
            cv2.waitKey(1)
        # Noise Filter: only consider a brick 'found' if confidence is > 25%
        # This prevents 'FIND' from skipping on noise/hands.
        reliable_found = found and confidence > 25
        if not reliable_found:
            brick_above = False
            brick_below = False

        return reliable_found, final_angle, final_dist, final_offset_x, confidence, cam_height, brick_above, brick_below

    def save_frame(self, filename):
        if self.current_frame is not None:
            cv2.imwrite(filename, self.current_frame)
            return True
        return False

    def close(self):
        self.cap.release()
        if not self.headless: cv2.destroyAllWindows()

if __name__ == "__main__":
    det = BrickDetector(save_folder=None)
    print("Running standalone test (Ctrl+C to stop)...")
    try:
        while True:
            det.read()
    except KeyboardInterrupt:
        det.close()
