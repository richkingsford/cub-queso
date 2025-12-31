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
from pathlib import Path

# --- CONFIG ---
CAMERA_INDEX = 0
WORLD_MODEL_FILE = Path(__file__).parent / "world_model.json"
MIN_AREA_THRESHOLD = 1000 
SHADOW_CUT_THRESHOLD = 80
CONFIDENCE_THRESHOLD = 60.0 # Minimum score (0-100) to accept a lock

class BrickDetector:
    def __init__(self, debug=True, save_folder=None, speed_optimize=False):
        self.debug = debug
        self.speed_optimize = speed_optimize
        self.headless = True 
        self.save_folder = save_folder
        self.current_frame = None
        self.raw_frame = None
        
        if self.save_folder and not os.path.exists(self.save_folder):
            try: os.makedirs(self.save_folder)
            except: pass

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) 
        
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4,1))
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.hsv_ranges = []
        # Default fallback
        # Default fallback
        self.search_margin_mm = 6.0
        self.notch_points_3d = []
        self.object_points, self.template_contour = self.load_world_model()
        
        self.debug_search_zones = []

        if self.debug and not self.speed_optimize:
            try: cv2.namedWindow("Brick Vision Debug")
            except: self.headless = True

    def draw_text_with_bg(self, img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, text_color=(255, 255, 255), thickness=1, bg_color=(0, 0, 0)):
        x, y = pos
        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(img, (x, y - h - 5), (x + w, y + 5), bg_color, -1)
        cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

    def hex_to_hsv_ranges(self, hex_str, h_margin, s_margin, v_margin):
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
        if WORLD_MODEL_FILE.exists():
            with open(WORLD_MODEL_FILE, 'r') as f:
                model = json.load(f)
                c = model['brick']['color']
                self.hsv_ranges = self.hex_to_hsv_ranges(
                    c.get('hex', '#AE363E'), c.get('hue_margin', 15),
                    c.get('sat_margin', 100), c.get('val_margin', 100)
                )
                
                # Load Search Margin
                self.search_margin_mm = model['brick']['notch'].get('search_margin_mm', 6.0)
                
                notch_pts = model['brick']['notch']['points_3d']
                self.notch_points_3d = notch_pts # Store for 2D projection logic
                p_map = {p['label']: [p['x'], p['y'], p['z']] for p in notch_pts}
                pts_3d = np.array([
                    p_map['bottom_left'], p_map['top_left'],
                    p_map['top_right'], p_map['bottom_right']
                ], dtype="double")
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
        Calculates a confidence score (0-100) based on how well the 
        detected geometry matches a standard brick profile.
        """
        score = 100.0
        reasons = []

        # 1. Bounding Box Analysis
        x, y, w, h = cv2.boundingRect(contour)
        
        # Get Notch Dimensions from the 2D points
        # Points: [BL, TL, TR, BR]
        notch_w = np.linalg.norm(notch_points[3] - notch_points[0])
        notch_h = np.linalg.norm(notch_points[1] - notch_points[0]) # Avg side height
        notch_center_x = (notch_points[0][0] + notch_points[3][0]) / 2.0
        notch_bottom_y = (notch_points[0][1] + notch_points[3][1]) / 2.0

        # --- CHECK A: WIDTH RATIO ---
        # A standard brick notch is usually 35-50% of the total brick width
        width_ratio = notch_w / float(w)
        expected_width_ratio = 0.45 
        diff_w = abs(width_ratio - expected_width_ratio)
        if diff_w > 0.2: 
            penalty = (diff_w - 0.2) * 100
            score -= penalty
            reasons.append(f"Bad Width Ratio: {width_ratio:.2f}")

        # --- CHECK B: HEIGHT RATIO (Critical for Hand rejection) ---
        # The notch is usually short compared to the full brick height (maybe 25-30%)
        # If the contour includes a hand, H will be huge, so ratio will be tiny (< 0.1)
        height_ratio = notch_h / float(h)
        expected_height_ratio = 0.25
        # We penalize heavily if the ratio is too small (contour too tall)
        if height_ratio < 0.10: 
            score -= 60 # Massive penalty for "Super Tall" contours (Hand)
            reasons.append(f"Contour Too Tall (Hand?): {height_ratio:.2f}")
        elif height_ratio > 0.60:
            score -= 40
            reasons.append(f"Contour Too Short: {height_ratio:.2f}")

        # --- CHECK C: BOTTOM ALIGNMENT ---
        # The notch bottom should be very close to the contour bottom
        cnt_bottom = y + h
        dist_from_bottom = abs(cnt_bottom - notch_bottom_y)
        if dist_from_bottom > (h * 0.15): # If notch is "floating" high up
            score -= 40
            reasons.append(f"Notch Floating: {dist_from_bottom:.1f}px")

        # --- CHECK D: CENTER ALIGNMENT ---
        cnt_center_x = x + (w / 2.0)
        dist_center = abs(cnt_center_x - notch_center_x)
        if dist_center > (w * 0.2): # Notch should be roughly centered
            score -= 20
            reasons.append("Notch Off-Center")

        return max(0.0, score), reasons

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
        margin_px = int(self.search_margin_mm * mm_per_pixel)
        
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
            
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        found, final_angle, final_dist, final_offset_x, max_y = False, 0.0, 0.0, 0.0, 0.0
        
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
        
        for cnt in clean_contours:
            if cv2.contourArea(cnt) < MIN_AREA_THRESHOLD: continue
            
            image_points, search_zones = self.get_notch_from_contour(mask, cnt)
            
            if self.debug and not self.speed_optimize and search_zones:
                for (zx, zy, zw, zh) in search_zones:
                    cv2.rectangle(frame, (zx, zy), (zx+zw, zy+zh), (0, 255, 255), 1)

            if image_points is not None:
                # --- CALCULATE CONFIDENCE ---
                confidence, reasons = self.calculate_fingerprint_confidence(cnt, image_points)
                print(f"Conf: {int(confidence)}% | Reasons: {reasons}")

                # Draw Visual Feedback
                color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 0, 255)
                
                if not self.speed_optimize:
                    epsilon = 0.008 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    cv2.drawContours(frame, [approx], -1, color, 2)
                    
                    p_bl, p_tl, p_tr, p_br = image_points
                    # Draw Notch Box
                    cv2.line(frame, (int(p_bl[0]), int(p_bl[1])), (int(p_tl[0]), int(p_tl[1])), (0, 165, 255), 2)
                    cv2.line(frame, (int(p_tr[0]), int(p_tr[1])), (int(p_br[0]), int(p_br[1])), (0, 165, 255), 2)
                    cv2.line(frame, (int(p_tl[0]), int(p_tl[1])), (int(p_tr[0]), int(p_tr[1])), (0, 165, 255), 2)
                    cv2.line(frame, (int(p_bl[0]), int(p_bl[1])), (int(p_br[0]), int(p_br[1])), (0, 255, 255), 2) # Angle Line

                    # Draw Status ONLY if rejected
                    if confidence < CONFIDENCE_THRESHOLD:
                         self.draw_text_with_bg(frame, "REJECTED", (int(x), int(y)-10), font_scale=0.6, text_color=color)

                # ONLY RETURN DATA IF CONFIDENCE IS HIGH
                if confidence >= CONFIDENCE_THRESHOLD:
                    success, rvec, tvec = cv2.solvePnP(
                        self.object_points, image_points, self.camera_matrix, 
                        self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE
                    )
                    
                    if success:
                        found = True
                        final_dist = np.linalg.norm(tvec)
                        final_offset_x = tvec[0][0]
                        max_y = np.max(image_points[:, 1])
                        
                        # Simple 2D Angle
                        p_bl = image_points[0]
                        p_br = image_points[3]
                        delta_y = p_br[1] - p_bl[1]
                        delta_x = p_br[0] - p_bl[0]
                        final_angle = math.degrees(math.atan2(delta_y, delta_x))
                        
                        if not self.speed_optimize:
                            line1 = f"Conf: {int(confidence)}% | Dist: {int(final_dist)}mm"
                            line2 = f"Ang: {int(final_angle)}deg | Off: {int(final_offset_x)}mm"
                            self.draw_text_with_bg(frame, line1, (10, 20), font_scale=0.35, text_color=(0, 255, 0))
                            self.draw_text_with_bg(frame, line2, (10, 40), font_scale=0.35, text_color=(0, 255, 0))
                        
                        # LOG ALL METRICS
                        print(f"[VISION] LOCKED | Conf: {int(confidence)}% | Dist: {int(final_dist)}mm | Ang: {int(final_angle)}deg | Off: {int(final_offset_x)}mm")
                        break # Locked on target
                else:
                    # If we found a candidate but rejected it, we continue searching 
                    # other contours, or just finish this frame as "not found"
                    pass

        overlay_w, overlay_h = 160, 120
        mask_small = cv2.resize(mask, (overlay_w, overlay_h))
        mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(mask_color, (0,0), (overlay_w-1, overlay_h-1), (0,255,0), 1)
        frame[0:overlay_h, frame.shape[1]-overlay_w:frame.shape[1]] = mask_color

        return found, final_angle, final_dist, final_offset_x, max_y, frame

    def read(self):
        ret, frame = self.cap.read()
        if not ret: return False, 0, 0, 0
        
        # Store RAW frame for ML training (clean, no text)
        self.raw_frame = frame.copy()
        
        found, final_angle, final_dist, final_offset_x, max_y, display_frame = self.process_frame(frame)
        self.current_frame = display_frame
        
        # SAVE SCREENSHOT IF ENABLED
        if self.save_folder is not None:
             timestamp = int(time.time() * 1000)
             filename = os.path.join(self.save_folder, f"frame_{timestamp}.jpg")
             cv2.imwrite(filename, display_frame)

        if self.debug and not self.headless and not self.speed_optimize:
            cv2.imshow("Brick Vision Debug", display_frame)
            cv2.waitKey(1)
        return found, final_angle, final_dist, final_offset_x, max_y

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