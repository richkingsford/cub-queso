import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class BrickPose:
    found: bool
    position: Tuple[float, float, float]  # x, y, z (mm)
    orientation: Tuple[float, float, float]  # roll, pitch, yaw (degrees)
    confidence: float # 0.0 to 1.0
    metadata: str

class LeiaVision:
    def __init__(self, debug=True):
        self.debug = debug
        self.camera_matrix = None
        self.dist_coeffs = None
        self.debug_frame = None
        
        # --- PHYSICAL CONSTANTS (mm) ---
        self.BRICK_W = 64.0
        self.BRICK_H = 48.0
        self.BRICK_D = 20.0
        
        # 3D Model Points (Center at 0,0,0) - Flat Face
        # 12-point face model matching world_model_brick.json
        # Y=48 is top (Notch), Y=0 is bottom
        self.face_points_3d = np.array([
            [-32, 0, 0], [-32, 38, 0], [-12, 38, 0], [-12, 48, 0], 
            [12, 48, 0], [12, 38, 0], [32, 38, 0], [32, 0, 0], 
            [16, 0, 0], [16, 8, 0], [-16, 8, 0], [-16, 0, 0]
        ], dtype="double")
        
        # TL, TR, BR, BL for PnP initialization (Match order_points output)
        self.box_points_3d = np.array([
            [-32, 48, 0], [ 32, 48, 0], [ 32, 0, 0], [-32, 0, 0]
        ], dtype="double")

        # --- TUNING CONSTANTS ---
        self.TARGET_ASPECT = self.BRICK_W / self.BRICK_H # 1.333
        
        # Create Reference Polygon for Shape Matching (Used for fast filtering)
        self.ref_poly = self.face_points_3d[:, :2].astype(np.int32)

        # --- STATE / SMOOTHING ---
        self.smooth_conf = 0.0
        self.last_pose = None
        self.ALPHA = 0.15 # Balanced smoothing for stability vs speed
        self.frames_since_last_pos = 0

    def init_camera(self, width=640, height=480):
        focal_length = width 
        center = (width/2, height/2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        self.dist_coeffs = np.zeros((4,1))

    def process(self, frame: np.ndarray) -> BrickPose:
        # 1. Preprocess - EXTREME WASH
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 9x9 Median Blur to wash away fine table texture/noise
        washed = cv2.medianBlur(gray, 9)
        # Gentle Gaussian to smooth remaining edges
        blurred = cv2.GaussianBlur(washed, (5, 5), 0)
        
        # 2. Structural Edge Detection
        # Larger block size (25) ignores local texture in favor of global structure
        edges = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            25, 4
        )
        
        # 3. Morphology Clustering (Merging dashed lines)
        kernel = np.ones((3,3), np.uint8)
        # MORPH_CLOSE with multiple iterations or large kernel to merge segments
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Strong Dilation to bridge any remaining gaps
        edges = cv2.dilate(edges, kernel, iterations=2) 
        
        # Compute Distance Transform once for scoring
        # (How far is every black pixel from the nearest white pixel)
        # We invert it: 0 at white edges, increasing distance away.
        dist_map = cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, 3)

        if self.debug:
            self.debug_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # 4. Contour Search
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        
        for cnt in contours:
            perimeter = cv2.arcLength(cnt, True)
            area = cv2.contourArea(cnt)
            # Brick is physically large (~200px perimeter in typical views)
            if perimeter < 200 or area < 2000: continue 

            # Rotated Rect for Pose Initialization
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect
            if w == 0 or h == 0: continue
            aspect = max(w, h) / min(w, h)
            # Brick aspect is ~1.33. Broad filter to allow perspective tilt (up to 3.5)
            if aspect > 3.5: continue 
            
            # --- 3D VERIFICATION (Distance Transform) ---
            box = cv2.boxPoints(rect)
            sorted_box = self.order_points(np.int32(box))
            
            # Solve Initial Pose
            success, rvec, tvec = cv2.solvePnP(
                self.box_points_3d, sorted_box.astype("double"),
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                # Score the 3D projection against the DISTANCE MAP
                proj_score = self.score_projection_dist(rvec, tvec, dist_map)
                
                if self.debug:
                    # Draw Cyan with Projection Score
                    cv2.drawContours(self.debug_frame, [cnt], -1, (255, 255, 0), 1)
                    cv2.putText(self.debug_frame, f"3D:{proj_score:.2f}", (int(cx), int(cy)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                # Score threshold: higher is cleaner. 0.3-0.5 is usually good for noisy scenes.
                if proj_score > 0.4: 
                    candidates.append({
                        'rvec': rvec,
                        'tvec': tvec,
                        'conf': proj_score,
                        'rect': rect
                    })

        # 5. Resolve Best
        current_pose = None
        if candidates:
            # Multi-sort: Prefer high confidence but also large area (size matters)
            candidates.sort(key=lambda x: x['conf'], reverse=True)
            best = candidates[0]
            
            # Draw Final Verification in Green
            if self.debug:
                self.draw_wireframe(best['rvec'], best['tvec'], (0, 255, 0))
            
            rmat, _ = cv2.Rodrigues(best['rvec'])
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            
            current_pose = BrickPose(
                found=True,
                position=(best['tvec'][0][0], best['tvec'][1][0], best['tvec'][2][0]),
                orientation=angles,
                confidence=best['conf'],
                metadata="Extreme_Wash_3D"
            )
        
        return self.smooth_output(current_pose)

    def score_projection_dist(self, rvec, tvec, dist_map) -> float:
        # Project the 12-point face model
        img_pts, _ = cv2.projectPoints(self.face_points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        img_pts = img_pts.reshape(-1, 2).astype(np.int32)
        
        # Measure error: average distance from projected points to nearest white edge
        total_error = 0
        samples = 0
        
        h, w = dist_map.shape
        for i in range(len(img_pts)):
            # Line segments
            p1 = img_pts[i]
            p2 = img_pts[(i+1)%len(img_pts)]
            
            # Sample 10 points along each segment
            for t in np.linspace(0, 1, 10):
                px = int(p1[0]*(1-t) + p2[0]*t)
                py = int(p1[1]*(1-t) + p2[1]*t)
                
                if 0 <= px < w and 0 <= py < h:
                    err = dist_map[py, px]
                    # Penalty increases quadratically with distance? 
                    # Simple linear for now: error of 0 is perfect, 10px is bad
                    total_error += err
                    samples += 1
        
        if samples == 0: return 0.0
        
        avg_dist = total_error / samples
        # Score = 1.0 at 0px distance, 0.0 at 10px distance
        score = max(0, 1.0 - (avg_dist / 10.0))
        return float(score)

    def draw_wireframe(self, rvec, tvec, color):
        img_pts, _ = cv2.projectPoints(self.face_points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        img_pts = img_pts.reshape(-1, 2).astype(np.int32)
        cv2.polylines(self.debug_frame, [img_pts], True, color, 2)
        cv2.drawFrameAxes(self.debug_frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 30.0)

    def smooth_output(self, pose: Optional[BrickPose]) -> BrickPose:
        if pose and pose.found:
            self.frames_since_last_pos = 0
            if self.last_pose is None:
                self.last_pose = pose
                self.smooth_conf = pose.confidence
            else:
                # Heavy EMA
                self.smooth_conf = (self.ALPHA * pose.confidence) + ((1-self.ALPHA) * self.smooth_conf)
                sp = []
                for i in range(3):
                    val = (self.ALPHA * pose.position[i]) + ((1-self.ALPHA) * self.last_pose.position[i])
                    sp.append(val)
                pose.position = tuple(sp)
                pose.confidence = self.smooth_conf
                self.last_pose = pose
            return pose
        else:
            self.frames_since_last_pos += 1
            if self.frames_since_last_pos > 5: # Hold for 5 frames
                self.last_pose = None
                return BrickPose(False, (0,0,0), (0,0,0), 0.0, "unknown")
            else:
                if self.last_pose:
                    p = self.last_pose
                    p.confidence *= 0.7 # Decay
                    return p
                return BrickPose(False, (0,0,0), (0,0,0), 0.0, "unknown")

    def order_points(self, pts):
        # pts expect [TL, TR, BR, BL] roughly
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # TL (min sum)
        rect[2] = pts[np.argmax(s)] # BR (max sum)
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # TR (min diff)
        rect[3] = pts[np.argmax(diff)] # BL (max diff)
        return rect
