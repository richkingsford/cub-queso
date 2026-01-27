import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import deque

@dataclass
class BrickPose:
    found: bool
    position: Tuple[float, float, float]  # x, y, z (mm)
    orientation: Tuple[float, float, float]  # roll, pitch, yaw (degrees)
    confidence: float
    marker_id: int
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    corners: Optional[np.ndarray] = None # (4, 2) image points
    brickAbove: bool = False
    brickBelow: bool = False

class ArucoBrickVision:
    def __init__(self, marker_size_mm: float = 20.0, debug: bool = True):
        self.marker_size = marker_size_mm
        self.debug = debug
        self.camera_matrix = None
        self.dist_coeffs = None
        self.debug_frame = None
        self.current_frame = None
        self.raw_frame = None
        self.cap = None
        
        # Stacking stability
        self.stack_history = deque(maxlen=6)
        
        # ArUco Setup (Loosened for partial marker detection)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_params.minMarkerPerimeterRate = 0.005  # even smaller for clipped markers
        self.aruco_params.polygonalApproxAccuracyRate = 0.1
        self.aruco_params.minCornerDistanceRate = 0.05
        self.aruco_params.minSideLengthCanonicalImg = 4 # Detect tiny partials
        
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # 3D Marker Points (Marker center is origin of its 3D space)
        # Assuming marker is flat on the brick face (Z=0)
        s = marker_size_mm / 2.0
        self.marker_points_3d = np.array([
            [-s,  s, 0], # Top Left
            [ s,  s, 0], # Top Right
            [ s, -s, 0], # Bottom Right
            [-s, -s, 0]  # Bottom Left
        ], dtype=np.float32)

        # Brick Dimensions (mm)
        self.BRICK_H = 48.0
        
        # Smoothing
        self.ALPHA = 0.2
        self.last_pose = None

    def init_camera(self, width: int = 640, height: int = 480):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Approximate camera matrix if none provided
        focal_length = width
        center = (width / 2, height / 2)
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    def read(self):
        if self.cap is None:
            self.init_camera()

        # Flush a few frames to avoid stale buffer data
        for _ in range(2):
            self.cap.grab()
        ret, frame = self.cap.read()
        if not ret:
            return False, 0, -1, 0, 0, 0, False, False

        self.raw_frame = frame.copy()
        pose = self.process(frame)
        self.current_frame = self.debug_frame if self.debug_frame is not None else frame
        
        if not pose.found:
            return False, 0, 0, 0, 0, 0, False, False

        # found, angle, dist, offset_x, conf, cam_h, above, below
        return (
            True, 
            pose.orientation[1], # Yaw-ish
            pose.position[2],    # Z dist
            pose.position[0],    # X offset
            pose.confidence,
            pose.position[1],    # Y height (cam relative)
            pose.brickAbove,
            pose.brickBelow
        )

    def close(self):
        if self.cap:
            self.cap.release()

    def process(self, frame: np.ndarray) -> BrickPose:
        if self.camera_matrix is None:
            self.init_camera(frame.shape[1], frame.shape[0])
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        if self.debug:
            self.debug_frame = frame.copy()
            # Removed drawDetectedMarkers (dots)

        best_marker = None
        
        if ids is not None:
            # Find centermost marker
            min_dist = float('inf')
            img_center = (frame.shape[1] / 2, frame.shape[0] / 2)
            
            poses = []
            for i in range(len(ids)):
                marker_id = int(ids[i][0])
                c = corners[i][0]
                
                # Estimate pose
                success, rvec, tvec = cv2.solvePnP(
                    self.marker_points_3d, c, self.camera_matrix, self.dist_coeffs
                )
                
                if success:
                    # image space distance to center
                    mcx = np.mean(c[:, 0])
                    mcy = np.mean(c[:, 1])
                    dist = np.sqrt((mcx - img_center[0])**2 + (mcy - img_center[1])**2)
                    
                    rmat, _ = cv2.Rodrigues(rvec)
                    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                    
                    pose = BrickPose(
                        found=True,
                        position=(tvec[0][0], tvec[1][0], tvec[2][0]),
                        orientation=angles,
                        confidence=100.0,
                        marker_id=marker_id,
                        rvec=rvec,
                        tvec=tvec,
                        corners=c
                    )
                    poses.append(pose)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_marker = pose

            # Detect stacking
            if best_marker:
                # 1. Check verified markers
                for pose in poses:
                    if pose.marker_id == best_marker.marker_id: continue
                    dz = abs(pose.position[2] - best_marker.position[2])
                    dy = pose.position[1] - best_marker.position[1]
                    dx = abs(pose.position[0] - best_marker.position[0])
                    
                    if dx < 30 and dz < 40: # Loosened for reliability
                        if -55 < dy < -40: best_marker.brickAbove = True
                        if  40 < dy < 55:  best_marker.brickBelow = True

                # 2. Check rejected markers for partial bricks
                if rejected and (not best_marker.brickAbove or not best_marker.brickBelow):
                    focal = self.camera_matrix[0,0]
                    dist_mm = best_marker.position[2]
                    expected_shift_px = focal * (self.BRICK_H / dist_mm)
                    expected_size_px = focal * (self.marker_size / dist_mm)
                    
                    # Project center to image space
                    best_img_pts, _ = cv2.projectPoints(np.array([[0,0,0]], dtype=np.float32), 
                                                       best_marker.rvec, best_marker.tvec, 
                                                       self.camera_matrix, self.dist_coeffs)
                    bcx, bcy = best_img_pts[0][0]

                    for rej in rejected:
                        corners = rej[0]
                        mcx = np.mean(corners[:, 0])
                        # Bottom-most point of this candidate (highest Y value in pixels)
                        max_y = np.max(corners[:, 1])
                        min_y = np.min(corners[:, 1])
                        
                        size_px = np.linalg.norm(corners[0] - corners[2]) / 1.414 
                        
                        # For clipped markers, size will be smaller. 
                        # Allow tiny size if it's right at the edge.
                        is_at_edge = (min_y < 10 or max_y > frame.shape[0] - 10)
                        min_size_f = 0.2 if is_at_edge else 0.4
                        if not (expected_size_px * min_size_f < size_px < expected_size_px * 1.6): continue
                        
                        idx = abs(mcx - bcx)
                        # idy based on bottom edge vs center
                        # expected shift to bottom edge is roughly 38mm (48 - 10)
                        dy_bottom = bcy - max_y # Correct: if max_y is above bcy, dy is positive
                        # expected_shift_bottom = focal * (38 / dist_mm)
                        # But simpler: use center if size is okay, or just check alignment
                        
                        if idx < expected_shift_px * 0.5: # Loosened alignment
                            # Check "Above" (Positive idy)
                            # center-center shift is 48mm. 
                            # If clipped at top, max_y (bottom edge) is the most reliable anchor.
                            # expected max_y for upper brick: bcy - (focal * (38 / dist_mm))
                            idy_center = bcy - np.mean(corners[:, 1])
                            
                            # Shifted check: if center is visible, use center.
                            # If clipped, idy_center will be lower than expected center-center shift.
                            # But max_y should still be at least ~28mm above center.
                            dist_to_bottom = bcy - max_y
                            expected_dist_to_bottom = focal * (38.0 / dist_mm)
                            
                            if not best_marker.brickAbove and (expected_dist_to_bottom * 0.7 < dist_to_bottom < expected_dist_to_bottom * 1.3):
                                best_marker.brickAbove = True
                            
                            # Check "Below" (Negative dy)
                            dist_to_top = bcy - min_y # Negative if below
                            expected_dist_to_top = focal * (-38.0 / dist_mm)
                            
                            if not best_marker.brickBelow and (expected_dist_to_top * 1.3 < dist_to_top < expected_dist_to_top * 0.7):
                                best_marker.brickBelow = True

        if best_marker:
            if self.last_pose:
                # Smoothing
                p = []
                for i in range(3):
                    p.append(self.ALPHA * best_marker.position[i] + (1 - self.ALPHA) * self.last_pose.position[i])
                best_marker.position = tuple(p)
            self.last_pose = best_marker
            return best_marker
        else:
            return BrickPose(False, (0, 0, 0), (0, 0, 0), 0.0, -1)
