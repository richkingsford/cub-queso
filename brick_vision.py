"""
Brick Vision V42 - "The Real World (JSON Model)"
1. DATA: Loads 'world_model.json' to get physical notch coordinates (points_3d).
2. VISION: Finds the 4 yellow notch dots on the video feed.
3. MATH: Uses solvePnP to calculate the brick's exact 3D rotation.
4. UI: Draws 3D Axes (Red/Green/Blue) on the live brick.
"""
import cv2
import numpy as np
import json
import sys
import math
from pathlib import Path

# --- CONFIG ---
CAMERA_INDEX = 0
WORLD_MODEL_FILE = Path(__file__).parent / "world_model.json"

# Camera Matrix (Will be approx. calibrated on first frame)
camera_matrix = None
dist_coeffs = np.zeros((4,1))

def load_world_model():
    if not WORLD_MODEL_FILE.exists():
        sys.exit(f"Error: {WORLD_MODEL_FILE} not found.")
    with open(WORLD_MODEL_FILE, 'r') as f:
        return json.load(f)

def init_camera_matrix(w, h):
    """
    Approximates camera optics so 3D math works.
    Focal length ~ Width is a standard webcam estimation.
    """
    global camera_matrix
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

def get_notch_2d_points(contour):
    """
    Finds the 4 notch corners in the 2D image (Pixels).
    Returns them in specific order: [BottomLeft, TopLeft, TopRight, BottomRight]
    to match the order of our 3D points in JSON.
    """
    x, y, w, h = cv2.boundingRect(contour)
    cutoff_y = y + (h * 0.60) 
    
    # Flatten contour
    points = [pt[0] for pt in contour]
    # Filter for bottom section
    bottom_points = [pt for pt in points if pt[1] > cutoff_y]
    bottom_points.sort(key=lambda p: p[0])
    
    if len(bottom_points) < 4: return None
    candidates = bottom_points[1:-1] # Trim feet
    if len(candidates) < 2: return None

    # Identify Notch Walls
    best_up_idx = -1; max_up = 0
    best_down_idx = -1; max_down = 0
    
    for i in range(len(candidates) - 1):
        step_up = candidates[i][1] - candidates[i+1][1]
        if step_up > max_up: max_up = step_up; best_up_idx = i
            
    start_search = best_up_idx + 1 if best_up_idx != -1 else 0
    for i in range(start_search, len(candidates) - 1):
        step_down = candidates[i+1][1] - candidates[i][1]
        if step_down > max_down: max_down = step_down; best_down_idx = i

    if max_up < h*0.05 or max_down < h*0.05: return None
        
    raw_p1 = candidates[best_up_idx]      
    raw_p2 = candidates[best_up_idx + 1]  
    raw_p3 = candidates[best_down_idx]    
    raw_p4 = candidates[best_down_idx + 1]
    
    # Snap vertical alignment for stability
    p1 = raw_p1 # BL
    p4 = raw_p4 # BR
    p2 = [raw_p1[0], raw_p2[1]] # TL (Snapped X to BL)
    p3 = [raw_p4[0], raw_p3[1]] # TR (Snapped X to BR)
    
    # Return as float array
    return np.array([p1, p2, p3, p4], dtype="double")

def calculate_yaw(rvec):
    """ Converts rotation vector to Yaw angle (Rotation around Y-axis) """
    R, _ = cv2.Rodrigues(rvec)
    # Standard Euler Angle decomposition
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    if sy < 1e-6:
        y = math.atan2(-R[2,0], sy)
    else:
        y = math.atan2(-R[2,0], sy)
    return math.degrees(y)

def draw_3d_axes(img, rvec, tvec, start_point):
    """ Draws X(Red)/Y(Green)/Z(Blue) axes starting from the notch """
    len_mm = 20.0
    axis_pts = np.float32([[0,0,0], [len_mm,0,0], [0,-len_mm,0], [0,0,-len_mm]]).reshape(-1,3)
    
    imgpts, _ = cv2.projectPoints(axis_pts, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.astype(int)
    
    origin = tuple(imgpts[0].ravel())
    img = cv2.line(img, origin, tuple(imgpts[1].ravel()), (0,0,255), 3) # X Red
    img = cv2.line(img, origin, tuple(imgpts[2].ravel()), (0,255,0), 3) # Y Green (Up)
    img = cv2.line(img, origin, tuple(imgpts[3].ravel()), (255,0,0), 3) # Z Blue (Depth)

def main():
    # 1. LOAD WORLD MODEL
    model = load_world_model()
    # Extract the 4 corner points [BL, TL, TR, BR]
    # Note: JSON order matters! Our code expects: BL, TL, TR, BR
    # Let's map them by label to be safe
    pts_map = {p['label']: [p['x'], p['y'], p['z']] for p in model['brick']['notch']['points_3d']}
    object_points = np.array([
        pts_map['bottom_left'],
        pts_map['top_left'],
        pts_map['top_right'],
        pts_map['bottom_right']
    ], dtype="double")
    
    print(f"Loaded 3D Model Points:\n{object_points}")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cv2.namedWindow("Master V42")
    
    cv2.createTrackbar("Sat Max", "Master V42", 52, 255, lambda x: None)
    cv2.createTrackbar("Val Min", "Master V42", 168, 255, lambda x: None)

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if camera_matrix is None:
            h, w = frame.shape[:2]
            init_camera_matrix(w, h)
            
        display = frame.copy()
        
        # HSV Filter
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        s_max = cv2.getTrackbarPos("Sat Max", "Master V42")
        v_min = cv2.getTrackbarPos("Val Min", "Master V42")
        mask = cv2.inRange(hsv, np.array([0, 0, v_min]), np.array([180, s_max, 255]))
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        live_cnt = None
        max_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 3000:
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if area > max_area:
                    max_area = area
                    live_cnt = approx

        if live_cnt is not None:
            cv2.drawContours(display, [live_cnt], -1, (0, 255, 0), 2)
            
            # 2. FIND 2D DOTS
            image_points = get_notch_2d_points(live_cnt)
            
            if image_points is not None:
                # Draw 2D Dots
                for pt in image_points:
                    cv2.circle(display, (int(pt[0]), int(pt[1])), 6, (0, 255, 255), -1)
                
                # 3. SOLVE 3D POSE
                success, rvec, tvec = cv2.solvePnP(
                    object_points, 
                    image_points, 
                    camera_matrix, 
                    dist_coeffs, 
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    # Draw Axes
                    draw_3d_axes(display, rvec, tvec, image_points[0])
                    
                    # Calculate Angle
                    yaw = calculate_yaw(rvec)
                    
                    # Show Result
                    label = f"3D YAW: {yaw:.1f} deg"
                    cv2.putText(display, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    
                    # Top Right Box - Model Info
                    box_w, box_h = 220, 100
                    h, w = display.shape[:2]
                    overlay = display[0:box_h, w-box_w:w]
                    cv2.rectangle(overlay, (0,0), (box_w, box_h), (50,50,50), -1)
                    cv2.putText(overlay, "World Model Active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
                    cv2.putText(overlay, "Method: PnP Solver", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                    display[0:box_h, w-box_w:w] = overlay

        # Robot View
        thumb = cv2.resize(mask, (200, 150))
        h, w = display.shape[:2]
        display[h-150:h, 0:200] = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(display, (0, h-150), (200, h), (255,255,0), 1)
        cv2.putText(display, "ROBOT VIEW", (5, h-135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        cv2.imshow("Master V42", display)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()