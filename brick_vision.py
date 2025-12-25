"""
Brick Vision V45 - "Range Finder"
1. MATH: Calculates Distance in mm using the PnP translation vector.
2. UI: Adds "Dist: X mm" to the top-right Info Box.
3. CORE: Retains V44 Reprojection Error for confidence.
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

camera_matrix = None
dist_coeffs = np.zeros((4,1))

def load_world_model():
    if not WORLD_MODEL_FILE.exists():
        sys.exit(f"Error: {WORLD_MODEL_FILE} not found.")
    with open(WORLD_MODEL_FILE, 'r') as f:
        return json.load(f)

def init_camera_matrix(w, h):
    global camera_matrix
    # Approximation: Standard webcams have a focal length roughly equal to width
    focal_length = w 
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

def get_notch_2d_points(contour):
    x, y, w, h = cv2.boundingRect(contour)
    cutoff_y = y + (h * 0.60) 
    
    points = [pt[0] for pt in contour]
    bottom_points = [pt for pt in points if pt[1] > cutoff_y]
    bottom_points.sort(key=lambda p: p[0])
    
    if len(bottom_points) < 4: return None
    candidates = bottom_points[1:-1]
    if len(candidates) < 2: return None

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
    
    p1 = raw_p1 
    p4 = raw_p4 
    p2 = [raw_p1[0], raw_p2[1]] 
    p3 = [raw_p4[0], raw_p3[1]] 
    
    notch_width = p4[0] - p1[0]
    if notch_width < 10: return None

    return np.array([p1, p2, p3, p4], dtype="double")

def calculate_yaw(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    y = math.atan2(-R[2,0], sy)
    return math.degrees(y)

def calculate_confidence(object_points, image_points, rvec, tvec):
    projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected_points = projected_points.reshape(-1, 2)
    error = cv2.norm(image_points, projected_points, cv2.NORM_L2) / len(image_points)
    confidence = max(0, 100 - (error * 5))
    return int(confidence)

def draw_3d_axes(img, rvec, tvec):
    len_mm = 25.0
    axis_pts = np.float32([[0,0,0], [len_mm,0,0], [0,-len_mm,0], [0,0,-len_mm]]).reshape(-1,3)
    imgpts, _ = cv2.projectPoints(axis_pts, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.astype(int)
    
    origin = tuple(imgpts[0].ravel())
    cv2.line(img, origin, tuple(imgpts[1].ravel()), (0,0,255), 3) # X Red
    cv2.line(img, origin, tuple(imgpts[2].ravel()), (0,255,0), 3) # Y Green
    cv2.line(img, origin, tuple(imgpts[3].ravel()), (255,0,0), 3) # Z Blue

def main():
    model = load_world_model()
    
    hsv_settings = model['brick'].get('hsv_thresholds', {'sat_max': 52, 'val_min': 168})
    
    pts_map = {p['label']: [p['x'], p['y'], p['z']] for p in model['brick']['notch']['points_3d']}
    object_points = np.array([
        pts_map['bottom_left'], pts_map['top_left'],
        pts_map['top_right'], pts_map['bottom_right']
    ], dtype="double")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cv2.namedWindow("Master V45")
    
    cv2.createTrackbar("Sat Max", "Master V45", hsv_settings['sat_max'], 255, lambda x: None)
    cv2.createTrackbar("Val Min", "Master V45", hsv_settings['val_min'], 255, lambda x: None)

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if camera_matrix is None:
            h, w = frame.shape[:2]
            init_camera_matrix(w, h)
            
        display = frame.copy()
        
        s_max = cv2.getTrackbarPos("Sat Max", "Master V45")
        v_min = cv2.getTrackbarPos("Val Min", "Master V45")
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, v_min]), np.array([180, s_max, 255]))
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

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

        final_yaw = 0.0
        final_conf = 0
        final_dist = 0.0
        is_tracking = False

        if live_cnt is not None:
            cv2.drawContours(display, [live_cnt], -1, (0, 255, 0), 2)
            
            image_points = get_notch_2d_points(live_cnt)
            
            if image_points is not None:
                is_tracking = True
                
                success, rvec, tvec = cv2.solvePnP(
                    object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    final_yaw = calculate_yaw(rvec)
                    final_conf = calculate_confidence(object_points, image_points, rvec, tvec)
                    
                    # --- NEW: DISTANCE CALCULATION ---
                    # tvec is [x, y, z] in mm. Norm gives straight line distance.
                    final_dist = np.linalg.norm(tvec)

                    color = (0, 255, 0)
                    if final_conf < 70: color = (0, 255, 255)
                    if final_conf < 40: color = (0, 0, 255)

                    for pt in image_points:
                        cv2.circle(display, (int(pt[0]), int(pt[1])), 5, color, -1)
                    
                    draw_3d_axes(display, rvec, tvec)

        # --- DRAW INFO BOX ---
        # Increased height to 135 to fit Distance
        box_w, box_h = 240, 135
        h, w = display.shape[:2]
        overlay = display[0:box_h, w-box_w:w]
        
        cv2.rectangle(overlay, (0,0), (box_w, box_h), (30,30,30), -1)
        
        cv2.putText(overlay, "BRICK TRACKER", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.line(overlay, (10, 32), (box_w-10, 32), (100,100,100), 1)

        if is_tracking:
            # Yaw
            cv2.putText(overlay, f"Yaw : {final_yaw:.1f} deg", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            
            # Distance (NEW)
            cv2.putText(overlay, f"Dist: {int(final_dist)} mm", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            
            # Confidence Bar
            bar_x = 10
            bar_y = 100
            bar_w = 150
            bar_h = 10
            
            conf_color = (0, 255, 0)
            if final_conf < 70: conf_color = (0, 255, 255)
            if final_conf < 40: conf_color = (0, 0, 255)

            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50,50,50), -1)
            fill_w = int(bar_w * (final_conf / 100.0))
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), conf_color, -1)
            
            cv2.putText(overlay, f"{final_conf}%", (bar_x + bar_w + 10, bar_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        else:
            cv2.putText(overlay, "NO NOTCH DETECTED", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        display[0:box_h, w-box_w:w] = overlay

        thumb = cv2.resize(mask, (200, 150))
        h, w = display.shape[:2]
        display[h-150:h, 0:200] = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(display, (0, h-150), (200, h), (255,255,0), 1)

        cv2.imshow("Master V45", display)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()