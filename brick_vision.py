"""
Brick Vision V27 - "The Perfect Notch"
1. Extracts raw notch points from the bottom of the Green Outline.
2. Analyzes height changes to find the "Left Wall" (Up) and "Right Wall" (Down).
3. Selects exactly 4 Corner Points.
4. "Snaps" the vertical walls to be perfectly straight for a clean UI.
"""
import cv2
import numpy as np
import json
import sys
from pathlib import Path

# --- CONFIG ---
CAMERA_INDEX = 0
SUBSET_DIR = Path(__file__).parent / "photos" / "angled_bricks_subset"
DB_FILE = Path(__file__).parent / "brick_database.json"

reference_data = []

def draw_bar(img, label, val_percent, x, y, color=(0, 255, 0)):
    w = 200
    h = 20
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), -1)
    fill_w = int(w * (val_percent / 100.0))
    cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1)
    cv2.putText(img, f"{label}: {int(val_percent)}%", (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

def load_references():
    global reference_data
    if not DB_FILE.exists():
        sys.exit(f"Error: {DB_FILE} not found.")
        
    with open(DB_FILE, 'r') as f:
        db = json.load(f)
        
    for filename, data in db.items():
        img_path = SUBSET_DIR / filename
        if not img_path.exists(): continue
        
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            best_cnt = max(contours, key=cv2.contourArea)
            reference_data.append({
                'angle': data['angle'],
                'contour': best_cnt,
            })
    print(f"Loaded {len(reference_data)} references.")

def get_four_notch_corners(contour):
    """
    Analyzes the contour to find exactly 4 points representing the notch.
    Returns list of 4 points: [BottomLeft, TopLeft, TopRight, BottomRight]
    """
    x, y, w, h = cv2.boundingRect(contour)
    cutoff_y = y + (h * 0.60) # Only look at bottom 40%
    
    # Extract raw points
    points = [pt[0] for pt in contour]
    # Filter for bottom section
    bottom_points = [pt for pt in points if pt[1] > cutoff_y]
    # Sort Left to Right
    bottom_points.sort(key=lambda p: p[0])
    
    # Need enough points to find jumps
    if len(bottom_points) < 4:
        return []

    # Trim outer edges (feet)
    notch_candidates = bottom_points[1:-1]
    
    if len(notch_candidates) < 2: return []

    # --- Find the Vertical Walls ---
    # We look for the biggest change in Y between consecutive points
    
    best_up_idx = -1
    max_up_jump = 0
    
    best_down_idx = -1
    max_down_jump = 0
    
    # 1. Find Left Wall (Jump UP -> Y decreases)
    # We only search the first half of the points to avoid confusion
    mid_index = len(notch_candidates) // 2
    
    for i in range(mid_index):
        # Current point Y - Next point Y
        jump = notch_candidates[i][1] - notch_candidates[i+1][1] 
        if jump > max_up_jump:
            max_up_jump = jump
            best_up_idx = i
            
    # 2. Find Right Wall (Jump DOWN -> Y increases)
    # We search the second half
    for i in range(mid_index, len(notch_candidates) - 1):
        # Next point Y - Current point Y
        jump = notch_candidates[i+1][1] - notch_candidates[i][1]
        if jump > max_down_jump:
            max_down_jump = jump
            best_down_idx = i
            
    if best_up_idx == -1 or best_down_idx == -1:
        return []
        
    # Extract the raw 4 corners
    p1 = notch_candidates[best_up_idx]      # Bottom Left
    p2 = notch_candidates[best_up_idx + 1]  # Top Left
    p3 = notch_candidates[best_down_idx]    # Top Right
    p4 = notch_candidates[best_down_idx + 1]# Bottom Right
    
    # --- Force Vertical Alignment (The "Snap") ---
    # Average the X for the left wall
    avg_x_left = int((p1[0] + p2[0]) / 2)
    p1 = (avg_x_left, p1[1])
    p2 = (avg_x_left, p2[1])
    
    # Average the X for the right wall
    avg_x_right = int((p3[0] + p4[0]) / 2)
    p3 = (avg_x_right, p3[1])
    p4 = (avg_x_right, p4[1])
    
    return [p1, p2, p3, p4]

def main():
    load_references()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cv2.namedWindow("Master V27")
    
    cv2.createTrackbar("Sat Max", "Master V27", 52, 255, lambda x: None)
    cv2.createTrackbar("Val Min", "Master V27", 168, 255, lambda x: None)

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        display = frame.copy()
        h, w = display.shape[:2]
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        s_max = cv2.getTrackbarPos("Sat Max", "Master V27")
        v_min = cv2.getTrackbarPos("Val Min", "Master V27")
        
        mask = cv2.inRange(hsv, np.array([0, 0, v_min]), np.array([180, s_max, 255]))
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        live_cnt = None
        max_area = 0
        brick_conf = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 3000:
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if area > max_area:
                    max_area = area
                    live_cnt = approx

        if max_area > 0:
            ideal_area = (h * w) * 0.25 
            ratio = min(max_area, ideal_area) / max(max_area, ideal_area)
            brick_conf = min(100, int(ratio * 100) + 40)
        
        draw_bar(display, "IS BRICK?", brick_conf, 20, 30, (0, 255, 0) if brick_conf > 60 else (0, 165, 255))

        if live_cnt is not None:
            cv2.drawContours(display, [live_cnt], -1, (0, 255, 0), 2)
            
            # --- NOTCH 4 CORNERS ---
            corners = get_four_notch_corners(live_cnt)
            
            if len(corners) == 4:
                p1, p2, p3, p4 = corners
                
                # Draw Lines (Red)
                cv2.line(display, p1, p2, (0, 0, 255), 3) # Left Wall
                cv2.line(display, p2, p3, (0, 0, 255), 3) # Roof
                cv2.line(display, p3, p4, (0, 0, 255), 3) # Right Wall
                
                # Draw Dots (Yellow, no text)
                for pt in corners:
                    cv2.circle(display, pt, 6, (0, 255, 255), -1)

            # --- ANGLE MATCHING ---
            best_match = None
            best_score = 100.0
            for ref in reference_data:
                score = cv2.matchShapes(live_cnt, ref['contour'], 1, 0.0)
                if score < best_score:
                    best_score = score
                    best_match = ref
            
            match_conf = max(0, (1.0 - (best_score * 3.0))) * 100
            draw_bar(display, "ANGLE CONFIDENCE", match_conf, 20, 80, (0, 255, 255))
            
            if best_match and match_conf > 40:
                bx, by, bw, bh = cv2.boundingRect(live_cnt)
                label = f"{best_match['angle']} deg"
                cv2.putText(display, label, (bx, by - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        thumb = cv2.resize(mask, (200, 150))
        display[h-150:h, w-200:w] = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(display, (w-200, h-150), (w, h), (255,255,0), 1)

        cv2.imshow("Master V27", display)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()