"""
Brick Vision V29 - "Anchored & Snapped"
1. STRICT SUBSET: Finds the 4 raw corners strictly on the Green Outline (inside the feet).
2. ANCHORING: Treats the bottom two points (P1, P4) as the "True Anchors".
3. SNAPPING: Forces the top two points (P2, P3) to align vertically with the anchors.
   - P2.x becomes P1.x
   - P3.x becomes P4.x
   - P2.y and P3.y remain "true" to the green outline height.
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

def get_anchored_notch(contour):
    """
    1. Find raw points on contour (Strict Subset).
    2. Identify P1 (BottomLeft) and P4 (BottomRight) as ANCHORS.
    3. Find P2 (TopLeft) and P3 (TopRight) from contour.
    4. Snap P2.x -> P1.x and P3.x -> P4.x to enforce verticality.
    """
    x, y, w, h = cv2.boundingRect(contour)
    cutoff_y = y + (h * 0.60) 
    
    # 1. Flatten and Filter
    points = [pt[0] for pt in contour]
    bottom_points = [pt for pt in points if pt[1] > cutoff_y]
    bottom_points.sort(key=lambda p: p[0])
    
    if len(bottom_points) < 4: return []

    # 2. Trim Feet (Rule: Horizontal Interior)
    candidates = bottom_points[1:-1]
    if len(candidates) < 2: return []

    # 3. Find Walls (Big Jumps)
    best_up_idx = -1
    max_up_step = 0
    
    search_limit = len(candidates) - 1
    for i in range(search_limit):
        step = candidates[i][1] - candidates[i+1][1] # Current Y - Next Y
        if step > max_up_step:
            max_up_step = step
            best_up_idx = i
            
    best_down_idx = -1
    max_down_step = 0
    start_search = best_up_idx + 1 if best_up_idx != -1 else 0
    
    for i in range(start_search, search_limit):
        step = candidates[i+1][1] - candidates[i][1] # Next Y - Current Y
        if step > max_down_step:
            max_down_step = step
            best_down_idx = i

    threshold = h * 0.05
    if max_up_step < threshold or max_down_step < threshold:
        return []
        
    # 4. Extract Raw Candidates
    raw_p1 = candidates[best_up_idx]      # Bottom Left
    raw_p2 = candidates[best_up_idx + 1]  # Top Left
    raw_p3 = candidates[best_down_idx]    # Top Right
    raw_p4 = candidates[best_down_idx + 1]# Bottom Right
    
    # 5. Apply Vertical Snapping (The "Notch Rule")
    # We trust the bottom points (p1, p4) as the anchors because they are on the ground.
    
    final_p1 = raw_p1
    final_p4 = raw_p4
    
    # Force P2 to be directly above P1
    final_p2 = [raw_p1[0], raw_p2[1]] 
    
    # Force P3 to be directly above P4
    final_p3 = [raw_p4[0], raw_p3[1]]
    
    # Convert to tuples for OpenCV
    return [tuple(final_p1), tuple(final_p2), tuple(final_p3), tuple(final_p4)]

def main():
    load_references()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cv2.namedWindow("Master V29")
    
    cv2.createTrackbar("Sat Max", "Master V29", 52, 255, lambda x: None)
    cv2.createTrackbar("Val Min", "Master V29", 168, 255, lambda x: None)

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        display = frame.copy()
        h, w = display.shape[:2]
        
        # HSV + Morphology
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        s_max = cv2.getTrackbarPos("Sat Max", "Master V29")
        v_min = cv2.getTrackbarPos("Val Min", "Master V29")
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
            
            # --- ANCHORED & SNAPPED NOTCH ---
            corners = get_anchored_notch(live_cnt)
            
            if len(corners) == 4:
                p1, p2, p3, p4 = corners
                
                # Draw Perfect Lines (Red)
                cv2.line(display, p1, p2, (0, 0, 255), 3) # Left Wall (Vertical)
                cv2.line(display, p2, p3, (0, 0, 255), 3) # Roof (Horizontal-ish)
                cv2.line(display, p3, p4, (0, 0, 255), 3) # Right Wall (Vertical)
                
                # Draw Dots (Yellow)
                for pt in corners:
                    cv2.circle(display, pt, 6, (0, 255, 255), -1)

            # Match Angle
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

        cv2.imshow("Master V29", display)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()