"""
Brick Vision V26 - "Trust The Green Line"
1. HSV Masking (Standard).
2. Green Outline (Standard).
3. Notch Logic: Strictly extracts points from the bottom edge of the Green Outline.
   - Filters points in the lower 40% of the bounding box.
   - Sorts them Left-to-Right.
   - Removes the first (Leftmost) and last (Rightmost).
   - Draws whatever remains as the Notch.
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

def extract_notch_points(contour):
    """
    1. Get Bounding Box.
    2. Keep only points in the bottom 40% of height.
    3. Sort by X.
    4. Remove First (Left Edge) and Last (Right Edge).
    5. Return the rest.
    """
    x, y, w, h = cv2.boundingRect(contour)
    cutoff_y = y + (h * 0.60) # Only look at bottom 40%
    
    # Extract points from contour (shape is N, 1, 2)
    points = [pt[0] for pt in contour]
    
    # Filter: Must be low enough (Y > cutoff)
    bottom_points = [pt for pt in points if pt[1] > cutoff_y]
    
    # Sort Left to Right
    bottom_points.sort(key=lambda p: p[0])
    
    # We need at least 3 points to trim edges and have something left
    if len(bottom_points) < 3:
        return []
        
    # Trim the Leftmost and Rightmost (The "Feet")
    notch_points = bottom_points[1:-1]
    
    return notch_points

def main():
    load_references()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cv2.namedWindow("Master V26")
    
    # Settings
    cv2.createTrackbar("Sat Max", "Master V26", 52, 255, lambda x: None)
    cv2.createTrackbar("Val Min", "Master V26", 168, 255, lambda x: None)

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        display = frame.copy()
        h, w = display.shape[:2]
        
        # 1. HSV FILTER
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        s_max = cv2.getTrackbarPos("Sat Max", "Master V26")
        v_min = cv2.getTrackbarPos("Val Min", "Master V26")
        
        lower_white = np.array([0, 0, v_min])
        upper_white = np.array([180, s_max, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # 2. GREEN OUTLINE
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

        # 3. NOTCH EXTRACTION
        if live_cnt is not None:
            # Draw Green Outline
            cv2.drawContours(display, [live_cnt], -1, (0, 255, 0), 2)
            
            # --- THE NEW LOGIC ---
            notch_pts = extract_notch_points(live_cnt)
            
            # Draw the extracted points
            if len(notch_pts) > 0:
                for i, pt in enumerate(notch_pts):
                    # Draw Yellow Dot for each notch point
                    cv2.circle(display, tuple(pt), 6, (0, 255, 255), -1)
                    
                    # Connect them with Red Line
                    if i > 0:
                        prev_pt = notch_pts[i-1]
                        cv2.line(display, tuple(prev_pt), tuple(pt), (0, 0, 255), 3)

                # Count points for debugging
                text_pt = notch_pts[0]
                cv2.putText(display, f"Notch Pts: {len(notch_pts)}", (text_pt[0], text_pt[1]-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Match Angle (Background task)
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

        cv2.imshow("Master V26", display)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()