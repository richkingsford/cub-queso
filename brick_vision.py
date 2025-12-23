"""
Brick Vision V23 - "Confidence & Polish"
1. HSV Filter (White vs Wood) - The clean signal.
2. Shape Matching - Uses the notch shape implicitly.
3. UI - Restored Confidence Bars & "Lock" visualization.
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

def get_centroid(cnt):
    M = cv2.moments(cnt)
    if M["m00"] == 0: return (0, 0)
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

def draw_bar(img, label, val_percent, x, y, color=(0, 255, 0)):
    """Draws a cool progress bar."""
    w = 200
    h = 20
    # Background
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), -1)
    # Fill
    fill_w = int(w * (val_percent / 100.0))
    cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, -1)
    # Border
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1)
    # Text
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
            cx, cy = get_centroid(best_cnt)
            reference_data.append({
                'angle': data['angle'],
                'points': data['points'], 
                'contour': best_cnt,
                'centroid': (cx, cy),
                'area': cv2.contourArea(best_cnt)
            })
    print(f"Loaded {len(reference_data)} references.")

def main():
    load_references()
    if not reference_data: sys.exit("No references!")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cv2.namedWindow("Master V23")
    
    # Defaults based on your last successful image
    cv2.createTrackbar("Sat Max", "Master V23", 60, 255, lambda x: None)
    cv2.createTrackbar("Val Min", "Master V23", 140, 255, lambda x: None)

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        display = frame.copy()
        h, w = display.shape[:2]
        
        # 1. HSV FILTER
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        s_max = cv2.getTrackbarPos("Sat Max", "Master V23")
        v_min = cv2.getTrackbarPos("Val Min", "Master V23")
        
        lower_white = np.array([0, 0, v_min])
        upper_white = np.array([180, s_max, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Cleanup
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # 2. FIND CONTOUR
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        live_cnt = None
        max_area = 0
        brick_conf = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 3000:
                # Smooth
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                if area > max_area:
                    max_area = area
                    live_cnt = approx

        # Calculate "Is Brick" Confidence based on area stability
        if max_area > 0:
            # We expect a brick to be decent size, but not whole screen
            ideal_area = (h * w) * 0.25 # Rough guess, 25% of screen
            ratio = min(max_area, ideal_area) / max(max_area, ideal_area)
            brick_conf = min(100, int(ratio * 100) + 40) # Boost slightly
        
        draw_bar(display, "IS BRICK?", brick_conf, 20, 30, (0, 255, 0) if brick_conf > 60 else (0, 165, 255))

        # 3. MATCHING
        if live_cnt is not None:
            # Draw Outline
            cv2.drawContours(display, [live_cnt], -1, (0, 255, 0), 2)
            live_cx, live_cy = get_centroid(live_cnt)
            
            # Find Match
            best_match = None
            best_score = 100.0
            
            for ref in reference_data:
                score = cv2.matchShapes(live_cnt, ref['contour'], 1, 0.0)
                if score < best_score:
                    best_score = score
                    best_match = ref
            
            # Convert Error Score to Confidence %
            # Score 0.0 = 100%, Score 0.5 = 0%
            match_conf = max(0, (1.0 - (best_score * 3.0))) * 100
            
            draw_bar(display, "ANGLE CONFIDENCE", match_conf, 20, 80, (0, 255, 255))

            if best_match and match_conf > 50: # Only draw if decent match
                
                # --- PROJECTION ---
                scale_factor = np.sqrt(max_area / best_match['area'])
                ref_cx, ref_cy = best_match['centroid']
                
                projected_points = []
                for (rx, ry) in best_match['points']:
                    vec_x = rx - ref_cx
                    vec_y = ry - ref_cy
                    final_x = int(live_cx + (vec_x * scale_factor))
                    final_y = int(live_cy + (vec_y * scale_factor))
                    projected_points.append((final_x, final_y))

                # DRAW NOTCH OVERLAY
                if len(projected_points) == 4:
                    p1, p2, p3, p4 = projected_points
                    # Cyan Walls
                    cv2.line(display, p1, p2, (255, 255, 0), 2)
                    cv2.line(display, p3, p4, (255, 255, 0), 2)
                    # RED NOTCH (Thick)
                    cv2.line(display, p2, p3, (0, 0, 255), 4)
                    
                    # Angle Text
                    label = f"{best_match['angle']} deg"
                    cv2.putText(display, label, (live_cx - 60, live_cy - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Thumbnail
        thumb = cv2.resize(mask, (200, 150))
        display[h-150:h, w-200:w] = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(display, (w-200, h-150), (w, h), (255,255,0), 1)

        cv2.imshow("Master V23", display)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()