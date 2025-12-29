"""
Brick Master Vision v7
Major Upgrade: Switched from Binary Threshold to Canny Edge Detection.
This fixes the "Gray Brick" contrast issue by looking for edges/gradients 
instead of absolute darkness.
"""
from __future__ import annotations

import sys
import random
import re
from pathlib import Path
from collections import Counter

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("OpenCV is missing.")


# --- CONFIGURATION ---
CAMERA_INDEX = 0

# k-NN Config
SAMPLE_SIZE = 50
K_NEIGHBORS = 10
PROCESS_W, PROCESS_H = 64, 48
CONFIDENCE_CUTOFF = 50.0

# Geometry Config: The "Bridge/H-Block"
IDEAL_COORDS = np.array([
    [0, 15], [30, 15], [30, 0], [80, 0], [80, 15], [110, 15], # Top
    [110, 75], # Right
    [80, 75], [80, 60], [30, 60], [30, 75], [0, 75] # Bottom Arch
], dtype=np.int32)

SHAPE_MATCH_TOLERANCE = 0.5 # Slightly relaxed for Canny results


def get_ideal_contour():
    temp_img = np.zeros((200, 200), dtype=np.uint8)
    shifted_coords = IDEAL_COORDS + [40, 50]
    cv2.fillPoly(temp_img, [shifted_coords], 255)
    contours, _ = cv2.findContours(temp_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


def process_knn_image(img):
    small = cv2.resize(img, (PROCESS_W, PROCESS_H))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur


def load_dataset(dataset_path: Path) -> list[dict]:
    print(f"Loading memory bank from {dataset_path}...")
    angle_files = {}
    all_files = sorted(list(dataset_path.glob("*.jpg")))
    
    if not all_files:
        print("Warning: No dataset found. Rough Angle detection will be disabled.")
        return []

    for f in all_files:
        match = re.search(r"(\d+(?:pt\d+)?)deg_", f.name)
        if match:
            angle = match.group(1).replace("pt", ".")
            if angle not in angle_files:
                angle_files[angle] = []
            angle_files[angle].append(f)

    memory_bank = []
    for angle, files in angle_files.items():
        selected = random.sample(files, min(len(files), SAMPLE_SIZE))
        for f in selected:
            img = cv2.imread(str(f))
            if img is not None:
                memory_bank.append({'angle': angle, 'data': process_knn_image(img)})

    print(f"Memory loaded: {len(memory_bank)} references.")
    return memory_bank


def draw_ghost_polygon(display_img):
    h, w = display_img.shape[:2]
    scale = 1.0 
    offset_x = w - 150
    offset_y = 50
    
    display_points = []
    for point in IDEAL_COORDS:
        px = int(point[0] * scale + offset_x)
        py = int(point[1] * scale + offset_y)
        display_points.append([px, py])
        
    pts = np.array(display_points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    cv2.polylines(display_img, [pts], True, (255, 200, 0), 2)
    cv2.putText(display_img, "Target Shape", (w - 180, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)


def main():
    base_path = Path(__file__).resolve().parent 
    dataset_path = base_path / "photos" / "angled_bricks_dataset"

    memory_bank = load_dataset(dataset_path)
    ideal_contour = get_ideal_contour()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Camera {CAMERA_INDEX} not found.")
        return

    cv2.namedWindow("Master Vision")
    
    # --- NEW CONTROLS FOR CANNY ---
    # Low: Edges weaker than this are rejected
    # High: Edges stronger than this are accepted immediately
    cv2.createTrackbar("Canny Low", "Master Vision", 50, 255, lambda x: None)
    cv2.createTrackbar("Canny High", "Master Vision", 150, 255, lambda x: None)

    print("\n--- Brick Master Vision v7 (Canny Edge Mode) ---")
    print("Adjust 'Canny Low' and 'Canny High' until the Right Screen clearly shows the brick outline.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display = frame.copy()
        
        # --- BRAIN 1: k-NN Rough Angle ---
        rough_angle_text = "Scanning..."
        knn_color = (0, 0, 255)
        
        if memory_bank:
            live_processed = process_knn_image(frame)
            scores = []
            for mem in memory_bank:
                diff = cv2.absdiff(live_processed, mem['data'])
                scores.append((np.sum(diff), mem['angle']))
            
            scores.sort(key=lambda x: x[0])
            top_k = scores[:K_NEIGHBORS]
            votes = [s[1] for s in top_k]
            if votes:
                winner_angle = Counter(votes).most_common(1)[0][0]
                avg_score = sum(s[0] for s in top_k) / K_NEIGHBORS
                confidence = max(0, min(100, 100 * (1 - (avg_score / 250000))))
                
                if confidence > CONFIDENCE_CUTOFF:
                    rough_angle_text = f"Rough Angle: {winner_angle}"
                    knn_color = (0, 255, 0)
        
        # --- BRAIN 2: Geometry Lock (CANNY VERSION) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Blur slightly to reduce noise (carpet textures, etc)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Get Slider Values
        canny_min = cv2.getTrackbarPos("Canny Low", "Master Vision")
        canny_max = cv2.getTrackbarPos("Canny High", "Master Vision")
        
        # 3. Detect Edges
        edges = cv2.Canny(blur, canny_min, canny_max)
        
        # 4. Dilate (Thicken) the edges
        # This connects small gaps in the outline so it forms a closed loop
        kernel = np.ones((3,3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 5. Find Contours on the edges
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        locked_cnt = None
        best_match_score = 100.0
        
        # Convert edges to color so we can draw on the right screen
        edges_bgr = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2BGR)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter noise (too small) and huge objects (monitor/desk edge)
            if area < 1000 or area > 100000:
                continue
            
            # Draw candidates in gray
            cv2.drawContours(edges_bgr, [cnt], -1, (50, 50, 50), 1)

            score = cv2.matchShapes(cnt, ideal_contour, 1, 0.0)
            
            if score < best_match_score:
                best_match_score = score
            
            if score < SHAPE_MATCH_TOLERANCE:
                if score == best_match_score:
                    locked_cnt = cnt

        # --- DRAW UI ---
        cv2.putText(display, rough_angle_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, knn_color, 2)
        
        if locked_cnt is not None:
            # Green Outline
            cv2.drawContours(display, [locked_cnt], -1, (0, 255, 0), 3)
            cv2.drawContours(edges_bgr, [locked_cnt], -1, (0, 255, 0), 3)
            
            # Draw Corner Dots
            epsilon = 0.02 * cv2.arcLength(locked_cnt, True)
            approx = cv2.approxPolyDP(locked_cnt, epsilon, True)
            for point in approx:
                px, py = point[0]
                cv2.circle(display, (px, py), 5, (0, 0, 255), -1)

            # Center Label
            M = cv2.moments(locked_cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                label = f"LOCKED: {len(approx)} pts"
                cv2.putText(display, label, (cx - 60, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        score_color = (0, 255, 0) if best_match_score < SHAPE_MATCH_TOLERANCE else (0, 165, 255)
        cv2.putText(display, f"Best Score: {best_match_score:.3f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)
        
        draw_ghost_polygon(display)

        # Show Canny Edge View on the right
        combined = np.hstack((display, edges_bgr))
        combined = cv2.resize(combined, (1200, 450))

        cv2.imshow("Master Vision", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()