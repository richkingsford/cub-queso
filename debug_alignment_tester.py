"""
Brick Alignment Tester v3: High Precision & Transparency
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
    sys.exit("OpenCV is missing. Please install it: pip install opencv-python")


# --- CONFIGURATION ---
CAMERA_INDEX = 0             
SAMPLE_SIZE = 50             # Memories per angle
K_NEIGHBORS = 10             # Increased to 10 voters
PROCESS_W, PROCESS_H = 64, 48 

# Confidence Calculation
MAX_DIFF_THRESHOLD = 250000  # The score that equals 0% confidence
CONFIDENCE_CUTOFF = 60.0     # Below this % -> "No see brick"


def process_image(img):
    """Standardizes an image for comparison."""
    small = cv2.resize(img, (PROCESS_W, PROCESS_H))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur


def load_dataset(dataset_path: Path) -> list[dict]:
    print(f"Scanning {dataset_path}...")
    angle_files = {}
    all_files = sorted(list(dataset_path.glob("*.jpg")))
    
    if not all_files:
        raise FileNotFoundError(f"No images found in {dataset_path}.")

    for f in all_files:
        match = re.search(r"(\d+(?:pt\d+)?)deg_", f.name)
        if match:
            angle = match.group(1).replace("pt", ".")
            if angle not in angle_files:
                angle_files[angle] = []
            angle_files[angle].append(f)

    memory_bank = []
    for angle, files in angle_files.items():
        selected_files = files
        if len(files) > SAMPLE_SIZE:
            selected_files = random.sample(files, SAMPLE_SIZE)
            
        print(f" -> Loading {len(selected_files)} images for {angle}°...")
        for f in selected_files:
            img = cv2.imread(str(f))
            if img is not None:
                processed = process_image(img)
                memory_bank.append({'angle': angle, 'data': processed})

    print(f"System Ready. Total memories: {len(memory_bank)}")
    return memory_bank


def main():
    base_path = Path(__file__).resolve().parent 
    dataset_path = base_path / "photos" / "angled_bricks_dataset"

    try:
        memory_bank = load_dataset(dataset_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Cannot open camera {CAMERA_INDEX}")
        return

    print("\n--- Brick Matcher v3 ---")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Comparison Logic
        live_processed = process_image(frame)
        
        scores = []
        for mem in memory_bank:
            diff = cv2.absdiff(live_processed, mem['data'])
            score_val = np.sum(diff)
            scores.append((score_val, mem['angle']))

        # Sort: Lowest score is best match
        scores.sort(key=lambda x: x[0])
        
        # Take top 10
        top_k = scores[:K_NEIGHBORS]
        
        # Calculate Consensus
        votes = [s[1] for s in top_k]
        vote_counts = Counter(votes)
        winner_angle, strength = vote_counts.most_common(1)[0]
        
        # Calculate Average Confidence
        avg_score_top_k = sum(s[0] for s in top_k) / K_NEIGHBORS
        confidence = max(0, min(100, 100 * (1 - (avg_score_top_k / MAX_DIFF_THRESHOLD))))

        # 2. Display Logic
        display = frame.copy()
        h, w = display.shape[:2]

        # Determine Main Status
        if confidence < CONFIDENCE_CUTOFF:
            main_text = "No see brick"
            main_color = (0, 0, 255) # Red
        else:
            main_text = f"Angle: {winner_angle}"
            main_color = (0, 255, 0) # Green

        # --- DRAW UI ---
        
        # A. Top Banner (Result)
        cv2.rectangle(display, (0, 0), (w, 60), (0,0,0), -1) 
        cv2.putText(display, main_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, main_color, 2)
        
        # Stats in banner
        stats = f"Conf: {confidence:.0f}%"
        cv2.putText(display, stats, (w - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        # B. Right Sidebar (The "Thinking" List)
        # Draw a semi-transparent box on the right
        overlay = display.copy()
        sidebar_w = 220
        cv2.rectangle(overlay, (w - sidebar_w, 60), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

        # Header for Sidebar
        cv2.putText(display, "Top 10 Matches:", (w - sidebar_w + 10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # List the top 10 voters
        y_offset = 110
        for i, (score, angle) in enumerate(top_k):
            # Highlight the ones that voted for the winner
            if angle == winner_angle and confidence >= CONFIDENCE_CUTOFF:
                row_color = (0, 255, 0) # Green text for winning votes
            else:
                row_color = (180, 180, 180) # Gray for others

            row_text = f"#{i+1}: {angle} deg"
            cv2.putText(display, row_text, (w - sidebar_w + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, row_color, 1)
            y_offset += 25

        cv2.imshow("Brick Alignment Tester v3", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()