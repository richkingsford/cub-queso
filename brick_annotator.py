"""
Brick Annotator Tool V3
- TARGET: 'photos/angled_bricks_subset'
- CONTROLS: 'Q' (Next), 'A' (Previous), 'ESC' (Quit).
- Auto-skips '0deg' images just in case.
"""
import cv2
import json
import re
from pathlib import Path

# --- CONFIG ---
# Make sure you create this folder and put your selected photos in it!
DATASET_DIR = Path(__file__).parent / "photos" / "angled_bricks_subset"
OUTPUT_FILE = Path(__file__).parent / "brick_database.json"

# Global state
current_points = []
current_img = None
display_img = None
database = {}

def click_event(event, x, y, flags, params):
    global current_points, display_img, current_img
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(current_points) < 4:
            current_points.append((x, y))
            redraw()

def redraw():
    global display_img, current_img, current_points
    if current_img is None: return
    display_img = current_img.copy()
    
    # Draw points
    for i, pt in enumerate(current_points):
        # 0,3 = Outer (Cyan), 1,2 = Inner (Red)
        color = (0, 0, 255) 
        if i == 0 or i == 3: color = (255, 255, 0)
            
        cv2.circle(display_img, pt, 5, color, -1)
        cv2.putText(display_img, str(i+1), (pt[0]+10, pt[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if i > 0:
            cv2.line(display_img, current_points[i-1], pt, (0, 255, 0), 1)

    cv2.imshow("Annotator V3", display_img)

def load_database():
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, 'r') as f:
                return json.load(f)
        except: return {}
    return {}

def save_database():
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(database, f, indent=4)
    print("Database saved.")

def main():
    global display_img, current_img, current_points, database

    if not DATASET_DIR.exists():
        print(f"Error: Folder not found: {DATASET_DIR}")
        print("Please create the folder 'angled_bricks_subset' inside 'photos' and copy your selected images there.")
        return

    database = load_database()
    
    all_files = sorted(list(DATASET_DIR.glob("*.jpg")) + list(DATASET_DIR.glob("*.png")))
    # Filter 0deg
    filtered_files = [f for f in all_files if "0deg" not in f.name and "0pt0deg" not in f.name]
    
    if not filtered_files:
        print("No images found in 'angled_bricks_subset' (excluding 0deg).")
        return

    print(f"Loaded {len(filtered_files)} images to annotate.")

    cv2.namedWindow("Annotator V3")
    cv2.setMouseCallback("Annotator V3", click_event)
    
    index = 0
    total = len(filtered_files)

    while True:
        f_path = filtered_files[index]
        filename = f_path.name
        
        # Parse Angle
        angle = "Unknown"
        match = re.search(r"(\d+(?:pt\d+)?)deg_", filename)
        if match: angle = match.group(1).replace("pt", ".")

        # Load existing points
        current_points = database.get(filename, {}).get('points', [])

        current_img = cv2.imread(str(f_path))
        if current_img is None:
            index += 1
            if index >= total: break
            continue

        redraw()
        
        while True:
            ui_view = display_img.copy()
            status = f"[{index+1}/{total}] {filename} ({angle} deg)"
            instr = "Q: Next | A: Prev | R: Reset | ESC: Quit"
            
            if len(current_points) == 4:
                cv2.putText(ui_view, "COMPLETE", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                prompts = ["Click Outer Left", "Click Inner Left", "Click Inner Right", "Click Outer Right"]
                if len(current_points) < 4:
                    cv2.putText(ui_view, prompts[len(current_points)], (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.putText(ui_view, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(ui_view, instr, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Annotator V3", ui_view)
            k = cv2.waitKey(20) & 0xFF
            
            # --- CONTROLS ---
            if k == 27: # ESC to Quit
                save_database()
                return

            if k == ord('r'): # Reset
                current_points = []
                redraw()

            # Q = Next
            if k == ord('q'): 
                if len(current_points) == 4:
                    database[filename] = {'angle': angle, 'points': current_points}
                    save_database()
                
                if index < total - 1:
                    index += 1
                    break
                else:
                    print("End of list.")

            # A = Previous
            if k == ord('a'):
                if len(current_points) == 4:
                    database[filename] = {'angle': angle, 'points': current_points}
                    save_database()

                if index > 0:
                    index -= 1
                    break
                else:
                    print("Start of list.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()