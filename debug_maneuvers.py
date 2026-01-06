"""
# debug_maneuvers.py
----------------
Contains the specific movement logic (skills) for the robot.
Separates 'how to move' from 'what job to do'.
"""
import time

# --- CONFIGURATION ---
ALIGN_TOLERANCE_DEG = 5.0   # +/- Degrees allowed for forward movement
CRAWL_SPEED = 0.30          # 30% Speed (approx PWM 120 - Strong enough to move!)

def crawl_forward_if_aligned(robot, detector):
    """
    The Safety-First Approach:
    1. Reads the latest vision frame.
    2. If the brick is lost OR the angle is bad -> STOP.
    3. ONLY if the brick is found AND angle is ~0 -> MOVE FORWARD.
    """
    # 1. Get the latest sensor data
    found, angle, dist, offset_x, max_y = detector.read()

    # 2. SAFETY CHECK: Do we have a lock?
    if not found:
        print(f"[MANEUVER] Target Lost. Stopping.")
        robot.stop()
        return

    # 3. ANGLE CHECK: Are we straight?
    # We want the angle to be effectively 0 (e.g., between -5 and +5)
    if abs(angle) > ALIGN_TOLERANCE_DEG:
        print(f"[MANEUVER] Bad Angle ({int(angle)}°). Stopping.")
        robot.stop()
        return

    # 4. GREEN LIGHT: Move Forward
    # UPDATED: We now use .drive() instead of .set_motors()
    print(f"[MANEUVER] LOCKED ({int(angle)}°) - Crawling...")
    robot.drive(CRAWL_SPEED)