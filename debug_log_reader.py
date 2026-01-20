#!/usr/bin/env python3
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

from helper_demo_log_utils import read_demo_log, normalize_objective_label
# ANSI Colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RED = "\033[91m"
BOLD = "\033[1m"
END = "\033[0m"

class LogEntry:
    def __init__(self, d):
        self.ts = d.get('timestamp', 0)
        self.pose = d.get('robot_pose', {})
        self.wall = d.get('wall_origin')
        self.brick = d.get('brick', {})
        self.lift = d.get('lift_height', 0)
        self.obj = normalize_objective_label(d.get('objective')) if d.get('objective') else "UNKNOWN"
        self.img = d.get('image_file')

DEMOS_DIR = Path(__file__).resolve().parent / "demos"

def find_sessions():
    demos_dir = DEMOS_DIR
    if not demos_dir.exists():
        return []
    
    sessions = []
    for d in os.listdir(demos_dir):
        p = demos_dir / d
        if p.is_dir():
            # Check for either a_log.json or log.json
            if (p / "a_log.json").exists() or (p / "log.json").exists():
                sessions.append(d)
            
    # Sort by name (timestamp based)
    sessions.sort(reverse=True)
    return sessions

def find_latest_log():
    sessions = find_sessions()
    if not sessions:
        return None
    latest_dir = DEMOS_DIR / sessions[0]
    a_log = latest_dir / "a_log.json"
    if a_log.exists():
        return str(a_log)
    return str(latest_dir / "log.json")

def format_pose(pose):
    return f"({pose.get('x',0):.1f}, {pose.get('y',0):.1f}, {pose.get('theta',0):.1f}°)"

def get_summary_line(entry):
    """Generates a one-line summary of the current state."""
    status = f"OBJ:{entry.obj}"
    
    if entry.brick.get('visible'):
        vision = f"{GREEN}Seen (Dist:{entry.brick.get('dist',0):.0f}mm, Ang:{entry.brick.get('angle',0):.1f}°){END}"
    else:
        vision = f"{RED}Searching...{END}"
        
    return f"{status} | {vision} | Pose:{format_pose(entry.pose)}"

def summarize_log(path):
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return

    print(f"{BOLD}{BLUE}Reading Log:{END} {path}")
    
    entries = []
    keyframes = []
    for data in read_demo_log(path):
        entry_type = data.get("type")
        if entry_type == "state":
            entries.append(LogEntry(data))
        elif entry_type == "keyframe":
            keyframes.append({
                "ts": data.get("timestamp", 0),
                "marker": data.get("marker")
            })

    if not entries:
        print(f"{RED}Log is empty or unreadable.{END}")
        return

    start_ts = entries[0].ts
    duration_total = entries[-1].ts - start_ts
    
    output_lines = []
    output_lines.append(f"SESSION SUMMARY: {os.path.dirname(path)}")
    output_lines.append(f"Total Entries: {len(entries)}")
    output_lines.append(f"Total Time: {duration_total:.1f} seconds")
    output_lines.append("-" * 60)
    output_lines.append("Timeline Summary:")

    def get_period_desc(e):
        vis = e.brick.get('visible', False)
        
        base_desc = ""
        if e.obj == "FIND_BRICK":
            if not vis:
                base_desc = "Searching for a brick..."
            else:
                base_desc = "Found a brick!"
        elif e.obj == "ALIGN_BRICK":
            base_desc = "Aligning with brick..."
        elif e.obj == "FIND_WALL":
            base_desc = "Finding the wall..."
        elif e.obj == "FIND_WALL2":
            base_desc = "Finding wall (return)..."
        elif e.obj == "POSITION_BRICK":
            base_desc = "Positioning brick..."
        elif e.obj == "SCOOP":
            base_desc = "Approaching/Scooping brick..."
        elif e.obj == "LIFT":
            base_desc = "Scooping/Lifting brick..."
        elif e.obj == "PLACE":
            base_desc = "Putting down brick..."
        else:
            base_desc = f"Objective: {e.obj}"
            
        return base_desc

    current_period = {
        'start': 0,
        'desc': None,
        'min_dist': float('inf'),
        'max_dist': float('-inf'),
        'vis_count': 0
    }
    wall_set_time = None

    # --- SUMMARIZATION LOOP ---
    periods = []
    
    current_p = {
        'start': 0,
        'obj': entries[0].obj,
        'held': entries[0].brick.get('held', False),
        'min_dist': float('inf'),
        'max_dist': float('-inf'),
        'vis_count': 0,
    }
    
    for i, e in enumerate(entries):
        elapsed = e.ts - start_ts
        dist = e.brick.get('dist', 0) if e.brick.get('visible') else None
        held = e.brick.get('held', False)
        
        # Merge if Objective AND State Flags are same
        state_match = (e.obj == current_p['obj'] and 
                       held == current_p['held'])
        
        if state_match:
            if dist is not None:
                current_p['min_dist'] = min(current_p['min_dist'], dist)
                current_p['max_dist'] = max(current_p['max_dist'], dist)
                current_p['vis_count'] += 1
        else:
            # Commit current
            current_p['end'] = elapsed
            periods.append(current_p)
            
            # Start new
            current_p = {
                'start': elapsed,
                'obj': e.obj,
                'held': held,
                'min_dist': float('inf'),
                'max_dist': float('-inf'),
                'vis_count': 0,
            }
            if dist is not None:
                current_p['min_dist'] = dist
                current_p['max_dist'] = dist
                current_p['vis_count'] = 1
                
    # Final flush
    current_p['end'] = entries[-1].ts - start_ts
    periods.append(current_p)

    # --- FORMAT OUTPUT ---
    obj_map = {
        "FIND_WALL": "Finding wall (FIND_WALL)",
        "FIND_BRICK": "Looking for brick (FIND_BRICK)",
        "ALIGN_BRICK": "Aligning with brick (ALIGN_BRICK)",
        "SCOOP": "Scooping brick (SCOOP)",
        "LIFT": "Lifting brick (LIFT)",
        "FIND_WALL2": "Finding wall again (FIND_WALL2)",
        "POSITION_BRICK": "Positioning brick (POSITION_BRICK)",
        "PLACE": "Lowering brick (PLACE)"
    }
    
    # ANSI for internal color markers
    GREEN_M = "\033[32m"
    
    for p in periods:
        desc = obj_map.get(p['obj'], f"Objective: {p['obj']}")
        
        # Add state markers
        if p['held']:
            desc += f" {BOLD}{GREEN_M}(HELD){END}"
        # Add vision info
        if p['vis_count'] > 0:
            d_min, d_max = p['min_dist'], p['max_dist']
            if abs(d_min - d_max) < 0.1:
                desc += f". Distance was {d_min:.0f}mm"
            else:
                desc += f". Distance was between {d_min:.0f} and {d_max:.0f}mm"
        else:
            # If we are held, we don't necessarily care about "No brick seen" noise
            if not p['held']:
                desc += ". (No brick seen)"
        
        output_lines.append(f"Sec {p['start']:4.1f} - {p['end']:4.1f}: {desc}")
    
    # --- SPECIAL EVENTS ---
    # Highlights for Success/Abort/Wall
    for kf in keyframes:
        elapsed = kf["ts"] - start_ts
        marker = kf.get("marker")
        if marker == "JOB_SUCCESS":
            output_lines.append(f"{BOLD}{GREEN}      >>> JOB SUCCESS CONFIRMED at {elapsed:.1f}s <<<{END}")
        elif marker == "JOB_ABORT":
            output_lines.append(f"{BOLD}{RED}      >>> JOB ABORTED at {elapsed:.1f}s <<<{END}")
            
    wall_set_time = None
    for e in entries:
        if e.wall and wall_set_time is None:
            wall_set_time = e.ts - start_ts
            w = e.wall
            output_lines.append(f"      >> WALL SET at x={w['x']:.0f}, y={w['y']:.0f} [at {wall_set_time:.1f}s]")
            break

    output_lines.append("-" * 60)

    # 1. Print to Terminal
    for line in output_lines:
        # Simple color replacement for terminal
        l = line.replace("Found a brick!", f"{GREEN}Found a brick!{END}")
        l = l.replace("WALL SET", f"{YELLOW}WALL SET{END}")
        print(l)
        
    # 2. Save to a_summary.txt
    summary_path = os.path.join(os.path.dirname(path), "a_summary.txt")
    try:
        with open(summary_path, 'w') as f:
            for line in output_lines:
                # Strip all potential color codes
                clean = line
                for code in [GREEN, RED, YELLOW, BLUE, MAGENTA, CYAN, BOLD, END, ORANGE_M, GREEN_M]:
                    clean = clean.replace(code, "")
                f.write(clean + "\n")
        print(f"\n{BOLD}{CYAN}Persistent summary saved to:{END} {summary_path}")
    except Exception as e:
        print(f"Failed to save summary file: {e}")

    print(f"\n{GREEN}Summary Complete.{END}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize robot session logs.")
    parser.add_argument("log_path", nargs="?", help="Path to log.json")
    parser.add_argument("--list", action="store_true", help="List all sessions")
    args = parser.parse_args()
    
    if args.list:
        sessions = find_sessions()
        if not sessions:
            print("No sessions found.")
        else:
            print(f"{BOLD}Available Sessions:{END}")
            for i, s in enumerate(sessions):
                print(f"  [{i}] {s}")
        sys.exit(0)

    log_path = args.log_path or find_latest_log()
    
    if not log_path:
        print(f"No log files found in {DEMOS_DIR}. Use the recorder first!")
        sys.exit(1)
        
    summarize_log(log_path)
