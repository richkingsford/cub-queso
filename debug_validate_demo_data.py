#!/usr/bin/env python3
import os
from pathlib import Path
import sys

from helper_demo_log_utils import read_demo_log, resolve_session_log, normalize_objective_label


def validate_session(session_path):
    log_path = resolve_session_log(session_path)
    if not log_path:
        return False, "Missing log file"
    
    try:
        entries = read_demo_log(log_path, strict=True)
    except Exception as e:
        return False, f"JSON Error: {e}"

    if not entries:
        return False, "Empty log"

    # Track objective transitions to detect runs
    objectives_sequence = []
    for e in entries:
        obj = normalize_objective_label(e.get('objective', '')) if e.get('objective') else ''
        if not objectives_sequence or objectives_sequence[-1] != obj:
            objectives_sequence.append(obj)
    
    # Check for all required objectives across the entire session
    found_objs = set(objectives_sequence)
    required_objs = {"FIND_BRICK", "ALIGN_BRICK", "SCOOP", "LIFT", "POSITION_BRICK", "PLACE"}
    missing = required_objs - found_objs
    if missing:
        return False, f"Missing Objectives: {missing}"

    # Detect complete runs (cycles that reach PLACE)
    # A run is complete if we see the sequence reach PLACE
    complete_runs = 0
    current_run_objs = []
    
    for obj in objectives_sequence:
        if obj == "FIND_BRICK" and current_run_objs:
            # Starting a new run, check if previous was complete
            if "PLACE" in current_run_objs:
                complete_runs += 1
            current_run_objs = ["FIND_BRICK"]
        else:
            current_run_objs.append(obj)
    
    # Check final run
    if "PLACE" in current_run_objs:
        complete_runs += 1
    
    if complete_runs == 0:
        return False, f"No complete runs (sequence: {' -> '.join(objectives_sequence)})"

    # Check for Vision Stability in ALIGN_BRICK/SCOOP
    align_vision_hits = 0
    scoop_vision_hits = 0
    for e in entries:
        obj = normalize_objective_label(e.get('objective'))
        visible = e.get('brick', {}).get('visible', False)
        if visible:
            if obj == "ALIGN_BRICK": align_vision_hits += 1
            elif obj == "SCOOP": scoop_vision_hits += 1

    if align_vision_hits < 5:
        return False, f"Poor ALIGN_BRICK vision ({align_vision_hits} hits)"
    
    # Success!
    total_runs = objectives_sequence.count("FIND_BRICK")
    return True, f"READY: {complete_runs}/{total_runs} runs, {len(entries)} entries"

def main():
    demos_dir = Path(__file__).resolve().parent / "demos"
    if not demos_dir.exists():
        print(f"Error: {demos_dir} not found.")
        return

    sessions = [d for d in os.listdir(demos_dir) if (demos_dir / d).is_dir()]
    sessions.sort(reverse=True)

    print(f"{'SESSION':<30} | {'STATUS':<10} | {'REASON/STATS'}")
    print("-" * 70)

    for s in sessions:
        path = demos_dir / s
        ok, reason = validate_session(str(path))
        status = "PASS" if ok else "FAIL"
        print(f"{s:<30} | {status:<10} | {reason}")

if __name__ == "__main__":
    main()
