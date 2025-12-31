#!/usr/bin/env python3
import json
import os
import sys

def validate_session(session_path):
    log_path = os.path.join(session_path, "a_log.json")
    if not os.path.exists(log_path):
        log_path = os.path.join(session_path, "log.json")
    
    if not os.path.exists(log_path):
        return False, "Missing log file"

    entries = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line in ["[", "]"]: continue
                if line.endswith(","): line = line[:-1]
                entries.append(json.loads(line))
    except Exception as e:
        return False, f"JSON Error: {e}"

    if not entries:
        return False, "Empty log"

    # Track objective transitions to detect runs
    objectives_sequence = []
    for e in entries:
        obj = e.get('objective', '')
        if not objectives_sequence or objectives_sequence[-1] != obj:
            objectives_sequence.append(obj)
    
    # Check for all required objectives across the entire session
    found_objs = set(objectives_sequence)
    required_objs = {"FIND", "ALIGN", "SCOOP", "LIFT", "PLACE"}
    missing = required_objs - found_objs
    if missing:
        return False, f"Missing Objectives: {missing}"

    # Detect complete runs (cycles that reach PLACE)
    # A run is complete if we see the sequence reach PLACE
    complete_runs = 0
    current_run_objs = []
    
    for obj in objectives_sequence:
        if obj == "FIND" and current_run_objs:
            # Starting a new run, check if previous was complete
            if "PLACE" in current_run_objs:
                complete_runs += 1
            current_run_objs = ["FIND"]
        else:
            current_run_objs.append(obj)
    
    # Check final run
    if "PLACE" in current_run_objs:
        complete_runs += 1
    
    if complete_runs == 0:
        return False, f"No complete runs (sequence: {' -> '.join(objectives_sequence)})"

    # Check for Vision Stability in ALIGN/SCOOP
    align_vision_hits = 0
    scoop_vision_hits = 0
    for e in entries:
        obj = e.get('objective')
        visible = e.get('brick', {}).get('visible', False)
        if visible:
            if obj == "ALIGN": align_vision_hits += 1
            elif obj == "SCOOP": scoop_vision_hits += 1

    if align_vision_hits < 5:
        return False, f"Poor ALIGN vision ({align_vision_hits} hits)"
    
    # Success!
    total_runs = objectives_sequence.count("FIND")
    return True, f"READY: {complete_runs}/{total_runs} runs, {len(entries)} entries"

def main():
    demos_dir = os.path.join(os.getcwd(), "demos")
    if not os.path.exists(demos_dir):
        print(f"Error: {demos_dir} not found.")
        return

    sessions = [d for d in os.listdir(demos_dir) if os.path.isdir(os.path.join(demos_dir, d))]
    sessions.sort(reverse=True)

    print(f"{'SESSION':<30} | {'STATUS':<10} | {'REASON/STATS'}")
    print("-" * 70)

    for s in sessions:
        path = os.path.join(demos_dir, s)
        ok, reason = validate_session(path)
        status = "PASS" if ok else "FAIL"
        print(f"{s:<30} | {status:<10} | {reason}")

if __name__ == "__main__":
    main()
