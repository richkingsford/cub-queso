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

    # Check for Objectives
    found_objs = set()
    for e in entries:
        found_objs.add(e.get('objective', ''))

    required_objs = {"FIND", "ALIGN", "SCOOP", "LIFT", "PLACE"}
    missing = required_objs - found_objs
    if missing:
        return False, f"Missing Objectives: {missing}"

    # Check for Success
    success = False
    for e in entries:
        evt = e.get('last_event')
        if evt and evt.get('type') == "JOB_SUCCESS":
            success = True
            break
    if not success:
        return False, "No JOB_SUCCESS event"

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
    
    return True, f"READY ({len(entries)} entries)"

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
