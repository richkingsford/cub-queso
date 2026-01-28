#!/usr/bin/env python3
"""
Test script to verify step state changes are being logged correctly.
"""
import sys

from helper_demo_log_utils import read_demo_log, normalize_step_label

def test_log(log_path):
    """Check if a log file has step transitions."""
    try:
        entries = read_demo_log(log_path, strict=True)
    except Exception as e:
        print(f"Error reading log: {e}")
        return
    
    if not entries:
        print("No entries found")
        return
    
    # Track step transitions
    steps = []
    for e in entries:
        obj = normalize_step_label(e.get('step')) if e.get('step') else "UNKNOWN"
        if not steps or steps[-1] != obj:
            steps.append(obj)
            timestamp = e.get('timestamp', 0)
            print(f"  {timestamp:.2f}s: {obj}")
    
    print(f"\nSummary:")
    print(f"  Total entries: {len(entries)}")
    print(f"  Unique steps: {set(obj for obj in steps)}")
    print(f"  Transitions: {' -> '.join(steps)}")
    
    # Check if all required steps are present
    required = {"FIND_BRICK", "ALIGN_BRICK", "SCOOP", "LIFT", "POSITION_BRICK", "PLACE"}
    found = set(normalize_step_label(e.get('step')) for e in entries if e.get('step'))
    missing = required - found
    
    if missing:
        print(f"  ⚠️  Missing steps: {missing}")
    else:
        print(f"  ✓ All steps present!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_step_logging.py <path_to_log.json>")
        sys.exit(1)
    
    test_log(sys.argv[1])
