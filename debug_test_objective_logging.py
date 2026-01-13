#!/usr/bin/env python3
"""
Test script to verify objective state changes are being logged correctly.
"""
import sys

from helper_demo_log_utils import read_demo_log, normalize_objective_label

def test_log(log_path):
    """Check if a log file has objective transitions."""
    try:
        entries = read_demo_log(log_path, strict=True)
    except Exception as e:
        print(f"Error reading log: {e}")
        return
    
    if not entries:
        print("No entries found")
        return
    
    # Track objective transitions
    objectives = []
    for e in entries:
        obj = normalize_objective_label(e.get('objective')) if e.get('objective') else "UNKNOWN"
        if not objectives or objectives[-1] != obj:
            objectives.append(obj)
            timestamp = e.get('timestamp', 0)
            print(f"  {timestamp:.2f}s: {obj}")
    
    print(f"\nSummary:")
    print(f"  Total entries: {len(entries)}")
    print(f"  Unique objectives: {set(obj for obj in objectives)}")
    print(f"  Transitions: {' -> '.join(objectives)}")
    
    # Check if all required objectives are present
    required = {"FIND_BRICK", "ALIGN_BRICK", "SCOOP", "LIFT", "POSITION_BRICK", "PLACE"}
    found = set(normalize_objective_label(e.get('objective')) for e in entries if e.get('objective'))
    missing = required - found
    
    if missing:
        print(f"  ⚠️  Missing objectives: {missing}")
    else:
        print(f"  ✓ All objectives present!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_objective_logging.py <path_to_log.json>")
        sys.exit(1)
    
    test_log(sys.argv[1])
