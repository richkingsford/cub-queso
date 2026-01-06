#!/usr/bin/env python3
"""
Test script to verify objective state changes are being logged correctly.
"""
import json
import sys

def test_log(log_path):
    """Check if a log file has objective transitions."""
    entries = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line in ["[", "]"]: 
                    continue
                if line.endswith(","): 
                    line = line[:-1]
                entries.append(json.loads(line))
    except Exception as e:
        print(f"Error reading log: {e}")
        return
    
    if not entries:
        print("No entries found")
        return
    
    # Track objective transitions
    objectives = []
    for e in entries:
        obj = e.get('objective', 'UNKNOWN')
        if not objectives or objectives[-1] != obj:
            objectives.append(obj)
            timestamp = e.get('timestamp', 0)
            print(f"  {timestamp:.2f}s: {obj}")
    
    print(f"\nSummary:")
    print(f"  Total entries: {len(entries)}")
    print(f"  Unique objectives: {set(obj for obj in objectives)}")
    print(f"  Transitions: {' -> '.join(objectives)}")
    
    # Check if all required objectives are present
    required = {"FIND", "ALIGN", "SCOOP", "LIFT", "PLACE"}
    found = set(e.get('objective') for e in entries)
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
