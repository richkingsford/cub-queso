#!/usr/bin/env python3
"""Debug script to test the timing loop"""
import time

# Simulate one event
duration = 0.1  # 100ms from log
command_interval = 0.1

print(f"Testing loop with duration={duration}s, interval={command_interval}s")

start_time = time.time()
iteration = 0

while True:
    elapsed = time.time() - start_time
    print(f"  Iteration {iteration}: elapsed={elapsed:.4f}s")
    
    if elapsed >= duration:
        print(f"  Breaking: elapsed ({elapsed:.4f}) >= duration ({duration})")
        break
    
    print(f"  Sending command...")
    iteration += 1
    
    # Sleep for command interval or remaining time, whichever is shorter
    remaining = duration - elapsed
    sleep_time = min(command_interval, remaining)
    print(f"  Sleeping for {sleep_time:.4f}s (remaining={remaining:.4f})")
    if sleep_time > 0:
        time.sleep(sleep_time)

total_time = time.time() - start_time
print(f"\nTotal iterations: {iteration}")
print(f"Total time: {total_time:.4f}s")
