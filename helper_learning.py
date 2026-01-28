import math
import numpy as np
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class Action:
    cmd: str
    speed: float

class BehavioralCloningPolicy:
    def __init__(self, k=5):
        self.k = k
        self.data_points = []
        self.actions = []
        self.cmd_map = {} # Map 'f','l', etc to integers for any fancy logic if needed
        
    def train(self, segments_by_obj):
        """
        Ingests demo segments and builds the memory bank.
        segments_by_obj: dict like { "ALIGN_BRICK": [seg1, seg2], ... }
        """
        self.policy_by_obj = {}
        
        for obj_name, segments in segments_by_obj.items():
            # We only care about SUCCESS/NOMINAL segments
            valid_segs = []
            if isinstance(segments, dict):
                valid_segs.extend(segments.get("SUCCESS", []))
                valid_segs.extend(segments.get("NOMINAL", []))
            elif isinstance(segments, list):
                valid_segs = segments
                
            if not valid_segs:
                continue

            points = []
            actions = []

            for seg in valid_segs:
                events = seg.get("events") or []
                states = seg.get("states") or []
                
                # Simple alignment: Map State[t] -> Action[t]
                # We need to correlate timestamps.
                # Since log rates are high, we can just find the closest state for each action event.
                
                # Sort states by time
                sorted_states = sorted(
                    [s for s in states if s.get("timestamp")], 
                    key=lambda x: x["timestamp"]
                )
                if not sorted_states:
                    continue
                    
                state_times = np.array([s["timestamp"] for s in sorted_states])
                
                for evt in events:
                    # Look for Motion Actions
                    etype = evt.get("type")
                    if etype != "action":
                        if etype == "event" and evt.get("event", {}).get("type") in ("forward","backward","left_turn","right_turn"):
                            # Handle different log formats if necessary
                            pass
                        continue
                        
                    cmd = evt.get("command") # f, b, l, r
                    if cmd not in ("f", "b", "l", "r"):
                        continue
                        
                    power = float(evt.get("power", 0)) / 255.0
                    if power <= 0.05:
                        continue
                        
                    ts = evt.get("timestamp")
                    if not ts:
                        continue
                        
                    # Find closest state
                    idx = np.searchsorted(state_times, ts)
                    # check left and right neighbor
                    best_idx = -1
                    best_diff = 999.0
                    
                    for i in (idx-1, idx):
                        if 0 <= i < len(state_times):
                            diff = abs(state_times[i] - ts)
                            if diff < best_diff:
                                best_diff = diff
                                best_idx = i
                                
                    if best_idx == -1 or best_diff > 0.2: # If state is stale (>200ms), skip
                        continue
                        
                    steps = sorted_states[best_idx]
                    brick = steps.get("brick")
                    if not brick or not brick.get("visible"):
                        continue
                        
                    # Extract Features
                    # Feature Vector: [offset_x, angle, dist]
                    # Normalize roughly to similar scales for Euclidean distance
                    # Dist: 0-500mm -> 0-1 (div by 500)
                    # Angle: 0-90deg -> 0-1 (div by 90)
                    # Offset: 0-100mm -> 0-1 (div by 100)
                    
                    f_offset = (brick.get("offset_x") or 0.0) / 100.0
                    f_angle = (brick.get("angle") or 0.0) / 90.0
                    f_dist = (brick.get("dist") or 0.0) / 500.0
                    
                    points.append([f_offset, f_angle, f_dist])
                    actions.append(Action(cmd, power))

            if points:
                self.policy_by_obj[obj_name] = {
                    "X": np.array(points, dtype=np.float32),
                    "Y": actions
                }
                print(f"[LEARN] {obj_name}: Trained on {len(points)} samples.")

    def query(self, step, world):
        """
        Returns (cmd, speed, confidence) based on k-NN
        """
        model = self.policy_by_obj.get(step)
        if not model:
            return None, 0.0, 0.0
            
        brick = world.brick or {}
        if not brick.get("visible"):
            return None, 0.0, 0.0
            
        f_offset = (brick.get("offset_x") or 0.0) / 100.0
        f_angle = (brick.get("angle") or 0.0) / 90.0
        f_dist = (brick.get("dist") or 0.0) / 500.0
        
        query_pt = np.array([f_offset, f_angle, f_dist], dtype=np.float32)
        
        # k-NN Logic
        # Calculate distances
        X = model["X"]
        dists = np.linalg.norm(X - query_pt, axis=1)
        
        # Get top k indices
        k = min(self.k, len(X))
        idx = np.argsort(dists)[:k]
        
        nearest_actions = [model["Y"][i] for i in idx]
        
        # Voting
        # Weight by 1/distance? Or just majority vote?
        # Let's do simple majority for command, average for speed
        
        votes = defaultdict(float)
        speed_sums = defaultdict(float)
        counts = defaultdict(int)
        
        for i, action in enumerate(nearest_actions):
            # Distance weighting: 1 / (d + eps)
            w = 1.0 / (dists[idx[i]] + 1e-4)
            votes[action.cmd] += w
            speed_sums[action.cmd] += action.speed * w
            counts[action.cmd] += 1 # just for tracking
            
        best_cmd = max(votes.items(), key=lambda item: item[1])[0]
        avg_speed = speed_sums[best_cmd] / votes[best_cmd]
        
        # Confidence logic? 
        # Ratio of best_cmd votes to total votes
        total_weight = sum(votes.values())
        confidence = votes[best_cmd] / total_weight if total_weight > 0 else 0.0
        
        return best_cmd, avg_speed, confidence
