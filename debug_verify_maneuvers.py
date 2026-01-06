import debug_maneuvers as maneuvers
import sys

# Mock Robot
class MockRobot:
    def stop(self): print("Robot stop")
    def drive(self, speed): print(f"Robot drive {speed}")

# Mock Detector
class MockDetector:
    def read(self):
        # Return 4 values to simulate the new brick_vision interface
        return True, 0.0, 100.0, 5.0 

robot = MockRobot()
detector = MockDetector()

print("Testing maneuvers.crawl_forward_if_aligned...")
try:
    maneuvers.crawl_forward_if_aligned(robot, detector)
    print("Test passed!")
except ValueError as e:
    print(f"Test failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Test failed with unexpected error: {e}")
    sys.exit(1)
