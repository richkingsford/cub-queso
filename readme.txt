Manual camera sweep utility for the dot-seeking robot prototype.

## Requirements

* Python 3.9+
* `pip install opencv-python numpy pyserial`
* Arduino-compatible continuous servo setup wired the same way as the original project (pin 5 handled by `robot_controller/servo.py`).

## Manual sweep capture (main.py)

Run the interactive capture loop:

```
python main.py --camera 1 --photos-dir photos --interval 1.0
```

Controls:

* Left arrow  – pan left (hold to keep moving)
* Right arrow – pan right
* Space       – stop the servo immediately
* ESC         – stop the loop and exit

Every time the loop starts it empties the target photos folder, then saves one JPEG per second (or whatever interval you pass). Filenames include timestamps for easy sorting.

## Camera smoke test (test_camera.py)

Capture a single frame to verify wiring and drivers:

```
python test_camera.py --camera 1 --output camera_test.jpg
```

The image is written next to `main.py`. Use this before the full sweep to make sure the camera index is correct.

## Annotating saved photos

If you need to draw circles/labels around colored dots after capturing, use the helpers in `robot_controller/vision.py`. For example, inside a Python shell:

```python
from robot_controller.vision import annotate_photo_directory
annotate_photo_directory("photos", "annotated")
```

This produces annotated copies in the `annotated` folder. Adjust HSV ranges in `robot_controller/vision.py` if your lighting conditions differ.
