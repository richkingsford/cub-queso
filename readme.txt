This app lets you manually control a robot to do a specific job. Our generation
1 robot uses a single micro servo motor on an Arduino Uno (pin 5) to sweep a
camera across a set of colored dots.

### Job definition

* **Start**: hold the camera on the **green** dot for 2 seconds.
* **Targets**: find the **purple**, **red**, and **dark blue** dots—in that
  order. For each dot, hold for 1 second, jiggle briefly, and hold for another
  second.
* **End**: hold on the **pink** dot for 2 seconds.

The Python app logs what color the camera is centered on and key transitions
like start, panning direction while searching, dot hits, jiggling, and end
signals.

### Running the orchestrator

1. Install Python 3.9+.
2. Run a simulated job timeline:

   ```bash
   python main.py --mode demo
   ```

3. Or run interactively, typing the observed color names as you manually pan
   the camera:

   ```bash
   python main.py --mode live
   ```

Every run executes quick smoke tests and attempts to open **camera 1** (USB),
printing a warning if it cannot be reached. If the OpenCV dependency is not
installed, the camera check is skipped.

Replace the default servo transport in `robot_controller/servo.py` with calls
to your Arduino library when you are ready to drive real hardware.
