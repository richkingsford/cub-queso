import serial
import time
import keyboard  # pip install keyboard

SERIAL_PORT = "COM6"   # your Arduino port
BAUD_RATE = 115200

MIN_PULSE = 1000
MAX_PULSE = 2000
STOP_PULSE = 1500
STEP = 50

def main():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # wait for Arduino reset

    pulse = STOP_PULSE
    ser.write(f"{pulse}\n".encode())
    print("Hold LEFT/RIGHT to spin. Press 's' to stop, 'q' to quit.")

    try:
        while True:
            if keyboard.is_pressed("left"):
                pulse = max(MIN_PULSE, pulse - STEP)
            elif keyboard.is_pressed("right"):
                pulse = min(MAX_PULSE, pulse + STEP)
            elif keyboard.is_pressed("s"):
                pulse = STOP_PULSE
            elif keyboard.is_pressed("q"):
                print("Quitting...")
                break

            # Always re‑send the current pulse so Arduino keeps driving
            ser.write(f"{pulse}\n".encode())
            print(f"Pulse: {pulse}")
            time.sleep(0.1)

    finally:
        ser.write(f"{STOP_PULSE}\n".encode())
        ser.close()

if __name__ == "__main__":
    main()