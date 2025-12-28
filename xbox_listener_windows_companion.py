import socket
import serial
import time
import sys

# --- CONFIGURATION ---
# Check if your Arduino is USB0 or ACM0 by running 'ls /dev/tty*'
SERIAL_PORT = '/dev/ttyCH341USB0' 
BAUD_RATE = 115200  # <--- FIXED (Matches Arduino)
HOST_IP = '0.0.0.0'
PORT = 65432

# 1. Setup Serial
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to Arduino on {SERIAL_PORT}")
    time.sleep(2) # Wait for reboot
    ser.reset_input_buffer()
except Exception as e:
    print(f"Error opening serial port: {e}")
    sys.exit(1)

# 2. Setup Network
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST_IP, PORT))
server_socket.listen(1)

print(f"=== LEIA LISTENER READY on Port {PORT} ===")

try:
    while True:
        print("Waiting for Xbox Controller...")
        conn, addr = server_socket.accept()
        print(f"Controller Connected! ({addr})")

        with conn:
            while True:
                data = conn.recv(1024)
                if not data: break # Laptop disconnected
                
                command = data.decode('utf-8')
                # print(f"Command: {command}") # Uncomment to debug
                ser.write(command.encode('utf-8'))

        print("Controller Disconnected. Resetting...")

except KeyboardInterrupt:
    server_socket.close()
    ser.close()
    print("\nBye!")