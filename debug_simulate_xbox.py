"""
test_xbox_client.py
-------------------
Simulates the Windows Xbox Client.
Sends commands to localhost:65432 to test record_demo.py.
"""
import socket
import time

HOST = '127.0.0.1'
PORT = 65432

def send_cmd(s, cmd):
    print(f"Sending: {cmd}")
    s.send(cmd.encode('utf-8'))
    time.sleep(0.1)

def main():
    print(f"Connecting to {HOST}:{PORT}...")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST, PORT))
        print("Connected.")
        
        # 0. Job Start
        print("Test: Job Start (BTN_Y)")
        send_cmd(s, "BTN_Y")
        time.sleep(1)
        
        # 0.5. Job Abort
        print("Test: Job Abort (BTN_B)")
        send_cmd(s, "BTN_B")
        time.sleep(1)
        
        # 1. Drive Forward
        print("Test: Driving Forward")
        for _ in range(5):
            send_cmd(s, "f 100 50")
            time.sleep(0.05)
            
        time.sleep(1)
        
        # 2. Cycle Step
        print("Test: Cycling Step (BTN_X)")
        send_cmd(s, "BTN_X")
        time.sleep(1)
        
        # 3. Confirm Success
        print("Test: Confirming Success (BTN_A)")
        send_cmd(s, "BTN_A")
        time.sleep(2)
        
        print("Done.")
        s.close()
    except ConnectionRefusedError:
        print("Could not connect. Is record_demo.py running?")

if __name__ == "__main__":
    main()
