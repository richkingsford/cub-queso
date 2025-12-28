import serial.tools.list_ports

print("Scanning for connected devices...")
ports = serial.tools.list_ports.comports()

if not ports:
    print("No devices found! (The computer really can't see the Arduino)")
else:
    for port, desc, hwid in ports:
        print(f"FOUND: {port}")
        print(f"       Desc: {desc}")
        print(f"       ID:   {hwid}")
        print("-" * 30)