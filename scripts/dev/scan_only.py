#!/usr/bin/env python3
"""只扫描 WiFi，不连接"""
import serial, time

# Hard reset first
print("Hard resetting ESP32...")
import subprocess
subprocess.run([
    "python3", "-m", "esptool",
    "--port", "/dev/cu.usbserial-10", "--chip", "esp32s3", "--baud", "115200",
    "run"
], capture_output=True)
time.sleep(1)

print("Reading serial output...")
ser = serial.Serial("/dev/cu.usbserial-10", 115200, timeout=0.5)
start = time.time()
buff = ""
try:
    while time.time() - start < 30:
        line = ser.readline().decode('utf-8', errors='replace')
        if line:
            print(line, end='', flush=True)
            buff += line
finally:
    ser.close()

print("\n" + "="*40)
if "duanduaniPhone" in buff:
    print("✅ FOUND duanduaniPhone in output!")
else:
    print("❌ NOT FOUND in 30 seconds")
