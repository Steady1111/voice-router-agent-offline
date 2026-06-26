#!/usr/bin/env python3
"""烧录 ESP32 并监控串口输出"""
import subprocess, serial, time, sys, os

FIRMWARE_DIR = "/Users/duanduan/CodingFiles/voice-router-agent-offline/firmware"
BUILD_DIR = os.path.join(FIRMWARE_DIR, ".pio/build/esp32-s3-devkitc-1")
PORT = "/dev/cu.usbserial-10"

# Step 1: Flash all partitions
print("=== Flashing ESP32 ===", flush=True)
files = [
    (0x0, os.path.join(BUILD_DIR, "bootloader.bin")),
    (0x8000, os.path.join(BUILD_DIR, "partitions.bin")),
    (0x10000, os.path.join(BUILD_DIR, "firmware.bin")),
]
args = [
    "python3", "-m", "esptool",
    "--port", PORT, "--chip", "esp32s3", "--baud", "115200",
    "--before", "default_reset", "--after", "hard_reset",
    "write_flash", "-z", "--flash_mode", "dio",
    "--flash_freq", "80m", "--flash_size", "8MB",
]
for addr, path in files:
    args.extend([hex(addr), path])

try:
    subprocess.run(args, check=True, timeout=180)
    print("Flash done.", flush=True)
except subprocess.TimeoutExpired:
    print("Flash timeout!", flush=True)
    sys.exit(1)

# Step 2: Monitor serial
print("=== Monitoring serial ===", flush=True)
time.sleep(1)  # Let ESP32 boot
try:
    ser = serial.Serial(PORT, 115200, timeout=0.5)
    start = time.time()
    buff = ""
    while time.time() - start < 35:
        line = ser.readline().decode('utf-8', errors='replace')
        if line:
            print(line, end='', flush=True)
            buff += line
    ser.close()
    
    print("\n=== Summary ===", flush=True)
    if "WiFi ok" in buff:
        print("SUCCESS: WiFi connected!", flush=True)
    elif "FOUND" in buff:
        print("PARTIAL: SSID found but not connecting", flush=True)
    else:
        print("FAILED: SSID not found or handshake timeout", flush=True)
        print("Last 200 chars:", buff[-200:] if buff else "(no output)", flush=True)
except Exception as e:
    print(f"Serial error: {e}", flush=True)
