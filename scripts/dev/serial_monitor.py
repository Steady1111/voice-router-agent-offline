#!/usr/bin/env python3
"""临时串口监视器 —— 读取 ESP32 串口输出并打印到终端"""
import serial
import sys
import time

PORT = '/dev/cu.usbserial-10'
BAUD = 115200
DURATION = 15  # 读取 15 秒

try:
    ser = serial.Serial(PORT, BAUD, timeout=0.5)
except Exception as e:
    print(f"无法打开串口 {PORT}: {e}")
    sys.exit(1)

print(f"=== 串口监视器 {PORT} @ {BAUD} (采集 {DURATION}s) ===")
ser.reset_input_buffer()

deadline = time.time() + DURATION
while time.time() < deadline:
    try:
        line = ser.readline().decode('utf-8', errors='replace')
        if line:
            print(line, end='', flush=True)
    except KeyboardInterrupt:
        break

ser.close()
print(f"\n=== 采集结束 ===")
