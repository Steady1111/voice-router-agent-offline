#!/usr/bin/env python3
import serial, time
ser = serial.Serial('/dev/cu.usbserial-10', 115200, timeout=0.3)
for _ in range(40):
    line = ser.readline().decode('utf-8', errors='replace')
    if line: print(line, end='', flush=True)
ser.close()
