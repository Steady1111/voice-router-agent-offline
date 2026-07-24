#!/usr/bin/env python3
"""串口监控 - 45秒"""
import serial, time
ser = serial.Serial('/dev/cu.usbserial-10', 115200, timeout=0.3)
start = time.time()
buff = ''
try:
    while time.time() - start < 30:
        line = ser.readline().decode('utf-8', errors='replace')
        if line:
            print(line, end='', flush=True)
            buff += line
finally:
    ser.close()
    if 'WiFi ok' in buff:
        print('\n✅ 连接成功!')
    elif 'FOUND' in buff and 'NOT FOUND' not in buff.split('FOUND')[-1][:50]:
        print('\n✅ 扫描到SSID!')
    else:
        print('\n❌ 仍未见SSID')
