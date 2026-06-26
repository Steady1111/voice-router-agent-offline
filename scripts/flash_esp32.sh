#!/bin/bash
# 烧录 ESP32-S3 固件
cd /Users/duanduan/CodingFiles/voice-router-agent-offline/firmware
# 先释放串口
fuser -k /dev/cu.usbserial-10 2>/dev/null
sleep 1
pio run -t upload --upload-port /dev/cu.usbserial-10
echo "=== FLASH DONE ==="
