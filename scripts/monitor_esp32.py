#!/usr/bin/env python3
"""监视 ESP32 串口输出并保存到文件。"""
import serial
import sys
import time

PORT = "/dev/cu.usbserial-10"
BAUD = 115200
LOG_FILE = "/tmp/esp32_monitor.log"

def main():
    print(f"连接到 {PORT} {BAUD}bps...")
    print(f"日志保存到 {LOG_FILE}")
    
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
    except Exception as e:
        print(f"连接失败: {e}")
        sys.exit(1)
    
    with open(LOG_FILE, "w") as f:
        print(f"开始监视... (Ctrl+C 停止)")
        try:
            while True:
                line = ser.readline().decode("utf-8", errors="replace").strip()
                if line:
                    timestamp = time.strftime("%H:%M:%S")
                    output = f"[{timestamp}] {line}"
                    print(output)
                    f.write(output + "\n")
                    f.flush()
        except KeyboardInterrupt:
            print("\n停止监视")
        except Exception as e:
            print(f"\n错误: {e}")
        finally:
            ser.close()

if __name__ == "__main__":
    main()
