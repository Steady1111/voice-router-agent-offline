# ESP32-S3 firmware

> English · [中文](README.zh-CN.md)

Push-to-talk mic node. Holds a button, streams 16 kHz / 16-bit / mono PCM
over WebSocket to the hub service, then waits for the next button press.

## Hardware

- ESP32-S3-DevKitC-1 (any flavor with WiFi)
- INMP441 I2S MEMS microphone
- A momentary push button (or use the BOOT key on the dev board)
- Optional: a 0.5 W speaker + I2S DAC for TTS playback (not wired in this
  sketch — the hub still plays TTS in the browser)

Default pin map:

| Signal      | ESP32-S3 pin |
|-------------|--------------|
| I2S BCLK    | GPIO 4       |
| I2S LRCLK   | GPIO 5       |
| I2S DIN     | GPIO 6       |
| Push button | GPIO 0 (BOOT)|
| Status LED  | GPIO 48 (onboard RGB) |

Change them in `include/config.h` (copied from `config.h.example`).

## Build

```bash
pip install platformio
cp include/config.h.example include/config.h
# edit config.h with your WiFi + hub host:port
pio run -t upload
pio device monitor
```

## Wire protocol

Matches the hub's `/ws/audio` endpoint:

```
client → server:
  {"event": "start", "sample_rate": 16000, "format": "pcm_s16le"}
  <binary PCM>
  {"event": "stop"}
server → client:
  {"event": "transcript", "text": "..."}
  {"event": "reply", "text": "...", "actions": [...]}
  ...
```

This sketch ignores the inbound `tts_audio` payload — playback happens in
the browser dashboard for now. Once you add a speaker, hook the binary
frames into an I2S DAC stream.
