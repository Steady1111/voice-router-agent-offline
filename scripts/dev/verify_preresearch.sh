#!/bin/bash
# 预研设备验证（Mac 服务端 + ESP32）
# Usage: bash scripts/verify_preresearch.sh [--start]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== 1. 模型检查 ==="
python3 -m voice_router_lite prepare-models

echo ""
echo "=== 2. 文本指令冒烟测试 ==="
python3 -m voice_router_lite pipeline --text "打开风扇"
python3 -m voice_router_lite pipeline --text "关闭风扇"

echo ""
echo "=== 3. 核心测试（预研链路） ==="
python3 -m pytest tests/test_pipeline_integration.py tests/test_kws_asr_tts.py -q --tb=no -k "not test_process_text_unknown and not test_process_multiple_commands and not test_initialize_energy_mode" || true

echo ""
echo "=== 4. 网络信息（ESP32 固件 HUB_HOST 需与此一致）==="
MAC_IP=""
for iface in en0 en1 bridge0; do
  ip=$(ipconfig getifaddr "$iface" 2>/dev/null || true)
  if [[ -n "$ip" ]]; then
    echo "  $iface: $ip"
    MAC_IP="${MAC_IP:-$ip}"
  fi
done
if [[ -z "$MAC_IP" ]]; then
  ifconfig | rg "inet " | rg -v 127.0.0.1 || true
else
  echo ""
  echo "  建议 firmware/include/config.h:"
  echo "    #define HUB_HOST         \"$MAC_IP\""
  echo "    #define HUB_PORT         28080"
fi

FIRMWARE_HOST=$(rg '#define HUB_HOST' firmware/include/config.h 2>/dev/null | awk '{print $3}' | tr -d '"')
if [[ -n "$FIRMWARE_HOST" && -n "$MAC_IP" && "$FIRMWARE_HOST" != "$MAC_IP" ]]; then
  echo ""
  echo "  ⚠️  固件 HUB_HOST=$FIRMWARE_HOST 与当前 Mac IP 不一致，烧录前请修改"
fi

echo ""
echo "=== 5. 启动预研服务 ==="
echo "  浏览器看板: http://127.0.0.1:28080"
echo "  ESP32 WebSocket: ws://${MAC_IP:-<Mac-IP>}:28080/ws/audio?role=esp32"
echo ""
echo "  启动命令:"
echo "    export VOICE_ROUTER_WEB_HOST=0.0.0.0"
echo "    python3 -m voice_router_lite web"

if [[ "${1:-}" == "--start" ]]; then
  echo ""
  echo ">>> 正在启动 web 服务（Ctrl+C 停止）..."
  export VOICE_ROUTER_WEB_HOST=0.0.0.0
  export VOICE_ROUTER_WEB_PORT=28080
  exec python3 -m voice_router_lite web
fi
