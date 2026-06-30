// DOM 缓存
const DEBUG = new URLSearchParams(location.search).get("debug") === "1";
const DEMO = new URLSearchParams(location.search).get("demo") === "1" || DEBUG;

const els = {
  talkBtn: document.getElementById("talk-btn"),
  ttsPlayer: document.getElementById("tts-player"),
  routerDot: document.getElementById("router-dot"),
  routerHeaderText: document.getElementById("router-header-text"),
  routerWan: document.getElementById("router-wan"),
  routerClients: document.getElementById("router-clients"),
  routerDaemon: document.getElementById("router-daemon"),
  routerRam: document.getElementById("router-ram"),
  routerCpu: document.getElementById("router-cpu"),
  routerLastCmd: document.getElementById("router-last-cmd"),
  fanDot: document.getElementById("fan-dot"),
  fanHeaderText: document.getElementById("fan-header-text"),
  deviceList: document.getElementById("device-list"),
  // 顶栏状态
  wakeDot: document.getElementById("wake-dot"),
  wakeText: document.getElementById("wake-text"),
  // 唤醒卡片
  wakeCard: document.getElementById("wake-card"),
  wakeLabel: document.getElementById("wake-label"),
  wakeTime: document.getElementById("wake-time"),
  // 对话
  chatPanel: document.getElementById("chat-panel"),
  chatList: document.getElementById("chat-list"),
  chatEmpty: document.getElementById("chat-empty"),
  // 文本调试
  textForm: document.getElementById("text-form"),
  textInput: document.getElementById("text-input"),
  suggestCmds: document.getElementById("suggest-cmds"),
  routerCmdSelect: document.getElementById("router-cmd-select"),
  // 性能监控
  monitorGrid: document.getElementById("monitor-grid"),
  monitorPanel: document.getElementById("monitor-panel"),
  monitorToggle: document.getElementById("monitor-toggle"),
  pipelineDemo: document.getElementById("pipeline-demo"),
  pipeWake: document.getElementById("pipe-wake"),
  pipeAsr: document.getElementById("pipe-asr"),
  pipeNlu: document.getElementById("pipe-nlu"),
};

// WebSocket / Audio 状态
let ws = null;
let audioCtx = null;
let mediaStream = null;
let workletNode = null;
let recording = false;
let pcmChunks = [];

// 设备图标映射
function updatePipelineStep(el, label, value) {
  if (!el) return;
  const span = el.querySelector("span");
  if (span) span.textContent = value || "—";
}

function setPipelineWake(keyword, wakeSource) {
  if (!keyword && !wakeSource) {
    updatePipelineStep(els.pipeWake, "① 唤醒", "—");
    return;
  }
  const src = wakeSource ? `（${wakeSource}）` : "";
  updatePipelineStep(els.pipeWake, "① 唤醒", `「${keyword || "小T小T"}」${src}`);
}

function clearPipelineWake() {
  updatePipelineStep(els.pipeWake, "① 唤醒", "—");
}

let pendingVoiceCommand = null;

function displayCommandText(result) {
  if (!result) return "";
  return (result.command || result.transcript || "").trim();
}

function setPipelineAsr(text) {
  updatePipelineStep(els.pipeAsr, "② 识别", text || "—");
}

function setPipelineNlu(result) {
  if (!result || !result.intent) {
    updatePipelineStep(els.pipeNlu, "③ 意图", "—");
    return;
  }
  const conf = result.confidence != null ? ` ${(result.confidence * 100).toFixed(0)}%` : "";
  const slots = result.slots && Object.keys(result.slots).length
    ? ` · ${Object.entries(result.slots).map(([k, v]) => `${k}=${v}`).join(", ")}`
    : "";
  updatePipelineStep(els.pipeNlu, "③ 意图", `${result.intent}${conf}${slots}`);
}

function applyDemoMode() {
  // 流水线与推荐指令始终展示；demo=1 仅作兼容标记
  if (els.pipelineDemo) els.pipelineDemo.hidden = false;
}

const DEVICE_ICONS = {
  light_livingroom: "💡",
  light_bedroom: "🛏️",
  led: "💡",
  light: "💡",
  fan: "🌀",
  ac: "❄️",
  air_conditioner: "❄️",
  relay: "🔌",
};

function wsUrl() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}/ws/audio?role=browser`;
}

function ensureSocket() {
  if (ws && ws.readyState <= 1) return ws;
  console.log("[WS] Connecting to", wsUrl());
  ws = new WebSocket(wsUrl());
  ws.binaryType = "arraybuffer";
  ws.onopen = () => console.log("[WS] Connected OK");
  ws.onmessage = onWsMessage;
  ws.onclose = () => {
    console.warn("[WS] Closed, will reconnect in 3s");
    ws = null;
    setTimeout(ensureSocket, 3000);
  };
  ws.onerror = (e) => console.warn("[WS] Error:", e);
  return ws;
}

// 处理服务端消息
let pendingAudioChunks = [];
let pendingTtsFormat = "mpeg";
let speechKeepAlive = null;
let preferredVoice = null;

// 中文女声优选（按流畅度/自然度排序；越靠前越优先）
const FEMALE_VOICE_PRIORITY = [
  /Xiaoxiao/i,       // Windows / Edge 神经网络女声
  /Xiaoyi/i,
  /Tingting/i,
  /Ting-Ting/i,      // macOS 女声
  /Meijia/i,
  /Sinji/i,
  /Sin-Ji/i,
  /Lili/i,
  /Huihui/i,
  /Xiaoni/i,
  /Xiaorui/i,
  /Yaoyao/i,
  /Female/i,
  /女/,
];

// 明确排除的男声
const MALE_VOICE_PATTERN =
  /Kangkang|Li-mu|Li-Mu|Yunjian|Yunxi|Yunfeng|Yunyang|男|Male|Daniel|Fred|Grandpa|Google 普通话（中国大陆）/i;

function stopSpeechKeepAlive() {
  if (speechKeepAlive) {
    clearInterval(speechKeepAlive);
    speechKeepAlive = null;
  }
}

function preloadSpeechVoices() {
  if (!window.speechSynthesis) return;
  const voices = window.speechSynthesis.getVoices();
  if (voices.length === 0) return;
  preferredVoice = pickChineseFemaleVoice(voices);
}

function pickChineseFemaleVoice(voices) {
  const zh = voices.filter((v) => {
    const lang = v.lang.replace("_", "-").toLowerCase();
    return lang.startsWith("zh");
  });
  if (!zh.length) return null;

  for (const pattern of FEMALE_VOICE_PRIORITY) {
    const hit = zh.find((v) => pattern.test(v.name) && !MALE_VOICE_PATTERN.test(v.name));
    if (hit) return hit;
  }

  const notMale = zh.filter((v) => !MALE_VOICE_PATTERN.test(v.name));
  // 优先本地语音包，通常比默认 Google 男声更自然
  const local = notMale.find((v) => v.localService);
  if (local) return local;

  return notMale[0] || null;
}

function ensureSpeechVoices() {
  return new Promise((resolve) => {
    if (!window.speechSynthesis) {
      resolve([]);
      return;
    }
    const existing = window.speechSynthesis.getVoices();
    if (existing.length > 0) {
      resolve(existing);
      return;
    }
    const onChange = () => {
      window.speechSynthesis.removeEventListener("voiceschanged", onChange);
      resolve(window.speechSynthesis.getVoices());
    };
    window.speechSynthesis.addEventListener("voiceschanged", onChange);
    setTimeout(() => {
      window.speechSynthesis.removeEventListener("voiceschanged", onChange);
      resolve(window.speechSynthesis.getVoices());
    }, 800);
  });
}

function preloadSpeechVoices() {
  if (!window.speechSynthesis) return;
  const voices = window.speechSynthesis.getVoices();
  if (voices.length === 0) return;
  preferredVoice = pickChineseFemaleVoice(voices);
  if (preferredVoice) {
    console.log("[TTS] 已选女声:", preferredVoice.name, preferredVoice.lang);
  }
}

if (window.speechSynthesis) {
  window.speechSynthesis.onvoiceschanged = preloadSpeechVoices;
  preloadSpeechVoices();
  ensureSpeechVoices().then((voices) => {
    if (voices.length) {
      preferredVoice = pickChineseFemaleVoice(voices);
      if (preferredVoice) {
        console.log("[TTS] 已选女声:", preferredVoice.name, preferredVoice.lang);
      }
    }
  });
}

async function speakReply(text) {
  if (!text || !window.speechSynthesis) return;
  stopSpeechKeepAlive();
  window.speechSynthesis.cancel();

  const voices = await ensureSpeechVoices();
  const voice = preferredVoice || pickChineseFemaleVoice(voices);
  if (voice) preferredVoice = voice;

  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = voice?.lang?.startsWith("zh") ? voice.lang : "zh-CN";
  utter.rate = 0.98;
  utter.pitch = 1.02;
  utter.volume = 1.0;
  if (voice) utter.voice = voice;
  utter.onend = stopSpeechKeepAlive;
  utter.onerror = stopSpeechKeepAlive;
  window.speechSynthesis.speak(utter);
  // Chrome/Safari：页面点击会暂停朗读，定时 resume 保持播报
  speechKeepAlive = setInterval(() => {
    if (!window.speechSynthesis.speaking) {
      stopSpeechKeepAlive();
      return;
    }
    if (window.speechSynthesis.paused) {
      window.speechSynthesis.resume();
    }
  }, 250);
}

function onWsMessage(ev) {
  if (typeof ev.data === "string") {
    const msg = JSON.parse(ev.data);
    switch (msg.event) {
      case "transcript":
        if (msg.text) setPipelineAsr(msg.text);
        break;
      case "reply": {
        const userCmd = displayCommandText(msg.result);
        pendingVoiceCommand = null;
        const src = msg.source || "browser";
        if (userCmd) {
          setPipelineAsr(userCmd);
          addChatMessage("user", userCmd, src);
        }
        setPipelineNlu(msg.result);
        addChatMessage("assistant", msg.text, src, msg.result);
        refreshDevices();
        refreshRouterStatus();
        if (src === "esp32") resetWakeStatus();
        setTimeout(() => speakReply(msg.text), 0);
        break;
      }
      case "tts_audio":
        pendingAudioChunks = [];
        pendingTtsFormat = msg.format || "mpeg";
        break;
      case "done":
        if (pendingAudioChunks.length > 0) {
          if (!(window.speechSynthesis && window.speechSynthesis.speaking)) {
            playTtsAudio(pendingAudioChunks);
          }
          pendingAudioChunks = [];
        }
        // 唤醒听指令期间，TTS done 不要清掉红灯/读秒
        if (esp32LedMode !== "wake") {
          resetWakeStatus();
        }
        break;
      case "wake_detected": {
        const src = msg.wake_source || "volume";
        setPipelineWake(msg.keyword, src);
        showWakeDetected(msg.keyword, src ? ` · ${src}` : "");
        break;
      }
      case "collecting":
        showCollecting(msg.text);
        break;
      case "error":
        addChatMessage("assistant", "出错了：" + msg.message, msg.source);
        speakReply(msg.message ? "出错了，" + msg.message : "出错了");
        break;
    }
    return;
  }
  pendingAudioChunks.push(ev.data);
}

function playTtsAudio(chunks) {
  const totalLen = chunks.reduce((s, c) => s + c.byteLength, 0);
  const merged = new Uint8Array(totalLen);
  let offset = 0;
  for (const c of chunks) {
    merged.set(new Uint8Array(c), offset);
    offset += c.byteLength;
  }
  const mime = pendingTtsFormat === "wav" ? "audio/wav" : "audio/mpeg";
  const blob = new Blob([merged], { type: mime });
  const url = URL.createObjectURL(blob);
  const player = els.ttsPlayer;
  if (player._lastUrl) URL.revokeObjectURL(player._lastUrl);
  player._lastUrl = url;
  player.src = url;
  player.play().catch(() => {});
}

// 对话气泡
function addChatMessage(role, text, source, nluResult) {
  if (!text) return;
  els.chatEmpty.style.display = "none";

  const now = new Date();
  const timeStr = now.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
  const sourceLabel = source === "esp32" ? " · ESP32" : source === "browser" ? " · 网页" : "";
  let hint = "";
  if (role === "assistant" && nluResult?.intent) {
    const conf = nluResult.confidence != null
      ? `${(nluResult.confidence * 100).toFixed(0)}%`
      : "—";
    hint = `<div class="chat-msg-hint">意图：${escapeHtml(nluResult.intent)}（置信度 ${conf}）</div>`;
  }

  const div = document.createElement("div");
  div.className = `chat-msg ${role}`;
  div.innerHTML = `
    <span class="chat-msg-label">${role === "user" ? "🧑 你" : "🤖 小T"}${sourceLabel}</span>
    <div class="chat-msg-bubble">${escapeHtml(text)}</div>
    ${hint}
    <span class="chat-msg-time">${timeStr}</span>
  `;
  els.chatList.appendChild(div);

  // 限制最多 15 条，超出删旧
  while (els.chatList.children.length > 15) {
    els.chatList.firstElementChild.remove();
  }

  // 滚动到底部
  els.chatPanel.scrollTop = els.chatPanel.scrollHeight;
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

// ====== 录音 ======
let recordingStartPromise = null;

function mergePcmChunks(chunks) {
  const total = chunks.reduce((s, c) => s + c.byteLength, 0);
  const merged = new Uint8Array(total);
  let offset = 0;
  for (const c of chunks) {
    merged.set(c, offset);
    offset += c.byteLength;
  }
  return merged;
}

function resamplePcm16To16k(pcmBytes, fromRate) {
  if (fromRate === 16000) return pcmBytes;
  const samples = new Int16Array(
    pcmBytes.buffer,
    pcmBytes.byteOffset,
    pcmBytes.byteLength / 2,
  );
  const ratio = 16000 / fromRate;
  const outLen = Math.max(1, Math.round(samples.length * ratio));
  const out = new Int16Array(outLen);
  for (let i = 0; i < outLen; i += 1) {
    const src = i / ratio;
    const idx = Math.floor(src);
    const frac = src - idx;
    const a = samples[Math.min(idx, samples.length - 1)] || 0;
    const b = samples[Math.min(idx + 1, samples.length - 1)] || 0;
    out[i] = Math.round(a + (b - a) * frac);
  }
  return new Uint8Array(out.buffer);
}

function resetTalkButton() {
  if (!els.talkBtn) return;
  els.talkBtn.classList.remove("recording");
  const label = els.talkBtn.querySelector(".talk-btn-text");
  if (label) label.textContent = "按住说话";
}

function micAccessHint() {
  const host = location.hostname;
  const isLocal = host === "localhost" || host === "127.0.0.1" || host === "[::1]";
  if (!window.isSecureContext && !isLocal) {
    return `当前地址 ${location.origin} 不是安全上下文，浏览器会禁用麦克风。请改用 http://localhost:28080 打开（本机），或使用下方文本框。`;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    return "当前窗口不支持麦克风（常见于 Cursor 内置预览）。请用 Chrome/Safari 打开 http://localhost:28080。";
  }
  return "";
}

let lastMicErrorText = "";
let lastMicErrorAt = 0;

function reportMicError(err) {
  const name = err?.name || "";
  let detail = err?.message || String(err);
  if (name === "NotAllowedError" || name === "PermissionDeniedError") {
    detail = "浏览器拒绝了麦克风权限，请在地址栏左侧打开网站设置并允许麦克风";
  } else if (name === "NotFoundError") {
    detail = "未检测到麦克风设备";
  } else if (detail === "当前浏览器不支持麦克风") {
    detail = micAccessHint() || detail;
  }
  const text = "麦克风不可用：" + detail;
  const now = Date.now();
  if (text === lastMicErrorText && now - lastMicErrorAt < 3000) return;
  lastMicErrorText = text;
  lastMicErrorAt = now;
  addChatMessage("assistant", text + "。路由器指令可直接用下方文本框。", "browser");
}

async function requestMicStream() {
  const hint = micAccessHint();
  if (hint) throw new Error(hint);

  const constraints = {
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  };
  if (navigator.mediaDevices?.getUserMedia) {
    return navigator.mediaDevices.getUserMedia(constraints);
  }
  const legacy = navigator.getUserMedia
    || navigator.webkitGetUserMedia
    || navigator.mozGetUserMedia;
  if (!legacy) {
    throw new Error(micAccessHint() || "当前浏览器不支持麦克风");
  }
  return new Promise((resolve, reject) => {
    legacy.call(navigator, constraints, resolve, reject);
  });
}

async function startRecording() {
  if (recording || recordingStartPromise) return;

  recordingStartPromise = (async () => {
    const stream = await requestMicStream();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const source = ctx.createMediaStreamSource(stream);

    const workletCode = `
      class PCM16Capture extends AudioWorkletProcessor {
        process(inputs) {
          const ch = inputs[0][0];
          if (!ch) return true;
          const out = new Int16Array(ch.length);
          for (let i = 0; i < ch.length; i++) {
            const s = Math.max(-1, Math.min(1, ch[i]));
            out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
          }
          this.port.postMessage(out.buffer, [out.buffer]);
          return true;
        }
      }
      registerProcessor("pcm16-capture", PCM16Capture);
    `;
    const blobUrl = URL.createObjectURL(new Blob([workletCode], { type: "text/javascript" }));
    await ctx.audioWorklet.addModule(blobUrl);
    URL.revokeObjectURL(blobUrl);

    const node = new AudioWorkletNode(ctx, "pcm16-capture");
    pcmChunks = [];
    node.port.onmessage = (e) => pcmChunks.push(new Uint8Array(e.data));
    source.connect(node);

    mediaStream = stream;
    audioCtx = ctx;
    workletNode = node;
    recording = true;

    els.talkBtn.classList.add("recording");
    els.talkBtn.querySelector(".talk-btn-text").textContent = "松开发送";
    clearPipelineWake();

    ensureSocket();
    const send = () => {
      ws.send(JSON.stringify({
        event: "start",
        sample_rate: 16000,
        format: "pcm_s16le",
      }));
    };
    if (ws.readyState === 1) send();
    else ws.addEventListener("open", send, { once: true });
  })();

  try {
    await recordingStartPromise;
  } catch (err) {
    console.error("[Talk] start failed:", err);
    recording = false;
    resetTalkButton();
    reportMicError(err);
  } finally {
    recordingStartPromise = null;
  }
}

async function stopRecording() {
  if (recordingStartPromise) {
    try {
      await recordingStartPromise;
    } catch {
      return;
    }
  }
  if (!recording) return;
  recording = false;
  resetTalkButton();

  if (workletNode) workletNode.disconnect();
  workletNode = null;
  const fromRate = audioCtx?.sampleRate || 48000;
  if (audioCtx) await audioCtx.close().catch(() => {});
  audioCtx = null;
  if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
  mediaStream = null;

  if (!ws || ws.readyState !== 1) {
    addChatMessage("assistant", "语音通道未连接，请刷新页面后重试。", "browser");
    pcmChunks = [];
    return;
  }

  const pcm = resamplePcm16To16k(mergePcmChunks(pcmChunks), fromRate);
  pcmChunks = [];

  if (pcm.byteLength < 3200) {
    ws.send(JSON.stringify({ event: "stop" }));
    addChatMessage("assistant", "录音太短，请按住按钮至少 1 秒再松开。", "browser");
    return;
  }

  for (let i = 0; i < pcm.byteLength; i += 8192) {
    ws.send(pcm.subarray(i, i + 8192));
  }
  ws.send(JSON.stringify({ event: "stop" }));
}

// 按住说话
if (els.talkBtn) {
  els.talkBtn.addEventListener("mousedown", startRecording);
  els.talkBtn.addEventListener("mouseup", stopRecording);
  els.talkBtn.addEventListener("mouseleave", stopRecording);
  els.talkBtn.addEventListener("touchstart", (e) => { e.preventDefault(); startRecording(); });
  els.talkBtn.addEventListener("touchend", (e) => { e.preventDefault(); stopRecording(); });
}

// F2 快捷键
document.addEventListener("keydown", (e) => {
  if (e.key === "F2" && !e.repeat) { e.preventDefault(); startRecording(); }
});
document.addEventListener("keyup", (e) => {
  if (e.key === "F2") { e.preventDefault(); stopRecording(); }
});

// ====== 文本指令 ======
async function runTextCommand(text) {
  const cmd = text.trim();
  if (!cmd) return;
  if (els.textInput) els.textInput.value = "";
  addChatMessage("user", cmd);
  clearPipelineWake();
  setPipelineAsr(cmd);
  const resp = await fetch("/api/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: cmd }),
  });
  const data = await resp.json();
  setPipelineNlu(data);
  addChatMessage("assistant", data.reply || "—", null, data);
  setTimeout(() => speakReply(data.reply), 0);
  refreshDevices();
}

if (els.textForm) {
  els.textForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    await runTextCommand(els.textInput?.value || "");
  });
}

if (els.suggestCmds) {
  els.suggestCmds.addEventListener("click", async (e) => {
    const btn = e.target.closest(".suggest-chip");
    if (!btn) return;
    const cmd = btn.getAttribute("data-cmd");
    if (cmd) await runTextCommand(cmd);
  });
}

const ROUTER_CMD_GROUPS = [
  {
    label: "重启路由器",
    intent: "router_reboot",
    commands: [
      "重启路由器",
      "重启路由",
      "重启网络",
      "网络卡了重启一下",
      "路由器死机了重启一下",
    ],
  },
  {
    label: "WiFi 开关",
    intent: "router_wifi_restart",
    commands: [
      "重启WiFi",
      "打开WiFi",
      "关闭WiFi",
      "WiFi连不上重启一下",
    ],
  },
  {
    label: "WiFi 配置",
    intent: "router_wifi_config",
    commands: [
      "WiFi密码多少",
      "改WiFi密码",
      "WiFi改个名字",
      "打开访客网络",
      "关闭访客网络",
      "切换到5G频段",
    ],
  },
  {
    label: "网络查询",
    intent: "router_network_query",
    commands: [
      "IP地址是多少",
      "谁连了WiFi",
      "连了几个设备",
      "网速怎么样",
      "路由器运行多久了",
      "内存剩多少",
      "CPU负载多少",
    ],
  },
  {
    label: "指示灯",
    intent: "router_led_control",
    commands: [
      "关闭路由器灯",
      "打开路由器灯",
    ],
  },
  {
    label: "设备管理",
    intent: "router_device_manage",
    commands: [
      "踢掉这个设备",
      "拉黑这部手机",
      "取消拉黑这个设备",
      "查看黑名单",
    ],
  },
  {
    label: "网络诊断",
    intent: "router_network_diag",
    commands: [
      "测一下网速",
      "Ping一下",
      "延迟多少",
      "网络诊断",
      "DNS正常吗",
    ],
  },
  {
    label: "系统运维",
    intent: "router_system",
    commands: [
      "检查固件更新",
      "备份配置",
      "恢复出厂设置",
      "查看系统日志",
      "设置定时重启",
    ],
  },
  {
    label: "QoS / 带宽",
    intent: "router_qos",
    commands: [
      "打开QoS",
      "关闭QoS",
      "限制这台电脑的网速",
      "给游戏机优先",
    ],
  },
  {
    label: "安全策略",
    intent: "router_security",
    commands: [
      "打开防火墙",
      "关闭防火墙",
      "打开家长控制",
      "打开VPN",
      "防蹭网",
    ],
  },
];

function buildRouterCmdSelect() {
  const sel = els.routerCmdSelect;
  if (!sel) return;
  for (const group of ROUTER_CMD_GROUPS) {
    const og = document.createElement("optgroup");
    og.label = group.label;
    for (const cmd of group.commands) {
      const opt = document.createElement("option");
      opt.value = cmd;
      opt.textContent = cmd;
      og.appendChild(opt);
    }
    sel.appendChild(og);
  }
}

if (els.routerCmdSelect) {
  els.routerCmdSelect.addEventListener("change", async () => {
    const cmd = els.routerCmdSelect.value;
    if (!cmd) return;
    els.routerCmdSelect.value = "";
    await runTextCommand(cmd);
  });
}

function resetWakeStatus() {
  esp32LedMode = "idle";
  stopWakeCountdown();
  if (els.wakeCard) {
    els.wakeCard.classList.remove("active");
    els.wakeLabel.textContent = "离线语音控制台";
    els.wakeTime.textContent = "";
  }
  els.wakeDot.classList.remove("active");
  els.wakeText.textContent = "等待唤醒";
  if (wakeTimer) clearTimeout(wakeTimer);
  wakeTimer = null;
  refreshDevices();
}

// ====== 路由器 / 风扇状态卡 ======
async function refreshRouterStatus() {
  try {
    const resp = await fetch("/api/router-status");
    const data = await resp.json();
    const wanOk = !!data.wan_up;
    if (els.routerDot) {
      els.routerDot.classList.toggle("ok", wanOk);
      els.routerDot.classList.toggle("bad", !wanOk);
    }
    if (els.routerHeaderText) {
      els.routerHeaderText.textContent = wanOk ? "路由器在线" : "路由器离线";
    }
    if (els.routerWan) els.routerWan.textContent = wanOk ? "已连接" : "断开";
    if (els.routerClients) {
      els.routerClients.textContent = data.lan_clients == null ? "—" : String(data.lan_clients);
    }
    if (els.routerDaemon) els.routerDaemon.textContent = data.voice_daemon || "—";
    if (els.routerRam && data.ram_used_mb != null) {
      const pct = data.ram_total_mb
        ? Math.round(data.ram_used_mb / data.ram_total_mb * 100)
        : null;
      els.routerRam.textContent = pct != null
        ? `${data.ram_used_mb} / ${data.ram_total_mb} MB（${pct}%）`
        : `${data.ram_used_mb} MB`;
    }
    if (els.routerCpu) {
      els.routerCpu.textContent = data.cpu_pct != null ? `${data.cpu_pct}%` : "—";
    }
    if (els.routerLastCmd) {
      els.routerLastCmd.textContent = data.last_command || "—";
    }
    return data;
  } catch {
    return null;
  }
}

async function refreshFanStatus() {
  try {
    const resp = await fetch("/api/esp32-status");
    const data = await resp.json();
    const connected = !!data.connected;
    if (els.fanDot) {
      els.fanDot.classList.toggle("ok", connected);
      els.fanDot.classList.toggle("bad", !connected);
    }
    if (els.fanHeaderText) {
      els.fanHeaderText.textContent = connected ? "风扇在线" : "风扇离线";
    }
    return data;
  } catch {
    return null;
  }
}

function renderDeviceRow({ icon, name, state, stateClass, meta }) {
  return `
    <li>
      <div class="device-item-left">
        <span class="device-icon">${icon}</span>
        <span class="device-name">${escapeHtml(name)}</span>
      </div>
      <div class="device-item-right">
        <span class="device-state ${stateClass}">${escapeHtml(state)}</span>
        ${meta ? `<span class="device-meta">${escapeHtml(meta)}</span>` : ""}
      </div>
    </li>`;
}

async function refreshDevices() {
  if (!els.deviceList) return;
  try {
    const [, esp] = await Promise.all([
      refreshRouterStatus(),
      refreshFanStatus(),
    ]);
    const espData = esp || {};
    const connected = !!espData.connected;
    const fanOn = connected && !!espData.fan?.on;
    const fanLevel = espData.fan?.level;
    const tempC = espData.temperature_c;

    const items = [
      {
        icon: "🌀",
        name: "风扇",
        state: !connected ? "离线" : (fanOn ? "开" : "关"),
        stateClass: fanOn ? "on" : "off",
        meta: connected && fanLevel != null ? `${fanLevel} 档` : "",
      },
      {
        icon: "🌡️",
        name: "温度",
        state: tempC != null ? `${Number(tempC).toFixed(1)}°C` : "—",
        stateClass: tempC != null ? "on" : "off",
        meta: connected ? "ESP32 传感器" : "",
      },
      {
        icon: "💡",
        name: "ESP32 指示灯",
        ...esp32IndicatorView(connected, fanOn),
      },
    ];

    els.deviceList.innerHTML = items.map(renderDeviceRow).join("");
  } catch { /* ignore */ }
}

// ====== 网络状态 ======
async function refreshNetwork() {
  await refreshDevices();
}

// ====== 唤醒状态 ======
let wakeTimer = null;
let esp32LedMode = "idle"; // idle | wake | fan_on
let wakeCountdownIv = null;
let wakeCountdownLeft = 0;

function stopWakeCountdown() {
  if (wakeCountdownIv) clearInterval(wakeCountdownIv);
  wakeCountdownIv = null;
  wakeCountdownLeft = 0;
}

function startWakeCountdown(seconds = 10) {
  stopWakeCountdown();
  wakeCountdownLeft = seconds;
  refreshDevices();
  wakeCountdownIv = setInterval(() => {
    wakeCountdownLeft -= 1;
    refreshDevices();
    if (wakeCountdownLeft <= 0) stopWakeCountdown();
  }, 1000);
}

function esp32IndicatorView(connected, fanOn) {
  if (!connected) {
    return { state: "离线", stateClass: "off", meta: "" };
  }
  if (esp32LedMode === "wake") {
    const meta = wakeCountdownLeft > 0 ? `听指令 ${wakeCountdownLeft}s` : "听指令中";
    return { state: "红灯", stateClass: "wake", meta };
  }
  if (fanOn) {
    return { state: "绿灯", stateClass: "on", meta: "风扇运行" };
  }
  return { state: "绿灯", stateClass: "on", meta: "待机监听" };
}

function showWakeDetected(keyword, sourceHint = "") {
  esp32LedMode = "wake";
  const now = new Date();
  const timeStr = now.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const label = keyword
    ? `已唤醒（${keyword}）${sourceHint}，正在听指令…`
    : `已唤醒${sourceHint}，正在听指令…`;

  if (els.wakeCard) {
    els.wakeCard.classList.add("active");
    els.wakeLabel.textContent = label;
    els.wakeTime.textContent = timeStr;
  }

  els.wakeDot.classList.add("active");
  els.wakeText.textContent = "唤醒 " + timeStr + (sourceHint || "");

  if (wakeTimer) clearTimeout(wakeTimer);
  wakeTimer = setTimeout(resetWakeStatus, 12000);
  startWakeCountdown(12);
  refreshDevices();
}

function showCollecting(text, { forEsp32 = true } = {}) {
  if (!forEsp32) return;
  esp32LedMode = "wake";
  if (els.wakeCard) els.wakeCard.classList.add("active");
  if (els.wakeLabel) els.wakeLabel.textContent = text || "正在听指令…";
  els.wakeDot.classList.add("active");
  els.wakeText.textContent = "录音中";
  startWakeCountdown(12);
  refreshDevices();
}

// ====== 性能监控 ======

// 七项指标定义（以路由器 128MB 为目标基准）
const MONITOR_CARDS = [
  {
    id: "audio",
    label: "音频电平",
    unit: "",
    template: "audio", // 自定义渲染：音量条
    icon: "🎙️",
  },
  {
    id: "ram",
    label: "内存占用",
    unit: "MB",
    template: true, // 自定义渲染：进度条布局
    critical: (v) => v > 90,
    warning: (v) => v > 70,
    icon: "🧠",
  },
  {
    id: "fragmentation",
    label: "内存碎片率",
    unit: "%",
    key: "fragmentation_pct",
    historyKey: null,
    critical: (v) => v > 50,
    warning: (v) => v > 20,
    icon: "🧩",
  },
  {
    id: "cpu",
    label: "CPU 占用",
    unit: "%",
    key: "cpu_percent",
    historyKey: null,
    critical: (v) => v > 80,
    warning: (v) => v > 50,
    icon: "⚡",
  },
  {
    id: "latency",
    label: "响应时延 (P95)",
    unit: "ms",
    key: null,
    extract: (s) => s.latency?.p95_ms ?? 0,
    historyKey: null,
    critical: (v) => v > 2000,
    warning: (v) => v > 1000,
    icon: "⏱️",
  },
  {
    id: "accuracy",
    label: "高置信度占比",
    unit: "%",
    key: null,
    extract: (s) => s.accuracy?.high_conf_pct ?? null,
    historyKey: null,
    cutLine: ">0.8",
    critical: (v) => v < 70,
    warning: (v) => v < 85,
    icon: "🎯",
  },
  {
    id: "mtbf",
    label: "平均无故障时间",
    unit: "",
    key: "mtbf_formatted",
    historyKey: null,
    critical: () => false,
    warning: () => false,
    icon: "🛡️",
  },
  {
    id: "thermal",
    label: "温度",
    unit: "°C",
    key: "thermal_celsius",
    historyKey: null,
    critical: (v) => v > 85,
    warning: (v) => v > 70,
    icon: "🌡️",
  },
];

// 历史数据缓存 (用于 sparkline)
const monitorHistory = {};

for (const card of MONITOR_CARDS) {
  monitorHistory[card.id] = [];
}

// 构建监控卡片 DOM（只执行一次）
function buildMonitorCards() {
  if (!els.monitorGrid) return;
  els.monitorGrid.innerHTML = MONITOR_CARDS.map((card) => {
    if (card.template === "audio") {
      // 音频电平：音量条 + 状态标签
      return `
        <div class="monitor-card" id="monitor-${card.id}">
          <div class="monitor-card-header">
            <span class="monitor-card-icon">${card.icon}</span>
            <span class="monitor-card-label">${card.label}</span>
            <span class="monitor-audio-state" id="monitor-audio-state">待机</span>
          </div>
          <div class="monitor-audio-meter-wrap">
            <div class="monitor-audio-meter" id="monitor-audio-meter">
              <div class="monitor-audio-bar" id="monitor-audio-bar"></div>
              <div class="monitor-audio-peak" id="monitor-audio-peak"></div>
            </div>
            <span class="monitor-audio-value" id="monitor-audio-value">--</span>
          </div>
          <div class="monitor-card-footer" id="monitor-audio-footer">等待 ESP32 连接…</div>
        </div>
      `;
    }
    if (card.template) {
      // 自定义模版：内存占用 → 进度条布局
      return `
        <div class="monitor-card" id="monitor-${card.id}">
          <div class="monitor-card-header">
            <span class="monitor-card-icon">${card.icon}</span>
            <span class="monitor-card-label">${card.label}</span>
          </div>
          <div class="monitor-card-body">
            <span class="monitor-card-value" id="monitor-${card.id}-cur">--</span>
            <span class="monitor-card-bar-label" id="monitor-${card.id}-pct">--</span>
          </div>
          <div class="monitor-bar-wrap">
            <div class="monitor-bar-fill" id="monitor-${card.id}-bar"></div>
          </div>
          <div class="monitor-card-footer" id="monitor-${card.id}-target">/ 128 MB</div>
        </div>
      `;
    }
    return `
      <div class="monitor-card" id="monitor-${card.id}">
        <div class="monitor-card-header">
          <span class="monitor-card-icon">${card.icon}</span>
          <span class="monitor-card-label">${card.label}</span>
        </div>
        <div class="monitor-card-body">
          <span class="monitor-card-value">--</span>
          <span class="monitor-card-unit">${card.unit}</span>
        </div>
        <canvas class="monitor-sparkline" id="spark-${card.id}" width="160" height="28"></canvas>
      </div>
    `;
  }).join("");
}

// 更新监控数据
async function refreshMonitor() {
  try {
    const resp = await fetch("/api/monitor");
    const snapshot = await resp.json();

    // 更新运行环境标识
    const hostLabel = document.getElementById("monitor-host");
    if (hostLabel) {
      const h = snapshot.host_info || {};
      const env = h.is_router ? "🎯 路由器" : `💻 ${h.system}`;
      hostLabel.textContent = `${env} · ${h.node || "?"} · 目标 128MB`;
    }

    for (const card of MONITOR_CARDS) {
      // 音频电平卡片特殊渲染
      if (card.template === "audio") {
        const audio = snapshot.audio || {};
        const energy = audio.energy || 0;
        const levelPct = audio.level_pct || 0;
        const state = audio.state || "idle";

        const barEl = document.getElementById("monitor-audio-bar");
        const peakEl = document.getElementById("monitor-audio-peak");
        const valueEl = document.getElementById("monitor-audio-value");
        const stateEl = document.getElementById("monitor-audio-state");
        const footerEl = document.getElementById("monitor-audio-footer");
        const cardEl = document.getElementById("monitor-audio");

        if (barEl) barEl.style.width = Math.min(levelPct, 100) + "%";
        if (peakEl) peakEl.style.width = Math.min(levelPct * 1.3, 100) + "%";
        if (valueEl) valueEl.textContent = energy >= 0.001 ? energy.toFixed(4) : "--";

        // 状态文字和颜色
        const stateMap = {
          "LISTENING": { text: "监听中", cls: "listening" },
          "COLLECTING": { text: "录音中", cls: "recording" },
          "idle": { text: "待机", cls: "idle" },
        };
        const si = stateMap[state] || stateMap["idle"];
        if (stateEl) {
          stateEl.textContent = si.text;
          stateEl.className = "monitor-audio-state " + si.cls;
        }
        if (footerEl) {
          if (energy < 0.005) footerEl.textContent = "无信号 · 背景噪声";
          else if (energy < 0.03) footerEl.textContent = "微弱信号 · 等待语音";
          else if (energy < 0.10) footerEl.textContent = "中等信号 · 可能有人说话";
          else footerEl.textContent = "强信号 · 语音活动中";
        }

        // 卡片状态颜色
        if (cardEl) {
          cardEl.classList.remove("good", "warning", "critical");
          if (energy > 0.10) cardEl.classList.add("good");
          else if (energy > 0.02) cardEl.classList.add("warning");
        }
        continue;
      }

      // 自定义模版卡片的特殊渲染
      if (card.template && card.id === "ram") {
        const cur = snapshot.current_ram_mb ?? 0;
        const pct = snapshot.ram_usage_pct ?? 0;
        const target = snapshot.target_ram_mb ?? 128;
        const cardEl = document.getElementById("monitor-ram");
        const curEl = document.getElementById("monitor-ram-cur");
        const pctEl = document.getElementById("monitor-ram-pct");
        const barEl = document.getElementById("monitor-ram-bar");
        const tgtEl = document.getElementById("monitor-ram-target");

        if (curEl) curEl.textContent = cur.toFixed(0) + " MB";
        if (pctEl) pctEl.textContent = pct.toFixed(0) + "%";
        if (barEl) barEl.style.width = Math.min(pct, 100) + "%";
        if (tgtEl) tgtEl.textContent = "目标 " + target + " MB";

        // 状态颜色
        if (cardEl) {
          cardEl.classList.remove("good", "warning", "critical");
          if (pct > 90) cardEl.classList.add("critical");
          else if (pct > 70) cardEl.classList.add("warning");
          else cardEl.classList.add("good");
        }

        // Sparkline
        const ramCanvas = document.getElementById("spark-ram");
        if (ramCanvas) {
          const history = monitorHistory["ram"];
          history.push(pct);
          if (history.length > 60) history.shift();
          drawSparkline(ramCanvas, history);
        }
        continue;
      }

      const value = card.extract ? card.extract(snapshot) : (snapshot[card.key] ?? null);
      const cardEl = document.getElementById(`monitor-${card.id}`);
      const valueEl = cardEl?.querySelector(".monitor-card-value");
      const sparkCanvas = document.getElementById(`spark-${card.id}`);

      if (card.id === "thermal") {
        const src = snapshot.thermal_source;
        let footerEl = cardEl?.querySelector(".monitor-card-footer");
        if (!footerEl && cardEl) {
          footerEl = document.createElement("div");
          footerEl.className = "monitor-card-footer";
          cardEl.appendChild(footerEl);
        }
        if (footerEl) {
          footerEl.textContent = src === "esp32"
            ? "ESP32 外接传感器"
            : (src === "host" ? "本机 thermal" : "等待 ESP32 上报");
        }
      }

      // 更新数值
      if (valueEl) {
        if (value === null || value === undefined) {
          valueEl.textContent = "N/A";
        } else if (typeof value === "number") {
          valueEl.textContent = value % 1 === 0 ? value.toFixed(0) : value.toFixed(1);
        } else {
          valueEl.textContent = value;
        }
      }

      // 状态颜色
      if (cardEl && typeof value === "number") {
        cardEl.classList.remove("good", "warning", "critical");
        if (card.critical(value)) {
          cardEl.classList.add("critical");
        } else if (card.warning(value)) {
          cardEl.classList.add("warning");
        } else {
          cardEl.classList.add("good");
        }
      }

      // Sparkline
      if (sparkCanvas && typeof value === "number") {
        const history = monitorHistory[card.id];
        history.push(value);
        if (history.length > 60) history.shift();
        drawSparkline(sparkCanvas, history);
      }
    }
  } catch (err) {
    const hostLabel = document.getElementById("monitor-host");
    if (hostLabel) hostLabel.textContent = "监控加载失败，请刷新页面";
    console.warn("[monitor]", err);
  }
}

// 绘制微型折线图
function drawSparkline(canvas, data) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  if (data.length < 2) return;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const pad = 2;

  ctx.beginPath();
  ctx.strokeStyle = "rgba(110,168,254,0.7)";
  ctx.lineWidth = 1.2;
  ctx.lineJoin = "round";

  const step = (w - pad * 2) / (data.length - 1);
  data.forEach((v, i) => {
    const x = pad + i * step;
    const y = pad + (1 - (v - min) / range) * (h - pad * 2);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // 渐变填充
  ctx.lineTo(pad + (data.length - 1) * step, h - pad);
  ctx.lineTo(pad, h - pad);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, pad, 0, h - pad);
  grad.addColorStop(0, "rgba(110,168,254,0.18)");
  grad.addColorStop(1, "rgba(110,168,254,0.01)");
  ctx.fillStyle = grad;
  ctx.fill();
}

function applyDebugMode() {
  document.body.classList.toggle("debug-mode", DEBUG);
}

// ====== 初始化 ======
function boot() {
  applyDebugMode();
  applyDemoMode();
  buildMonitorCards();
  buildRouterCmdSelect();
  refreshDevices();
  refreshMonitor();
  setInterval(refreshDevices, 5000);
  setInterval(refreshMonitor, 2000);
  ensureSocket();
  const micHint = micAccessHint();
  if (micHint && els.chatEmpty) {
    els.chatEmpty.textContent = micHint + " 路由器指令请用下方文本框。";
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
