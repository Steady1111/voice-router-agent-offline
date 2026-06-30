// ESP32-S3 音频采集 + 设备执行节点固件：
// 服务端 KWS 检测唤醒词 → 经 WebSocket 发 start_record →
// ESP32 经 I2S 从 INMP441 采集 16kHz/16bit/单声道 PCM → 流式发往中枢 →
// 服务端 ASR/NLU 处理后发 stop_record → ESP32 回到 KWS 静默监听。
#include <Arduino.h>
#include <ArduinoJson.h>
#include <U8g2lib.h>
#include <WebSocketsClient.h>
#include <WiFi.h>
#include <driver/i2s.h>
#include <time.h>

#include "config.h"



namespace {

// ── 数码管硬件定时器扫描（250Hz ISR，与 main loop 完全解耦，永不闪烁）──
hw_timer_t* g_disp_timer = nullptr;
volatile uint8_t g_disp_digits_v[4] = {10, 10, 10, 10};  // ISR 读取的 digits 副本

constexpr uint8_t DISP_DASH  = 10;
constexpr uint8_t DISP_BLANK = 11;
constexpr uint8_t SEG_PINS[] = {DISP_SEG_A, DISP_SEG_B, DISP_SEG_C, DISP_SEG_D,
                                DISP_SEG_E, DISP_SEG_F, DISP_SEG_G, DISP_SEG_DP};
constexpr uint8_t DIG_PINS[] = {DISP_DIG1, DISP_DIG2, DISP_DIG3, DISP_DIG4};

// 段码 PROGMEM：位 0=a, 1=b, 2=c, 3=d, 4=e, 5=f, 6=g, 7=dp
const uint8_t SEG_PATTERNS[] PROGMEM = {
    0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F, 0x40,
};

// ISR：每 4ms 触发一次，扫描一位数码管（共阳：段 LOW=亮，位 HIGH=开）
void IRAM_ATTR dispTimerISR() {
    static uint8_t cur = 0;
    digitalWrite(DIG_PINS[cur], LOW);               // 关当前位
    cur = (cur + 1) % 4;                            // 下一位置
    for (int i = 0; i < 8; ++i) digitalWrite(SEG_PINS[i], HIGH); // 关所有段
    uint8_t v = g_disp_digits_v[cur];
    if (v <= 9) {
        uint8_t pat = pgm_read_byte(&SEG_PATTERNS[v]);
        for (int i = 0; i < 8; ++i)
            if (pat & (1 << i)) digitalWrite(SEG_PINS[i], LOW);
    } else if (v == DISP_DASH) {
        digitalWrite(DISP_SEG_G, LOW);
    }
    digitalWrite(DIG_PINS[cur], HIGH);              // 开新位
}

void dispSetNumber(uint16_t num) {
    if (num > 9999) num = 9999;
    g_disp_digits_v[0] = (num / 1000) % 10;
    g_disp_digits_v[1] = (num / 100) % 10;
    g_disp_digits_v[2] = (num / 10) % 10;
    g_disp_digits_v[3] = num % 10;
}

void dispShowDash() {
    for (int i = 0; i < 4; ++i) g_disp_digits_v[i] = DISP_DASH;
}

void dispInit() {
    for (int p = DISP_SEG_A; p <= DISP_SEG_DP; ++p) { pinMode(p, OUTPUT); digitalWrite(p, HIGH); }
    for (int i = 0; i < 4; ++i) { pinMode(DIG_PINS[i], OUTPUT); digitalWrite(DIG_PINS[i], LOW); }
    dispShowDash();
    // 硬件定时器：80 预分频 → 1MHz，每 4000 ticks (4ms) = 250Hz 扫描
    g_disp_timer = timerBegin(0, 80, true);
    timerAttachInterrupt(g_disp_timer, &dispTimerISR, true);
    timerAlarmWrite(g_disp_timer, 4000, true);
    timerAlarmEnable(g_disp_timer);
}

constexpr i2s_port_t I2S_PORT = I2S_NUM_0;
WebSocketsClient ws;
U8G2_SSD1306_128X64_NONAME_F_SW_I2C u8g2(U8G2_R0, OLED_SCL_PIN, OLED_SDA_PIN, U8X8_PIN_NONE);
volatile bool g_streaming = false;     // 录音模式（有倒计时，15s超时）
volatile bool g_kws_listening = false; // KWS静默监听模式（无倒计时，持续传音频）
unsigned long g_last_ping_ms = 0;
unsigned long g_last_debug_ms = 0;
int16_t g_frame_buf[FRAME_SAMPLES];

struct {
    bool     fan_on = false;
    uint8_t  fan_speed = 0;
    uint8_t  fan_speed_level = 0;
    bool     recording = false;
    unsigned long rec_start_ms = 0;
} g_disp;

uint8_t  g_vu_level = 0;
unsigned long g_boot_ms = 0;
unsigned long g_last_oled_ms = 0;
int g_date_year = 2026, g_date_mon = 6, g_date_mday = 1, g_date_wday = 0;
int g_date_hour = 0, g_date_min = 0;
unsigned long g_fan_led_until = 0;  // 风扇开启后蓝灯短暂提示，到期恢复绿灯

#if TEMP_SENSOR_ENABLE
// NTC 热敏电阻模块（AO/DO/GND/VCC），用 ADC 读 AO 脚电压
// B=3950, R0=10kΩ@25°C, 模块电压分压：NTC 接 GND，固定电阻接 VCC
constexpr float NTC_R0     = 10000.0f;   // 25°C 时热敏电阻值
constexpr float NTC_T0     = 298.15f;    // 25°C = 298.15K
constexpr float NTC_B      = 3950.0f;    // B 值
constexpr float NTC_R_FIXED = 10000.0f;  // 模块上固定分压电阻
constexpr float NTC_VCC    = 3.3f;       // 供电电压

float g_temp_c = NAN;
bool g_temp_ok = false;
unsigned long g_last_temp_read_ms = 0;
unsigned long g_last_temp_tx_ms = 0;
#endif

// ============== 以下函数无变动 ==============

void setStatusLed(uint8_t r, uint8_t g, uint8_t b) {
#if STATUS_LED_PIN >= 0
    neopixelWrite(STATUS_LED_PIN, r, g, b);
#else
    (void)r; (void)g; (void)b;
#endif
}

void configureI2S() {
    i2s_config_t cfg = {};
    cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
    cfg.sample_rate = SAMPLE_RATE_HZ;
    cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
    cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
    cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
    cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
    cfg.dma_buf_count = 6;
    cfg.dma_buf_len = FRAME_SAMPLES;
    cfg.use_apll = false;
    cfg.tx_desc_auto_clear = false;
    cfg.fixed_mclk = 0;
    i2s_pin_config_t pins = {};
    pins.bck_io_num = I2S_BCLK_PIN;
    pins.ws_io_num = I2S_LRCLK_PIN;
    pins.data_in_num = I2S_DIN_PIN;
    pins.data_out_num = I2S_PIN_NO_CHANGE;
    i2s_driver_install(I2S_PORT, &cfg, 0, nullptr);
    i2s_set_pin(I2S_PORT, &pins);
    i2s_zero_dma_buffer(I2S_PORT);
}

bool readFrameAsInt16() {
    static int32_t raw[FRAME_SAMPLES];
    size_t bytes_read = 0;
    esp_err_t err = i2s_read(I2S_PORT, raw, sizeof(raw), &bytes_read, pdMS_TO_TICKS(100));
    if (err != ESP_OK || bytes_read == 0) return false;
    size_t samples = bytes_read / sizeof(int32_t);
    for (size_t i = 0; i < samples; ++i) {
        int32_t s = raw[i] >> 16;
        if (s > 32767) s = 32767;
        else if (s < -32768) s = -32768;
        g_frame_buf[i] = (int16_t)s;
    }
    for (size_t i = samples; i < FRAME_SAMPLES; ++i) g_frame_buf[i] = 0;
    return true;
}

uint16_t estimateRpm(uint8_t level) {
    switch (level) {
        case 1: return 500;
        case 2: return 1100;
        case 3: return 1800;
        case 4: return 2600;
        case 5: return 3600;
        default: return 0;
    }
}

#if TEMP_SENSOR_ENABLE
void readTemperature() {
    // 多次采样取平均，减少 ADC 噪声
    constexpr int N = 16;
    long sum = 0;
    for (int i = 0; i < N; i++) {
        sum += analogRead(TEMP_ONEWIRE_PIN);
        delay(2);
    }
    float avg = (float)sum / (float)N;
    // ADC 12bit: 0-4095 → 0-VCC
    float v = avg * NTC_VCC / 4095.0f;
    // 电压分压：NTC 在下，固定电阻在上 → R_ntc = R_fixed * V / (VCC - V)
    if (v < 0.05f || v > (NTC_VCC - 0.05f)) {
        g_temp_ok = false;
        return;
    }
    float r_ntc = NTC_R_FIXED * v / (NTC_VCC - v);
    // B 参数方程: 1/T = 1/T0 + (1/B) * ln(R/R0)
    float temp_k = 1.0f / (1.0f / NTC_T0 + logf(r_ntc / NTC_R0) / NTC_B);
    g_temp_c = temp_k - 273.15f;
    g_temp_ok = (g_temp_c > -20.0f && g_temp_c < 100.0f);
}

void sendTemperatureTelemetry() {
    if (!g_temp_ok || !ws.isConnected()) return;
    char out[72];
    snprintf(out, sizeof(out),
             "{\"event\":\"telemetry\",\"temperature_c\":%.1f}", g_temp_c);
    ws.sendTXT(out);
}
#endif

void updateOled() {
    u8g2.clearBuffer();
    char buf[32];
    int y = 0;
    struct tm ti;
    if (getLocalTime(&ti)) {
        g_date_year = ti.tm_year + 1900;
        g_date_mon  = ti.tm_mon + 1;
        g_date_mday = ti.tm_mday;
        g_date_wday = ti.tm_wday;
        g_date_hour = ti.tm_hour;
        g_date_min  = ti.tm_min;
    }
    const char* weekdays[] = {"周日","周一","周二","周三","周四","周五","周六"};
    u8g2.setFont(u8g2_font_5x7_tf);
    snprintf(buf, sizeof(buf), "%02d-%02d %02d:%02d %s",
             g_date_mon, g_date_mday, g_date_hour, g_date_min, weekdays[g_date_wday]);
    int date_w = u8g2.getStrWidth(buf);
    u8g2.drawStr((128 - date_w) / 2, 7, buf);
    unsigned long uptime_sec = (millis() - g_boot_ms) / 1000;
    unsigned long h = uptime_sec / 3600;
    unsigned long m = (uptime_sec % 3600) / 60;
    unsigned long s = uptime_sec % 60;
    y = 18;
    u8g2.setFont(u8g2_font_6x10_tf);
    if (g_streaming) {
        unsigned long rec = (millis() - g_disp.rec_start_ms) / 1000;
        snprintf(buf, sizeof(buf), "REC %02lu:%02lu", rec / 60, rec % 60);
        u8g2.drawStr(0, y, buf);
        int bx = 70, bw_max = 56;
        int bw = g_vu_level * bw_max / 20;
        if (bw < 1 && g_vu_level > 0) bw = 1;
        if (bw > 0) u8g2.drawBox(bx, y - 8, bw, 8);
        u8g2.drawFrame(bx, y - 8, bw_max, 8);
    } else {
        snprintf(buf, sizeof(buf), "VR %02lu:%02lu:%02lu", h, m, s);
        u8g2.drawStr(0, y, buf);
    }
    y += 11;
    u8g2.setFont(u8g2_font_5x7_tf);
    snprintf(buf, sizeof(buf), "WiFi:%ddBm  %s",
             WiFi.RSSI(), ws.isConnected() ? "WS:OK" : "WS:DOWN");
    u8g2.drawStr(0, y, buf);
    y += 9;
    u8g2.drawStr(0, y, WiFi.localIP().toString().c_str());
    y += 9;
    u8g2.setFont(u8g2_font_6x10_tf);
    if (g_disp.fan_on && g_disp.fan_speed_level > 0) {
        uint16_t rpm = estimateRpm(g_disp.fan_speed_level);
        snprintf(buf, sizeof(buf), "Fan:Lv%d ~%uRPM", g_disp.fan_speed_level, rpm);
    } else {
        snprintf(buf, sizeof(buf), "Fan: OFF");
    }
    u8g2.drawStr(0, y, buf);
    y += 11;
#if TEMP_SENSOR_ENABLE
    if (g_temp_ok) {
        snprintf(buf, sizeof(buf), "Temp: %.1fC", g_temp_c);
    } else {
        snprintf(buf, sizeof(buf), "Temp: --.-C");
    }
    u8g2.drawStr(0, y, buf);
#endif
    u8g2.sendBuffer();
}

void connectWifi() {
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);

    // 多次扫描，因为 iPhone 热点有时扫描不到
    bool found = false;
    int scan_n = 0;
    for (int scan_round = 0; scan_round < 3 && !found; scan_round++) {
        if (scan_round > 0) delay(1000);
        unsigned long t0 = millis();
        int n = WiFi.scanNetworks(false, true);  // async=false, show_hidden=true
        unsigned long t1 = millis();
        scan_n = n;
        Serial.printf("WiFi: scan #%d took %lums, found %d networks\n", scan_round, t1-t0, n);
        for (int i = 0; i < n; i++) {
            wifi_auth_mode_t auth = WiFi.encryptionType(i);
            if (WiFi.SSID(i) == WIFI_SSID) {
                found = true;
                Serial.printf("  >>> FOUND: %s  (%d dBm) ch=%d\n",
                    WiFi.SSID(i).c_str(), WiFi.RSSI(i), WiFi.channel(i));
            }
        }
        if (!found) {
            // 如果没找到目标网络，只打印摘要
            Serial.printf("  (SSID '%s' not in scan)\n", WIFI_SSID);
        }
    }
    if (!found) {
        Serial.printf("WiFi: SSID '%s' NOT FOUND after %d scans\n", WIFI_SSID, 3);
    }

    // 无论扫描结果如何，都尝试连接
    Serial.printf("WiFi: connecting to %s...\n", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    setStatusLed(8, 8, 0);

    int retry = 0;
    int last_status = -1;
    while (WiFi.status() != WL_CONNECTED) {
        delay(250);
        retry++;
        int st = (int)WiFi.status();
        if (st != last_status) {
            Serial.printf("\n  [%ds] status=%d", retry/4, st);
            last_status = st;
        } else {
            Serial.print(".");
        }
        if (retry > 240) {  // 60 秒超时
            int st2 = (int)WiFi.status();
            Serial.printf("\nWiFi FAIL after 60s status=%d\n", st2);
            delay(2000);
            ESP.restart();
        }
    }
    Serial.printf("\nWiFi ok, ip=%s, rssi=%d\n",
        WiFi.localIP().toString().c_str(), WiFi.RSSI());
    configTime(8 * 3600, 0, "ntp.aliyun.com", "pool.ntp.org", "time.apple.com");
    Serial.print("NTP syncing...");
    struct tm ti;
    int ntp_retry = 0;
    while (!getLocalTime(&ti) && ntp_retry < 40) {
        delay(250);
        Serial.print(".");
        ntp_retry++;
    }
    if (ntp_retry < 40) {
        Serial.printf("\nNTP ok: %04d-%02d-%02d %02d:%02d:%02d\n",
            ti.tm_year + 1900, ti.tm_mon + 1, ti.tm_mday,
            ti.tm_hour, ti.tm_min, ti.tm_sec);
    } else {
        Serial.println("\nNTP timeout, using default date");
    }
}

// 前置声明（解决函数间互相调用的编译顺序问题）
void startKwsListening();
void stopKwsListening();
void startRecording();
void stopRecording();

void sendStartEvent() {
    JsonDocument doc;
    doc["event"] = "start";
    doc["sample_rate"] = SAMPLE_RATE_HZ;
    doc["format"] = "pcm_s16le";
    char out[128];
    size_t n = serializeJson(doc, out, sizeof(out));
    ws.sendTXT(out, n);
}

void sendStopEvent() {
    ws.sendTXT("{\"event\":\"stop\"}");
}

void sendFanAck(bool on, int speedLevel) {
    char out[96];
    snprintf(out, sizeof(out),
             "{\"event\":\"device_state\",\"device_type\":\"fan\","
             "\"state\":{\"on\":%s,\"level\":%d}}",
             on ? "true" : "false", speedLevel);
    ws.sendTXT(out);
}

void startKwsListening() {
    g_disp.recording = false;
    stopRecording();  // 先退出录音模式
    setStatusLed(0, 8, 0);  // 绿灯：KWS 监听中（必须在 return 前）
    dispShowDash();          // 显示 ---- 待机
    if (g_kws_listening) return;
    g_kws_listening = true;
    Serial.println("kws listening…");
}

void stopKwsListening() {
    if (!g_kws_listening) return;
    g_kws_listening = false;
    Serial.println("kws listening stopped");
}

void startRecording() {
    stopKwsListening();  // 先退出 KWS 监听模式
    if (g_streaming) return;
    g_streaming = true;
    g_disp.recording = true;
    g_disp.rec_start_ms = millis();
    if (ws.isConnected()) sendStartEvent();
    setStatusLed(16, 0, 0);
    digitalWrite(LED_PIN, HIGH);
    Serial.println("recording (wake-triggered)…");
}

void stopRecording() {
    if (!g_streaming) return;
    g_streaming = false;
    g_disp.recording = false;
    if (ws.isConnected()) sendStopEvent();
    bool conn = ws.isConnected();
    setStatusLed(conn ? 0 : 8, conn ? 8 : 0, 0);
    digitalWrite(LED_PIN, LOW);
    Serial.println(conn ? "…sent" : "…stopped (ws offline)");
}

void onWsEvent(WStype_t type, uint8_t* payload, size_t length) {
    switch (type) {
        case WStype_CONNECTED:
            Serial.println("ws connected");
            setStatusLed(0, 8, 0);
            break;
        case WStype_DISCONNECTED:
            Serial.println("ws disconnected");
            setStatusLed(8, 0, 0);
            break;
        case WStype_TEXT: {
            Serial.printf("ws < %.*s\n", (int)length, payload);
            JsonDocument doc;
            DeserializationError err = deserializeJson(doc, payload, length);
            if (err) break;
            const char* event = doc["event"];
            if (!event) break;
            if (strcmp(event, "start_kws") == 0) {
                startKwsListening();
            } else if (strcmp(event, "stop_kws") == 0) {
                stopKwsListening();
            } else if (strcmp(event, "start_record") == 0) {
                startRecording();
            } else if (strcmp(event, "stop_record") == 0) {
                stopRecording();
            } else if (strcmp(event, "wake_detected") == 0) {
                g_disp.recording = true;
                g_disp.rec_start_ms = millis();
                g_streaming = true;  // ← 必须设，否则 15s 超时不会触发
                setStatusLed(16, 0, 0);
                digitalWrite(LED_PIN, HIGH);
                Serial.println("wake detected → red timer, say command");
            } else if (strcmp(event, "reply") == 0) {
                const char* text = doc["text"] | "";
                Serial.printf("reply: %s\n", text);
                g_disp.recording = false;
                g_streaming = false; // ← 同步重置
                setStatusLed(0, 8, 0);
                digitalWrite(LED_PIN, LOW);
                dispShowDash();
            } else if (strcmp(event, "fan") == 0) {
                const char* action = doc["action"] | "off";
                int speed = doc["speed"] | 255;
                int speedLevel = doc["speed_level"] | 3;
                Serial.printf("[FAN DEBUG] Received: action=%s speed=%d\n", action, speed);
                Serial.printf("[FAN DEBUG] Before: FAN_PIN=%d, PWM ch0=%d\n",
                              digitalRead(FAN_PIN), ledcRead(0));
                digitalWrite(TB6612_STBY_PIN, HIGH);
                if (strcmp(action, "on") == 0) {
                    if (speed < 102) speed = 102;
                    digitalWrite(FAN_PIN, HIGH);
                    ledcWrite(0, speed);
                    g_disp.fan_on = true;
                    g_disp.fan_speed = (uint8_t)speed;
                    g_disp.fan_speed_level = (uint8_t)speedLevel;
                    g_fan_led_until = millis() + 1500;
                    setStatusLed(0, 0, 16);
                    Serial.printf("[FAN DEBUG] After ON: FAN_PIN=%d, PWM ch0=%d\n",
                                  digitalRead(FAN_PIN), ledcRead(0));
                    Serial.printf("Fan ON  speed=%d level=%d\n", speed, speedLevel);
                    sendFanAck(true, speedLevel);
                } else {
                    digitalWrite(FAN_PIN, LOW);
                    ledcWrite(0, 0);
                    g_disp.fan_on = false;
                    g_disp.fan_speed = 0;
                    g_disp.fan_speed_level = 0;
                    g_fan_led_until = 0;
                    setStatusLed(0, 8, 0);
                    Serial.printf("[FAN DEBUG] After OFF: FAN_PIN=%d, PWM ch0=%d\n",
                                  digitalRead(FAN_PIN), ledcRead(0));
                    Serial.println("Fan OFF");
                    sendFanAck(false, 0);
                }
            }
            break;
        }
        case WStype_BIN:
            Serial.printf("ws bin %u bytes\n", (unsigned)length);
            break;
        default:
            break;
    }
}

}  // namespace

void setup() {
    Serial.begin(115200);
    delay(200);
    Serial.println("voice-router-agent firmware boot v4 (timer-disp)");
    g_boot_ms = millis();

    setStatusLed(0, 0, 8);
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    pinMode(FAN_PIN, OUTPUT);
    digitalWrite(FAN_PIN, LOW);
    ledcSetup(0, 5000, 8);
    ledcAttachPin(FAN_PWM_PIN, 0);
    ledcWrite(0, 0);
    pinMode(TB6612_STBY_PIN, OUTPUT);
    digitalWrite(TB6612_STBY_PIN, HIGH);

    dispInit();

#if TEMP_SENSOR_ENABLE
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);  // 0-3.3V 量程
    readTemperature();
    Serial.printf("Temp: NTC on GPIO %d (ADC), first=%.1fC ok=%d\n",
                  TEMP_ONEWIRE_PIN, g_temp_ok ? g_temp_c : -999.0f, g_temp_ok);
#endif

    // 启动自检：逐位亮「0」
    const uint8_t testDigs[] = {DISP_DIG1, DISP_DIG2, DISP_DIG3, DISP_DIG4};
    for (int i = 0; i < 4; i++) {
        timerAlarmDisable(g_disp_timer);  // 暂停 ISR，避免干扰
        for (int j = 0; j < 4; j++) digitalWrite(testDigs[j], LOW);
        uint8_t pat = 0x3F;
        for (int s = 0; s < 7; s++)
            digitalWrite(10 + s, (pat & (1 << s)) ? LOW : HIGH);
        digitalWrite(testDigs[i], HIGH);
        Serial.printf("SEG-TEST: digit[%d] pin=%d\n", i, testDigs[i]);
        delay(500);
    }
    for (int j = 0; j < 4; j++) digitalWrite(testDigs[j], LOW);
    timerAlarmEnable(g_disp_timer);  // 恢复 ISR 扫描

    configureI2S();

    u8g2.setI2CAddress(OLED_I2C_ADDR * 2);
    Serial.printf("OLED: trying addr 0x%02X (bus: 0x%02X)...\n", OLED_I2C_ADDR, OLED_I2C_ADDR * 2);
    if (!u8g2.begin()) {
        const uint8_t alt = 0x3D;
        Serial.printf("OLED: trying alt addr 0x%02X (bus: 0x%02X)...\n", alt, alt * 2);
        u8g2.setI2CAddress(alt * 2);
        if (!u8g2.begin()) {
            Serial.println("OLED FAIL - check wiring (SDA->41, SCL->42)");
        } else {
            Serial.println("OLED OK (alt addr 0x3D)");
            u8g2.setContrast(128);
            updateOled();
        }
    } else {
        Serial.println("OLED OK");
        u8g2.setContrast(128);
        updateOled();
    }

    pinMode(BUTTON_PIN, INPUT_PULLUP);

    connectWifi();

    ws.begin(HUB_HOST, HUB_PORT, HUB_WS_PATH);
    ws.onEvent(onWsEvent);
    ws.setReconnectInterval(2000);
}

void loop() {
    // 0. 录音超时保护：15 秒自动停止（仅录音模式）
    if (g_streaming && (millis() - g_disp.rec_start_ms > 15000)) {
        Serial.println("recording timeout, auto-stop");
        stopRecording();
        startKwsListening();  // 自动回到 KWS 监听模式
    }

    // 1b. 风扇蓝灯提示到期 → 恢复 KWS 绿灯
    if (g_fan_led_until > 0 && millis() > g_fan_led_until) {
        g_fan_led_until = 0;
        if (g_kws_listening && ws.isConnected()) {
            setStatusLed(0, 8, 0);
        }
    }

    // 2. 更新数码管显示内容（ISR 自动扫描，main loop 只管数据）
    if (g_disp.recording) {
        unsigned long elapsed = (millis() - g_disp.rec_start_ms) / 1000;
        dispSetNumber((uint16_t)elapsed);
    } else if (g_disp.fan_on && g_disp.fan_speed > 0) {
        uint16_t rpm = estimateRpm(g_disp.fan_speed_level);
        dispSetNumber(rpm);
    } else {
        dispShowDash();  // KWS 监听中或空闲：显示 ----
    }

    // 2. WebSocket 轮询（多发 BIN 时也要及时收控制指令）
    for (int i = 0; i < 4; i++) ws.loop();

    // 3. 应用层心跳
    if (ws.isConnected() && millis() - g_last_ping_ms > 15000) {
        g_last_ping_ms = millis();
        ws.sendTXT("{\"event\":\"ping\"}");
    }

    // 4. OLED 刷新（每 500ms）
    {
        unsigned long now = millis();
        if (now - g_last_oled_ms > 500) {
            g_last_oled_ms = now;
            updateOled();
        }
    }

#if TEMP_SENSOR_ENABLE
    // 4b. 温度采样与上报
    {
        unsigned long now = millis();
        if (now - g_last_temp_read_ms > 2000) {
            g_last_temp_read_ms = now;
            readTemperature();
        }
        if (g_temp_ok && now - g_last_temp_tx_ms > 5000) {
            g_last_temp_tx_ms = now;
            sendTemperatureTelemetry();
        }
    }
#endif

    // 5. 串口调试输出（每 5 秒）
    {
        unsigned long now = millis();
        if (now - g_last_debug_ms > 5000) {
            g_last_debug_ms = now;
            Serial.printf("DEBUG: streaming=%d kws=%d ws=%d fan=%d vu=%d\n",
                g_streaming, g_kws_listening, ws.isConnected(), g_disp.fan_on, g_vu_level);
#if TEMP_SENSOR_ENABLE
            Serial.printf("DEBUG: temp=%.1fC ok=%d\n", g_temp_c, g_temp_ok);
#endif
            Serial.printf("DEBUG: DISP fan_on=%d fan_speed=%d rpm=%d digits=[%d%d%d%d]\n",
                g_disp.fan_on, g_disp.fan_speed, estimateRpm(g_disp.fan_speed_level),
                g_disp_digits_v[0], g_disp_digits_v[1], g_disp_digits_v[2], g_disp_digits_v[3]);
        }
    }

    // 6. 音频发送：每轮尽量排空 I2S DMA，避免 main loop 其它任务导致丢帧
    if ((g_streaming || g_kws_listening) && ws.isConnected()) {
        int drained = 0;
        while (drained < 16 && readFrameAsInt16()) {
            int32_t peak = 0;
            for (int i = 0; i < FRAME_SAMPLES; i++) {
                int32_t a = abs((int32_t)g_frame_buf[i]);
                if (a > peak) peak = a;
            }
            g_vu_level = 0;
            if (peak > 0) {
                g_vu_level = (uint8_t)(20.0f * log10f(1.0f + peak) / log10f(1.0f + 32768.0f));
                if (g_vu_level > 20) g_vu_level = 20;
            }
            ws.sendBIN((uint8_t*)g_frame_buf, sizeof(g_frame_buf));
            drained++;
            if ((drained & 3) == 0) ws.loop();
        }
        ws.loop();
    } else {
        delay(2);
    }

    // 7. 风扇状态调试：每 5 秒检查一次风扇引脚状态
    {
        static unsigned long last_fan_check = 0;
        unsigned long now = millis();
        if (now - last_fan_check > 5000) {
            last_fan_check = now;
            int stby = digitalRead(TB6612_STBY_PIN);
            Serial.printf("[FAN DEBUG] Loop check: FAN_PIN=%d, PWM ch0=%d, STBY=%d, fan_on=%d, fan_speed=%d\n",
                          digitalRead(FAN_PIN), ledcRead(0), stby, g_disp.fan_on, g_disp.fan_speed);
            if (stby != HIGH) {
                Serial.printf("[FAN DEBUG] WARNING: STBY pin is %d, setting to HIGH\n", stby);
                digitalWrite(TB6612_STBY_PIN, HIGH);
            }
        }
    }

    // 8. 按键唤醒（预研：GPIO1 按下触发，比对着话筒喊更可靠）
    {
        static bool last_btn = false;
        bool btn = digitalRead(BUTTON_PIN) == LOW;
        if (btn && !last_btn && ws.isConnected() && g_kws_listening) {
            ws.sendTXT("{\"event\":\"button_wake\"}");
            Serial.println("button wake sent");
        }
        last_btn = btn;
    }
}
