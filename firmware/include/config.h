#pragma once

// WiFi
#define WIFI_SSID        "duanduaniPhone"
#define WIFI_PASSWORD    "12345679u"

// 中枢服务（你的 Mac）
#define HUB_HOST         "172.20.10.6"
#define HUB_PORT         28080
#define HUB_WS_PATH      "/ws/audio?role=esp32"

// 音频与 I2S 引脚
#define SAMPLE_RATE_HZ   16000
#define I2S_BCLK_PIN     4   // 位时钟 BCLK ← INMP441 SCK
#define I2S_LRCLK_PIN    5   // 帧时钟 LRCLK/WS ← INMP441 WS
#define I2S_DIN_PIN      6   // 数据输入 DIN ← INMP441 SD

// 按键（内部上拉，按下为低电平）
// GPIO 0 (BOOT) 在该开发板上始终被拉低，改用 GPIO 1
#define BUTTON_PIN       1

// 板载 RGB 状态灯引脚
#define STATUS_LED_PIN   48

// 外部 LED（发光二极管，长脚接 GPIO 7，短脚接 GND，中间串 220Ω-1KΩ 限流电阻）
#define LED_PIN          7

// 风扇控制（TB6612 电机驱动）
#define FAN_PIN          8   // AIN1：HIGH=转，LOW=停
#define FAN_PWM_PIN      9   // PWMA：PWM 调速（0-255），不用调速时接 3V3
#define TB6612_STBY_PIN  3   // STBY：必须接 HIGH 才能工作，悬空会随机待机导致电机停转

// 四位数码管（0.56" 4位共阳，直接 GPIO 驱动，动态扫描）
// 段引脚（均串 470Ω 限流电阻到数码管对应段）
#define DISP_SEG_A       10
#define DISP_SEG_B       11
#define DISP_SEG_C       12
#define DISP_SEG_D       13
#define DISP_SEG_E       14
#define DISP_SEG_F       15
#define DISP_SEG_G       16
#define DISP_SEG_DP      17
// 位选引脚（共阴：低电平选中该位）
#define DISP_DIG1        18
#define DISP_DIG2        19
#define DISP_DIG3        20
#define DISP_DIG4        21

// 每帧 PCM 的采样点数（16kHz 下约 32ms）
#define FRAME_SAMPLES    512

// 0.96" OLED 显示屏（I2C，SSD1306 128x64）
#define OLED_SDA_PIN     41
#define OLED_SCL_PIN     42
// I2C 地址通常是 0x3C（有些模块是 0x3D）
#define OLED_I2C_ADDR    0x3C
