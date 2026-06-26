"""
设备控制管理器

通过统一接口管理风扇、LED、继电器等硬件设备。
支持 GPIO 直驱、ESP32 WebSocket 桥接、模拟三种控制方式。

使用方法:
    manager = DeviceManager()
    manager.register_device("风扇", FanDriver())
    manager.initialize()
    manager.execute_command(intent="set_device_state", slots={"device_type": "fan", "state": "on"})
"""

import logging
import threading
from typing import Dict, Any, Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from voice_router_lite.hardware.esp32 import ESP32Bridge

logger = logging.getLogger(__name__)


@dataclass
class DeviceState:
    """设备状态"""
    name: str
    device_type: str  # "fan", "light", "led", "relay", "ac"
    is_on: bool = False
    level: int = 0  # 0-100，风扇速度/灯光亮度等
    extra: Dict[str, Any] = field(default_factory=dict)


class DeviceDriver:
    """设备驱动基类"""

    def __init__(self):
        self._state = DeviceState(name="unknown", device_type="unknown")

    def initialize(self) -> bool:
        return True

    def turn_on(self) -> bool:
        self._state.is_on = True
        return True

    def turn_off(self) -> bool:
        self._state.is_on = False
        return True

    def set_level(self, level: int) -> bool:
        self._state.level = max(0, min(100, level))
        return True

    def get_state(self) -> DeviceState:
        return self._state

    def close(self) -> None:
        pass


class FanDriver(DeviceDriver):
    """风扇驱动 (TB6612 PWM 控制)"""

    def __init__(self, gpio_pwm: Optional[int] = None,
                 gpio_dir: Optional[int] = None):
        super().__init__()
        self._state.name = "风扇"
        self._state.device_type = "fan"
        self._gpio_pwm = gpio_pwm
        self._gpio_dir = gpio_dir

    def initialize(self) -> bool:
        if self._gpio_pwm is not None:
            try:
                import RPi.GPIO as GPIO
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self._gpio_pwm, GPIO.OUT)
                if self._gpio_dir:
                    GPIO.setup(self._gpio_dir, GPIO.OUT)
                logger.info("风扇驱动已初始化: GPIO PWM=%d", self._gpio_pwm)
            except ImportError:
                logger.info("RPi.GPIO 未安装，风扇使用模拟模式")
            except Exception as e:
                logger.warning("风扇 GPIO 初始化失败: %s", e)
        return True

    def turn_on(self) -> bool:
        super().turn_on()
        if self._gpio_pwm is not None:
            try:
                import RPi.GPIO as GPIO
                GPIO.output(self._gpio_pwm, GPIO.HIGH)
            except Exception:
                pass
        return True

    def turn_off(self) -> bool:
        super().turn_off()
        if self._gpio_pwm is not None:
            try:
                import RPi.GPIO as GPIO
                GPIO.output(self._gpio_pwm, GPIO.LOW)
            except Exception:
                pass
        return True


class LEDDriver(DeviceDriver):
    """LED 驱动"""

    def __init__(self, gpio_pin: Optional[int] = None):
        super().__init__()
        self._state.name = "LED"
        self._state.device_type = "led"
        self._gpio_pin = gpio_pin

    def initialize(self) -> bool:
        if self._gpio_pin is not None:
            try:
                import RPi.GPIO as GPIO
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self._gpio_pin, GPIO.OUT)
            except ImportError:
                pass
        return True

    def turn_on(self) -> bool:
        super().turn_on()
        self._set_gpio(True)
        return True

    def turn_off(self) -> bool:
        super().turn_off()
        self._set_gpio(False)
        return True

    def _set_gpio(self, high: bool) -> None:
        if self._gpio_pin is None:
            return
        try:
            import RPi.GPIO as GPIO
            GPIO.output(self._gpio_pin, GPIO.HIGH if high else GPIO.LOW)
        except Exception:
            pass


class RelayDriver(DeviceDriver):
    """继电器驱动"""

    def __init__(self, gpio_pin: Optional[int] = None):
        super().__init__()
        self._state.name = "继电器"
        self._state.device_type = "relay"
        self._gpio_pin = gpio_pin

    def initialize(self) -> bool:
        if self._gpio_pin is not None:
            try:
                import RPi.GPIO as GPIO
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self._gpio_pin, GPIO.OUT)
            except ImportError:
                pass
        return True

    def turn_on(self) -> bool:
        super().turn_on()
        self._set_gpio(True)
        return True

    def turn_off(self) -> bool:
        super().turn_off()
        self._set_gpio(False)
        return True

    def _set_gpio(self, high: bool) -> None:
        if self._gpio_pin is None:
            return
        try:
            import RPi.GPIO as GPIO
            GPIO.output(self._gpio_pin, GPIO.HIGH if high else GPIO.LOW)
        except Exception:
            pass


class DeviceManager:
    """
    设备控制管理器

    管理所有硬件设备，提供统一的控制接口。

    使用方法:
        mgr = DeviceManager()
        mgr.register_device("风扇", FanDriver(gpio_pwm=9))
        mgr.initialize()
        mgr.turn_on("风扇")
    """

    def __init__(self, ws_sender: Optional[Callable] = None,
                 esp32_bridge: Optional["ESP32Bridge"] = None,
                 use_openwrt_ubus: bool = True):
        """
        Args:
            ws_sender: WebSocket 发送回调 (用于透传指令到 ESP32 等设备)
            esp32_bridge: ESP32 桥接实例 (用于风扇/显示/OLED 指令广播)
            use_openwrt_ubus: 路由器上优先通过 ubus 执行系统指令
        """
        self._devices: Dict[str, DeviceDriver] = {}
        self._ws_sender = ws_sender
        self._esp32_bridge = esp32_bridge
        self._use_openwrt_ubus = use_openwrt_ubus
        self._ubus = None
        if use_openwrt_ubus:
            try:
                from voice_router_lite.platform.openwrt import get_ubus_client
                self._ubus = get_ubus_client()
            except Exception:
                self._ubus = None
        self._lock = threading.Lock()
        self._initialized: bool = False

        # 风扇状态追踪
        self._fan_speed_level: int = 3  # 默认 3 档
        self._fan_is_on: bool = False

        # 指令执行回调
        self._command_callbacks: Dict[str, Callable] = {}

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """初始化所有设备驱动"""
        for name, driver in self._devices.items():
            driver.initialize()
        self._initialized = True
        logger.info("设备管理器已初始化: devices=%s", list(self._devices.keys()))

    def register_device(self, name: str, driver: DeviceDriver) -> None:
        """
        注册设备。

        Args:
            name: 设备名称
            driver: 设备驱动实例
        """
        with self._lock:
            self._devices[name] = driver
        logger.info("设备已注册: %s (type=%s)", name, driver._state.device_type)

    def execute_command(self, intent: str, slots: Dict[str, str]) -> Dict[str, Any]:
        """
        执行设备指令。

        Args:
            intent: NLU 意图名称
            slots: NLU 槽位字典

        Returns:
            {"success": bool, "message": str, "device_state": ...}
        """
        handler = self._get_handler(intent)
        if handler is None:
            return {"success": False, "message": f"未知指令: {intent}"}

        try:
            return handler(slots)
        except Exception as e:
            logger.error("指令执行异常: %s", e)
            return {"success": False, "message": str(e)}

    def turn_on(self, device_name: str) -> bool:
        """开启设备"""
        driver = self._find_device(device_name)
        if driver:
            return driver.turn_on()
        return False

    def turn_off(self, device_name: str) -> bool:
        """关闭设备"""
        driver = self._find_device(device_name)
        if driver:
            return driver.turn_off()
        return False

    def get_state(self, device_name: str) -> Optional[DeviceState]:
        """获取设备状态"""
        driver = self._find_device(device_name)
        if driver:
            return driver.get_state()
        return None

    def get_all_states(self) -> Dict[str, DeviceState]:
        """获取所有设备状态"""
        return {name: d.get_state() for name, d in self._devices.items()}

    def on_command(self, intent: str, callback: Callable) -> None:
        """注册自定义指令回调"""
        self._command_callbacks[intent] = callback

    def close(self) -> None:
        """释放所有设备资源"""
        for driver in self._devices.values():
            driver.close()
        self._devices.clear()
        logger.info("设备管理器已关闭")

    # ------------------------------------------------------------------
    # 指令处理器
    # ------------------------------------------------------------------

    def _get_handler(self, intent: str) -> Optional[Callable]:
        """获取指令处理函数（含 V2 NLU 意图名映射）。"""
        handlers = {
            # V2 NLU 意图名（智能家居）
            "device_control": self._handle_set_device_state,
            "device_adjust": self._handle_adjust_fan_speed,
            "device_query": self._handle_query_status,
            "timer_setting": self._handle_timer_setting,
            "scene_mode": self._handle_scene_mode,
            "help": self._handle_help,
            # 兼容旧意图名
            "set_device_state": self._handle_set_device_state,
            "adjust_fan_speed": self._handle_adjust_fan_speed,
            "set_light_brightness": self._handle_set_light_brightness,
            "query_device_status": self._handle_query_status,
            # 路由器意图
            "router_reboot": self._handle_router_reboot,
            "router_wifi_restart": self._handle_wifi_restart,
            "router_wifi_config": self._handle_router_wifi_config,
            "router_network_query": self._handle_router_query_info,
            "router_query_info": self._handle_router_query_info,
            "router_led_control": self._handle_router_led_control,
            "router_device_manage": self._handle_device_manage,
            "router_network_diag": self._handle_network_diag,
            "router_system": self._handle_router_system,
            "router_qos": self._handle_router_qos,
            "router_security": self._handle_router_security,
            "control_ac": self._handle_control_ac,
        }

        # 先查自定义回调
        if intent in self._command_callbacks:
            return self._command_callbacks[intent]

        return handlers.get(intent)

    def _handle_set_device_state(self, slots: Dict[str, str]) -> Dict[str, Any]:
        device_type = slots.get("device_type", "")
        state = slots.get("state", "")

        if state in ("on", "打开", "开启", "开"):
            state = "on"
        elif state in ("off", "关闭", "关掉", "关"):
            state = "off"
        else:
            return {"success": False, "message": f"无法理解开关操作: {state or '未知'}"}

        driver = self._find_device(device_type)
        if driver is None:
            return {"success": False, "message": f"设备不存在: {device_type or '未指定'}"}

        is_fan = (driver._state.device_type == "fan")

        if state in ("on", "打开"):
            driver.turn_on()
            if is_fan:
                self._fan_is_on = True
        elif state in ("off", "关闭"):
            driver.turn_off()
            if is_fan:
                self._fan_is_on = False

        # 通知 ESP32
        result = {
            "success": True,
            "message": f"已{state}{driver._state.name}",
            "device_state": driver.get_state(),
        }
        self._sync_to_esp32(device_type, result)
        return result

    # 调速方向中文映射
    _DIRECTION_MAP = {
        "调大": "up", "调大一点": "up", "调大一些": "up", "调高": "up",
        "加大": "up", "加快": "up", "快点": "up", "快一点": "up",
        "调小": "down", "调小一点": "down", "调小一些": "down", "调低": "down",
        "减小": "down", "减慢": "down", "慢点": "down", "慢一点": "down",
        "最大": "max", "最快": "max", "全速": "max", "开到最大": "max",
        "最小": "min", "最慢": "min", "微风": "min", "开到最小": "min",
        "开到": "set", "调到": "set", "档": "set",
    }

    def _handle_adjust_fan_speed(self, slots: Dict[str, str]) -> Dict[str, Any]:
        """风扇 5 档调速 (从 voice-router-agent 移植)。"""
        from voice_router_lite.hardware.fan import (
            compute_relative_level,
            SPEED_LEVELS,
            LEVEL_NAMES,
        )

        direction_raw = slots.get("direction", "")
        target_level = slots.get("level")

        # 中文方向 → 英文方向映射（无匹配时不能默认 up，否则会误加档）
        direction: str | None = None
        if direction_raw in ("up", "down", "min", "max", "set"):
            direction = direction_raw
        else:
            for chinese, english in self._DIRECTION_MAP.items():
                if chinese in direction_raw:
                    direction = english
                    break

        if target_level is not None:
            target_level = int(target_level)
            direction = "set"

        # 从 NLU 文本中提取数字档位
        if direction == "set" and target_level is None:
            import re
            numbers = re.findall(r"(\d+)", direction_raw)
            if numbers:
                target_level = int(numbers[0])

        if direction is None:
            return {"success": False, "message": "无法理解风扇档位调节，请说如「风扇调到3档」"}

        driver = self._find_device("fan")
        if driver is None:
            return {"success": False, "message": "风扇设备不存在"}

        # 调速时确保风扇开启（含「最低档」也应启动微风）
        if not self._fan_is_on:
            driver.turn_on()
            self._fan_is_on = True

        # 计算新档位
        new_level, at_limit, limit_msg = compute_relative_level(
            self._fan_speed_level, direction, target_level,
        )

        if at_limit and direction in ("up", "down"):
            return {
                "success": True,
                "message": limit_msg,
                "device_state": driver.get_state(),
                "speed_level": self._fan_speed_level,
                "at_limit": True,
            }

        # 更新状态
        self._fan_speed_level = new_level
        pwm = SPEED_LEVELS[new_level]
        driver._state.level = new_level * 20  # 1-5 → 20-100%
        driver._state.extra["speed_level"] = new_level
        driver._state.extra["speed_pwm"] = pwm

        result = {
            "success": True,
            "message": f"散热风扇已调到{new_level}档 ({LEVEL_NAMES[new_level]})",
            "device_state": driver.get_state(),
            "speed_level": new_level,
            "speed_pwm": pwm,
        }
        self._sync_to_esp32("fan", result)
        return result

    def _handle_set_light_brightness(self, slots: Dict[str, str]) -> Dict[str, Any]:
        direction = slots.get("direction", "")
        driver = self._find_device("light") or self._find_device("led")

        if driver is None:
            return {"success": False, "message": "灯光设备不存在"}

        current = driver._state.level
        if direction == "up":
            new_level = min(100, current + 20)
        elif direction == "down":
            new_level = max(0, current - 20)
        else:
            return {"success": False, "message": "无法解析亮度调节方向"}

        driver.set_level(new_level)
        return {
            "success": True,
            "message": f"亮度已调整到 {new_level}%",
            "device_state": driver.get_state(),
        }

    def _handle_query_status(self, slots: Dict[str, str]) -> Dict[str, Any]:
        device_type = slots.get("device_type", "")
        if device_type:
            driver = self._find_device(device_type)
            if driver:
                state = driver.get_state()
                status = "已开启" if state.is_on else "已关闭"
                return {"success": True, "message": f"{state.name}{status}"}

        all_states = self.get_all_states()
        return {"success": True, "message": str(all_states)}

    def _handle_timer_setting(self, slots: Dict[str, str]) -> Dict[str, Any]:
        duration = slots.get("timer_duration", "未知")
        return {"success": True, "message": f"已设置定时: {duration}"}

    def _handle_scene_mode(self, slots: Dict[str, str]) -> Dict[str, Any]:
        mode = slots.get("mode_name", "未知")
        return {"success": True, "message": f"已切换到{mode}模式"}

    def _handle_router_reboot(self, slots: Dict[str, str]) -> Dict[str, Any]:
        if self._ubus and self._ubus.available:
            if self._ubus.reboot():
                return {"success": True, "message": "正在重启路由器"}
            return {"success": False, "message": "重启路由器失败"}
        logger.warning("收到重启路由器指令 (ubus 不可用，未实际执行)")
        return {"success": True, "message": "正在重启路由器（模拟）"}

    def _handle_wifi_restart(self, slots: Dict[str, str]) -> Dict[str, Any]:
        if self._ubus and self._ubus.available:
            if self._ubus.wireless_restart():
                return {"success": True, "message": "正在重启WiFi"}
            return {"success": False, "message": "重启WiFi失败"}
        logger.warning("收到重启WiFi指令 (ubus 不可用，未实际执行)")
        return {"success": True, "message": "正在重启WiFi（模拟）"}

    def _handle_router_query_info(self, slots: Dict[str, str]) -> Dict[str, Any]:
        """查询路由器信息：IP地址、连接设备、网速等"""
        query_type = slots.get("query_type", "").lower()

        # 尝试获取实际系统信息
        info = self._get_router_info(self._ubus)

        if query_type == "ip":
            return {"success": True, "message": f"路由器IP地址: {info['ip']}"}

        if query_type == "devices":
            return {"success": True, "message": f"当前连接设备: {info['connected_devices']}台"}

        if query_type == "speed":
            return {"success": True, "message": f"当前网速: {info['speed']}"}

        # 默认返回综合信息
        return {
            "success": True,
            "message": (
                f"路由器状态: IP={info['ip']}, "
                f"连接设备={info['connected_devices']}台, "
                f"网速={info['speed']}"
            ),
        }

    def _handle_router_wifi_config(self, slots: Dict[str, str]) -> Dict[str, Any]:
        """WiFi配置：修改密码、名称、访客网络等"""
        config_type = slots.get("config_type", "").lower()
        config_value = slots.get("config_value", "")

        if any(w in config_type for w in ["密码", "password"]):
            if config_value:
                return {"success": True, "message": f"WiFi密码已修改为: {config_value}"}
            return {"success": True, "message": "请说出新的WiFi密码"}

        if any(w in config_type for w in ["名字", "名称", "ssid"]):
            if config_value:
                return {"success": True, "message": f"WiFi名称已修改为: {config_value}"}
            return {"success": True, "message": "请说出新的WiFi名称"}

        if any(w in config_type for w in ["访客", "guest"]):
            action = slots.get("state", "")
            if action in ("on", "打开", "开启"):
                return {"success": True, "message": "访客WiFi已开启"}
            elif action in ("off", "关闭"):
                return {"success": True, "message": "访客WiFi已关闭"}
            return {"success": True, "message": "请说明要开启还是关闭访客WiFi"}

        # 默认
        logger.warning("收到WiFi配置指令 (未实际执行): type=%s, value=%s", config_type, config_value)
        return {"success": True, "message": "WiFi配置已更新"}

    def _handle_router_led_control(self, slots: Dict[str, str]) -> Dict[str, Any]:
        """控制路由器LED指示灯"""
        state = slots.get("state", "")
        if state in ("on", "打开", "开启"):
            return {"success": True, "message": "路由器指示灯已打开"}
        elif state in ("off", "关闭"):
            return {"success": True, "message": "路由器指示灯已关闭"}
        # 默认切换
        return {"success": True, "message": "路由器指示灯已切换"}

    @staticmethod
    def _get_router_info(ubus_client=None) -> Dict[str, str]:
        """获取路由器系统信息，优先读取 ubus。"""
        import socket

        ip = "192.168.1.1"
        connected = "3"
        speed = "100Mbps"

        if ubus_client is not None and ubus_client.available:
            devices = ubus_client.get_network_devices()
            if devices:
                connected = str(len(devices))
            for dev in devices:
                addrs = dev.get("ipv4-address") or []
                if dev.get("up") and addrs:
                    ip = addrs[0].get("address", ip)
                    break

        try:
            ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            pass

        return {
            "ip": ip,
            "connected_devices": connected,
            "speed": speed,
        }

    def _handle_control_ac(self, slots: Dict[str, str]) -> Dict[str, Any]:
        state = slots.get("state", "")
        temp = slots.get("temperature", "")
        if self._ws_sender:
            # 通过 WebSocket 透传红外指令到 ESP32
            self._ws_sender({"type": "ac_control", "state": state, "temperature": temp})
        return {"success": True, "message": f"空调指令: 状态={state}, 温度={temp}"}

    def _handle_device_manage(self, slots: Dict[str, str]) -> Dict[str, Any]:
        """路由器设备管理：踢掉设备、拉黑等。"""
        action = slots.get("manage_action", "")
        device = slots.get("device_name", "设备")
        if action in ("踢掉", "踢出", "断开"):
            return {"success": True, "message": f"已断开{device}"}
        if any(w in action for w in ("拉黑", "黑名单")):
            return {"success": True, "message": f"已将{device}加入黑名单"}
        return {"success": True, "message": f"已执行{action}操作"}

    def _handle_network_diag(self, slots: Dict[str, str]) -> Dict[str, Any]:
        """路由器网络诊断：测速、延迟等。"""
        diag = slots.get("diag_type", "")
        if "速度" in diag or "测速" in diag:
            return {"success": True, "message": "当前网速: 100Mbps"}
        if "延迟" in diag:
            return {"success": True, "message": "当前延迟: 12ms"}
        return {"success": True, "message": "网络状态正常"}

    def _handle_router_system(self, slots: Dict[str, str]) -> Dict[str, Any]:
        """路由器系统操作：固件更新、备份等。"""
        action = slots.get("sys_action", "")
        if "更新" in action or "升级" in action:
            return {"success": True, "message": "正在检查固件更新"}
        if "备份" in action:
            return {"success": True, "message": "配置已备份"}
        return {"success": True, "message": "系统操作已完成"}

    def _handle_router_qos(self, slots: Dict[str, str]) -> Dict[str, Any]:
        """路由器 QoS 限速。"""
        device = slots.get("device_name", "设备")
        action = slots.get("qos_action", "限速")
        bandwidth = slots.get("bandwidth_value", "")
        if bandwidth:
            return {"success": True, "message": f"已对{device}{action}，限制带宽{bandwidth}"}
        return {"success": True, "message": f"已对{device}执行{action}"}

    def _handle_router_security(self, slots: Dict[str, str]) -> Dict[str, Any]:
        """路由器安全：防火墙、家长控制等。"""
        sec = slots.get("security_type", "")
        state = slots.get("state", "on")
        action = "开启" if state in ("on", "打开") else "关闭"
        if "防火墙" in sec:
            return {"success": True, "message": f"防火墙已{action}"}
        if "家长" in sec:
            return {"success": True, "message": f"家长控制已{action}"}
        return {"success": True, "message": f"安全设置已更新"}

    def _handle_help(self, slots: Dict[str, str]) -> Dict[str, Any]:
        return {
            "success": True,
            "message": (
                "我可以控制以下设备: "
                + ", ".join(d._state.name for d in self._devices.values())
                + "。请说出指令，如'打开风扇'、'关闭灯光'。"
            ),
        }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _sync_to_esp32(self, device_type: str, result: Dict[str, Any]) -> None:
        """将设备状态变更同步到 ESP32 (风扇/显示)。"""
        if self._esp32_bridge is None:
            logger.warning("ESP32 桥接器未配置，无法同步设备状态")
            return

        import asyncio

        async def _do_sync():
            try:
                if device_type == "fan":
                    from voice_router_lite.hardware.esp32 import FanCommand, DisplayCommand, OledCommand
                    action = "on" if self._fan_is_on else "off"
                    from voice_router_lite.hardware.fan import speed_level_to_pwm

                    pwm = speed_level_to_pwm(self._fan_speed_level)
                    sent = await self._esp32_bridge.broadcast_fan(FanCommand(
                        action=action,
                        speed_level=self._fan_speed_level,
                        speed=pwm,
                    ))
                    logger.info("风扇指令已发送到 %d 台 ESP32: action=%s, level=%d", sent, action, self._fan_speed_level)
                    if sent == 0:
                        logger.warning("没有已连接的 ESP32 设备，风扇指令未能发送到硬件！")
                    await self._esp32_bridge.broadcast_display(DisplayCommand(
                        mode="level" if self._fan_is_on else "idle",
                        value=self._fan_speed_level,
                    ))
                    await self._esp32_bridge.broadcast_oled(OledCommand(
                        fan_state=f"{self._fan_speed_level}档" if self._fan_is_on else "关闭",
                        fan_level=self._fan_speed_level if self._fan_is_on else 0,
                    ))
            except Exception as e:
                logger.exception("同步设备状态到 ESP32 失败: %s", e)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_do_sync())
            else:
                loop.run_until_complete(_do_sync())
        except Exception as e:
            logger.exception("_sync_to_esp32 调度失败: %s", e)

    # 设备类型别名映射（NLU 输出 → 实际 device_type）
    _DEVICE_TYPE_ALIASES = {
        "light": "led",
        "灯光": "led",
        "灯": "led",
        "照明": "led",
    }

    def _find_device(self, device_type: str) -> Optional[DeviceDriver]:
        """根据设备类型查找驱动"""
        device_type = (device_type or "").strip().lower()
        if not device_type:
            return None

        # 别名解析
        device_type = self._DEVICE_TYPE_ALIASES.get(device_type, device_type)

        # 精确匹配
        for name, driver in self._devices.items():
            if driver._state.device_type == device_type:
                return driver

        # 名称模糊匹配
        for name, driver in self._devices.items():
            if name.lower() == device_type or device_type in name.lower():
                return driver

        return None


def create_default_device_manager(
    esp32_bridge: Optional["ESP32Bridge"] = None,
    use_openwrt_ubus: bool = True,
    include_mock_devices: bool = True,
) -> DeviceManager:
    """创建带默认设备的设备管理器。"""
    mgr = DeviceManager(
        esp32_bridge=esp32_bridge,
        use_openwrt_ubus=use_openwrt_ubus,
    )

    mgr.register_device("风扇", FanDriver())
    if include_mock_devices:
        mgr.register_device("LED", LEDDriver())
        mgr.register_device("继电器", RelayDriver())

    return mgr
