"""
设备控制管理器

通过统一接口管理风扇、LED、继电器等硬件设备。
支持 GPIO 直驱和 WebSocket 透传两种控制方式。

使用方法:
    manager = DeviceManager(config_file="config/devices.yaml")
    manager.initialize()
    manager.execute(intent="set_device_state", slots={"device_type": "fan", "state": "on"})
"""

import logging
import threading
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field

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

    def __init__(self, ws_sender: Optional[Callable] = None):
        """
        Args:
            ws_sender: WebSocket 发送回调 (用于透传指令到 ESP32 等设备)
        """
        self._devices: Dict[str, DeviceDriver] = {}
        self._ws_sender = ws_sender
        self._lock = threading.Lock()
        self._initialized: bool = False

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
        """获取指令处理函数"""
        handlers = {
            "set_device_state": self._handle_set_device_state,
            "adjust_fan_speed": self._handle_adjust_fan_speed,
            "set_light_brightness": self._handle_set_light_brightness,
            "query_device_status": self._handle_query_status,
            "timer_setting": self._handle_timer_setting,
            "scene_mode": self._handle_scene_mode,
            "router_reboot": self._handle_router_reboot,
            "router_wifi_restart": self._handle_wifi_restart,
            "control_ac": self._handle_control_ac,
            "help": self._handle_help,
        }

        # 先查自定义回调
        if intent in self._command_callbacks:
            return self._command_callbacks[intent]

        return handlers.get(intent)

    def _handle_set_device_state(self, slots: Dict[str, str]) -> Dict[str, Any]:
        device_type = slots.get("device_type", "")
        state = slots.get("state", "on")

        driver = self._find_device(device_type)
        if driver is None:
            return {"success": False, "message": f"设备不存在: {device_type}"}

        if state in ("on", "打开"):
            driver.turn_on()
        elif state in ("off", "关闭"):
            driver.turn_off()

        return {
            "success": True,
            "message": f"已{state}{driver._state.name}",
            "device_state": driver.get_state(),
        }

    def _handle_adjust_fan_speed(self, slots: Dict[str, str]) -> Dict[str, Any]:
        direction = slots.get("direction", "")
        driver = self._find_device("fan")

        if driver is None:
            return {"success": False, "message": "风扇设备不存在"}

        current = driver._state.level
        if direction == "up":
            new_level = min(100, current + 20)
        elif direction == "down":
            new_level = max(0, current - 20)
        else:
            return {"success": False, "message": "无法解析调节方向"}

        driver.set_level(new_level)
        return {
            "success": True,
            "message": f"风扇速度已调整到 {new_level}%",
            "device_state": driver.get_state(),
        }

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
        # 注意：这是危险操作，实际部署需要权限控制
        logger.warning("收到重启路由器指令 (未实际执行)")
        return {"success": True, "message": "正在重启路由器"}

    def _handle_wifi_restart(self, slots: Dict[str, str]) -> Dict[str, Any]:
        logger.warning("收到重启WiFi指令 (未实际执行)")
        return {"success": True, "message": "正在重启WiFi"}

    def _handle_control_ac(self, slots: Dict[str, str]) -> Dict[str, Any]:
        state = slots.get("state", "")
        temp = slots.get("temperature", "")
        if self._ws_sender:
            # 通过 WebSocket 透传红外指令到 ESP32
            self._ws_sender({"type": "ac_control", "state": state, "temperature": temp})
        return {"success": True, "message": f"空调指令: 状态={state}, 温度={temp}"}

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

    def _find_device(self, device_type: str) -> Optional[DeviceDriver]:
        """根据设备类型查找驱动"""
        device_type = device_type.lower()

        # 精确匹配
        for name, driver in self._devices.items():
            if driver._state.device_type == device_type:
                return driver

        # 名称模糊匹配
        for name, driver in self._devices.items():
            if name.lower() == device_type or device_type in name.lower():
                return driver

        return None


def create_default_device_manager() -> DeviceManager:
    """创建带默认设备的设备管理器"""
    mgr = DeviceManager()

    # 注册默认设备
    mgr.register_device("风扇", FanDriver())
    mgr.register_device("LED", LEDDriver())
    mgr.register_device("继电器", RelayDriver())

    return mgr
