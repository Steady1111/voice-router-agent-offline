"""
设备控制模块测试：DeviceManager、各驱动、指令执行
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_router_lite.device.manager import (
    DeviceManager, DeviceDriver, DeviceState,
    FanDriver, LEDDriver, RelayDriver,
    create_default_device_manager,
)


# ======================================================================
# DeviceDriver 基类
# ======================================================================

class TestDeviceDriver:
    """设备驱动基类"""

    def test_base_driver(self):
        d = DeviceDriver()
        assert d.initialize() is True
        assert d._state.is_on is False

        d.turn_on()
        assert d._state.is_on is True

        d.turn_off()
        assert d._state.is_on is False

        d.set_level(50)
        assert d._state.level == 50

        d.set_level(150)  # 超限钳位
        assert d._state.level == 100

        d.set_level(-10)
        assert d._state.level == 0

    def test_get_state(self):
        d = DeviceDriver()
        state = d.get_state()
        assert isinstance(state, DeviceState)
        assert state.device_type == "unknown"


# ======================================================================
# FanDriver
# ======================================================================

class TestFanDriver:
    """风扇驱动"""

    @pytest.fixture
    def fan(self):
        return FanDriver()

    def test_initialize(self, fan):
        assert fan.initialize() is True

    def test_turn_on_off(self, fan):
        fan.initialize()
        assert fan._state.is_on is False
        fan.turn_on()
        assert fan._state.is_on is True
        fan.turn_off()
        assert fan._state.is_on is False

    def test_set_level(self, fan):
        fan.initialize()
        fan.set_level(75)
        assert fan._state.level == 75

    def test_get_state(self, fan):
        fan.initialize()
        fan.turn_on()
        fan.set_level(60)
        state = fan.get_state()
        assert state.name == "风扇"
        assert state.device_type == "fan"
        assert state.is_on is True
        assert state.level == 60


# ======================================================================
# LEDDriver
# ======================================================================

class TestLEDDriver:
    """LED 驱动"""

    @pytest.fixture
    def led(self):
        return LEDDriver()

    def test_initialize(self, led):
        assert led.initialize() is True

    def test_turn_on_off(self, led):
        led.initialize()
        led.turn_on()
        assert led._state.is_on is True
        led.turn_off()
        assert led._state.is_on is False

    def test_state_attributes(self, led):
        led.initialize()
        state = led.get_state()
        assert state.name == "LED"
        assert state.device_type == "led"


# ======================================================================
# RelayDriver
# ======================================================================

class TestRelayDriver:
    """继电器驱动"""

    @pytest.fixture
    def relay(self):
        return RelayDriver()

    def test_initialize(self, relay):
        assert relay.initialize() is True

    def test_turn_on_off(self, relay):
        relay.initialize()
        relay.turn_on()
        assert relay._state.is_on is True
        relay.turn_off()
        assert relay._state.is_on is False

    def test_state_attributes(self, relay):
        relay.initialize()
        state = relay.get_state()
        assert state.name == "继电器"
        assert state.device_type == "relay"


# ======================================================================
# DeviceManager
# ======================================================================

class TestDeviceManager:
    """设备管理器"""

    @pytest.fixture
    def mgr(self):
        m = create_default_device_manager()
        m.initialize()
        yield m
        m.close()

    def test_default_devices_registered(self, mgr):
        states = mgr.get_all_states()
        assert "风扇" in states
        assert "LED" in states
        assert "继电器" in states

    def test_register_custom_device(self, mgr):
        class CustomDriver(DeviceDriver):
            def __init__(self):
                super().__init__()
                self._state = DeviceState(name="自定义", device_type="custom")

        mgr.register_device("自定义", CustomDriver())
        state = mgr.get_state("自定义")
        assert state.device_type == "custom"

    def test_turn_on_device(self, mgr):
        assert mgr.turn_on("风扇") is True
        state = mgr.get_state("风扇")
        assert state.is_on is True

    def test_turn_off_device(self, mgr):
        mgr.turn_on("风扇")
        assert mgr.turn_off("风扇") is True
        state = mgr.get_state("风扇")
        assert state.is_on is False

    def test_get_nonexistent_device(self, mgr):
        assert mgr.get_state("不存在的设备") is None

    def test_turn_on_nonexistent(self, mgr):
        assert mgr.turn_on("ghost") is False

    def test_turn_off_nonexistent(self, mgr):
        assert mgr.turn_off("ghost") is False

    # ---- 指令执行 ----

    def test_execute_set_device_state_on(self, mgr):
        result = mgr.execute_command(
            "set_device_state", {"device_type": "fan", "state": "on"}
        )
        assert result["success"] is True
        assert "风扇" in result["message"]

    def test_execute_set_device_state_off(self, mgr):
        mgr.turn_on("风扇")
        result = mgr.execute_command(
            "set_device_state", {"device_type": "fan", "state": "off"}
        )
        assert result["success"] is True

    def test_execute_nonexistent_device(self, mgr):
        result = mgr.execute_command(
            "set_device_state", {"device_type": "blender", "state": "on"}
        )
        assert result["success"] is False
        assert "不存在" in result["message"]

    def test_execute_unknown_intent(self, mgr):
        result = mgr.execute_command("do_something", {})
        assert result["success"] is False
        assert "未知" in result["message"]

    def test_execute_adjust_fan_speed_up(self, mgr):
        result = mgr.execute_command(
            "adjust_fan_speed", {"direction": "up"}
        )
        assert result["success"] is True
        state = mgr.get_state("风扇")
        assert state.level > 0

    def test_execute_adjust_fan_speed_down(self, mgr):
        mgr._devices["风扇"].set_level(100)
        result = mgr.execute_command(
            "adjust_fan_speed", {"direction": "down"}
        )
        assert result["success"] is True
        state = mgr.get_state("风扇")
        assert state.level < 100

    def test_execute_adjust_fan_speed_invalid(self, mgr):
        result = mgr.execute_command(
            "device_adjust", {}
        )
        # V2 handler: empty slots gracefully defaults to "up" direction
        assert result["success"] is True

    def test_execute_light_brightness(self, mgr):
        result = mgr.execute_command(
            "device_adjust", {"direction": "up"}
        )
        assert result["success"] is True

    def test_execute_light_brightness_no_device(self, mgr):
        """没有专门 light 设备时找到 led"""
        # 默认只有 LED，无 light — 应该能找到 LED
        result = mgr.execute_command(
            "set_light_brightness", {"direction": "up"}
        )
        assert result["success"] is True

    def test_execute_query_status(self, mgr):
        result = mgr.execute_command(
            "query_device_status", {"device_type": "fan"}
        )
        assert result["success"] is True

    def test_execute_timer_setting(self, mgr):
        result = mgr.execute_command(
            "timer_setting", {"timer_duration": "10分钟"}
        )
        assert result["success"] is True

    def test_execute_scene_mode(self, mgr):
        result = mgr.execute_command(
            "scene_mode", {"mode_name": "睡眠"}
        )
        assert result["success"] is True
        assert "睡眠" in result["message"]

    def test_execute_help(self, mgr):
        result = mgr.execute_command("help", {})
        assert result["success"] is True
        # 帮助信息应包含已注册设备
        assert "风扇" in result["message"]

    def test_execute_router_reboot(self, mgr):
        result = mgr.execute_command("router_reboot", {})
        assert result["success"] is True
        assert "重启" in result["message"]

    def test_execute_wifi_restart(self, mgr):
        result = mgr.execute_command("router_wifi_restart", {})
        assert result["success"] is True
        assert "WiFi" in result["message"]

    def test_execute_control_ac(self, mgr):
        result = mgr.execute_command(
            "control_ac", {"state": "on", "temperature": "26"}
        )
        assert result["success"] is True

    # ---- 自定义回调 ----

    def test_custom_command_callback(self, mgr):
        received = {}

        def handler(slots):
            received["called"] = True
            received["slots"] = slots
            return {"success": True, "message": "custom"}

        mgr.on_command("custom_intent", handler)
        result = mgr.execute_command("custom_intent", {"key": "value"})
        assert result["success"] is True
        assert received["called"] is True
        assert received["slots"] == {"key": "value"}

    # ---- 状态管理 ----

    def test_get_all_states_format(self, mgr):
        states = mgr.get_all_states()
        assert isinstance(states, dict)
        for name, state in states.items():
            assert isinstance(state, DeviceState)
            assert hasattr(state, "is_on")
            assert hasattr(state, "level")

    def test_find_device_by_type(self, mgr):
        driver = mgr._find_device("fan")
        assert driver is not None
        assert driver._state.device_type == "fan"

    def test_find_device_by_name(self, mgr):
        driver = mgr._find_device("风扇")
        assert driver is not None

    def test_find_device_case_insensitive(self, mgr):
        driver = mgr._find_device("FAN")
        assert driver is not None
        assert driver._state.device_type == "fan"


# ======================================================================
# DeviceState
# ======================================================================

class TestDeviceState:
    """设备状态数据类"""

    def test_default_values(self):
        s = DeviceState(name="test", device_type="custom")
        assert s.name == "test"
        assert s.device_type == "custom"
        assert s.is_on is False
        assert s.level == 0
        assert s.extra == {}

    def test_custom_values(self):
        s = DeviceState(
            name="风扇", device_type="fan",
            is_on=True, level=75,
            extra={"rpm": 3000},
        )
        assert s.is_on is True
        assert s.level == 75
        assert s.extra["rpm"] == 3000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
