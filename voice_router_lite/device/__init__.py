"""设备控制子模块"""
from voice_router_lite.device.manager import (
    DeviceManager,
    DeviceDriver,
    FanDriver,
    LEDDriver,
    RelayDriver,
    DeviceState,
    create_default_device_manager,
)
