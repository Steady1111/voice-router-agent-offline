"""Platform-specific helpers for OpenWrt router deployment."""

from voice_router_lite.platform.cgroups import apply_resource_limits
from voice_router_lite.platform.openwrt import UBusClient, get_ubus_client

__all__ = ["UBusClient", "get_ubus_client", "apply_resource_limits"]
