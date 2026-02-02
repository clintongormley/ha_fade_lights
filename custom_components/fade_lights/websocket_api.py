"""WebSocket API for Fade Lights panel."""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.components import websocket_api
from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    area_registry as ar,
    entity_registry as er,
    floor_registry as fr,
)

from .const import DOMAIN


def async_register_websocket_api(hass: HomeAssistant) -> None:
    """Register WebSocket API commands."""
    websocket_api.async_register_command(hass, ws_get_lights)
    websocket_api.async_register_command(hass, ws_save_light_config)


async def async_get_lights(hass: HomeAssistant) -> dict[str, Any]:
    """Get all lights grouped by floor and area."""
    entity_reg = er.async_get(hass)
    area_reg = ar.async_get(hass)
    floor_reg = fr.async_get(hass)

    storage_data = hass.data.get(DOMAIN, {}).get("data", {})

    # Build floor -> area -> lights structure
    floors_dict: dict[str | None, dict] = {}

    for entity in entity_reg.entities.values():
        # Only include lights (not groups)
        if not entity.entity_id.startswith("light."):
            continue

        # Skip light groups (they have entity_id in state attributes)
        state = hass.states.get(entity.entity_id)
        if state and "entity_id" in state.attributes:
            continue

        # Get area and floor info
        area = area_reg.async_get_area(entity.area_id) if entity.area_id else None
        floor_id = area.floor_id if area else None

        # Get or create floor entry
        if floor_id not in floors_dict:
            floor = floor_reg.floors.get(floor_id) if floor_id else None
            floors_dict[floor_id] = {
                "floor_id": floor_id,
                "name": floor.name if floor else "No Floor",
                "areas": {},
            }

        # Get or create area entry
        area_id = area.id if area else None
        if area_id not in floors_dict[floor_id]["areas"]:
            floors_dict[floor_id]["areas"][area_id] = {
                "area_id": area_id,
                "name": area.name if area else "No Area",
                "lights": [],
            }

        # Get light config from storage
        light_config = storage_data.get(entity.entity_id, {})

        # Add light to area
        floors_dict[floor_id]["areas"][area_id]["lights"].append({
            "entity_id": entity.entity_id,
            "name": entity.name or entity.entity_id,
            "min_delay_ms": light_config.get("min_delay_ms"),
            "exclude": light_config.get("exclude", False),
            "use_native_transition": light_config.get("use_native_transition", True),
        })

    # Convert to list format
    result = []
    for floor_data in floors_dict.values():
        floor_data["areas"] = list(floor_data["areas"].values())
        result.append(floor_data)

    # Sort: floors with names first, then "No Floor"
    result.sort(key=lambda f: (f["floor_id"] is None, f["name"]))

    return {"floors": result}


@websocket_api.websocket_command({"type": "fade_lights/get_lights"})
@websocket_api.async_response
async def ws_get_lights(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Handle get_lights WebSocket command."""
    result = await async_get_lights(hass)
    connection.send_result(msg["id"], result)


async def async_save_light_config(
    hass: HomeAssistant,
    entity_id: str,
    min_delay_ms: int | None = None,
    exclude: bool | None = None,
    use_native_transition: bool | None = None,
    *,
    clear_min_delay: bool = False,
) -> dict[str, bool]:
    """Save per-light configuration.

    Args:
        hass: Home Assistant instance
        entity_id: The light entity ID
        min_delay_ms: Optional minimum delay in milliseconds (50-1000)
        exclude: Optional flag to exclude light from fades
        use_native_transition: Optional flag for native transition support
        clear_min_delay: If True and min_delay_ms is None, remove the setting

    Returns:
        Dict with success status
    """
    data = hass.data.get(DOMAIN, {}).get("data", {})

    if entity_id not in data:
        data[entity_id] = {}

    # Update only provided fields
    if min_delay_ms is not None:
        data[entity_id]["min_delay_ms"] = min_delay_ms
    elif clear_min_delay:
        data[entity_id].pop("min_delay_ms", None)

    if exclude is not None:
        data[entity_id]["exclude"] = exclude

    if use_native_transition is not None:
        data[entity_id]["use_native_transition"] = use_native_transition

    # Save to disk
    store = hass.data[DOMAIN]["store"]
    await store.async_save(data)

    return {"success": True}


@websocket_api.websocket_command(
    {
        "type": "fade_lights/save_light_config",
        vol.Required("entity_id"): str,
        vol.Optional("min_delay_ms"): vol.Any(None, vol.All(int, vol.Range(min=50, max=1000))),
        vol.Optional("exclude"): bool,
        vol.Optional("use_native_transition"): bool,
    }
)
@websocket_api.async_response
async def ws_save_light_config(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Handle save_light_config WebSocket command."""
    entity_id = msg["entity_id"]

    # Determine if we should clear min_delay_ms
    clear_min_delay = "min_delay_ms" in msg and msg["min_delay_ms"] is None

    result = await async_save_light_config(
        hass,
        entity_id,
        min_delay_ms=msg.get("min_delay_ms"),
        exclude=msg.get("exclude"),
        use_native_transition=msg.get("use_native_transition"),
        clear_min_delay=clear_min_delay,
    )

    connection.send_result(msg["id"], result)
