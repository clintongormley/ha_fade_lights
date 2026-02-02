"""WebSocket API for Fade Lights panel."""

from __future__ import annotations

import asyncio
from typing import Any

import voluptuous as vol
from homeassistant.components import websocket_api
from homeassistant.components.light.const import DOMAIN as LIGHT_DOMAIN
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    floor_registry as fr,
)

from .const import AUTOCONFIGURE_MAX_PARALLEL, DOMAIN


def async_register_websocket_api(hass: HomeAssistant) -> None:
    """Register WebSocket API commands."""
    websocket_api.async_register_command(hass, ws_get_lights)
    websocket_api.async_register_command(hass, ws_save_light_config)
    websocket_api.async_register_command(hass, ws_autoconfigure)


async def async_get_lights(hass: HomeAssistant) -> dict[str, Any]:
    """Get all lights grouped by floor and area."""
    entity_reg = er.async_get(hass)
    device_reg = dr.async_get(hass)
    area_reg = ar.async_get(hass)
    floor_reg = fr.async_get(hass)

    storage_data = hass.data.get(DOMAIN, {}).get("data", {})

    # Build floor -> area -> lights structure
    floors_dict: dict[str | None, dict] = {}

    for entity in entity_reg.entities.values():
        # Only include lights (not groups)
        if not entity.entity_id.startswith("light."):
            continue

        # Skip disabled entities
        if entity.disabled_by is not None:
            continue

        # Skip light groups (they have entity_id in state attributes)
        state = hass.states.get(entity.entity_id)
        if state and "entity_id" in state.attributes:
            continue

        # Get area_id from entity, or from device if entity doesn't have one
        area_id = entity.area_id
        if not area_id and entity.device_id:
            device = device_reg.async_get(entity.device_id)
            if device:
                area_id = device.area_id

        # Get area and floor info
        area = area_reg.async_get_area(area_id) if area_id else None
        floor_id = area.floor_id if area else None

        # Get or create floor entry
        if floor_id not in floors_dict:
            floor = floor_reg.floors.get(floor_id) if floor_id else None
            floors_dict[floor_id] = {
                "floor_id": floor_id,
                "name": floor.name if floor else "No Floor",
                "icon": floor.icon if floor else None,
                "areas": {},
            }

        # Get or create area entry
        area_key = area.id if area else None
        if area_key not in floors_dict[floor_id]["areas"]:
            floors_dict[floor_id]["areas"][area_key] = {
                "area_id": area_key,
                "name": area.name if area else "No Area",
                "icon": area.icon if area else None,
                "lights": [],
            }

        # Get light config from storage
        light_config = storage_data.get(entity.entity_id, {})

        # Get friendly name: prefer state name, then entity original_name, then entity_id
        friendly_name = None
        if state:
            friendly_name = state.attributes.get("friendly_name")
        if not friendly_name:
            friendly_name = entity.original_name or entity.name or entity.entity_id

        # Get icon: prefer state icon, then entity icon
        icon = None
        if state:
            icon = state.attributes.get("icon")
        if not icon:
            icon = entity.icon

        # Add light to area
        floors_dict[floor_id]["areas"][area_key]["lights"].append({
            "entity_id": entity.entity_id,
            "name": friendly_name,
            "icon": icon,
            "min_delay_ms": light_config.get("min_delay_ms"),
            "exclude": light_config.get("exclude", False),
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
    *,
    clear_min_delay: bool = False,
) -> dict[str, bool]:
    """Save per-light configuration.

    Args:
        hass: Home Assistant instance
        entity_id: The light entity ID
        min_delay_ms: Optional minimum delay in milliseconds (50-1000)
        exclude: Optional flag to exclude light from fades
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
        clear_min_delay=clear_min_delay,
    )

    connection.send_result(msg["id"], result)


def _expand_light_groups(hass: HomeAssistant, entity_ids: list[str]) -> list[str]:
    """Expand light groups to individual light entities.

    Light groups have an entity_id attribute containing member lights.
    Expands iteratively (not recursively) and deduplicates results.

    Args:
        hass: Home Assistant instance
        entity_ids: List of entity IDs (may include groups)

    Returns:
        List of individual light entity IDs (no groups)
    """
    pending = list(entity_ids)
    result: set[str] = set()
    light_prefix = f"{LIGHT_DOMAIN}."

    while pending:
        entity_id = pending.pop()
        state = hass.states.get(entity_id)

        if state is None:
            continue

        # Check if this is a group (has entity_id attribute with member lights)
        if ATTR_ENTITY_ID in state.attributes:
            group_members = state.attributes[ATTR_ENTITY_ID]
            if isinstance(group_members, str):
                group_members = [group_members]
            # Filter to lights only (groups can technically contain non-lights)
            pending.extend(m for m in group_members if m.startswith(light_prefix))
        else:
            result.add(entity_id)

    return list(result)


def _get_light_config(hass: HomeAssistant, entity_id: str) -> dict[str, Any]:
    """Get per-light configuration from storage.

    Args:
        hass: Home Assistant instance
        entity_id: The light entity ID

    Returns:
        Config dict for the light, or empty dict if not configured
    """
    return hass.data.get(DOMAIN, {}).get("data", {}).get(entity_id, {})


@websocket_api.websocket_command(
    {
        "type": "fade_lights/autoconfigure",
        vol.Required("entity_ids"): [str],
    }
)
@websocket_api.async_response
async def ws_autoconfigure(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Handle autoconfigure WebSocket command.

    This is a subscription handler that streams events back to the frontend
    as each light is tested. Events are sent for:
    - started: When a light test begins
    - result: When a light test completes successfully
    - error: When a light test fails
    """
    # Import here to avoid circular import (autoconfigure imports async_save_light_config)
    from .autoconfigure import async_test_light_delay  # noqa: PLC0415

    entity_ids = msg["entity_ids"]

    # Expand groups to individual entity IDs
    expanded_entities = _expand_light_groups(hass, entity_ids)

    # Filter out excluded lights
    filtered_entities = [
        eid for eid in expanded_entities
        if not _get_light_config(hass, eid).get("exclude", False)
    ]

    # Create cancellation event for unsubscribe support
    cancel_event = asyncio.Event()

    def cancel_subscription() -> None:
        """Cancel the autoconfigure subscription."""
        cancel_event.set()

    # Register subscription so client can unsubscribe
    connection.subscriptions[msg["id"]] = cancel_subscription

    # Create semaphore to limit parallel testing
    semaphore = asyncio.Semaphore(AUTOCONFIGURE_MAX_PARALLEL)

    async def test_light(entity_id: str) -> None:
        """Test a single light and send events."""
        # Check if cancelled before waiting for semaphore
        if cancel_event.is_set():
            return

        # Acquire semaphore before testing
        async with semaphore:
            # Check if cancelled after acquiring semaphore
            if cancel_event.is_set():
                return

            # Send started event after acquiring semaphore (actual testing begins)
            connection.send_message(
                websocket_api.event_message(
                    msg["id"],
                    {"type": "started", "entity_id": entity_id},
                )
            )

            try:
                result = await async_test_light_delay(hass, entity_id)

                # Check if cancelled before sending result
                if cancel_event.is_set():
                    return

                if "error" in result:
                    # Send error event
                    connection.send_message(
                        websocket_api.event_message(
                            msg["id"],
                            {
                                "type": "error",
                                "entity_id": entity_id,
                                "message": result["error"],
                            },
                        )
                    )
                else:
                    # Send result event
                    connection.send_message(
                        websocket_api.event_message(
                            msg["id"],
                            {
                                "type": "result",
                                "entity_id": entity_id,
                                "min_delay_ms": result["min_delay_ms"],
                            },
                        )
                    )
            except Exception as err:  # noqa: BLE001
                # Check if cancelled before sending error
                if cancel_event.is_set():
                    return
                # Send error event for unexpected exceptions
                connection.send_message(
                    websocket_api.event_message(
                        msg["id"],
                        {
                            "type": "error",
                            "entity_id": entity_id,
                            "message": str(err),
                        },
                    )
                )

    # Spawn tasks for all lights
    tasks = [test_light(entity_id) for entity_id in filtered_entities]

    # Run all tasks concurrently (semaphore limits parallelism)
    if tasks:
        await asyncio.gather(*tasks)

    # Send final result to close the subscription (only if not cancelled)
    if not cancel_event.is_set():
        connection.send_result(msg["id"])
