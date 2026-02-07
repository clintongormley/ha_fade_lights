"""WebSocket API for Fado panel."""

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
)
from homeassistant.helpers import (
    device_registry as dr,
)
from homeassistant.helpers import (
    entity_registry as er,
)

from .const import (
    AUTOCONFIGURE_MAX_PARALLEL,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MIN_STEP_DELAY_MS,
    DOMAIN,
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARNING,
    MIN_STEP_DELAY_MS,
    OPTION_LOG_LEVEL,
    OPTION_MIN_STEP_DELAY_MS,
)
from .notifications import _notify_unconfigured_lights


def async_register_websocket_api(hass: HomeAssistant) -> None:
    """Register WebSocket API commands."""
    websocket_api.async_register_command(hass, ws_get_lights)
    websocket_api.async_register_command(hass, ws_save_light_config)
    websocket_api.async_register_command(hass, ws_autoconfigure)
    websocket_api.async_register_command(hass, ws_test_native_transitions)
    websocket_api.async_register_command(hass, ws_get_settings)
    websocket_api.async_register_command(hass, ws_save_settings)


async def async_get_lights(hass: HomeAssistant) -> dict[str, Any]:
    """Get all lights grouped by area."""
    entity_reg = er.async_get(hass)
    device_reg = dr.async_get(hass)
    area_reg = ar.async_get(hass)

    storage_data = hass.data.get(DOMAIN, {}).get("data", {})

    # Build area -> lights structure
    areas_dict: dict[str | None, dict] = {}

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

        # Get area info
        area = area_reg.async_get_area(area_id) if area_id else None

        # Get or create area entry
        area_key = area.id if area else None
        if area_key not in areas_dict:
            areas_dict[area_key] = {
                "area_id": area_key,
                "name": area.name if area else "Unknown",
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
        areas_dict[area_key]["lights"].append(
            {
                "entity_id": entity.entity_id,
                "name": friendly_name,
                "icon": icon,
                "min_delay_ms": light_config.get("min_delay_ms"),
                "exclude": light_config.get("exclude", False),
                "native_transitions": light_config.get("native_transitions"),
                "min_brightness": light_config.get("min_brightness"),
            }
        )

    # Convert to sorted list: alphabetical with "Unknown" (area_id=None) at the bottom
    result = sorted(
        areas_dict.values(),
        key=lambda a: (a["area_id"] is None, a["name"]),
    )

    return {"areas": result}


@websocket_api.websocket_command({"type": "fado/get_lights"})
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
    native_transitions: bool | str | None = None,
    min_brightness: int | None = None,
    *,
    clear_min_delay: bool = False,
    clear_native_transitions: bool = False,
    clear_min_brightness: bool = False,
) -> dict[str, bool]:
    """Save per-light configuration.

    Args:
        hass: Home Assistant instance
        entity_id: The light entity ID
        min_delay_ms: Optional minimum delay in milliseconds (50-1000)
        exclude: Optional flag to exclude light from fades
        native_transitions: Optional flag for native transition support
        min_brightness: Optional minimum brightness value (1-255) that keeps light on
        clear_min_delay: If True and min_delay_ms is None, remove the setting
        clear_native_transitions: If True and native_transitions is None, remove the setting
        clear_min_brightness: If True and min_brightness is None, remove the setting

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

    if native_transitions is not None:
        data[entity_id]["native_transitions"] = native_transitions
    elif clear_native_transitions:
        data[entity_id].pop("native_transitions", None)

    if min_brightness is not None:
        data[entity_id]["min_brightness"] = min_brightness
    elif clear_min_brightness:
        data[entity_id].pop("min_brightness", None)

    # Save to disk
    store = hass.data[DOMAIN]["store"]
    await store.async_save(data)

    # Check if notification should be updated/dismissed
    await _notify_unconfigured_lights(hass)

    return {"success": True}


@websocket_api.websocket_command(
    {
        "type": "fado/save_light_config",
        vol.Required("entity_id"): str,
        vol.Optional("min_delay_ms"): vol.Any(None, vol.All(int, vol.Range(min=50, max=1000))),
        vol.Optional("exclude"): bool,
        vol.Optional("native_transitions"): vol.Any(None, bool, "disable"),
        vol.Optional("min_brightness"): vol.Any(None, vol.All(int, vol.Range(min=1, max=255))),
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

    # Determine if we should clear fields
    clear_min_delay = "min_delay_ms" in msg and msg["min_delay_ms"] is None
    clear_native_transitions = "native_transitions" in msg and msg["native_transitions"] is None
    clear_min_brightness = "min_brightness" in msg and msg["min_brightness"] is None

    result = await async_save_light_config(
        hass,
        entity_id,
        min_delay_ms=msg.get("min_delay_ms"),
        exclude=msg.get("exclude"),
        native_transitions=msg.get("native_transitions"),
        min_brightness=msg.get("min_brightness"),
        clear_min_delay=clear_min_delay,
        clear_native_transitions=clear_native_transitions,
        clear_min_brightness=clear_min_brightness,
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
        "type": "fado/autoconfigure",
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

    Both delay testing and native transitions testing are performed.
    """
    # Import here to avoid circular import (autoconfigure imports async_save_light_config)
    from .autoconfigure import async_autoconfigure_light  # noqa: PLC0415

    entity_ids = msg["entity_ids"]

    # Expand groups to individual entity IDs
    expanded_entities = _expand_light_groups(hass, entity_ids)

    # Filter out excluded lights
    filtered_entities = [
        eid for eid in expanded_entities if not _get_light_config(hass, eid).get("exclude", False)
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

    # Get testing_lights set for exclusion during autoconfigure
    testing_lights: set[str] = hass.data.get(DOMAIN, {}).get("testing_lights", set())

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

            # Add to testing set to exclude from fades during test
            testing_lights.add(entity_id)
            try:
                # Send started event after acquiring semaphore (actual testing begins)
                connection.send_message(
                    websocket_api.event_message(
                        msg["id"],
                        {"type": "started", "entity_id": entity_id},
                    )
                )

                # Run full autoconfigure (delay + native transitions, with state restoration)
                result = await async_autoconfigure_light(hass, entity_id)

                # Check if cancelled before sending result
                if cancel_event.is_set():
                    return

                if "error" in result and "min_delay_ms" not in result:
                    # Both tests failed
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
                    # At least one test succeeded
                    connection.send_message(
                        websocket_api.event_message(
                            msg["id"],
                            {
                                "type": "result",
                                "entity_id": entity_id,
                                "min_delay_ms": result.get("min_delay_ms"),
                                "native_transitions": result.get("native_transitions"),
                                "min_brightness": result.get("min_brightness"),
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
            finally:
                # Always remove from testing set when done
                testing_lights.discard(entity_id)

    # Spawn tasks for all lights
    tasks = [test_light(entity_id) for entity_id in filtered_entities]

    # Run all tasks concurrently (semaphore limits parallelism)
    if tasks:
        await asyncio.gather(*tasks)

    # Send final result to close the subscription (only if not cancelled)
    if not cancel_event.is_set():
        connection.send_result(msg["id"])


@websocket_api.websocket_command(
    {
        "type": "fado/test_native_transitions",
        vol.Required("entity_id"): str,
        vol.Optional("transition_s", default=2.0): vol.Coerce(float),
    }
)
@websocket_api.async_response
async def ws_test_native_transitions(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Handle test_native_transitions WebSocket command.

    Tests if a light supports native transitions by sending a command
    with a transition time and measuring how long until the state changes.
    """
    from .autoconfigure import async_test_native_transitions  # noqa: PLC0415

    entity_id = msg["entity_id"]
    transition_s = msg["transition_s"]

    result = await async_test_native_transitions(hass, entity_id, transition_s)

    if "error" in result:
        connection.send_error(msg["id"], "test_failed", result["error"])
    else:
        connection.send_result(msg["id"], result)


def _get_config_entry(hass: HomeAssistant):
    """Get the Fado config entry."""
    entries = hass.config_entries.async_entries(DOMAIN)
    return entries[0] if entries else None


@websocket_api.websocket_command({"type": "fado/get_settings"})
@websocket_api.async_response
async def ws_get_settings(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Handle get_settings WebSocket command."""
    entry = _get_config_entry(hass)
    if not entry:
        connection.send_error(msg["id"], "not_found", "Config entry not found")
        return

    default_min_delay_ms = entry.options.get(OPTION_MIN_STEP_DELAY_MS, DEFAULT_MIN_STEP_DELAY_MS)
    log_level = entry.options.get(OPTION_LOG_LEVEL, DEFAULT_LOG_LEVEL)

    connection.send_result(
        msg["id"],
        {
            "default_min_delay_ms": default_min_delay_ms,
            "log_level": log_level,
        },
    )


@websocket_api.websocket_command(
    {
        "type": "fado/save_settings",
        vol.Optional("default_min_delay_ms"): vol.All(
            int, vol.Range(min=MIN_STEP_DELAY_MS, max=1000)
        ),
        vol.Optional("log_level"): vol.In([LOG_LEVEL_WARNING, LOG_LEVEL_INFO, LOG_LEVEL_DEBUG]),
    }
)
@websocket_api.async_response
async def ws_save_settings(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Handle save_settings WebSocket command."""
    entry = _get_config_entry(hass)
    if not entry:
        connection.send_error(msg["id"], "not_found", "Config entry not found")
        return

    new_options = dict(entry.options)

    if "default_min_delay_ms" in msg:
        new_options[OPTION_MIN_STEP_DELAY_MS] = msg["default_min_delay_ms"]
        # Update runtime data
        if DOMAIN in hass.data and "min_step_delay_ms" in hass.data[DOMAIN]:
            hass.data[DOMAIN]["min_step_delay_ms"] = msg["default_min_delay_ms"]

    if "log_level" in msg:
        new_options[OPTION_LOG_LEVEL] = msg["log_level"]
        # Apply log level immediately via HA's logger service
        await _apply_log_level(hass, msg["log_level"])

    hass.config_entries.async_update_entry(entry, options=new_options)

    connection.send_result(msg["id"], {"success": True})


async def _apply_log_level(hass: HomeAssistant, level: str) -> None:
    """Apply log level to Home Assistant's logger system."""
    # Map our level names to Python logging level names
    level_map = {
        LOG_LEVEL_WARNING: "warning",
        LOG_LEVEL_INFO: "info",
        LOG_LEVEL_DEBUG: "debug",
    }
    python_level = level_map.get(level, "warning")

    # Use HA's logger service to set the level
    await hass.services.async_call(
        "logger",
        "set_level",
        {f"custom_components.{DOMAIN}": python_level},
    )
