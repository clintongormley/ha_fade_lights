"""The Fade Lights integration."""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

from homeassistant.components.light import DOMAIN as LIGHT_DOMAIN
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_ENTITY_ID,
    EVENT_STATE_CHANGED,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_OFF,
    STATE_ON,
)
from homeassistant.core import (
    Event,
    HomeAssistant,
    ServiceCall,
    callback,
)
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.storage import Store
from homeassistant.helpers.typing import ConfigType

from .const import (
    ATTR_BRIGHTNESS_PCT,
    ATTR_FORCE,
    ATTR_TRANSITION,
    AUTO_BRIGHTNESS_TARGET,
    AUTO_BRIGHTNESS_THRESHOLD,
    DEFAULT_BRIGHTNESS_PCT,
    DEFAULT_FORCE,
    DEFAULT_TRANSITION,
    DOMAIN,
    OPTION_AUTO_BRIGHTNESS_TARGET,
    OPTION_AUTO_BRIGHTNESS_THRESHOLD,
    OPTION_DEFAULT_BRIGHTNESS_PCT,
    OPTION_DEFAULT_TRANSITION,
    SERVICE_FADE_LIGHTS,
    STORAGE_KEY,
    STORAGE_VERSION,
)

_LOGGER = logging.getLogger(__name__)

# Track active fade tasks
# A new task should first cancel the running task
ACTIVE_FADES: dict[str, asyncio.Task] = {}


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Fade Lights component."""
    # Check if we already have a config entry
    if not hass.config_entries.async_entries(DOMAIN):
        # Create a config entry automatically
        hass.async_create_task(
            hass.config_entries.flow.async_init(DOMAIN, context={"source": "import"})
        )
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Fade Lights from a config entry."""
    # Initialize storage to persist brightness levels across restarts
    store = Store(hass, STORAGE_VERSION, STORAGE_KEY)
    storage_data = await store.async_load() or {}

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN] = {
        "store": store,
        "data": storage_data,
    }

    # Get configurable defaults from options
    default_brightness = entry.options.get(
        OPTION_DEFAULT_BRIGHTNESS_PCT, DEFAULT_BRIGHTNESS_PCT
    )
    default_transition = entry.options.get(
        OPTION_DEFAULT_TRANSITION, DEFAULT_TRANSITION
    )

    # Register service
    async def handle_fade_lights(call: ServiceCall) -> None:
        """Handle the fade_lights service call."""
        entity_ids = call.data.get(ATTR_ENTITY_ID)
        brightness_pct = call.data.get(ATTR_BRIGHTNESS_PCT, default_brightness)
        transition = call.data.get(ATTR_TRANSITION, default_transition)
        force = call.data.get(ATTR_FORCE, DEFAULT_FORCE)

        if isinstance(entity_ids, str):
            entity_ids = [e.strip() for e in entity_ids.split(",")]

        # Expand groups and validate entities
        expanded_entities = await _expand_entity_ids(hass, entity_ids)

        _LOGGER.debug("Parallelizing fade_lights across: %s", expanded_entities)

        transition_ms = _transition_in_ms(transition)
        _LOGGER.debug("Transition in ms is: %s", transition_ms)

        # Create tasks for each light
        tasks = []
        for entity_id in expanded_entities:
            if force or await _is_automated(hass, entity_id):
                task = asyncio.create_task(
                    _fade_light(
                        hass,
                        entity_id,
                        brightness_pct,
                        transition_ms,
                        call.context,
                    )
                )
                tasks.append(task)
            else:
                _LOGGER.debug(
                    "Aborting because light '%s' has been changed manually", entity_id
                )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    hass.services.async_register(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        handle_fade_lights,
        schema=None,  # We'll use services.yaml for schema
    )

    # Get configurable auto-brightness settings from options
    auto_brightness_threshold = entry.options.get(
        OPTION_AUTO_BRIGHTNESS_THRESHOLD, AUTO_BRIGHTNESS_THRESHOLD
    )
    auto_brightness_target = entry.options.get(
        OPTION_AUTO_BRIGHTNESS_TARGET, AUTO_BRIGHTNESS_TARGET
    )

    # Convert percentage threshold to 0-255 brightness value for comparison
    threshold_brightness = int(auto_brightness_threshold / 100 * 255)

    # Register event listener for auto-brightness
    @callback
    def handle_light_on(event: Event) -> None:
        """Handle light turned on event."""
        if event.event_type != EVENT_STATE_CHANGED:
            return

        old_state = event.data.get("old_state")
        new_state = event.data.get("new_state")

        if not old_state or not new_state:
            return

        entity_id = new_state.entity_id

        # Check conditions matching the pyscript trigger
        if (
            entity_id.startswith("light.")
            and new_state.state == STATE_ON
            and old_state.state == STATE_OFF
            and new_state.attributes.get(ATTR_ENTITY_ID) is None  # Ignore group helpers
            and new_state.context.parent_id is None  # Ignore automations
            and "brightness" in new_state.attributes.get("supported_color_modes", [])
            and int(new_state.attributes.get("brightness", 0)) < threshold_brightness
        ):
            _LOGGER.debug(
                "Light %s turned on - setting brightness to %s%%",
                entity_id,
                auto_brightness_target,
            )
            hass.async_create_task(
                hass.services.async_call(
                    LIGHT_DOMAIN,
                    SERVICE_TURN_ON,
                    {
                        ATTR_ENTITY_ID: entity_id,
                        "brightness": int(auto_brightness_target / 100 * 255),
                    },
                )
            )

    entry.async_on_unload(hass.bus.async_listen(EVENT_STATE_CHANGED, handle_light_on))

    # Register options update listener
    entry.async_on_unload(entry.add_update_listener(async_update_options))

    return True


async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    # Cancel all active fade tasks
    for task in ACTIVE_FADES.values():
        task.cancel()
    ACTIVE_FADES.clear()

    # Unregister service
    hass.services.async_remove(DOMAIN, SERVICE_FADE_LIGHTS)

    # Clear data
    hass.data.pop(DOMAIN, None)

    return True


async def _fade_light(
    hass: HomeAssistant,
    entity_id: str,
    brightness_pct: int,
    transition_ms: int,
    context: Any,
) -> None:
    """Fade a single light to the specified brightness."""
    # Cancel any existing fade for this light
    if entity_id in ACTIVE_FADES:
        ACTIVE_FADES[entity_id].cancel()
        try:
            await ACTIVE_FADES[entity_id]
        except asyncio.CancelledError:
            pass

    # Create and track new task
    current_task = asyncio.current_task()
    if current_task:
        ACTIVE_FADES[entity_id] = current_task

    try:
        await _execute_fade(hass, entity_id, brightness_pct, transition_ms, context)
    finally:
        # Clean up task reference
        ACTIVE_FADES.pop(entity_id, None)


async def _execute_fade(
    hass: HomeAssistant,
    entity_id: str,
    brightness_pct: int,
    transition_ms: int,
    context: Any,
) -> None:
    """Execute the fade operation."""
    current_level = _get_current_level(hass, entity_id)
    current_level = 0 if current_level is None else current_level
    start_level = current_level
    new_level = current_level

    end_level = int(brightness_pct / 100 * 255)

    # Get entity state
    state = hass.states.get(entity_id)
    if not state:
        _LOGGER.warning("Entity %s not found", entity_id)
        return

    # Lights which don't support fading get turned on or off
    if "brightness" not in state.attributes.get("supported_color_modes", []):
        if brightness_pct == 0:
            light_service = SERVICE_TURN_OFF
            _LOGGER.debug("Turning off light %s", entity_id)
        else:
            light_service = SERVICE_TURN_ON
            _LOGGER.debug("Turning on light %s", entity_id)

        await hass.services.async_call(
            LIGHT_DOMAIN,
            light_service,
            {ATTR_ENTITY_ID: entity_id},
            context=context,
            blocking=True,
        )
        return

    _LOGGER.debug(
        "Fading light %s from %s to %s in %sms",
        entity_id,
        start_level,
        end_level,
        transition_ms,
    )

    if start_level == end_level:
        _LOGGER.debug(
            "Aborting: start level of light %s is already end level", entity_id
        )
        return

    delay_ms = round(abs(transition_ms / (end_level - start_level)))
    delay_ms = 100 if delay_ms < 100 else delay_ms

    num_steps = math.ceil(transition_ms / (delay_ms + 30)) or 1
    delta = (end_level - start_level) / num_steps
    delta = math.ceil(delta) if delta > 0 else math.floor(delta)

    _LOGGER.debug(
        "Fading in %s step(s) of delta %s with a delay of %sms",
        num_steps,
        delta,
        delay_ms,
    )

    for i in range(num_steps):
        current_level = _get_current_level(hass, entity_id)

        _LOGGER.debug("%s: current brightness of %s is %s", i, entity_id, current_level)

        # End loop if the current brightness is being changed from outside this process
        if not new_level - 5 <= current_level <= new_level + 5:
            _LOGGER.debug(
                "Aborting fade on light '%s' because the brightness has been changed by another process",
                entity_id,
            )
            return

        new_level = start_level + delta * (i + 1)
        new_level = min(max(new_level, 0), 255)
        if delta * (new_level - end_level) > 0 or i == num_steps - 1:
            new_level = end_level

        if new_level == 1:
            new_level = 0 if delta < 0 else 2
        if new_level == 0:
            _LOGGER.debug("%s: Turning off light '%s'", i, entity_id)
            await hass.services.async_call(
                LIGHT_DOMAIN,
                SERVICE_TURN_OFF,
                {ATTR_ENTITY_ID: entity_id},
                context=context,
                blocking=True,
            )
        else:
            _LOGGER.debug(
                "%s: setting new level %s on light '%s'", i, new_level, entity_id
            )
            await hass.services.async_call(
                LIGHT_DOMAIN,
                SERVICE_TURN_ON,
                {
                    ATTR_ENTITY_ID: entity_id,
                    "brightness": int(new_level),
                },
                context=context,
                blocking=True,
            )

        await asyncio.sleep(delay_ms / 1000)

    await _store_current_level(hass, entity_id)


def _transition_in_ms(transition: int | str) -> int:
    """Convert transition time to milliseconds."""
    if isinstance(transition, int):
        return transition * 1000

    t = f"00:00:{transition}"
    parts = t.split(":")
    return 1000 * (int(parts[-3]) * 3600 + int(parts[-2]) * 60 + int(parts[-1]))


def _get_current_level(hass: HomeAssistant, entity_id: str) -> int:
    """Get current brightness level of a light."""
    state = hass.states.get(entity_id)
    if not state:
        return 0
    current_level = state.attributes.get("brightness", 0)
    return 0 if current_level is None else current_level


async def _expand_entity_ids(hass: HomeAssistant, entity_ids: list[str]) -> list[str]:
    """Expand light groups recursively."""
    result = []
    _LOGGER.debug("Flattening '%s'", entity_ids)

    for entity_id in entity_ids:
        if not entity_id.startswith("light."):
            raise ServiceValidationError(f"Entity_id '{entity_id}' is not a light")

        state = hass.states.get(entity_id)
        if state is None:
            _LOGGER.error("Unknown light '%s'", entity_id)
            continue

        # Check if it's a group (has entity_id attribute)
        if ATTR_ENTITY_ID in state.attributes:
            # Recursively expand group members
            group_entities = state.attributes[ATTR_ENTITY_ID]
            if isinstance(group_entities, str):
                group_entities = [group_entities]
            result.extend(await _expand_entity_ids(hass, group_entities))
        else:
            _LOGGER.debug("Flattened %s", entity_id)
            result.append(entity_id)

    return list(set(result))


async def _store_current_level(hass: HomeAssistant, entity_id: str) -> None:
    """Store the current brightness level."""
    current_level = _get_current_level(hass, entity_id)
    key = entity_id.replace(".", "_")

    storage_data = hass.data[DOMAIN]["data"]
    storage_data[key] = current_level

    # Save to persistent storage
    store: Store = hass.data[DOMAIN]["store"]
    await store.async_save(storage_data)

    _LOGGER.debug("Stored %s in '%s'", current_level, key)


async def _is_automated(hass: HomeAssistant, entity_id: str) -> bool:
    """Check if a light is being controlled by automation."""
    storage_data = hass.data[DOMAIN]["data"]
    key = entity_id.replace(".", "_")

    stored_level = storage_data.get(key, 0)
    current_level = _get_current_level(hass, entity_id)

    _LOGGER.debug(
        "Light '%s' has current_level=%s and stored_level=%s",
        entity_id,
        current_level,
        stored_level,
    )

    return current_level == 0 or stored_level == current_level
