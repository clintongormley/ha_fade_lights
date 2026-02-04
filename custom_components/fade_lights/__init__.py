"""The Fade Lights integration."""

from __future__ import annotations

import asyncio
import logging
import math
from homeassistant.components.light import (
    ATTR_BRIGHTNESS,
    ATTR_SUPPORTED_COLOR_MODES,
    DOMAIN as LIGHT_DOMAIN,
)
from homeassistant.components.light.const import ColorMode
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
    Context,
    Event,
    HomeAssistant,
    ServiceCall,
    callback,
)
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers.storage import Store
from homeassistant.helpers.typing import ConfigType

from .const import (
    ATTR_BRIGHTNESS_PCT,
    ATTR_TRANSITION,
    DEFAULT_BRIGHTNESS_PCT,
    DEFAULT_STEP_DELAY_MS,
    DEFAULT_TRANSITION,
    DOMAIN,
    KEY_CURR_BRIGHTNESS,
    KEY_ORIG_BRIGHTNESS,
    OPTION_DEFAULT_BRIGHTNESS_PCT,
    OPTION_DEFAULT_TRANSITION,
    OPTION_STEP_DELAY_MS,
    SERVICE_FADE_LIGHTS,
    STORAGE_KEY,
    STORAGE_VERSION,
)

_LOGGER = logging.getLogger(__name__)

# Track active fade tasks
ACTIVE_FADES: dict[str, asyncio.Task] = {}

# Track contexts created by this integration
ACTIVE_CONTEXTS: set[str] = set()


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Fade Lights component."""
    if not hass.config_entries.async_entries(DOMAIN):
        hass.async_create_task(
            hass.config_entries.flow.async_init(DOMAIN, context={"source": "import"})
        )
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Fade Lights from a config entry."""
    store = Store(hass, STORAGE_VERSION, STORAGE_KEY)
    storage_data = await store.async_load() or {}

    # Migrate from v1 to v2 storage format
    storage_data = _migrate_storage(storage_data)

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN] = {
        "store": store,
        "data": storage_data,
    }

    default_brightness = entry.options.get(
        OPTION_DEFAULT_BRIGHTNESS_PCT, DEFAULT_BRIGHTNESS_PCT
    )
    default_transition = entry.options.get(
        OPTION_DEFAULT_TRANSITION, DEFAULT_TRANSITION
    )
    step_delay_ms = entry.options.get(OPTION_STEP_DELAY_MS, DEFAULT_STEP_DELAY_MS)

    async def handle_fade_lights(call: ServiceCall) -> None:
        """Handle the fade_lights service call."""
        entity_ids = call.data.get(ATTR_ENTITY_ID)
        brightness_pct = call.data.get(ATTR_BRIGHTNESS_PCT, default_brightness)
        transition = call.data.get(ATTR_TRANSITION, default_transition)

        if isinstance(entity_ids, str):
            entity_ids = [e.strip() for e in entity_ids.split(",")]

        expanded_entities = await _expand_entity_ids(hass, entity_ids)

        _LOGGER.debug("Fading lights: %s", expanded_entities)

        transition_ms = transition * 1000
        _LOGGER.debug("Transition in ms: %s", transition_ms)

        # Create a context for this fade operation
        context = Context()
        ACTIVE_CONTEXTS.add(context.id)

        try:
            tasks = [
                asyncio.create_task(
                    _fade_light(
                        hass,
                        entity_id,
                        brightness_pct,
                        transition_ms,
                        context,
                        step_delay_ms,
                    )
                )
                for entity_id in expanded_entities
            ]

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            ACTIVE_CONTEXTS.discard(context.id)

    hass.services.async_register(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        handle_fade_lights,
        schema=None,
    )

    @callback
    def handle_light_state_change(event: Event) -> None:
        """Handle all light state changes."""
        new_state = event.data.get("new_state")
        old_state = event.data.get("old_state")

        if not new_state:
            return

        entity_id = new_state.entity_id

        if not entity_id.startswith("light."):
            return

        # Ignore changes from our own fade operations
        if event.context.id in ACTIVE_CONTEXTS:
            return

        # Ignore group helpers
        if new_state.attributes.get(ATTR_ENTITY_ID) is not None:
            return

        # Light turned ON (was OFF)
        if old_state and old_state.state == STATE_OFF and new_state.state == STATE_ON:
            # Check if light supports brightness
            if ColorMode.BRIGHTNESS not in new_state.attributes.get(
                ATTR_SUPPORTED_COLOR_MODES, []
            ):
                return

            orig_brightness = _get_orig_brightness(hass, entity_id)
            if orig_brightness > 0:
                current_brightness = new_state.attributes.get(ATTR_BRIGHTNESS, 0)
                if current_brightness != orig_brightness:
                    _LOGGER.debug(
                        "Restoring %s to original brightness %s",
                        entity_id,
                        orig_brightness,
                    )
                    hass.async_create_task(
                        hass.services.async_call(
                            LIGHT_DOMAIN,
                            SERVICE_TURN_ON,
                            {
                                ATTR_ENTITY_ID: entity_id,
                                ATTR_BRIGHTNESS: orig_brightness,
                            },
                        )
                    )
            return

        # Light turned OFF - no action needed
        if new_state.state == STATE_OFF:
            return

        # Brightness changed while light was already ON
        if old_state and old_state.state == STATE_ON and new_state.state == STATE_ON:
            new_brightness = new_state.attributes.get(ATTR_BRIGHTNESS)
            old_brightness = old_state.attributes.get(ATTR_BRIGHTNESS)

            if new_brightness != old_brightness and new_brightness is not None:
                # Cancel active fade if any
                if entity_id in ACTIVE_FADES:
                    _LOGGER.debug(
                        "Cancelling fade on %s due to manual brightness change",
                        entity_id,
                    )
                    ACTIVE_FADES[entity_id].cancel()

                # Store new brightness as original
                _LOGGER.debug(
                    "Storing manual brightness change for %s: %s",
                    entity_id,
                    new_brightness,
                )
                _store_orig_brightness(hass, entity_id, new_brightness)

    entry.async_on_unload(
        hass.bus.async_listen(EVENT_STATE_CHANGED, handle_light_state_change)
    )
    entry.async_on_unload(entry.add_update_listener(async_update_options))

    return True


async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    for task in ACTIVE_FADES.values():
        task.cancel()
    ACTIVE_FADES.clear()
    ACTIVE_CONTEXTS.clear()

    hass.services.async_remove(DOMAIN, SERVICE_FADE_LIGHTS)
    hass.data.pop(DOMAIN, None)

    return True


def _migrate_storage(storage_data: dict) -> dict:
    """Migrate storage from v1 to v2 format."""
    migrated = {}
    for key, value in storage_data.items():
        if isinstance(value, int):
            # v1 format: single integer
            migrated[key] = {
                KEY_ORIG_BRIGHTNESS: value,
                KEY_CURR_BRIGHTNESS: value,
            }
        elif isinstance(value, dict):
            # Already v2 format
            migrated[key] = value
        else:
            # Unknown format, skip
            _LOGGER.warning("Unknown storage format for %s: %s", key, value)
    return migrated


async def _fade_light(
    hass: HomeAssistant,
    entity_id: str,
    brightness_pct: int,
    transition_ms: int,
    context: Context,
    step_delay_ms: int,
) -> None:
    """Fade a single light to the specified brightness."""
    if entity_id in ACTIVE_FADES:
        ACTIVE_FADES[entity_id].cancel()
        try:
            await ACTIVE_FADES[entity_id]
        except asyncio.CancelledError:
            pass

    current_task = asyncio.current_task()
    if current_task:
        ACTIVE_FADES[entity_id] = current_task

    try:
        await _execute_fade(
            hass, entity_id, brightness_pct, transition_ms, context, step_delay_ms
        )
    except asyncio.CancelledError:
        _LOGGER.debug("Fade cancelled for %s", entity_id)
    finally:
        ACTIVE_FADES.pop(entity_id, None)


async def _execute_fade(
    hass: HomeAssistant,
    entity_id: str,
    brightness_pct: int,
    transition_ms: int,
    context: Context,
    step_delay_ms: int,
) -> None:
    """Execute the fade operation."""
    state = hass.states.get(entity_id)
    if not state:
        _LOGGER.warning("Entity %s not found", entity_id)
        return

    # Handle non-dimmable lights
    if ColorMode.BRIGHTNESS not in state.attributes.get(ATTR_SUPPORTED_COLOR_MODES, []):
        if brightness_pct == 0:
            _LOGGER.debug("Turning off non-dimmable light %s", entity_id)
            await hass.services.async_call(
                LIGHT_DOMAIN,
                SERVICE_TURN_OFF,
                {ATTR_ENTITY_ID: entity_id},
                context=context,
                blocking=True,
            )
        else:
            _LOGGER.debug("Turning on non-dimmable light %s", entity_id)
            await hass.services.async_call(
                LIGHT_DOMAIN,
                SERVICE_TURN_ON,
                {ATTR_ENTITY_ID: entity_id},
                context=context,
                blocking=True,
            )
        return

    start_level = _get_current_level(hass, entity_id)
    end_level = int(brightness_pct / 100 * 255)

    _LOGGER.debug(
        "Fading %s from %s to %s in %sms",
        entity_id,
        start_level,
        end_level,
        transition_ms,
    )

    if start_level == end_level:
        _LOGGER.debug("Already at target brightness for %s", entity_id)
        return

    # Store starting brightness as curr
    _store_curr_brightness(hass, entity_id, start_level)

    # Calculate steps
    level_diff = abs(end_level - start_level)
    delay_ms = round(transition_ms / level_diff) if level_diff > 0 else step_delay_ms
    delay_ms = max(delay_ms, step_delay_ms)

    num_steps = math.ceil(transition_ms / (delay_ms + 30)) or 1
    delta = (end_level - start_level) / num_steps
    delta = math.ceil(delta) if delta > 0 else math.floor(delta)

    _LOGGER.debug(
        "Fading in %s steps of delta %s with delay %sms",
        num_steps,
        delta,
        delay_ms,
    )

    for i in range(num_steps):
        curr = _get_curr_brightness(hass, entity_id)
        new_level = curr + delta
        new_level = max(0, min(255, new_level))

        # Ensure we hit the target on the last step
        if (delta > 0 and new_level > end_level) or (delta < 0 and new_level < end_level):
            new_level = end_level

        if i == num_steps - 1:
            new_level = end_level

        # Handle brightness level 1 edge case
        if new_level == 1:
            new_level = 0 if delta < 0 else 2

        _store_curr_brightness(hass, entity_id, new_level)

        if new_level == 0:
            _LOGGER.debug("Step %s: Turning off %s", i, entity_id)
            await hass.services.async_call(
                LIGHT_DOMAIN,
                SERVICE_TURN_OFF,
                {ATTR_ENTITY_ID: entity_id},
                context=context,
                blocking=True,
            )
        else:
            _LOGGER.debug("Step %s: Setting %s to %s", i, entity_id, new_level)
            await hass.services.async_call(
                LIGHT_DOMAIN,
                SERVICE_TURN_ON,
                {
                    ATTR_ENTITY_ID: entity_id,
                    ATTR_BRIGHTNESS: int(new_level),
                },
                context=context,
                blocking=True,
            )

        await asyncio.sleep(delay_ms / 1000)

    # Store orig brightness if we faded to a non-zero value
    if end_level > 0:
        _store_orig_brightness(hass, entity_id, end_level)
        await _save_storage(hass)

    _LOGGER.debug("Fade complete for %s", entity_id)


def _get_current_level(hass: HomeAssistant, entity_id: str) -> int:
    """Get current brightness level from the light entity."""
    state = hass.states.get(entity_id)
    if not state:
        return 0
    brightness = state.attributes.get(ATTR_BRIGHTNESS)
    if brightness is None:
        return 0
    return int(brightness)


def _get_storage_key(entity_id: str) -> str:
    """Get storage key for an entity."""
    return entity_id.replace(".", "_")


def _get_orig_brightness(hass: HomeAssistant, entity_id: str) -> int:
    """Get stored original brightness for an entity."""
    key = _get_storage_key(entity_id)
    storage_data = hass.data.get(DOMAIN, {}).get("data", {})
    entity_data = storage_data.get(key, {})
    return entity_data.get(KEY_ORIG_BRIGHTNESS, 0)


def _get_curr_brightness(hass: HomeAssistant, entity_id: str) -> int:
    """Get stored current brightness for an entity."""
    key = _get_storage_key(entity_id)
    storage_data = hass.data.get(DOMAIN, {}).get("data", {})
    entity_data = storage_data.get(key, {})
    return entity_data.get(KEY_CURR_BRIGHTNESS, 0)


def _store_orig_brightness(hass: HomeAssistant, entity_id: str, level: int) -> None:
    """Store original brightness for an entity."""
    key = _get_storage_key(entity_id)
    storage_data = hass.data[DOMAIN]["data"]
    if key not in storage_data:
        storage_data[key] = {}
    storage_data[key][KEY_ORIG_BRIGHTNESS] = level


def _store_curr_brightness(hass: HomeAssistant, entity_id: str, level: int) -> None:
    """Store current brightness for an entity."""
    key = _get_storage_key(entity_id)
    storage_data = hass.data[DOMAIN]["data"]
    if key not in storage_data:
        storage_data[key] = {}
    storage_data[key][KEY_CURR_BRIGHTNESS] = level


async def _save_storage(hass: HomeAssistant) -> None:
    """Save storage data to disk."""
    store: Store = hass.data[DOMAIN]["store"]
    await store.async_save(hass.data[DOMAIN]["data"])


async def _expand_entity_ids(hass: HomeAssistant, entity_ids: list[str]) -> list[str]:
    """Expand light groups recursively."""
    result = []
    _LOGGER.debug("Expanding entity IDs: %s", entity_ids)

    for entity_id in entity_ids:
        if not entity_id.startswith("light."):
            raise ServiceValidationError(f"Entity '{entity_id}' is not a light")

        state = hass.states.get(entity_id)
        if state is None:
            _LOGGER.error("Unknown light '%s'", entity_id)
            continue

        if ATTR_ENTITY_ID in state.attributes:
            group_entities = state.attributes[ATTR_ENTITY_ID]
            if isinstance(group_entities, str):
                group_entities = [group_entities]
            result.extend(await _expand_entity_ids(hass, group_entities))
        else:
            result.append(entity_id)

    return list(set(result))
