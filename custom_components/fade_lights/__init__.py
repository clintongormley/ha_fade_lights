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
)

_LOGGER = logging.getLogger(__name__)

# Track active fade tasks and their cancellation events
ACTIVE_FADES: dict[str, asyncio.Task] = {}
FADE_CANCEL_EVENTS: dict[str, asyncio.Event] = {}

# Track the expected brightness for each light during fading
# This helps detect manual changes even when context is inherited
FADE_EXPECTED_BRIGHTNESS: dict[str, int] = {}

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
    store = Store(hass, 1, STORAGE_KEY)
    storage_data = await store.async_load() or {}

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
        brightness_pct = int(call.data.get(ATTR_BRIGHTNESS_PCT, default_brightness))
        transition = float(call.data.get(ATTR_TRANSITION, default_transition))

        if isinstance(entity_ids, str):
            entity_ids = [e.strip() for e in entity_ids.split(",")]

        expanded_entities = await _expand_entity_ids(hass, entity_ids)

        transition_ms = transition * 1000

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

        new_brightness = new_state.attributes.get(ATTR_BRIGHTNESS)

        # Check if this is from our own fade operations
        is_our_context = event.context.id in ACTIVE_CONTEXTS

        # Even if it's "our" context, check if the brightness is wildly different
        # from what we expected - this indicates a manual change that inherited our context
        if is_our_context and entity_id in FADE_EXPECTED_BRIGHTNESS:
            expected = FADE_EXPECTED_BRIGHTNESS[entity_id]
            # Allow some tolerance for rounding (light may not hit exact value)
            tolerance = 5
            if new_brightness is not None and abs(new_brightness - expected) > tolerance:
                # This is actually a manual change, don't ignore it
                is_our_context = False

        if is_our_context:
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
                    _LOGGER.debug("Restoring %s to brightness %s", entity_id, orig_brightness)
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

        # Light turned OFF manually - cancel any active fade
        if new_state.state == STATE_OFF:
            if entity_id in ACTIVE_FADES:
                _LOGGER.debug("Cancelling fade on %s due to manual turn off", entity_id)
                if entity_id in FADE_CANCEL_EVENTS:
                    FADE_CANCEL_EVENTS[entity_id].set()
                ACTIVE_FADES[entity_id].cancel()
            return

        # Brightness changed while light was already ON
        if old_state and old_state.state == STATE_ON and new_state.state == STATE_ON:
            old_brightness = old_state.attributes.get(ATTR_BRIGHTNESS)

            if new_brightness != old_brightness and new_brightness is not None:
                # Cancel active fade if any
                if entity_id in ACTIVE_FADES:
                    _LOGGER.debug("Cancelling fade on %s due to manual change", entity_id)
                    if entity_id in FADE_CANCEL_EVENTS:
                        FADE_CANCEL_EVENTS[entity_id].set()
                    ACTIVE_FADES[entity_id].cancel()

                # Store new brightness as original
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
    for event in FADE_CANCEL_EVENTS.values():
        event.set()
    for task in ACTIVE_FADES.values():
        task.cancel()
    ACTIVE_FADES.clear()
    FADE_CANCEL_EVENTS.clear()
    FADE_EXPECTED_BRIGHTNESS.clear()
    ACTIVE_CONTEXTS.clear()

    hass.services.async_remove(DOMAIN, SERVICE_FADE_LIGHTS)
    hass.data.pop(DOMAIN, None)

    return True


async def _fade_light(
    hass: HomeAssistant,
    entity_id: str,
    brightness_pct: int,
    transition_ms: int,
    context: Context,
    step_delay_ms: int,
) -> None:
    """Fade a single light to the specified brightness."""
    # Cancel any existing fade for this entity
    if entity_id in ACTIVE_FADES:
        if entity_id in FADE_CANCEL_EVENTS:
            FADE_CANCEL_EVENTS[entity_id].set()
        ACTIVE_FADES[entity_id].cancel()
        try:
            await ACTIVE_FADES[entity_id]
        except asyncio.CancelledError:
            pass

    # Create cancellation event for this fade
    cancel_event = asyncio.Event()
    FADE_CANCEL_EVENTS[entity_id] = cancel_event

    current_task = asyncio.current_task()
    if current_task:
        ACTIVE_FADES[entity_id] = current_task

    try:
        await _execute_fade(
            hass, entity_id, brightness_pct, transition_ms, context, step_delay_ms, cancel_event
        )
    except asyncio.CancelledError:
        pass
    finally:
        ACTIVE_FADES.pop(entity_id, None)
        FADE_CANCEL_EVENTS.pop(entity_id, None)
        FADE_EXPECTED_BRIGHTNESS.pop(entity_id, None)


async def _execute_fade(
    hass: HomeAssistant,
    entity_id: str,
    brightness_pct: int,
    transition_ms: int,
    context: Context,
    step_delay_ms: int,
    cancel_event: asyncio.Event,
) -> None:
    """Execute the fade operation."""
    state = hass.states.get(entity_id)
    if not state:
        _LOGGER.warning("Entity %s not found", entity_id)
        return

    # Handle non-dimmable lights
    if ColorMode.BRIGHTNESS not in state.attributes.get(ATTR_SUPPORTED_COLOR_MODES, []):
        if brightness_pct == 0:
            await hass.services.async_call(
                LIGHT_DOMAIN,
                SERVICE_TURN_OFF,
                {ATTR_ENTITY_ID: entity_id},
                context=context,
                blocking=True,
            )
        else:
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

    if start_level == end_level:
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

    _LOGGER.debug("Fading %s from %s to %s in %s steps", entity_id, start_level, end_level, num_steps)

    for i in range(num_steps):
        # Check for cancellation at start of each step
        if cancel_event.is_set():
            return

        curr = _get_curr_brightness(hass, entity_id)
        new_level = curr + delta
        new_level = max(0, min(255, new_level))

        # Ensure we hit the target on the last step
        if (delta > 0 and new_level > end_level) or (
            delta < 0 and new_level < end_level
        ):
            new_level = end_level

        if i == num_steps - 1:
            new_level = end_level

        # Handle brightness level 1 edge case
        if new_level == 1:
            new_level = 0 if delta < 0 else 2

        _store_curr_brightness(hass, entity_id, new_level)

        # Track what brightness we expect so we can detect manual changes
        FADE_EXPECTED_BRIGHTNESS[entity_id] = new_level

        if new_level == 0:
            await hass.services.async_call(
                LIGHT_DOMAIN,
                SERVICE_TURN_OFF,
                {ATTR_ENTITY_ID: entity_id},
                context=context,
                blocking=True,
            )
        else:
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

        # Check for cancellation after service call completes
        if cancel_event.is_set():
            return

        await asyncio.sleep(delay_ms / 1000)

    # Store orig brightness if we faded to a non-zero value
    if end_level > 0:
        _store_orig_brightness(hass, entity_id, end_level)
        await _save_storage(hass)


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
