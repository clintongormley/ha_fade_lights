"""The Fade Lights integration."""

from __future__ import annotations

import asyncio
import logging
import math
import time

from homeassistant.components.light import (
    ATTR_BRIGHTNESS,
    ATTR_SUPPORTED_COLOR_MODES,
)
from homeassistant.components.light.const import DOMAIN as LIGHT_DOMAIN
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
    DEFAULT_MIN_STEP_DELAY_MS,
    DEFAULT_TRANSITION,
    DOMAIN,
    OPTION_DEFAULT_BRIGHTNESS_PCT,
    OPTION_DEFAULT_TRANSITION,
    OPTION_MIN_STEP_DELAY_MS,
    SERVICE_FADE_LIGHTS,
    STORAGE_KEY,
)

_LOGGER = logging.getLogger(__name__)

# =============================================================================
# Global State Tracking
# =============================================================================
# These module-level dictionaries track fade operations across all config entries.
# They are keyed by entity_id (e.g., "light.bedroom") for O(1) lookups during
# state change events, which fire frequently.

# Maps entity_id -> asyncio.Task for the currently running fade.
# Used to cancel an in-progress fade when a new fade starts or manual change occurs.
ACTIVE_FADES: dict[str, asyncio.Task] = {}

# Maps entity_id -> asyncio.Event used to signal cancellation to the fade loop.
# We use an Event in addition to Task.cancel() because the fade loop needs to
# check for cancellation at safe points (between service calls, not mid-call).
FADE_CANCEL_EVENTS: dict[str, asyncio.Event] = {}

# Maps entity_id -> set of recent expected brightness values during an active fade.
# This is critical for detecting physical switch changes: when a state change
# event arrives, we compare actual brightness to expected. If they differ by
# more than ±3 (tolerance for device rounding), we treat it as manual intervention
# even if the event has our context ID (which can happen with some integrations).
#
# We track a SET of recent values because state change events arrive asynchronously
# and may be delayed. By the time an event arrives, the fade may have already moved
# to a new brightness level. We need to accept any of the recent expected values.
FADE_EXPECTED_BRIGHTNESS: dict[str, set[int]] = {}

# Maps entity_id -> True when manual intervention was just detected.
# Used to suppress stale state events from the cancelled fade. When a user
# manually changes a light during a fade, delayed state events from previous
# fade steps may still arrive. Without this flag, those events could trigger
# unintended brightness restoration. Cleared after _cancel_and_wait_for_fade.
FADE_INTERRUPTED: dict[str, bool] = {}


async def async_setup(hass: HomeAssistant, _config: ConfigType) -> bool:
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

    default_brightness = entry.options.get(OPTION_DEFAULT_BRIGHTNESS_PCT, DEFAULT_BRIGHTNESS_PCT)
    default_transition = entry.options.get(OPTION_DEFAULT_TRANSITION, DEFAULT_TRANSITION)
    min_step_delay_ms = entry.options.get(OPTION_MIN_STEP_DELAY_MS, DEFAULT_MIN_STEP_DELAY_MS)

    async def handle_fade_lights(call: ServiceCall) -> None:
        """Handle the fade_lights service call."""
        entity_ids_raw = call.data.get(ATTR_ENTITY_ID)
        brightness_pct = int(call.data.get(ATTR_BRIGHTNESS_PCT, default_brightness))
        transition = float(call.data.get(ATTR_TRANSITION, default_transition))

        if entity_ids_raw is None:
            return

        if isinstance(entity_ids_raw, str):
            entity_ids = [e.strip() for e in entity_ids_raw.split(",")]
        else:
            entity_ids = list(entity_ids_raw)

        expanded_entities = await _expand_entity_ids(hass, entity_ids)

        transition_ms = int(transition * 1000)

        tasks = [
            asyncio.create_task(
                _fade_light(
                    hass,
                    entity_id,
                    brightness_pct,
                    transition_ms,
                    min_step_delay_ms,
                )
            )
            for entity_id in expanded_entities
        ]

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    hass.services.async_register(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        handle_fade_lights,
        schema=None,
    )

    # =========================================================================
    # State Change Handler
    # =========================================================================
    # This callback fires for EVERY state change in Home Assistant that affects
    # lights. It detects manual intervention during fades and spawns async
    # handlers when needed.
    #
    # Key insight: By waiting for fade cleanup before applying state, we
    # eliminate race conditions with late-arriving fade calls.
    @callback
    def handle_light_state_change(event: Event) -> None:
        """Handle all light state changes."""
        new_state = event.data.get("new_state")

        if not new_state:
            return
        if new_state.domain != LIGHT_DOMAIN:
            return

        # Ignore group helpers (lights that contain other lights).
        # Groups have an entity_id attribute listing their member lights.
        # We only care about individual lights, not the group aggregate.
        if new_state.attributes.get(ATTR_ENTITY_ID) is not None:
            return

        entity_id = new_state.entity_id
        new_brightness = new_state.attributes.get(ATTR_BRIGHTNESS)
        is_fading = entity_id in ACTIVE_FADES
        old_state = event.data.get("old_state")

        # Suppress stale events during fade cleanup. When manual intervention
        # is detected, we set this flag to ignore delayed events from previous
        # fade steps that arrive before cleanup completes.
        if entity_id in FADE_INTERRUPTED:
            _LOGGER.debug(
                "(%s) -> Ignoring stale event during fade cleanup (%s/%s)",
                entity_id,
                new_state.state,
                new_brightness,
            )
            return

        _LOGGER.debug(
            "(%s) -> state=%s, brightness=%s, is_fading=%s",
            entity_id,
            new_state.state,
            new_brightness,
            is_fading,
        )

        # =====================================================================
        # Check for manual intervention during fade
        # =====================================================================
        # During a fade, we track what brightness we EXPECT. If actual differs,
        # someone changed it manually. We spawn an async handler to cancel the
        # fade and apply the correct state.
        if is_fading and entity_id in FADE_EXPECTED_BRIGHTNESS:
            expected_values = FADE_EXPECTED_BRIGHTNESS[entity_id]
            tolerance = 3

            # Check if state matches ANY expected value (our fade step, not manual)
            # We check multiple values because state change events can be delayed
            is_our_fade = False
            if new_state.state == STATE_OFF and 0 in expected_values:
                is_our_fade = True
            elif new_state.state == STATE_ON and new_brightness is not None:
                # Check if brightness matches any expected value within tolerance
                for expected in expected_values:
                    if expected > 0 and abs(new_brightness - expected) <= tolerance:
                        is_our_fade = True
                        break

            if is_our_fade:
                # State matches expected - this is from our fade, ignore it
                _LOGGER.debug(
                    "(%s) -> State matches expected during fade, ignoring",
                    entity_id,
                )
                return

            # Manual change detected - set flag to suppress stale events and spawn handler
            _LOGGER.debug(
                "(%s) -> Manual intervention during fade: expected_values=%s, got=%s/%s",
                entity_id,
                expected_values,
                new_state.state,
                new_brightness,
            )
            FADE_INTERRUPTED[entity_id] = True
            # Cancel and wait for any active fade to fully clean up

            hass.async_create_task(_restore_manual_state(hass, entity_id, old_state, new_state))
            return

        # =====================================================================
        # Normal state handling (no active fade)
        # =====================================================================

        # OFF -> ON: Restore to stored original brightness
        if old_state and old_state.state == STATE_OFF and new_state.state == STATE_ON:
            _LOGGER.debug("(%s) -> Light turned OFF->ON", entity_id)

            # Non-dimmable lights can't have brightness restored
            if ColorMode.BRIGHTNESS not in new_state.attributes.get(ATTR_SUPPORTED_COLOR_MODES, []):
                return

            # Restore to the stored original brightness if we have one
            orig_brightness = _get_orig_brightness(hass, entity_id)
            _LOGGER.debug(
                "(%s) -> orig_brightness=%s, current_brightness=%s",
                entity_id,
                orig_brightness,
                new_brightness,
            )
            if orig_brightness > 0:
                current_brightness = new_state.attributes.get(ATTR_BRIGHTNESS, 0)
                if current_brightness != orig_brightness:
                    _LOGGER.debug("(%s) -> Restoring to brightness %s", entity_id, orig_brightness)
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

        # ON -> ON with brightness change: Store new brightness as original
        if old_state and old_state.state == STATE_ON and new_state.state == STATE_ON:
            old_brightness = old_state.attributes.get(ATTR_BRIGHTNESS)

            if new_brightness != old_brightness and new_brightness is not None:
                _LOGGER.debug(
                    "(%s) -> Storing new brightness as original: %s", entity_id, new_brightness
                )
                _store_orig_brightness(hass, entity_id, new_brightness)

    entry.async_on_unload(hass.bus.async_listen(EVENT_STATE_CHANGED, handle_light_state_change))
    entry.async_on_unload(entry.add_update_listener(async_update_options))

    return True


async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, _entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    for event in FADE_CANCEL_EVENTS.values():
        event.set()
    for task in ACTIVE_FADES.values():
        task.cancel()
    ACTIVE_FADES.clear()
    FADE_CANCEL_EVENTS.clear()
    FADE_EXPECTED_BRIGHTNESS.clear()
    FADE_INTERRUPTED.clear()

    hass.services.async_remove(DOMAIN, SERVICE_FADE_LIGHTS)
    hass.data.pop(DOMAIN, None)

    return True


# =============================================================================
# Fade Execution
# =============================================================================


async def _fade_light(
    hass: HomeAssistant,
    entity_id: str,
    brightness_pct: int,
    transition_ms: int,
    min_step_delay_ms: int,
) -> None:
    """Fade a single light to the specified brightness.

    This is the entry point for fading a single light. It handles:
    - Cancelling any existing fade for the same entity
    - Setting up tracking state (ACTIVE_FADES, FADE_CANCEL_EVENTS)
    - Delegating to _execute_fade for the actual work
    - Cleaning up tracking state when done (success, cancel, or error)
    """
    # Cancel any existing fade for this entity - only one fade per light at a time
    await _cancel_and_wait_for_fade(entity_id)

    # Create cancellation event for this fade. We use an Event rather than just
    # Task.cancel() because we need to check for cancellation at specific points
    # in the fade loop (between service calls), not interrupt mid-call.
    cancel_event = asyncio.Event()
    FADE_CANCEL_EVENTS[entity_id] = cancel_event

    # Track this task so external code can cancel it
    current_task = asyncio.current_task()
    if current_task:
        ACTIVE_FADES[entity_id] = current_task

    try:
        await _execute_fade(
            hass, entity_id, brightness_pct, transition_ms, min_step_delay_ms, cancel_event
        )
    except asyncio.CancelledError:
        pass  # Normal cancellation, not an error
    finally:
        # Clean up tracking state regardless of how the fade ended
        ACTIVE_FADES.pop(entity_id, None)
        FADE_CANCEL_EVENTS.pop(entity_id, None)
        FADE_EXPECTED_BRIGHTNESS.pop(entity_id, None)


# =============================================================================
# Fade Step Helpers
# =============================================================================


def _calculate_fade_steps(
    start_level: int,
    end_level: int,
    transition_ms: int,
    min_step_delay_ms: int,
) -> tuple[int, int, float]:
    """Calculate fade step parameters.

    Returns:
        Tuple of (num_steps, delta_per_step, delay_ms_per_step)
    """
    level_diff = abs(end_level - start_level)

    # Maximum steps we can fit in the transition time
    max_steps_by_time = max(1, int(transition_ms / min_step_delay_ms))

    # Can't have more steps than brightness levels to change
    num_steps = min(max_steps_by_time, level_diff)

    # Calculate brightness change per step (at least 1)
    delta = (end_level - start_level) / num_steps
    delta = math.ceil(delta) if delta > 0 else math.floor(delta)

    # Recalculate steps based on rounded delta
    actual_steps = math.ceil(level_diff / abs(delta))

    # Delay between steps to spread evenly across transition
    delay_ms = transition_ms / actual_steps

    return (actual_steps, delta, delay_ms)


def _calculate_next_brightness(
    current_level: int,
    end_level: int,
    delta: int,
    is_last_step: bool,
) -> int:
    """Calculate the next brightness level for a fade step.

    Handles clamping, final step targeting, and the brightness=1 edge case.
    """
    new_level = current_level + delta
    new_level = max(0, min(255, new_level))

    # Ensure we hit target on last step or if we've overshot
    if (
        is_last_step
        or (delta > 0 and new_level > end_level)
        or (delta < 0 and new_level < end_level)
    ):
        new_level = end_level

    # Skip brightness=1 (many lights behave oddly at this level)
    if new_level == 1:
        new_level = 0 if delta < 0 else 2

    return new_level


def _track_expected_brightness(entity_id: str, new_level: int, delta: int) -> None:
    """Track expected brightness for manual intervention detection.

    Maintains a set of the 2 most recent expected values. This allows
    the state change handler to detect manual changes even when events
    arrive delayed.
    """
    expected_set = FADE_EXPECTED_BRIGHTNESS[entity_id]
    expected_set.add(new_level)
    if len(expected_set) > 2:
        # Remove oldest value (furthest from target based on direction)
        if delta > 0:
            expected_set.discard(min(expected_set))
        else:
            expected_set.discard(max(expected_set))


async def _apply_brightness(hass: HomeAssistant, entity_id: str, level: int) -> None:
    """Apply a brightness level to a light.

    Handles the special case of level 0 (turn off) vs positive levels (turn on).
    """
    if level == 0:
        await hass.services.async_call(
            LIGHT_DOMAIN,
            SERVICE_TURN_OFF,
            {ATTR_ENTITY_ID: entity_id},
            blocking=True,
        )
    else:
        await hass.services.async_call(
            LIGHT_DOMAIN,
            SERVICE_TURN_ON,
            {ATTR_ENTITY_ID: entity_id, ATTR_BRIGHTNESS: level},
            blocking=True,
        )


async def _sleep_remaining_step_time(step_start: float, delay_ms: float) -> None:
    """Sleep for the remaining time in a fade step.

    Subtracts elapsed time from target delay to maintain consistent fade duration
    regardless of how long the service call took.
    """
    elapsed_ms = (time.monotonic() - step_start) * 1000
    sleep_ms = max(0, delay_ms - elapsed_ms)
    if sleep_ms > 0:
        await asyncio.sleep(sleep_ms / 1000)


async def _finalize_fade(
    hass: HomeAssistant,
    entity_id: str,
    end_level: int,
    cancel_event: asyncio.Event,
) -> None:
    """Store final brightness after successful fade completion.

    Only stores if we faded to a non-zero brightness and weren't cancelled.
    If we faded to 0 (off), we keep the previous orig_brightness so that
    turning the light back on restores to where it was before.
    """
    if end_level > 0 and not cancel_event.is_set():
        _store_orig_brightness(hass, entity_id, end_level)
        await _save_storage(hass)


# =============================================================================
# Fade Execution
# =============================================================================


async def _execute_fade(
    hass: HomeAssistant,
    entity_id: str,
    brightness_pct: int,
    transition_ms: int,
    min_step_delay_ms: int,
    cancel_event: asyncio.Event,
) -> None:
    """Execute the fade operation."""
    state = hass.states.get(entity_id)
    if not state:
        _LOGGER.warning("Entity %s not found", entity_id)
        return

    # Handle non-dimmable lights (on/off only)
    if ColorMode.BRIGHTNESS not in state.attributes.get(ATTR_SUPPORTED_COLOR_MODES, []):
        await _apply_brightness(hass, entity_id, 255 if brightness_pct > 0 else 0)
        return

    brightness = state.attributes.get(ATTR_BRIGHTNESS)
    start_level = int(brightness) if brightness is not None else 0
    end_level = int(brightness_pct / 100 * 255)

    if start_level == end_level:
        return

    # Store original brightness if not already stored (for restoration after fade)
    existing_orig = _get_orig_brightness(hass, entity_id)
    if existing_orig == 0 and start_level > 0:
        _store_orig_brightness(hass, entity_id, start_level)

    # Initialize expected brightness tracking
    FADE_EXPECTED_BRIGHTNESS[entity_id] = {start_level}
    current_level = start_level

    # Calculate fade parameters
    num_steps, delta, delay_ms = _calculate_fade_steps(
        start_level, end_level, transition_ms, min_step_delay_ms
    )

    _LOGGER.debug(
        "Fading %s from %s to %s in %s steps", entity_id, start_level, end_level, num_steps
    )

    # Execute fade loop
    for i in range(num_steps):
        step_start = time.monotonic()

        if cancel_event.is_set():
            return

        current_level = _calculate_next_brightness(
            current_level, end_level, delta, is_last_step=(i == num_steps - 1)
        )
        _track_expected_brightness(entity_id, current_level, delta)

        await _apply_brightness(hass, entity_id, current_level)

        if cancel_event.is_set():
            return

        await _sleep_remaining_step_time(step_start, delay_ms)

    await _finalize_fade(hass, entity_id, end_level, cancel_event)


# =============================================================================
# Storage Helpers
# =============================================================================
# These functions manage persistent storage for brightness tracking.
#
# Storage structure (in hass.data[DOMAIN]["data"]):
# {
#     "light.bedroom": 200,  # Original brightness before fade (for restoration)
#     "light.kitchen": 150,
#     ...
# }
#
# Note: Current brightness during fade is tracked in-memory via
# FADE_EXPECTED_BRIGHTNESS, not in persistent storage.


def _get_orig_brightness(hass: HomeAssistant, entity_id: str) -> int:
    """Get stored original brightness for an entity."""
    storage_data = hass.data.get(DOMAIN, {}).get("data", {})
    return storage_data.get(entity_id, 0)


def _store_orig_brightness(hass: HomeAssistant, entity_id: str, level: int) -> None:
    """Store original brightness for an entity."""
    hass.data[DOMAIN]["data"][entity_id] = level


async def _save_storage(hass: HomeAssistant) -> None:
    """Save storage data to disk."""
    store: Store = hass.data[DOMAIN]["store"]
    await store.async_save(hass.data[DOMAIN]["data"])


# =============================================================================
# State Change Helpers
# =============================================================================


async def _cancel_and_wait_for_fade(entity_id: str) -> None:
    """Cancel any active fade for entity and wait for cleanup.

    This function cancels an active fade task and waits for it to be fully
    removed from ACTIVE_FADES. Since asyncio.sleep is interruptible, the
    cleanup happens quickly after cancellation.
    """
    _LOGGER.debug("(%s) -> Cancelling fade", entity_id)
    if entity_id not in ACTIVE_FADES:
        _LOGGER.debug("  -> Fade not in ACTIVE_FADES")
        return

    task = ACTIVE_FADES[entity_id]

    # Signal cancellation via the cancel event if available
    if entity_id in FADE_CANCEL_EVENTS:
        _LOGGER.debug("  -> Cancelling event")
        FADE_CANCEL_EVENTS[entity_id].set()

    # Cancel the task
    if task.done():
        _LOGGER.debug("  -> Task done")
    else:
        _LOGGER.debug("  -> Cancelling task")
        task.cancel()

    # Wait for task to complete (with timeout to avoid infinite wait)
    max_wait = 50  # 50 * 0.01 = 0.5 seconds max wait
    for _ in range(max_wait):
        _LOGGER.debug("  -> Waiting for task to disappear (%s)", _)
        if entity_id not in ACTIVE_FADES:
            _LOGGER.debug("  -> Task disappeared")
            await asyncio.sleep(0.1)
            return
        """
        Is this required? Surely it'll be handled by finally?
        if task.done():
            _LOGGER.debug("  -> Task done, cleaning up")
            # Task is done but cleanup hasn't happened yet - do it manually
            ACTIVE_FADES.pop(entity_id, None)
            FADE_CANCEL_EVENTS.pop(entity_id, None)
            FADE_EXPECTED_BRIGHTNESS.pop(entity_id, None)
            await asyncio.sleep(0.1)
            return
        """
        await asyncio.sleep(0.01)

    if entity_id in ACTIVE_FADES:
        _LOGGER.debug("(%s) -> Timed out waiting for fade task to be cancelled", entity_id)
    return


def _get_intended_brightness(
    hass: HomeAssistant,
    entity_id: str,
    old_state,
    new_state,
) -> int | None:
    """Determine the intended brightness from a manual intervention.

    Returns:
        0: Light should be OFF
        >0: Light should be ON at this brightness
        None: Could not determine (integration unloaded)
    """
    if DOMAIN not in hass.data:
        return None

    if new_state.state == STATE_OFF:
        return 0

    new_brightness = new_state.attributes.get(ATTR_BRIGHTNESS)

    if old_state and old_state.state == STATE_OFF:
        # OFF -> ON: restore to original brightness
        orig = _get_orig_brightness(hass, entity_id)
        return orig if orig > 0 else new_brightness

    # ON -> ON: use the brightness from the event
    return new_brightness


async def _restore_manual_state(
    hass: HomeAssistant,
    entity_id: str,
    old_state,
    new_state,
) -> None:
    """Restore intended state after manual intervention during fade.

    When manual intervention is detected during a fade, late fade events may
    overwrite the user's intended state. This function:
    1. Cancels the fade and waits for cleanup
    2. Compares current state to intended state
    3. Restores intended state if they differ

    The intended brightness encodes both state and brightness:
    - 0 means OFF
    - >0 means ON at that brightness
    """
    _LOGGER.debug("(%s) -> in _restore_manual_state", entity_id)
    await _cancel_and_wait_for_fade(entity_id)

    intended = _get_intended_brightness(hass, entity_id, old_state, new_state)
    _LOGGER.debug("(%s) -> got intended brightness (%s)", entity_id, intended)
    if intended is None:
        _clear_fade_interrupted(entity_id)
        return

    # Store as new original brightness (for future OFF->ON restore)
    if intended > 0:
        _LOGGER.debug("(%s) -> storing original brightness (%s)", entity_id, intended)
        _store_orig_brightness(hass, entity_id, intended)

    # Get current state after fade cleanup
    current_state = hass.states.get(entity_id)
    if not current_state:
        _LOGGER.debug("(%s) -> no current state found, exiting", entity_id)
        _clear_fade_interrupted(entity_id)
        return

    current = current_state.attributes.get(ATTR_BRIGHTNESS) or 0
    if current_state.state == STATE_OFF:
        current = 0
        _LOGGER.debug("(%s) -> got current brightness (%s)", entity_id, current)

    # Restore if current differs from intended
    if intended == 0 and current != 0:
        _LOGGER.debug("(%s) -> turning light off as intended", entity_id)
        await hass.services.async_call(
            LIGHT_DOMAIN,
            SERVICE_TURN_OFF,
            {ATTR_ENTITY_ID: entity_id},
            blocking=True,
        )
        await asyncio.sleep(0.1)
    elif intended > 0 and current != intended:
        _LOGGER.debug("(%s) -> setting light brightness (%s) as intended", entity_id, intended)
        await hass.services.async_call(
            LIGHT_DOMAIN,
            SERVICE_TURN_ON,
            {ATTR_ENTITY_ID: entity_id, ATTR_BRIGHTNESS: intended},
            blocking=True,
        )
        await asyncio.sleep(0.1)

    _clear_fade_interrupted(entity_id)


def _clear_fade_interrupted(entity_id: str) -> None:
    # Clear the interrupted flag now that cleanup is complete
    _LOGGER.debug("(%s) -> Clearing FADE_INTERRUPTED", entity_id)
    FADE_INTERRUPTED.pop(entity_id, None)


# =============================================================================
# Entity Expansion
# =============================================================================


async def _expand_entity_ids(hass: HomeAssistant, entity_ids: list[str]) -> list[str]:
    """Expand light groups recursively.

    Light groups in Home Assistant have an entity_id attribute containing
    their member lights. This function recursively expands any groups to
    get the individual light entities, then deduplicates the result.

    Example:
        Input: ["light.living_room_group", "light.bedroom"]
        If light.living_room_group contains [light.lamp, light.ceiling]
        Output: ["light.lamp", "light.ceiling", "light.bedroom"]
    """
    result = []

    for entity_id in entity_ids:
        if not entity_id.startswith("light."):
            raise ServiceValidationError(f"Entity '{entity_id}' is not a light")

        state = hass.states.get(entity_id)
        if state is None:
            _LOGGER.error("Unknown light '%s'", entity_id)
            continue

        # Check if this is a group (has entity_id attribute with member lights)
        if ATTR_ENTITY_ID in state.attributes:
            group_entities = state.attributes[ATTR_ENTITY_ID]
            if isinstance(group_entities, str):
                group_entities = [group_entities]
            # Recursively expand in case of nested groups
            result.extend(await _expand_entity_ids(hass, group_entities))
        else:
            result.append(entity_id)

    return list(set(result))
