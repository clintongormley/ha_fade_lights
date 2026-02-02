"""The Fade Lights integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import asyncio
import logging
import time

import voluptuous as vol
from homeassistant.components import panel_custom
from homeassistant.components.http import StaticPathConfig
from homeassistant.components.light import (
    ATTR_BRIGHTNESS,
    ATTR_SUPPORTED_COLOR_MODES,
)
from homeassistant.components.light import (
    ATTR_COLOR_TEMP_KELVIN as HA_ATTR_COLOR_TEMP_KELVIN,
)
from homeassistant.components.light import (
    ATTR_HS_COLOR as HA_ATTR_HS_COLOR,
)
from homeassistant.components.light.const import DOMAIN as LIGHT_DOMAIN
from homeassistant.components.light.const import ColorMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_ENTITY_ID,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_OFF,
    STATE_ON,
)
from homeassistant.core import (
    Event,
    EventStateChangedData,
    HomeAssistant,
    ServiceCall,
    State,
    callback,
)
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.event import TrackStates, async_track_state_change_filtered
from homeassistant.helpers.service import remove_entity_service_fields
from homeassistant.helpers.storage import Store
from homeassistant.helpers.target import (
    TargetSelection,
    async_extract_referenced_entity_ids,
)
from homeassistant.helpers.typing import ConfigType

from .const import (
    DEFAULT_MIN_STEP_DELAY_MS,
    DOMAIN,
    FADE_CANCEL_TIMEOUT_S,
    OPTION_MIN_STEP_DELAY_MS,
    SERVICE_FADE_LIGHTS,
    STORAGE_KEY,
)
from .expected_state import ExpectedState, ExpectedValues
from .fade_change import FadeChange, FadeStep
from .fade_params import FadeParams
from .websocket_api import async_register_websocket_api

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


# Maps entity_id -> ExpectedState for tracking expected state during fades
# and waiting for state change events after service calls.
FADE_EXPECTED_STATE: dict[str, ExpectedState] = {}

# Maps entity_id -> asyncio.Condition to signal when fade cleanup completes.
# Waiters use this instead of polling to know when a cancelled fade has finished.
FADE_COMPLETE_CONDITIONS: dict[str, asyncio.Condition] = {}

# Maps entity_id -> queue of States from manual interventions.
# First entry is the old_state when queue was created; subsequent entries are intended states.
# Used to compare adjacent states when determining brightness transitions.
INTENDED_STATE_QUEUE: dict[str, list[State]] = {}

# Maps entity_id -> running restore Task to avoid spawning duplicates.
# Only one restore task runs per entity; subsequent events update LATEST_INTENDED_STATE.
RESTORE_TASKS: dict[str, asyncio.Task] = {}


# =============================================================================
# Integration Setup
# =============================================================================


async def async_setup(hass: HomeAssistant, _config: ConfigType) -> bool:
    """Set up the Fade Lights component."""
    if not hass.config_entries.async_entries(DOMAIN):
        hass.async_create_task(
            hass.config_entries.flow.async_init(DOMAIN, context={"source": "import"})
        )
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Fade Lights from a config entry."""
    store: Store[dict[str, int]] = Store(hass, 1, STORAGE_KEY)
    storage_data = await store.async_load() or {}

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN] = {
        "store": store,
        "data": storage_data,
        "min_step_delay_ms": entry.options.get(OPTION_MIN_STEP_DELAY_MS, DEFAULT_MIN_STEP_DELAY_MS),
    }

    async def handle_fade_lights(call: ServiceCall) -> None:
        """Service handler wrapper."""
        await _handle_fade_lights(hass, call)

    @callback
    def handle_light_state_change(event: Event[EventStateChangedData]) -> None:
        """Event handler wrapper."""
        _handle_light_state_change(hass, event)

    hass.services.async_register(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        handle_fade_lights,
        schema=cv.make_entity_service_schema({}, extra=vol.ALLOW_EXTRA),
    )

    # Track only light domain state changes (more efficient than listening to all events)
    tracker = async_track_state_change_filtered(
        hass,
        TrackStates(False, set(), {LIGHT_DOMAIN}),
        handle_light_state_change,
    )
    entry.async_on_unload(tracker.async_remove)
    entry.async_on_unload(entry.add_update_listener(async_update_options))

    # Register WebSocket API
    async_register_websocket_api(hass)

    # Register panel (only if HTTP component is available - won't be in tests)
    if hass.http is not None:
        # Register static path for frontend files
        await hass.http.async_register_static_paths(
            [StaticPathConfig(
                "/fade_lights_panel",
                str(Path(__file__).parent / "frontend"),
                cache_headers=False,  # Disable caching during development
            )]
        )

        # Register the panel
        await panel_custom.async_register_panel(
            hass,
            frontend_url_path="fade-lights",
            webcomponent_name="fade-lights-panel",
            sidebar_title="Fade Lights",
            sidebar_icon="mdi:lightbulb-variant",
            module_url="/fade_lights_panel/panel.js",
            require_admin=False,
        )

    return True


async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, _entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    for event in FADE_CANCEL_EVENTS.values():
        event.set()

    # Cancel all active fade tasks
    fade_tasks = list(ACTIVE_FADES.values())
    for task in fade_tasks:
        task.cancel()

    # Cancel all restore tasks
    restore_tasks = list(RESTORE_TASKS.values())
    for task in restore_tasks:
        task.cancel()

    all_tasks = fade_tasks + restore_tasks
    if all_tasks:
        await asyncio.gather(*all_tasks, return_exceptions=True)

    ACTIVE_FADES.clear()
    FADE_CANCEL_EVENTS.clear()
    FADE_EXPECTED_STATE.clear()
    FADE_COMPLETE_CONDITIONS.clear()
    INTENDED_STATE_QUEUE.clear()
    RESTORE_TASKS.clear()

    hass.services.async_remove(DOMAIN, SERVICE_FADE_LIGHTS)
    hass.data.pop(DOMAIN, None)

    return True


# =============================================================================
# Service Handler: fade_lights
# =============================================================================


async def _handle_fade_lights(hass: HomeAssistant, call: ServiceCall) -> None:
    """Handle the fade_lights service call."""
    domain_data = hass.data.get(DOMAIN, {})
    min_step_delay_ms = domain_data.get("min_step_delay_ms", DEFAULT_MIN_STEP_DELAY_MS)

    # Remove target fields (entity_id, device_id, area_id, etc.) from service data
    # before parsing fade parameters - these are handled separately via TargetSelection
    service_data = remove_entity_service_fields(call)
    fade_params = FadeParams.from_service_data(service_data)

    if not fade_params.has_target() and not fade_params.has_from_target():
        _LOGGER.debug("No fade parameters specified, nothing to do")
        return

    # Resolve targets to entity IDs
    target_selection = TargetSelection(call.data)
    selected = async_extract_referenced_entity_ids(hass, target_selection)
    all_entity_ids = selected.referenced | selected.indirectly_referenced

    # Filter to light domain and expand groups
    light_prefix = f"{LIGHT_DOMAIN}."
    light_entity_ids = [eid for eid in all_entity_ids if eid.startswith(light_prefix)]
    expanded_entities = _expand_light_groups(hass, light_entity_ids)

    if not expanded_entities:
        _LOGGER.debug("No light entities found in target")
        return

    tasks = []
    for entity_id in expanded_entities:
        state = hass.states.get(entity_id)
        if state and not _can_apply_fade_params(state, fade_params):
            _LOGGER.info(
                "%s: Skipping - light cannot apply any requested fade parameters",
                entity_id,
            )
            continue
        tasks.append(
            asyncio.create_task(
                _fade_light(
                    hass,
                    entity_id,
                    fade_params,
                    min_step_delay_ms,
                )
            )
        )

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _fade_light(
    hass: HomeAssistant,
    entity_id: str,
    fade_params: FadeParams,
    min_step_delay_ms: int,
) -> None:
    """Fade a single light to the specified brightness and/or color.

    This is the entry point for fading a single light. It handles:
    - Cancelling any existing fade for the same entity
    - Setting up tracking state (ACTIVE_FADES, FADE_CANCEL_EVENTS)
    - Delegating to _execute_fade for the actual work
    - Cleaning up tracking state when done (success, cancel, or error)
    """
    # Get per-light config and determine effective delay
    light_config = _get_light_config(hass, entity_id)
    effective_delay = light_config.get("min_delay_ms") or min_step_delay_ms

    # Cancel any existing fade for this entity - only one fade per light at a time
    await _cancel_and_wait_for_fade(entity_id)

    # Create cancellation event for this fade. We use an Event rather than just
    # Task.cancel() because we need to check for cancellation at specific points
    # in the fade loop (between service calls), not interrupt mid-call.
    cancel_event = asyncio.Event()
    FADE_CANCEL_EVENTS[entity_id] = cancel_event

    # Create completion condition for waiters to know when cleanup is done
    complete_condition = asyncio.Condition()
    FADE_COMPLETE_CONDITIONS[entity_id] = complete_condition

    # Track this task so external code can cancel it
    current_task = asyncio.current_task()
    if current_task:
        ACTIVE_FADES[entity_id] = current_task

    try:
        await _execute_fade(hass, entity_id, fade_params, effective_delay, cancel_event)
    except asyncio.CancelledError:
        pass  # Normal cancellation, not an error
    finally:
        # Clean up tracking state regardless of how the fade ended
        ACTIVE_FADES.pop(entity_id, None)
        FADE_CANCEL_EVENTS.pop(entity_id, None)
        # Note: FADE_EXPECTED_STATE is NOT cleared here - values persist
        # for event matching and are pruned when next fade starts

        # Notify any waiters that cleanup is complete
        condition = FADE_COMPLETE_CONDITIONS.pop(entity_id, None)
        if condition:
            async with condition:
                condition.notify_all()


async def _execute_fade(
    hass: HomeAssistant,
    entity_id: str,
    fade_params: FadeParams,
    min_step_delay_ms: int,
    cancel_event: asyncio.Event,
) -> None:
    """Execute the fade operation using FadeChange iterator pattern.

    Uses _resolve_fade to create a single FadeChange that handles all fade types
    including hybrid transitions internally. The iterator generates steps
    seamlessly across mode switches.
    """
    state = hass.states.get(entity_id)
    if not state:
        _LOGGER.warning("%s: Entity not found", entity_id)
        return

    # Store original brightness for restoration after OFF->ON
    # Update if: (1) nothing stored yet, or (2) user changed brightness since last fade
    current_brightness = state.attributes.get(ATTR_BRIGHTNESS)
    start_brightness = int(current_brightness) if current_brightness is not None else 0
    existing_orig = _get_orig_brightness(hass, entity_id)
    if start_brightness > 0 and start_brightness != existing_orig:
        _store_orig_brightness(hass, entity_id, start_brightness)

    # Get stored brightness for auto-turn-on when fading color from off
    stored_brightness = existing_orig if existing_orig > 0 else start_brightness

    # Resolve fade parameters into a configured FadeChange
    fade = FadeChange.resolve(fade_params, state.attributes, min_step_delay_ms, stored_brightness)

    if fade is None:
        _LOGGER.debug("%s: Nothing to fade", entity_id)
        return

    total_steps = fade.step_count()
    delay_ms = fade.delay_ms()

    _LOGGER.info(
        "%s: Fading in %s steps, (brightness=%s->%s, hs=%s->%s, mireds=%s->%s, "
        "hybrid=%s, crossover_step=%s, delay_ms=%s)",
        entity_id,
        total_steps,
        fade.start_brightness,
        fade.end_brightness,
        fade.start_hs,
        fade.end_hs,
        fade.start_mireds,
        fade.end_mireds,
        fade._hybrid_direction,
        fade._crossover_step,
        delay_ms,
    )

    # Execute fade steps
    while fade.has_next():
        step_start = time.monotonic()

        if cancel_event.is_set():
            return

        step = fade.next_step()

        # Track expected values for manual intervention detection
        expected = ExpectedValues(
            brightness=step.brightness,
            hs_color=step.hs_color,
            color_temp_kelvin=step.color_temp_kelvin,
        )
        _add_expected_values(entity_id, expected)

        await _apply_step(hass, entity_id, step)

        if cancel_event.is_set():
            return

        # Sleep remaining time (skip after last step)
        if fade.has_next():
            await _sleep_remaining_step_time(step_start, delay_ms)

    # Wait for any late events and clear expected state
    expected_state = FADE_EXPECTED_STATE.get(entity_id)
    if expected_state:
        _LOGGER.debug(
            "%s: Fade finished. Waiting for expected state events to be flushed (remaining: %d)",
            entity_id,
            len(expected_state.values),
        )
        await expected_state.wait_and_clear()

    # Store final brightness after successful fade completion
    if not cancel_event.is_set():
        final_brightness = fade.end_brightness

        if final_brightness is not None and final_brightness > 0:
            _store_orig_brightness(hass, entity_id, final_brightness)
            await _save_storage(hass)
            _LOGGER.info("%s: Fade complete at brightness %s", entity_id, final_brightness)
        elif final_brightness == 0:
            _LOGGER.info("%s: Fade complete (turned off)", entity_id)
        else:
            _LOGGER.info("%s: Fade complete", entity_id)


async def _apply_step(
    hass: HomeAssistant,
    entity_id: str,
    step: FadeStep,
) -> None:
    """Apply a fade step to a light.

    Handles brightness, hs_color, and color_temp_kelvin in a single service call.
    If brightness is 0, turns off the light. If step is empty, does nothing.

    Args:
        hass: Home Assistant instance
        entity_id: Light entity ID
        step: The fade step to apply
    """
    # Build service data based on what's in the step
    service_data: dict = {ATTR_ENTITY_ID: entity_id}

    if step.brightness is not None:
        if step.brightness == 0:
            # Turn off - no other attributes needed
            await hass.services.async_call(
                LIGHT_DOMAIN,
                SERVICE_TURN_OFF,
                {ATTR_ENTITY_ID: entity_id},
                blocking=True,
            )
            return
        service_data[ATTR_BRIGHTNESS] = step.brightness

    if step.hs_color is not None:
        service_data[HA_ATTR_HS_COLOR] = step.hs_color

    if step.color_temp_kelvin is not None:
        service_data[HA_ATTR_COLOR_TEMP_KELVIN] = step.color_temp_kelvin

    _LOGGER.debug("%s", service_data)

    # Only call service if there's something to set (beyond entity_id)
    if len(service_data) > 1:
        await hass.services.async_call(
            LIGHT_DOMAIN,
            SERVICE_TURN_ON,
            service_data,
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


def _expand_light_groups(hass: HomeAssistant, entity_ids: list[str]) -> list[str]:
    """Expand light groups to individual light entities.

    Light groups have an entity_id attribute containing member lights.
    Expands iteratively (not recursively) and deduplicates results.
    Lights with exclude=True in their config are filtered out.

    Example:
        Input: ["light.living_room_group", "light.bedroom"]
        If light.living_room_group contains [light.lamp, light.ceiling]
        Output: ["light.lamp", "light.ceiling", "light.bedroom"]
    """
    pending = list(entity_ids)
    result: set[str] = set()
    light_prefix = f"{LIGHT_DOMAIN}."

    while pending:
        entity_id = pending.pop()
        state = hass.states.get(entity_id)

        if state is None:
            _LOGGER.warning("%s: Entity not found, skipping", entity_id)
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

    # Filter out excluded lights
    return [
        eid for eid in result
        if not _get_light_config(hass, eid).get("exclude", False)
    ]


def _can_apply_fade_params(state: State, params: FadeParams) -> bool:
    """Check if a light can perform at least one of the requested fade operations.

    Returns True if the light can do ANY of:
    - Brightness fade (any light, including on/off only)
    - HS color fade (if requested and light supports HS/RGB/RGBW/RGBWW/XY)
    - Color temp fade (if requested and light supports COLOR_TEMP)
    """
    modes = set(state.attributes.get(ATTR_SUPPORTED_COLOR_MODES, []))

    # Check if brightness is requested - any light can handle it
    # (on/off lights get turned on/off, dimmable lights get faded)
    brightness_requested = (
        params.brightness_pct is not None or params.from_brightness_pct is not None
    )
    if brightness_requested:
        return True

    # Check if HS color is requested and light supports it
    hs_requested = params.hs_color is not None or params.from_hs_color is not None
    hs_capable = modes & {
        ColorMode.HS,
        ColorMode.RGB,
        ColorMode.RGBW,
        ColorMode.RGBWW,
        ColorMode.XY,
    }
    if hs_requested and hs_capable:
        return True

    # Check if color temp is requested and light supports it
    color_temp_requested = (
        params.color_temp_kelvin is not None or params.from_color_temp_kelvin is not None
    )
    return color_temp_requested and ColorMode.COLOR_TEMP in modes


# =============================================================================
# State Change Handler
# =============================================================================


@callback
def _handle_light_state_change(hass: HomeAssistant, event: Event[EventStateChangedData]) -> None:
    """Handle light state changes - detects manual intervention and tracks brightness."""
    new_state: State | None = event.data.get("new_state")
    old_state: State | None = event.data.get("old_state")

    if not _should_process_state_change(new_state):
        return

    # Type narrowing: new_state is guaranteed non-None after _should_process_state_change
    assert new_state is not None

    entity_id = new_state.entity_id

    # Skip excluded lights entirely
    if _get_light_config(hass, entity_id).get("exclude", False):
        return

    # Check if this is an expected state change (from our service calls)
    if _match_and_remove_expected(entity_id, new_state):
        return

    # During fade or restore: if we get here, state didn't match expected - manual intervention
    is_during_fade = entity_id in ACTIVE_FADES and entity_id in FADE_EXPECTED_STATE
    is_during_restore = entity_id in RESTORE_TASKS
    if is_during_fade or is_during_restore:
        # Manual intervention detected - add to intended state queue
        _LOGGER.info(
            "%s: Manual intervention detected (state=%s, brightness=%s)",
            entity_id,
            new_state.state,
            new_state.attributes.get(ATTR_BRIGHTNESS),
        )

        # Initialize queue with old_state if this is the first manual event
        if entity_id not in INTENDED_STATE_QUEUE:
            INTENDED_STATE_QUEUE[entity_id] = [old_state] if old_state else []

        # Append the new intended state
        INTENDED_STATE_QUEUE[entity_id].append(new_state)

        # Only spawn restore task if one isn't already running
        if entity_id not in RESTORE_TASKS:
            task = hass.async_create_task(_restore_intended_state(hass, entity_id))
            RESTORE_TASKS[entity_id] = task
        else:
            _LOGGER.debug("%s: Restore task already running, queued intended state", entity_id)
        return

    # Normal state handling (no active fade)
    if _is_off_to_on_transition(old_state, new_state):
        _handle_off_to_on(hass, entity_id, new_state)
        return

    if _is_brightness_change(old_state, new_state):
        new_brightness = new_state.attributes.get(ATTR_BRIGHTNESS)
        if new_brightness is not None:
            _LOGGER.debug("%s: Storing new brightness as original: %s", entity_id, new_brightness)
            _store_orig_brightness(hass, entity_id, new_brightness)


def _should_process_state_change(new_state: State | None) -> bool:
    """Check if this state change should be processed."""
    if not new_state:
        return False
    if new_state.domain != LIGHT_DOMAIN:
        return False
    # Ignore group helpers (lights that contain other lights)
    return new_state.attributes.get(ATTR_ENTITY_ID) is None


def _match_and_remove_expected(entity_id: str, new_state: State) -> bool:
    """Check if state matches expected, remove if found, notify if empty.

    Returns True if this was an expected state change (caller should ignore it).
    """
    expected_state = FADE_EXPECTED_STATE.get(entity_id)
    if not expected_state or expected_state.is_empty:
        return False

    # Build ExpectedValues from the new state
    if new_state.state == STATE_OFF:
        actual = ExpectedValues(brightness=0)
    else:
        brightness = new_state.attributes.get(ATTR_BRIGHTNESS)
        if brightness is None:
            return False

        # Extract color attributes
        hs_raw = new_state.attributes.get(HA_ATTR_HS_COLOR)
        hs_color = (float(hs_raw[0]), float(hs_raw[1])) if hs_raw else None

        # Read kelvin directly from state attributes
        kelvin_raw = new_state.attributes.get(HA_ATTR_COLOR_TEMP_KELVIN)
        color_temp_kelvin = int(kelvin_raw) if kelvin_raw else None

        actual = ExpectedValues(
            brightness=brightness,
            hs_color=hs_color,
            color_temp_kelvin=color_temp_kelvin,
        )

    matched = expected_state.match_and_remove(actual)
    return matched is not None


def _is_off_to_on_transition(old_state: State | None, new_state: State) -> bool:
    """Check if this is an OFF -> ON transition."""
    return old_state is not None and old_state.state == STATE_OFF and new_state.state == STATE_ON


def _is_brightness_change(old_state: State | None, new_state: State) -> bool:
    """Check if this is an ON -> ON brightness change."""
    if not old_state or old_state.state != STATE_ON or new_state.state != STATE_ON:
        return False
    old_brightness = old_state.attributes.get(ATTR_BRIGHTNESS)
    new_brightness = new_state.attributes.get(ATTR_BRIGHTNESS)
    return new_brightness is not None and new_brightness != old_brightness


def _handle_off_to_on(hass: HomeAssistant, entity_id: str, new_state: State) -> None:
    """Handle OFF -> ON transition by restoring original brightness."""
    if DOMAIN not in hass.data:
        return
    _LOGGER.info("%s: Light turned on", entity_id)

    # Non-dimmable lights can't have brightness restored
    if ColorMode.BRIGHTNESS not in new_state.attributes.get(ATTR_SUPPORTED_COLOR_MODES, []):
        return

    orig_brightness = _get_orig_brightness(hass, entity_id)
    current_brightness = new_state.attributes.get(ATTR_BRIGHTNESS, 0)

    _LOGGER.debug(
        "%s: orig_brightness=%s, current_brightness=%s",
        entity_id,
        orig_brightness,
        current_brightness,
    )

    if orig_brightness > 0 and current_brightness != orig_brightness:
        _LOGGER.info("%s: Restoring to brightness %s", entity_id, orig_brightness)
        hass.async_create_task(_restore_original_brightness(hass, entity_id, orig_brightness))


async def _restore_original_brightness(
    hass: HomeAssistant,
    entity_id: str,
    brightness: int,
) -> None:
    """Restore original brightness and wait for confirmation."""
    _add_expected_brightness(entity_id, brightness)
    await hass.services.async_call(
        LIGHT_DOMAIN,
        SERVICE_TURN_ON,
        {ATTR_ENTITY_ID: entity_id, ATTR_BRIGHTNESS: brightness},
        blocking=True,
    )
    await _wait_until_stale_events_flushed(entity_id)


# =============================================================================
# Manual Intervention
# =============================================================================


async def _restore_intended_state(
    hass: HomeAssistant,
    entity_id: str,
) -> None:
    """Restore intended state after manual intervention during fade.

    When manual intervention is detected during a fade, late fade events may
    overwrite the user's intended state. This function:
    1. Cancels the fade and waits for cleanup
    2. Loops: reads most recent intended state from queue and restores if needed
    3. Continues until no more intended states are queued

    The queue structure is: [old_state, intended_1, intended_2, ...]
    - First entry is the state before the first manual intervention
    - Subsequent entries are intended states from manual interventions

    When processing, we compare adjacent states to determine transitions
    (e.g., OFF->ON vs ON->ON) and only store original brightness when
    the previous state had non-zero brightness.
    """
    try:
        if DOMAIN not in hass.data:
            return
        _LOGGER.debug(
            "%s: Waiting for state events to flush before restoring intended state",
            entity_id,
        )
        await _cancel_and_wait_for_fade(entity_id)

        # Loop until no more intended states are queued
        # (handles case where another manual event arrives during restore)
        while True:
            # Get the queue for this entity
            queue = INTENDED_STATE_QUEUE.get(entity_id, [])

            # Need at least 2 entries: previous state + intended state
            if len(queue) < 2:
                _LOGGER.debug("%s: No more intended states in queue, done", entity_id)
                INTENDED_STATE_QUEUE.pop(entity_id, None)
                break

            # Get the most recent intended state (last in queue)
            intended_state = queue[-1]
            # Get the previous state (second to last) for comparison
            previous_state = queue[-2]

            # Remove intended_state and all previous states from queue
            # Keep only the intended_state as the new "previous" for any future events
            INTENDED_STATE_QUEUE[entity_id] = [intended_state]

            intended_brightness = _get_intended_brightness(
                hass, entity_id, previous_state, intended_state
            )
            _LOGGER.debug("%s: Got intended brightness (%s)", entity_id, intended_brightness)
            if intended_brightness is None:
                break

            # Store as new original brightness only if:
            # - Previous state had non-zero brightness (was ON, not coming from OFF)
            # - Intended brightness is > 0 (not turning off)
            # - Brightness is actually changing
            # This ensures we track the user's intended brightness for OFF->ON restoration
            previous_brightness = (
                previous_state.attributes.get(ATTR_BRIGHTNESS, 0)
                if previous_state and previous_state.state != STATE_OFF
                else 0
            )
            if (
                previous_brightness > 0
                and intended_brightness > 0
                and intended_brightness != previous_brightness
            ):
                _LOGGER.debug(
                    "%s: Storing original brightness (%s) from transition %s->%s",
                    entity_id,
                    intended_brightness,
                    previous_brightness,
                    intended_brightness,
                )
                _store_orig_brightness(hass, entity_id, intended_brightness)

            # Get current state after fade cleanup
            current_state = hass.states.get(entity_id)
            if not current_state:
                _LOGGER.debug("%s: No current state found, exiting", entity_id)
                break

            current_brightness = current_state.attributes.get(ATTR_BRIGHTNESS) or 0
            if current_state.state == STATE_OFF:
                current_brightness = 0

            # Handle OFF case
            if intended_brightness == 0:
                if current_brightness != 0:
                    _LOGGER.info("%s: Restoring to off as intended", entity_id)
                    _add_expected_brightness(entity_id, 0)
                    await hass.services.async_call(
                        LIGHT_DOMAIN,
                        SERVICE_TURN_OFF,
                        {ATTR_ENTITY_ID: entity_id},
                        blocking=True,
                    )
                    await _wait_until_stale_events_flushed(entity_id)
                else:
                    _LOGGER.debug("%s: already off, nothing to restore", entity_id)
                continue  # Check for more intended states

            # Handle ON case - check brightness and colors
            # Build service data for restoration
            service_data: dict = {ATTR_ENTITY_ID: entity_id}
            need_restore = False

            # Check brightness
            if current_brightness != intended_brightness:
                service_data[ATTR_BRIGHTNESS] = intended_brightness
                need_restore = True

            # Get intended colors from manual intervention (intended_state)
            intended_hs = intended_state.attributes.get(HA_ATTR_HS_COLOR)
            intended_kelvin = intended_state.attributes.get(HA_ATTR_COLOR_TEMP_KELVIN)

            # Get current colors
            current_hs = current_state.attributes.get(HA_ATTR_HS_COLOR)
            current_kelvin = current_state.attributes.get(HA_ATTR_COLOR_TEMP_KELVIN)

            # Check HS color
            if intended_hs and intended_hs != current_hs:
                service_data[HA_ATTR_HS_COLOR] = intended_hs
                need_restore = True

            # Check color temp (mutually exclusive with HS)
            if (
                intended_kelvin
                and intended_kelvin != current_kelvin
                and HA_ATTR_HS_COLOR not in service_data
            ):
                service_data[HA_ATTR_COLOR_TEMP_KELVIN] = intended_kelvin
                need_restore = True

            if need_restore:
                _LOGGER.info("%s: Restoring intended state: %s", entity_id, service_data)

                # Track expected values (ExpectedValues uses kelvin)
                _add_expected_values(
                    entity_id,
                    ExpectedValues(
                        brightness=service_data.get(ATTR_BRIGHTNESS),
                        hs_color=service_data.get(HA_ATTR_HS_COLOR),
                        color_temp_kelvin=service_data.get(HA_ATTR_COLOR_TEMP_KELVIN),
                    ),
                )

                await hass.services.async_call(
                    LIGHT_DOMAIN,
                    SERVICE_TURN_ON,
                    service_data,
                    blocking=True,
                )
                await _wait_until_stale_events_flushed(entity_id)
            else:
                _LOGGER.debug("%s: already in intended state, nothing to restore", entity_id)
    finally:
        # Clean up restore task tracking
        RESTORE_TASKS.pop(entity_id, None)


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


# =============================================================================
# Fade Cancellation & Synchronization
# =============================================================================


async def _cancel_and_wait_for_fade(entity_id: str) -> None:
    """Cancel any active fade for entity and wait for cleanup.

    This function cancels an active fade task and waits for it to be fully
    removed from ACTIVE_FADES using an asyncio.Condition for efficient
    notification instead of polling.
    """
    _LOGGER.debug("%s: Cancelling fade", entity_id)
    if entity_id not in ACTIVE_FADES:
        _LOGGER.debug("%s: Fade not in ACTIVE_FADES", entity_id)
        return

    task = ACTIVE_FADES[entity_id]
    condition = FADE_COMPLETE_CONDITIONS.get(entity_id)

    # Signal cancellation via the cancel event if available
    if entity_id in FADE_CANCEL_EVENTS:
        _LOGGER.debug("%s: Setting cancel event", entity_id)
        FADE_CANCEL_EVENTS[entity_id].set()

    # Cancel the task
    if task.done():
        _LOGGER.debug("%s: Task already done", entity_id)
    else:
        _LOGGER.debug("%s: Cancelling task", entity_id)
        task.cancel()

    # Wait for cleanup using Condition
    if condition:
        async with condition:
            try:
                await asyncio.wait_for(
                    condition.wait_for(lambda: entity_id not in ACTIVE_FADES),
                    timeout=FADE_CANCEL_TIMEOUT_S,
                )
                _LOGGER.debug("%s: Task cleanup complete", entity_id)
            except TimeoutError:
                _LOGGER.debug("%s: Timed out waiting for fade task cleanup", entity_id)

    await _wait_until_stale_events_flushed(entity_id)


def _add_expected_values(entity_id: str, values: ExpectedValues) -> None:
    """Register expected values before making a service call."""
    if entity_id not in FADE_EXPECTED_STATE:
        FADE_EXPECTED_STATE[entity_id] = ExpectedState(entity_id=entity_id)
    FADE_EXPECTED_STATE[entity_id].add(values)


def _add_expected_brightness(entity_id: str, brightness: int) -> None:
    """Register an expected brightness value (convenience wrapper)."""
    _add_expected_values(entity_id, ExpectedValues(brightness=brightness))


async def _wait_until_stale_events_flushed(
    entity_id: str,
    timeout: float = 5.0,
) -> None:
    """Wait until all expected brightness values have been confirmed via state changes."""
    expected_state = FADE_EXPECTED_STATE.get(entity_id)
    if not expected_state or expected_state.is_empty:
        return

    _LOGGER.debug(
        "%s: waiting for expected state events to be flushed (remaining: %d)",
        entity_id,
        len(expected_state.values),
    )
    condition = expected_state.get_condition()
    try:
        async with condition:
            await asyncio.wait_for(
                condition.wait_for(lambda: expected_state.is_empty),
                timeout=timeout,
            )
    except TimeoutError:
        _LOGGER.warning(
            "%s: Timed out waiting for state events to flush (remaining: %d)",
            entity_id,
            len(expected_state.values),
        )


# =============================================================================
# Storage Helpers
# =============================================================================


def _get_light_config(hass: HomeAssistant, entity_id: str) -> dict[str, Any]:
    """Get per-light configuration.

    Returns the config dict for the light, or empty dict if not configured.
    """
    return hass.data.get(DOMAIN, {}).get("data", {}).get(entity_id, {})


def _get_orig_brightness(hass: HomeAssistant, entity_id: str) -> int:
    """Get stored original brightness for an entity."""
    return _get_light_config(hass, entity_id).get("orig_brightness", 0)


def _store_orig_brightness(hass: HomeAssistant, entity_id: str, level: int) -> None:
    """Store original brightness for an entity."""
    if DOMAIN not in hass.data:
        return
    data = hass.data[DOMAIN]["data"]
    if entity_id not in data:
        data[entity_id] = {}
    data[entity_id]["orig_brightness"] = level


async def _save_storage(hass: HomeAssistant) -> None:
    """Save storage data to disk."""
    if DOMAIN not in hass.data:
        return
    store: Store = hass.data[DOMAIN]["store"]
    await store.async_save(hass.data[DOMAIN]["data"])
