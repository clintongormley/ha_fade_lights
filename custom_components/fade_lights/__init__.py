"""The Fade Lights integration."""

from __future__ import annotations

import asyncio
import contextlib
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

# Maps entity_id -> expected brightness (0-255) during an active fade.
# This is critical for detecting physical switch changes: when a state change
# event arrives, we compare actual brightness to expected. If they differ by
# more than ±3 (tolerance for device rounding), we treat it as manual intervention
# even if the event has our context ID (which can happen with some integrations).
FADE_EXPECTED_BRIGHTNESS: dict[str, int] = {}

# Set of context IDs created by this integration's service calls.
# When we call light.turn_on/turn_off, Home Assistant creates state change events
# with the context we provided. By tracking our context IDs, we can identify
# which state changes are from our own operations vs external sources.
ACTIVE_CONTEXTS: set[str] = set()

# Set of entity_ids that had fades recently cancelled (not completed normally).
# This handles a race condition: when a fade is cancelled, its task cleanup may
# remove the entity from ACTIVE_FADES before all pending state change events
# from the fade's service calls have been processed. Without this set, those
# late-arriving events would trigger the "store new brightness as original" logic
# in Case 3, overwriting the pre-fade original brightness we want to preserve.
# Entities are added when cancelled and removed on next OFF->ON transition.
RECENTLY_CANCELLED_FADES: set[str] = set()


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
                        min_step_delay_ms,
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

    # =========================================================================
    # State Change Handler
    # =========================================================================
    # This callback fires for EVERY state change in Home Assistant that affects
    # lights. It handles three main scenarios:
    #
    # 1. Light turned ON from OFF -> Possibly restore original brightness
    # 2. Light turned OFF -> Cancel any active fade, mark for no-restore
    # 3. Brightness changed while ON -> Cancel fade and/or update orig brightness
    #
    # The tricky part is distinguishing OUR changes (from fade operations) from
    # EXTERNAL changes (user via app, physical switch, automations, etc.)
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

        # First pass: check if this event came from our own service calls.
        # We track our context IDs in ACTIVE_CONTEXTS when making calls.
        is_our_context = event.context.id in ACTIVE_CONTEXTS

        _LOGGER.debug(
            "(%s) -> state=%s, brightness=%s, is_fading=%s, is_our_context=%s, context_id=%s, active=%s",
            entity_id,
            new_state.state,
            new_brightness,
            is_fading,
            is_our_context,
            event.context.id,
            ACTIVE_CONTEXTS,
        )

        # Track if second-pass detection overrode is_our_context. This indicates
        # the event came from our context but had unexpected brightness, which
        # happens with late-arriving fade events or physical switch changes that
        # inherited our context. In these cases, we should NOT store the brightness.
        detected_via_brightness_mismatch = False

        # Second pass: detect physical switch changes during active fades.
        #
        # Problem: Physical switches (and some integrations) don't create new
        # contexts - they may inherit or use the existing context. So we can't
        # rely solely on context.id to detect manual changes.
        #
        # Solution: During a fade, we track what brightness we EXPECT the light
        # to be at (FADE_EXPECTED_BRIGHTNESS). If the actual brightness differs
        # by more than the tolerance (±3 for device rounding), someone changed
        # it manually - even if the context ID matches ours.
        if is_fading and entity_id in FADE_EXPECTED_BRIGHTNESS:
            expected = FADE_EXPECTED_BRIGHTNESS[entity_id]
            tolerance = 3

            # Two cases indicate manual intervention:
            # 1. Light turned OFF unexpectedly (and we weren't fading to 0)
            # 2. Brightness differs from expected by more than tolerance
            if (new_state.state == STATE_OFF and expected != 0) or (
                new_brightness is not None and abs(new_brightness - expected) > tolerance
            ):
                _LOGGER.debug(
                    "(%s) -> Unexpected state during fade: expected=%s, got=%s, manual",
                    entity_id,
                    expected,
                    new_brightness,
                )
                detected_via_brightness_mismatch = is_our_context
                is_our_context = False  # Treat as external change

        # If this is definitely our own change, ignore it - no action needed

        if is_our_context:
            _LOGGER.debug(
                "(%s) -> Ignoring (our context), state: %s, brightness: %s",
                entity_id,
                new_state.state,
                new_brightness,
            )
            return

        # =====================================================================
        # Case 1: Light turned ON from OFF - possibly restore brightness
        # =====================================================================
        old_state = event.data.get("old_state")
        if old_state and old_state.state == STATE_OFF and new_state.state == STATE_ON:
            _LOGGER.debug("(%s) -> Light turned OFF->ON", entity_id)

            # Skip restoration if this is a late-arriving event from a cancelled fade.
            # When the user turns off a light during a fade, there may be in-flight
            # service calls that complete after cancellation, causing spurious ON events.
            if entity_id in RECENTLY_CANCELLED_FADES:
                _LOGGER.debug("(%s) -> Skipping restore (recently cancelled fade)", entity_id)
                RECENTLY_CANCELLED_FADES.discard(entity_id)
                return

            # Clear the recently-cancelled flag now that the light is back on.
            # This allows future manual brightness changes to be stored as new
            # original brightness values (until the next fade).
            RECENTLY_CANCELLED_FADES.discard(entity_id)

            # Non-dimmable lights can't have brightness restored
            if ColorMode.BRIGHTNESS not in new_state.attributes.get(ATTR_SUPPORTED_COLOR_MODES, []):
                return

            # Restore to the stored original brightness if we have one.
            # The orig_brightness is set at the START of a fade and preserved even
            # if the fade is interrupted. This ensures turning the light back on
            # restores to pre-fade brightness, not some intermediate value.
            orig_brightness = _get_orig_brightness(hass, entity_id)
            _LOGGER.debug(
                "(%s) -> orig_brightness=%s, current_brightness=%s",
                entity_id,
                orig_brightness,
                new_brightness,
            )
            if orig_brightness > 0:
                current_brightness = new_state.attributes.get(ATTR_BRIGHTNESS, 0)
                # Only restore if current brightness differs from original.
                # Note: We can't distinguish "turn on only" from "turn on with brightness"
                # so we always restore. This is a known limitation documented in README.
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

        # =====================================================================
        # Case 2: Light turned OFF - cancel fade
        # =====================================================================
        if new_state.state == STATE_OFF:
            if entity_id in ACTIVE_FADES:
                _LOGGER.debug("(%s) -> Cancelling fade due to manual turn off", entity_id)

                # Mark as recently cancelled. This prevents late-arriving state change
                # events from the fade's own service calls from being stored as orig
                # brightness in Case 3. Cleared when light turns back on in Case 1.
                RECENTLY_CANCELLED_FADES.add(entity_id)

                # Signal the fade loop to stop and cancel the task
                if entity_id in FADE_CANCEL_EVENTS:
                    FADE_CANCEL_EVENTS[entity_id].set()
                ACTIVE_FADES[entity_id].cancel()

                # Race condition protection: The fade loop might be mid-service-call
                # when we cancel it. If it's calling turn_on with low brightness,
                # that call might complete AFTER we detect the manual turn-off.
                # Sending an explicit turn_off ensures the light ends up off.
                hass.async_create_task(
                    hass.services.async_call(
                        LIGHT_DOMAIN,
                        SERVICE_TURN_OFF,
                        {ATTR_ENTITY_ID: entity_id},
                    )
                )
            return

        # =====================================================================
        # Case 3: Brightness changed while light is ON - update orig or cancel
        # =====================================================================
        if old_state and old_state.state == STATE_ON and new_state.state == STATE_ON:
            old_brightness = old_state.attributes.get(ATTR_BRIGHTNESS)

            if new_brightness != old_brightness and new_brightness is not None:
                # Cancel active fade if any
                if is_fading:
                    _LOGGER.debug("(%s) -> Cancelling fade due to manual change", entity_id)

                    if entity_id in FADE_CANCEL_EVENTS:
                        FADE_CANCEL_EVENTS[entity_id].set()
                    ACTIVE_FADES[entity_id].cancel()

                # Store new brightness as original - but only if this is a genuine
                # user action, not a late-arriving event from our fade.
                #
                # Skip storing if:
                # 1. detected_via_brightness_mismatch: Event came from our context
                #    but had unexpected brightness (late fade event or physical switch
                #    that inherited our context)
                # 2. RECENTLY_CANCELLED_FADES: Catch late events that arrive after
                #    the fade task cleaned up but before we clear the flag
                should_skip = (
                    detected_via_brightness_mismatch
                    or entity_id in RECENTLY_CANCELLED_FADES
                )
                if not should_skip:
                    _store_orig_brightness(hass, entity_id, new_brightness)

    entry.async_on_unload(hass.bus.async_listen(EVENT_STATE_CHANGED, handle_light_state_change))
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
    RECENTLY_CANCELLED_FADES.clear()

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
    context: Context,
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
    if entity_id in ACTIVE_FADES:
        if entity_id in FADE_CANCEL_EVENTS:
            FADE_CANCEL_EVENTS[entity_id].set()
        ACTIVE_FADES[entity_id].cancel()
        # Wait for the old fade to actually stop before starting the new one
        with contextlib.suppress(asyncio.CancelledError):
            await ACTIVE_FADES[entity_id]

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
            hass, entity_id, brightness_pct, transition_ms, context, min_step_delay_ms, cancel_event
        )
    except asyncio.CancelledError:
        pass  # Normal cancellation, not an error
    finally:
        # Clean up tracking state regardless of how the fade ended
        ACTIVE_FADES.pop(entity_id, None)
        FADE_CANCEL_EVENTS.pop(entity_id, None)
        FADE_EXPECTED_BRIGHTNESS.pop(entity_id, None)


async def _execute_fade(
    hass: HomeAssistant,
    entity_id: str,
    brightness_pct: int,
    transition_ms: int,
    context: Context,
    min_step_delay_ms: int,
    cancel_event: asyncio.Event,
) -> None:
    """Execute the fade operation.

    This function contains the core fade logic:
    1. Validate entity and handle non-dimmable lights
    2. Store original brightness for later restoration
    3. Calculate step count and timing
    4. Execute the fade loop with cancellation checks
    """
    state = hass.states.get(entity_id)
    if not state:
        _LOGGER.warning("Entity %s not found", entity_id)
        return

    # Handle non-dimmable lights (on/off only, no brightness control)
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

    brightness = state.attributes.get(ATTR_BRIGHTNESS)
    start_level = int(brightness) if brightness is not None else 0
    end_level = int(brightness_pct / 100 * 255)

    if start_level == end_level:
        return

    # -------------------------------------------------------------------------
    # Original Brightness Preservation
    # -------------------------------------------------------------------------
    # Store the starting brightness as "orig" ONLY if we don't already have one.
    # This handles the situation where one fade is interrupted by a second fade
    # The start_level should be the current brightness, but the original_brightness
    # should be taken from before the first fade starts.
    #
    # Scenario: Light at 200, fade to 50 starts
    #  - We store orig=200 at fade start
    #  - Light reaches 100 then the fade is interrupted by a new fade to 255
    #  - The new fade doesn't update orig, but leaves it at 200

    existing_orig = _get_orig_brightness(hass, entity_id)
    if existing_orig == 0 and start_level > 0:
        _store_orig_brightness(hass, entity_id, start_level)

    # Track starting brightness for delta calculation during the fade loop
    FADE_EXPECTED_BRIGHTNESS[entity_id] = start_level

    # -------------------------------------------------------------------------
    # Step Calculation
    # -------------------------------------------------------------------------
    # Goal: Complete the fade in transition_ms, with smooth steps.
    #
    # Constraints:
    # - Each step must take at least min_step_delay_ms (default 50ms)
    # - Each step changes brightness by at least 1 level
    # - We must hit the exact target at the end
    #
    # Example: Fade from 200 to 100 (100 levels) over 5000ms with 50ms min delay
    # - max_steps_by_time = 5000 / 50 = 100 steps
    # - level_diff = 100 levels
    # - We can do 100 steps of 1 level each, 50ms apart
    level_diff = abs(end_level - start_level)
    if level_diff == 0:
        return

    # Maximum steps we can fit in the transition time (given minimum delay per step)
    max_steps_by_time = max(1, int(transition_ms / min_step_delay_ms))

    # We can't have more steps than brightness levels to change
    num_steps = min(max_steps_by_time, level_diff)

    # Calculate how much brightness changes per step (must be at least 1).
    # Round up for increases, down for decreases, to ensure we don't undershoot.
    delta = (end_level - start_level) / num_steps
    delta = math.ceil(delta) if delta > 0 else math.floor(delta)

    # Recalculate actual steps needed based on rounded delta.
    # This ensures we don't overshoot or have extra steps at the end.
    actual_steps = math.ceil(level_diff / abs(delta))

    # Calculate delay between steps to spread them evenly across transition time
    delay_ms = transition_ms / actual_steps
    num_steps = actual_steps

    _LOGGER.debug(
        "Fading %s from %s to %s in %s steps", entity_id, start_level, end_level, num_steps
    )

    # -------------------------------------------------------------------------
    # Fade Loop
    # -------------------------------------------------------------------------
    # Each iteration:
    # 1. Check for cancellation (from manual change or new fade)
    # 2. Calculate and apply the next brightness level
    # 3. Wait for the appropriate delay
    for i in range(num_steps):
        step_start = time.monotonic()

        # Check for cancellation at start of each step.
        # This catches cancellations that happened during the previous sleep.
        if cancel_event.is_set():
            return

        # Calculate next brightness level based on our tracked expected value.
        # We use FADE_EXPECTED_BRIGHTNESS rather than reading from entity state because:
        # - Entity state might be stale (async update)
        # - Manual changes would throw off our delta calculation
        curr = FADE_EXPECTED_BRIGHTNESS[entity_id]
        new_level = curr + delta
        new_level = max(0, min(255, new_level))

        # Ensure we hit the target on the last step (avoid off-by-one from rounding)
        if (delta > 0 and new_level > end_level) or (delta < 0 and new_level < end_level):
            new_level = end_level

        if i == num_steps - 1:
            new_level = end_level

        # Handle brightness level 1 edge case.
        # Many lights treat brightness=1 oddly (some turn off, some go to min).
        # Skip level 1 entirely by going to 0 (off) or 2 depending on direction.
        if new_level == 1:
            new_level = 0 if delta < 0 else 2

        # Track expected brightness for both delta calculation and physical switch detection.
        # The state change handler compares actual vs expected to detect
        # manual intervention even when context ID matches ours.
        FADE_EXPECTED_BRIGHTNESS[entity_id] = new_level

        # Apply the brightness change
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

        # Check for cancellation after service call completes.
        # The service call might have triggered a state change that cancelled us.
        if cancel_event.is_set():
            return

        # Sleep for remaining time in this step.
        # We subtract elapsed time to maintain consistent fade duration
        # regardless of how long the service call took.
        elapsed_ms = (time.monotonic() - step_start) * 1000
        sleep_ms = max(0, delay_ms - elapsed_ms)
        if sleep_ms > 0:
            await asyncio.sleep(sleep_ms / 1000)

    # -------------------------------------------------------------------------
    # Fade Complete
    # -------------------------------------------------------------------------
    # On successful completion to a non-zero brightness, update the stored
    # orig_brightness to the new level. This is the user's new "intended"
    # brightness for future restoration.
    #
    # If we faded to 0 (off), we keep the previous orig_brightness so that
    # turning the light back on restores to where it was before the fade.
    if end_level > 0:
        _store_orig_brightness(hass, entity_id, end_level)
        await _save_storage(hass)


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
