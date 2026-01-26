"""The Fade Lights integration."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field

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
    State,
    callback,
)
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers.storage import Store
from homeassistant.helpers.typing import ConfigType

from .const import (
    ATTR_BRIGHTNESS_PCT,
    ATTR_TRANSITION,
    BRIGHTNESS_TOLERANCE,
    DEFAULT_BRIGHTNESS_PCT,
    DEFAULT_MIN_STEP_DELAY_MS,
    DEFAULT_TRANSITION,
    DOMAIN,
    FADE_CANCEL_TIMEOUT_S,
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

@dataclass
class ExpectedState:
    """Track expected brightness values and provide synchronization for waiting."""

    values: dict[int, float] = field(default_factory=dict)  # brightness -> timestamp
    _condition: asyncio.Condition | None = field(default=None, repr=False)

    def add(self, brightness: int) -> None:
        """Add an expected brightness value with current timestamp."""
        self.values[brightness] = time.monotonic()

    def get_condition(self) -> asyncio.Condition:
        """Get or create the condition for waiting."""
        if self._condition is None:
            self._condition = asyncio.Condition()
        return self._condition

    def match_and_remove(
        self,
        state: str,
        brightness: int | None,
        tolerance: int,
    ) -> int | None:
        """Match state against expected values, remove if found, notify if empty.

        Args:
            state: The light state (STATE_ON or STATE_OFF)
            brightness: The brightness value from the state (None if off)
            tolerance: Brightness tolerance for matching

        Returns:
            The matched brightness value, or None if no match.
        """
        matched_value: int | None = None

        # Check for OFF match
        if state == STATE_OFF and 0 in self.values:
            matched_value = 0
        # Check for brightness match with tolerance
        elif state == STATE_ON and brightness is not None:
            for expected in self.values:
                if expected > 0 and abs(brightness - expected) <= tolerance:
                    matched_value = expected
                    break

        if matched_value is None:
            return None

        # Remove matched value
        del self.values[matched_value]

        # Notify condition if set is now empty
        if not self.values and self._condition is not None:
            # Schedule notification (can't await in callback context)
            asyncio.get_event_loop().call_soon(
                lambda c=self._condition: asyncio.create_task(self._notify(c))
            )

        return matched_value

    def prune(self, threshold: float = 5.0) -> None:
        """Remove values older than threshold seconds."""
        now = time.monotonic()
        stale_keys = [
            brightness
            for brightness, timestamp in self.values.items()
            if now - timestamp > threshold
        ]
        for key in stale_keys:
            del self.values[key]

    @property
    def is_empty(self) -> bool:
        """Check if there are no expected values."""
        return not self.values

    @staticmethod
    async def _notify(condition: asyncio.Condition) -> None:
        """Notify all waiters on the condition."""
        async with condition:
            condition.notify_all()


# Maps entity_id -> ExpectedState for tracking expected brightness during fades
# and waiting for state change events after service calls.
FADE_EXPECTED_BRIGHTNESS: dict[str, ExpectedState] = {}

# Maps entity_id -> True when manual intervention was just detected.
# Used to suppress stale state events from the cancelled fade. When a user
# manually changes a light during a fade, delayed state events from previous
# fade steps may still arrive. Without this flag, those events could trigger
# unintended brightness restoration. Cleared after _cancel_and_wait_for_fade.
FADE_INTERRUPTED: dict[str, bool] = {}

# Maps entity_id -> asyncio.Condition to signal when fade cleanup completes.
# Waiters use this instead of polling to know when a cancelled fade has finished.
FADE_COMPLETE_CONDITIONS: dict[str, asyncio.Condition] = {}


async def async_setup(hass: HomeAssistant, _config: ConfigType) -> bool:
    """Set up the Fade Lights component."""
    if not hass.config_entries.async_entries(DOMAIN):
        hass.async_create_task(
            hass.config_entries.flow.async_init(DOMAIN, context={"source": "import"})
        )
    return True


# =============================================================================
# Service and Event Handlers
# =============================================================================


async def _handle_fade_lights(hass: HomeAssistant, call: ServiceCall) -> None:
    """Handle the fade_lights service call."""
    domain_data = hass.data.get(DOMAIN, {})
    default_brightness = domain_data.get("default_brightness", DEFAULT_BRIGHTNESS_PCT)
    default_transition = domain_data.get("default_transition", DEFAULT_TRANSITION)
    min_step_delay_ms = domain_data.get("min_step_delay_ms", DEFAULT_MIN_STEP_DELAY_MS)

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


@callback
def _handle_light_state_change(hass: HomeAssistant, event: Event) -> None:
    """Handle light state changes - detects manual intervention and tracks brightness."""
    new_state: State | None = event.data.get("new_state")
    old_state: State | None = event.data.get("old_state")

    if not _should_process_state_change(new_state):
        return

    # Type narrowing: new_state is guaranteed non-None after _should_process_state_change
    assert new_state is not None

    entity_id = new_state.entity_id

    if _is_stale_event(entity_id, new_state):
        return

    _log_state_change(entity_id, new_state)

    # Check if this is an expected state change (from our service calls)
    if _match_and_remove_expected(entity_id, new_state):
        _LOGGER.debug("(%s) -> State matches expected, removed from tracking", entity_id)
        return

    # During fade: if we get here, state didn't match expected - manual intervention
    if entity_id in ACTIVE_FADES and entity_id in FADE_EXPECTED_BRIGHTNESS:
        # Manual intervention detected
        _LOGGER.debug(
            "(%s) -> Manual intervention during fade: got=%s/%s",
            entity_id,
            new_state.state,
            new_state.attributes.get(ATTR_BRIGHTNESS),
        )
        FADE_INTERRUPTED[entity_id] = True
        hass.async_create_task(_restore_manual_state(hass, entity_id, old_state, new_state))
        return

    # Normal state handling (no active fade)
    if _is_off_to_on_transition(old_state, new_state):
        _handle_off_to_on(hass, entity_id, new_state)
        return

    if _is_brightness_change(old_state, new_state):
        new_brightness = new_state.attributes.get(ATTR_BRIGHTNESS)
        _LOGGER.debug("(%s) -> Storing new brightness as original: %s", entity_id, new_brightness)
        _store_orig_brightness(hass, entity_id, new_brightness)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Fade Lights from a config entry."""
    store = Store(hass, 1, STORAGE_KEY)
    storage_data = await store.async_load() or {}

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN] = {
        "store": store,
        "data": storage_data,
        "default_brightness": entry.options.get(
            OPTION_DEFAULT_BRIGHTNESS_PCT, DEFAULT_BRIGHTNESS_PCT
        ),
        "default_transition": entry.options.get(OPTION_DEFAULT_TRANSITION, DEFAULT_TRANSITION),
        "min_step_delay_ms": entry.options.get(OPTION_MIN_STEP_DELAY_MS, DEFAULT_MIN_STEP_DELAY_MS),
    }

    async def handle_fade_lights(call: ServiceCall) -> None:
        """Service handler wrapper."""
        await _handle_fade_lights(hass, call)

    @callback
    def handle_light_state_change(event: Event) -> None:
        """Event handler wrapper."""
        _handle_light_state_change(hass, event)

    hass.services.async_register(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        handle_fade_lights,
        schema=None,
    )

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

    tasks = list(ACTIVE_FADES.values())
    for task in tasks:
        task.cancel()

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    ACTIVE_FADES.clear()
    FADE_CANCEL_EVENTS.clear()
    FADE_EXPECTED_BRIGHTNESS.clear()
    FADE_INTERRUPTED.clear()
    FADE_COMPLETE_CONDITIONS.clear()

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

    # Create completion condition for waiters to know when cleanup is done
    complete_condition = asyncio.Condition()
    FADE_COMPLETE_CONDITIONS[entity_id] = complete_condition

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
        # Note: FADE_EXPECTED_BRIGHTNESS is NOT cleared here - values persist
        # for event matching and are pruned when next fade starts

        # Notify any waiters that cleanup is complete
        condition = FADE_COMPLETE_CONDITIONS.pop(entity_id, None)
        if condition:
            async with condition:
                condition.notify_all()


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
# Fade Loop
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
    _add_expected_brightness(entity_id, start_level)
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
        _add_expected_brightness(entity_id, current_level)

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
    if DOMAIN not in hass.data:
        return
    hass.data[DOMAIN]["data"][entity_id] = level


async def _save_storage(hass: HomeAssistant) -> None:
    """Save storage data to disk."""
    if DOMAIN not in hass.data:
        return
    store: Store = hass.data[DOMAIN]["store"]
    await store.async_save(hass.data[DOMAIN]["data"])


# =============================================================================
# State Change Helpers
# =============================================================================


# --- Event Filtering ---


def _should_process_state_change(new_state: State | None) -> bool:
    """Check if this state change should be processed."""
    if not new_state:
        return False
    if new_state.domain != LIGHT_DOMAIN:
        return False
    # Ignore group helpers (lights that contain other lights)
    return new_state.attributes.get(ATTR_ENTITY_ID) is None


def _is_stale_event(entity_id: str, new_state: State) -> bool:
    """Check if event should be suppressed during fade cleanup."""
    if entity_id not in FADE_INTERRUPTED:
        return False
    _LOGGER.debug(
        "(%s) -> Ignoring stale event during fade cleanup (%s/%s)",
        entity_id,
        new_state.state,
        new_state.attributes.get(ATTR_BRIGHTNESS),
    )
    return True


def _log_state_change(entity_id: str, new_state: State) -> None:
    """Log state change details."""
    _LOGGER.debug(
        "(%s) -> state=%s, brightness=%s, is_fading=%s",
        entity_id,
        new_state.state,
        new_state.attributes.get(ATTR_BRIGHTNESS),
        entity_id in ACTIVE_FADES,
    )


# --- Expected State Matching ---


def _match_and_remove_expected(entity_id: str, new_state: State) -> bool:
    """Check if state matches expected, remove if found, notify if empty.

    Returns True if this was an expected state change (caller should ignore it).
    """
    expected_state = FADE_EXPECTED_BRIGHTNESS.get(entity_id)
    if not expected_state or expected_state.is_empty:
        return False

    new_brightness = new_state.attributes.get(ATTR_BRIGHTNESS)
    matched = expected_state.match_and_remove(
        new_state.state, new_brightness, BRIGHTNESS_TOLERANCE
    )

    if matched is not None:
        _LOGGER.debug(
            "(%s) -> Matched expected brightness %s, remaining: %s",
            entity_id,
            matched,
            list(expected_state.values.keys()),
        )
        return True

    return False


# --- State Transition Predicates ---


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


# --- State Transition Handlers ---


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


def _handle_off_to_on(hass: HomeAssistant, entity_id: str, new_state: State) -> None:
    """Handle OFF -> ON transition by restoring original brightness."""
    if DOMAIN not in hass.data:
        return
    _LOGGER.debug("(%s) -> Light turned OFF->ON", entity_id)

    # Non-dimmable lights can't have brightness restored
    if ColorMode.BRIGHTNESS not in new_state.attributes.get(ATTR_SUPPORTED_COLOR_MODES, []):
        return

    orig_brightness = _get_orig_brightness(hass, entity_id)
    current_brightness = new_state.attributes.get(ATTR_BRIGHTNESS, 0)

    _LOGGER.debug(
        "(%s) -> orig_brightness=%s, current_brightness=%s",
        entity_id,
        orig_brightness,
        current_brightness,
    )

    if orig_brightness > 0 and current_brightness != orig_brightness:
        _LOGGER.debug("(%s) -> Restoring to brightness %s", entity_id, orig_brightness)
        hass.async_create_task(
            _restore_original_brightness(hass, entity_id, orig_brightness)
        )


# --- Fade Cancellation ---


async def _cancel_and_wait_for_fade(entity_id: str) -> None:
    """Cancel any active fade for entity and wait for cleanup.

    This function cancels an active fade task and waits for it to be fully
    removed from ACTIVE_FADES using an asyncio.Condition for efficient
    notification instead of polling.
    """
    _LOGGER.debug("(%s) -> Cancelling fade", entity_id)
    if entity_id not in ACTIVE_FADES:
        _LOGGER.debug("  -> Fade not in ACTIVE_FADES")
        return

    task = ACTIVE_FADES[entity_id]
    condition = FADE_COMPLETE_CONDITIONS.get(entity_id)

    # Signal cancellation via the cancel event if available
    if entity_id in FADE_CANCEL_EVENTS:
        _LOGGER.debug("  -> Setting cancel event")
        FADE_CANCEL_EVENTS[entity_id].set()

    # Cancel the task
    if task.done():
        _LOGGER.debug("  -> Task already done")
    else:
        _LOGGER.debug("  -> Cancelling task")
        task.cancel()

    # Wait for cleanup using Condition
    if condition:
        async with condition:
            try:
                await asyncio.wait_for(
                    condition.wait_for(lambda: entity_id not in ACTIVE_FADES),
                    timeout=FADE_CANCEL_TIMEOUT_S,
                )
                _LOGGER.debug("  -> Task cleanup complete")
            except TimeoutError:
                _LOGGER.debug("(%s) -> Timed out waiting for fade task cleanup", entity_id)

    await _wait_until_stale_events_flushed(entity_id)


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
    if DOMAIN not in hass.data:
        _clear_fade_interrupted(entity_id)
        return
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
        _add_expected_brightness(entity_id, 0)
        await hass.services.async_call(
            LIGHT_DOMAIN,
            SERVICE_TURN_OFF,
            {ATTR_ENTITY_ID: entity_id},
            blocking=True,
        )
        await _wait_until_stale_events_flushed(entity_id)
    elif intended > 0 and current != intended:
        _LOGGER.debug("(%s) -> setting light brightness (%s) as intended", entity_id, intended)
        _add_expected_brightness(entity_id, intended)
        await hass.services.async_call(
            LIGHT_DOMAIN,
            SERVICE_TURN_ON,
            {ATTR_ENTITY_ID: entity_id, ATTR_BRIGHTNESS: intended},
            blocking=True,
        )
        await _wait_until_stale_events_flushed(entity_id)

    _clear_fade_interrupted(entity_id)


def _clear_fade_interrupted(entity_id: str) -> None:
    """Clear the interrupted flag after fade cleanup is complete."""
    _LOGGER.debug("(%s) -> Clearing FADE_INTERRUPTED", entity_id)
    FADE_INTERRUPTED.pop(entity_id, None)


def _add_expected_brightness(entity_id: str, brightness: int) -> None:
    """Register an expected brightness value before making a service call."""
    if entity_id not in FADE_EXPECTED_BRIGHTNESS:
        FADE_EXPECTED_BRIGHTNESS[entity_id] = ExpectedState()
    FADE_EXPECTED_BRIGHTNESS[entity_id].add(brightness)


async def _wait_until_stale_events_flushed(
    entity_id: str,
    timeout: float = 5.0,
) -> None:
    """Wait until all expected brightness values have been confirmed via state changes."""
    expected_state = FADE_EXPECTED_BRIGHTNESS.get(entity_id)
    if not expected_state or expected_state.is_empty:
        return

    condition = expected_state.get_condition()
    try:
        async with condition:
            await asyncio.wait_for(
                condition.wait_for(lambda: expected_state.is_empty),
                timeout=timeout,
            )
    except TimeoutError:
        _LOGGER.warning(
            "(%s) Timed out waiting for state events to flush (remaining: %s)",
            entity_id,
            list(expected_state.values.keys()),
        )


def _prune_expected_brightness(entity_id: str) -> None:
    """Remove stale expected values when starting a new fade."""
    expected_state = FADE_EXPECTED_BRIGHTNESS.get(entity_id)
    if not expected_state:
        return

    expected_state.prune()
    if expected_state.is_empty:
        FADE_EXPECTED_BRIGHTNESS.pop(entity_id, None)


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
