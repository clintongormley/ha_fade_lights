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
from homeassistant.util.color import (
    color_RGB_to_hs,
    color_rgbw_to_rgb,
    color_rgbww_to_rgb,
    color_xy_to_hs,
)

from .const import (
    ATTR_BRIGHTNESS_PCT,
    ATTR_COLOR_TEMP_KELVIN,
    ATTR_FROM,
    ATTR_HS_COLOR,
    ATTR_RGB_COLOR,
    ATTR_RGBW_COLOR,
    ATTR_RGBWW_COLOR,
    ATTR_TRANSITION,
    ATTR_XY_COLOR,
    BRIGHTNESS_TOLERANCE,
    COLOR_PARAMS,
    DEFAULT_BRIGHTNESS_PCT,
    DEFAULT_MIN_STEP_DELAY_MS,
    DEFAULT_TRANSITION,
    DOMAIN,
    FADE_CANCEL_TIMEOUT_S,
    OPTION_MIN_STEP_DELAY_MS,
    SERVICE_FADE_LIGHTS,
    STALE_THRESHOLD,
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

    entity_id: str
    values: dict[int, float] = field(default_factory=dict)  # brightness -> timestamp
    _condition: asyncio.Condition | None = field(default=None, repr=False)

    def add(self, brightness: int) -> None:
        """Add an expected brightness value with current timestamp."""
        self.values[brightness] = time.monotonic()
        _LOGGER.debug(
            "%s: ExpectedState.add(%s) -> values=%s",
            self.entity_id,
            brightness,
            list(self.values.keys()),
        )

    def get_condition(self) -> asyncio.Condition:
        """Get or create the condition for waiting, pruning stale values first."""
        _LOGGER.debug(
            "%s: ExpectedState.get_condition() values=%s",
            self.entity_id,
            list(self.values.keys()),
        )

        # Prune stale values
        now = time.monotonic()
        stale_keys = [
            brightness
            for brightness, timestamp in self.values.items()
            if now - timestamp > STALE_THRESHOLD
        ]
        if stale_keys:
            _LOGGER.debug(
                "%s: ExpectedState.get_condition() removing stale keys: %s",
                self.entity_id,
                stale_keys,
            )
        for key in stale_keys:
            del self.values[key]

        _LOGGER.debug(
            "%s: ExpectedState.get_condition() after prune=%s",
            self.entity_id,
            list(self.values.keys()),
        )

        if self._condition is None:
            self._condition = asyncio.Condition()

        # Notify if all values were pruned
        if not self.values:
            _LOGGER.debug(
                "%s: ExpectedState.get_condition -> values empty, notifying condition",
                self.entity_id,
            )
            asyncio.get_event_loop().call_soon(
                lambda c=self._condition: asyncio.create_task(self._notify(c))
            )

        return self._condition

    def match_and_remove(self, brightness: int) -> int | None:
        """Match brightness against expected values, remove if found, notify if empty.

        Args:
            brightness: The brightness value to match (0 for off)

        Returns:
            The matched brightness value, or None if no match.
        """
        _LOGGER.debug(
            "%s: ExpectedState.match_and_remove(%s) values=%s",
            self.entity_id,
            brightness,
            list(self.values.keys()),
        )
        matched_value: int | None = None

        if brightness == 0:
            if 0 in self.values:
                matched_value = 0
        else:
            for expected in self.values:
                if abs(brightness - expected) <= BRIGHTNESS_TOLERANCE:
                    matched_value = expected
                    break

        if matched_value is None:
            _LOGGER.debug(
                "%s: ExpectedState.match_and_remove(%s) -> no match found",
                self.entity_id,
                brightness,
            )
            return None

        # Remove matched value
        del self.values[matched_value]
        _LOGGER.debug(
            "%s: ExpectedState.match_and_remove(%s) matched=%s now=%s",
            self.entity_id,
            brightness,
            matched_value,
            list(self.values.keys()),
        )

        # Notify condition if set is now empty
        if not self.values and self._condition is not None:
            _LOGGER.debug(
                "%s: ExpectedState.match_and_remove -> values empty, notifying condition",
                self.entity_id,
            )
            # Schedule notification (can't await in callback context)
            asyncio.get_event_loop().call_soon(
                lambda c=self._condition: asyncio.create_task(self._notify(c))
            )

        return matched_value

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

# Maps entity_id -> asyncio.Condition to signal when fade cleanup completes.
# Waiters use this instead of polling to know when a cancelled fade has finished.
FADE_COMPLETE_CONDITIONS: dict[str, asyncio.Condition] = {}


@dataclass
class FadeParams:
    """Normalized parameters for a fade operation.

    All color inputs are converted to internal representations:
    - RGB/RGBW/RGBWW/XY colors -> hs_color (hue 0-360, saturation 0-100)
    - color_temp_kelvin -> color_temp_mireds
    """

    brightness_pct: int | None = None
    hs_color: tuple[float, float] | None = None  # (hue, saturation)
    color_temp_mireds: int | None = None
    transition_ms: int = DEFAULT_TRANSITION * 1000

    # Starting values (from: parameter)
    from_brightness_pct: int | None = None
    from_hs_color: tuple[float, float] | None = None
    from_color_temp_mireds: int | None = None


def _validate_color_params(data: dict) -> None:
    """Validate that at most one color parameter is specified.

    Also validates the from: parameter if present.

    Raises:
        ServiceValidationError: If multiple color parameters are provided.
    """
    # Validate main params
    specified = [param for param in COLOR_PARAMS if param in data]
    if len(specified) > 1:
        raise ServiceValidationError(
            f"Only one color parameter allowed, got: {', '.join(sorted(specified))}"
        )

    # Validate from: params
    from_data = data.get(ATTR_FROM, {})
    if from_data:
        from_specified = [param for param in COLOR_PARAMS if param in from_data]
        if len(from_specified) > 1:
            params = ", ".join(sorted(from_specified))
            raise ServiceValidationError(
                f"Only one color parameter allowed in 'from:', got: {params}"
            )


def _validate_color_ranges(data: dict) -> None:
    """Validate color parameter value ranges.

    Raises:
        ServiceValidationError: If values are out of valid ranges.
    """
    _validate_color_ranges_dict(data, "")

    from_data = data.get(ATTR_FROM, {})
    if from_data:
        _validate_color_ranges_dict(from_data, "from: ")


def _validate_color_ranges_dict(data: dict, prefix: str) -> None:
    """Validate color ranges in a single dict."""
    if ATTR_HS_COLOR in data:
        hs = data[ATTR_HS_COLOR]
        if not (0 <= hs[0] <= 360):
            raise ServiceValidationError(f"{prefix}Hue must be between 0 and 360, got {hs[0]}")
        if not (0 <= hs[1] <= 100):
            raise ServiceValidationError(
                f"{prefix}Saturation must be between 0 and 100, got {hs[1]}"
            )

    if ATTR_RGB_COLOR in data:
        rgb = data[ATTR_RGB_COLOR]
        for val in rgb[:3]:
            if not (0 <= val <= 255):
                raise ServiceValidationError(
                    f"{prefix}RGB values must be between 0 and 255, got {val}"
                )

    if ATTR_RGBW_COLOR in data:
        rgbw = data[ATTR_RGBW_COLOR]
        for val in rgbw[:4]:
            if not (0 <= val <= 255):
                raise ServiceValidationError(
                    f"{prefix}RGBW values must be between 0 and 255, got {val}"
                )

    if ATTR_RGBWW_COLOR in data:
        rgbww = data[ATTR_RGBWW_COLOR]
        for val in rgbww[:5]:
            if not (0 <= val <= 255):
                raise ServiceValidationError(
                    f"{prefix}RGBWW values must be between 0 and 255, got {val}"
                )

    if ATTR_XY_COLOR in data:
        xy = data[ATTR_XY_COLOR]
        for val in xy[:2]:
            if not (0 <= val <= 1):
                raise ServiceValidationError(
                    f"{prefix}XY values must be between 0 and 1, got {val}"
                )

    if ATTR_COLOR_TEMP_KELVIN in data:
        kelvin = data[ATTR_COLOR_TEMP_KELVIN]
        if not (1000 <= kelvin <= 40000):
            raise ServiceValidationError(
                f"{prefix}Color temp must be between 1000K and 40000K, got {kelvin}K"
            )


def _parse_color_params(data: dict) -> FadeParams:
    """Parse and convert color parameters to internal representation.

    Converts:
    - rgb_color, rgbw_color, rgbww_color, xy_color -> hs_color
    - color_temp_kelvin -> color_temp_mireds

    Also handles the 'from:' parameter for starting values.

    Args:
        data: Service call data dictionary

    Returns:
        FadeParams with normalized color values
    """
    params = FadeParams()

    # Parse brightness
    if ATTR_BRIGHTNESS_PCT in data:
        params.brightness_pct = int(data[ATTR_BRIGHTNESS_PCT])

    # Parse target color
    hs, mireds = _extract_color(data)
    params.hs_color = hs
    params.color_temp_mireds = mireds

    # Parse from: parameter
    from_data = data.get(ATTR_FROM, {})
    if from_data:
        if ATTR_BRIGHTNESS_PCT in from_data:
            params.from_brightness_pct = int(from_data[ATTR_BRIGHTNESS_PCT])

        from_hs, from_mireds = _extract_color(from_data)
        params.from_hs_color = from_hs
        params.from_color_temp_mireds = from_mireds

    return params


def _extract_color(data: dict) -> tuple[tuple[float, float] | None, int | None]:
    """Extract color from data dict, converting to HS or mireds.

    Returns:
        Tuple of (hs_color, color_temp_mireds) - one will be None
    """
    # Handle HS color (pass through)
    if ATTR_HS_COLOR in data:
        hs = data[ATTR_HS_COLOR]
        return (float(hs[0]), float(hs[1])), None

    # Handle RGB -> HS
    if ATTR_RGB_COLOR in data:
        rgb = data[ATTR_RGB_COLOR]
        hs = color_RGB_to_hs(rgb[0], rgb[1], rgb[2])
        return hs, None

    # Handle RGBW -> RGB -> HS
    if ATTR_RGBW_COLOR in data:
        rgbw = data[ATTR_RGBW_COLOR]
        rgb = color_rgbw_to_rgb(rgbw[0], rgbw[1], rgbw[2], rgbw[3])
        hs = color_RGB_to_hs(rgb[0], rgb[1], rgb[2])
        return hs, None

    # Handle RGBWW -> RGB -> HS
    if ATTR_RGBWW_COLOR in data:
        rgbww = data[ATTR_RGBWW_COLOR]
        rgb = color_rgbww_to_rgb(
            rgbww[0], rgbww[1], rgbww[2], rgbww[3], rgbww[4], min_kelvin=2700, max_kelvin=6500
        )
        hs = color_RGB_to_hs(rgb[0], rgb[1], rgb[2])
        return hs, None

    # Handle XY -> HS
    if ATTR_XY_COLOR in data:
        xy = data[ATTR_XY_COLOR]
        hs = color_xy_to_hs(xy[0], xy[1])
        return hs, None

    # Handle color temperature
    if ATTR_COLOR_TEMP_KELVIN in data:
        kelvin = data[ATTR_COLOR_TEMP_KELVIN]
        return None, int(1_000_000 / kelvin)

    return None, None


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
    store = Store(hass, 1, STORAGE_KEY)
    storage_data = await store.async_load() or {}

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN] = {
        "store": store,
        "data": storage_data,
        "default_brightness": DEFAULT_BRIGHTNESS_PCT,
        "default_transition": DEFAULT_TRANSITION,
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
    FADE_COMPLETE_CONDITIONS.clear()

    hass.services.async_remove(DOMAIN, SERVICE_FADE_LIGHTS)
    hass.data.pop(DOMAIN, None)

    return True


# =============================================================================
# Service Handler: fade_lights
# =============================================================================


async def _handle_fade_lights(hass: HomeAssistant, call: ServiceCall) -> None:
    """Handle the fade_lights service call."""
    domain_data = hass.data.get(DOMAIN, {})
    default_brightness = domain_data.get("default_brightness", DEFAULT_BRIGHTNESS_PCT)
    default_transition = domain_data.get("default_transition", DEFAULT_TRANSITION)
    min_step_delay_ms = domain_data.get("min_step_delay_ms", DEFAULT_MIN_STEP_DELAY_MS)

    # Validate color parameters
    _validate_color_params(call.data)
    _validate_color_ranges(call.data)

    brightness_pct = int(call.data.get(ATTR_BRIGHTNESS_PCT, default_brightness))
    transition_ms = int(1000 * float(call.data.get(ATTR_TRANSITION, default_transition)))

    expanded_entities = _expand_entity_ids(hass, call.data.get(ATTR_ENTITY_ID))
    if not expanded_entities:
        return

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
        _LOGGER.warning("%s: Entity not found", entity_id)
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

    current_level = start_level

    # Calculate fade parameters
    num_steps, delta, delay_ms = _calculate_fade_steps(
        start_level, end_level, transition_ms, min_step_delay_ms
    )

    _LOGGER.info(
        "%s: Fading from %s to %s in %s steps", entity_id, start_level, end_level, num_steps
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

    # Store final brightness after successful fade completion
    if not cancel_event.is_set():
        if end_level > 0:
            _store_orig_brightness(hass, entity_id, end_level)
            await _save_storage(hass)
            _LOGGER.info("%s: Fade complete at brightness %s", entity_id, end_level)
        else:
            _LOGGER.info("%s: Fade complete (turned off)", entity_id)


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


def _expand_entity_ids(hass: HomeAssistant, entity_ids_raw: str | list[str] | None) -> list[str]:
    """Expand entity IDs, handling groups and various input formats.

    Accepts raw input from service call and handles:
    - None: returns empty list
    - Comma-separated string: splits into list
    - List of strings: uses as-is

    Light groups are expanded iteratively (not recursively) to get
    individual light entities. Results are deduplicated.

    Example:
        Input: "light.living_room_group, light.bedroom"
        If light.living_room_group contains [light.lamp, light.ceiling]
        Output: ["light.lamp", "light.ceiling", "light.bedroom"]
    """
    if entity_ids_raw is None:
        return []

    if isinstance(entity_ids_raw, str):
        pending = [e.strip() for e in entity_ids_raw.split(",")]
    else:
        pending = list(entity_ids_raw)

    result: set[str] = set()

    while pending:
        entity_id = pending.pop()

        if not entity_id.startswith("light."):
            raise ServiceValidationError(f"Entity '{entity_id}' is not a light")

        state = hass.states.get(entity_id)
        if state is None:
            _LOGGER.error("%s: Unknown light", entity_id)
            continue

        # Check if this is a group (has entity_id attribute with member lights)
        if ATTR_ENTITY_ID in state.attributes:
            group_entities = state.attributes[ATTR_ENTITY_ID]
            if isinstance(group_entities, str):
                group_entities = [group_entities]
            pending.extend(group_entities)
        else:
            result.add(entity_id)

    return list(result)


# =============================================================================
# State Change Handler
# =============================================================================


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

    _log_state_change(entity_id, new_state)

    # Check if this is an expected state change (from our service calls)
    if _match_and_remove_expected(entity_id, new_state):
        _LOGGER.debug("%s: State matches expected, removed from tracking", entity_id)
        return

    # During fade: if we get here, state didn't match expected - manual intervention
    if entity_id in ACTIVE_FADES and entity_id in FADE_EXPECTED_BRIGHTNESS:
        # Manual intervention detected
        _LOGGER.info(
            "%s: Manual intervention detected (state=%s, brightness=%s)",
            entity_id,
            new_state.state,
            new_state.attributes.get(ATTR_BRIGHTNESS),
        )
        hass.async_create_task(_restore_intended_state(hass, entity_id, old_state, new_state))
        return

    # Normal state handling (no active fade)
    if _is_off_to_on_transition(old_state, new_state):
        _handle_off_to_on(hass, entity_id, new_state)
        return

    if _is_brightness_change(old_state, new_state):
        new_brightness = new_state.attributes.get(ATTR_BRIGHTNESS)
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


def _log_state_change(entity_id: str, new_state: State) -> None:
    """Log state change details."""
    _LOGGER.debug(
        "%s: State change - state=%s, brightness=%s, is_fading=%s",
        entity_id,
        new_state.state,
        new_state.attributes.get(ATTR_BRIGHTNESS),
        entity_id in ACTIVE_FADES,
    )


def _match_and_remove_expected(entity_id: str, new_state: State) -> bool:
    """Check if state matches expected, remove if found, notify if empty.

    Returns True if this was an expected state change (caller should ignore it).
    """
    expected_state = FADE_EXPECTED_BRIGHTNESS.get(entity_id)
    if not expected_state or expected_state.is_empty:
        return False

    # Normalize state to brightness: OFF -> 0, ON -> actual brightness
    if new_state.state == STATE_OFF:
        brightness = 0
    else:
        brightness = new_state.attributes.get(ATTR_BRIGHTNESS)
        if brightness is None:
            return False

    matched = expected_state.match_and_remove(brightness)

    if matched is not None:
        _LOGGER.debug(
            "%s: Matched expected brightness %s, remaining: %s",
            entity_id,
            matched,
            list(expected_state.values.keys()),
        )
        return True

    return False


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
        return
    _LOGGER.debug("%s: In _restore_intended_state", entity_id)
    await _cancel_and_wait_for_fade(entity_id)

    intended = _get_intended_brightness(hass, entity_id, old_state, new_state)
    _LOGGER.debug("%s: Got intended brightness (%s)", entity_id, intended)
    if intended is None:
        return

    # Store as new original brightness (for future OFF->ON restore)
    if intended > 0:
        _LOGGER.debug("%s: Storing original brightness (%s)", entity_id, intended)
        _store_orig_brightness(hass, entity_id, intended)

    # Get current state after fade cleanup
    current_state = hass.states.get(entity_id)
    if not current_state:
        _LOGGER.debug("%s: No current state found, exiting", entity_id)
        return

    current = current_state.attributes.get(ATTR_BRIGHTNESS) or 0
    if current_state.state == STATE_OFF:
        current = 0
        _LOGGER.debug("%s: Got current brightness (%s)", entity_id, current)

    # Restore if current differs from intended
    if intended == 0 and current != 0:
        _LOGGER.info("%s: Restoring to off as intended", entity_id)
        _add_expected_brightness(entity_id, 0)
        await hass.services.async_call(
            LIGHT_DOMAIN,
            SERVICE_TURN_OFF,
            {ATTR_ENTITY_ID: entity_id},
            blocking=True,
        )
        await _wait_until_stale_events_flushed(entity_id)
    elif intended > 0 and current != intended:
        _LOGGER.info("%s: Restoring to brightness %s as intended", entity_id, intended)
        _add_expected_brightness(entity_id, intended)
        await hass.services.async_call(
            LIGHT_DOMAIN,
            SERVICE_TURN_ON,
            {ATTR_ENTITY_ID: entity_id, ATTR_BRIGHTNESS: intended},
            blocking=True,
        )
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


def _add_expected_brightness(entity_id: str, brightness: int) -> None:
    """Register an expected brightness value before making a service call."""
    if entity_id not in FADE_EXPECTED_BRIGHTNESS:
        FADE_EXPECTED_BRIGHTNESS[entity_id] = ExpectedState(entity_id=entity_id)
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
            "%s: Timed out waiting for state events to flush (remaining: %s)",
            entity_id,
            list(expected_state.values.keys()),
        )


# =============================================================================
# Storage Helpers
# =============================================================================


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
