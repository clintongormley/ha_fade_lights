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
    COLOR_PARAMS,
    DEFAULT_MIN_STEP_DELAY_MS,
    DEFAULT_TRANSITION,
    DOMAIN,
    FADE_CANCEL_TIMEOUT_S,
    MIN_BRIGHTNESS_DELTA,
    MIN_HUE_DELTA,
    MIN_MIREDS_DELTA,
    MIN_SATURATION_DELTA,
    OPTION_MIN_STEP_DELAY_MS,
    PLANCKIAN_LOCUS_HS,
    PLANCKIAN_LOCUS_SATURATION_THRESHOLD,
    SERVICE_FADE_LIGHTS,
    STORAGE_KEY,
)
from .models import ExpectedState, ExpectedValues, FadeChange, FadeParams, FadeStep

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


def _validate_single_color_param(data: dict, context: str = "") -> None:
    """Validate that at most one color parameter is specified in data.

    Args:
        data: Dictionary to check for color parameters.
        context: Optional context string for error message (e.g., "in 'from:'").

    Raises:
        ServiceValidationError: If multiple color parameters are provided.
    """
    specified = [param for param in COLOR_PARAMS if param in data]
    if len(specified) > 1:
        ctx = f" {context}" if context else ""
        raise ServiceValidationError(
            f"Only one color parameter allowed{ctx}, got: {', '.join(sorted(specified))}"
        )


def _validate_color_params(data: dict) -> None:
    """Validate that at most one color parameter is specified.

    Also validates the from: parameter if present.

    Raises:
        ServiceValidationError: If multiple color parameters are provided.
    """
    _validate_single_color_param(data)

    from_data = data.get(ATTR_FROM, {})
    if from_data:
        _validate_single_color_param(from_data, "in 'from:'")


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
    """Validate color and brightness ranges in a single dict."""
    if ATTR_BRIGHTNESS_PCT in data:
        brightness = data[ATTR_BRIGHTNESS_PCT]
        if not (0 <= brightness <= 100):
            raise ServiceValidationError(
                f"{prefix}Brightness must be between 0 and 100, got {brightness}"
            )

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


def _extract_fade_values(
    data: dict,
) -> tuple[int | None, tuple[float, float] | None, int | None]:
    """Extract brightness, HS color, and mireds from data dict.

    Returns:
        Tuple of (brightness_pct, hs_color, color_temp_mireds)
    """
    brightness_pct = int(data[ATTR_BRIGHTNESS_PCT]) if ATTR_BRIGHTNESS_PCT in data else None
    hs, mireds = _extract_color(data)
    return brightness_pct, hs, mireds


def _validate_and_parse_color_params(data: dict) -> FadeParams:
    """Validate and parse color parameters to internal representation.

    Validates:
    - At most one color parameter is specified
    - Color parameter values are within valid ranges

    Converts:
    - rgb_color, rgbw_color, rgbww_color, xy_color -> hs_color
    - color_temp_kelvin -> color_temp_mireds

    Also handles the 'from:' parameter for starting values.

    Args:
        data: Service call data dictionary

    Returns:
        FadeParams with normalized color values

    Raises:
        ServiceValidationError: If validation fails
    """
    _validate_color_params(data)
    _validate_color_ranges(data)

    params = FadeParams()

    params.brightness_pct, params.hs_color, params.color_temp_mireds = _extract_fade_values(data)

    from_data = data.get(ATTR_FROM, {})
    if from_data:
        (
            params.from_brightness_pct,
            params.from_hs_color,
            params.from_color_temp_mireds,
        ) = _extract_fade_values(from_data)

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
    FADE_EXPECTED_STATE.clear()
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
    min_step_delay_ms = domain_data.get("min_step_delay_ms", DEFAULT_MIN_STEP_DELAY_MS)

    fade_params = _validate_and_parse_color_params(call.data)

    if not fade_params.has_target() and not fade_params.has_from_target():
        _LOGGER.debug("No fade parameters specified, nothing to do")
        return

    transition_ms = int(1000 * float(call.data.get(ATTR_TRANSITION, DEFAULT_TRANSITION)))

    expanded_entities = _expand_entity_ids(hass, call.data.get(ATTR_ENTITY_ID))
    if not expanded_entities:
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
                    transition_ms,
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
    transition_ms: int,
    min_step_delay_ms: int,
) -> None:
    """Fade a single light to the specified brightness and/or color.

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
            hass, entity_id, fade_params, transition_ms, min_step_delay_ms, cancel_event
        )
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
    transition_ms: int,
    min_step_delay_ms: int,
    cancel_event: asyncio.Event,
) -> None:
    """Execute the fade operation using FadeChange iterator pattern.

    Uses _calculate_changes to generate FadeChange phases, then iterates
    through each phase using the iterator pattern (has_next/next_step).
    """
    state = hass.states.get(entity_id)
    if not state:
        _LOGGER.warning("%s: Entity not found", entity_id)
        return

    # Handle non-dimmable lights (on/off only)
    supported_modes = state.attributes.get(ATTR_SUPPORTED_COLOR_MODES, [])
    has_any_color = any(
        mode in supported_modes
        for mode in [
            ColorMode.HS,
            ColorMode.RGB,
            ColorMode.RGBW,
            ColorMode.RGBWW,
            ColorMode.XY,
            ColorMode.COLOR_TEMP,
        ]
    )
    if ColorMode.BRIGHTNESS not in supported_modes and not has_any_color:
        # On/off only light
        target_brightness = fade_params.brightness_pct or 0
        await _apply_step(hass, entity_id, FadeStep(brightness=255 if target_brightness > 0 else 0))
        return

    # Store original brightness if not already stored
    current_brightness = state.attributes.get(ATTR_BRIGHTNESS)
    start_brightness = int(current_brightness) if current_brightness is not None else 0
    existing_orig = _get_orig_brightness(hass, entity_id)
    if existing_orig == 0 and start_brightness > 0:
        _store_orig_brightness(hass, entity_id, start_brightness)

    # Get current color state for "nothing to fade" detection
    current_hs, current_mireds = _get_current_color_state(state)

    # Determine start and end values for change detection
    end_brightness = int(fade_params.brightness_pct / 100 * 255) if fade_params.brightness_pct is not None else None
    effective_start_brightness = (
        int(fade_params.from_brightness_pct / 100 * 255)
        if fade_params.from_brightness_pct is not None
        else start_brightness
    )

    # Check HS color
    start_hs = fade_params.from_hs_color if fade_params.from_hs_color is not None else current_hs
    end_hs = fade_params.hs_color

    # Check mireds
    start_mireds = (
        fade_params.from_color_temp_mireds
        if fade_params.from_color_temp_mireds is not None
        else current_mireds
    )
    end_mireds = fade_params.color_temp_mireds

    # Check if anything is actually changing
    brightness_changing = fade_params.brightness_pct is not None and effective_start_brightness != end_brightness
    hs_changing = end_hs is not None and start_hs != end_hs
    mireds_changing = end_mireds is not None and start_mireds != end_mireds

    if not brightness_changing and not hs_changing and not mireds_changing:
        _LOGGER.debug("%s: Nothing to fade", entity_id)
        return

    # Set transition_ms in params for _calculate_changes
    fade_params.transition_ms = transition_ms

    # Calculate changes (returns list of FadeChange phases)
    phases = _calculate_changes(fade_params, state.attributes, min_step_delay_ms)

    # Sum total steps
    total_steps = sum(p.step_count() for p in phases)

    _LOGGER.info("%s: Fading in %s phases (%s total steps)", entity_id, len(phases), total_steps)

    # Execute each phase
    for phase_idx, phase in enumerate(phases):
        phase.reset()
        delay_ms = phase.delay_ms()

        while phase.has_next():
            step_start = time.monotonic()

            if cancel_event.is_set():
                return

            step = phase.next_step()

            # Track expected values for manual intervention detection
            expected = ExpectedValues(
                brightness=step.brightness,
                hs_color=step.hs_color,
                color_temp_mireds=step.color_temp_mireds,
            )
            _add_expected_values(entity_id, expected)

            await _apply_step(hass, entity_id, step)

            if cancel_event.is_set():
                return

            # Sleep remaining time (skip after last step of last phase)
            if phase.has_next() or phase_idx < len(phases) - 1:
                await _sleep_remaining_step_time(step_start, delay_ms)

    # Store final brightness after successful fade completion
    if not cancel_event.is_set():
        # Get final brightness from last phase's end_brightness (efficient)
        final_brightness = phases[-1].end_brightness
        if final_brightness is None:
            # Try to get from params
            if fade_params.brightness_pct is not None:
                final_brightness = int(fade_params.brightness_pct / 100 * 255)

        if final_brightness is not None and final_brightness > 0:
            _store_orig_brightness(hass, entity_id, final_brightness)
            await _save_storage(hass)
            _LOGGER.info("%s: Fade complete at brightness %s", entity_id, final_brightness)
        elif final_brightness == 0:
            _LOGGER.info("%s: Fade complete (turned off)", entity_id)
        else:
            _LOGGER.info("%s: Fade complete", entity_id)


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


def _calculate_step_count(
    brightness_change: int | None,
    hue_change: float | None,
    sat_change: float | None,
    mireds_change: int | None,
    transition_ms: int,
    min_step_delay_ms: int,
) -> int:
    """Calculate optimal step count based on change magnitude and time constraints.

    The algorithm:
    1. Calculate ideal steps per dimension: change / minimum_delta
    2. Take the maximum across all changing dimensions (smoothest dimension wins)
    3. Constrain by time: if ideal_steps * min_step_delay_ms > transition_ms, use time-limited steps

    Args:
        brightness_change: Absolute brightness change (0-255 scale), or None if not changing
        hue_change: Absolute hue change in degrees (0-360), or None if not changing
        sat_change: Absolute saturation change (0-100), or None if not changing
        mireds_change: Absolute mireds change, or None if not changing
        transition_ms: Total transition time in milliseconds
        min_step_delay_ms: Minimum delay between steps in milliseconds

    Returns:
        Optimal number of steps (at least 1)
    """
    ideal_steps = []

    if brightness_change is not None and brightness_change > 0:
        ideal_steps.append(brightness_change // MIN_BRIGHTNESS_DELTA)
    if hue_change is not None and hue_change > 0:
        ideal_steps.append(int(hue_change / MIN_HUE_DELTA))
    if sat_change is not None and sat_change > 0:
        ideal_steps.append(int(sat_change / MIN_SATURATION_DELTA))
    if mireds_change is not None and mireds_change > 0:
        ideal_steps.append(mireds_change // MIN_MIREDS_DELTA)

    ideal = max(ideal_steps) if ideal_steps else 1
    max_by_time = transition_ms // min_step_delay_ms

    return max(1, min(ideal, max_by_time))


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


async def _apply_step(hass: HomeAssistant, entity_id: str, step: FadeStep) -> None:
    """Apply a fade step to a light.

    Handles brightness, hs_color, and color_temp_mireds in a single service call.
    If brightness is 0, turns off the light. If step is empty, does nothing.
    Color temp is converted from mireds to kelvin for the service call.
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

    if step.color_temp_mireds is not None:
        # Convert mireds to kelvin for service call
        kelvin = int(1_000_000 / step.color_temp_mireds)
        service_data[HA_ATTR_COLOR_TEMP_KELVIN] = kelvin

    # Only call service if there's something to set (beyond entity_id)
    if len(service_data) > 1:
        await hass.services.async_call(
            LIGHT_DOMAIN,
            SERVICE_TURN_ON,
            service_data,
            blocking=True,
        )


def _get_current_color_state(state: State) -> tuple[tuple[float, float] | None, int | None]:
    """Extract current HS color and color temp from light state.

    Args:
        state: Home Assistant State object for a light entity

    Returns:
        Tuple of (hs_color, color_temp_mireds) - either or both may be None
    """
    hs_color = None
    color_temp_mireds = None

    # Get HS color (may be tuple or list)
    hs_raw = state.attributes.get(HA_ATTR_HS_COLOR)
    if hs_raw is not None:
        hs_color = (float(hs_raw[0]), float(hs_raw[1]))

    # Get color temp in mireds (stored as "color_temp" in state attributes)
    mireds_raw = state.attributes.get("color_temp")
    if mireds_raw is not None:
        color_temp_mireds = int(mireds_raw)

    return hs_color, color_temp_mireds


def _resolve_start_brightness(params: FadeParams, state: dict) -> int | None:
    """Resolve starting brightness from params.from_brightness_pct or current state.

    Args:
        params: FadeParams with optional from_brightness_pct
        state: Light attributes dict from state.attributes

    Returns:
        Starting brightness (0-255 scale), or None if not targeting brightness
    """
    if params.from_brightness_pct is not None:
        return int(params.from_brightness_pct * 255 / 100)
    # When light is off (no brightness in state), treat as 0
    brightness = state.get(ATTR_BRIGHTNESS)
    return int(brightness) if brightness is not None else 0


def _resolve_end_brightness(params: FadeParams, state: dict) -> int | None:
    """Resolve ending brightness from params.brightness_pct.

    Args:
        params: FadeParams with optional brightness_pct
        state: Light attributes dict (unused, kept for API consistency)

    Returns:
        Ending brightness (0-255 scale), or None if not specified
    """
    if params.brightness_pct is not None:
        return int(params.brightness_pct * 255 / 100)
    return None


def _resolve_start_hs(params: FadeParams, state: dict) -> tuple[float, float] | None:
    """Resolve starting HS from params.from_hs_color or current state.

    Args:
        params: FadeParams with optional from_hs_color
        state: Light attributes dict from state.attributes

    Returns:
        Starting HS color (hue 0-360, saturation 0-100), or None if not available
    """
    if params.from_hs_color is not None:
        return params.from_hs_color
    return state.get(HA_ATTR_HS_COLOR)


def _resolve_start_mireds(params: FadeParams, state: dict) -> int | None:
    """Resolve starting mireds from params.from_color_temp_mireds or current state.

    Args:
        params: FadeParams with optional from_color_temp_mireds
        state: Light attributes dict from state.attributes

    Returns:
        Starting color temperature in mireds, or None if not available
    """
    if params.from_color_temp_mireds is not None:
        return params.from_color_temp_mireds
    return state.get("color_temp")


async def _sleep_remaining_step_time(step_start: float, delay_ms: float) -> None:
    """Sleep for the remaining time in a fade step.

    Subtracts elapsed time from target delay to maintain consistent fade duration
    regardless of how long the service call took.
    """
    elapsed_ms = (time.monotonic() - step_start) * 1000
    sleep_ms = max(0, delay_ms - elapsed_ms)
    if sleep_ms > 0:
        await asyncio.sleep(sleep_ms / 1000)


def _interpolate_hue(start: float, end: float, t: float) -> float:
    """Interpolate hue using circular short-path.

    Hue is circular (0-360 wraps around). This function always takes the
    shortest path between two hues.

    Args:
        start: Starting hue (0-360)
        end: Ending hue (0-360)
        t: Interpolation factor (0.0 = start, 1.0 = end)

    Returns:
        Interpolated hue in range [0, 360)
    """
    diff = end - start

    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360

    result = start + diff * t
    return result % 360


def _is_on_planckian_locus(hs_color: tuple[float, float]) -> bool:
    """Check if an HS color is on or near the Planckian locus.

    The Planckian locus represents the colors of blackbody radiation
    (color temperatures). Colors on the locus have low saturation
    (white/off-white appearance).

    Args:
        hs_color: Tuple of (hue 0-360, saturation 0-100)

    Returns:
        True if the color is close enough to the locus to transition
        directly to mireds-based fading.
    """
    _, saturation = hs_color
    return saturation <= PLANCKIAN_LOCUS_SATURATION_THRESHOLD


def _hs_to_mireds(hs_color: tuple[float, float]) -> int:
    """Convert an HS color to approximate mireds using Planckian locus lookup.

    Finds the closest matching color temperature on the Planckian locus
    based on hue matching. Used when transitioning from HS to color temp.

    Args:
        hs_color: Tuple of (hue 0-360, saturation 0-100)

    Returns:
        Approximate color temperature in mireds
    """
    hue, saturation = hs_color

    # For very low saturation, return neutral white
    if saturation < 3:
        return 286  # ~3500K neutral white

    # Find closest match in the lookup table based on hue
    best_mireds = 286  # Default to neutral white
    best_distance = float("inf")

    for mireds, (locus_hue, _) in PLANCKIAN_LOCUS_HS:
        # Calculate hue distance (circular)
        distance = abs(hue - locus_hue)
        if distance > 180:
            distance = 360 - distance

        if distance < best_distance:
            best_distance = distance
            best_mireds = mireds

    return best_mireds


def _mireds_to_hs(mireds: int) -> tuple[float, float]:
    """Convert mireds to approximate HS using Planckian locus lookup.

    Interpolates between lookup table entries to find the HS color
    that corresponds to the given color temperature.

    Args:
        mireds: Color temperature in mireds

    Returns:
        Tuple of (hue 0-360, saturation 0-100)
    """
    # Handle values outside the lookup range
    if mireds <= PLANCKIAN_LOCUS_HS[0][0]:
        return PLANCKIAN_LOCUS_HS[0][1]
    if mireds >= PLANCKIAN_LOCUS_HS[-1][0]:
        return PLANCKIAN_LOCUS_HS[-1][1]

    # Find the two bracketing entries
    for i in range(len(PLANCKIAN_LOCUS_HS) - 1):
        lower_mireds, lower_hs = PLANCKIAN_LOCUS_HS[i]
        upper_mireds, upper_hs = PLANCKIAN_LOCUS_HS[i + 1]

        if lower_mireds <= mireds <= upper_mireds:
            # Interpolate between the two entries
            t = (mireds - lower_mireds) / (upper_mireds - lower_mireds)
            hue = lower_hs[0] + (upper_hs[0] - lower_hs[0]) * t
            sat = lower_hs[1] + (upper_hs[1] - lower_hs[1]) * t
            return (round(hue, 2), round(sat, 2))

    # Fallback (should not reach here)
    return (38.0, 12.0)  # Neutral white


def _calculate_hs_to_mireds_changes(
    start_brightness: int | None,
    end_brightness: int | None,
    start_hs: tuple[float, float],
    end_mireds: int,
    transition_ms: int,
    min_step_delay_ms: int,
) -> list[FadeChange]:
    """Calculate HS -> mireds hybrid transition (two phases).

    Phase 1 (70%): Fade HS toward Planckian locus
    Phase 2 (30%): Fade mireds to target
    """
    # If already on locus, skip HS phase and just do mireds
    if _is_on_planckian_locus(start_hs):
        start_mireds = _hs_to_mireds(start_hs)
        return [FadeChange(
            start_brightness=start_brightness,
            end_brightness=end_brightness,
            start_mireds=start_mireds,
            end_mireds=end_mireds,
            transition_ms=transition_ms,
            min_step_delay_ms=min_step_delay_ms,
        )]

    # Find intersection point on Planckian locus
    intersection_hs = _mireds_to_hs(end_mireds)
    intersection_mireds = _hs_to_mireds(intersection_hs)

    # Split timing 70/30
    phase1_ms = int(transition_ms * 0.7)
    phase2_ms = transition_ms - phase1_ms

    # Split brightness proportionally
    mid_brightness = None
    if start_brightness is not None and end_brightness is not None:
        brightness_change = end_brightness - start_brightness
        mid_brightness = start_brightness + int(brightness_change * 0.7)

    return [
        FadeChange(
            start_brightness=start_brightness,
            end_brightness=mid_brightness,
            start_hs=start_hs,
            end_hs=intersection_hs,
            transition_ms=phase1_ms,
            min_step_delay_ms=min_step_delay_ms,
        ),
        FadeChange(
            start_brightness=mid_brightness,
            end_brightness=end_brightness,
            start_mireds=intersection_mireds,
            end_mireds=end_mireds,
            transition_ms=phase2_ms,
            min_step_delay_ms=min_step_delay_ms,
        ),
    ]


def _calculate_mireds_to_hs_changes(
    start_brightness: int | None,
    end_brightness: int | None,
    start_mireds: int,
    end_hs: tuple[float, float],
    transition_ms: int,
    min_step_delay_ms: int,
) -> list[FadeChange]:
    """Calculate mireds -> HS hybrid transition (two phases).

    Phase 1 (30%): Fade mireds to Planckian intersection
    Phase 2 (70%): Fade HS to target
    """
    # Find the locus point closest to target HS
    target_locus_mireds = _hs_to_mireds(end_hs)

    # If already close to target mireds, skip mireds phase
    if abs(start_mireds - target_locus_mireds) < 10:
        start_hs_from_mireds = _mireds_to_hs(start_mireds)
        return [FadeChange(
            start_brightness=start_brightness,
            end_brightness=end_brightness,
            start_hs=start_hs_from_mireds,
            end_hs=end_hs,
            transition_ms=transition_ms,
            min_step_delay_ms=min_step_delay_ms,
        )]

    # Split timing 30/70
    phase1_ms = int(transition_ms * 0.3)
    phase2_ms = transition_ms - phase1_ms

    # Split brightness proportionally
    mid_brightness = None
    if start_brightness is not None and end_brightness is not None:
        brightness_change = end_brightness - start_brightness
        mid_brightness = start_brightness + int(brightness_change * 0.3)

    # Get HS at the target mireds point on locus
    locus_hs = _mireds_to_hs(target_locus_mireds)

    return [
        FadeChange(
            start_brightness=start_brightness,
            end_brightness=mid_brightness,
            start_mireds=start_mireds,
            end_mireds=target_locus_mireds,
            transition_ms=phase1_ms,
            min_step_delay_ms=min_step_delay_ms,
        ),
        FadeChange(
            start_brightness=mid_brightness,
            end_brightness=end_brightness,
            start_hs=locus_hs,
            end_hs=end_hs,
            transition_ms=phase2_ms,
            min_step_delay_ms=min_step_delay_ms,
        ),
    ]


def _calculate_changes(
    params: FadeParams,
    current_state: dict,
    min_step_delay_ms: int,
) -> list[FadeChange]:
    """Calculate fade changes, dispatching to hybrid calculators if needed.

    Returns a list of FadeChange phases:
    - Simple fades: single FadeChange
    - Hybrid transitions (HS<->mireds): two FadeChange phases
    """
    # Resolve start values from state or params.from_*
    start_brightness = _resolve_start_brightness(params, current_state)
    start_hs = _resolve_start_hs(params, current_state)
    start_mireds = _resolve_start_mireds(params, current_state)

    # Resolve end values from params
    end_brightness = _resolve_end_brightness(params, current_state)
    end_hs = params.hs_color
    end_mireds = params.color_temp_mireds

    # Detect hybrid transitions
    # HS -> mireds: starting with HS color (not on Planckian locus) and targeting mireds
    if (
        start_hs is not None
        and end_mireds is not None
        and end_hs is None
        and not _is_on_planckian_locus(start_hs)
    ):
        return _calculate_hs_to_mireds_changes(
            start_brightness,
            end_brightness,
            start_hs,
            end_mireds,
            params.transition_ms,
            min_step_delay_ms,
        )

    # mireds -> HS: starting with color temp and targeting HS color
    if start_mireds is not None and end_hs is not None and end_mireds is None:
        return _calculate_mireds_to_hs_changes(
            start_brightness,
            end_brightness,
            start_mireds,
            end_hs,
            params.transition_ms,
            min_step_delay_ms,
        )

    # Simple fade (single phase)
    return [FadeChange(
        start_brightness=start_brightness,
        end_brightness=end_brightness,
        start_hs=start_hs if end_hs is not None else None,
        end_hs=end_hs,
        start_mireds=start_mireds if end_mireds is not None else None,
        end_mireds=end_mireds,
        transition_ms=params.transition_ms,
        min_step_delay_ms=min_step_delay_ms,
    )]


def _build_hs_to_mireds_steps(
    start_hs: tuple[float, float],
    end_mireds: int,
    transition_ms: int,
    min_step_delay_ms: int,
) -> list[FadeStep]:
    """Build hybrid step sequence from HS color to mireds.

    If the starting HS is already on the Planckian locus, generates
    pure mireds-based steps. Otherwise, first fades the HS toward
    the locus (reducing saturation), then switches to mireds.

    Args:
        start_hs: Starting (hue, saturation)
        end_mireds: Target color temperature in mireds
        transition_ms: Total transition time in milliseconds
        min_step_delay_ms: Minimum delay between steps

    Returns:
        List of FadeStep objects transitioning from HS to mireds
    """
    # If already on locus, just do mireds-based fading
    if _is_on_planckian_locus(start_hs):
        start_mireds = _hs_to_mireds(start_hs)
        return _build_fade_steps(
            start_brightness=None,
            end_brightness=None,
            start_hs=None,
            end_hs=None,
            start_mireds=start_mireds,
            end_mireds=end_mireds,
            transition_ms=transition_ms,
            min_step_delay_ms=min_step_delay_ms,
        )

    # Off locus: need to transition HS toward locus first, then to mireds
    # Get the target HS on the locus (what the end_mireds looks like in HS)
    target_locus_hs = _mireds_to_hs(end_mireds)

    # Calculate changes for HS phase
    hue_diff = abs(start_hs[0] - target_locus_hs[0])
    if hue_diff > 180:
        hue_diff = 360 - hue_diff
    sat_diff = abs(start_hs[1] - target_locus_hs[1])

    # Calculate changes for mireds phase
    locus_mireds = _hs_to_mireds(target_locus_hs)
    mireds_diff = abs(end_mireds - locus_mireds)

    # Calculate total step count based on all changes
    total_steps = _calculate_step_count(
        brightness_change=None,
        hue_change=hue_diff,
        sat_change=sat_diff,
        mireds_change=mireds_diff,
        transition_ms=transition_ms,
        min_step_delay_ms=min_step_delay_ms,
    )

    # Use 70% of steps for HS transition, 30% for final mireds adjustment
    hs_steps_count = max(1, int(total_steps * 0.7))
    mireds_steps_count = max(1, total_steps - hs_steps_count)

    steps = []

    # Phase 1: HS toward the locus target
    for i in range(1, hs_steps_count + 1):
        t = i / hs_steps_count
        hue = _interpolate_hue(start_hs[0], target_locus_hs[0], t)
        sat = start_hs[1] + (target_locus_hs[1] - start_hs[1]) * t
        steps.append(FadeStep(hs_color=(round(hue, 2), round(sat, 2))))

    # Phase 2: Mireds from locus point to target
    for i in range(1, mireds_steps_count + 1):
        t = i / mireds_steps_count
        mireds = round(locus_mireds + (end_mireds - locus_mireds) * t)
        steps.append(FadeStep(color_temp_mireds=mireds))

    return steps


def _build_mireds_to_hs_steps(
    start_mireds: int,
    end_hs: tuple[float, float],
    transition_ms: int,
    min_step_delay_ms: int,
) -> list[FadeStep]:
    """Build hybrid step sequence from mireds to HS color.

    When a light is in color temp mode and needs to transition to an HS color,
    this function creates a smooth transition:
    1. First, fade mireds along the Planckian locus toward the target hue (~30%)
    2. Then, switch to HS and fade to the final target (~70%)

    This is the symmetric counterpart to _build_hs_to_mireds_steps.

    Args:
        start_mireds: Starting color temperature in mireds
        end_hs: Target (hue, saturation)
        transition_ms: Total transition time in milliseconds
        min_step_delay_ms: Minimum delay between steps

    Returns:
        List of FadeStep objects transitioning from mireds to HS
    """
    # Find the locus point closest to the target hue
    target_locus_mireds = _hs_to_mireds(end_hs)

    # If already at or very close to target mireds, skip mireds phase
    if abs(start_mireds - target_locus_mireds) < 10:
        # Just do HS fade from locus point
        start_hs = _mireds_to_hs(start_mireds)
        return _build_fade_steps(
            start_brightness=None,
            end_brightness=None,
            start_hs=start_hs,
            end_hs=end_hs,
            start_mireds=None,
            end_mireds=None,
            transition_ms=transition_ms,
            min_step_delay_ms=min_step_delay_ms,
        )

    # Calculate changes for mireds phase
    mireds_diff = abs(target_locus_mireds - start_mireds)

    # Calculate changes for HS phase
    locus_hs = _mireds_to_hs(target_locus_mireds)
    hue_diff = abs(locus_hs[0] - end_hs[0])
    if hue_diff > 180:
        hue_diff = 360 - hue_diff
    sat_diff = abs(locus_hs[1] - end_hs[1])

    # Calculate total step count based on all changes
    total_steps = _calculate_step_count(
        brightness_change=None,
        hue_change=hue_diff,
        sat_change=sat_diff,
        mireds_change=mireds_diff,
        transition_ms=transition_ms,
        min_step_delay_ms=min_step_delay_ms,
    )

    # Split: ~30% mireds, ~70% HS (opposite of HS->mireds which is 70/30)
    mireds_steps_count = max(1, int(total_steps * 0.3))
    hs_steps_count = max(1, total_steps - mireds_steps_count)

    steps = []

    # Phase 1: Mireds along locus toward target hue
    for i in range(1, mireds_steps_count + 1):
        t = i / mireds_steps_count
        mireds = round(start_mireds + (target_locus_mireds - start_mireds) * t)
        steps.append(FadeStep(color_temp_mireds=mireds))

    # Phase 2: HS from locus point to final target
    for i in range(1, hs_steps_count + 1):
        t = i / hs_steps_count
        hue = _interpolate_hue(locus_hs[0], end_hs[0], t)
        sat = locus_hs[1] + (end_hs[1] - locus_hs[1]) * t
        steps.append(FadeStep(hs_color=(round(hue, 2), round(sat, 2))))

    return steps


def _build_fade_steps(
    start_brightness: int | None,
    end_brightness: int | None,
    start_hs: tuple[float, float] | None,
    end_hs: tuple[float, float] | None,
    start_mireds: int | None,
    end_mireds: int | None,
    transition_ms: int,
    min_step_delay_ms: int,
) -> list[FadeStep]:
    """Build array of interpolated fade steps.

    Calculates optimal step count based on the largest change dimension,
    constrained by transition time and minimum step delay.
    """
    # Calculate changes for each dimension
    brightness_change: int | None = None
    hue_change: float | None = None
    sat_change: float | None = None
    mireds_change: int | None = None

    if start_brightness is not None and end_brightness is not None:
        brightness_change = abs(end_brightness - start_brightness)

    if start_hs is not None and end_hs is not None:
        hue_diff = abs(end_hs[0] - start_hs[0])
        if hue_diff > 180:
            hue_diff = 360 - hue_diff
        hue_change = hue_diff
        sat_change = abs(end_hs[1] - start_hs[1])

    if start_mireds is not None and end_mireds is not None:
        mireds_change = abs(end_mireds - start_mireds)

    num_steps = _calculate_step_count(
        brightness_change=brightness_change,
        hue_change=hue_change,
        sat_change=sat_change,
        mireds_change=mireds_change,
        transition_ms=transition_ms,
        min_step_delay_ms=min_step_delay_ms,
    )

    steps = []
    for i in range(1, num_steps + 1):
        t = i / num_steps
        step = FadeStep()

        if start_brightness is not None and end_brightness is not None:
            brightness = round(start_brightness + (end_brightness - start_brightness) * t)
            # Skip brightness level 1 (many lights behave oddly at this level)
            if brightness == 1:
                brightness = 0 if end_brightness < start_brightness else 2
            step.brightness = brightness

        if start_hs is not None and end_hs is not None:
            hue = _interpolate_hue(start_hs[0], end_hs[0], t)
            sat = start_hs[1] + (end_hs[1] - start_hs[1]) * t
            step.hs_color = (round(hue, 2), round(sat, 2))

        if start_mireds is not None and end_mireds is not None:
            step.color_temp_mireds = round(start_mireds + (end_mireds - start_mireds) * t)

        steps.append(step)

    return steps


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
        params.color_temp_mireds is not None or params.from_color_temp_mireds is not None
    )
    return color_temp_requested and ColorMode.COLOR_TEMP in modes


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
    if entity_id in ACTIVE_FADES and entity_id in FADE_EXPECTED_STATE:
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

        mireds_raw = new_state.attributes.get("color_temp")
        color_temp_mireds = int(mireds_raw) if mireds_raw else None

        actual = ExpectedValues(
            brightness=brightness,
            hs_color=hs_color,
            color_temp_mireds=color_temp_mireds,
        )

    matched = expected_state.match_and_remove(actual)

    if matched is not None:
        _LOGGER.debug(
            "%s: Matched expected %s, remaining: %d",
            entity_id,
            matched,
            len(expected_state.values),
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
    2. Compares current state to intended state (brightness + color)
    3. Restores intended state if they differ

    The intended brightness encodes both state and brightness:
    - 0 means OFF
    - >0 means ON at that brightness

    Colors from the manual intervention are also restored.
    Note: Only brightness is persisted to storage, not colors.
    """
    if DOMAIN not in hass.data:
        return
    _LOGGER.debug("%s: In _restore_intended_state", entity_id)
    await _cancel_and_wait_for_fade(entity_id)

    intended_brightness = _get_intended_brightness(hass, entity_id, old_state, new_state)
    _LOGGER.debug("%s: Got intended brightness (%s)", entity_id, intended_brightness)
    if intended_brightness is None:
        return

    # Store as new original brightness (for future OFF->ON restore)
    # Note: We only store brightness, not colors
    if intended_brightness > 0:
        _LOGGER.debug("%s: Storing original brightness (%s)", entity_id, intended_brightness)
        _store_orig_brightness(hass, entity_id, intended_brightness)

    # Get current state after fade cleanup
    current_state = hass.states.get(entity_id)
    if not current_state:
        _LOGGER.debug("%s: No current state found, exiting", entity_id)
        return

    current_brightness = current_state.attributes.get(ATTR_BRIGHTNESS) or 0
    if current_state.state == STATE_OFF:
        current_brightness = 0

    # Handle OFF case
    if intended_brightness == 0 and current_brightness != 0:
        _LOGGER.info("%s: Restoring to off as intended", entity_id)
        _add_expected_brightness(entity_id, 0)
        await hass.services.async_call(
            LIGHT_DOMAIN,
            SERVICE_TURN_OFF,
            {ATTR_ENTITY_ID: entity_id},
            blocking=True,
        )
        await _wait_until_stale_events_flushed(entity_id)
        return

    # Handle ON case - check brightness and colors
    if intended_brightness > 0:
        # Build service data for restoration
        service_data: dict = {ATTR_ENTITY_ID: entity_id}
        need_restore = False

        # Check brightness
        if current_brightness != intended_brightness:
            service_data[ATTR_BRIGHTNESS] = intended_brightness
            need_restore = True

        # Get intended colors from manual intervention (new_state)
        intended_hs = new_state.attributes.get(HA_ATTR_HS_COLOR)
        intended_mireds = new_state.attributes.get("color_temp")

        # Get current colors
        current_hs = current_state.attributes.get(HA_ATTR_HS_COLOR)
        current_mireds = current_state.attributes.get("color_temp")

        # Check HS color
        if intended_hs and intended_hs != current_hs:
            service_data[HA_ATTR_HS_COLOR] = intended_hs
            need_restore = True

        # Check color temp (mutually exclusive with HS)
        if (
            intended_mireds
            and intended_mireds != current_mireds
            and HA_ATTR_HS_COLOR not in service_data
        ):
            # Convert mireds to kelvin
            kelvin = int(1_000_000 / intended_mireds)
            service_data[HA_ATTR_COLOR_TEMP_KELVIN] = kelvin
            need_restore = True

        if need_restore:
            _LOGGER.info("%s: Restoring intended state: %s", entity_id, service_data)

            # Track expected values
            _add_expected_values(
                entity_id,
                ExpectedValues(
                    brightness=service_data.get(ATTR_BRIGHTNESS),
                    hs_color=service_data.get(HA_ATTR_HS_COLOR),
                    color_temp_mireds=(
                        int(1_000_000 / service_data[HA_ATTR_COLOR_TEMP_KELVIN])
                        if HA_ATTR_COLOR_TEMP_KELVIN in service_data
                        else None
                    ),
                ),
            )

            await hass.services.async_call(
                LIGHT_DOMAIN,
                SERVICE_TURN_ON,
                service_data,
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
