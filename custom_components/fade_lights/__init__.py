"""The Fade Lights integration."""

from __future__ import annotations

import asyncio
import logging
import time

from homeassistant.components.light import (
    ATTR_BRIGHTNESS,
    ATTR_COLOR_TEMP_KELVIN as HA_ATTR_COLOR_TEMP_KELVIN,
    ATTR_HS_COLOR as HA_ATTR_HS_COLOR,
    ATTR_MAX_COLOR_TEMP_KELVIN as HA_ATTR_MAX_COLOR_TEMP_KELVIN,
    ATTR_MIN_COLOR_TEMP_KELVIN as HA_ATTR_MIN_COLOR_TEMP_KELVIN,
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
    DEFAULT_MIN_STEP_DELAY_MS,
    DOMAIN,
    FADE_CANCEL_TIMEOUT_S,
    HYBRID_HS_PHASE_RATIO,
    OPTION_MIN_STEP_DELAY_MS,
    PLANCKIAN_LOCUS_HS,
    PLANCKIAN_LOCUS_SATURATION_THRESHOLD,
    SERVICE_FADE_LIGHTS,
    STORAGE_KEY,
)
from .expected_state import ExpectedState, ExpectedValues
from .fade_change import FadeChange, FadeStep
from .fade_params import FadeParams

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

    fade_params = FadeParams.from_service_data(call.data)

    if not fade_params.has_target() and not fade_params.has_from_target():
        _LOGGER.debug("No fade parameters specified, nothing to do")
        return

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
        await _execute_fade(hass, entity_id, fade_params, min_step_delay_ms, cancel_event)
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

    Uses resolve_fade to create a single FadeChange that handles all fade types
    including hybrid transitions internally. The iterator generates steps
    seamlessly across mode switches.
    """
    state = hass.states.get(entity_id)
    if not state:
        _LOGGER.warning("%s: Entity not found", entity_id)
        return

    # Store original brightness if not already stored
    current_brightness = state.attributes.get(ATTR_BRIGHTNESS)
    start_brightness = int(current_brightness) if current_brightness is not None else 0
    existing_orig = _get_orig_brightness(hass, entity_id)
    if existing_orig == 0 and start_brightness > 0:
        _store_orig_brightness(hass, entity_id, start_brightness)

    # Get stored brightness for auto-turn-on when fading color from off
    stored_brightness = existing_orig if existing_orig > 0 else start_brightness

    # Resolve fade parameters into a configured FadeChange
    fade = resolve_fade(fade_params, state.attributes, min_step_delay_ms, stored_brightness)

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


async def _apply_step(hass: HomeAssistant, entity_id: str, step: FadeStep) -> None:
    """Apply a fade step to a light.

    Handles brightness, hs_color, and color_temp_kelvin in a single service call.
    If brightness is 0, turns off the light. If step is empty, does nothing.
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

    # Only call service if there's something to set (beyond entity_id)
    if len(service_data) > 1:
        await hass.services.async_call(
            LIGHT_DOMAIN,
            SERVICE_TURN_ON,
            service_data,
            blocking=True,
        )


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

    Only returns HS if the light is actually in HS mode (not emulated HS from COLOR_TEMP).
    If from_color_temp_kelvin is explicitly set, returns None (user wants to start from color_temp).

    Args:
        params: FadeParams with optional from_hs_color
        state: Light attributes dict from state.attributes

    Returns:
        Starting HS color (hue 0-360, saturation 0-100), or None if not available
    """
    # If from_color_temp_kelvin is explicitly set, user wants to start from color_temp, not HS
    if params.from_color_temp_kelvin is not None:
        return None
    if params.from_hs_color is not None:
        return params.from_hs_color
    # Only use state HS if light is actually in HS mode (not emulated from COLOR_TEMP)
    color_mode = state.get("color_mode")
    if color_mode == ColorMode.COLOR_TEMP:
        return None
    return state.get(HA_ATTR_HS_COLOR)


def _resolve_start_mireds(params: FadeParams, state: dict) -> int | None:
    """Resolve starting mireds from params.from_color_temp_kelvin or current state.

    FadeParams stores kelvin, but FadeChange needs mireds for linear interpolation.
    This function handles the kelvin->mireds conversion at the boundary.
    Only returns mireds if the light is in COLOR_TEMP mode, or if color_mode is unknown.
    Does NOT return mireds if light is explicitly in HS/RGB mode (those would be emulated).
    If from_hs_color is explicitly set, returns None (user wants to start from HS).

    Args:
        params: FadeParams with optional from_color_temp_kelvin
        state: Light attributes dict from state.attributes

    Returns:
        Starting color temperature in mireds, or None if not available
    """
    # If from_hs_color is explicitly set, user wants to start from HS, not color_temp
    if params.from_hs_color is not None:
        return None
    if params.from_color_temp_kelvin is not None:
        return int(1_000_000 / params.from_color_temp_kelvin)
    # Check color_mode to avoid using emulated values
    color_mode = state.get("color_mode")
    # If color_mode is explicitly HS or other color mode, don't use color_temp (it's emulated)
    if color_mode is not None and color_mode != ColorMode.COLOR_TEMP:
        return None
    # Either color_mode is COLOR_TEMP or unknown - use kelvin if available
    kelvin = state.get(HA_ATTR_COLOR_TEMP_KELVIN)
    if kelvin is not None:
        return int(1_000_000 / kelvin)
    return None


def _resolve_end_mireds(params: FadeParams) -> int | None:
    """Resolve ending mireds from params.color_temp_kelvin.

    Args:
        params: FadeParams with optional color_temp_kelvin

    Returns:
        Ending color temperature in mireds, or None if not specified
    """
    if params.color_temp_kelvin is not None:
        return int(1_000_000 / params.color_temp_kelvin)
    return None


def _get_supported_color_modes(state_attributes: dict) -> set[ColorMode]:
    """Extract supported color modes from state attributes.

    Args:
        state_attributes: Light state attributes dict

    Returns:
        Set of supported ColorMode values
    """
    modes_raw = state_attributes.get(ATTR_SUPPORTED_COLOR_MODES, [])
    return set(modes_raw)


def _supports_brightness(supported_modes: set[ColorMode]) -> bool:
    """Check if light supports brightness control (dimming).

    Args:
        supported_modes: Set of supported ColorMode values

    Returns:
        True if light can be dimmed
    """
    # Any color mode implies brightness support except ONOFF and UNKNOWN
    dimmable_modes = {
        ColorMode.BRIGHTNESS,
        ColorMode.HS,
        ColorMode.RGB,
        ColorMode.RGBW,
        ColorMode.RGBWW,
        ColorMode.XY,
        ColorMode.COLOR_TEMP,
    }
    return bool(supported_modes & dimmable_modes)


def _supports_hs(supported_modes: set[ColorMode]) -> bool:
    """Check if light supports HS color.

    Args:
        supported_modes: Set of supported ColorMode values

    Returns:
        True if light can use HS color
    """
    hs_modes = {
        ColorMode.HS,
        ColorMode.RGB,
        ColorMode.RGBW,
        ColorMode.RGBWW,
        ColorMode.XY,
    }
    return bool(supported_modes & hs_modes)


def _supports_color_temp(supported_modes: set[ColorMode]) -> bool:
    """Check if light supports color temperature.

    Args:
        supported_modes: Set of supported ColorMode values

    Returns:
        True if light can use color temperature
    """
    return ColorMode.COLOR_TEMP in supported_modes


def resolve_fade(
    params: FadeParams,
    state_attributes: dict,
    min_step_delay_ms: int,
    stored_brightness: int = 0,
) -> FadeChange | None:
    """Resolve fade parameters against light state, returning configured FadeChange.

    This function consolidates all resolution and filtering logic:
    - Extracts light capabilities from state attributes
    - Resolves start values from params or state
    - Converts kelvin to mireds (with bounds clamping)
    - Detects hybrid transition scenarios (HS <-> color temp)
    - Filters/converts based on light capabilities
    - Handles non-dimmable lights (single step, zero delay)
    - Auto-fades brightness from 0 when targeting color from off state
    - Returns None if nothing to fade

    Args:
        params: FadeParams from service call
        state_attributes: Light state attributes dict
        min_step_delay_ms: Minimum delay between steps in milliseconds
        stored_brightness: Previously stored brightness (for auto-turn-on from off)

    Returns:
        Configured FadeChange, or None if nothing to fade
    """
    # Extract light capabilities from state
    supported_color_modes = _get_supported_color_modes(state_attributes)
    can_dim = _supports_brightness(supported_color_modes)
    can_hs = _supports_hs(supported_color_modes)
    can_color_temp = _supports_color_temp(supported_color_modes)

    # Extract color temp bounds (kelvin -> mireds with inversion)
    min_kelvin = state_attributes.get(HA_ATTR_MIN_COLOR_TEMP_KELVIN)
    max_kelvin = state_attributes.get(HA_ATTR_MAX_COLOR_TEMP_KELVIN)
    min_mireds = int(1_000_000 / max_kelvin) if max_kelvin else None
    max_mireds = int(1_000_000 / min_kelvin) if min_kelvin else None

    # Resolve brightness values
    start_brightness = _resolve_start_brightness(params, state_attributes)
    end_brightness = _resolve_end_brightness(params, state_attributes)

    # Handle non-dimmable lights (on/off only)
    if not can_dim:
        # Only process if brightness is being targeted
        if end_brightness is None:
            return None
        # Single step with zero delay - just set on/off
        target = 255 if end_brightness > 0 else 0
        return FadeChange(
            start_brightness=start_brightness,
            end_brightness=target,
            transition_ms=0,
            min_step_delay_ms=min_step_delay_ms,
        )

    # Resolve color values
    start_hs = _resolve_start_hs(params, state_attributes)
    end_hs = params.hs_color
    start_mireds = _resolve_start_mireds(params, state_attributes)
    end_mireds = _resolve_end_mireds(params)

    # Clamp mireds to light's supported range
    if start_mireds is not None:
        start_mireds = _clamp_mireds(start_mireds, min_mireds, max_mireds)
    if end_mireds is not None:
        end_mireds = _clamp_mireds(end_mireds, min_mireds, max_mireds)

    # Handle capability filtering - convert unsupported color modes
    if end_mireds is not None and not can_color_temp and can_hs:
        # Convert target color temp to equivalent HS on Planckian locus
        end_hs = _mireds_to_hs(end_mireds)
        end_mireds = None
    if end_hs is not None and not can_hs and can_color_temp:
        # Convert target HS to nearest mireds (if low saturation)
        if _is_on_planckian_locus(end_hs):
            end_mireds = _hs_to_mireds(end_hs)
            end_mireds = _clamp_mireds(end_mireds, min_mireds, max_mireds)
        end_hs = None

    # Filter out unsupported modes entirely
    if not can_hs:
        start_hs = None
        end_hs = None
    if not can_color_temp:
        start_mireds = None
        end_mireds = None

    # Handle HS on Planckian locus -> mireds transition:
    # When start_hs is on the locus (low saturation) and we're targeting mireds,
    # convert start_hs to equivalent start_mireds for smooth interpolation
    if (
        start_hs is not None
        and start_mireds is None
        and end_mireds is not None
        and _is_on_planckian_locus(start_hs)
    ):
        start_mireds = _hs_to_mireds(start_hs)
        start_mireds = _clamp_mireds(start_mireds, min_mireds, max_mireds)
        start_hs = None  # Clear HS since we're doing a pure mireds fade
        _LOGGER.debug(
            "resolve_fade: Converted on-locus start_hs to start_mireds=%s",
            start_mireds,
        )

    # Auto-turn-on: when fading color from off state without explicit brightness,
    # automatically fade brightness from 0 to stored value (or full brightness)
    if (
        start_brightness == 0
        and end_brightness is None
        and (end_hs is not None or end_mireds is not None)
    ):
        # Use stored brightness if available, otherwise full brightness
        end_brightness = stored_brightness if stored_brightness > 0 else 255
        _LOGGER.debug(
            "resolve_fade: Auto-turn-on from off state, end_brightness=%s",
            end_brightness,
        )

    # Check if anything is changing
    brightness_changing = end_brightness is not None and start_brightness != end_brightness
    hs_changing = end_hs is not None and start_hs != end_hs
    mireds_changing = end_mireds is not None and start_mireds != end_mireds

    if not brightness_changing and not hs_changing and not mireds_changing:
        return None  # Nothing to fade

    # Detect hybrid transitions and configure FadeChange accordingly
    hybrid_direction = None
    crossover_step = None
    crossover_hs = None
    crossover_mireds = None

    # HS -> mireds: starting with HS color and targeting mireds
    if (
        start_hs is not None
        and end_mireds is not None
        and end_hs is None
        and start_mireds is None
        and not _is_on_planckian_locus(start_hs)
    ):
        hybrid_direction = "hs_to_mireds"
        # Find crossover point on Planckian locus
        crossover_hs = _mireds_to_hs(end_mireds)
        crossover_mireds = _hs_to_mireds(crossover_hs)
        crossover_mireds = _clamp_mireds(crossover_mireds, min_mireds, max_mireds)

    # mireds -> HS: starting with color temp and targeting HS
    elif (
        start_mireds is not None and end_hs is not None and end_mireds is None and start_hs is None
    ):
        hybrid_direction = "mireds_to_hs"
        # Find crossover point on Planckian locus closest to target HS
        crossover_mireds = _hs_to_mireds(end_hs)
        crossover_mireds = _clamp_mireds(crossover_mireds, min_mireds, max_mireds)
        crossover_hs = _mireds_to_hs(crossover_mireds)

    # Handle missing start values for non-hybrid transitions:
    # When starting from off/unknown state with a color target but no start value,
    # use the closest boundary as start for a visible transition.
    # (Hybrid transitions handle this differently.)
    if not hybrid_direction:
        if end_mireds is not None and start_mireds is None:
            # Use closest boundary (min or max mireds) as start for a visible transition
            if min_mireds is not None and max_mireds is not None:
                dist_to_min = abs(end_mireds - min_mireds)
                dist_to_max = abs(end_mireds - max_mireds)
                start_mireds = min_mireds if dist_to_min <= dist_to_max else max_mireds
            elif min_mireds is not None:
                start_mireds = min_mireds
            elif max_mireds is not None:
                start_mireds = max_mireds
            else:
                # No bounds available, use target as start (no color transition)
                start_mireds = end_mireds
            _LOGGER.debug(
                "resolve_fade: No start_mireds, using boundary mireds=%s as start",
                start_mireds,
            )
        if end_hs is not None and start_hs is None:
            # For HS with no start, use white (0, 0) as start for visible transition
            start_hs = (0.0, 0.0)
            _LOGGER.debug(
                "resolve_fade: No start_hs, using white (0, 0) as start",
            )

    # Create FadeChange
    fade = FadeChange(
        start_brightness=start_brightness if brightness_changing else None,
        end_brightness=end_brightness if brightness_changing else None,
        start_hs=start_hs if (hs_changing or hybrid_direction) else None,
        end_hs=end_hs if (hs_changing or hybrid_direction) else None,
        start_mireds=start_mireds if (mireds_changing or hybrid_direction) else None,
        end_mireds=end_mireds if (mireds_changing or hybrid_direction) else None,
        transition_ms=params.transition_ms,
        min_step_delay_ms=min_step_delay_ms,
        _hybrid_direction=hybrid_direction,
        _crossover_hs=crossover_hs,
        _crossover_mireds=crossover_mireds,
    )

    # Calculate crossover step if hybrid
    if hybrid_direction:
        total_steps = fade.step_count()
        if hybrid_direction == "hs_to_mireds":
            crossover_step = int(total_steps * HYBRID_HS_PHASE_RATIO)
        else:  # mireds_to_hs
            crossover_step = int(total_steps * (1 - HYBRID_HS_PHASE_RATIO))
        # Update the crossover step (need to set directly since it's a private field)
        fade._crossover_step = crossover_step  # noqa: SLF001

    return fade


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


def _clamp_mireds(mireds: int, min_mireds: int | None, max_mireds: int | None) -> int:
    """Clamp mireds to the light's supported range.

    Args:
        mireds: The mireds value to clamp
        min_mireds: Minimum mireds (coolest/highest kelvin), or None for no limit
        max_mireds: Maximum mireds (warmest/lowest kelvin), or None for no limit

    Returns:
        Clamped mireds value
    """
    if min_mireds is None and max_mireds is None:
        return mireds
    result = mireds
    if min_mireds is not None:
        result = max(result, min_mireds)
    if max_mireds is not None:
        result = min(result, max_mireds)
    return result


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
        params.color_temp_kelvin is not None or params.from_color_temp_kelvin is not None
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

    # Check if this is an expected state change (from our service calls)
    if _match_and_remove_expected(entity_id, new_state):
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

    if matched is not None:
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
        intended_kelvin = new_state.attributes.get(HA_ATTR_COLOR_TEMP_KELVIN)

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
