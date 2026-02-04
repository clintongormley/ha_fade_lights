"""FadeChange and FadeStep models for the Fade Lights integration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from homeassistant.components.light import ATTR_BRIGHTNESS, ATTR_SUPPORTED_COLOR_MODES
from homeassistant.components.light import ATTR_COLOR_TEMP_KELVIN as HA_ATTR_COLOR_TEMP_KELVIN
from homeassistant.components.light import ATTR_HS_COLOR as HA_ATTR_HS_COLOR
from homeassistant.components.light import ATTR_MAX_COLOR_TEMP_KELVIN as HA_ATTR_MAX_COLOR_TEMP_KELVIN
from homeassistant.components.light import ATTR_MIN_COLOR_TEMP_KELVIN as HA_ATTR_MIN_COLOR_TEMP_KELVIN
from homeassistant.components.light.const import ColorMode

from .const import (
    HYBRID_HS_PHASE_RATIO,
    MIN_BRIGHTNESS_DELTA,
    MIN_HUE_DELTA,
    MIN_MIREDS_DELTA,
    MIN_SATURATION_DELTA,
    PLANCKIAN_LOCUS_HS,
    PLANCKIAN_LOCUS_SATURATION_THRESHOLD,
)

if TYPE_CHECKING:
    from .fade_params import FadeParams

_LOGGER = logging.getLogger(__name__)


# =============================================================================
# Resolution Helper Functions
# =============================================================================


def _resolve_start_brightness(params: FadeParams, state: dict[str, Any]) -> int:
    """Resolve starting brightness from params.from_brightness_pct or current state.

    Args:
        params: FadeParams with optional from_brightness_pct
        state: Light attributes dict from state.attributes

    Returns:
        Starting brightness (0-255 scale)
    """
    if params.from_brightness_pct is not None:
        return int(params.from_brightness_pct * 255 / 100)
    # When light is off (no brightness in state), treat as 0
    brightness = state.get(ATTR_BRIGHTNESS)
    return int(brightness) if brightness is not None else 0


def _resolve_end_brightness(params: FadeParams) -> int | None:
    """Resolve ending brightness from params.brightness_pct.

    Args:
        params: FadeParams with optional brightness_pct

    Returns:
        Ending brightness (0-255 scale), or None if not specified
    """
    if params.brightness_pct is not None:
        return int(params.brightness_pct * 255 / 100)
    return None


def _resolve_start_hs(
    params: FadeParams, state: dict[str, Any]
) -> tuple[float, float] | None:
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


def _resolve_start_mireds(params: FadeParams, state: dict[str, Any]) -> int | None:
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


def _get_supported_color_modes(state_attributes: dict[str, Any]) -> set[ColorMode]:
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


# =============================================================================
# Color Conversion Utilities (Planckian Locus)
# =============================================================================


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


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FadeStep:
    """A single step in a fade sequence.

    All values are optional - only include attributes being faded.
    Color temperature is in kelvin for direct use with Home Assistant.
    """

    brightness: int | None = None
    hs_color: tuple[float, float] | None = None
    color_temp_kelvin: int | None = None


@dataclass
class FadeChange:  # pylint: disable=too-many-instance-attributes
    """A fade operation with flat step generation and hybrid transition support.

    This class represents a change from start to end values for brightness,
    HS color, and/or color temperature. It generates steps on-demand via
    an iterator pattern rather than pre-building a list.

    Hybrid transitions (HS <-> color temp) are handled internally by tracking
    a crossover point where the color mode switches. This enables flat step
    generation for ease-in/ease-out implementation.

    Color temperature is stored internally as mireds for linear interpolation,
    but converted to kelvin when generating FadeStep output.

    All start/end values are optional - only include dimensions being faded.
    """

    # Brightness (0-255 scale)
    start_brightness: int | None = None
    end_brightness: int | None = None

    # HS color (hue 0-360, saturation 0-100)
    start_hs: tuple[float, float] | None = None
    end_hs: tuple[float, float] | None = None

    # Color temperature (mireds, internal use for linear interpolation)
    start_mireds: int | None = None
    end_mireds: int | None = None

    # Timing
    transition_ms: int = 0
    min_step_delay_ms: int = 100

    # Hybrid transition tracking (private)
    # "hs_to_mireds" | "mireds_to_hs" | None
    _hybrid_direction: str | None = field(default=None, repr=False)
    _crossover_step: int | None = field(default=None, repr=False)
    _crossover_hs: tuple[float, float] | None = field(default=None, repr=False)
    _crossover_mireds: int | None = field(default=None, repr=False)

    # Iterator state (private)
    _current_step: int = field(default=0, repr=False)
    _step_count: int | None = field(default=None, repr=False)

    @classmethod
    def resolve(
        cls,
        params: FadeParams,
        state_attributes: dict[str, Any],
        min_step_delay_ms: int,
        stored_brightness: int = 0,
    ) -> FadeChange | None:
        """Factory that resolves fade parameters against light capabilities.

        This method consolidates all resolution and filtering logic:
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
        end_brightness = _resolve_end_brightness(params)

        # Handle non-dimmable lights (on/off only)
        if not can_dim:
            # Only process if brightness is being targeted
            if end_brightness is None:
                return None
            # Single step with zero delay - just set on/off
            target = 255 if end_brightness > 0 else 0
            return cls(
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
                "FadeChange.resolve: Converted on-locus start_hs to start_mireds=%s",
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
                "FadeChange.resolve: Auto-turn-on from off state, end_brightness=%s",
                end_brightness,
            )

        # Check if anything is changing
        brightness_changing = (
            end_brightness is not None and start_brightness != end_brightness
        )
        hs_changing = end_hs is not None and start_hs != end_hs
        mireds_changing = end_mireds is not None and start_mireds != end_mireds

        if not brightness_changing and not hs_changing and not mireds_changing:
            return None  # Nothing to fade

        # Detect hybrid transitions and configure FadeChange accordingly
        hybrid_direction = None
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
            start_mireds is not None
            and end_hs is not None
            and end_mireds is None
            and start_hs is None
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
                    start_mireds = (
                        min_mireds if dist_to_min <= dist_to_max else max_mireds
                    )
                elif min_mireds is not None:
                    start_mireds = min_mireds
                elif max_mireds is not None:
                    start_mireds = max_mireds
                else:
                    # No bounds available, use target as start (no color transition)
                    start_mireds = end_mireds
                _LOGGER.debug(
                    "FadeChange.resolve: No start_mireds, using boundary mireds=%s as start",
                    start_mireds,
                )
            if end_hs is not None and start_hs is None:
                # For HS with no start, use white (0, 0) as start for visible transition
                start_hs = (0.0, 0.0)
                _LOGGER.debug(
                    "FadeChange.resolve: No start_hs, using white (0, 0) as start",
                )

        # Create FadeChange
        fade = cls(
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

    def step_count(self) -> int:
        """Calculate optimal step count based on change magnitude and time constraints.

        The algorithm:
        1. Calculate ideal steps per dimension: change / minimum_delta
        2. Take the maximum across all changing dimensions (smoothest dimension wins)
        3. Constrain by time: if ideal * min_step_delay_ms > transition_ms, use time-limited

        Returns:
            Optimal number of steps (at least 1)
        """
        if self._step_count is not None:
            return self._step_count

        ideal_steps: list[int] = []

        # Brightness change
        if self.start_brightness is not None and self.end_brightness is not None:
            brightness_change = abs(self.end_brightness - self.start_brightness)
            if brightness_change > 0:
                ideal_steps.append(brightness_change // MIN_BRIGHTNESS_DELTA)

        # HS color change (for non-hybrid or hybrid HS phases)
        if self._hybrid_direction == "hs_to_mireds":
            # HS phase: from start_hs to crossover_hs
            if self.start_hs is not None and self._crossover_hs is not None:
                self._add_hs_steps(ideal_steps, self.start_hs, self._crossover_hs)
            # Mireds phase: from crossover_mireds to end_mireds
            if self._crossover_mireds is not None and self.end_mireds is not None:
                mireds_change = abs(self.end_mireds - self._crossover_mireds)
                if mireds_change > 0:
                    ideal_steps.append(mireds_change // MIN_MIREDS_DELTA)
        elif self._hybrid_direction == "mireds_to_hs":
            # Mireds phase: from start_mireds to crossover_mireds
            if self.start_mireds is not None and self._crossover_mireds is not None:
                mireds_change = abs(self._crossover_mireds - self.start_mireds)
                if mireds_change > 0:
                    ideal_steps.append(mireds_change // MIN_MIREDS_DELTA)
            # HS phase: from crossover_hs to end_hs
            if self._crossover_hs is not None and self.end_hs is not None:
                self._add_hs_steps(ideal_steps, self._crossover_hs, self.end_hs)
        else:
            # Non-hybrid: standard HS and mireds handling
            if self.start_hs is not None and self.end_hs is not None:
                self._add_hs_steps(ideal_steps, self.start_hs, self.end_hs)

            if self.start_mireds is not None and self.end_mireds is not None:
                mireds_change = abs(self.end_mireds - self.start_mireds)
                if mireds_change > 0:
                    ideal_steps.append(mireds_change // MIN_MIREDS_DELTA)

        ideal = max(ideal_steps) if ideal_steps else 1
        max_by_time = (
            self.transition_ms // self.min_step_delay_ms
            if self.min_step_delay_ms > 0
            else ideal
        )

        self._step_count = max(1, min(ideal, max_by_time))
        return self._step_count

    def _add_hs_steps(
        self,
        ideal_steps: list[int],
        start_hs: tuple[float, float],
        end_hs: tuple[float, float],
    ) -> None:
        """Add ideal step counts for HS color change to the list."""
        hue_diff = abs(end_hs[0] - start_hs[0])
        # Handle hue wraparound (shortest path)
        if hue_diff > 180:
            hue_diff = 360 - hue_diff
        if hue_diff > 0:
            ideal_steps.append(int(hue_diff / MIN_HUE_DELTA))

        sat_diff = abs(end_hs[1] - start_hs[1])
        if sat_diff > 0:
            ideal_steps.append(int(sat_diff / MIN_SATURATION_DELTA))

    def delay_ms(self) -> float:
        """Calculate delay between steps.

        Returns:
            Delay in milliseconds, or 0 if step_count <= 1.
        """
        count = self.step_count()
        if count <= 1:
            return 0.0
        return self.transition_ms / count

    def reset(self) -> None:
        """Reset iterator to beginning."""
        self._current_step = 0

    def has_next(self) -> bool:
        """Check if more steps remain."""
        return self._current_step < self.step_count()

    def next_step(self) -> FadeStep:
        """Generate and return next step using interpolation.

        For hybrid transitions, emits different color attributes before/after
        the crossover point. Color temperature is converted from internal
        mireds to kelvin.

        Raises:
            StopIteration: If no more steps remain.
        """
        if not self.has_next():
            raise StopIteration

        self._current_step += 1
        count = self.step_count()

        # Use t=1.0 for the last step to hit target exactly
        t = self._current_step / count

        # Handle hybrid transitions
        if self._hybrid_direction is not None:
            return self._interpolate_hybrid_step(t)

        # Non-hybrid: standard interpolation
        return FadeStep(
            brightness=self._interpolate_brightness(t),
            hs_color=self._interpolate_hs(t),
            color_temp_kelvin=self._interpolate_color_temp_kelvin(t),
        )

    def _interpolate_hybrid_step(self, t: float) -> FadeStep:
        """Interpolate a step for hybrid HS <-> mireds transitions.

        Args:
            t: Overall interpolation factor (0.0 = start, 1.0 = end)

        Returns:
            FadeStep with appropriate color attribute based on phase
        """
        crossover_step = self._crossover_step or 0
        count = self.step_count()

        # Calculate crossover_t (the t value at crossover)
        crossover_t = crossover_step / count if count > 0 else 0.5

        if self._hybrid_direction == "hs_to_mireds":
            # Before/at crossover: emit hs_color
            if self._current_step <= crossover_step:
                # Map t to [0, crossover_t] -> [0, 1] for HS interpolation
                phase_t = t / crossover_t if crossover_t > 0 else 1.0
                hs_color = self._interpolate_hs_between(
                    self.start_hs, self._crossover_hs, phase_t
                )
                return FadeStep(
                    brightness=self._interpolate_brightness(t),
                    hs_color=hs_color,
                )
            # After crossover: emit color_temp_kelvin
            # Map t from [crossover_t, 1] -> [0, 1] for mireds interpolation
            remaining_t = 1.0 - crossover_t
            phase_t = (t - crossover_t) / remaining_t if remaining_t > 0 else 1.0
            mireds = self._interpolate_mireds_between(
                self._crossover_mireds, self.end_mireds, phase_t
            )
            kelvin = int(1_000_000 / mireds) if mireds else None
            return FadeStep(
                brightness=self._interpolate_brightness(t),
                color_temp_kelvin=kelvin,
            )

        # mireds_to_hs
        # Before/at crossover: emit color_temp_kelvin
        if self._current_step <= crossover_step:
            # Map t to [0, crossover_t] -> [0, 1] for mireds interpolation
            phase_t = t / crossover_t if crossover_t > 0 else 1.0
            mireds = self._interpolate_mireds_between(
                self.start_mireds, self._crossover_mireds, phase_t
            )
            kelvin = int(1_000_000 / mireds) if mireds else None
            return FadeStep(
                brightness=self._interpolate_brightness(t),
                color_temp_kelvin=kelvin,
            )
        # After crossover: emit hs_color
        # Map t from [crossover_t, 1] -> [0, 1] for HS interpolation
        remaining_t = 1.0 - crossover_t
        phase_t = (t - crossover_t) / remaining_t if remaining_t > 0 else 1.0
        hs_color = self._interpolate_hs_between(self._crossover_hs, self.end_hs, phase_t)
        return FadeStep(
            brightness=self._interpolate_brightness(t),
            hs_color=hs_color,
        )

    def _interpolate_hs_between(
        self,
        start: tuple[float, float] | None,
        end: tuple[float, float] | None,
        t: float,
    ) -> tuple[float, float] | None:
        """Interpolate HS color between two points."""
        if start is None or end is None:
            return None

        start_hue, start_sat = start
        end_hue, end_sat = end

        hue_diff = end_hue - start_hue
        if hue_diff > 180:
            hue_diff -= 360
        elif hue_diff < -180:
            hue_diff += 360

        hue = (start_hue + hue_diff * t) % 360
        sat = start_sat + (end_sat - start_sat) * t

        return (round(hue, 2), round(sat, 2))

    def _interpolate_mireds_between(
        self,
        start: int | None,
        end: int | None,
        t: float,
    ) -> int | None:
        """Interpolate mireds between two points."""
        if start is None or end is None:
            return None
        return round(start + (end - start) * t)

    def _interpolate_brightness(self, t: float) -> int | None:
        """Interpolate brightness at factor t.

        Args:
            t: Interpolation factor (0.0 = start, 1.0 = end)

        Returns:
            Interpolated brightness, or None if brightness not set.
            Skips brightness level 1 (many lights behave oddly at this level).
        """
        if self.start_brightness is None or self.end_brightness is None:
            return None
        brightness = round(
            self.start_brightness + (self.end_brightness - self.start_brightness) * t
        )
        # Skip brightness level 1 (many lights behave oddly at this level)
        if brightness == 1:
            brightness = 0 if self.end_brightness < self.start_brightness else 2
        return brightness

    def _interpolate_hs(self, t: float) -> tuple[float, float] | None:
        """Interpolate HS color at factor t, handling hue wraparound.

        Args:
            t: Interpolation factor (0.0 = start, 1.0 = end)

        Returns:
            Interpolated (hue, saturation), or None if HS not set.
        """
        return self._interpolate_hs_between(self.start_hs, self.end_hs, t)

    def _interpolate_color_temp_kelvin(self, t: float) -> int | None:
        """Interpolate color temperature, returning kelvin.

        Interpolation is done in mireds (linear in color space),
        then converted to kelvin for the output.

        Args:
            t: Interpolation factor (0.0 = start, 1.0 = end)

        Returns:
            Interpolated color temperature in kelvin, or None if not set.
        """
        mireds = self._interpolate_mireds_between(self.start_mireds, self.end_mireds, t)
        if mireds is None:
            return None
        return int(1_000_000 / mireds)
