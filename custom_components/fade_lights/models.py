"""Data models for the Fade Lights integration."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from .const import (
    BRIGHTNESS_TOLERANCE,
    DEFAULT_TRANSITION,
    HUE_TOLERANCE,
    MIN_BRIGHTNESS_DELTA,
    MIN_HUE_DELTA,
    MIN_MIREDS_DELTA,
    MIN_SATURATION_DELTA,
    MIREDS_TOLERANCE,
    SATURATION_TOLERANCE,
    STALE_THRESHOLD,
)

_LOGGER = logging.getLogger(__name__)


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

    def has_target(self) -> bool:
        """Check if any fade target values are specified."""
        return (
            self.brightness_pct is not None
            or self.hs_color is not None
            or self.color_temp_mireds is not None
        )

    def has_from_target(self) -> bool:
        """Check if any from values are specified."""
        return (
            self.from_brightness_pct is not None
            or self.from_hs_color is not None
            or self.from_color_temp_mireds is not None
        )


@dataclass
class FadeStep:
    """A single step in a fade sequence.

    All values are optional - only include attributes being faded.
    """

    brightness: int | None = None
    hs_color: tuple[float, float] | None = None
    color_temp_mireds: int | None = None


@dataclass
class FadeChange:
    """A single phase of a fade operation with iterator-based step generation.

    This class represents a change from start to end values for brightness,
    HS color, and/or color temperature. It generates steps on-demand via
    an iterator pattern rather than pre-building a list.

    All start/end values are optional - only include dimensions being faded.
    """

    # Brightness (0-255 scale)
    start_brightness: int | None = None
    end_brightness: int | None = None

    # HS color (hue 0-360, saturation 0-100)
    start_hs: tuple[float, float] | None = None
    end_hs: tuple[float, float] | None = None

    # Color temperature (mireds)
    start_mireds: int | None = None
    end_mireds: int | None = None

    # Timing
    transition_ms: int = 0
    min_step_delay_ms: int = 100

    # Iterator state (private)
    _current_step: int = field(default=0, repr=False)
    _step_count: int | None = field(default=None, repr=False)

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

        # HS color change
        if self.start_hs is not None and self.end_hs is not None:
            hue_diff = abs(self.end_hs[0] - self.start_hs[0])
            # Handle hue wraparound (shortest path)
            if hue_diff > 180:
                hue_diff = 360 - hue_diff
            if hue_diff > 0:
                ideal_steps.append(int(hue_diff / MIN_HUE_DELTA))

            sat_diff = abs(self.end_hs[1] - self.start_hs[1])
            if sat_diff > 0:
                ideal_steps.append(int(sat_diff / MIN_SATURATION_DELTA))

        # Mireds change
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

        Raises:
            StopIteration: If no more steps remain.
        """
        if not self.has_next():
            raise StopIteration

        self._current_step += 1
        count = self.step_count()

        # Use t=1.0 for the last step to hit target exactly
        t = self._current_step / count

        return FadeStep(
            brightness=self._interpolate_brightness(t),
            hs_color=self._interpolate_hs(t),
            color_temp_mireds=self._interpolate_mireds(t),
        )

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
        if self.start_hs is None or self.end_hs is None:
            return None

        # Interpolate hue with wraparound (shortest path)
        start_hue, start_sat = self.start_hs
        end_hue, end_sat = self.end_hs

        hue_diff = end_hue - start_hue
        if hue_diff > 180:
            hue_diff -= 360
        elif hue_diff < -180:
            hue_diff += 360

        hue = (start_hue + hue_diff * t) % 360

        # Linear interpolation for saturation
        sat = start_sat + (end_sat - start_sat) * t

        return (round(hue, 2), round(sat, 2))

    def _interpolate_mireds(self, t: float) -> int | None:
        """Interpolate mireds at factor t.

        Args:
            t: Interpolation factor (0.0 = start, 1.0 = end)

        Returns:
            Interpolated mireds, or None if mireds not set.
        """
        if self.start_mireds is None or self.end_mireds is None:
            return None
        return round(self.start_mireds + (self.end_mireds - self.start_mireds) * t)


@dataclass
class ExpectedValues:
    """Expected state values for manual intervention detection.

    Only includes dimensions being actively faded. None means
    that dimension is not being tracked for this fade.
    """

    brightness: int | None = None
    hs_color: tuple[float, float] | None = None
    color_temp_mireds: int | None = None


@dataclass
class ExpectedState:
    """Track expected values (brightness + colors) and provide synchronization for waiting."""

    entity_id: str
    values: list[tuple[ExpectedValues, float]] = field(default_factory=list)
    _condition: asyncio.Condition | None = field(default=None, repr=False)

    def add(self, expected: ExpectedValues) -> None:
        """Add an expected value with current timestamp."""
        self.values.append((expected, time.monotonic()))
        _LOGGER.debug(
            "%s: ExpectedState.add(%s) -> count=%d",
            self.entity_id,
            expected,
            len(self.values),
        )

    def get_condition(self) -> asyncio.Condition:
        """Get or create the condition for waiting, pruning stale values first."""
        _LOGGER.debug(
            "%s: ExpectedState.get_condition() count=%d",
            self.entity_id,
            len(self.values),
        )

        # Prune stale values
        now = time.monotonic()
        stale_count = sum(1 for _, ts in self.values if now - ts > STALE_THRESHOLD)
        if stale_count:
            _LOGGER.debug(
                "%s: ExpectedState.get_condition() removing %d stale entries",
                self.entity_id,
                stale_count,
            )
        self.values = [(v, ts) for v, ts in self.values if now - ts <= STALE_THRESHOLD]

        _LOGGER.debug(
            "%s: ExpectedState.get_condition() after prune count=%d",
            self.entity_id,
            len(self.values),
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

    def match_and_remove(self, actual: ExpectedValues) -> ExpectedValues | None:
        """Match actual values against expected values, remove if found, notify if empty.

        Args:
            actual: The actual values from the light state

        Returns:
            The matched ExpectedValues, or None if no match.
        """
        _LOGGER.debug(
            "%s: ExpectedState.match_and_remove(%s) count=%d",
            self.entity_id,
            actual,
            len(self.values),
        )

        matched_index: int | None = None
        matched_value: ExpectedValues | None = None

        for i, (expected, _) in enumerate(self.values):
            if self._values_match(expected, actual):
                matched_index = i
                matched_value = expected
                break

        if matched_value is None:
            _LOGGER.debug(
                "%s: ExpectedState.match_and_remove(%s) -> no match found",
                self.entity_id,
                actual,
            )
            return None

        # Remove matched value
        del self.values[matched_index]
        _LOGGER.debug(
            "%s: ExpectedState.match_and_remove(%s) matched=%s remaining=%d",
            self.entity_id,
            actual,
            matched_value,
            len(self.values),
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

    def _values_match(self, expected: ExpectedValues, actual: ExpectedValues) -> bool:
        """Check if actual values match expected values within tolerances.

        Only checks dimensions that are being tracked (non-None in expected).
        Untracked dimensions in expected are ignored.
        """
        # Check brightness if tracked
        if expected.brightness is not None:
            if actual.brightness is None:
                return False
            # Special case: brightness 0 must match exactly
            if expected.brightness == 0:
                if actual.brightness != 0:
                    return False
            elif abs(expected.brightness - actual.brightness) > BRIGHTNESS_TOLERANCE:
                return False

        # Check HS color if tracked
        if expected.hs_color is not None:
            if actual.hs_color is None:
                return False
            if not self._hs_match(expected.hs_color, actual.hs_color):
                return False

        # Check mireds if tracked
        if expected.color_temp_mireds is not None:
            if actual.color_temp_mireds is None:
                return False
            if abs(expected.color_temp_mireds - actual.color_temp_mireds) > MIREDS_TOLERANCE:
                return False

        return True

    @staticmethod
    def _hs_match(
        expected: tuple[float, float],
        actual: tuple[float, float],
    ) -> bool:
        """Check if two HS colors match within tolerance, handling hue wraparound."""
        expected_hue, expected_sat = expected
        actual_hue, actual_sat = actual

        # Check saturation first (simple linear comparison)
        if abs(expected_sat - actual_sat) > SATURATION_TOLERANCE:
            return False

        # Check hue with wraparound (0 and 360 are the same)
        hue_diff = abs(expected_hue - actual_hue)
        if hue_diff > 180:
            hue_diff = 360 - hue_diff

        return hue_diff <= HUE_TOLERANCE

    @property
    def is_empty(self) -> bool:
        """Check if there are no expected values."""
        return not self.values

    @staticmethod
    async def _notify(condition: asyncio.Condition) -> None:
        """Notify all waiters on the condition."""
        async with condition:
            condition.notify_all()
