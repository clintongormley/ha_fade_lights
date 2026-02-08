"""ExpectedState and ExpectedValues models for the Fado integration."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from .const import (
    BRIGHTNESS_TOLERANCE,
    HUE_TOLERANCE,
    KELVIN_TOLERANCE,
    SATURATION_TOLERANCE,
    STALE_THRESHOLD,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class ExpectedValues:
    """Expected state values for manual intervention detection.

    Only includes dimensions being actively faded. None means
    that dimension is not being tracked for this fade.

    When from_* fields are present, indicates a transition range
    (used with native_transitions=True). Matching accepts values
    between from_* and target. When from_* is None, uses point
    matching with tolerance (existing behavior).
    """

    # Target values (always present when tracking that dimension)
    brightness: int | None = None
    hs_color: tuple[float, float] | None = None
    color_temp_kelvin: int | None = None

    # Source values for transitions (present when use_transition=True)
    from_brightness: int | None = None
    from_hs_color: tuple[float, float] | None = None
    from_color_temp_kelvin: int | None = None

    def __str__(self) -> str:
        """Format without class name for cleaner logs."""
        parts = []
        if self.brightness is not None:
            if self.from_brightness is not None:
                parts.append(f"brightness={self.from_brightness}->{self.brightness}")
            else:
                parts.append(f"brightness={self.brightness}")
        if self.hs_color is not None:
            if self.from_hs_color is not None:
                parts.append(f"hs_color={self.from_hs_color}->{self.hs_color}")
            else:
                parts.append(f"hs_color={self.hs_color}")
        if self.color_temp_kelvin is not None:
            if self.from_color_temp_kelvin is not None:
                parts.append(
                    f"color_temp_kelvin={self.from_color_temp_kelvin}->{self.color_temp_kelvin}"
                )
            else:
                parts.append(f"color_temp_kelvin={self.color_temp_kelvin}")
        return "(" + (", ".join(parts) if parts else "empty") + ")"


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
            "%s: add%s -> count=%d",
            self.entity_id,
            expected,
            len(self.values),
        )

    def get_condition(self) -> asyncio.Condition:
        """Get or create the condition for waiting, pruning stale values first."""
        _LOGGER.debug(
            "%s: get_condition() count=%d",
            self.entity_id,
            len(self.values),
        )

        # Prune stale values
        now = time.monotonic()
        stale_count = sum(1 for _, ts in self.values if now - ts > STALE_THRESHOLD)
        if stale_count:
            _LOGGER.debug(
                "%s: -> removing %d stale entries",
                self.entity_id,
                stale_count,
            )
        self.values = [(v, ts) for v, ts in self.values if now - ts <= STALE_THRESHOLD]

        _LOGGER.debug(
            "%s: -> after prune count=%d",
            self.entity_id,
            len(self.values),
        )

        if self._condition is None:
            self._condition = asyncio.Condition()

        # Notify if all values were pruned
        if not self.values:
            _LOGGER.debug(
                "%s: -> values empty, notifying condition",
                self.entity_id,
            )
            condition = self._condition
            asyncio.get_event_loop().call_soon(lambda: asyncio.create_task(self._notify(condition)))

        return self._condition

    def match_and_remove(self, actual: ExpectedValues) -> ExpectedValues | None:
        """Match actual values against expected values.

        Only removes from queue on exact match (final target reached).
        Range matches are kept in queue to wait for final value event.

        Args:
            actual: The actual values from the light state

        Returns:
            The matched ExpectedValues, or None if no match.
        """
        _LOGGER.debug(
            "%s: match_and_remove%s count=%d",
            self.entity_id,
            actual,
            len(self.values),
        )

        matched_index: int | None = None
        matched_value: ExpectedValues | None = None
        match_type: str | None = None

        for i, (expected, _) in enumerate(self.values):
            match_result = self._values_match(expected, actual)
            if match_result is not None:
                matched_index = i
                matched_value = expected
                match_type = match_result
                break

        if matched_index is None or matched_value is None or match_type is None:
            _LOGGER.debug("%s: -> no match found", self.entity_id)
            return None

        # Only remove on exact match (final value reached)
        if match_type == "exact":
            # Remove matched value and all older entries (handles event coalescing)
            del self.values[: matched_index + 1]
            _LOGGER.debug(
                "%s: -> matched=%s (exact) remaining=%d",
                self.entity_id,
                matched_value,
                len(self.values),
            )

            # Notify condition if set is now empty
            if not self.values and self._condition is not None:
                _LOGGER.debug(
                    "%s: -> values empty, notifying condition",
                    self.entity_id,
                )
                condition = self._condition
                asyncio.get_event_loop().call_soon(
                    lambda: asyncio.create_task(self._notify(condition))
                )
        else:
            # Range match - keep in queue for final value
            _LOGGER.debug(
                "%s: -> matched=%s (range) NOT removing, remaining=%d",
                self.entity_id,
                matched_value,
                len(self.values),
            )

        return matched_value

    def _values_match(self, expected: ExpectedValues, actual: ExpectedValues) -> str | None:
        """Check if actual values match expected values.

        Returns:
            "exact" - Matches target within tolerance (should remove from queue)
            "range" - Within transition range (keep in queue for final value)
            None - No match
        """
        match_types = []

        # Check brightness if tracked
        if expected.brightness is not None:
            brightness_match = self._brightness_match(expected, actual)
            if brightness_match is None:
                return None
            match_types.append(brightness_match)

        # Check HS color if tracked
        if expected.hs_color is not None:
            hs_match = self._hs_match(expected, actual)
            if hs_match is None:
                return None
            match_types.append(hs_match)

        # Check color temp if tracked
        if expected.color_temp_kelvin is not None:
            kelvin_match = self._kelvin_match(expected, actual)
            if kelvin_match is None:
                return None
            match_types.append(kelvin_match)

        # Return weakest match type (range < exact)
        if "range" in match_types:
            return "range"
        return "exact"

    def _brightness_match(self, expected: ExpectedValues, actual: ExpectedValues) -> str | None:
        """Check if brightness matches. Returns match type or None."""
        if actual.brightness is None or expected.brightness is None:
            return None

        # Phase 1: Exact match with tolerance (prioritize target)
        if expected.brightness == 0:
            if actual.brightness == 0:
                return "exact"
        elif abs(expected.brightness - actual.brightness) <= BRIGHTNESS_TOLERANCE:
            return "exact"

        # Phase 2: Range match (only if transitioning)
        if expected.from_brightness is not None:
            min_val = min(expected.from_brightness, expected.brightness)
            max_val = max(expected.from_brightness, expected.brightness)
            if min_val <= actual.brightness <= max_val:
                return "range"

        return None

    def _hs_match(self, expected: ExpectedValues, actual: ExpectedValues) -> str | None:
        """Check if HS color matches. Returns match type or None."""
        if actual.hs_color is None or expected.hs_color is None:
            return None

        # Phase 1: Exact match with tolerance (prioritize target)
        if self._hs_exact_match(expected.hs_color, actual.hs_color):
            return "exact"

        # Phase 2: Range match (only if transitioning)
        if expected.from_hs_color is not None and self._hs_range_match(
            expected.from_hs_color, expected.hs_color, actual.hs_color
        ):
            return "range"

        return None

    def _hs_exact_match(
        self,
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

    def _hs_range_match(
        self,
        from_hs: tuple[float, float],
        to_hs: tuple[float, float],
        actual_hs: tuple[float, float],
    ) -> bool:
        """Check if actual HS is within transition range from_hs -> to_hs."""
        from_hue, from_sat = from_hs
        to_hue, to_sat = to_hs
        actual_hue, actual_sat = actual_hs

        # Check saturation (simple range)
        min_sat = min(from_sat, to_sat)
        max_sat = max(from_sat, to_sat)
        if not (min_sat <= actual_sat <= max_sat):
            return False

        # Check hue (handle wraparound)
        hue_diff = abs(from_hue - to_hue)
        if hue_diff > 180:  # Wraparound case
            # Accept if actual is in either range
            min_hue = min(from_hue, to_hue)
            max_hue = max(from_hue, to_hue)
            # Accept values outside the "gap"
            return actual_hue >= max_hue or actual_hue <= min_hue
        else:  # No wraparound
            min_hue = min(from_hue, to_hue)
            max_hue = max(from_hue, to_hue)
            return min_hue <= actual_hue <= max_hue

    def _kelvin_match(self, expected: ExpectedValues, actual: ExpectedValues) -> str | None:
        """Check if color temp kelvin matches. Returns match type or None."""
        if actual.color_temp_kelvin is None or expected.color_temp_kelvin is None:
            return None

        # Phase 1: Exact match with tolerance (prioritize target)
        if abs(expected.color_temp_kelvin - actual.color_temp_kelvin) <= KELVIN_TOLERANCE:
            return "exact"

        # Phase 2: Range match (only if transitioning)
        if expected.from_color_temp_kelvin is not None:
            min_val = min(expected.from_color_temp_kelvin, expected.color_temp_kelvin)
            max_val = max(expected.from_color_temp_kelvin, expected.color_temp_kelvin)
            if min_val <= actual.color_temp_kelvin <= max_val:
                return "range"

        return None

    @property
    def is_empty(self) -> bool:
        """Check if there are no expected values."""
        return not self.values

    async def wait_and_clear(self, timeout: float = 0.5) -> None:
        """Wait for pending events then clear all remaining entries.

        Call this after a fade completes to allow late events to flush
        before clearing stale entries.

        Args:
            timeout: Maximum seconds to wait for events (default 0.5s)
        """
        if self.values:
            condition = self.get_condition()
            try:
                async with condition:
                    await asyncio.wait_for(condition.wait(), timeout=timeout)
            except TimeoutError:
                pass  # Timeout is expected - just clear below

        # Clear any remaining entries
        self.values.clear()
        _LOGGER.debug(
            "%s: wait_and_clear() completed",
            self.entity_id,
        )

    @staticmethod
    async def _notify(condition: asyncio.Condition) -> None:
        """Notify all waiters on the condition."""
        async with condition:
            condition.notify_all()
