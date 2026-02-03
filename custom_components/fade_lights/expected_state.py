"""ExpectedState and ExpectedValues models for the Fade Lights integration."""

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
        """Match actual values against expected values, remove if found, notify if empty.

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

        for i, (expected, _) in enumerate(self.values):
            if self._values_match(expected, actual):
                matched_index = i
                matched_value = expected
                break

        if matched_index is None or matched_value is None:
            _LOGGER.debug("%s: -> no match found", self.entity_id)
            return None

        # Remove matched value and all older entries (handles event coalescing)
        del self.values[: matched_index + 1]
        _LOGGER.debug(
            "%s: -> matched=%s remaining=%d",
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
            # Schedule notification (can't await in callback context)
            condition = self._condition
            asyncio.get_event_loop().call_soon(lambda: asyncio.create_task(self._notify(condition)))

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

        # Check color temp if tracked
        if expected.color_temp_kelvin is not None:
            if actual.color_temp_kelvin is None:
                return False
            if abs(expected.color_temp_kelvin - actual.color_temp_kelvin) > KELVIN_TOLERANCE:
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
