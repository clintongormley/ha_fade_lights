"""FadeChange and FadeStep models for the Fade Lights integration."""

from __future__ import annotations

from dataclasses import dataclass, field

from .const import (
    MIN_BRIGHTNESS_DELTA,
    MIN_HUE_DELTA,
    MIN_MIREDS_DELTA,
    MIN_SATURATION_DELTA,
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