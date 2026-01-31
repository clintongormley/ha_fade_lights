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
        hs_color = self._interpolate_hs_between(
            self._crossover_hs, self.end_hs, phase_t
        )
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
