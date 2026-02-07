"""Tests for the FadeChange dataclass."""

from __future__ import annotations

import pytest

from custom_components.fado.fade_change import FadeChange, FadeStep


class TestFadeChangeStepCount:
    """Test step count calculation for FadeChange."""

    def test_brightness_only(self) -> None:
        """Test step count for brightness-only change."""
        change = FadeChange(
            start_brightness=100,
            end_brightness=200,
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        # 100 brightness change, 1000ms, 100ms min -> ideal=100, time_limited=10 -> 10
        assert change.step_count() == 10

    def test_time_limited(self) -> None:
        """Test step count is limited by time."""
        change = FadeChange(
            start_brightness=0,
            end_brightness=255,
            transition_ms=500,
            min_step_delay_ms=100,
        )
        # 255 brightness change, 500ms, 100ms min -> ideal=255, time_limited=5 -> 5
        assert change.step_count() == 5

    def test_change_limited(self) -> None:
        """Test step count is limited by change magnitude."""
        change = FadeChange(
            start_brightness=100,
            end_brightness=105,
            transition_ms=10000,
            min_step_delay_ms=100,
        )
        # 5 brightness change, 10000ms, 100ms min -> ideal=5, time_limited=100 -> 5
        assert change.step_count() == 5

    def test_hue_change(self) -> None:
        """Test step count with hue change."""
        change = FadeChange(
            start_hs=(0.0, 50.0),
            end_hs=(180.0, 50.0),
            transition_ms=18000,
            min_step_delay_ms=100,
        )
        # 180 hue change, 18000ms, 100ms min -> ideal=180, time_limited=180 -> 180
        assert change.step_count() == 180

    def test_saturation_change(self) -> None:
        """Test step count with saturation change."""
        change = FadeChange(
            start_hs=(50.0, 0.0),
            end_hs=(50.0, 50.0),
            transition_ms=10000,
            min_step_delay_ms=100,
        )
        # 50 saturation change, 10000ms, 100ms min -> ideal=50, time_limited=100 -> 50
        assert change.step_count() == 50

    def test_mireds_change(self) -> None:
        """Test step count with mireds change."""
        change = FadeChange(
            start_mireds=200,
            end_mireds=400,
            transition_ms=10000,
            min_step_delay_ms=100,
        )
        # 200 mireds change / 5 = 40 ideal, 10000ms/100ms = 100 time limited -> 40
        assert change.step_count() == 40

    def test_hue_wraparound(self) -> None:
        """Test step count handles hue wraparound correctly."""
        change = FadeChange(
            start_hs=(350.0, 50.0),
            end_hs=(20.0, 50.0),
            transition_ms=10000,
            min_step_delay_ms=100,
        )
        # Hue wraps: 350 -> 20 is only 30 degrees via 0, not 330
        # ideal = 30, time_limited = 100 -> 30
        assert change.step_count() == 30

    def test_multiple_dimensions_max_wins(self) -> None:
        """Test that largest change dimension determines step count."""
        change = FadeChange(
            start_brightness=100,
            end_brightness=150,  # 50 change
            start_hs=(0.0, 50.0),
            end_hs=(100.0, 50.0),  # 100 hue change
            transition_ms=15000,
            min_step_delay_ms=100,
        )
        # brightness ideal=50, hue ideal=100 -> max=100, time_limited=150 -> 100
        assert change.step_count() == 100

    def test_minimum_one_step(self) -> None:
        """Test at least one step is returned."""
        change = FadeChange(
            start_brightness=100,
            end_brightness=100,  # no change
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        assert change.step_count() >= 1

    def test_no_changes_returns_one(self) -> None:
        """Test no changes still returns one step."""
        change = FadeChange(transition_ms=1000, min_step_delay_ms=100)
        assert change.step_count() == 1

    def test_step_count_cached(self) -> None:
        """Test step count is cached after first calculation."""
        change = FadeChange(
            start_brightness=100,
            end_brightness=200,
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        count1 = change.step_count()
        count2 = change.step_count()
        assert count1 == count2
        assert change._step_count == count1


class TestFadeChangeDelay:
    """Test delay calculation for FadeChange."""

    def test_basic_delay(self) -> None:
        """Test basic delay calculation."""
        change = FadeChange(
            start_brightness=100,
            end_brightness=200,
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        # 10 steps, 1000ms transition -> 100ms delay
        assert change.delay_ms() == 100.0

    def test_non_integer_delay(self) -> None:
        """Test delay with non-integer result."""
        change = FadeChange(
            start_brightness=0,
            end_brightness=30,
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        # 10 steps (time limited), 1000ms -> 100ms delay
        assert change.delay_ms() == 100.0

    def test_single_step_zero_delay(self) -> None:
        """Test that single step returns zero delay."""
        change = FadeChange(
            start_brightness=100,
            end_brightness=100,
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        # 1 step -> delay is 0
        assert change.delay_ms() == 0.0


class TestFadeChangeIterator:
    """Test iterator functionality of FadeChange."""

    def test_reset(self) -> None:
        """Test reset resets iterator state."""
        change = FadeChange(
            start_brightness=100,
            end_brightness=200,
            transition_ms=500,
            min_step_delay_ms=100,
        )
        change.next_step()
        change.next_step()
        change.reset()
        assert change._current_step == 0

    def test_has_next_initial(self) -> None:
        """Test has_next returns True initially."""
        change = FadeChange(
            start_brightness=100,
            end_brightness=200,
            transition_ms=500,
            min_step_delay_ms=100,
        )
        assert change.has_next() is True

    def test_has_next_after_all_steps(self) -> None:
        """Test has_next returns False after all steps consumed."""
        change = FadeChange(
            start_brightness=100,
            end_brightness=200,
            transition_ms=200,  # 2 steps
            min_step_delay_ms=100,
        )
        change.next_step()
        change.next_step()
        assert change.has_next() is False

    def test_next_step_returns_fade_step(self) -> None:
        """Test next_step returns FadeStep instance."""
        change = FadeChange(
            start_brightness=100,
            end_brightness=200,
            transition_ms=100,  # 1 step
            min_step_delay_ms=100,
        )
        step = change.next_step()
        assert isinstance(step, FadeStep)

    def test_next_step_raises_when_exhausted(self) -> None:
        """Test next_step raises StopIteration when no more steps."""
        change = FadeChange(
            start_brightness=100,
            end_brightness=200,
            transition_ms=100,  # 1 step
            min_step_delay_ms=100,
        )
        change.next_step()
        with pytest.raises(StopIteration):
            change.next_step()

    def test_iterator_generates_correct_count(self) -> None:
        """Test iterator generates exactly step_count steps."""
        change = FadeChange(
            start_brightness=100,
            end_brightness=200,
            transition_ms=500,
            min_step_delay_ms=100,
        )
        steps = []
        while change.has_next():
            steps.append(change.next_step())
        assert len(steps) == change.step_count()


class TestFadeChangeInterpolateBrightness:
    """Test brightness interpolation."""

    def test_interpolate_middle(self) -> None:
        """Test interpolation at t=0.5."""
        change = FadeChange(
            start_brightness=100,
            end_brightness=200,
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        result = change._interpolate_brightness(0.5)
        assert result == 150

    def test_interpolate_end(self) -> None:
        """Test interpolation at t=1.0."""
        change = FadeChange(
            start_brightness=100,
            end_brightness=200,
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        result = change._interpolate_brightness(1.0)
        assert result == 200

    def test_interpolate_start(self) -> None:
        """Test interpolation at t=0.0."""
        change = FadeChange(
            start_brightness=100,
            end_brightness=200,
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        result = change._interpolate_brightness(0.0)
        assert result == 100

    def test_interpolate_none_when_no_brightness(self) -> None:
        """Test interpolation returns None when brightness not set."""
        change = FadeChange(
            start_hs=(0.0, 50.0),
            end_hs=(100.0, 50.0),
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        result = change._interpolate_brightness(0.5)
        assert result is None

    def test_interpolate_fade_down(self) -> None:
        """Test interpolation when fading down."""
        change = FadeChange(
            start_brightness=200,
            end_brightness=100,
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        result = change._interpolate_brightness(0.5)
        assert result == 150


class TestFadeChangeInterpolateHS:
    """Test HS color interpolation."""

    def test_interpolate_hs_simple(self) -> None:
        """Test simple HS interpolation."""
        change = FadeChange(
            start_hs=(0.0, 50.0),
            end_hs=(100.0, 100.0),
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        result = change._interpolate_hs(0.5)
        assert result is not None
        assert result[0] == 50.0  # hue
        assert result[1] == 75.0  # saturation

    def test_interpolate_hs_wraparound(self) -> None:
        """Test HS interpolation with hue wraparound."""
        change = FadeChange(
            start_hs=(350.0, 50.0),
            end_hs=(20.0, 50.0),
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        result = change._interpolate_hs(0.5)
        assert result is not None
        # Halfway between 350 and 20 via 0 is 5.0
        assert result[0] == 5.0
        assert result[1] == 50.0

    def test_interpolate_hs_end(self) -> None:
        """Test HS interpolation at t=1.0 hits target exactly."""
        change = FadeChange(
            start_hs=(350.0, 30.0),
            end_hs=(20.0, 80.0),
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        result = change._interpolate_hs(1.0)
        assert result == (20.0, 80.0)

    def test_interpolate_hs_none_when_no_hs(self) -> None:
        """Test HS interpolation returns None when HS not set."""
        change = FadeChange(
            start_brightness=100,
            end_brightness=200,
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        result = change._interpolate_hs(0.5)
        assert result is None


class TestFadeChangeInterpolateColorTempKelvin:
    """Test color temp interpolation (returns kelvin from internal mireds)."""

    def test_interpolate_color_temp_kelvin_simple(self) -> None:
        """Test simple color temp kelvin interpolation.

        Internal mireds: 200-400, midpoint at t=0.5 is 300 mireds = 3333K
        """
        change = FadeChange(
            start_mireds=200,  # 5000K
            end_mireds=400,  # 2500K
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        result = change._interpolate_color_temp_kelvin(0.5)
        # At t=0.5, mireds = 300, kelvin = 1_000_000/300 = 3333
        assert result == 3333

    def test_interpolate_color_temp_kelvin_end(self) -> None:
        """Test color temp kelvin interpolation at t=1.0."""
        change = FadeChange(
            start_mireds=200,  # 5000K
            end_mireds=400,  # 2500K
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        result = change._interpolate_color_temp_kelvin(1.0)
        # At t=1.0, mireds = 400, kelvin = 1_000_000/400 = 2500
        assert result == 2500

    def test_interpolate_color_temp_kelvin_none_when_not_set(self) -> None:
        """Test color temp kelvin interpolation returns None when not set."""
        change = FadeChange(
            start_brightness=100,
            end_brightness=200,
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        result = change._interpolate_color_temp_kelvin(0.5)
        assert result is None


class TestFadeChangeNextStepValues:
    """Test that next_step produces correct values."""

    def test_brightness_progression(self) -> None:
        """Test brightness progresses correctly through steps."""
        change = FadeChange(
            start_brightness=100,
            end_brightness=200,
            transition_ms=500,
            min_step_delay_ms=100,
        )
        steps = []
        while change.has_next():
            steps.append(change.next_step())

        # Should be 5 steps
        assert len(steps) == 5
        # Final step should hit target exactly
        assert steps[-1].brightness == 200
        # All should have brightness set
        for step in steps:
            assert step.brightness is not None
        # Should be monotonically increasing
        brightnesses = [s.brightness for s in steps]
        assert brightnesses == sorted(brightnesses)

    def test_brightness_fade_down(self) -> None:
        """Test brightness progression when fading down."""
        change = FadeChange(
            start_brightness=200,
            end_brightness=100,
            transition_ms=500,
            min_step_delay_ms=100,
        )
        steps = []
        while change.has_next():
            steps.append(change.next_step())

        assert steps[-1].brightness == 100
        brightnesses = [s.brightness for s in steps]
        assert brightnesses == sorted(brightnesses, reverse=True)

    def test_hs_progression(self) -> None:
        """Test HS color progresses correctly through steps."""
        change = FadeChange(
            start_hs=(0.0, 50.0),
            end_hs=(100.0, 100.0),
            transition_ms=500,
            min_step_delay_ms=100,
        )
        steps = []
        while change.has_next():
            steps.append(change.next_step())

        assert len(steps) == 5
        assert steps[-1].hs_color == (100.0, 100.0)
        for step in steps:
            assert step.hs_color is not None
            assert step.brightness is None

    def test_combined_brightness_and_hs(self) -> None:
        """Test combined brightness and HS in same change."""
        change = FadeChange(
            start_brightness=100,
            end_brightness=200,
            start_hs=(0.0, 50.0),
            end_hs=(100.0, 80.0),
            transition_ms=500,
            min_step_delay_ms=100,
        )
        steps = []
        while change.has_next():
            steps.append(change.next_step())

        for step in steps:
            assert step.brightness is not None
            assert step.hs_color is not None

        assert steps[-1].brightness == 200
        assert steps[-1].hs_color == (100.0, 80.0)

    def test_color_temp_progression(self) -> None:
        """Test color temp (kelvin) progresses correctly through steps."""
        change = FadeChange(
            start_mireds=200,  # 5000K
            end_mireds=400,  # 2500K
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        steps = []
        while change.has_next():
            steps.append(change.next_step())

        # FadeStep outputs kelvin (converted from internal mireds)
        assert steps[-1].color_temp_kelvin == 2500  # 400 mireds = 2500K
        for step in steps:
            assert step.color_temp_kelvin is not None

    def test_no_start_value_in_steps(self) -> None:
        """Test that the start value is not included in steps."""
        change = FadeChange(
            start_brightness=100,
            end_brightness=200,
            transition_ms=500,
            min_step_delay_ms=100,
        )
        step = change.next_step()
        assert step.brightness != 100  # First step is not the start value


class TestFadeChangeEasing:
    """Test easing integration in FadeChange."""

    def test_linear_easing_produces_uniform_steps(self) -> None:
        """Test linear easing produces evenly spaced brightness values."""
        from custom_components.fado.easing import linear

        change = FadeChange(
            start_brightness=0,
            end_brightness=100,
            transition_ms=500,
            min_step_delay_ms=100,
            _easing_func=linear,
        )
        steps = []
        while change.has_next():
            steps.append(change.next_step())

        # With 5 steps and linear easing: 20, 40, 60, 80, 100
        # (brightness 1 is skipped -> 2)
        brightnesses = [s.brightness for s in steps]
        assert brightnesses[-1] == 100  # Final hits target

        # Check roughly uniform spacing (within 1 due to rounding)
        diffs = [brightnesses[i + 1] - brightnesses[i] for i in range(len(brightnesses) - 1)]
        # All diffs should be roughly equal for linear
        assert all(18 <= d <= 22 for d in diffs)

    def test_ease_out_quad_faster_at_start(self) -> None:
        """Test ease_out_quad produces larger steps at the start."""
        from custom_components.fado.easing import ease_out_quad

        change = FadeChange(
            start_brightness=0,
            end_brightness=100,
            transition_ms=500,
            min_step_delay_ms=100,
            _easing_func=ease_out_quad,
        )
        steps = []
        while change.has_next():
            steps.append(change.next_step())

        brightnesses = [s.brightness for s in steps]
        assert brightnesses[-1] == 100  # Final hits target

        # First step should be larger than last step difference (ease_out = fast start)
        # First diff: steps[0].brightness - 0 (start)
        # Since brightness 1 is skipped to 2, first step is at eased position
        first_step_brightness = brightnesses[0]
        brightnesses[-1] - brightnesses[-2]

        # ease_out_quad at t=0.2 gives 0.36, so first step ~36 brightness
        # Linear would give 20. So ease_out should have larger first step.
        assert first_step_brightness > 30  # Faster at start

    def test_ease_in_quad_slower_at_start(self) -> None:
        """Test ease_in_quad produces smaller steps at the start."""
        from custom_components.fado.easing import ease_in_quad

        change = FadeChange(
            start_brightness=0,
            end_brightness=100,
            transition_ms=500,
            min_step_delay_ms=100,
            _easing_func=ease_in_quad,
        )
        steps = []
        while change.has_next():
            steps.append(change.next_step())

        brightnesses = [s.brightness for s in steps]
        assert brightnesses[-1] == 100  # Final hits target

        # First step should be smaller than linear would produce
        # ease_in_quad at t=0.2 gives 0.04, so first step ~4 brightness
        # (but brightness 1 is skipped to 2)
        first_step_brightness = brightnesses[0]
        # Linear would give 20, ease_in should give much less
        assert first_step_brightness < 10  # Slower at start

    def test_easing_applied_only_to_brightness(self) -> None:
        """Test that easing is applied to brightness but not to HS or mireds."""
        from custom_components.fado.easing import ease_out_quad

        change = FadeChange(
            start_brightness=0,
            end_brightness=100,
            start_hs=(0.0, 50.0),
            end_hs=(100.0, 100.0),
            transition_ms=500,
            min_step_delay_ms=100,
            _easing_func=ease_out_quad,
        )
        steps = []
        while change.has_next():
            steps.append(change.next_step())

        # Check brightness uses easing (larger first step)
        first_brightness = steps[0].brightness
        assert first_brightness > 30  # Eased (ease_out = fast start)

        # Check HS uses linear interpolation (not eased)
        # At t=0.2 (first step), linear HS should be:
        # hue: 0 + (100-0)*0.2 = 20
        # sat: 50 + (100-50)*0.2 = 60
        first_hs = steps[0].hs_color
        assert first_hs is not None
        assert 18 <= first_hs[0] <= 22  # Hue ~20 (linear)
        assert 58 <= first_hs[1] <= 62  # Sat ~60 (linear)

    def test_easing_final_step_hits_target(self) -> None:
        """Test that final step always hits target exactly regardless of easing."""
        from custom_components.fado.easing import (
            ease_in_cubic,
            ease_in_out_sine,
            ease_out_cubic,
        )

        for easing_func in [ease_in_cubic, ease_out_cubic, ease_in_out_sine]:
            change = FadeChange(
                start_brightness=50,
                end_brightness=200,
                transition_ms=500,
                min_step_delay_ms=100,
                _easing_func=easing_func,
            )
            steps = []
            while change.has_next():
                steps.append(change.next_step())

            assert steps[-1].brightness == 200, f"Final step should hit target with {easing_func}"

    def test_easing_mireds_uses_linear(self) -> None:
        """Test that color temperature interpolation is always linear."""
        from custom_components.fado.easing import ease_out_quad

        change = FadeChange(
            start_mireds=200,  # 5000K
            end_mireds=400,  # 2500K
            transition_ms=500,
            min_step_delay_ms=100,
            _easing_func=ease_out_quad,  # Should not affect mireds
        )
        steps = []
        while change.has_next():
            steps.append(change.next_step())

        # Check mireds uses linear interpolation
        # With 5 steps: t = 0.2, 0.4, 0.6, 0.8, 1.0
        # mireds should be: 240, 280, 320, 360, 400
        # kelvin: 4166, 3571, 3125, 2777, 2500
        kelvins = [s.color_temp_kelvin for s in steps]
        assert kelvins[-1] == 2500  # Final hits target

        # Linear spacing in mireds means roughly uniform kelvin diffs
        # (mireds are linear, kelvin is 1/mireds so not perfectly uniform)
        # But first diff should be similar to others, not affected by ease_out
        # For linear mireds: 200->240 = 40 mireds change per step
        # At step 1: mireds = 200 + 40 = 240 -> kelvin = 4166
        assert 4100 <= kelvins[0] <= 4200  # Linear, not eased

    def test_easing_skips_duplicate_steps(self) -> None:
        """Test that steps with identical values due to easing are skipped.

        With aggressive easing, consecutive steps can round to the same value.
        The iterator should skip these duplicates to avoid redundant service calls.
        Note: The last step is always emitted to ensure target is hit, so it may
        duplicate the previous step if the penultimate step already hit target.
        """
        from custom_components.fado.easing import ease_in_cubic

        # Very slow fade: 5 unit brightness change over 20 steps (50ms delay over 1000ms)
        # With ease_in_cubic (aggressive easing), early steps will round to same value
        change = FadeChange(
            start_brightness=100,
            end_brightness=105,  # Only 5 units change
            transition_ms=1000,
            min_step_delay_ms=50,  # Would give 20 steps without optimization
            _easing_func=ease_in_cubic,
        )

        steps = []
        while change.has_next():
            steps.append(change.next_step())

        brightnesses = [s.brightness for s in steps]

        # No consecutive duplicates EXCEPT for the last step (which always emits
        # to ensure target is hit, even if previous step already reached target)
        for i in range(1, len(brightnesses) - 1):
            assert brightnesses[i] != brightnesses[i - 1], (
                f"Step {i} has same brightness as step {i - 1}: {brightnesses[i]}"
            )

        # Should have fewer steps than the theoretical 20 due to duplicate skipping
        # With only 5 units of change spread over 20 steps with aggressive easing,
        # many steps will round to the same value and be skipped
        assert len(steps) < 20, f"Should skip some duplicate steps, got {len(steps)}"

        # First and last step should still be correct
        assert brightnesses[0] is not None
        assert brightnesses[-1] == 105  # Final target reached

    def test_reset_clears_last_emitted_step(self) -> None:
        """Test that reset() clears the duplicate tracking state."""
        change = FadeChange(
            start_brightness=100,
            end_brightness=110,
            transition_ms=500,
            min_step_delay_ms=100,
        )

        # Consume some steps
        change.next_step()
        change.next_step()

        # Reset should clear state
        change.reset()

        # Should start fresh (not skip based on previous run's last step)
        first_step = change.next_step()
        assert first_step.brightness is not None
        assert first_step.brightness > 100


class TestFadeChangeResolveMinBrightness:
    """Test min_brightness parameter in FadeChange.resolve()."""

    def test_brightness_pct_1_maps_to_min_brightness_when_higher(self) -> None:
        """Test brightness_pct=1 maps to min_brightness when min_brightness > normal conversion.

        Normal conversion: 1% of 255 = 2.55 -> rounds to 3.
        When min_brightness > 3, use min_brightness instead (special case).
        """
        from custom_components.fado.fade_params import FadeParams

        params = FadeParams(brightness_pct=1, transition_ms=1000)
        state = {
            "brightness": 100,
            "supported_color_modes": ["brightness"],
        }

        # With min_brightness=10, brightness_pct=1 should use min_brightness
        fade = FadeChange.resolve(params, state, min_step_delay_ms=100, min_brightness=10)

        assert fade is not None
        assert fade.end_brightness == 10  # Uses min_brightness, not 3

    def test_brightness_pct_1_normal_conversion_when_min_brightness_is_1(self) -> None:
        """Test brightness_pct=1 converts normally when min_brightness is 1.

        Normal conversion: 1% of 255 = 2.55 -> truncated to 2.
        When min_brightness=1, just use normal conversion (clamped to min 1).
        """
        from custom_components.fado.fade_params import FadeParams

        params = FadeParams(brightness_pct=1, transition_ms=1000)
        state = {
            "brightness": 100,
            "supported_color_modes": ["brightness"],
        }

        # With default min_brightness=1, brightness_pct=1 should convert normally
        fade = FadeChange.resolve(params, state, min_step_delay_ms=100, min_brightness=1)

        assert fade is not None
        assert fade.end_brightness == 2  # int(1 * 255 / 100) = 2

    def test_brightness_pct_converts_correctly_to_255_scale(self) -> None:
        """Test brightness_pct values convert correctly to 0-255 scale."""
        from custom_components.fado.fade_params import FadeParams

        # Test various percentages (using int() for truncation)
        test_cases = [
            (50, 127),  # 50% -> 127.5 -> 127
            (100, 255),  # 100% -> 255
            (25, 63),  # 25% -> 63.75 -> 63
            (10, 25),  # 10% -> 25.5 -> 25
        ]

        state = {
            "brightness": 200,
            "supported_color_modes": ["brightness"],
        }

        for pct, expected in test_cases:
            params = FadeParams(brightness_pct=pct, transition_ms=1000)
            fade = FadeChange.resolve(params, state, min_step_delay_ms=100, min_brightness=1)

            assert fade is not None, f"FadeChange should be created for {pct}%"
            assert fade.end_brightness == expected, (
                f"brightness_pct={pct} should convert to {expected}, got {fade.end_brightness}"
            )

    def test_raw_brightness_used_directly(self) -> None:
        """Test brightness (raw 1-255) values are used directly without conversion."""
        from custom_components.fado.fade_params import FadeParams

        params = FadeParams(brightness=150, transition_ms=1000)
        state = {
            "brightness": 100,
            "supported_color_modes": ["brightness"],
        }

        fade = FadeChange.resolve(params, state, min_step_delay_ms=100, min_brightness=1)

        assert fade is not None
        assert fade.end_brightness == 150  # Used directly

    def test_raw_brightness_clamped_to_min_brightness(self) -> None:
        """Test raw brightness value is clamped to min_brightness floor."""
        from custom_components.fado.fade_params import FadeParams

        params = FadeParams(brightness=5, transition_ms=1000)
        state = {
            "brightness": 100,
            "supported_color_modes": ["brightness"],
        }

        # min_brightness=10 should clamp brightness=5 up to 10
        fade = FadeChange.resolve(params, state, min_step_delay_ms=100, min_brightness=10)

        assert fade is not None
        assert fade.end_brightness == 10  # Clamped to min_brightness

    def test_brightness_pct_clamped_to_min_brightness_floor(self) -> None:
        """Test converted brightness_pct value is clamped to min_brightness floor."""
        from custom_components.fado.fade_params import FadeParams

        # 5% of 255 = 12.75 -> 12, but min_brightness=20 should clamp it
        params = FadeParams(brightness_pct=5, transition_ms=1000)
        state = {
            "brightness": 100,
            "supported_color_modes": ["brightness"],
        }

        fade = FadeChange.resolve(params, state, min_step_delay_ms=100, min_brightness=20)

        assert fade is not None
        assert fade.end_brightness == 20  # Clamped to min_brightness

    def test_start_brightness_clamped_when_fading_from_off(self) -> None:
        """Test starting brightness is clamped to min_brightness when light is off."""
        from custom_components.fado.fade_params import FadeParams

        params = FadeParams(brightness_pct=50, transition_ms=1000)
        state = {
            # Light is off - no brightness in state
            "supported_color_modes": ["brightness"],
        }

        fade = FadeChange.resolve(params, state, min_step_delay_ms=100, min_brightness=10)

        assert fade is not None
        # Start should be clamped from 0 to min_brightness
        assert fade.start_brightness == 10

    def test_start_brightness_clamped_when_current_brightness_is_low(self) -> None:
        """Test starting brightness is clamped when current brightness < min_brightness."""
        from custom_components.fado.fade_params import FadeParams

        params = FadeParams(brightness_pct=50, transition_ms=1000)
        state = {
            "brightness": 5,  # Current brightness is below min_brightness
            "supported_color_modes": ["brightness"],
        }

        fade = FadeChange.resolve(params, state, min_step_delay_ms=100, min_brightness=10)

        assert fade is not None
        # Start should be clamped from 5 to min_brightness=10
        assert fade.start_brightness == 10

    def test_from_brightness_pct_handling(self) -> None:
        """Test from_brightness_pct is converted and clamped correctly."""
        from custom_components.fado.fade_params import FadeParams

        # from_brightness_pct=2 -> 5 (int(2*255/100)=5), but min_brightness=10
        params = FadeParams(
            brightness_pct=50,
            from_brightness_pct=2,
            transition_ms=1000,
        )
        state = {
            "brightness": 100,
            "supported_color_modes": ["brightness"],
        }

        fade = FadeChange.resolve(params, state, min_step_delay_ms=100, min_brightness=10)

        assert fade is not None
        assert fade.start_brightness == 10  # Clamped from 5 to min_brightness

    def test_from_brightness_raw_handling(self) -> None:
        """Test from_brightness (raw) is used directly and clamped correctly."""
        from custom_components.fado.fade_params import FadeParams

        params = FadeParams(
            brightness_pct=50,
            from_brightness=5,  # Raw value below min_brightness
            transition_ms=1000,
        )
        state = {
            "brightness": 100,
            "supported_color_modes": ["brightness"],
        }

        fade = FadeChange.resolve(params, state, min_step_delay_ms=100, min_brightness=10)

        assert fade is not None
        assert fade.start_brightness == 10  # Clamped from 5 to min_brightness

    def test_from_brightness_pct_1_special_case(self) -> None:
        """Test from_brightness_pct=1 maps to min_brightness when min_brightness > 2."""
        from custom_components.fado.fade_params import FadeParams

        params = FadeParams(
            brightness_pct=100,
            from_brightness_pct=1,  # Special case: "dimmest possible"
            transition_ms=1000,
        )
        state = {
            "brightness": 100,
            "supported_color_modes": ["brightness"],
        }

        fade = FadeChange.resolve(params, state, min_step_delay_ms=100, min_brightness=15)

        assert fade is not None
        assert fade.start_brightness == 15  # Uses min_brightness for 1%

    def test_both_endpoints_clamped_same_value_returns_none(self) -> None:
        """Test that when both endpoints clamp to same value, nothing to fade."""
        from custom_components.fado.fade_params import FadeParams

        params = FadeParams(
            brightness=5,  # Below min_brightness, will clamp to 10
            from_brightness=3,  # Also below min_brightness, will clamp to 10
            transition_ms=1000,
        )
        state = {
            "brightness": 100,
            "supported_color_modes": ["brightness"],
        }

        fade = FadeChange.resolve(params, state, min_step_delay_ms=100, min_brightness=10)

        # Both should be clamped to min_brightness (10)
        # Since they're the same after clamping, nothing to fade
        assert fade is None  # No change when both endpoints are the same

    def test_end_brightness_zero_not_clamped(self) -> None:
        """Test that end brightness of 0 is NOT clamped (allows fade to off)."""
        from custom_components.fado.fade_params import FadeParams

        params = FadeParams(brightness_pct=0, transition_ms=1000)
        state = {
            "brightness": 100,
            "supported_color_modes": ["brightness"],
        }

        # min_brightness=10, but target is 0 (off) - should not clamp
        fade = FadeChange.resolve(params, state, min_step_delay_ms=100, min_brightness=10)

        assert fade is not None
        assert fade.end_brightness == 0  # Not clamped to min_brightness

    def test_fade_respects_min_brightness_during_interpolation(self) -> None:
        """Test that fading from min_brightness ensures all steps stay above min."""
        from custom_components.fado.fade_params import FadeParams

        params = FadeParams(brightness_pct=100, transition_ms=500)
        state = {
            # Light is off
            "supported_color_modes": ["brightness"],
        }

        fade = FadeChange.resolve(params, state, min_step_delay_ms=100, min_brightness=10)

        assert fade is not None
        assert fade.start_brightness == 10  # Clamped from 0
        assert fade.end_brightness == 255

        # All interpolated steps should be >= min_brightness
        while fade.has_next():
            step = fade.next_step()
            assert step.brightness is not None
            assert step.brightness >= 10, (
                f"Step brightness {step.brightness} is below min_brightness"
            )

    def test_default_min_brightness_is_1(self) -> None:
        """Test that default min_brightness is 1 (backward compatibility)."""
        from custom_components.fado.fade_params import FadeParams

        params = FadeParams(brightness_pct=1, transition_ms=1000)
        state = {
            "brightness": 100,
            "supported_color_modes": ["brightness"],
        }

        # Default min_brightness should be 1, so normal conversion applies
        fade = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert fade is not None
        assert fade.end_brightness == 2  # int(1 * 255 / 100) = 2

    def test_min_brightness_with_stored_brightness_auto_turn_on(self) -> None:
        """Test min_brightness is used when auto-turning on with stored brightness."""
        from custom_components.fado.fade_params import FadeParams

        # Only targeting color (no explicit brightness)
        params = FadeParams(hs_color=(180.0, 50.0), transition_ms=1000)
        state = {
            # Light is off (no brightness)
            "supported_color_modes": ["hs"],
        }

        fade = FadeChange.resolve(
            params, state, min_step_delay_ms=100, stored_brightness=0, min_brightness=10
        )

        assert fade is not None
        # Start should be clamped to min_brightness since light is off
        # Auto-turn-on should also use min_brightness as floor
        assert fade.start_brightness == 10
