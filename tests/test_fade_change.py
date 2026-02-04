"""Tests for the FadeChange dataclass."""

from __future__ import annotations

import pytest

from custom_components.fade_lights.fade_change import FadeChange, FadeStep


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
        from custom_components.fade_lights.easing import linear

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
        from custom_components.fade_lights.easing import ease_out_quad

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
        last_diff = brightnesses[-1] - brightnesses[-2]

        # ease_out_quad at t=0.2 gives 0.36, so first step ~36 brightness
        # Linear would give 20. So ease_out should have larger first step.
        assert first_step_brightness > 30  # Faster at start

    def test_ease_in_quad_slower_at_start(self) -> None:
        """Test ease_in_quad produces smaller steps at the start."""
        from custom_components.fade_lights.easing import ease_in_quad

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
        from custom_components.fade_lights.easing import ease_out_quad

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
        from custom_components.fade_lights.easing import (
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
        from custom_components.fade_lights.easing import ease_out_quad

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
