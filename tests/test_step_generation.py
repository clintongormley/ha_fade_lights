"""Tests for fade step generation."""

from __future__ import annotations

from custom_components.fade_lights import _build_fade_steps, _interpolate_hue


class TestInterpolateHue:
    """Test circular short-path hue interpolation."""

    def test_simple_forward(self) -> None:
        """Test simple forward interpolation (0 -> 100)."""
        result = _interpolate_hue(0, 100, 0.5)
        assert result == 50

    def test_simple_backward(self) -> None:
        """Test simple backward interpolation (100 -> 0)."""
        result = _interpolate_hue(100, 0, 0.5)
        assert result == 50

    def test_short_path_through_zero(self) -> None:
        """Test short path crosses 0/360 boundary (350 -> 20)."""
        result = _interpolate_hue(350, 20, 0.5)
        assert result == 5

    def test_short_path_through_zero_reverse(self) -> None:
        """Test short path crosses 0/360 boundary reverse (20 -> 350)."""
        result = _interpolate_hue(20, 350, 0.5)
        assert result == 5

    def test_long_arc_avoided(self) -> None:
        """Test that long arc is avoided (180 -> 190 should NOT go through 0)."""
        result = _interpolate_hue(180, 190, 0.5)
        assert result == 185

    def test_exactly_180_degrees_apart(self) -> None:
        """Test when start and end are exactly 180 degrees apart."""
        result = _interpolate_hue(0, 180, 0.5)
        assert result == 90

    def test_at_boundaries(self) -> None:
        """Test interpolation at t=0 and t=1."""
        assert _interpolate_hue(100, 200, 0.0) == 100
        assert _interpolate_hue(100, 200, 1.0) == 200

    def test_wrap_result_normalized(self) -> None:
        """Test result is always in 0-360 range."""
        result = _interpolate_hue(350, 20, 0.9)
        assert 0 <= result < 360


class TestBuildFadeSteps:
    """Test fade step generation."""

    def test_brightness_only_fade(self) -> None:
        """Test generating steps for brightness-only fade."""
        steps = _build_fade_steps(
            start_brightness=100,
            end_brightness=200,
            start_hs=None,
            end_hs=None,
            start_mireds=None,
            end_mireds=None,
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        assert len(steps) == 10
        assert steps[0].brightness is not None
        assert steps[0].hs_color is None
        assert steps[-1].brightness == 200
        brightnesses = [s.brightness for s in steps]
        assert brightnesses == sorted(brightnesses)

    def test_brightness_fade_down(self) -> None:
        """Test generating steps for brightness fade down."""
        steps = _build_fade_steps(
            start_brightness=200,
            end_brightness=50,
            start_hs=None,
            end_hs=None,
            start_mireds=None,
            end_mireds=None,
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        assert steps[-1].brightness == 50
        brightnesses = [s.brightness for s in steps]
        assert brightnesses == sorted(brightnesses, reverse=True)

    def test_hs_color_fade(self) -> None:
        """Test generating steps for HS color fade."""
        steps = _build_fade_steps(
            start_brightness=None,
            end_brightness=None,
            start_hs=(100.0, 50.0),
            end_hs=(200.0, 80.0),
            start_mireds=None,
            end_mireds=None,
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        assert len(steps) == 10
        assert steps[0].hs_color is not None
        assert steps[-1].hs_color == (200.0, 80.0)
        assert steps[0].brightness is None

    def test_hs_color_short_path(self) -> None:
        """Test HS fade uses short path for hue."""
        steps = _build_fade_steps(
            start_brightness=None,
            end_brightness=None,
            start_hs=(350.0, 50.0),
            end_hs=(20.0, 50.0),
            start_mireds=None,
            end_mireds=None,
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        mid_step = steps[len(steps) // 2]
        assert mid_step.hs_color is not None
        mid_hue = mid_step.hs_color[0]
        assert mid_hue < 30 or mid_hue > 340

    def test_color_temp_fade(self) -> None:
        """Test generating steps for color temperature fade."""
        steps = _build_fade_steps(
            start_brightness=None,
            end_brightness=None,
            start_hs=None,
            end_hs=None,
            start_mireds=250,
            end_mireds=400,
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        assert len(steps) == 10
        assert steps[0].color_temp_mireds is not None
        assert steps[-1].color_temp_mireds == 400

    def test_combined_brightness_and_hs(self) -> None:
        """Test fading both brightness and HS together."""
        steps = _build_fade_steps(
            start_brightness=100,
            end_brightness=200,
            start_hs=(0.0, 50.0),
            end_hs=(100.0, 80.0),
            start_mireds=None,
            end_mireds=None,
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        assert steps[0].brightness is not None
        assert steps[0].hs_color is not None
        assert steps[-1].brightness == 200
        assert steps[-1].hs_color == (100.0, 80.0)

    def test_step_count_limited_by_time(self) -> None:
        """Test step count is limited by transition_ms / min_step_delay_ms."""
        steps = _build_fade_steps(
            start_brightness=0,
            end_brightness=255,
            start_hs=None,
            end_hs=None,
            start_mireds=None,
            end_mireds=None,
            transition_ms=500,
            min_step_delay_ms=100,
        )
        assert len(steps) <= 5

    def test_step_count_limited_by_change(self) -> None:
        """Test step count limited by actual change magnitude."""
        steps = _build_fade_steps(
            start_brightness=100,
            end_brightness=105,
            start_hs=None,
            end_hs=None,
            start_mireds=None,
            end_mireds=None,
            transition_ms=10000,
            min_step_delay_ms=100,
        )
        assert len(steps) <= 10

    def test_minimum_one_step(self) -> None:
        """Test at least one step is generated."""
        steps = _build_fade_steps(
            start_brightness=100,
            end_brightness=100,
            start_hs=None,
            end_hs=None,
            start_mireds=None,
            end_mireds=None,
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        assert len(steps) >= 1
        assert steps[-1].brightness == 100

    def test_no_start_equals_end_in_steps(self) -> None:
        """Test that start value is not included in steps."""
        steps = _build_fade_steps(
            start_brightness=100,
            end_brightness=200,
            start_hs=None,
            end_hs=None,
            start_mireds=None,
            end_mireds=None,
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        assert steps[0].brightness != 100
