"""Tests for Planckian locus functions."""

from __future__ import annotations

from custom_components.fade_lights import (
    _build_hs_to_mireds_steps,
    _hs_to_mireds,
    _is_on_planckian_locus,
    _mireds_to_hs,
)
from custom_components.fade_lights.models import FadeStep


class TestIsOnPlanckianLocus:
    """Test detection of HS colors on the Planckian locus."""

    def test_pure_white_is_on_locus(self) -> None:
        """Test that pure white (0 saturation) is on the locus."""
        assert _is_on_planckian_locus((0.0, 0.0)) is True
        assert _is_on_planckian_locus((180.0, 0.0)) is True

    def test_low_saturation_warm_white_is_on_locus(self) -> None:
        """Test that low saturation warm white is on the locus."""
        assert _is_on_planckian_locus((35.0, 10.0)) is True
        assert _is_on_planckian_locus((32.0, 14.0)) is True

    def test_low_saturation_cool_white_is_on_locus(self) -> None:
        """Test that low saturation cool white is on the locus."""
        assert _is_on_planckian_locus((210.0, 5.0)) is True
        assert _is_on_planckian_locus((220.0, 8.0)) is True

    def test_high_saturation_is_not_on_locus(self) -> None:
        """Test that high saturation colors are NOT on the locus."""
        assert _is_on_planckian_locus((35.0, 50.0)) is False
        assert _is_on_planckian_locus((210.0, 50.0)) is False
        assert _is_on_planckian_locus((120.0, 80.0)) is False

    def test_saturated_colors_not_on_locus(self) -> None:
        """Test that saturated colors like red, green, blue are NOT on locus."""
        assert _is_on_planckian_locus((0.0, 100.0)) is False
        assert _is_on_planckian_locus((120.0, 100.0)) is False
        assert _is_on_planckian_locus((240.0, 100.0)) is False

    def test_threshold_boundary(self) -> None:
        """Test behavior at saturation threshold boundary."""
        assert _is_on_planckian_locus((35.0, 15.0)) is True
        assert _is_on_planckian_locus((35.0, 16.0)) is False


class TestHsToMireds:
    """Test conversion from HS to approximate mireds using lookup table."""

    def test_cool_daylight_hue(self) -> None:
        """Test cool daylight hue maps to cool mireds."""
        # Hue ~220 is cool daylight (~6500K = 154 mireds)
        mireds = _hs_to_mireds((220.0, 5.0))
        assert 140 <= mireds <= 180

    def test_warm_white_hue(self) -> None:
        """Test warm white hue maps to warm mireds."""
        # Hue ~35 is warm white (~3000K = 333 mireds)
        mireds = _hs_to_mireds((35.0, 18.0))
        assert 300 <= mireds <= 370

    def test_neutral_white_hue(self) -> None:
        """Test neutral white maps to middle mireds."""
        # Hue ~42 is neutral (~4000K = 250 mireds)
        mireds = _hs_to_mireds((42.0, 8.0))
        assert 220 <= mireds <= 290

    def test_candlelight_hue(self) -> None:
        """Test very warm hue maps to high mireds."""
        # Hue ~28 is candlelight (~2000K = 500 mireds)
        mireds = _hs_to_mireds((28.0, 45.0))
        assert 450 <= mireds <= 550

    def test_pure_white_defaults_to_neutral(self) -> None:
        """Test pure white (0 saturation) returns neutral mireds."""
        mireds = _hs_to_mireds((0.0, 0.0))
        # Should return something reasonable in the middle range
        assert 200 <= mireds <= 400

    def test_returns_int(self) -> None:
        """Test that result is an integer."""
        mireds = _hs_to_mireds((35.0, 10.0))
        assert isinstance(mireds, int)


class TestMiredsToHs:
    """Test conversion from mireds to HS using Planckian locus lookup."""

    def test_cool_daylight_mireds(self) -> None:
        """Test cool daylight mireds maps to cool hue."""
        # 154 mireds = 6500K = cool daylight
        hs = _mireds_to_hs(154)
        hue, sat = hs
        assert 200 <= hue <= 230  # Cool blue-ish hue
        assert sat < 15  # Low saturation

    def test_warm_white_mireds(self) -> None:
        """Test warm white mireds maps to warm hue."""
        # 333 mireds = 3000K = warm white
        hs = _mireds_to_hs(333)
        hue, sat = hs
        assert 30 <= hue <= 45  # Warm amber hue
        assert 10 <= sat <= 25

    def test_neutral_white_mireds(self) -> None:
        """Test neutral mireds maps to neutral hue."""
        # 286 mireds = 3500K = neutral
        hs = _mireds_to_hs(286)
        hue, sat = hs
        assert 35 <= hue <= 45
        assert 8 <= sat <= 18

    def test_candlelight_mireds(self) -> None:
        """Test candlelight mireds maps to very warm hue."""
        # 500 mireds = 2000K = candlelight
        hs = _mireds_to_hs(500)
        hue, sat = hs
        assert 25 <= hue <= 35  # Very warm amber
        assert sat >= 40  # Higher saturation

    def test_interpolation_between_points(self) -> None:
        """Test that values between lookup points are interpolated."""
        # 310 is between 303 (36, 15) and 333 (35, 18)
        hs = _mireds_to_hs(310)
        hue, sat = hs
        assert 35 <= hue <= 36
        assert 15 <= sat <= 18

    def test_extrapolation_below_range(self) -> None:
        """Test mireds below lookup range returns coolest value."""
        hs = _mireds_to_hs(100)  # Below 154
        hue, _ = hs
        assert 210 <= hue <= 230  # Should be cool

    def test_extrapolation_above_range(self) -> None:
        """Test mireds above lookup range returns warmest value."""
        hs = _mireds_to_hs(600)  # Above 500
        hue, sat = hs
        assert 25 <= hue <= 30  # Very warm
        assert sat >= 40

    def test_returns_tuple(self) -> None:
        """Test that result is a tuple of two floats."""
        hs = _mireds_to_hs(300)
        assert isinstance(hs, tuple)
        assert len(hs) == 2
        assert isinstance(hs[0], float)
        assert isinstance(hs[1], float)


class TestBuildHsToMiredsSteps:
    """Test hybrid HS→mireds step generation."""

    def test_on_locus_hs_goes_straight_to_mireds(self) -> None:
        """Test that HS already on locus generates only mireds steps."""
        # Starting with low saturation (on locus) going to mireds
        steps = _build_hs_to_mireds_steps(
            start_hs=(35.0, 10.0),  # Warm white, on locus
            end_mireds=400,
            transition_ms=1000,
            min_step_delay_ms=100,
        )

        # All steps should have mireds, no HS
        for step in steps:
            assert step.color_temp_mireds is not None
            assert step.hs_color is None

        # Last step should be target mireds
        assert steps[-1].color_temp_mireds == 400

    def test_off_locus_hs_transitions_through_locus(self) -> None:
        """Test that HS off locus first fades toward locus, then to mireds."""
        # Starting with saturated red, going to warm white mireds
        steps = _build_hs_to_mireds_steps(
            start_hs=(0.0, 80.0),  # Saturated red, NOT on locus
            end_mireds=333,  # Warm white
            transition_ms=1000,
            min_step_delay_ms=100,
        )

        # Should have some steps with HS (fading toward locus)
        hs_steps = [s for s in steps if s.hs_color is not None]
        assert len(hs_steps) > 0

        # Should end with mireds steps
        mireds_steps = [s for s in steps if s.color_temp_mireds is not None]
        assert len(mireds_steps) > 0

        # Last step should be target mireds
        assert steps[-1].color_temp_mireds == 333
        assert steps[-1].hs_color is None

    def test_hs_saturation_decreases_toward_locus(self) -> None:
        """Test that HS saturation decreases as we approach the locus."""
        steps = _build_hs_to_mireds_steps(
            start_hs=(120.0, 100.0),  # Saturated green
            end_mireds=286,  # Neutral white
            transition_ms=1000,
            min_step_delay_ms=100,
        )

        # Get the HS steps
        hs_steps = [s for s in steps if s.hs_color is not None]
        if len(hs_steps) > 1:
            saturations = [s.hs_color[1] for s in hs_steps]
            # Saturation should generally decrease
            assert saturations[-1] < saturations[0]

    def test_minimum_one_step(self) -> None:
        """Test at least one step is generated."""
        steps = _build_hs_to_mireds_steps(
            start_hs=(35.0, 5.0),  # Already near target
            end_mireds=333,
            transition_ms=100,
            min_step_delay_ms=100,
        )

        assert len(steps) >= 1

    def test_step_count_respects_timing(self) -> None:
        """Test step count is limited by transition time."""
        steps = _build_hs_to_mireds_steps(
            start_hs=(0.0, 80.0),
            end_mireds=333,
            transition_ms=300,
            min_step_delay_ms=100,
        )

        # Max 3 steps (300ms / 100ms)
        assert len(steps) <= 3

    def test_returns_list_of_fade_steps(self) -> None:
        """Test return type is list of FadeStep."""
        steps = _build_hs_to_mireds_steps(
            start_hs=(35.0, 10.0),
            end_mireds=333,
            transition_ms=1000,
            min_step_delay_ms=100,
        )

        assert isinstance(steps, list)
        for step in steps:
            assert isinstance(step, FadeStep)
