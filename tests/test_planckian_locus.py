"""Tests for Planckian locus functions."""

from __future__ import annotations

from custom_components.fade_lights import _is_on_planckian_locus


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