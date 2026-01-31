"""Tests for fade step generation helper functions."""

from __future__ import annotations

from custom_components.fade_lights import _interpolate_hue


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
