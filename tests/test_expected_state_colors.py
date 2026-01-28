"""Tests for ExpectedState color matching functionality."""

from __future__ import annotations

from custom_components.fade_lights import ExpectedState
from custom_components.fade_lights.models import ExpectedValues


class TestExpectedStateColorMatching:
    """Test ExpectedState with ExpectedValues for color matching."""

    def test_match_brightness_only(self) -> None:
        """Test matching when only brightness is tracked."""
        state = ExpectedState(entity_id="light.test")
        state.add(ExpectedValues(brightness=128))

        # Exact match
        result = state.match_and_remove(ExpectedValues(brightness=128))
        assert result is not None
        assert result.brightness == 128
        assert state.is_empty

    def test_match_brightness_with_tolerance(self) -> None:
        """Test brightness within +/-3 tolerance matches."""
        state = ExpectedState(entity_id="light.test")
        state.add(ExpectedValues(brightness=128))

        # Within tolerance (+3)
        result = state.match_and_remove(ExpectedValues(brightness=131))
        assert result is not None
        assert result.brightness == 128
        assert state.is_empty

    def test_no_match_brightness_outside_tolerance(self) -> None:
        """Test brightness outside tolerance doesn't match."""
        state = ExpectedState(entity_id="light.test")
        state.add(ExpectedValues(brightness=128))

        # Outside tolerance (+4)
        result = state.match_and_remove(ExpectedValues(brightness=132))
        assert result is None
        assert not state.is_empty

    def test_match_hs_color_only(self) -> None:
        """Test matching when only HS color is tracked."""
        state = ExpectedState(entity_id="light.test")
        state.add(ExpectedValues(hs_color=(180.0, 50.0)))

        # Exact match
        result = state.match_and_remove(ExpectedValues(hs_color=(180.0, 50.0)))
        assert result is not None
        assert result.hs_color == (180.0, 50.0)
        assert state.is_empty

    def test_match_hs_with_tolerance(self) -> None:
        """Test HS within tolerance (hue +/-5, sat +/-3) matches."""
        state = ExpectedState(entity_id="light.test")
        state.add(ExpectedValues(hs_color=(180.0, 50.0)))

        # Within tolerance (hue +5, sat +3)
        result = state.match_and_remove(ExpectedValues(hs_color=(185.0, 53.0)))
        assert result is not None
        assert result.hs_color == (180.0, 50.0)
        assert state.is_empty

    def test_no_match_hue_outside_tolerance(self) -> None:
        """Test hue outside +/-5 tolerance doesn't match."""
        state = ExpectedState(entity_id="light.test")
        state.add(ExpectedValues(hs_color=(180.0, 50.0)))

        # Outside tolerance (hue +6)
        result = state.match_and_remove(ExpectedValues(hs_color=(186.0, 50.0)))
        assert result is None
        assert not state.is_empty

    def test_no_match_saturation_outside_tolerance(self) -> None:
        """Test saturation outside +/-3 tolerance doesn't match."""
        state = ExpectedState(entity_id="light.test")
        state.add(ExpectedValues(hs_color=(180.0, 50.0)))

        # Outside tolerance (sat +4)
        result = state.match_and_remove(ExpectedValues(hs_color=(180.0, 54.0)))
        assert result is None
        assert not state.is_empty

    def test_hue_wraparound_matching(self) -> None:
        """Test hue 358 matches hue 2 (both within 5 degrees of 0)."""
        state = ExpectedState(entity_id="light.test")
        state.add(ExpectedValues(hs_color=(358.0, 50.0)))

        # Wraparound: 358 and 2 are 4 degrees apart (within 5 degree tolerance)
        result = state.match_and_remove(ExpectedValues(hs_color=(2.0, 50.0)))
        assert result is not None
        assert result.hs_color == (358.0, 50.0)
        assert state.is_empty

    def test_match_mireds_only(self) -> None:
        """Test matching when only mireds is tracked."""
        state = ExpectedState(entity_id="light.test")
        state.add(ExpectedValues(color_temp_mireds=300))

        # Exact match
        result = state.match_and_remove(ExpectedValues(color_temp_mireds=300))
        assert result is not None
        assert result.color_temp_mireds == 300
        assert state.is_empty

    def test_match_mireds_with_tolerance(self) -> None:
        """Test mireds within +/-10 tolerance matches."""
        state = ExpectedState(entity_id="light.test")
        state.add(ExpectedValues(color_temp_mireds=300))

        # Within tolerance (+10)
        result = state.match_and_remove(ExpectedValues(color_temp_mireds=310))
        assert result is not None
        assert result.color_temp_mireds == 300
        assert state.is_empty

    def test_no_match_mireds_outside_tolerance(self) -> None:
        """Test mireds outside tolerance doesn't match."""
        state = ExpectedState(entity_id="light.test")
        state.add(ExpectedValues(color_temp_mireds=300))

        # Outside tolerance (+11)
        result = state.match_and_remove(ExpectedValues(color_temp_mireds=311))
        assert result is None
        assert not state.is_empty

    def test_match_brightness_and_hs(self) -> None:
        """Test both brightness and HS must match."""
        state = ExpectedState(entity_id="light.test")
        state.add(ExpectedValues(brightness=128, hs_color=(180.0, 50.0)))

        # Both match
        result = state.match_and_remove(ExpectedValues(brightness=128, hs_color=(180.0, 50.0)))
        assert result is not None
        assert result.brightness == 128
        assert result.hs_color == (180.0, 50.0)
        assert state.is_empty

    def test_no_match_when_brightness_wrong(self) -> None:
        """Test color matches but brightness doesn't = no match."""
        state = ExpectedState(entity_id="light.test")
        state.add(ExpectedValues(brightness=128, hs_color=(180.0, 50.0)))

        # Color matches, brightness doesn't
        result = state.match_and_remove(ExpectedValues(brightness=200, hs_color=(180.0, 50.0)))
        assert result is None
        assert not state.is_empty

    def test_no_match_when_color_wrong(self) -> None:
        """Test brightness matches but color doesn't = no match."""
        state = ExpectedState(entity_id="light.test")
        state.add(ExpectedValues(brightness=128, hs_color=(180.0, 50.0)))

        # Brightness matches, color doesn't
        result = state.match_and_remove(ExpectedValues(brightness=128, hs_color=(90.0, 50.0)))
        assert result is None
        assert not state.is_empty

    def test_ignores_untracked_dimensions(self) -> None:
        """Test if only tracking brightness, color in actual doesn't matter."""
        state = ExpectedState(entity_id="light.test")
        state.add(ExpectedValues(brightness=128))  # Only tracking brightness

        # Actual has color, but we only check brightness
        result = state.match_and_remove(ExpectedValues(brightness=128, hs_color=(180.0, 50.0)))
        assert result is not None
        assert result.brightness == 128
        assert state.is_empty
