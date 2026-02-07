"""Tests for ExpectedState color matching functionality."""

from __future__ import annotations

from custom_components.fado import ExpectedState
from custom_components.fado.expected_state import ExpectedValues


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

    def test_match_kelvin_only(self) -> None:
        """Test matching when only kelvin is tracked."""
        state = ExpectedState(entity_id="light.test")
        state.add(ExpectedValues(color_temp_kelvin=3333))

        # Exact match
        result = state.match_and_remove(ExpectedValues(color_temp_kelvin=3333))
        assert result is not None
        assert result.color_temp_kelvin == 3333
        assert state.is_empty

    def test_match_kelvin_with_tolerance(self) -> None:
        """Test kelvin within +/-100 tolerance matches."""
        state = ExpectedState(entity_id="light.test")
        state.add(ExpectedValues(color_temp_kelvin=3333))

        # Within tolerance (+100)
        result = state.match_and_remove(ExpectedValues(color_temp_kelvin=3433))
        assert result is not None
        assert result.color_temp_kelvin == 3333
        assert state.is_empty

    def test_no_match_kelvin_outside_tolerance(self) -> None:
        """Test kelvin outside tolerance doesn't match."""
        state = ExpectedState(entity_id="light.test")
        state.add(ExpectedValues(color_temp_kelvin=3333))

        # Outside tolerance (+101)
        result = state.match_and_remove(ExpectedValues(color_temp_kelvin=3434))
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

    def test_brightness_range_match_intermediate_value(self) -> None:
        """Test range matching accepts intermediate brightness values."""
        expected_state = ExpectedState(entity_id="light.test")

        # Transition from 10 -> 100
        expected = ExpectedValues(brightness=100, from_brightness=10)
        expected_state.add(expected)

        # Intermediate value during transition
        actual = ExpectedValues(brightness=50)

        matched = expected_state.match_and_remove(actual)
        assert matched == expected
        # Should NOT remove (range match, waiting for final value)
        assert len(expected_state.values) == 1

    def test_brightness_range_match_final_value(self) -> None:
        """Test range matching removes on final target value."""
        expected_state = ExpectedState(entity_id="light.test")

        # Transition from 10 -> 100
        expected = ExpectedValues(brightness=100, from_brightness=10)
        expected_state.add(expected)

        # Final value (within tolerance of target)
        actual = ExpectedValues(brightness=98)

        matched = expected_state.match_and_remove(actual)
        assert matched == expected
        # Should remove (exact match)
        assert len(expected_state.values) == 0

    def test_brightness_range_match_outside_range(self) -> None:
        """Test range matching rejects values outside range."""
        expected_state = ExpectedState(entity_id="light.test")

        # Transition from 10 -> 100
        expected = ExpectedValues(brightness=100, from_brightness=10)
        expected_state.add(expected)

        # Value outside range
        actual = ExpectedValues(brightness=150)

        matched = expected_state.match_and_remove(actual)
        assert matched is None
        assert len(expected_state.values) == 1

    def test_brightness_point_match_unchanged(self) -> None:
        """Test point matching behavior unchanged when no from_brightness."""
        expected_state = ExpectedState(entity_id="light.test")

        # Point-based (no from_brightness)
        expected = ExpectedValues(brightness=100)
        expected_state.add(expected)

        # Within tolerance
        actual = ExpectedValues(brightness=98)

        matched = expected_state.match_and_remove(actual)
        assert matched == expected
        # Should remove (exact match)
        assert len(expected_state.values) == 0

    def test_hs_range_match_no_wraparound(self) -> None:
        """Test HS range matching without hue wraparound."""
        expected_state = ExpectedState(entity_id="light.test")

        # Transition from (100, 50) -> (150, 80)
        expected = ExpectedValues(
            hs_color=(150.0, 80.0),
            from_hs_color=(100.0, 50.0)
        )
        expected_state.add(expected)

        # Intermediate value
        actual = ExpectedValues(hs_color=(125.0, 65.0))

        matched = expected_state.match_and_remove(actual)
        assert matched == expected
        assert len(expected_state.values) == 1  # Range match, not removed

    def test_hs_range_match_with_wraparound(self) -> None:
        """Test HS range matching with hue wraparound (350->10)."""
        expected_state = ExpectedState(entity_id="light.test")

        # Transition from (350, 50) -> (10, 50)
        expected = ExpectedValues(
            hs_color=(10.0, 50.0),
            from_hs_color=(350.0, 50.0)
        )
        expected_state.add(expected)

        # Intermediate values in wraparound range (not near target)
        for test_hue in [355.0, 0.0]:
            actual = ExpectedValues(hs_color=(test_hue, 50.0))
            matched = expected_state.match_and_remove(actual)
            assert matched == expected
            assert len(expected_state.values) == 1  # Range match, not removed

        # Value close to target (within tolerance) - should be exact match
        actual = ExpectedValues(hs_color=(8.0, 50.0))
        matched = expected_state.match_and_remove(actual)
        assert matched == expected
        assert len(expected_state.values) == 0  # Exact match, removed

    def test_hs_range_match_wraparound_rejects_gap(self) -> None:
        """Test HS range matching rejects values in the wraparound gap."""
        expected_state = ExpectedState(entity_id="light.test")

        # Transition from (350, 50) -> (10, 50)
        expected = ExpectedValues(
            hs_color=(10.0, 50.0),
            from_hs_color=(350.0, 50.0)
        )
        expected_state.add(expected)

        # Value in the gap (should be rejected)
        actual = ExpectedValues(hs_color=(180.0, 50.0))

        matched = expected_state.match_and_remove(actual)
        assert matched is None

    def test_hs_exact_match_removes(self) -> None:
        """Test HS exact match removes from queue."""
        expected_state = ExpectedState(entity_id="light.test")

        # Transition with range
        expected = ExpectedValues(
            hs_color=(150.0, 80.0),
            from_hs_color=(100.0, 50.0)
        )
        expected_state.add(expected)

        # Target value (within tolerance)
        actual = ExpectedValues(hs_color=(148.0, 79.0))

        matched = expected_state.match_and_remove(actual)
        assert matched == expected
        assert len(expected_state.values) == 0  # Exact match, removed

    def test_kelvin_range_match_intermediate_value(self) -> None:
        """Test kelvin range matching accepts intermediate values."""
        expected_state = ExpectedState(entity_id="light.test")

        # Transition from 2700K -> 6500K
        expected = ExpectedValues(
            color_temp_kelvin=6500,
            from_color_temp_kelvin=2700
        )
        expected_state.add(expected)

        # Intermediate value
        actual = ExpectedValues(color_temp_kelvin=4000)

        matched = expected_state.match_and_remove(actual)
        assert matched == expected
        assert len(expected_state.values) == 1  # Range match

    def test_kelvin_range_match_final_value(self) -> None:
        """Test kelvin range matching removes on target value."""
        expected_state = ExpectedState(entity_id="light.test")

        # Transition from 2700K -> 6500K
        expected = ExpectedValues(
            color_temp_kelvin=6500,
            from_color_temp_kelvin=2700
        )
        expected_state.add(expected)

        # Target value (within tolerance)
        actual = ExpectedValues(color_temp_kelvin=6450)

        matched = expected_state.match_and_remove(actual)
        assert matched == expected
        assert len(expected_state.values) == 0  # Exact match, removed

    def test_kelvin_range_match_outside_range(self) -> None:
        """Test kelvin range matching rejects out of range values."""
        expected_state = ExpectedState(entity_id="light.test")

        # Transition from 2700K -> 6500K
        expected = ExpectedValues(
            color_temp_kelvin=6500,
            from_color_temp_kelvin=2700
        )
        expected_state.add(expected)

        # Out of range
        actual = ExpectedValues(color_temp_kelvin=7000)

        matched = expected_state.match_and_remove(actual)
        assert matched is None
