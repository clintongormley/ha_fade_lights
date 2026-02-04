"""Tests for hybrid change calculator functions."""

from __future__ import annotations

from custom_components.fade_lights import (
    _calculate_hs_to_mireds_changes,
    _calculate_mireds_to_hs_changes,
    _hs_to_mireds,
    _mireds_to_hs,
)
from custom_components.fade_lights.models import FadeChange


class TestCalculateHsToMiredsChanges:
    """Test HS -> mireds hybrid transition calculator."""

    def test_returns_list_of_fade_changes(self) -> None:
        """Test that the function returns a list of FadeChange objects."""
        changes = _calculate_hs_to_mireds_changes(
            start_brightness=100,
            end_brightness=200,
            start_hs=(0.0, 80.0),  # Off locus
            end_mireds=333,
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        assert isinstance(changes, list)
        for change in changes:
            assert isinstance(change, FadeChange)

    def test_on_locus_returns_single_mireds_change(self) -> None:
        """Test that HS on locus returns single mireds-only change."""
        changes = _calculate_hs_to_mireds_changes(
            start_brightness=100,
            end_brightness=200,
            start_hs=(35.0, 10.0),  # On locus (low saturation)
            end_mireds=400,
            transition_ms=1000,
            min_step_delay_ms=100,
        )

        assert len(changes) == 1
        change = changes[0]

        # Should have mireds transition
        assert change.start_mireds is not None
        assert change.end_mireds == 400

        # Should NOT have HS transition
        assert change.start_hs is None
        assert change.end_hs is None

        # Should have brightness transition
        assert change.start_brightness == 100
        assert change.end_brightness == 200

    def test_off_locus_returns_two_phases(self) -> None:
        """Test that HS off locus returns two-phase transition."""
        changes = _calculate_hs_to_mireds_changes(
            start_brightness=100,
            end_brightness=200,
            start_hs=(0.0, 80.0),  # Off locus (saturated red)
            end_mireds=333,
            transition_ms=1000,
            min_step_delay_ms=100,
        )

        assert len(changes) == 2

    def test_phase1_is_hs_transition(self) -> None:
        """Test that phase 1 fades HS toward locus."""
        changes = _calculate_hs_to_mireds_changes(
            start_brightness=100,
            end_brightness=200,
            start_hs=(120.0, 100.0),  # Saturated green
            end_mireds=286,
            transition_ms=1000,
            min_step_delay_ms=100,
        )

        phase1 = changes[0]

        # Phase 1 should have HS transition
        assert phase1.start_hs == (120.0, 100.0)
        assert phase1.end_hs is not None
        # End HS should be on/near the locus (low saturation)
        assert phase1.end_hs[1] < 20  # Low saturation

        # Phase 1 should NOT have mireds
        assert phase1.start_mireds is None
        assert phase1.end_mireds is None

    def test_phase2_is_mireds_transition(self) -> None:
        """Test that phase 2 fades mireds to target."""
        changes = _calculate_hs_to_mireds_changes(
            start_brightness=100,
            end_brightness=200,
            start_hs=(120.0, 100.0),  # Saturated green
            end_mireds=333,
            transition_ms=1000,
            min_step_delay_ms=100,
        )

        phase2 = changes[1]

        # Phase 2 should have mireds transition
        assert phase2.start_mireds is not None
        assert phase2.end_mireds == 333

        # Phase 2 should NOT have HS
        assert phase2.start_hs is None
        assert phase2.end_hs is None

    def test_timing_split_70_30(self) -> None:
        """Test that timing is split 70/30 between phases."""
        changes = _calculate_hs_to_mireds_changes(
            start_brightness=100,
            end_brightness=200,
            start_hs=(0.0, 80.0),
            end_mireds=333,
            transition_ms=1000,
            min_step_delay_ms=100,
        )

        phase1, phase2 = changes[0], changes[1]

        assert phase1.transition_ms == 700
        assert phase2.transition_ms == 300

    def test_brightness_split_proportionally(self) -> None:
        """Test that brightness is split proportionally to time."""
        changes = _calculate_hs_to_mireds_changes(
            start_brightness=100,
            end_brightness=200,
            start_hs=(0.0, 80.0),
            end_mireds=333,
            transition_ms=1000,
            min_step_delay_ms=100,
        )

        phase1, phase2 = changes[0], changes[1]

        # Phase 1: 100 -> 170 (70% of 100-point change = 70)
        assert phase1.start_brightness == 100
        assert phase1.end_brightness == 170

        # Phase 2: 170 -> 200 (remaining 30)
        assert phase2.start_brightness == 170
        assert phase2.end_brightness == 200

    def test_no_brightness_changes(self) -> None:
        """Test handling when no brightness change is specified."""
        changes = _calculate_hs_to_mireds_changes(
            start_brightness=None,
            end_brightness=None,
            start_hs=(0.0, 80.0),
            end_mireds=333,
            transition_ms=1000,
            min_step_delay_ms=100,
        )

        phase1, phase2 = changes[0], changes[1]

        assert phase1.start_brightness is None
        assert phase1.end_brightness is None
        assert phase2.start_brightness is None
        assert phase2.end_brightness is None

    def test_min_step_delay_passed_through(self) -> None:
        """Test that min_step_delay_ms is passed to FadeChange objects."""
        changes = _calculate_hs_to_mireds_changes(
            start_brightness=100,
            end_brightness=200,
            start_hs=(0.0, 80.0),
            end_mireds=333,
            transition_ms=1000,
            min_step_delay_ms=50,
        )

        for change in changes:
            assert change.min_step_delay_ms == 50


class TestCalculateMiredsToHsChanges:
    """Test mireds -> HS hybrid transition calculator."""

    def test_returns_list_of_fade_changes(self) -> None:
        """Test that the function returns a list of FadeChange objects."""
        changes = _calculate_mireds_to_hs_changes(
            start_brightness=100,
            end_brightness=200,
            start_mireds=333,
            end_hs=(120.0, 80.0),  # Green, off locus
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        assert isinstance(changes, list)
        for change in changes:
            assert isinstance(change, FadeChange)

    def test_close_mireds_returns_single_hs_change(self) -> None:
        """Test that close mireds returns single HS-only change."""
        # Get the target locus mireds for the end HS
        end_hs = (35.0, 18.0)  # On locus warm white
        target_locus_mireds = _hs_to_mireds(end_hs)

        changes = _calculate_mireds_to_hs_changes(
            start_brightness=100,
            end_brightness=200,
            start_mireds=target_locus_mireds,  # Same as target
            end_hs=end_hs,
            transition_ms=1000,
            min_step_delay_ms=100,
        )

        assert len(changes) == 1
        change = changes[0]

        # Should have HS transition
        assert change.start_hs is not None
        assert change.end_hs == end_hs

        # Should NOT have mireds transition
        assert change.start_mireds is None
        assert change.end_mireds is None

    def test_far_mireds_returns_two_phases(self) -> None:
        """Test that distant mireds returns two-phase transition."""
        changes = _calculate_mireds_to_hs_changes(
            start_brightness=100,
            end_brightness=200,
            start_mireds=154,  # Cool daylight (far from warm)
            end_hs=(35.0, 50.0),  # Warm saturated
            transition_ms=1000,
            min_step_delay_ms=100,
        )

        assert len(changes) == 2

    def test_phase1_is_mireds_transition(self) -> None:
        """Test that phase 1 fades mireds along locus."""
        changes = _calculate_mireds_to_hs_changes(
            start_brightness=100,
            end_brightness=200,
            start_mireds=154,  # Cool daylight
            end_hs=(35.0, 50.0),  # Warm saturated
            transition_ms=1000,
            min_step_delay_ms=100,
        )

        phase1 = changes[0]

        # Phase 1 should have mireds transition
        assert phase1.start_mireds == 154
        assert phase1.end_mireds is not None

        # Phase 1 should NOT have HS
        assert phase1.start_hs is None
        assert phase1.end_hs is None

    def test_phase2_is_hs_transition(self) -> None:
        """Test that phase 2 fades HS to target."""
        changes = _calculate_mireds_to_hs_changes(
            start_brightness=100,
            end_brightness=200,
            start_mireds=154,
            end_hs=(120.0, 80.0),  # Green
            transition_ms=1000,
            min_step_delay_ms=100,
        )

        phase2 = changes[1]

        # Phase 2 should have HS transition
        assert phase2.start_hs is not None
        assert phase2.end_hs == (120.0, 80.0)

        # Phase 2 should NOT have mireds
        assert phase2.start_mireds is None
        assert phase2.end_mireds is None

    def test_timing_split_30_70(self) -> None:
        """Test that timing is split 30/70 between phases."""
        changes = _calculate_mireds_to_hs_changes(
            start_brightness=100,
            end_brightness=200,
            start_mireds=154,
            end_hs=(120.0, 80.0),
            transition_ms=1000,
            min_step_delay_ms=100,
        )

        phase1, phase2 = changes[0], changes[1]

        assert phase1.transition_ms == 300
        assert phase2.transition_ms == 700

    def test_brightness_split_proportionally(self) -> None:
        """Test that brightness is split proportionally to time."""
        changes = _calculate_mireds_to_hs_changes(
            start_brightness=100,
            end_brightness=200,
            start_mireds=154,
            end_hs=(120.0, 80.0),
            transition_ms=1000,
            min_step_delay_ms=100,
        )

        phase1, phase2 = changes[0], changes[1]

        # Phase 1: 100 -> 130 (30% of 100-point change = 30)
        assert phase1.start_brightness == 100
        assert phase1.end_brightness == 130

        # Phase 2: 130 -> 200 (remaining 70)
        assert phase2.start_brightness == 130
        assert phase2.end_brightness == 200

    def test_no_brightness_changes(self) -> None:
        """Test handling when no brightness change is specified."""
        changes = _calculate_mireds_to_hs_changes(
            start_brightness=None,
            end_brightness=None,
            start_mireds=154,
            end_hs=(120.0, 80.0),
            transition_ms=1000,
            min_step_delay_ms=100,
        )

        phase1, phase2 = changes[0], changes[1]

        assert phase1.start_brightness is None
        assert phase1.end_brightness is None
        assert phase2.start_brightness is None
        assert phase2.end_brightness is None

    def test_min_step_delay_passed_through(self) -> None:
        """Test that min_step_delay_ms is passed to FadeChange objects."""
        changes = _calculate_mireds_to_hs_changes(
            start_brightness=100,
            end_brightness=200,
            start_mireds=154,
            end_hs=(120.0, 80.0),
            transition_ms=1000,
            min_step_delay_ms=50,
        )

        for change in changes:
            assert change.min_step_delay_ms == 50

    def test_phase2_starts_from_locus_hs(self) -> None:
        """Test that phase 2 starts from HS on the locus."""
        changes = _calculate_mireds_to_hs_changes(
            start_brightness=100,
            end_brightness=200,
            start_mireds=333,  # Warm white
            end_hs=(120.0, 80.0),  # Green
            transition_ms=1000,
            min_step_delay_ms=100,
        )

        phase2 = changes[1]

        # Phase 2's start_hs should be on the locus (derived from mireds)
        assert phase2.start_hs is not None
        # It should have low saturation (on locus)
        assert phase2.start_hs[1] < 50  # Reasonable for locus colors


class TestHybridCalculatorEdgeCases:
    """Test edge cases for both hybrid calculators."""

    def test_hs_to_mireds_threshold_boundary(self) -> None:
        """Test behavior at saturation threshold boundary."""
        # Just below threshold (on locus)
        changes_on = _calculate_hs_to_mireds_changes(
            start_brightness=100,
            end_brightness=200,
            start_hs=(35.0, 15.0),  # At threshold
            end_mireds=333,
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        assert len(changes_on) == 1  # Single phase

        # Just above threshold (off locus)
        changes_off = _calculate_hs_to_mireds_changes(
            start_brightness=100,
            end_brightness=200,
            start_hs=(35.0, 16.0),  # Above threshold
            end_mireds=333,
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        assert len(changes_off) == 2  # Two phases

    def test_mireds_to_hs_near_target_threshold(self) -> None:
        """Test behavior when mireds is close to target."""
        end_hs = (35.0, 50.0)
        target_mireds = _hs_to_mireds(end_hs)

        # Just within threshold (9 mireds away)
        changes_close = _calculate_mireds_to_hs_changes(
            start_brightness=100,
            end_brightness=200,
            start_mireds=target_mireds + 9,
            end_hs=end_hs,
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        assert len(changes_close) == 1  # Single phase

        # Just outside threshold (10+ mireds away)
        changes_far = _calculate_mireds_to_hs_changes(
            start_brightness=100,
            end_brightness=200,
            start_mireds=target_mireds + 50,
            end_hs=end_hs,
            transition_ms=1000,
            min_step_delay_ms=100,
        )
        assert len(changes_far) == 2  # Two phases

    def test_hs_to_mireds_phases_connect(self) -> None:
        """Test that phases connect properly (phase1 end matches phase2 start)."""
        changes = _calculate_hs_to_mireds_changes(
            start_brightness=100,
            end_brightness=200,
            start_hs=(0.0, 80.0),
            end_mireds=333,
            transition_ms=1000,
            min_step_delay_ms=100,
        )

        phase1, phase2 = changes[0], changes[1]

        # Brightness should connect
        assert phase1.end_brightness == phase2.start_brightness

        # The HS endpoint should correspond to the mireds startpoint
        # (both representing the same point on the locus)
        if phase1.end_hs is not None:
            expected_mireds = _hs_to_mireds(phase1.end_hs)
            assert phase2.start_mireds == expected_mireds

    def test_mireds_to_hs_phases_connect(self) -> None:
        """Test that phases connect properly (phase1 end matches phase2 start)."""
        changes = _calculate_mireds_to_hs_changes(
            start_brightness=100,
            end_brightness=200,
            start_mireds=154,
            end_hs=(120.0, 80.0),
            transition_ms=1000,
            min_step_delay_ms=100,
        )

        phase1, phase2 = changes[0], changes[1]

        # Brightness should connect
        assert phase1.end_brightness == phase2.start_brightness

        # The mireds endpoint should correspond to the HS startpoint
        # (both representing the same point on the locus)
        if phase1.end_mireds is not None:
            expected_hs = _mireds_to_hs(phase1.end_mireds)
            assert phase2.start_hs == expected_hs

    def test_fade_changes_can_generate_steps(self) -> None:
        """Test that returned FadeChange objects can generate steps."""
        changes = _calculate_hs_to_mireds_changes(
            start_brightness=100,
            end_brightness=200,
            start_hs=(0.0, 80.0),
            end_mireds=333,
            transition_ms=1000,
            min_step_delay_ms=100,
        )

        total_steps = 0
        for change in changes:
            while change.has_next():
                step = change.next_step()
                assert step is not None
                total_steps += 1

        assert total_steps > 0
