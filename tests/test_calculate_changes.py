"""Tests for the _calculate_changes dispatcher function."""

from __future__ import annotations

from homeassistant.components.light import ATTR_BRIGHTNESS
from homeassistant.components.light import ATTR_HS_COLOR as HA_ATTR_HS_COLOR

from custom_components.fade_lights import _calculate_changes
from custom_components.fade_lights.models import FadeChange, FadeParams


class TestCalculateChangesBasicStructure:
    """Test basic return type and structure of _calculate_changes."""

    def test_returns_list_of_fade_changes(self) -> None:
        """Test that the function returns a list of FadeChange objects."""
        params = FadeParams(brightness_pct=50, transition_ms=1000)
        state = {ATTR_BRIGHTNESS: 100}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        assert isinstance(changes, list)
        for change in changes:
            assert isinstance(change, FadeChange)

    def test_returns_at_least_one_change(self) -> None:
        """Test that at least one FadeChange is always returned."""
        params = FadeParams(brightness_pct=50, transition_ms=1000)
        state = {ATTR_BRIGHTNESS: 100}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        assert len(changes) >= 1


class TestCalculateChangesSimpleBrightnessFade:
    """Test simple brightness-only fade scenarios."""

    def test_brightness_only_fade_returns_single_change(self) -> None:
        """Test that brightness-only fade returns single FadeChange."""
        params = FadeParams(brightness_pct=50, transition_ms=1000)
        state = {ATTR_BRIGHTNESS: 100}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        assert len(changes) == 1

    def test_brightness_fade_values(self) -> None:
        """Test that brightness values are correctly resolved."""
        params = FadeParams(brightness_pct=75, transition_ms=1000)
        state = {ATTR_BRIGHTNESS: 100}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        change = changes[0]
        assert change.start_brightness == 100
        # 75% of 255 = 191
        assert change.end_brightness == 191

    def test_brightness_fade_from_override(self) -> None:
        """Test that from_brightness_pct overrides state brightness."""
        params = FadeParams(
            brightness_pct=100,
            from_brightness_pct=25,
            transition_ms=1000,
        )
        state = {ATTR_BRIGHTNESS: 200}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        change = changes[0]
        # 25% of 255 = 63
        assert change.start_brightness == 63
        assert change.end_brightness == 255

    def test_brightness_fade_no_color_attributes(self) -> None:
        """Test that brightness-only fade has no color attributes."""
        params = FadeParams(brightness_pct=50, transition_ms=1000)
        state = {ATTR_BRIGHTNESS: 100}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        change = changes[0]
        assert change.start_hs is None
        assert change.end_hs is None
        assert change.start_mireds is None
        assert change.end_mireds is None


class TestCalculateChangesSimpleHsFade:
    """Test simple HS color fade scenarios."""

    def test_hs_color_fade_returns_single_change(self) -> None:
        """Test that HS color fade without mireds target returns single change."""
        params = FadeParams(
            hs_color=(120.0, 80.0),
            transition_ms=1000,
        )
        state = {HA_ATTR_HS_COLOR: (60.0, 50.0)}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        assert len(changes) == 1

    def test_hs_color_fade_values(self) -> None:
        """Test that HS color values are correctly resolved."""
        params = FadeParams(
            hs_color=(120.0, 80.0),
            transition_ms=1000,
        )
        state = {HA_ATTR_HS_COLOR: (60.0, 50.0)}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        change = changes[0]
        assert change.start_hs == (60.0, 50.0)
        assert change.end_hs == (120.0, 80.0)

    def test_hs_color_fade_from_override(self) -> None:
        """Test that from_hs_color overrides state HS color."""
        params = FadeParams(
            hs_color=(240.0, 100.0),
            from_hs_color=(0.0, 100.0),
            transition_ms=1000,
        )
        state = {HA_ATTR_HS_COLOR: (180.0, 50.0)}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        change = changes[0]
        assert change.start_hs == (0.0, 100.0)
        assert change.end_hs == (240.0, 100.0)

    def test_hs_color_only_no_mireds(self) -> None:
        """Test that HS-only fade has no mireds attributes."""
        params = FadeParams(
            hs_color=(120.0, 80.0),
            transition_ms=1000,
        )
        state = {HA_ATTR_HS_COLOR: (60.0, 50.0)}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        change = changes[0]
        assert change.start_mireds is None
        assert change.end_mireds is None


class TestCalculateChangesSimpleMiredsFade:
    """Test simple mireds color temperature fade scenarios."""

    def test_mireds_fade_returns_single_change(self) -> None:
        """Test that mireds-only fade returns single FadeChange."""
        params = FadeParams(
            color_temp_mireds=333,
            transition_ms=1000,
        )
        state = {"color_temp": 250}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        assert len(changes) == 1

    def test_mireds_fade_values(self) -> None:
        """Test that mireds values are correctly resolved."""
        params = FadeParams(
            color_temp_mireds=400,
            transition_ms=1000,
        )
        state = {"color_temp": 200}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        change = changes[0]
        assert change.start_mireds == 200
        assert change.end_mireds == 400

    def test_mireds_fade_from_override(self) -> None:
        """Test that from_color_temp_mireds overrides state mireds."""
        params = FadeParams(
            color_temp_mireds=500,
            from_color_temp_mireds=154,
            transition_ms=1000,
        )
        state = {"color_temp": 333}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        change = changes[0]
        assert change.start_mireds == 154
        assert change.end_mireds == 500

    def test_mireds_only_no_hs(self) -> None:
        """Test that mireds-only fade has no HS attributes."""
        params = FadeParams(
            color_temp_mireds=333,
            transition_ms=1000,
        )
        state = {"color_temp": 250}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        change = changes[0]
        assert change.start_hs is None
        assert change.end_hs is None


class TestCalculateChangesHybridHsToMireds:
    """Test hybrid HS -> mireds transition detection."""

    def test_off_locus_hs_to_mireds_returns_two_changes(self) -> None:
        """Test that off-locus HS to mireds returns two-phase hybrid transition."""
        params = FadeParams(
            color_temp_mireds=333,
            transition_ms=1000,
        )
        # High saturation HS (off Planckian locus)
        state = {HA_ATTR_HS_COLOR: (120.0, 80.0)}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        assert len(changes) == 2

    def test_off_locus_hs_to_mireds_first_phase_is_hs(self) -> None:
        """Test that first phase is HS transition."""
        params = FadeParams(
            color_temp_mireds=333,
            transition_ms=1000,
        )
        state = {HA_ATTR_HS_COLOR: (120.0, 80.0)}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        phase1 = changes[0]
        assert phase1.start_hs is not None
        assert phase1.end_hs is not None
        assert phase1.start_mireds is None
        assert phase1.end_mireds is None

    def test_off_locus_hs_to_mireds_second_phase_is_mireds(self) -> None:
        """Test that second phase is mireds transition."""
        params = FadeParams(
            color_temp_mireds=333,
            transition_ms=1000,
        )
        state = {HA_ATTR_HS_COLOR: (120.0, 80.0)}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        phase2 = changes[1]
        assert phase2.start_mireds is not None
        assert phase2.end_mireds == 333
        assert phase2.start_hs is None
        assert phase2.end_hs is None

    def test_on_locus_hs_to_mireds_returns_single_change(self) -> None:
        """Test that on-locus HS to mireds returns single mireds-only change."""
        params = FadeParams(
            color_temp_mireds=333,
            transition_ms=1000,
        )
        # Low saturation HS (on Planckian locus)
        state = {HA_ATTR_HS_COLOR: (35.0, 10.0)}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        # Should be single phase since already on locus
        assert len(changes) == 1
        change = changes[0]
        # Should be mireds transition (HS was converted to mireds)
        assert change.end_mireds == 333

    def test_hs_to_mireds_with_both_hs_and_mireds_target_is_not_hybrid(self) -> None:
        """Test that specifying both HS and mireds targets uses simple fade (not hybrid)."""
        params = FadeParams(
            hs_color=(240.0, 100.0),  # Specifying HS target
            color_temp_mireds=333,  # And also mireds target
            transition_ms=1000,
        )
        state = {HA_ATTR_HS_COLOR: (120.0, 80.0)}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        # Should be single phase since end_hs is specified (condition end_hs is None fails)
        assert len(changes) == 1


class TestCalculateChangesHybridMiredsToHs:
    """Test hybrid mireds -> HS transition detection."""

    def test_mireds_to_hs_returns_two_changes(self) -> None:
        """Test that mireds to HS returns two-phase hybrid transition."""
        params = FadeParams(
            hs_color=(120.0, 80.0),
            transition_ms=1000,
        )
        state = {"color_temp": 333}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        assert len(changes) == 2

    def test_mireds_to_hs_first_phase_is_mireds(self) -> None:
        """Test that first phase is mireds transition."""
        params = FadeParams(
            hs_color=(120.0, 80.0),
            transition_ms=1000,
        )
        state = {"color_temp": 154}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        phase1 = changes[0]
        assert phase1.start_mireds == 154
        assert phase1.end_mireds is not None
        assert phase1.start_hs is None
        assert phase1.end_hs is None

    def test_mireds_to_hs_second_phase_is_hs(self) -> None:
        """Test that second phase is HS transition."""
        params = FadeParams(
            hs_color=(120.0, 80.0),
            transition_ms=1000,
        )
        state = {"color_temp": 154}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        phase2 = changes[1]
        assert phase2.start_hs is not None
        assert phase2.end_hs == (120.0, 80.0)
        assert phase2.start_mireds is None
        assert phase2.end_mireds is None

    def test_mireds_to_hs_with_both_targets_is_not_hybrid(self) -> None:
        """Test that specifying both mireds and HS targets uses simple fade."""
        params = FadeParams(
            hs_color=(120.0, 80.0),  # HS target
            color_temp_mireds=400,  # Also mireds target
            transition_ms=1000,
        )
        state = {"color_temp": 154}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        # Should be single phase since end_mireds is specified (condition end_mireds is None fails)
        assert len(changes) == 1


class TestCalculateChangesCombinedFades:
    """Test combined brightness + color fades."""

    def test_brightness_and_hs_combined(self) -> None:
        """Test combined brightness and HS color fade."""
        params = FadeParams(
            brightness_pct=75,
            hs_color=(180.0, 90.0),
            transition_ms=1000,
        )
        state = {
            ATTR_BRIGHTNESS: 100,
            HA_ATTR_HS_COLOR: (60.0, 50.0),
        }

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        assert len(changes) == 1
        change = changes[0]
        assert change.start_brightness == 100
        assert change.end_brightness == 191
        assert change.start_hs == (60.0, 50.0)
        assert change.end_hs == (180.0, 90.0)

    def test_brightness_and_mireds_combined(self) -> None:
        """Test combined brightness and mireds fade."""
        params = FadeParams(
            brightness_pct=50,
            color_temp_mireds=400,
            transition_ms=1000,
        )
        state = {
            ATTR_BRIGHTNESS: 200,
            "color_temp": 250,
        }

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        assert len(changes) == 1
        change = changes[0]
        assert change.start_brightness == 200
        assert change.end_brightness == 127
        assert change.start_mireds == 250
        assert change.end_mireds == 400

    def test_brightness_with_hs_to_mireds_hybrid(self) -> None:
        """Test combined brightness and hybrid HS->mireds transition."""
        params = FadeParams(
            brightness_pct=100,
            color_temp_mireds=333,
            transition_ms=1000,
        )
        state = {
            ATTR_BRIGHTNESS: 50,
            HA_ATTR_HS_COLOR: (120.0, 80.0),  # Off locus
        }

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        assert len(changes) == 2
        # Both phases should have brightness transitions
        assert changes[0].start_brightness is not None
        assert changes[0].end_brightness is not None
        assert changes[1].start_brightness is not None
        assert changes[1].end_brightness == 255


class TestCalculateChangesTimingParameters:
    """Test that timing parameters are correctly passed through."""

    def test_transition_ms_passed_to_simple_change(self) -> None:
        """Test that transition_ms is passed to simple FadeChange."""
        params = FadeParams(brightness_pct=50, transition_ms=2000)
        state = {ATTR_BRIGHTNESS: 100}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        assert changes[0].transition_ms == 2000

    def test_min_step_delay_ms_passed_to_simple_change(self) -> None:
        """Test that min_step_delay_ms is passed to simple FadeChange."""
        params = FadeParams(brightness_pct=50, transition_ms=1000)
        state = {ATTR_BRIGHTNESS: 100}

        changes = _calculate_changes(params, state, min_step_delay_ms=75)

        assert changes[0].min_step_delay_ms == 75

    def test_timing_passed_to_hybrid_changes(self) -> None:
        """Test that timing parameters are passed to hybrid changes."""
        params = FadeParams(
            color_temp_mireds=333,
            transition_ms=1000,
        )
        state = {HA_ATTR_HS_COLOR: (120.0, 80.0)}

        changes = _calculate_changes(params, state, min_step_delay_ms=50)

        for change in changes:
            assert change.min_step_delay_ms == 50


class TestCalculateChangesEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_state(self) -> None:
        """Test with empty state dictionary."""
        params = FadeParams(brightness_pct=50, transition_ms=1000)
        state = {}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        assert len(changes) == 1
        change = changes[0]
        assert change.start_brightness is None
        assert change.end_brightness == 127

    def test_no_end_values(self) -> None:
        """Test with no end values specified (all defaults)."""
        params = FadeParams(transition_ms=1000)
        state = {ATTR_BRIGHTNESS: 100}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        # Should return a FadeChange but with no end values
        assert len(changes) == 1
        change = changes[0]
        assert change.end_brightness is None
        assert change.end_hs is None
        assert change.end_mireds is None

    def test_start_hs_excluded_when_no_end_hs(self) -> None:
        """Test that start_hs is excluded when end_hs is not specified."""
        params = FadeParams(
            brightness_pct=50,
            transition_ms=1000,
        )
        state = {
            ATTR_BRIGHTNESS: 100,
            HA_ATTR_HS_COLOR: (120.0, 80.0),  # HS in state but not targeting it
        }

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        change = changes[0]
        # start_hs should be None because we're not fading to HS
        assert change.start_hs is None
        assert change.end_hs is None

    def test_start_mireds_excluded_when_no_end_mireds(self) -> None:
        """Test that start_mireds is excluded when end_mireds is not specified."""
        params = FadeParams(
            brightness_pct=50,
            transition_ms=1000,
        )
        state = {
            ATTR_BRIGHTNESS: 100,
            "color_temp": 333,  # Mireds in state but not targeting it
        }

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        change = changes[0]
        # start_mireds should be None because we're not fading to mireds
        assert change.start_mireds is None
        assert change.end_mireds is None

    def test_start_values_included_when_end_values_specified(self) -> None:
        """Test that start values are included when corresponding end values exist."""
        params = FadeParams(
            hs_color=(240.0, 100.0),
            color_temp_mireds=400,
            transition_ms=1000,
        )
        state = {
            HA_ATTR_HS_COLOR: (120.0, 50.0),
            "color_temp": 250,
        }

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        change = changes[0]
        # Both start values should be included because both end values are specified
        assert change.start_hs == (120.0, 50.0)
        assert change.end_hs == (240.0, 100.0)
        assert change.start_mireds == 250
        assert change.end_mireds == 400

    def test_saturation_threshold_boundary_on_locus(self) -> None:
        """Test saturation at exactly the threshold is considered on locus."""
        params = FadeParams(
            color_temp_mireds=333,
            transition_ms=1000,
        )
        # Saturation at threshold (15) - should be on locus
        state = {HA_ATTR_HS_COLOR: (35.0, 15.0)}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        # On locus means single phase
        assert len(changes) == 1

    def test_saturation_threshold_boundary_off_locus(self) -> None:
        """Test saturation just above threshold is considered off locus."""
        params = FadeParams(
            color_temp_mireds=333,
            transition_ms=1000,
        )
        # Saturation just above threshold (16) - should be off locus
        state = {HA_ATTR_HS_COLOR: (35.0, 16.0)}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        # Off locus means two phases (hybrid)
        assert len(changes) == 2


class TestCalculateChangesFadeChangeIterator:
    """Test that returned FadeChange objects can generate steps."""

    def test_simple_change_generates_steps(self) -> None:
        """Test that simple FadeChange can generate steps."""
        params = FadeParams(brightness_pct=50, transition_ms=1000)
        state = {ATTR_BRIGHTNESS: 200}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        change = changes[0]
        assert change.step_count() >= 1

        total_steps = 0
        while change.has_next():
            step = change.next_step()
            assert step is not None
            total_steps += 1

        assert total_steps == change.step_count()

    def test_hybrid_changes_generate_steps(self) -> None:
        """Test that hybrid FadeChange objects can generate steps."""
        params = FadeParams(
            color_temp_mireds=333,
            transition_ms=1000,
        )
        state = {HA_ATTR_HS_COLOR: (120.0, 80.0)}

        changes = _calculate_changes(params, state, min_step_delay_ms=100)

        total_steps = 0
        for change in changes:
            while change.has_next():
                step = change.next_step()
                assert step is not None
                total_steps += 1

        assert total_steps > 0
