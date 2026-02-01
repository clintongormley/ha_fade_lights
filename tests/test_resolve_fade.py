"""Tests for the _resolve_fade function."""

from __future__ import annotations

from homeassistant.components.light import ATTR_BRIGHTNESS, ATTR_SUPPORTED_COLOR_MODES
from homeassistant.components.light import ATTR_COLOR_TEMP_KELVIN as HA_ATTR_COLOR_TEMP_KELVIN
from homeassistant.components.light import ATTR_HS_COLOR as HA_ATTR_HS_COLOR
from homeassistant.components.light.const import ColorMode

from custom_components.fade_lights import _resolve_fade
from custom_components.fade_lights.fade_change import FadeChange
from custom_components.fade_lights.fade_params import FadeParams


class TestResolveFadeBasicStructure:
    """Test basic return type and structure of _resolve_fade."""

    def test_returns_fade_change_or_none(self) -> None:
        """Test that the function returns a FadeChange object or None."""
        params = FadeParams(brightness_pct=50, transition_ms=1000)
        state = {
            ATTR_BRIGHTNESS: 100,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert isinstance(change, FadeChange)

    def test_returns_none_when_nothing_to_fade(self) -> None:
        """Test that None is returned when there's nothing to change."""
        params = FadeParams(transition_ms=1000)  # No target values
        state = {
            ATTR_BRIGHTNESS: 100,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is None

    def test_returns_none_when_target_equals_current(self) -> None:
        """Test that None is returned when target equals current state."""
        params = FadeParams(brightness_pct=50, transition_ms=1000)  # 50% = 127
        state = {
            ATTR_BRIGHTNESS: 127,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is None


class TestResolveFadeSimpleBrightnessFade:
    """Test simple brightness-only fade scenarios."""

    def test_brightness_fade_values(self) -> None:
        """Test that brightness values are correctly resolved."""
        params = FadeParams(brightness_pct=75, transition_ms=1000)
        state = {
            ATTR_BRIGHTNESS: 100,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
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
        state = {
            ATTR_BRIGHTNESS: 200,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
        # 25% of 255 = 63
        assert change.start_brightness == 63
        assert change.end_brightness == 255

    def test_brightness_fade_no_color_attributes(self) -> None:
        """Test that brightness-only fade has no color attributes."""
        params = FadeParams(brightness_pct=50, transition_ms=1000)
        state = {
            ATTR_BRIGHTNESS: 100,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.start_hs is None
        assert change.end_hs is None
        assert change.start_mireds is None
        assert change.end_mireds is None


class TestResolveFadeSimpleHsFade:
    """Test simple HS color fade scenarios."""

    def test_hs_color_fade_values(self) -> None:
        """Test that HS color values are correctly resolved."""
        params = FadeParams(
            hs_color=(120.0, 80.0),
            transition_ms=1000,
        )
        state = {
            HA_ATTR_HS_COLOR: (60.0, 50.0),
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.start_hs == (60.0, 50.0)
        assert change.end_hs == (120.0, 80.0)

    def test_hs_color_fade_from_override(self) -> None:
        """Test that from_hs_color overrides state HS color."""
        params = FadeParams(
            hs_color=(240.0, 100.0),
            from_hs_color=(0.0, 100.0),
            transition_ms=1000,
        )
        state = {
            HA_ATTR_HS_COLOR: (180.0, 50.0),
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.start_hs == (0.0, 100.0)
        assert change.end_hs == (240.0, 100.0)

    def test_hs_color_only_no_mireds(self) -> None:
        """Test that HS-only fade has no mireds attributes."""
        params = FadeParams(
            hs_color=(120.0, 80.0),
            transition_ms=1000,
        )
        state = {
            HA_ATTR_HS_COLOR: (60.0, 50.0),
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.start_mireds is None
        assert change.end_mireds is None


class TestResolveFadeSimpleColorTempFade:
    """Test simple color temperature fade scenarios."""

    def test_color_temp_fade_values(self) -> None:
        """Test that color temp values are correctly converted to mireds."""
        params = FadeParams(
            color_temp_kelvin=2500,  # 400 mireds
            transition_ms=1000,
        )
        state = {
            HA_ATTR_COLOR_TEMP_KELVIN: 5000,  # 200 mireds
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.COLOR_TEMP],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.start_mireds == 200
        assert change.end_mireds == 400

    def test_color_temp_fade_from_override(self) -> None:
        """Test that from_color_temp_kelvin overrides state kelvin."""
        params = FadeParams(
            color_temp_kelvin=2000,  # 500 mireds
            from_color_temp_kelvin=6500,  # 153 mireds (int(1_000_000/6500))
            transition_ms=1000,
        )
        state = {
            HA_ATTR_COLOR_TEMP_KELVIN: 3000,  # ~333 mireds
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.COLOR_TEMP],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.start_mireds == 153
        assert change.end_mireds == 500

    def test_color_temp_only_no_hs(self) -> None:
        """Test that color temp only fade has no HS attributes."""
        params = FadeParams(
            color_temp_kelvin=3000,  # ~333 mireds
            transition_ms=1000,
        )
        state = {
            HA_ATTR_COLOR_TEMP_KELVIN: 4000,  # ~250 mireds
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.COLOR_TEMP],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.start_hs is None
        assert change.end_hs is None


class TestResolveFadeHybridTransitions:
    """Test hybrid HS <-> color temp transition detection."""

    def test_off_locus_hs_to_color_temp_is_hybrid(self) -> None:
        """Test that off-locus HS to color temp creates hybrid FadeChange."""
        params = FadeParams(
            color_temp_kelvin=3000,  # ~333 mireds
            transition_ms=1000,
        )
        # High saturation HS (off Planckian locus)
        state = {
            HA_ATTR_HS_COLOR: (120.0, 80.0),
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS, ColorMode.COLOR_TEMP],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
        # Should be hybrid - has _hybrid_direction set
        assert change._hybrid_direction == "hs_to_mireds"
        assert change._crossover_step is not None
        assert change._crossover_hs is not None
        assert change._crossover_mireds is not None

    def test_color_temp_to_hs_is_hybrid(self) -> None:
        """Test that color temp to HS creates hybrid FadeChange."""
        params = FadeParams(
            hs_color=(120.0, 80.0),
            transition_ms=1000,
        )
        state = {
            HA_ATTR_COLOR_TEMP_KELVIN: 3000,  # ~333 mireds
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS, ColorMode.COLOR_TEMP],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
        # Should be hybrid - has _hybrid_direction set
        assert change._hybrid_direction == "mireds_to_hs"
        assert change._crossover_step is not None
        assert change._crossover_hs is not None
        assert change._crossover_mireds is not None

    def test_on_locus_hs_to_color_temp_is_not_hybrid(self) -> None:
        """Test that on-locus HS to color temp is simple (not hybrid)."""
        params = FadeParams(
            color_temp_kelvin=3000,  # ~333 mireds
            transition_ms=1000,
        )
        # Low saturation HS (on Planckian locus)
        state = {
            HA_ATTR_HS_COLOR: (35.0, 10.0),
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS, ColorMode.COLOR_TEMP],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
        # Should NOT be hybrid
        assert change._hybrid_direction is None

    def test_hs_to_hs_is_not_hybrid(self) -> None:
        """Test that HS to HS is simple (not hybrid)."""
        params = FadeParams(
            hs_color=(240.0, 100.0),
            transition_ms=1000,
        )
        state = {
            HA_ATTR_HS_COLOR: (120.0, 80.0),
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS, ColorMode.COLOR_TEMP],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change._hybrid_direction is None

    def test_both_targets_specified_is_not_hybrid(self) -> None:
        """Test that specifying both HS and color temp targets is not hybrid."""
        params = FadeParams(
            hs_color=(240.0, 100.0),  # Specifying HS target
            color_temp_kelvin=3000,  # And also color temp target
            transition_ms=1000,
        )
        state = {
            HA_ATTR_HS_COLOR: (120.0, 80.0),
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS, ColorMode.COLOR_TEMP],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change._hybrid_direction is None


class TestResolveFadeHybridStepGeneration:
    """Test that hybrid FadeChange generates correct steps across crossover."""

    def test_hs_to_mireds_generates_hs_before_crossover(self) -> None:
        """Test that HS->mireds hybrid emits hs_color before crossover."""
        params = FadeParams(
            color_temp_kelvin=3000,
            transition_ms=1000,
        )
        state = {
            HA_ATTR_HS_COLOR: (120.0, 80.0),
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS, ColorMode.COLOR_TEMP],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change._hybrid_direction == "hs_to_mireds"

        # Get first step (should be HS)
        step = change.next_step()
        assert step.hs_color is not None
        assert step.color_temp_kelvin is None

    def test_hs_to_mireds_generates_color_temp_after_crossover(self) -> None:
        """Test that HS->mireds hybrid emits color_temp_kelvin after crossover."""
        params = FadeParams(
            color_temp_kelvin=3000,
            transition_ms=1000,
        )
        state = {
            HA_ATTR_HS_COLOR: (120.0, 80.0),
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS, ColorMode.COLOR_TEMP],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
        crossover = change._crossover_step or 0

        # Iterate to after crossover
        steps_with_hs = 0
        steps_with_kelvin = 0
        while change.has_next():
            step = change.next_step()
            if step.hs_color is not None:
                steps_with_hs += 1
            if step.color_temp_kelvin is not None:
                steps_with_kelvin += 1

        # Should have both HS and color_temp steps
        assert steps_with_hs > 0
        assert steps_with_kelvin > 0

    def test_mireds_to_hs_generates_color_temp_before_crossover(self) -> None:
        """Test that mireds->HS hybrid emits color_temp_kelvin before crossover."""
        params = FadeParams(
            hs_color=(120.0, 80.0),
            transition_ms=1000,
        )
        state = {
            HA_ATTR_COLOR_TEMP_KELVIN: 3000,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS, ColorMode.COLOR_TEMP],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change._hybrid_direction == "mireds_to_hs"

        # Get first step (should be color_temp)
        step = change.next_step()
        assert step.color_temp_kelvin is not None
        assert step.hs_color is None

    def test_mireds_to_hs_generates_hs_after_crossover(self) -> None:
        """Test that mireds->HS hybrid emits hs_color after crossover."""
        params = FadeParams(
            hs_color=(120.0, 80.0),
            transition_ms=1000,
        )
        state = {
            HA_ATTR_COLOR_TEMP_KELVIN: 3000,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS, ColorMode.COLOR_TEMP],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None

        # Iterate through all steps
        steps_with_hs = 0
        steps_with_kelvin = 0
        while change.has_next():
            step = change.next_step()
            if step.hs_color is not None:
                steps_with_hs += 1
            if step.color_temp_kelvin is not None:
                steps_with_kelvin += 1

        # Should have both color_temp and HS steps
        assert steps_with_kelvin > 0
        assert steps_with_hs > 0


class TestResolveFadeNonDimmableLights:
    """Test handling of non-dimmable (on/off only) lights."""

    def test_non_dimmable_light_returns_single_step(self) -> None:
        """Test that non-dimmable light gets single step FadeChange."""
        params = FadeParams(brightness_pct=100, transition_ms=1000)
        state = {
            ATTR_BRIGHTNESS: 0,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.ONOFF],  # On/off only
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.end_brightness == 255
        assert change.transition_ms == 0  # Zero transition for on/off
        assert change.step_count() == 1

    def test_non_dimmable_light_turns_off(self) -> None:
        """Test that non-dimmable light can be turned off."""
        params = FadeParams(brightness_pct=0, transition_ms=1000)
        state = {
            ATTR_BRIGHTNESS: 255,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.ONOFF],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.end_brightness == 0

    def test_non_dimmable_light_ignores_color_params(self) -> None:
        """Test that non-dimmable light ignores color parameters."""
        params = FadeParams(
            brightness_pct=100,
            hs_color=(120.0, 80.0),  # This should be ignored
            transition_ms=1000,
        )
        state = {
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.ONOFF],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.end_hs is None
        assert change.end_mireds is None


class TestResolveFadeCapabilityFiltering:
    """Test capability-based filtering of color modes."""

    def test_color_temp_converted_to_hs_when_not_supported(self) -> None:
        """Test that color temp is converted to HS when light only supports HS."""
        params = FadeParams(
            color_temp_kelvin=3000,  # Target color temp
            transition_ms=1000,
        )
        state = {
            HA_ATTR_HS_COLOR: (120.0, 50.0),
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS],  # Only HS, no COLOR_TEMP
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
        # Should have converted to HS
        assert change.end_hs is not None
        assert change.end_mireds is None

    def test_hs_dropped_when_only_color_temp_supported(self) -> None:
        """Test that HS is dropped when light only supports color temp."""
        params = FadeParams(
            hs_color=(120.0, 80.0),  # High saturation - can't convert
            transition_ms=1000,
        )
        state = {
            HA_ATTR_COLOR_TEMP_KELVIN: 3000,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.COLOR_TEMP],  # Only color temp, no HS
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        # Should return None since HS can't be applied
        assert change is None

    def test_hs_converted_to_mireds_when_on_locus(self) -> None:
        """Test that low-saturation HS is converted to mireds when appropriate."""
        params = FadeParams(
            hs_color=(35.0, 10.0),  # On locus - can convert
            transition_ms=1000,
        )
        state = {
            HA_ATTR_COLOR_TEMP_KELVIN: 5000,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.COLOR_TEMP],  # Only color temp
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
        # Should have converted to mireds
        assert change.end_mireds is not None
        assert change.end_hs is None


class TestResolveFadeTimingParameters:
    """Test that timing parameters are correctly passed through."""

    def test_transition_ms_passed_to_change(self) -> None:
        """Test that transition_ms is passed to FadeChange."""
        params = FadeParams(brightness_pct=50, transition_ms=2000)
        state = {
            ATTR_BRIGHTNESS: 100,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.transition_ms == 2000

    def test_min_step_delay_ms_passed_to_change(self) -> None:
        """Test that min_step_delay_ms is passed to FadeChange."""
        params = FadeParams(brightness_pct=50, transition_ms=1000)
        state = {
            ATTR_BRIGHTNESS: 100,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=75)

        assert change is not None
        assert change.min_step_delay_ms == 75


class TestResolveFadeEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_state(self) -> None:
        """Test with empty state dictionary (light off)."""
        params = FadeParams(brightness_pct=50, transition_ms=1000)
        state = {
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
        # When light is off (no brightness in state), start_brightness is 0
        assert change.start_brightness == 0
        assert change.end_brightness == 127

    def test_saturation_threshold_boundary_on_locus(self) -> None:
        """Test saturation at exactly the threshold is considered on locus."""
        params = FadeParams(
            color_temp_kelvin=3000,
            transition_ms=1000,
        )
        # Saturation at threshold (15) - should be on locus
        state = {
            HA_ATTR_HS_COLOR: (35.0, 15.0),
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS, ColorMode.COLOR_TEMP],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
        # On locus means NOT hybrid
        assert change._hybrid_direction is None

    def test_saturation_threshold_boundary_off_locus(self) -> None:
        """Test saturation just above threshold is considered off locus."""
        params = FadeParams(
            color_temp_kelvin=3000,
            transition_ms=1000,
        )
        # Saturation just above threshold (16) - should be off locus
        state = {
            HA_ATTR_HS_COLOR: (35.0, 16.0),
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS, ColorMode.COLOR_TEMP],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
        # Off locus means hybrid
        assert change._hybrid_direction == "hs_to_mireds"


class TestResolveFadeFadeChangeIterator:
    """Test that returned FadeChange objects can generate steps."""

    def test_simple_change_generates_steps(self) -> None:
        """Test that simple FadeChange can generate steps."""
        params = FadeParams(brightness_pct=50, transition_ms=1000)
        state = {
            ATTR_BRIGHTNESS: 200,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.step_count() >= 1

        total_steps = 0
        while change.has_next():
            step = change.next_step()
            assert step is not None
            total_steps += 1

        assert total_steps == change.step_count()

    def test_hybrid_change_generates_steps(self) -> None:
        """Test that hybrid FadeChange can generate steps."""
        params = FadeParams(
            color_temp_kelvin=3000,
            transition_ms=1000,
        )
        state = {
            HA_ATTR_HS_COLOR: (120.0, 80.0),
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS, ColorMode.COLOR_TEMP],
        }

        change = _resolve_fade(params, state, min_step_delay_ms=100)

        assert change is not None

        total_steps = 0
        while change.has_next():
            step = change.next_step()
            assert step is not None
            total_steps += 1

        assert total_steps > 0
        assert total_steps == change.step_count()
