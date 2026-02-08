"""Tests for all parameter combinations in the fado integration.

This file tests all combinations from the parameter matrix:
- From: off, on with HS, on with CT
- To: brightness, HS, CT, combinations, hybrid transitions
- With and without from: overrides
"""

from __future__ import annotations

from homeassistant.components.light import ATTR_BRIGHTNESS, ATTR_SUPPORTED_COLOR_MODES
from homeassistant.components.light import ATTR_COLOR_TEMP_KELVIN as HA_ATTR_COLOR_TEMP_KELVIN
from homeassistant.components.light import ATTR_HS_COLOR as HA_ATTR_HS_COLOR
from homeassistant.components.light.const import ColorMode

from custom_components.fado.fade_change import FadeChange
from custom_components.fado.fade_params import FadeParams


class TestFromOffState:
    """Test fades starting from off state (brightness=0 or None).

    Note: With min_brightness=1 (default), start_brightness is clamped to 1 when light is off.
    """

    def test_off_to_brightness(self) -> None:
        """#1: off → b:100 - fade brightness from min_brightness to 100."""
        params = FadeParams(brightness_pct=100, transition_ms=1000)
        state = {
            ATTR_BRIGHTNESS: None,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None
        # With min_brightness=1 (default), start from 1 not 0
        assert change.start_brightness == 1
        assert change.end_brightness == 255

    def test_off_to_hs_auto_turn_on(self) -> None:
        """#2: off → hs:[100,100] - auto-turn-on + fade HS from white."""
        params = FadeParams(hs_color=(100.0, 100.0), transition_ms=1000)
        state = {
            ATTR_BRIGHTNESS: None,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS],
        }

        # stored_brightness=0 means use 255
        change = FadeChange.resolve(params, state, min_step_delay_ms=100, stored_brightness=0)

        assert change is not None
        # Auto-turn-on: brightness min_brightness→255
        assert change.start_brightness == 1  # clamped from 0 to min_brightness
        assert change.end_brightness == 255
        # HS from white (0,0) to target
        assert change.start_hs == (0.0, 0.0)
        assert change.end_hs == (100.0, 100.0)

    def test_off_to_hs_uses_stored_brightness(self) -> None:
        """#2 variant: off → hs uses stored brightness if available."""
        params = FadeParams(hs_color=(100.0, 100.0), transition_ms=1000)
        state = {
            ATTR_BRIGHTNESS: None,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS],
        }

        # stored_brightness=200 from previous session
        change = FadeChange.resolve(params, state, min_step_delay_ms=100, stored_brightness=200)

        assert change is not None
        # Start brightness clamped to min_brightness
        assert change.start_brightness == 1
        assert change.end_brightness == 200  # Uses stored

    def test_off_to_color_temp_auto_turn_on(self) -> None:
        """#3: off → ct:4000 - auto-turn-on + fade CT from boundary."""
        params = FadeParams(color_temp_kelvin=4000, transition_ms=1000)
        state = {
            ATTR_BRIGHTNESS: None,
            "min_color_temp_kelvin": 2500,
            "max_color_temp_kelvin": 6500,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.COLOR_TEMP],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100, stored_brightness=0)

        assert change is not None
        # Auto-turn-on, with start clamped to min_brightness
        assert change.start_brightness == 1
        assert change.end_brightness == 255
        # CT from boundary (min=400 mireds, max=153 mireds, target=250 mireds)
        # Target 4000K = 250 mireds, closer to max (153) than min (400)
        assert change.start_mireds == 153  # max boundary (6500K)
        assert change.end_mireds == 250  # 4000K

    def test_off_to_brightness_and_hs(self) -> None:
        """#4: off → b:50,hs:[100,100] - fade brightness min_brightness→50, HS from white."""
        params = FadeParams(
            brightness_pct=50,
            hs_color=(100.0, 100.0),
            transition_ms=1000,
        )
        state = {
            ATTR_BRIGHTNESS: None,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None
        # Start brightness clamped to min_brightness
        assert change.start_brightness == 1
        assert change.end_brightness == 127  # 50%
        assert change.start_hs == (0.0, 0.0)  # white
        assert change.end_hs == (100.0, 100.0)

    def test_off_to_brightness_and_color_temp(self) -> None:
        """#5: off → b:50,ct:4000 - fade brightness min_brightness→50, CT from boundary."""
        params = FadeParams(
            brightness_pct=50,
            color_temp_kelvin=4000,
            transition_ms=1000,
        )
        state = {
            ATTR_BRIGHTNESS: None,
            "min_color_temp_kelvin": 2500,
            "max_color_temp_kelvin": 6500,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.COLOR_TEMP],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None
        # Start brightness clamped to min_brightness
        assert change.start_brightness == 1
        assert change.end_brightness == 127  # 50%
        assert change.start_mireds == 153  # boundary (6500K, closer to 4000K)
        assert change.end_mireds == 250  # 4000K


class TestFromOnHsState:
    """Test fades starting from on state with HS color."""

    def test_hs_to_brightness_only(self) -> None:
        """#6: b:50,hs:[10,10] → b:100 - fade brightness only."""
        params = FadeParams(brightness_pct=100, transition_ms=1000)
        state = {
            ATTR_BRIGHTNESS: 127,  # 50%
            HA_ATTR_HS_COLOR: (10.0, 10.0),
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.start_brightness == 127
        assert change.end_brightness == 255
        # No color change
        assert change.start_hs is None
        assert change.end_hs is None

    def test_hs_to_hs(self) -> None:
        """#7: b:50,hs:[10,10] → hs:[100,100] - fade HS only."""
        params = FadeParams(hs_color=(100.0, 100.0), transition_ms=1000)
        state = {
            ATTR_BRIGHTNESS: 127,
            HA_ATTR_HS_COLOR: (10.0, 10.0),
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None
        # No brightness change
        assert change.start_brightness is None
        assert change.end_brightness is None
        # HS changes
        assert change.start_hs == (10.0, 10.0)
        assert change.end_hs == (100.0, 100.0)

    def test_hs_to_brightness_and_hs(self) -> None:
        """#10: b:50,hs:[10,10] → b:100,hs:[100,100] - fade both."""
        params = FadeParams(
            brightness_pct=100,
            hs_color=(100.0, 100.0),
            transition_ms=1000,
        )
        state = {
            ATTR_BRIGHTNESS: 127,
            HA_ATTR_HS_COLOR: (10.0, 10.0),
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.start_brightness == 127
        assert change.end_brightness == 255
        assert change.start_hs == (10.0, 10.0)
        assert change.end_hs == (100.0, 100.0)

    def test_hs_to_color_temp_hybrid(self) -> None:
        """#12: b:50,hs:[10,10] → ct:4000 - hybrid HS→CT transition."""
        params = FadeParams(color_temp_kelvin=4000, transition_ms=1000)
        state = {
            ATTR_BRIGHTNESS: 127,
            HA_ATTR_HS_COLOR: (120.0, 80.0),  # Off locus (high saturation)
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS, ColorMode.COLOR_TEMP],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.hybrid_direction == "hs_to_mireds"
        assert change.crossover_step is not None

    def test_hs_to_brightness_and_color_temp_hybrid(self) -> None:
        """#13: b:50,hs:[10,10] → b:100,ct:4000 - brightness + hybrid."""
        params = FadeParams(
            brightness_pct=100,
            color_temp_kelvin=4000,
            transition_ms=1000,
        )
        state = {
            ATTR_BRIGHTNESS: 127,
            HA_ATTR_HS_COLOR: (120.0, 80.0),  # Off locus
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS, ColorMode.COLOR_TEMP],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None
        # Brightness changes
        assert change.start_brightness == 127
        assert change.end_brightness == 255
        # Hybrid transition
        assert change.hybrid_direction == "hs_to_mireds"

    def test_hs_to_off(self) -> None:
        """#16: b:50,hs:[10,10] → b:0 - fade to off."""
        params = FadeParams(brightness_pct=0, transition_ms=1000)
        state = {
            ATTR_BRIGHTNESS: 127,
            HA_ATTR_HS_COLOR: (10.0, 10.0),
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.start_brightness == 127
        assert change.end_brightness == 0

    def test_hs_to_off_with_hs_change(self) -> None:
        """#18: b:50,hs:[10,10] → b:0,hs:[100,100] - fade to off with HS change."""
        params = FadeParams(
            brightness_pct=0,
            hs_color=(100.0, 100.0),
            transition_ms=1000,
        )
        state = {
            ATTR_BRIGHTNESS: 127,
            HA_ATTR_HS_COLOR: (10.0, 10.0),
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.start_brightness == 127
        assert change.end_brightness == 0
        assert change.start_hs == (10.0, 10.0)
        assert change.end_hs == (100.0, 100.0)

    def test_hs_to_off_with_hybrid(self) -> None:
        """#20: b:50,hs:[10,10] → b:0,ct:4000 - fade to off with hybrid."""
        params = FadeParams(
            brightness_pct=0,
            color_temp_kelvin=4000,
            transition_ms=1000,
        )
        state = {
            ATTR_BRIGHTNESS: 127,
            HA_ATTR_HS_COLOR: (120.0, 80.0),  # Off locus
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS, ColorMode.COLOR_TEMP],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.start_brightness == 127
        assert change.end_brightness == 0
        assert change.hybrid_direction == "hs_to_mireds"


class TestFromOnColorTempState:
    """Test fades starting from on state with color temperature."""

    def test_ct_to_brightness_only(self) -> None:
        """#8: b:50,ct:3000 → b:100 - fade brightness only."""
        params = FadeParams(brightness_pct=100, transition_ms=1000)
        state = {
            ATTR_BRIGHTNESS: 127,
            HA_ATTR_COLOR_TEMP_KELVIN: 3000,
            "color_mode": ColorMode.COLOR_TEMP,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.COLOR_TEMP],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.start_brightness == 127
        assert change.end_brightness == 255
        # No color change
        assert change.start_mireds is None
        assert change.end_mireds is None

    def test_ct_to_ct(self) -> None:
        """#9: b:50,ct:3000 → ct:6000 - fade CT only."""
        params = FadeParams(color_temp_kelvin=6000, transition_ms=1000)
        state = {
            ATTR_BRIGHTNESS: 127,
            HA_ATTR_COLOR_TEMP_KELVIN: 3000,
            "color_mode": ColorMode.COLOR_TEMP,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.COLOR_TEMP],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None
        # No brightness change
        assert change.start_brightness is None
        assert change.end_brightness is None
        # CT changes (3000K=333 mireds, 6000K=166 mireds)
        assert change.start_mireds == 333
        assert change.end_mireds == 166

    def test_ct_to_brightness_and_ct(self) -> None:
        """#11: b:50,ct:3000 → b:100,ct:6000 - fade both."""
        params = FadeParams(
            brightness_pct=100,
            color_temp_kelvin=6000,
            transition_ms=1000,
        )
        state = {
            ATTR_BRIGHTNESS: 127,
            HA_ATTR_COLOR_TEMP_KELVIN: 3000,
            "color_mode": ColorMode.COLOR_TEMP,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.COLOR_TEMP],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.start_brightness == 127
        assert change.end_brightness == 255
        assert change.start_mireds == 333
        assert change.end_mireds == 166

    def test_ct_to_hs_hybrid(self) -> None:
        """#14: b:50,ct:3000 → hs:[10,10] - hybrid CT→HS transition."""
        params = FadeParams(hs_color=(120.0, 80.0), transition_ms=1000)
        state = {
            ATTR_BRIGHTNESS: 127,
            HA_ATTR_COLOR_TEMP_KELVIN: 3000,
            "color_mode": ColorMode.COLOR_TEMP,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS, ColorMode.COLOR_TEMP],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.hybrid_direction == "mireds_to_hs"
        assert change.crossover_step is not None

    def test_ct_to_brightness_and_hs_hybrid(self) -> None:
        """#15: b:50,ct:3000 → b:100,hs:[10,10] - brightness + hybrid."""
        params = FadeParams(
            brightness_pct=100,
            hs_color=(120.0, 80.0),
            transition_ms=1000,
        )
        state = {
            ATTR_BRIGHTNESS: 127,
            HA_ATTR_COLOR_TEMP_KELVIN: 3000,
            "color_mode": ColorMode.COLOR_TEMP,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS, ColorMode.COLOR_TEMP],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.start_brightness == 127
        assert change.end_brightness == 255
        assert change.hybrid_direction == "mireds_to_hs"

    def test_ct_to_off(self) -> None:
        """#17: b:50,ct:3000 → b:0 - fade to off."""
        params = FadeParams(brightness_pct=0, transition_ms=1000)
        state = {
            ATTR_BRIGHTNESS: 127,
            HA_ATTR_COLOR_TEMP_KELVIN: 3000,
            "color_mode": ColorMode.COLOR_TEMP,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.COLOR_TEMP],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.start_brightness == 127
        assert change.end_brightness == 0

    def test_ct_to_off_with_ct_change(self) -> None:
        """#19: b:50,ct:3000 → b:0,ct:6000 - fade to off with CT change."""
        params = FadeParams(
            brightness_pct=0,
            color_temp_kelvin=6000,
            transition_ms=1000,
        )
        state = {
            ATTR_BRIGHTNESS: 127,
            HA_ATTR_COLOR_TEMP_KELVIN: 3000,
            "color_mode": ColorMode.COLOR_TEMP,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.COLOR_TEMP],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.start_brightness == 127
        assert change.end_brightness == 0
        assert change.start_mireds == 333
        assert change.end_mireds == 166

    def test_ct_to_off_with_hybrid(self) -> None:
        """#21: b:50,ct:3000 → b:0,hs:[10,10] - fade to off with hybrid."""
        params = FadeParams(
            brightness_pct=0,
            hs_color=(120.0, 80.0),
            transition_ms=1000,
        )
        state = {
            ATTR_BRIGHTNESS: 127,
            HA_ATTR_COLOR_TEMP_KELVIN: 3000,
            "color_mode": ColorMode.COLOR_TEMP,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS, ColorMode.COLOR_TEMP],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.start_brightness == 127
        assert change.end_brightness == 0
        assert change.hybrid_direction == "mireds_to_hs"


class TestFromParamOverrides:
    """Test fades using from: parameter to override current state."""

    def test_from_hs_to_ct_hybrid(self) -> None:
        """#22: Current CT, from: HS → CT target - hybrid HS→CT."""
        params = FadeParams(
            color_temp_kelvin=4000,
            from_hs_color=(240.0, 100.0),  # Blue
            transition_ms=1000,
        )
        # Current state is CT, but from: overrides to HS
        state = {
            ATTR_BRIGHTNESS: 255,
            HA_ATTR_COLOR_TEMP_KELVIN: 3000,
            "color_mode": ColorMode.COLOR_TEMP,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS, ColorMode.COLOR_TEMP],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None
        # Should be hybrid because from: specifies HS
        assert change.hybrid_direction == "hs_to_mireds"
        assert change.start_hs == (240.0, 100.0)

    def test_from_ct_to_hs_hybrid(self) -> None:
        """#23: Current HS, from: CT → HS target - hybrid CT→HS."""
        params = FadeParams(
            hs_color=(120.0, 80.0),
            from_color_temp_kelvin=3000,
            transition_ms=1000,
        )
        # Current state is HS, but from: overrides to CT
        state = {
            ATTR_BRIGHTNESS: 255,
            HA_ATTR_HS_COLOR: (60.0, 50.0),
            "color_mode": ColorMode.HS,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS, ColorMode.COLOR_TEMP],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None
        # Should be hybrid because from: specifies CT
        assert change.hybrid_direction == "mireds_to_hs"
        assert change.start_mireds == 333  # 3000K

    def test_from_brightness_override(self) -> None:
        """from: brightness_pct overrides current brightness."""
        params = FadeParams(
            brightness_pct=100,
            from_brightness_pct=10,
            transition_ms=1000,
        )
        state = {
            ATTR_BRIGHTNESS: 200,  # Current is 200, but from: says 10%
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.start_brightness == 25  # 10% of 255
        assert change.end_brightness == 255

    def test_from_hs_override(self) -> None:
        """from: hs_color overrides current HS color."""
        params = FadeParams(
            hs_color=(240.0, 100.0),
            from_hs_color=(0.0, 100.0),  # Red start
            transition_ms=1000,
        )
        state = {
            ATTR_BRIGHTNESS: 200,
            HA_ATTR_HS_COLOR: (120.0, 50.0),  # Current is green
            "color_mode": ColorMode.HS,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.start_hs == (0.0, 100.0)  # From override
        assert change.end_hs == (240.0, 100.0)

    def test_from_ct_override(self) -> None:
        """from: color_temp_kelvin overrides current CT."""
        params = FadeParams(
            color_temp_kelvin=6000,
            from_color_temp_kelvin=2700,
            transition_ms=1000,
        )
        state = {
            ATTR_BRIGHTNESS: 200,
            HA_ATTR_COLOR_TEMP_KELVIN: 4000,  # Current is 4000K
            "color_mode": ColorMode.COLOR_TEMP,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.COLOR_TEMP],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.start_mireds == 370  # 2700K from override
        assert change.end_mireds == 166  # 6000K


class TestStepGeneration:
    """Test that FadeChange objects generate correct steps."""

    def test_hybrid_hs_to_ct_generates_both_phases(self) -> None:
        """Hybrid HS→CT generates HS steps then CT steps."""
        params = FadeParams(color_temp_kelvin=4000, transition_ms=2000)
        state = {
            ATTR_BRIGHTNESS: 200,
            HA_ATTR_HS_COLOR: (120.0, 80.0),
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS, ColorMode.COLOR_TEMP],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.hybrid_direction == "hs_to_mireds"

        hs_steps = 0
        ct_steps = 0
        while change.has_next():
            step = change.next_step()
            if step.hs_color is not None:
                hs_steps += 1
            if step.color_temp_kelvin is not None:
                ct_steps += 1

        assert hs_steps > 0, "Should have HS steps"
        assert ct_steps > 0, "Should have CT steps"

    def test_hybrid_ct_to_hs_generates_both_phases(self) -> None:
        """Hybrid CT→HS generates CT steps then HS steps."""
        params = FadeParams(hs_color=(120.0, 80.0), transition_ms=2000)
        state = {
            ATTR_BRIGHTNESS: 200,
            HA_ATTR_COLOR_TEMP_KELVIN: 3000,
            "color_mode": ColorMode.COLOR_TEMP,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS, ColorMode.COLOR_TEMP],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None
        assert change.hybrid_direction == "mireds_to_hs"

        hs_steps = 0
        ct_steps = 0
        while change.has_next():
            step = change.next_step()
            if step.hs_color is not None:
                hs_steps += 1
            if step.color_temp_kelvin is not None:
                ct_steps += 1

        assert ct_steps > 0, "Should have CT steps"
        assert hs_steps > 0, "Should have HS steps"

    def test_combined_brightness_and_color_fade(self) -> None:
        """Combined fade generates steps with both attributes."""
        params = FadeParams(
            brightness_pct=100,
            hs_color=(240.0, 100.0),
            transition_ms=1000,
        )
        state = {
            ATTR_BRIGHTNESS: 100,
            HA_ATTR_HS_COLOR: (60.0, 50.0),
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS],
        }

        change = FadeChange.resolve(params, state, min_step_delay_ms=100)

        assert change is not None

        # First step should have both brightness and HS
        step = change.next_step()
        assert step.brightness is not None
        assert step.hs_color is not None
