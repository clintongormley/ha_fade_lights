"""Tests for resolver helper functions."""

from __future__ import annotations

from homeassistant.components.light import ATTR_BRIGHTNESS
from homeassistant.components.light import ATTR_COLOR_TEMP_KELVIN as HA_ATTR_COLOR_TEMP_KELVIN
from homeassistant.components.light import ATTR_HS_COLOR as HA_ATTR_HS_COLOR

from custom_components.fado.fade_change import (
    _resolve_end_brightness,
    _resolve_start_brightness,
    _resolve_start_hs,
    _resolve_start_mireds,
)
from custom_components.fado.fade_params import FadeParams


class TestResolveStartBrightness:
    """Test _resolve_start_brightness function."""

    def test_uses_from_brightness_pct_when_specified(self) -> None:
        """Test that from_brightness_pct takes precedence over state."""
        params = FadeParams(from_brightness_pct=50)
        state = {ATTR_BRIGHTNESS: 200}

        result = _resolve_start_brightness(params, state, min_brightness=1)

        # 50% of 255 = 127.5, truncated to 127
        assert result == 127

    def test_uses_current_state_when_no_from_brightness(self) -> None:
        """Test that current state is used when from_brightness_pct is None."""
        params = FadeParams()
        state = {ATTR_BRIGHTNESS: 180}

        result = _resolve_start_brightness(params, state, min_brightness=1)

        assert result == 180

    def test_returns_min_brightness_when_light_off(self) -> None:
        """Test that min_brightness is returned when light is off (no brightness in state)."""
        params = FadeParams()
        state = {}

        # With default min_brightness=1, start from 1 not 0
        result = _resolve_start_brightness(params, state, min_brightness=1)
        assert result == 1

        # With min_brightness=10, start from 10
        result = _resolve_start_brightness(params, state, min_brightness=10)
        assert result == 10

    def test_from_brightness_pct_zero_clamped_to_min(self) -> None:
        """Test handling of 0% brightness (clamped to min_brightness)."""
        params = FadeParams(from_brightness_pct=0)
        state = {ATTR_BRIGHTNESS: 255}

        # 0% converts to 0, but gets clamped to min_brightness
        result = _resolve_start_brightness(params, state, min_brightness=1)
        assert result == 1

        result = _resolve_start_brightness(params, state, min_brightness=10)
        assert result == 10

    def test_from_brightness_pct_100(self) -> None:
        """Test handling of 100% brightness."""
        params = FadeParams(from_brightness_pct=100)
        state = {ATTR_BRIGHTNESS: 50}

        result = _resolve_start_brightness(params, state, min_brightness=1)

        assert result == 255

    def test_state_brightness_none_returns_min(self) -> None:
        """Test handling of None brightness in state (light off)."""
        params = FadeParams()
        state = {ATTR_BRIGHTNESS: None}

        # When light is off (brightness is None), return min_brightness
        result = _resolve_start_brightness(params, state, min_brightness=1)
        assert result == 1

        result = _resolve_start_brightness(params, state, min_brightness=5)
        assert result == 5

    def test_from_brightness_raw_used_directly(self) -> None:
        """Test that from_brightness (raw) is used directly."""
        params = FadeParams(from_brightness=150)
        state = {ATTR_BRIGHTNESS: 200}

        result = _resolve_start_brightness(params, state, min_brightness=1)

        assert result == 150  # Raw value used directly

    def test_from_brightness_raw_clamped_to_min(self) -> None:
        """Test that from_brightness (raw) is clamped to min_brightness."""
        params = FadeParams(from_brightness=5)
        state = {ATTR_BRIGHTNESS: 200}

        result = _resolve_start_brightness(params, state, min_brightness=10)

        assert result == 10  # Clamped to min_brightness

    def test_from_brightness_pct_1_special_case(self) -> None:
        """Test from_brightness_pct=1 maps to min_brightness when higher."""
        params = FadeParams(from_brightness_pct=1)
        state = {ATTR_BRIGHTNESS: 200}

        # Normal conversion: 1% of 255 = 2.55 -> truncated to 2
        # With min_brightness=10, use min_brightness instead
        result = _resolve_start_brightness(params, state, min_brightness=10)
        assert result == 10

        # With min_brightness=1, use normal conversion (but clamp to min 1)
        result = _resolve_start_brightness(params, state, min_brightness=1)
        assert result == 2


class TestResolveEndBrightness:
    """Test _resolve_end_brightness function."""

    def test_uses_brightness_pct_when_specified(self) -> None:
        """Test that brightness_pct is converted to 0-255 scale."""
        params = FadeParams(brightness_pct=75)

        result = _resolve_end_brightness(params, min_brightness=1)

        # 75% of 255 = 191.25, truncated to 191
        assert result == 191

    def test_returns_none_when_not_specified(self) -> None:
        """Test that None is returned when brightness_pct is not set."""
        params = FadeParams()

        result = _resolve_end_brightness(params, min_brightness=1)

        assert result is None

    def test_brightness_pct_zero_not_clamped(self) -> None:
        """Test handling of 0% end brightness (not clamped - allows turn off)."""
        params = FadeParams(brightness_pct=0)

        # 0 is a special case - not clamped (allows fading to off)
        result = _resolve_end_brightness(params, min_brightness=1)
        assert result == 0

        result = _resolve_end_brightness(params, min_brightness=10)
        assert result == 0  # Still 0, not clamped

    def test_brightness_pct_100(self) -> None:
        """Test handling of 100% end brightness."""
        params = FadeParams(brightness_pct=100)

        result = _resolve_end_brightness(params, min_brightness=1)

        assert result == 255

    def test_state_ignored(self) -> None:
        """Test that state is ignored for end brightness (unlike start).

        This function only uses params, not state - demonstrating the design
        that end brightness is explicitly specified, not derived from state.
        """
        params = FadeParams()

        result = _resolve_end_brightness(params, min_brightness=1)

        # Should return None since no brightness_pct specified
        assert result is None

    def test_brightness_raw_used_directly(self) -> None:
        """Test that brightness (raw) is used directly."""
        params = FadeParams(brightness=150)

        result = _resolve_end_brightness(params, min_brightness=1)

        assert result == 150  # Raw value used directly

    def test_brightness_raw_clamped_to_min(self) -> None:
        """Test that brightness (raw) is clamped to min_brightness."""
        params = FadeParams(brightness=5)

        result = _resolve_end_brightness(params, min_brightness=10)

        assert result == 10  # Clamped to min_brightness

    def test_brightness_pct_1_special_case(self) -> None:
        """Test brightness_pct=1 maps to min_brightness when higher."""
        params = FadeParams(brightness_pct=1)

        # Normal conversion: 1% of 255 = 2.55 -> truncated to 2
        # With min_brightness=10, use min_brightness instead
        result = _resolve_end_brightness(params, min_brightness=10)
        assert result == 10

        # With min_brightness=1, use normal conversion (but clamp to min 1)
        result = _resolve_end_brightness(params, min_brightness=1)
        assert result == 2

    def test_low_brightness_clamped_to_min(self) -> None:
        """Test that low but non-zero brightness is clamped to min_brightness."""
        params = FadeParams(brightness_pct=2)  # 2% = 5

        result = _resolve_end_brightness(params, min_brightness=10)

        assert result == 10  # Clamped to min_brightness


class TestResolveStartHs:
    """Test _resolve_start_hs function."""

    def test_uses_from_hs_color_when_specified(self) -> None:
        """Test that from_hs_color takes precedence over state."""
        params = FadeParams(from_hs_color=(120.0, 80.0))
        state = {HA_ATTR_HS_COLOR: (240.0, 50.0)}

        result = _resolve_start_hs(params, state)

        assert result == (120.0, 80.0)

    def test_uses_current_state_when_no_from_hs(self) -> None:
        """Test that current state is used when from_hs_color is None."""
        params = FadeParams()
        state = {HA_ATTR_HS_COLOR: (30.0, 100.0)}

        result = _resolve_start_hs(params, state)

        assert result == (30.0, 100.0)

    def test_returns_none_when_both_missing(self) -> None:
        """Test that None is returned when neither source has HS color."""
        params = FadeParams()
        state = {}

        result = _resolve_start_hs(params, state)

        assert result is None

    def test_edge_case_hue_zero(self) -> None:
        """Test handling of hue=0 (red)."""
        params = FadeParams(from_hs_color=(0.0, 100.0))
        state = {}

        result = _resolve_start_hs(params, state)

        assert result == (0.0, 100.0)

    def test_edge_case_saturation_zero(self) -> None:
        """Test handling of saturation=0 (white)."""
        params = FadeParams(from_hs_color=(180.0, 0.0))
        state = {}

        result = _resolve_start_hs(params, state)

        assert result == (180.0, 0.0)

    def test_state_returns_list_converted_to_tuple(self) -> None:
        """Test that list from state is returned as-is (caller handles conversion)."""
        # Note: The resolver returns the raw state value; conversion to tuple
        # happens elsewhere if needed
        params = FadeParams()
        state = {HA_ATTR_HS_COLOR: [60.0, 75.0]}

        result = _resolve_start_hs(params, state)

        # Returns the list as-is
        assert result == [60.0, 75.0]


class TestResolveStartMireds:
    """Test _resolve_start_mireds function.

    This function takes kelvin from FadeParams and state, but returns mireds
    for use with FadeChange which uses mireds internally for linear interpolation.
    """

    def test_uses_from_color_temp_kelvin_when_specified(self) -> None:
        """Test that from_color_temp_kelvin takes precedence over state."""
        params = FadeParams(from_color_temp_kelvin=4000)  # 250 mireds
        state = {HA_ATTR_COLOR_TEMP_KELVIN: 2500}  # 400 mireds

        result = _resolve_start_mireds(params, state)

        assert result == 250

    def test_uses_current_state_when_no_from_kelvin(self) -> None:
        """Test that current state is used when from_color_temp_kelvin is None."""
        params = FadeParams()
        state = {HA_ATTR_COLOR_TEMP_KELVIN: 3003}  # ~333 mireds

        result = _resolve_start_mireds(params, state)

        assert result == 333

    def test_returns_none_when_both_missing(self) -> None:
        """Test that None is returned when neither source has color temp."""
        params = FadeParams()
        state = {}

        result = _resolve_start_mireds(params, state)

        assert result is None

    def test_warm_white_kelvin(self) -> None:
        """Test handling of warm white color temp (low kelvin)."""
        params = FadeParams(from_color_temp_kelvin=2000)  # 500 mireds
        state = {}

        result = _resolve_start_mireds(params, state)

        assert result == 500

    def test_cool_white_kelvin(self) -> None:
        """Test handling of cool white color temp (high kelvin)."""
        params = FadeParams(from_color_temp_kelvin=6500)  # 153 mireds (int(1_000_000/6500))
        state = {}

        result = _resolve_start_mireds(params, state)

        assert result == 153

    def test_state_with_kelvin(self) -> None:
        """Test handling of kelvin from state."""
        params = FadeParams()
        state = {HA_ATTR_COLOR_TEMP_KELVIN: 3500}  # 285 mireds (int(1_000_000/3500))

        result = _resolve_start_mireds(params, state)

        assert result == 285
