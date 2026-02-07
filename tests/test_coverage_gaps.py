"""Tests to cover specific code paths identified from coverage analysis."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.components.light import ATTR_BRIGHTNESS, ATTR_SUPPORTED_COLOR_MODES
from homeassistant.components.light.const import ColorMode
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.fado.const import (
    ATTR_BRIGHTNESS_PCT,
    ATTR_FROM,
    ATTR_RGBW_COLOR,
    ATTR_RGBWW_COLOR,
    ATTR_XY_COLOR,
    DOMAIN,
    SERVICE_FADO,
)
from custom_components.fado.expected_state import ExpectedValues
from custom_components.fado.fade_change import FadeChange
from custom_components.fado.fade_params import FadeParams


class TestNoFadeParameters:
    """Test early return when no fade parameters are specified (lines 171-172)."""

    async def test_no_fade_params_returns_early(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test service returns early when no fade parameters specified."""
        with patch(
            "custom_components.fado.coordinator.FadeCoordinator._fade_light",
            new_callable=AsyncMock,
        ) as mock_fade_light:
            # Call with only target, no brightness, colors, or from params
            await hass.services.async_call(
                DOMAIN,
                SERVICE_FADO,
                {
                    # No brightness_pct, no color params, no from params
                },
                target={"entity_id": mock_light_entity},
                blocking=True,
            )
            await hass.async_block_till_done()

            # _fade_light should NOT be called since there's nothing to fade
            assert mock_fade_light.call_count == 0


class TestNonDimmableLightNoTarget:
    """Test non-dimmable light with no brightness target (line 610)."""

    def test_non_dimmable_no_brightness_target_returns_none(self) -> None:
        """Test FadeChange.resolve returns None for non-dimmable with no brightness target."""
        # State attributes for a non-dimmable light (ONOFF only)
        state_attributes = {
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.ONOFF],
            # No brightness attribute
        }

        # FadeParams with only color target, no brightness
        fade_params = FadeParams(
            brightness_pct=None,
            hs_color=(200.0, 80.0),  # Color target but no brightness
        )

        result = FadeChange.resolve(fade_params, state_attributes, 50)

        # Should return None because light can't dim and no brightness target
        assert result is None


class TestMiredsBoundaryFallback:
    """Test start_mireds boundary fallback logic (lines 733, 735)."""

    def test_only_min_mireds_available(self) -> None:
        """Test fallback to min_mireds when only min bound exists.

        When only max_color_temp_kelvin is available (no min), we only have
        min_mireds (from max kelvin). The code should use min_mireds as start.
        """
        # State with only max_color_temp_kelvin (gives us only min_mireds)
        state_attributes = {
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.COLOR_TEMP],
            ATTR_BRIGHTNESS: 200,
            "max_color_temp_kelvin": 6500,  # max kelvin = min mireds (~154)
            # No min_color_temp_kelvin - this means no max_mireds
        }

        # Target color temp with no starting color temp in state
        fade_params = FadeParams(
            color_temp_kelvin=4000,  # ~250 mireds
        )

        result = FadeChange.resolve(fade_params, state_attributes, 50)

        # Should return a FadeChange (not None)
        assert result is not None

    def test_only_max_mireds_available(self) -> None:
        """Test fallback to max_mireds when only max bound exists.

        When only min_color_temp_kelvin is available (no max), we only have
        max_mireds (from min kelvin). The code should use max_mireds as start.
        """
        # State with only min_color_temp_kelvin (gives us only max_mireds)
        state_attributes = {
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.COLOR_TEMP],
            ATTR_BRIGHTNESS: 200,
            "min_color_temp_kelvin": 2000,  # min kelvin = max mireds (500)
            # No max_color_temp_kelvin - this means no min_mireds
        }

        # Target color temp with no starting color temp in state
        fade_params = FadeParams(
            color_temp_kelvin=4000,  # ~250 mireds
        )

        result = FadeChange.resolve(fade_params, state_attributes, 50)

        # Should return a FadeChange (not None)
        assert result is not None


class TestExpectedValuesStr:
    """Test ExpectedValues __str__ method (lines 35-42)."""

    def test_str_with_brightness_only(self) -> None:
        """Test string representation with only brightness."""
        ev = ExpectedValues(brightness=128)
        result = str(ev)
        assert "brightness=128" in result
        assert "hs_color" not in result
        assert "color_temp_kelvin" not in result

    def test_str_with_hs_color_only(self) -> None:
        """Test string representation with only hs_color."""
        ev = ExpectedValues(hs_color=(200.0, 80.0))
        result = str(ev)
        assert "hs_color=(200.0, 80.0)" in result
        assert "brightness" not in result
        assert "color_temp_kelvin" not in result

    def test_str_with_color_temp_only(self) -> None:
        """Test string representation with only color_temp_kelvin."""
        ev = ExpectedValues(color_temp_kelvin=4000)
        result = str(ev)
        assert "color_temp_kelvin=4000" in result
        assert "brightness" not in result
        assert "hs_color" not in result

    def test_str_with_all_values(self) -> None:
        """Test string representation with all values."""
        ev = ExpectedValues(
            brightness=200,
            hs_color=(120.0, 50.0),
            color_temp_kelvin=3000,
        )
        result = str(ev)
        assert "brightness=200" in result
        assert "hs_color=(120.0, 50.0)" in result
        assert "color_temp_kelvin=3000" in result

    def test_str_empty(self) -> None:
        """Test string representation with no values."""
        ev = ExpectedValues()
        result = str(ev)
        assert "empty" in result


class TestFadeParamsValidationErrors:
    """Test FadeParams validation error paths."""

    def test_unknown_top_level_parameter(self) -> None:
        """Test error for unknown top-level parameter (line 162)."""
        with pytest.raises(ServiceValidationError, match="Unknown parameter"):
            FadeParams.from_service_data(
                {
                    ATTR_BRIGHTNESS_PCT: 50,
                    "invalid_top_level_param": "value",  # Unknown param at top level
                }
            )

    def test_unknown_from_parameter(self) -> None:
        """Test error for unknown parameter in 'from' dict (line 173)."""
        with pytest.raises(ServiceValidationError, match="Unknown parameter.*'from'"):
            FadeParams.from_service_data(
                {
                    ATTR_BRIGHTNESS_PCT: 50,
                    ATTR_FROM: {
                        ATTR_BRIGHTNESS_PCT: 0,
                        "invalid_param": "value",  # Unknown param in from
                    },
                }
            )

    def test_brightness_out_of_range(self) -> None:
        """Test error for brightness outside 0-100 (line 195)."""
        with pytest.raises(ServiceValidationError, match="Brightness must be between"):
            FadeParams.from_service_data(
                {
                    ATTR_BRIGHTNESS_PCT: 150,  # > 100 is invalid
                }
            )

    def test_rgbw_value_out_of_range(self) -> None:
        """Test error for RGBW value outside 0-255 (line 224)."""
        with pytest.raises(ServiceValidationError, match="RGBW"):
            FadeParams.from_service_data(
                {
                    ATTR_RGBW_COLOR: [100, 200, 300, 50],  # 300 is invalid
                }
            )

    def test_rgbww_value_out_of_range(self) -> None:
        """Test error for RGBWW value outside 0-255 (line 232)."""
        with pytest.raises(ServiceValidationError, match="RGBWW"):
            FadeParams.from_service_data(
                {
                    ATTR_RGBWW_COLOR: [100, 200, 50, 300, 50],  # 300 is invalid
                }
            )

    def test_xy_value_out_of_range(self) -> None:
        """Test error for XY value outside 0-1 (line 240)."""
        with pytest.raises(ServiceValidationError, match="XY"):
            FadeParams.from_service_data(
                {
                    ATTR_XY_COLOR: [0.5, 1.5],  # 1.5 is invalid
                }
            )

    def test_rgbw_negative_value(self) -> None:
        """Test error for negative RGBW value."""
        with pytest.raises(ServiceValidationError, match="RGBW"):
            FadeParams.from_service_data(
                {
                    ATTR_RGBW_COLOR: [-1, 200, 100, 50],  # -1 is invalid
                }
            )

    def test_rgbww_negative_value(self) -> None:
        """Test error for negative RGBWW value."""
        with pytest.raises(ServiceValidationError, match="RGBWW"):
            FadeParams.from_service_data(
                {
                    ATTR_RGBWW_COLOR: [100, -10, 50, 50, 50],  # -10 is invalid
                }
            )

    def test_xy_negative_value(self) -> None:
        """Test error for negative XY value."""
        with pytest.raises(ServiceValidationError, match="XY"):
            FadeParams.from_service_data(
                {
                    ATTR_XY_COLOR: [-0.1, 0.5],  # -0.1 is invalid
                }
            )
