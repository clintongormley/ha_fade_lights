"""Tests for color parameter parsing and validation."""

from __future__ import annotations

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.fado.const import (
    ATTR_BRIGHTNESS_PCT,
    ATTR_COLOR_TEMP_KELVIN,
    ATTR_HS_COLOR,
    ATTR_RGB_COLOR,
    ATTR_RGBW_COLOR,
    ATTR_RGBWW_COLOR,
    ATTR_TRANSITION,
    ATTR_XY_COLOR,
    DOMAIN,
    SERVICE_FADO,
)


class TestColorParameterValidation:
    """Test that only one color parameter is allowed."""

    async def test_rejects_multiple_color_params(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test service rejects calls with multiple color parameters."""
        with pytest.raises(ServiceValidationError, match="Only one color parameter"):
            await hass.services.async_call(
                DOMAIN,
                SERVICE_FADO,
                {
                    ATTR_HS_COLOR: [200, 80],
                    ATTR_COLOR_TEMP_KELVIN: 4000,
                },
                target={"entity_id": mock_light_entity},
                blocking=True,
            )

    async def test_rejects_hs_and_rgb(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test service rejects hs_color with rgb_color."""
        with pytest.raises(ServiceValidationError, match="Only one color parameter"):
            await hass.services.async_call(
                DOMAIN,
                SERVICE_FADO,
                {
                    ATTR_HS_COLOR: [200, 80],
                    ATTR_RGB_COLOR: [255, 128, 0],
                },
                target={"entity_id": mock_light_entity},
                blocking=True,
            )

    async def test_accepts_single_hs_color(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test service accepts a single hs_color parameter."""
        # Should not raise
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADO,
            {
                ATTR_HS_COLOR: [200, 80],
                ATTR_TRANSITION: 0.1,
            },
            target={"entity_id": mock_light_entity},
            blocking=True,
        )

    async def test_accepts_single_color_temp(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test service accepts a single color_temp_kelvin parameter."""
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADO,
            {
                ATTR_COLOR_TEMP_KELVIN: 4000,
                ATTR_TRANSITION: 0.1,
            },
            target={"entity_id": mock_light_entity},
            blocking=True,
        )

    async def test_accepts_brightness_with_color(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test service accepts brightness_pct alongside a color parameter."""
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADO,
            {
                ATTR_BRIGHTNESS_PCT: 80,
                ATTR_HS_COLOR: [200, 80],
                ATTR_TRANSITION: 0.1,
            },
            target={"entity_id": mock_light_entity},
            blocking=True,
        )


class TestColorConversions:
    """Test color format conversions to internal representations."""

    async def test_rgb_converts_to_hs(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test rgb_color is converted to hs_color internally."""
        from custom_components.fado.fade_params import FadeParams

        params = FadeParams.from_service_data({ATTR_RGB_COLOR: [255, 0, 0]})

        # Pure red should be hue ~0, saturation 100
        assert params.hs_color is not None
        assert abs(params.hs_color[0] - 0) < 1  # hue ~0
        assert abs(params.hs_color[1] - 100) < 1  # saturation 100

    async def test_rgbw_converts_to_hs(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test rgbw_color is converted to hs_color internally."""
        from custom_components.fado.fade_params import FadeParams

        # Green with some white
        params = FadeParams.from_service_data({ATTR_RGBW_COLOR: [0, 255, 0, 50]})

        assert params.hs_color is not None
        # Green is hue ~120
        assert 115 < params.hs_color[0] < 125

    async def test_rgbww_converts_to_hs(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test rgbww_color is converted to hs_color internally."""
        from custom_components.fado.fade_params import FadeParams

        # Blue with some whites
        params = FadeParams.from_service_data({ATTR_RGBWW_COLOR: [0, 0, 255, 30, 20]})

        assert params.hs_color is not None
        # Blue is hue ~240
        assert 235 < params.hs_color[0] < 245

    async def test_xy_converts_to_hs(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test xy_color is converted to hs_color internally."""
        from custom_components.fado.fade_params import FadeParams

        # Red-ish xy coordinates
        params = FadeParams.from_service_data({ATTR_XY_COLOR: [0.64, 0.33]})

        assert params.hs_color is not None
        # Should be reddish (hue near 0 or 360)
        assert params.hs_color[0] < 30 or params.hs_color[0] > 330

    async def test_hs_passes_through(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test hs_color passes through unchanged."""
        from custom_components.fado.fade_params import FadeParams

        params = FadeParams.from_service_data({ATTR_HS_COLOR: [180, 75]})

        assert params.hs_color == (180, 75)

    async def test_color_temp_stored_as_kelvin(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test color_temp_kelvin is stored directly (no conversion)."""
        from custom_components.fado.fade_params import FadeParams

        params = FadeParams.from_service_data({ATTR_COLOR_TEMP_KELVIN: 4000})

        assert params.color_temp_kelvin == 4000
        assert params.hs_color is None

    async def test_no_color_returns_none(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test no color params returns None for both."""
        from custom_components.fado.fade_params import FadeParams

        params = FadeParams.from_service_data({ATTR_BRIGHTNESS_PCT: 50})

        assert params.hs_color is None
        assert params.color_temp_kelvin is None


class TestFromParameter:
    """Test the from: parameter for specifying starting values."""

    async def test_from_brightness(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test from: parameter with brightness_pct."""
        from custom_components.fado.const import ATTR_FROM
        from custom_components.fado.fade_params import FadeParams

        params = FadeParams.from_service_data(
            {
                ATTR_BRIGHTNESS_PCT: 100,
                ATTR_FROM: {ATTR_BRIGHTNESS_PCT: 0},
            }
        )

        assert params.brightness_pct == 100
        assert params.from_brightness_pct == 0

    async def test_from_hs_color(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test from: parameter with hs_color."""
        from custom_components.fado.const import ATTR_FROM
        from custom_components.fado.fade_params import FadeParams

        params = FadeParams.from_service_data(
            {
                ATTR_HS_COLOR: [200, 80],
                ATTR_FROM: {ATTR_HS_COLOR: [0, 0]},
            }
        )

        assert params.hs_color == (200, 80)
        assert params.from_hs_color == (0, 0)

    async def test_from_color_temp(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test from: parameter with color_temp_kelvin."""
        from custom_components.fado.const import ATTR_FROM
        from custom_components.fado.fade_params import FadeParams

        params = FadeParams.from_service_data(
            {
                ATTR_COLOR_TEMP_KELVIN: 4000,
                ATTR_FROM: {ATTR_COLOR_TEMP_KELVIN: 2700},
            }
        )

        assert params.color_temp_kelvin == 4000
        assert params.from_color_temp_kelvin == 2700

    async def test_from_rgb_converts_to_hs(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test from: parameter converts rgb_color to hs."""
        from custom_components.fado.const import ATTR_FROM
        from custom_components.fado.fade_params import FadeParams

        params = FadeParams.from_service_data(
            {
                ATTR_HS_COLOR: [200, 80],
                ATTR_FROM: {ATTR_RGB_COLOR: [255, 0, 0]},  # Red
            }
        )

        assert params.from_hs_color is not None
        assert abs(params.from_hs_color[0] - 0) < 1  # hue ~0 (red)

    async def test_from_validates_single_color(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test from: parameter validates only one color param."""
        with pytest.raises(ServiceValidationError, match="Only one color parameter"):
            await hass.services.async_call(
                DOMAIN,
                SERVICE_FADO,
                {
                    ATTR_HS_COLOR: [200, 80],
                    "from": {
                        ATTR_HS_COLOR: [0, 0],
                        ATTR_COLOR_TEMP_KELVIN: 2700,
                    },
                },
                target={"entity_id": mock_light_entity},
                blocking=True,
            )


class TestValueRangeValidation:
    """Test parameter value range validation."""

    async def test_rejects_invalid_hue(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test service rejects hue outside 0-360."""
        with pytest.raises(ServiceValidationError, match="[Hh]ue"):
            await hass.services.async_call(
                DOMAIN,
                SERVICE_FADO,
                {
                    ATTR_HS_COLOR: [400, 50],  # Invalid hue
                },
                target={"entity_id": mock_light_entity},
                blocking=True,
            )

    async def test_rejects_invalid_saturation(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test service rejects saturation outside 0-100."""
        with pytest.raises(ServiceValidationError, match="[Ss]aturation"):
            await hass.services.async_call(
                DOMAIN,
                SERVICE_FADO,
                {
                    ATTR_HS_COLOR: [200, 150],  # Invalid saturation
                },
                target={"entity_id": mock_light_entity},
                blocking=True,
            )

    async def test_rejects_invalid_rgb(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test service rejects RGB values outside 0-255."""
        with pytest.raises(ServiceValidationError, match="RGB"):
            await hass.services.async_call(
                DOMAIN,
                SERVICE_FADO,
                {
                    ATTR_RGB_COLOR: [300, 128, 0],  # Invalid R
                },
                target={"entity_id": mock_light_entity},
                blocking=True,
            )

    async def test_rejects_invalid_color_temp(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test service rejects color temp outside reasonable range."""
        with pytest.raises(ServiceValidationError, match="[Cc]olor temp"):
            await hass.services.async_call(
                DOMAIN,
                SERVICE_FADO,
                {
                    ATTR_COLOR_TEMP_KELVIN: 500,  # Too low
                },
                target={"entity_id": mock_light_entity},
                blocking=True,
            )

    async def test_accepts_valid_ranges(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test service accepts valid parameter ranges."""
        # Should not raise
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADO,
            {
                ATTR_HS_COLOR: [360, 100],  # Max valid values
                ATTR_TRANSITION: 0.1,
            },
            target={"entity_id": mock_light_entity},
            blocking=True,
        )
