"""Tests for color parameter parsing and validation."""

from __future__ import annotations

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.fade_lights.const import (
    ATTR_BRIGHTNESS_PCT,
    ATTR_COLOR_TEMP_KELVIN,
    ATTR_HS_COLOR,
    ATTR_RGB_COLOR,
    ATTR_RGBW_COLOR,
    ATTR_RGBWW_COLOR,
    ATTR_TRANSITION,
    ATTR_XY_COLOR,
    DOMAIN,
    SERVICE_FADE_LIGHTS,
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
                SERVICE_FADE_LIGHTS,
                {
                    "entity_id": mock_light_entity,
                    ATTR_HS_COLOR: [200, 80],
                    ATTR_COLOR_TEMP_KELVIN: 4000,
                },
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
                SERVICE_FADE_LIGHTS,
                {
                    "entity_id": mock_light_entity,
                    ATTR_HS_COLOR: [200, 80],
                    ATTR_RGB_COLOR: [255, 128, 0],
                },
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
            SERVICE_FADE_LIGHTS,
            {
                "entity_id": mock_light_entity,
                ATTR_HS_COLOR: [200, 80],
                ATTR_TRANSITION: 0.1,
            },
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
            SERVICE_FADE_LIGHTS,
            {
                "entity_id": mock_light_entity,
                ATTR_COLOR_TEMP_KELVIN: 4000,
                ATTR_TRANSITION: 0.1,
            },
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
            SERVICE_FADE_LIGHTS,
            {
                "entity_id": mock_light_entity,
                ATTR_BRIGHTNESS_PCT: 80,
                ATTR_HS_COLOR: [200, 80],
                ATTR_TRANSITION: 0.1,
            },
            blocking=True,
        )