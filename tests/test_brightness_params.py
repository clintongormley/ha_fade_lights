"""Tests for brightness parameter (raw 1-255) handling."""

from __future__ import annotations

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.fado.const import (
    ATTR_BRIGHTNESS,
    ATTR_BRIGHTNESS_PCT,
    ATTR_FROM,
    ATTR_TRANSITION,
    DOMAIN,
    SERVICE_FADE_LIGHTS,
)
from custom_components.fado.fade_params import FadeParams


class TestBrightnessAndBrightnessPctMutualExclusion:
    """Test that brightness and brightness_pct cannot be specified together."""

    async def test_rejects_both_brightness_and_brightness_pct(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test service rejects calls with both brightness and brightness_pct."""
        with pytest.raises(
            ServiceValidationError, match="Cannot specify both brightness_pct and brightness"
        ):
            await hass.services.async_call(
                DOMAIN,
                SERVICE_FADE_LIGHTS,
                {
                    ATTR_BRIGHTNESS_PCT: 50,
                    ATTR_BRIGHTNESS: 128,
                },
                target={"entity_id": mock_light_entity},
                blocking=True,
            )

    async def test_rejects_both_from_brightness_and_from_brightness_pct(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test service rejects from: with both brightness and brightness_pct."""
        with pytest.raises(
            ServiceValidationError,
            match="Cannot specify both brightness_pct and brightness in 'from:'",
        ):
            await hass.services.async_call(
                DOMAIN,
                SERVICE_FADE_LIGHTS,
                {
                    ATTR_BRIGHTNESS_PCT: 100,
                    ATTR_FROM: {
                        ATTR_BRIGHTNESS_PCT: 0,
                        ATTR_BRIGHTNESS: 0,
                    },
                },
                target={"entity_id": mock_light_entity},
                blocking=True,
            )

    def test_rejects_both_via_from_service_data(self) -> None:
        """Test from_service_data rejects both brightness params."""
        with pytest.raises(
            ServiceValidationError, match="Cannot specify both brightness_pct and brightness"
        ):
            FadeParams.from_service_data(
                {
                    ATTR_BRIGHTNESS_PCT: 50,
                    ATTR_BRIGHTNESS: 128,
                }
            )

    def test_rejects_both_in_from_via_from_service_data(self) -> None:
        """Test from_service_data rejects both brightness params in from:."""
        with pytest.raises(
            ServiceValidationError,
            match="Cannot specify both brightness_pct and brightness in 'from:'",
        ):
            FadeParams.from_service_data(
                {
                    ATTR_BRIGHTNESS_PCT: 100,
                    ATTR_FROM: {
                        ATTR_BRIGHTNESS_PCT: 0,
                        ATTR_BRIGHTNESS: 0,
                    },
                }
            )


class TestBrightnessStorage:
    """Test that brightness raw value is stored correctly in FadeParams."""

    def test_stores_brightness_raw_value(self) -> None:
        """Test brightness value is stored in FadeParams."""
        params = FadeParams.from_service_data({ATTR_BRIGHTNESS: 128})

        assert params.brightness == 128
        assert params.brightness_pct is None

    def test_stores_brightness_pct_separately(self) -> None:
        """Test brightness_pct is stored when brightness is not provided."""
        params = FadeParams.from_service_data({ATTR_BRIGHTNESS_PCT: 50})

        assert params.brightness_pct == 50
        assert params.brightness is None

    def test_stores_from_brightness_raw_value(self) -> None:
        """Test from_brightness value is stored in FadeParams."""
        params = FadeParams.from_service_data(
            {
                ATTR_BRIGHTNESS: 255,
                ATTR_FROM: {ATTR_BRIGHTNESS: 1},
            }
        )

        assert params.brightness == 255
        assert params.from_brightness == 1
        assert params.from_brightness_pct is None

    def test_stores_from_brightness_pct_separately(self) -> None:
        """Test from_brightness_pct is stored when from_brightness is not provided."""
        params = FadeParams.from_service_data(
            {
                ATTR_BRIGHTNESS_PCT: 100,
                ATTR_FROM: {ATTR_BRIGHTNESS_PCT: 0},
            }
        )

        assert params.from_brightness_pct == 0
        assert params.from_brightness is None


class TestBrightnessRangeValidation:
    """Test brightness value range validation (1-255)."""

    async def test_rejects_brightness_below_1(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test service rejects brightness below 1."""
        with pytest.raises(ServiceValidationError, match="[Bb]rightness.*between 1 and 255"):
            await hass.services.async_call(
                DOMAIN,
                SERVICE_FADE_LIGHTS,
                {
                    ATTR_BRIGHTNESS: 0,
                },
                target={"entity_id": mock_light_entity},
                blocking=True,
            )

    async def test_rejects_brightness_above_255(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test service rejects brightness above 255."""
        with pytest.raises(ServiceValidationError, match="[Bb]rightness.*between 1 and 255"):
            await hass.services.async_call(
                DOMAIN,
                SERVICE_FADE_LIGHTS,
                {
                    ATTR_BRIGHTNESS: 256,
                },
                target={"entity_id": mock_light_entity},
                blocking=True,
            )

    async def test_rejects_from_brightness_below_1(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test service rejects from: brightness below 1."""
        with pytest.raises(ServiceValidationError, match="from:.*[Bb]rightness.*between 1 and 255"):
            await hass.services.async_call(
                DOMAIN,
                SERVICE_FADE_LIGHTS,
                {
                    ATTR_BRIGHTNESS: 255,
                    ATTR_FROM: {ATTR_BRIGHTNESS: 0},
                },
                target={"entity_id": mock_light_entity},
                blocking=True,
            )

    async def test_accepts_brightness_at_minimum(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test service accepts brightness at minimum value (1)."""
        # Should not raise
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_BRIGHTNESS: 1,
                ATTR_TRANSITION: 0.1,
            },
            target={"entity_id": mock_light_entity},
            blocking=True,
        )

    async def test_accepts_brightness_at_maximum(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        mock_light_entity: str,
    ) -> None:
        """Test service accepts brightness at maximum value (255)."""
        # Should not raise
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_BRIGHTNESS: 255,
                ATTR_TRANSITION: 0.1,
            },
            target={"entity_id": mock_light_entity},
            blocking=True,
        )

    def test_from_service_data_rejects_brightness_below_1(self) -> None:
        """Test from_service_data rejects brightness below 1."""
        with pytest.raises(ServiceValidationError, match="[Bb]rightness.*between 1 and 255"):
            FadeParams.from_service_data({ATTR_BRIGHTNESS: 0})

    def test_from_service_data_rejects_brightness_above_255(self) -> None:
        """Test from_service_data rejects brightness above 255."""
        with pytest.raises(ServiceValidationError, match="[Bb]rightness.*between 1 and 255"):
            FadeParams.from_service_data({ATTR_BRIGHTNESS: 256})

    def test_from_service_data_accepts_valid_brightness(self) -> None:
        """Test from_service_data accepts valid brightness values."""
        params = FadeParams.from_service_data({ATTR_BRIGHTNESS: 128})
        assert params.brightness == 128


class TestHasTargetWithBrightness:
    """Test has_target() and has_from_target() with brightness field."""

    def test_has_target_true_with_brightness(self) -> None:
        """Test has_target returns True when brightness is set."""
        params = FadeParams(brightness=128)
        assert params.has_target() is True

    def test_has_target_true_with_brightness_pct(self) -> None:
        """Test has_target returns True when brightness_pct is set."""
        params = FadeParams(brightness_pct=50)
        assert params.has_target() is True

    def test_has_target_false_when_no_targets(self) -> None:
        """Test has_target returns False when no targets are set."""
        params = FadeParams()
        assert params.has_target() is False

    def test_has_from_target_true_with_from_brightness(self) -> None:
        """Test has_from_target returns True when from_brightness is set."""
        params = FadeParams(from_brightness=1)
        assert params.has_from_target() is True

    def test_has_from_target_true_with_from_brightness_pct(self) -> None:
        """Test has_from_target returns True when from_brightness_pct is set."""
        params = FadeParams(from_brightness_pct=0)
        assert params.has_from_target() is True

    def test_has_from_target_false_when_no_from_targets(self) -> None:
        """Test has_from_target returns False when no from targets are set."""
        params = FadeParams(brightness=128)
        assert params.has_from_target() is False


class TestBrightnessAsKnownParameter:
    """Test that brightness is recognized as a valid parameter."""

    def test_brightness_is_valid_main_param(self) -> None:
        """Test brightness is accepted as a main parameter."""
        # Should not raise ServiceValidationError for unknown parameter
        params = FadeParams.from_service_data({ATTR_BRIGHTNESS: 128})
        assert params.brightness == 128

    def test_brightness_is_valid_from_param(self) -> None:
        """Test brightness is accepted in from: block."""
        params = FadeParams.from_service_data(
            {
                ATTR_BRIGHTNESS: 255,
                ATTR_FROM: {ATTR_BRIGHTNESS: 1},
            }
        )
        assert params.from_brightness == 1
