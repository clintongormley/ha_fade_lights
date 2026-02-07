"""Tests for _apply_step function."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.fado.const import DOMAIN
from custom_components.fado.coordinator import FadeCoordinator
from custom_components.fado.fade_change import FadeStep


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance with FadeCoordinator."""
    hass = MagicMock()
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    return hass


@pytest.fixture
def coordinator(mock_hass):
    """Create a FadeCoordinator with mock hass."""
    return FadeCoordinator(
        hass=mock_hass,
        entry=MagicMock(),
        store=MagicMock(),
        data={},
        min_step_delay_ms=100,
    )


class TestApplyStep:
    """Test _apply_step function."""

    @pytest.mark.asyncio
    async def test_brightness_only(
        self, mock_hass: MagicMock, coordinator: FadeCoordinator
    ) -> None:
        """Test applying a step with only brightness."""
        step = FadeStep(brightness=128)
        await coordinator._apply_step("light.test", step)

        mock_hass.services.async_call.assert_called_once_with(
            "light",
            "turn_on",
            {"entity_id": "light.test", "brightness": 128},
            blocking=True,
        )

    @pytest.mark.asyncio
    async def test_brightness_zero_turns_off(
        self, mock_hass: MagicMock, coordinator: FadeCoordinator
    ) -> None:
        """Test applying brightness=0 turns off the light."""
        step = FadeStep(brightness=0)
        await coordinator._apply_step("light.test", step)

        mock_hass.services.async_call.assert_called_once_with(
            "light",
            "turn_off",
            {"entity_id": "light.test"},
            blocking=True,
        )

    @pytest.mark.asyncio
    async def test_hs_color_only(self, mock_hass: MagicMock, coordinator: FadeCoordinator) -> None:
        """Test applying a step with only HS color."""
        step = FadeStep(hs_color=(120.0, 80.0))
        await coordinator._apply_step("light.test", step)

        mock_hass.services.async_call.assert_called_once_with(
            "light",
            "turn_on",
            {"entity_id": "light.test", "hs_color": (120.0, 80.0)},
            blocking=True,
        )

    @pytest.mark.asyncio
    async def test_color_temp_kelvin_passed_directly(
        self, mock_hass: MagicMock, coordinator: FadeCoordinator
    ) -> None:
        """Test applying color temp passes kelvin directly."""
        step = FadeStep(color_temp_kelvin=4000)
        await coordinator._apply_step("light.test", step)

        mock_hass.services.async_call.assert_called_once_with(
            "light",
            "turn_on",
            {"entity_id": "light.test", "color_temp_kelvin": 4000},
            blocking=True,
        )

    @pytest.mark.asyncio
    async def test_brightness_and_hs_color(
        self, mock_hass: MagicMock, coordinator: FadeCoordinator
    ) -> None:
        """Test applying a step with brightness and HS color."""
        step = FadeStep(brightness=200, hs_color=(50.0, 90.0))
        await coordinator._apply_step("light.test", step)

        mock_hass.services.async_call.assert_called_once_with(
            "light",
            "turn_on",
            {"entity_id": "light.test", "brightness": 200, "hs_color": (50.0, 90.0)},
            blocking=True,
        )

    @pytest.mark.asyncio
    async def test_brightness_and_color_temp(
        self, mock_hass: MagicMock, coordinator: FadeCoordinator
    ) -> None:
        """Test applying a step with brightness and color temp."""
        step = FadeStep(brightness=150, color_temp_kelvin=2500)
        await coordinator._apply_step("light.test", step)

        mock_hass.services.async_call.assert_called_once_with(
            "light",
            "turn_on",
            {"entity_id": "light.test", "brightness": 150, "color_temp_kelvin": 2500},
            blocking=True,
        )

    @pytest.mark.asyncio
    async def test_empty_step_no_service_call(
        self, mock_hass: MagicMock, coordinator: FadeCoordinator
    ) -> None:
        """Test that an empty step doesn't make a service call."""
        step = FadeStep()
        await coordinator._apply_step("light.test", step)

        # Empty steps don't call any service (nothing to set)
        mock_hass.services.async_call.assert_not_called()
