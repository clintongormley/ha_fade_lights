"""Tests for _apply_step function."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.fade_lights import _apply_step
from custom_components.fade_lights.fade_change import FadeStep


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    hass = MagicMock()
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    return hass


class TestApplyStep:
    """Test _apply_step function."""

    @pytest.mark.asyncio
    async def test_brightness_only(self, mock_hass: MagicMock) -> None:
        """Test applying a step with only brightness."""
        step = FadeStep(brightness=128)
        await _apply_step(mock_hass, "light.test", step)

        mock_hass.services.async_call.assert_called_once_with(
            "light",
            "turn_on",
            {"entity_id": "light.test", "brightness": 128, "transition": 0.1},
            blocking=True,
        )

    @pytest.mark.asyncio
    async def test_brightness_zero_turns_off(self, mock_hass: MagicMock) -> None:
        """Test applying brightness=0 turns off the light."""
        step = FadeStep(brightness=0)
        await _apply_step(mock_hass, "light.test", step)

        mock_hass.services.async_call.assert_called_once_with(
            "light",
            "turn_off",
            {"entity_id": "light.test"},
            blocking=True,
        )

    @pytest.mark.asyncio
    async def test_hs_color_only(self, mock_hass: MagicMock) -> None:
        """Test applying a step with only HS color."""
        step = FadeStep(hs_color=(120.0, 80.0))
        await _apply_step(mock_hass, "light.test", step)

        mock_hass.services.async_call.assert_called_once_with(
            "light",
            "turn_on",
            {"entity_id": "light.test", "hs_color": (120.0, 80.0), "transition": 0.1},
            blocking=True,
        )

    @pytest.mark.asyncio
    async def test_color_temp_kelvin_passed_directly(self, mock_hass: MagicMock) -> None:
        """Test applying color temp passes kelvin directly."""
        step = FadeStep(color_temp_kelvin=4000)
        await _apply_step(mock_hass, "light.test", step)

        mock_hass.services.async_call.assert_called_once_with(
            "light",
            "turn_on",
            {"entity_id": "light.test", "color_temp_kelvin": 4000, "transition": 0.1},
            blocking=True,
        )

    @pytest.mark.asyncio
    async def test_brightness_and_hs_color(self, mock_hass: MagicMock) -> None:
        """Test applying a step with brightness and HS color."""
        step = FadeStep(brightness=200, hs_color=(50.0, 90.0))
        await _apply_step(mock_hass, "light.test", step)

        mock_hass.services.async_call.assert_called_once_with(
            "light",
            "turn_on",
            {"entity_id": "light.test", "brightness": 200, "hs_color": (50.0, 90.0), "transition": 0.1},
            blocking=True,
        )

    @pytest.mark.asyncio
    async def test_brightness_and_color_temp(self, mock_hass: MagicMock) -> None:
        """Test applying a step with brightness and color temp."""
        step = FadeStep(brightness=150, color_temp_kelvin=2500)
        await _apply_step(mock_hass, "light.test", step)

        mock_hass.services.async_call.assert_called_once_with(
            "light",
            "turn_on",
            {"entity_id": "light.test", "brightness": 150, "color_temp_kelvin": 2500, "transition": 0.1},
            blocking=True,
        )

    @pytest.mark.asyncio
    async def test_empty_step_still_applies_transition(self, mock_hass: MagicMock) -> None:
        """Test that an empty step still applies native transition for smoothing."""
        step = FadeStep()
        await _apply_step(mock_hass, "light.test", step)

        # Even empty steps apply native transition for flicker-free fading
        mock_hass.services.async_call.assert_called_once_with(
            "light",
            "turn_on",
            {"entity_id": "light.test", "transition": 0.1},
            blocking=True,
        )
