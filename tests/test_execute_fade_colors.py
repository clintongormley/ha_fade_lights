"""Tests for _execute_fade with color support."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.fado.coordinator import FadeCoordinator
from custom_components.fado.fade_params import FadeParams


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
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
        store=MagicMock(async_save=AsyncMock()),
        data={},
        min_step_delay_ms=100,
    )


@pytest.fixture
def cancel_event():
    """Create a cancel event that is not set."""
    return asyncio.Event()


@pytest.fixture
def mock_state_on():
    """Create a mock state for a light that is on with brightness."""
    state = MagicMock()
    state.state = "on"
    state.attributes = {
        "brightness": 128,
        "supported_color_modes": ["brightness", "hs", "color_temp"],
        "hs_color": (0.0, 0.0),
        "color_temp_kelvin": 3333,  # ~300 mireds equivalent
    }
    return state


class TestExecuteFadeWithColors:
    """Test _execute_fade with color parameters."""

    @pytest.mark.asyncio
    async def test_fade_hs_color_only(
        self,
        mock_hass: MagicMock,
        coordinator: FadeCoordinator,
        cancel_event: asyncio.Event,
        mock_state_on: MagicMock,
    ) -> None:
        """Test fading HS color without brightness change."""
        mock_hass.states.get = MagicMock(return_value=mock_state_on)

        params = FadeParams(
            hs_color=(120.0, 80.0),  # Target green
        )

        params.transition_ms = 1000
        await coordinator._execute_fade(
            "light.test",
            params,
            100,  # min_step_delay_ms
            cancel_event,
        )

        # Should have called turn_on with hs_color
        calls = mock_hass.services.async_call.call_args_list
        assert len(calls) > 0
        # Last call should have the target HS color
        last_call = calls[-1]
        assert last_call[0][0] == "light"  # domain
        assert last_call[0][1] == "turn_on"  # service
        assert "hs_color" in last_call[0][2]

    @pytest.mark.asyncio
    async def test_fade_color_temp_only(
        self,
        mock_hass: MagicMock,
        coordinator: FadeCoordinator,
        cancel_event: asyncio.Event,
        mock_state_on: MagicMock,
    ) -> None:
        """Test fading color temperature without brightness change."""
        mock_hass.states.get = MagicMock(return_value=mock_state_on)

        params = FadeParams(
            color_temp_kelvin=2500,  # Target warm white (400 mireds equivalent)
        )

        params.transition_ms = 1000
        await coordinator._execute_fade(
            "light.test",
            params,
            100,  # min_step_delay_ms
            cancel_event,
        )

        # Should have called turn_on with color_temp_kelvin
        calls = mock_hass.services.async_call.call_args_list
        assert len(calls) > 0
        # Last call should have color temp
        last_call = calls[-1]
        assert last_call[0][0] == "light"
        assert last_call[0][1] == "turn_on"
        assert "color_temp_kelvin" in last_call[0][2]

    @pytest.mark.asyncio
    async def test_fade_brightness_and_hs_color(
        self,
        mock_hass: MagicMock,
        coordinator: FadeCoordinator,
        cancel_event: asyncio.Event,
        mock_state_on: MagicMock,
    ) -> None:
        """Test fading both brightness and HS color together."""
        mock_hass.states.get = MagicMock(return_value=mock_state_on)

        params = FadeParams(
            brightness_pct=100,  # Target full brightness
            hs_color=(240.0, 100.0),  # Target blue
        )

        params.transition_ms = 1000
        await coordinator._execute_fade(
            "light.test",
            params,
            100,  # min_step_delay_ms
            cancel_event,
        )

        # Should have called turn_on with both brightness and hs_color
        calls = mock_hass.services.async_call.call_args_list
        assert len(calls) > 0
        # Last call should have both
        last_call = calls[-1]
        assert last_call[0][0] == "light"
        assert last_call[0][1] == "turn_on"
        assert "brightness" in last_call[0][2]
        assert "hs_color" in last_call[0][2]

    @pytest.mark.asyncio
    async def test_brightness_only_fade_unchanged(
        self,
        mock_hass: MagicMock,
        coordinator: FadeCoordinator,
        cancel_event: asyncio.Event,
        mock_state_on: MagicMock,
    ) -> None:
        """Test that brightness-only fade still works (regression test)."""
        mock_hass.states.get = MagicMock(return_value=mock_state_on)

        params = FadeParams(
            brightness_pct=100,  # No color params
            transition_ms=500,
        )

        await coordinator._execute_fade(
            "light.test",
            params,
            100,  # min_step_delay_ms
            cancel_event,
        )

        calls = mock_hass.services.async_call.call_args_list
        assert len(calls) > 0
        # Should have brightness in calls
        has_brightness = any(
            "brightness" in call[0][2]
            for call in calls
            if len(call[0]) > 2 and call[0][1] == "turn_on"
        )
        assert has_brightness
