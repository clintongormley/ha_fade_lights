"""Integration tests for mireds-to-HS hybrid fade transitions."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components.light.const import ColorMode
from homeassistant.const import STATE_ON

from custom_components.fade_lights import _execute_fade, _calculate_changes
from custom_components.fade_lights.fade_change import FadeChange
from custom_components.fade_lights.models import FadeParams


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    hass = MagicMock()
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    hass.data = {
        "fade_lights": {
            "data": {},
            "store": MagicMock(),
            "min_step_delay_ms": 100,
        }
    }
    return hass


@pytest.fixture
def color_temp_light_state():
    """Create a light state in COLOR_TEMP mode."""
    state = MagicMock()
    state.state = STATE_ON
    state.attributes = {
        "brightness": 200,
        "color_mode": ColorMode.COLOR_TEMP,
        "color_temp": 333,  # ~3000K in mireds
        "supported_color_modes": [ColorMode.COLOR_TEMP, ColorMode.HS],
    }
    return state


class TestMiredsToHsFade:
    """Integration tests for mireds-to-HS transitions."""

    @pytest.mark.asyncio
    async def test_color_temp_to_hs_uses_hybrid_fade(self, mock_hass, color_temp_light_state):
        """Test that COLOR_TEMP mode light fading to HS uses hybrid transition.

        Uses _calculate_changes which returns FadeChange phases directly.
        """
        mock_hass.states.get = MagicMock(return_value=color_temp_light_state)

        cancel_event = asyncio.Event()

        fade_params = FadeParams(
            brightness_pct=None,
            hs_color=(0.0, 100.0),  # Red
        )

        with patch("custom_components.fade_lights._save_storage", new_callable=AsyncMock):
            # Use _calculate_changes to verify it returns hybrid phases
            fade_params.transition_ms = 3000
            phases = _calculate_changes(
                fade_params,
                color_temp_light_state.attributes,
                min_step_delay_ms=100,
            )

            # Verify we get two phases (hybrid transition)
            assert len(phases) == 2

            # First phase should be mireds transition
            assert phases[0].start_mireds == 333
            assert phases[0].end_mireds is not None
            assert phases[0].start_hs is None
            assert phases[0].end_hs is None

            # Second phase should be HS transition
            assert phases[1].start_hs is not None
            assert phases[1].end_hs == (0.0, 100.0)
            assert phases[1].start_mireds is None
            assert phases[1].end_mireds is None

            # Also run _execute_fade to ensure it executes without error
            await _execute_fade(
                mock_hass,
                "light.test",
                fade_params,
                transition_ms=3000,
                min_step_delay_ms=100,
                cancel_event=cancel_event,
            )

    @pytest.mark.asyncio
    async def test_hs_mode_light_does_not_use_hybrid(self, mock_hass):
        """Test that HS mode light uses standard fade, not hybrid."""
        hs_state = MagicMock()
        hs_state.state = STATE_ON
        hs_state.attributes = {
            "brightness": 200,
            "color_mode": ColorMode.HS,
            "hs_color": (200.0, 50.0),
            "supported_color_modes": [ColorMode.COLOR_TEMP, ColorMode.HS],
        }
        mock_hass.states.get = MagicMock(return_value=hs_state)

        cancel_event = asyncio.Event()

        fade_params = FadeParams(
            brightness_pct=None,
            hs_color=(0.0, 100.0),
        )

        with patch("custom_components.fade_lights._save_storage", new_callable=AsyncMock):
            # Use _calculate_changes to verify it returns single phase
            fade_params.transition_ms = 3000
            phases = _calculate_changes(
                fade_params,
                hs_state.attributes,
                min_step_delay_ms=100,
            )

            # Should be single phase (simple HS to HS fade)
            assert len(phases) == 1
            assert phases[0].start_hs == (200.0, 50.0)
            assert phases[0].end_hs == (0.0, 100.0)

            await _execute_fade(
                mock_hass,
                "light.test",
                fade_params,
                transition_ms=3000,
                min_step_delay_ms=100,
                cancel_event=cancel_event,
            )

    @pytest.mark.asyncio
    async def test_color_temp_to_mireds_uses_standard_fade(self, mock_hass, color_temp_light_state):
        """Test that COLOR_TEMP to mireds uses standard fade (no mode switch needed)."""
        mock_hass.states.get = MagicMock(return_value=color_temp_light_state)

        cancel_event = asyncio.Event()

        fade_params = FadeParams(
            brightness_pct=None,
            color_temp_mireds=200,  # Target is mireds, not HS
        )

        with patch("custom_components.fade_lights._save_storage", new_callable=AsyncMock):
            # Use _calculate_changes to verify it returns single phase
            fade_params.transition_ms = 3000
            phases = _calculate_changes(
                fade_params,
                color_temp_light_state.attributes,
                min_step_delay_ms=100,
            )

            # Should be single phase (simple mireds to mireds fade)
            assert len(phases) == 1
            assert phases[0].start_mireds == 333
            assert phases[0].end_mireds == 200

            await _execute_fade(
                mock_hass,
                "light.test",
                fade_params,
                transition_ms=3000,
                min_step_delay_ms=100,
                cancel_event=cancel_event,
            )
