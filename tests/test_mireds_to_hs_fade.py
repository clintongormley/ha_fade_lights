"""Integration tests for color-temp-to-HS hybrid fade transitions."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components.light import ATTR_COLOR_TEMP_KELVIN as HA_ATTR_COLOR_TEMP_KELVIN
from homeassistant.components.light.const import ColorMode
from homeassistant.const import STATE_ON

from custom_components.fade_lights import _execute_fade, resolve_fade
from custom_components.fade_lights.fade_change import FadeChange
from custom_components.fade_lights.fade_params import FadeParams


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
        HA_ATTR_COLOR_TEMP_KELVIN: 3003,  # ~333 mireds equivalent
        "supported_color_modes": [ColorMode.COLOR_TEMP, ColorMode.HS],
    }
    return state


class TestColorTempToHsFade:
    """Integration tests for color-temp-to-HS transitions."""

    @pytest.mark.asyncio
    async def test_color_temp_to_hs_uses_hybrid_fade(self, mock_hass, color_temp_light_state):
        """Test that COLOR_TEMP mode light fading to HS uses hybrid transition.

        Uses resolve_fade which returns a single FadeChange with hybrid support.
        """
        mock_hass.states.get = MagicMock(return_value=color_temp_light_state)

        cancel_event = asyncio.Event()

        fade_params = FadeParams(
            brightness_pct=None,
            hs_color=(0.0, 100.0),  # Red
            transition_ms=3000,
        )

        with patch("custom_components.fade_lights._save_storage", new_callable=AsyncMock):
            # Use resolve_fade to verify it returns hybrid FadeChange
            supported_modes = {ColorMode.COLOR_TEMP, ColorMode.HS}
            change = resolve_fade(
                fade_params,
                color_temp_light_state.attributes,
                supported_modes,
                min_step_delay_ms=100,
            )

            # Verify we get a hybrid FadeChange
            assert change is not None
            assert change._hybrid_direction == "mireds_to_hs"
            assert change._crossover_step is not None

            # Verify start mireds comes from state
            assert change.start_mireds == 333  # 3003K -> 333 mireds

            # Verify end HS is the target
            assert change.end_hs == (0.0, 100.0)

            # Also run _execute_fade to ensure it executes without error
            await _execute_fade(
                mock_hass,
                "light.test",
                fade_params,
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
            transition_ms=3000,
        )

        with patch("custom_components.fade_lights._save_storage", new_callable=AsyncMock):
            # Use resolve_fade to verify it returns non-hybrid FadeChange
            supported_modes = {ColorMode.COLOR_TEMP, ColorMode.HS}
            change = resolve_fade(
                fade_params,
                hs_state.attributes,
                supported_modes,
                min_step_delay_ms=100,
            )

            # Should be simple HS-to-HS fade (NOT hybrid)
            assert change is not None
            assert change._hybrid_direction is None
            assert change.start_hs == (200.0, 50.0)
            assert change.end_hs == (0.0, 100.0)

            await _execute_fade(
                mock_hass,
                "light.test",
                fade_params,
                min_step_delay_ms=100,
                cancel_event=cancel_event,
            )

    @pytest.mark.asyncio
    async def test_color_temp_to_color_temp_uses_standard_fade(self, mock_hass, color_temp_light_state):
        """Test that COLOR_TEMP to color temp uses standard fade (no mode switch needed)."""
        mock_hass.states.get = MagicMock(return_value=color_temp_light_state)

        cancel_event = asyncio.Event()

        fade_params = FadeParams(
            brightness_pct=None,
            color_temp_kelvin=5000,  # Target is kelvin (200 mireds), not HS
            transition_ms=3000,
        )

        with patch("custom_components.fade_lights._save_storage", new_callable=AsyncMock):
            # Use resolve_fade to verify it returns non-hybrid FadeChange
            supported_modes = {ColorMode.COLOR_TEMP, ColorMode.HS}
            change = resolve_fade(
                fade_params,
                color_temp_light_state.attributes,
                supported_modes,
                min_step_delay_ms=100,
            )

            # Should be simple mireds fade (NOT hybrid)
            assert change is not None
            assert change._hybrid_direction is None
            assert change.start_mireds == 333  # 3003K -> 333 mireds
            assert change.end_mireds == 200  # 5000K -> 200 mireds

            await _execute_fade(
                mock_hass,
                "light.test",
                fade_params,
                min_step_delay_ms=100,
                cancel_event=cancel_event,
            )

    @pytest.mark.asyncio
    async def test_hybrid_fade_generates_both_color_types(self, mock_hass, color_temp_light_state):
        """Test that hybrid fade generates both color_temp and hs_color steps."""
        mock_hass.states.get = MagicMock(return_value=color_temp_light_state)

        fade_params = FadeParams(
            brightness_pct=None,
            hs_color=(0.0, 100.0),  # Red
            transition_ms=3000,
        )

        supported_modes = {ColorMode.COLOR_TEMP, ColorMode.HS}
        change = resolve_fade(
            fade_params,
            color_temp_light_state.attributes,
            supported_modes,
            min_step_delay_ms=100,
        )

        assert change is not None
        assert change._hybrid_direction == "mireds_to_hs"

        # Iterate through all steps and verify we get both color types
        steps_with_color_temp = 0
        steps_with_hs = 0

        while change.has_next():
            step = change.next_step()
            if step.color_temp_kelvin is not None:
                steps_with_color_temp += 1
            if step.hs_color is not None:
                steps_with_hs += 1

        # Should have steps of both types
        assert steps_with_color_temp > 0, "Expected steps with color_temp_kelvin"
        assert steps_with_hs > 0, "Expected steps with hs_color"


class TestHsToColorTempFade:
    """Integration tests for HS-to-color-temp transitions."""

    @pytest.mark.asyncio
    async def test_hs_to_color_temp_uses_hybrid_fade(self, mock_hass):
        """Test that HS mode light fading to color temp uses hybrid transition."""
        hs_state = MagicMock()
        hs_state.state = STATE_ON
        hs_state.attributes = {
            "brightness": 200,
            "color_mode": ColorMode.HS,
            "hs_color": (120.0, 100.0),  # Saturated green (off locus)
            "supported_color_modes": [ColorMode.COLOR_TEMP, ColorMode.HS],
        }
        mock_hass.states.get = MagicMock(return_value=hs_state)

        cancel_event = asyncio.Event()

        fade_params = FadeParams(
            brightness_pct=None,
            color_temp_kelvin=4000,  # Warm white
            transition_ms=3000,
        )

        with patch("custom_components.fade_lights._save_storage", new_callable=AsyncMock):
            supported_modes = {ColorMode.COLOR_TEMP, ColorMode.HS}
            change = resolve_fade(
                fade_params,
                hs_state.attributes,
                supported_modes,
                min_step_delay_ms=100,
            )

            # Should be hybrid HS->mireds transition
            assert change is not None
            assert change._hybrid_direction == "hs_to_mireds"
            assert change._crossover_step is not None
            assert change.start_hs == (120.0, 100.0)
            assert change.end_mireds == 250  # 4000K = 250 mireds

            await _execute_fade(
                mock_hass,
                "light.test",
                fade_params,
                min_step_delay_ms=100,
                cancel_event=cancel_event,
            )

    @pytest.mark.asyncio
    async def test_hs_on_locus_to_color_temp_not_hybrid(self, mock_hass):
        """Test that HS on Planckian locus fading to color temp is NOT hybrid."""
        hs_state = MagicMock()
        hs_state.state = STATE_ON
        hs_state.attributes = {
            "brightness": 200,
            "color_mode": ColorMode.HS,
            "hs_color": (35.0, 10.0),  # Low saturation (on locus)
            "supported_color_modes": [ColorMode.COLOR_TEMP, ColorMode.HS],
        }
        mock_hass.states.get = MagicMock(return_value=hs_state)

        fade_params = FadeParams(
            brightness_pct=None,
            color_temp_kelvin=4000,
            transition_ms=3000,
        )

        supported_modes = {ColorMode.COLOR_TEMP, ColorMode.HS}
        change = resolve_fade(
            fade_params,
            hs_state.attributes,
            supported_modes,
            min_step_delay_ms=100,
        )

        # Should NOT be hybrid (on locus can go directly to mireds)
        assert change is not None
        assert change._hybrid_direction is None
