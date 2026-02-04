"""Tests for color temp bounds clamping."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components.light import (
    ATTR_BRIGHTNESS,
    ATTR_COLOR_TEMP_KELVIN,
    ATTR_MAX_COLOR_TEMP_KELVIN,
    ATTR_MIN_COLOR_TEMP_KELVIN,
    ATTR_SUPPORTED_COLOR_MODES,
)
from homeassistant.components.light.const import ColorMode
from homeassistant.const import STATE_ON

from custom_components.fade_lights import (
    DOMAIN,
    _clamp_mireds,
    resolve_fade,
)
from custom_components.fade_lights.fade_params import FadeParams


class TestClampMireds:
    """Test _clamp_mireds helper function."""

    def test_clamp_within_bounds(self) -> None:
        """Test value within bounds is unchanged."""
        result = _clamp_mireds(300, min_mireds=200, max_mireds=400)
        assert result == 300

    def test_clamp_below_min(self) -> None:
        """Test value below min is clamped to min."""
        result = _clamp_mireds(100, min_mireds=200, max_mireds=400)
        assert result == 200

    def test_clamp_above_max(self) -> None:
        """Test value above max is clamped to max."""
        result = _clamp_mireds(500, min_mireds=200, max_mireds=400)
        assert result == 400

    def test_clamp_no_bounds(self) -> None:
        """Test value unchanged when no bounds provided."""
        result = _clamp_mireds(300, min_mireds=None, max_mireds=None)
        assert result == 300

    def test_clamp_only_min(self) -> None:
        """Test clamping with only min bound."""
        assert _clamp_mireds(100, min_mireds=200, max_mireds=None) == 200
        assert _clamp_mireds(300, min_mireds=200, max_mireds=None) == 300

    def test_clamp_only_max(self) -> None:
        """Test clamping with only max bound."""
        assert _clamp_mireds(500, min_mireds=None, max_mireds=400) == 400
        assert _clamp_mireds(300, min_mireds=None, max_mireds=400) == 300


class TestResolveFadeWithBounds:
    """Test resolve_fade with color temp bounds."""

    def test_color_temp_clamped_to_min_kelvin(self) -> None:
        """Test color temp below min_kelvin is clamped."""
        # Light has min 2000K (500 mireds), max 6500K (153 mireds)
        # User requests 1500K which is below min
        params = FadeParams(
            transition_ms=1000,
            color_temp_kelvin=1500,  # Below min 2000K
        )
        state = {
            ATTR_BRIGHTNESS: 128,
            ATTR_COLOR_TEMP_KELVIN: 4000,
            ATTR_MIN_COLOR_TEMP_KELVIN: 2000,
            ATTR_MAX_COLOR_TEMP_KELVIN: 6500,
        }
        supported_modes = {ColorMode.COLOR_TEMP}

        # min_mireds = 1_000_000 / 6500 = 153
        # max_mireds = 1_000_000 / 2000 = 500
        change = resolve_fade(
            params, state, supported_modes, min_step_delay_ms=100, min_mireds=153, max_mireds=500
        )

        assert change is not None
        # End mireds should be clamped to max_mireds (500) = 2000K
        # 1500K = 666 mireds, which should clamp to 500 mireds
        assert change.end_mireds == 500

    def test_color_temp_clamped_to_max_kelvin(self) -> None:
        """Test color temp above max_kelvin is clamped."""
        # Light has min 2000K (500 mireds), max 6500K (153 mireds)
        # User requests 8000K which is above max
        params = FadeParams(
            transition_ms=1000,
            color_temp_kelvin=8000,  # Above max 6500K
        )
        state = {
            ATTR_BRIGHTNESS: 128,
            ATTR_COLOR_TEMP_KELVIN: 4000,
            ATTR_MIN_COLOR_TEMP_KELVIN: 2000,
            ATTR_MAX_COLOR_TEMP_KELVIN: 6500,
        }
        supported_modes = {ColorMode.COLOR_TEMP}

        # min_mireds = 1_000_000 / 6500 = 153
        # max_mireds = 1_000_000 / 2000 = 500
        change = resolve_fade(
            params, state, supported_modes, min_step_delay_ms=100, min_mireds=153, max_mireds=500
        )

        assert change is not None
        # End mireds should be clamped to min_mireds (153) = 6500K
        # 8000K = 125 mireds, which should clamp to 153 mireds
        assert change.end_mireds == 153

    def test_color_temp_within_bounds_unchanged(self) -> None:
        """Test color temp within bounds is unchanged."""
        params = FadeParams(
            transition_ms=1000,
            color_temp_kelvin=4000,  # Within bounds
        )
        state = {
            ATTR_BRIGHTNESS: 128,
            ATTR_COLOR_TEMP_KELVIN: 3000,
            ATTR_MIN_COLOR_TEMP_KELVIN: 2000,
            ATTR_MAX_COLOR_TEMP_KELVIN: 6500,
        }
        supported_modes = {ColorMode.COLOR_TEMP}

        # min_mireds = 1_000_000 / 6500 = 153
        # max_mireds = 1_000_000 / 2000 = 500
        change = resolve_fade(
            params, state, supported_modes, min_step_delay_ms=100, min_mireds=153, max_mireds=500
        )

        assert change is not None
        # 4000K = 250 mireds, should be unchanged
        assert change.end_mireds == 250

    def test_from_color_temp_clamped(self) -> None:
        """Test from_color_temp_kelvin is clamped to bounds."""
        params = FadeParams(
            transition_ms=1000,
            color_temp_kelvin=4000,
            from_color_temp_kelvin=1500,  # Below min 2000K
        )
        state = {
            ATTR_BRIGHTNESS: 128,
            ATTR_COLOR_TEMP_KELVIN: 4000,
            ATTR_MIN_COLOR_TEMP_KELVIN: 2000,
            ATTR_MAX_COLOR_TEMP_KELVIN: 6500,
        }
        supported_modes = {ColorMode.COLOR_TEMP}

        # min_mireds = 1_000_000 / 6500 = 153
        # max_mireds = 1_000_000 / 2000 = 500
        change = resolve_fade(
            params, state, supported_modes, min_step_delay_ms=100, min_mireds=153, max_mireds=500
        )

        assert change is not None
        # Start mireds should be clamped to max_mireds (500) = 2000K
        # 1500K = 666 mireds, which should clamp to 500 mireds
        assert change.start_mireds == 500

    def test_no_bounds_no_clamping(self) -> None:
        """Test that without bounds, values are not clamped."""
        params = FadeParams(
            transition_ms=1000,
            color_temp_kelvin=1500,  # Would be below typical min
        )
        state = {
            ATTR_BRIGHTNESS: 128,
            ATTR_COLOR_TEMP_KELVIN: 4000,
            # No min/max bounds
        }
        supported_modes = {ColorMode.COLOR_TEMP}

        change = resolve_fade(params, state, supported_modes, min_step_delay_ms=100)

        assert change is not None
        # 1500K = 666 mireds, should NOT be clamped
        assert change.end_mireds == 666


class TestHybridTransitionsWithBounds:
    """Test hybrid transitions with bounds clamping."""

    def test_hs_to_mireds_end_mireds_clamped(self) -> None:
        """Test HS->mireds hybrid clamps end_mireds to max."""
        # User requests 1500K (666 mireds), but light max is 500 mireds (2000K)
        params = FadeParams(
            transition_ms=2000,
            color_temp_kelvin=1500,  # Will be clamped
        )
        state = {
            "hs_color": (120.0, 100.0),  # Saturated green - off locus
        }
        supported_modes = {ColorMode.HS, ColorMode.COLOR_TEMP}

        change = resolve_fade(
            params, state, supported_modes, min_step_delay_ms=100, min_mireds=153, max_mireds=500
        )

        assert change is not None
        # Should be hybrid transition
        assert change._hybrid_direction == "hs_to_mireds"
        # End mireds should be clamped to 500
        assert change.end_mireds == 500

    def test_mireds_to_hs_start_mireds_clamped(self) -> None:
        """Test mireds->HS hybrid clamps start_mireds."""
        # Start at 666 mireds (1500K), but light max is 500 mireds (2000K)
        params = FadeParams(
            transition_ms=2000,
            hs_color=(120.0, 100.0),  # Saturated green
            from_color_temp_kelvin=1500,  # Will be clamped
        )
        state = {
            ATTR_COLOR_TEMP_KELVIN: 1500,  # Will be clamped
        }
        supported_modes = {ColorMode.HS, ColorMode.COLOR_TEMP}

        change = resolve_fade(
            params, state, supported_modes, min_step_delay_ms=100, min_mireds=153, max_mireds=500
        )

        assert change is not None
        # Should be hybrid transition
        assert change._hybrid_direction == "mireds_to_hs"
        # Start mireds should be clamped to 500
        assert change.start_mireds == 500


class TestExecuteFadeWithBounds:
    """Test _execute_fade extracts and uses bounds correctly."""

    @pytest.fixture
    def mock_hass(self):
        """Create a mock Home Assistant instance."""
        hass = MagicMock()
        hass.data = {DOMAIN: {"data": {}, "store": MagicMock()}}
        hass.services = MagicMock()
        hass.services.async_call = AsyncMock()
        return hass

    async def test_bounds_extracted_from_state(self, mock_hass) -> None:
        """Test that min/max kelvin are extracted from state attributes."""
        from custom_components.fade_lights import _execute_fade

        # Light with color temp bounds
        state = MagicMock()
        state.attributes = {
            ATTR_BRIGHTNESS: 128,
            ATTR_COLOR_TEMP_KELVIN: 4000,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.COLOR_TEMP],
            ATTR_MIN_COLOR_TEMP_KELVIN: 2000,
            ATTR_MAX_COLOR_TEMP_KELVIN: 6500,
        }
        mock_hass.states.get.return_value = state

        params = FadeParams(
            transition_ms=100,
            color_temp_kelvin=1500,  # Below min
        )

        cancel_event = MagicMock()
        cancel_event.is_set.return_value = False

        with patch("custom_components.fade_lights._apply_step", new_callable=AsyncMock) as mock_apply:
            await _execute_fade(mock_hass, "light.test", params, 50, cancel_event)

            # Verify _apply_step was called
            assert mock_apply.called
            # Get the LAST step that was applied (final target)
            call_args = mock_apply.call_args_list[-1]
            step = call_args[0][2]  # Third arg is the step

            # The color temp should be clamped to min kelvin (2000K)
            # 1500K would be 666 mireds, clamped to 500 mireds = 2000K
            assert step.color_temp_kelvin == 2000

    async def test_no_bounds_no_clamping(self, mock_hass) -> None:
        """Test that without bounds, color temp is not clamped."""
        from custom_components.fade_lights import _execute_fade

        # Light without color temp bounds
        state = MagicMock()
        state.attributes = {
            ATTR_BRIGHTNESS: 128,
            ATTR_COLOR_TEMP_KELVIN: 4000,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.COLOR_TEMP],
            # No min/max bounds
        }
        mock_hass.states.get.return_value = state

        params = FadeParams(
            transition_ms=100,
            color_temp_kelvin=1500,  # Would be clamped if bounds existed
        )

        cancel_event = MagicMock()
        cancel_event.is_set.return_value = False

        with patch("custom_components.fade_lights._apply_step", new_callable=AsyncMock) as mock_apply:
            await _execute_fade(mock_hass, "light.test", params, 50, cancel_event)

            # Verify _apply_step was called
            assert mock_apply.called
            # Get the LAST step that was applied (final target)
            call_args = mock_apply.call_args_list[-1]
            step = call_args[0][2]  # Third arg is the step

            # The color temp should NOT be clamped - 666 mireds = 1500K
            # (allowing for rounding: int(1_000_000/666) = 1501)
            assert step.color_temp_kelvin in (1500, 1501, 1502)
