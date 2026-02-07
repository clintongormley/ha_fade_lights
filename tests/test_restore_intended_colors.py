"""Tests for _restore_intended_state with color support."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components.light import ATTR_BRIGHTNESS
from homeassistant.components.light import ATTR_COLOR_TEMP_KELVIN as HA_ATTR_COLOR_TEMP_KELVIN
from homeassistant.components.light import ATTR_HS_COLOR as HA_ATTR_HS_COLOR
from homeassistant.const import STATE_ON

from custom_components.fado import (
    DOMAIN,
    INTENDED_STATE_QUEUE,
    _restore_intended_state,
)


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    hass = MagicMock()
    hass.data = {DOMAIN: {"data": {}, "store": MagicMock()}}
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    hass.states = MagicMock()
    return hass


class TestRestoreIntendedColors:
    """Test _restore_intended_state with color support."""

    @pytest.mark.asyncio
    async def test_restore_includes_hs_color(self, mock_hass):
        """Test restoration includes HS color from manual intervention."""
        entity_id = "light.test"

        # Old state (before intervention)
        old_state = MagicMock()
        old_state.state = STATE_ON
        old_state.attributes = {ATTR_BRIGHTNESS: 100, HA_ATTR_HS_COLOR: (50.0, 30.0)}

        # New state (the manual intervention)
        new_state = MagicMock()
        new_state.state = STATE_ON
        new_state.entity_id = entity_id
        new_state.attributes = {ATTR_BRIGHTNESS: 150, HA_ATTR_HS_COLOR: (200.0, 60.0)}

        # Current state after fade cleanup (different from intended)
        current_state = MagicMock()
        current_state.state = STATE_ON
        current_state.attributes = {ATTR_BRIGHTNESS: 80, HA_ATTR_HS_COLOR: (100.0, 40.0)}
        mock_hass.states.get.return_value = current_state

        # Set up intended state queue: [old_state, intended_state]
        INTENDED_STATE_QUEUE[entity_id] = [old_state, new_state]

        try:
            with (
                patch(
                    "custom_components.fado._cancel_and_wait_for_fade",
                    new_callable=AsyncMock,
                ),
                patch(
                    "custom_components.fado._wait_until_stale_events_flushed",
                    new_callable=AsyncMock,
                ),
            ):
                await _restore_intended_state(mock_hass, entity_id)

            # Verify service call includes HS color
            mock_hass.services.async_call.assert_called()
            call_args = mock_hass.services.async_call.call_args
            service_data = call_args[0][2]  # Third positional arg is the data dict

            assert service_data.get(ATTR_BRIGHTNESS) == 150
            assert service_data.get(HA_ATTR_HS_COLOR) == (200.0, 60.0)
        finally:
            INTENDED_STATE_QUEUE.pop(entity_id, None)

    @pytest.mark.asyncio
    async def test_restore_includes_color_temp(self, mock_hass):
        """Test restoration includes color temp from manual intervention."""
        entity_id = "light.test"

        # Old state
        old_state = MagicMock()
        old_state.state = STATE_ON
        old_state.attributes = {ATTR_BRIGHTNESS: 100, HA_ATTR_COLOR_TEMP_KELVIN: 3333}

        # New state (the manual intervention) - user set to 2500K
        new_state = MagicMock()
        new_state.state = STATE_ON
        new_state.entity_id = entity_id
        new_state.attributes = {ATTR_BRIGHTNESS: 150, HA_ATTR_COLOR_TEMP_KELVIN: 2500}

        # Current state after fade cleanup (different from intended) - at 4000K
        current_state = MagicMock()
        current_state.state = STATE_ON
        current_state.attributes = {ATTR_BRIGHTNESS: 80, HA_ATTR_COLOR_TEMP_KELVIN: 4000}
        mock_hass.states.get.return_value = current_state

        # Set up intended state queue: [old_state, intended_state]
        INTENDED_STATE_QUEUE[entity_id] = [old_state, new_state]

        try:
            with (
                patch(
                    "custom_components.fado._cancel_and_wait_for_fade",
                    new_callable=AsyncMock,
                ),
                patch(
                    "custom_components.fado._wait_until_stale_events_flushed",
                    new_callable=AsyncMock,
                ),
            ):
                await _restore_intended_state(mock_hass, entity_id)

            # Verify service call includes color temp in kelvin
            mock_hass.services.async_call.assert_called()
            call_args = mock_hass.services.async_call.call_args
            service_data = call_args[0][2]

            assert service_data.get(ATTR_BRIGHTNESS) == 150
            assert service_data.get(HA_ATTR_COLOR_TEMP_KELVIN) == 2500
        finally:
            INTENDED_STATE_QUEUE.pop(entity_id, None)

    @pytest.mark.asyncio
    async def test_no_restore_when_current_matches_intended(self, mock_hass):
        """Test no service call when current state matches intended."""
        entity_id = "light.test"

        old_state = MagicMock()
        old_state.state = STATE_ON
        old_state.attributes = {ATTR_BRIGHTNESS: 100}

        new_state = MagicMock()
        new_state.state = STATE_ON
        new_state.entity_id = entity_id
        new_state.attributes = {ATTR_BRIGHTNESS: 150, HA_ATTR_HS_COLOR: (200.0, 60.0)}

        # Current state matches intended
        current_state = MagicMock()
        current_state.state = STATE_ON
        current_state.attributes = {ATTR_BRIGHTNESS: 150, HA_ATTR_HS_COLOR: (200.0, 60.0)}
        mock_hass.states.get.return_value = current_state

        # Set up intended state queue: [old_state, intended_state]
        INTENDED_STATE_QUEUE[entity_id] = [old_state, new_state]

        try:
            with (
                patch(
                    "custom_components.fado._cancel_and_wait_for_fade",
                    new_callable=AsyncMock,
                ),
                patch(
                    "custom_components.fado._wait_until_stale_events_flushed",
                    new_callable=AsyncMock,
                ),
            ):
                await _restore_intended_state(mock_hass, entity_id)

            # No restoration needed
            mock_hass.services.async_call.assert_not_called()
        finally:
            INTENDED_STATE_QUEUE.pop(entity_id, None)
