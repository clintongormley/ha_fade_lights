"""Fixtures for Fade Lights integration tests."""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.components.light import ATTR_BRIGHTNESS, ATTR_SUPPORTED_COLOR_MODES, ColorMode
from homeassistant.const import ATTR_ENTITY_ID, STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant, ServiceCall
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.fade_lights.const import DOMAIN


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(
    enable_custom_integrations: None,
) -> Generator[None]:
    """Enable custom integrations for all tests."""
    yield


@pytest.fixture
def mock_config_entry() -> MockConfigEntry:
    """Create a mock config entry for the Fade Lights integration."""
    return MockConfigEntry(
        domain=DOMAIN,
        title="Fade Lights",
        data={},
        options={},
        unique_id="fade_lights_unique",
    )


@pytest.fixture
async def init_integration(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> MockConfigEntry:
    """Set up the Fade Lights integration for testing."""
    mock_config_entry.add_to_hass(hass)

    with patch(
        "custom_components.fade_lights.Store",
        return_value=AsyncMock(async_load=AsyncMock(return_value={}), async_save=AsyncMock()),
    ):
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()

    return mock_config_entry


@pytest.fixture
def mock_light_entity(hass: HomeAssistant) -> str:
    """Create a mock dimmable light entity at brightness 200."""
    entity_id = "light.test_light"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 200,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )
    return entity_id


@pytest.fixture
def mock_light_off(hass: HomeAssistant) -> str:
    """Create a mock dimmable light entity that is off."""
    entity_id = "light.test_light_off"
    hass.states.async_set(
        entity_id,
        STATE_OFF,
        {
            ATTR_BRIGHTNESS: None,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )
    return entity_id


@pytest.fixture
def mock_non_dimmable_light(hass: HomeAssistant) -> str:
    """Create a mock non-dimmable (on/off only) light entity."""
    entity_id = "light.non_dimmable_light"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.ONOFF],
        },
    )
    return entity_id


@pytest.fixture
def mock_light_group(hass: HomeAssistant, mock_light_entity: str, mock_light_off: str) -> str:
    """Create a mock light group containing other lights."""
    entity_id = "light.test_group"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 150,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
            ATTR_ENTITY_ID: [mock_light_entity, mock_light_off],
        },
    )
    return entity_id


@pytest.fixture
def captured_calls(hass: HomeAssistant) -> list[ServiceCall]:
    """Capture light service calls for verification."""
    calls: list[ServiceCall] = []

    async def mock_service_handler(call: ServiceCall) -> None:
        """Record the service call and update light state."""
        calls.append(call)

        entity_id = call.data.get(ATTR_ENTITY_ID)
        if entity_id:
            entity_ids = entity_id if isinstance(entity_id, list) else [entity_id]
            for eid in entity_ids:
                current_state = hass.states.get(eid)
                if current_state:
                    current_attrs = dict(current_state.attributes)

                    if call.service == "turn_on":
                        new_state = STATE_ON
                        if ATTR_BRIGHTNESS in call.data:
                            current_attrs[ATTR_BRIGHTNESS] = call.data[ATTR_BRIGHTNESS]
                    elif call.service == "turn_off":
                        new_state = STATE_OFF
                        current_attrs[ATTR_BRIGHTNESS] = None
                    else:
                        new_state = current_state.state

                    hass.states.async_set(eid, new_state, current_attrs)

    hass.services.async_register("light", "turn_on", mock_service_handler)
    hass.services.async_register("light", "turn_off", mock_service_handler)

    return calls
