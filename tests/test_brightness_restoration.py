"""Tests for Fade Lights brightness restoration logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.components.light import (
    ATTR_BRIGHTNESS,
    ATTR_SUPPORTED_COLOR_MODES,
)
from homeassistant.components.light.const import ColorMode
from homeassistant.const import ATTR_ENTITY_ID, STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant, ServiceCall
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.fade_lights.const import DOMAIN


@pytest.fixture
def service_calls(hass: HomeAssistant) -> list[ServiceCall]:
    """Capture light service calls and update light state accordingly.

    This fixture:
    - Registers light.turn_on and light.turn_off services
    - Captures all service calls to a list for assertion
    - Updates the mock light state based on the call
    """
    calls: list[ServiceCall] = []

    async def mock_turn_on(call: ServiceCall) -> None:
        """Handle turn_on service call."""
        calls.append(call)

        entity_id = call.data.get(ATTR_ENTITY_ID)
        if entity_id:
            entity_ids = entity_id if isinstance(entity_id, list) else [entity_id]
            for eid in entity_ids:
                current_state = hass.states.get(eid)
                if current_state:
                    current_attrs = dict(current_state.attributes)
                    if ATTR_BRIGHTNESS in call.data:
                        current_attrs[ATTR_BRIGHTNESS] = call.data[ATTR_BRIGHTNESS]
                    elif current_attrs.get(ATTR_BRIGHTNESS) is None:
                        # If turning on without brightness and light was off, set default
                        current_attrs[ATTR_BRIGHTNESS] = 255
                    hass.states.async_set(eid, STATE_ON, current_attrs)

    async def mock_turn_off(call: ServiceCall) -> None:
        """Handle turn_off service call."""
        calls.append(call)

        entity_id = call.data.get(ATTR_ENTITY_ID)
        if entity_id:
            entity_ids = entity_id if isinstance(entity_id, list) else [entity_id]
            for eid in entity_ids:
                current_state = hass.states.get(eid)
                if current_state:
                    current_attrs = dict(current_state.attributes)
                    current_attrs[ATTR_BRIGHTNESS] = None
                    hass.states.async_set(eid, STATE_OFF, current_attrs)

    hass.services.async_register("light", "turn_on", mock_turn_on)
    hass.services.async_register("light", "turn_off", mock_turn_off)

    return calls


def _get_turn_on_calls(calls: list[ServiceCall]) -> list[ServiceCall]:
    """Filter for turn_on calls only."""
    return [c for c in calls if c.service == "turn_on"]


async def test_restore_brightness_on_turn_on(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test light restored to orig brightness when turned on."""
    entity_id = "light.test_restore"
    stored_brightness = 180

    # Create mock storage with stored brightness (nested dict format)
    mock_storage_data = {
        entity_id: {"orig_brightness": stored_brightness},
    }

    mock_config_entry.add_to_hass(hass)

    mock_store = AsyncMock()
    mock_store.async_load = AsyncMock(return_value=mock_storage_data)
    mock_store.async_save = AsyncMock()

    with patch(
        "custom_components.fade_lights.Store",
        return_value=mock_store,
    ):
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()

    # Create dimmable light that is OFF
    hass.states.async_set(
        entity_id,
        STATE_OFF,
        {
            ATTR_BRIGHTNESS: None,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )
    await hass.async_block_till_done()

    # Simulate turning the light ON at a different brightness (e.g., 100)
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 100,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )
    await hass.async_block_till_done()

    # The integration should have called turn_on to restore to stored brightness
    turn_on_calls = _get_turn_on_calls(service_calls)
    assert len(turn_on_calls) >= 1

    # The restore call should set the stored brightness
    restore_call = turn_on_calls[-1]
    assert restore_call.data.get(ATTR_ENTITY_ID) == entity_id
    assert restore_call.data.get(ATTR_BRIGHTNESS) == stored_brightness


async def test_no_restore_if_no_stored_brightness(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test no restore if orig_brightness is 0 or not stored."""
    entity_id = "light.test_no_stored"

    # No stored brightness data
    mock_config_entry.add_to_hass(hass)

    mock_store = AsyncMock()
    mock_store.async_load = AsyncMock(return_value={})
    mock_store.async_save = AsyncMock()

    with patch(
        "custom_components.fade_lights.Store",
        return_value=mock_store,
    ):
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()

    # Create dimmable light that is OFF
    hass.states.async_set(
        entity_id,
        STATE_OFF,
        {
            ATTR_BRIGHTNESS: None,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )
    await hass.async_block_till_done()

    # Turn the light ON
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 100,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )
    await hass.async_block_till_done()

    # No restore calls should have been made since no stored brightness
    turn_on_calls = _get_turn_on_calls(service_calls)
    assert len(turn_on_calls) == 0


async def test_no_restore_if_already_at_orig(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test no extra call if turned on at correct brightness."""
    entity_id = "light.test_already_correct"
    stored_brightness = 150

    # Create mock storage with stored brightness (nested dict format)
    mock_storage_data = {
        entity_id: {"orig_brightness": stored_brightness},
    }

    mock_config_entry.add_to_hass(hass)

    mock_store = AsyncMock()
    mock_store.async_load = AsyncMock(return_value=mock_storage_data)
    mock_store.async_save = AsyncMock()

    with patch(
        "custom_components.fade_lights.Store",
        return_value=mock_store,
    ):
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()

    # Create dimmable light that is OFF
    hass.states.async_set(
        entity_id,
        STATE_OFF,
        {
            ATTR_BRIGHTNESS: None,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )
    await hass.async_block_till_done()

    # Turn the light ON at the exact stored brightness
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: stored_brightness,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )
    await hass.async_block_till_done()

    # No restore call needed since already at correct brightness
    turn_on_calls = _get_turn_on_calls(service_calls)
    assert len(turn_on_calls) == 0


async def test_no_restore_for_non_dimmable(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test non-dimmable lights skip restoration."""
    entity_id = "light.test_non_dimmable"
    stored_brightness = 180

    # Create mock storage with stored brightness (nested dict format)
    mock_storage_data = {
        entity_id: {"orig_brightness": stored_brightness},
    }

    mock_config_entry.add_to_hass(hass)

    mock_store = AsyncMock()
    mock_store.async_load = AsyncMock(return_value=mock_storage_data)
    mock_store.async_save = AsyncMock()

    with patch(
        "custom_components.fade_lights.Store",
        return_value=mock_store,
    ):
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()

    # Create NON-dimmable light that is OFF (ColorMode.ONOFF instead of BRIGHTNESS)
    hass.states.async_set(
        entity_id,
        STATE_OFF,
        {
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.ONOFF],
        },
    )
    await hass.async_block_till_done()

    # Turn the light ON
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.ONOFF],
        },
    )
    await hass.async_block_till_done()

    # No restore call should be made for non-dimmable lights
    turn_on_calls = _get_turn_on_calls(service_calls)
    assert len(turn_on_calls) == 0


async def test_storage_persists_across_reload(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test stored brightness survives integration reload."""
    entity_id = "light.test_persist"
    stored_brightness = 200

    # Create mock storage with stored brightness (nested dict format)
    mock_storage_data = {
        entity_id: {"orig_brightness": stored_brightness},
    }

    mock_config_entry.add_to_hass(hass)

    mock_store = AsyncMock()
    mock_store.async_load = AsyncMock(return_value=mock_storage_data)
    mock_store.async_save = AsyncMock()

    with patch(
        "custom_components.fade_lights.Store",
        return_value=mock_store,
    ):
        # First setup
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()

        # Verify storage data is loaded
        assert DOMAIN in hass.data
        stored_orig = hass.data[DOMAIN]["data"].get(entity_id, {}).get("orig_brightness", 0)
        assert stored_orig == stored_brightness

        # Unload the integration
        await hass.config_entries.async_unload(mock_config_entry.entry_id)
        await hass.async_block_till_done()

        # Re-setup (simulating reload)
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()

    # Verify storage data is still available after reload
    assert DOMAIN in hass.data
    stored_orig = hass.data[DOMAIN]["data"].get(entity_id, {}).get("orig_brightness", 0)
    assert stored_orig == stored_brightness

    # Create dimmable light that is OFF
    hass.states.async_set(
        entity_id,
        STATE_OFF,
        {
            ATTR_BRIGHTNESS: None,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )
    await hass.async_block_till_done()

    # Turn on at different brightness
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 50,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )
    await hass.async_block_till_done()

    # Should restore to stored brightness
    turn_on_calls = _get_turn_on_calls(service_calls)
    assert len(turn_on_calls) >= 1
    restore_call = turn_on_calls[-1]
    assert restore_call.data.get(ATTR_BRIGHTNESS) == stored_brightness


async def test_restore_uses_correct_brightness(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test restores to exact stored value."""
    entity_id = "light.test_exact_restore"
    # Use a specific non-round number to verify exact restoration
    stored_brightness = 137

    # Nested dict format: entity_id -> config dict
    mock_storage_data = {
        entity_id: {"orig_brightness": stored_brightness},
    }

    mock_config_entry.add_to_hass(hass)

    mock_store = AsyncMock()
    mock_store.async_load = AsyncMock(return_value=mock_storage_data)
    mock_store.async_save = AsyncMock()

    with patch(
        "custom_components.fade_lights.Store",
        return_value=mock_store,
    ):
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()

    # Create dimmable light that is OFF
    hass.states.async_set(
        entity_id,
        STATE_OFF,
        {
            ATTR_BRIGHTNESS: None,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )
    await hass.async_block_till_done()

    # Turn on at a completely different brightness
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 50,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )
    await hass.async_block_till_done()

    # Verify the exact stored brightness value is used for restoration
    turn_on_calls = _get_turn_on_calls(service_calls)
    assert len(turn_on_calls) >= 1

    restore_call = turn_on_calls[-1]
    assert restore_call.data.get(ATTR_ENTITY_ID) == entity_id
    # The key assertion: exact value must match
    assert restore_call.data.get(ATTR_BRIGHTNESS) == stored_brightness


async def test_no_restore_for_excluded_light(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test that excluded lights don't get brightness restored."""
    entity_id = "light.excluded"

    # Configure as excluded with stored brightness
    hass.data[DOMAIN]["data"][entity_id] = {
        "orig_brightness": 200,
        "exclude": True,
    }

    # Start with light off
    hass.states.async_set(
        entity_id,
        STATE_OFF,
        {ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS]},
    )
    await hass.async_block_till_done()
    service_calls.clear()

    # Turn on at low brightness
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 50,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )
    await hass.async_block_till_done()

    # Should NOT have called turn_on to restore brightness
    turn_on_calls = [c for c in service_calls if c.service == "turn_on"]
    assert len(turn_on_calls) == 0
