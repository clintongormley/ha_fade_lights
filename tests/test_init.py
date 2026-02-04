"""Tests for Fade Lights integration initialization."""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, patch

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.fade_lights import (
    ACTIVE_FADES,
    FADE_CANCEL_EVENTS,
    FADE_EXPECTED_BRIGHTNESS,
)
from custom_components.fade_lights.const import DOMAIN, SERVICE_FADE_LIGHTS


async def test_setup_entry_registers_service(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test the integration registers the fade_lights service."""
    assert DOMAIN in hass.data
    assert hass.services.has_service(DOMAIN, SERVICE_FADE_LIGHTS)


async def test_setup_entry_loads_storage(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Test the integration loads storage data on setup."""
    mock_config_entry.add_to_hass(hass)

    # Create mock storage data
    mock_storage_data = {
        "light_test_light": {"orig": 200, "curr": 150},
        "light_another_light": {"orig": 100, "curr": 100},
    }

    mock_store = AsyncMock()
    mock_store.async_load = AsyncMock(return_value=mock_storage_data)
    mock_store.async_save = AsyncMock()

    with patch(
        "custom_components.fade_lights.Store",
        return_value=mock_store,
    ):
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()

    # Verify storage was loaded
    mock_store.async_load.assert_called_once()

    # Verify data is accessible in hass.data
    assert DOMAIN in hass.data
    assert hass.data[DOMAIN]["data"] == mock_storage_data
    assert hass.data[DOMAIN]["store"] is mock_store


async def test_unload_entry_removes_service(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test the integration removes the fade_lights service on unload."""
    # Verify service exists before unload
    assert hass.services.has_service(DOMAIN, SERVICE_FADE_LIGHTS)

    await hass.config_entries.async_unload(init_integration.entry_id)
    await hass.async_block_till_done()

    # Verify service is removed after unload
    assert not hass.services.has_service(DOMAIN, SERVICE_FADE_LIGHTS)


async def test_unload_entry_clears_tracking_dicts(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test the integration clears tracking dicts on unload."""
    # Populate the tracking dicts to simulate an active fade
    test_task = asyncio.create_task(asyncio.sleep(10))
    test_event = asyncio.Event()

    ACTIVE_FADES["light.test_light"] = test_task
    FADE_CANCEL_EVENTS["light.test_light"] = test_event
    FADE_EXPECTED_BRIGHTNESS["light.test_light"] = 128

    # Unload the integration
    await hass.config_entries.async_unload(init_integration.entry_id)
    await hass.async_block_till_done()

    # Verify all tracking dicts are cleared
    assert len(ACTIVE_FADES) == 0
    assert len(FADE_CANCEL_EVENTS) == 0
    assert len(FADE_EXPECTED_BRIGHTNESS) == 0

    # Verify the cancel event was set (to stop any active fades)
    assert test_event.is_set()

    # Clean up the test task
    test_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await test_task


async def test_unload_entry_state(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test the config entry state is NOT_LOADED after unload."""
    # Verify entry is loaded before unload
    assert init_integration.state is ConfigEntryState.LOADED

    await hass.config_entries.async_unload(init_integration.entry_id)
    await hass.async_block_till_done()

    # Verify entry state is NOT_LOADED after unload
    assert init_integration.state is ConfigEntryState.NOT_LOADED

    # Verify domain is removed from hass.data
    assert DOMAIN not in hass.data


async def test_options_update_reloads_entry(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test that updating options triggers a config entry reload."""
    # Verify entry is loaded
    assert init_integration.state is ConfigEntryState.LOADED

    # Update options with a patch to track the reload
    with patch(
        "custom_components.fade_lights.Store",
        return_value=AsyncMock(async_load=AsyncMock(return_value={}), async_save=AsyncMock()),
    ):
        hass.config_entries.async_update_entry(
            init_integration,
            options={"default_brightness_pct": 50},
        )
        await hass.async_block_till_done()

    # Entry should still be loaded after reload completes
    assert init_integration.state is ConfigEntryState.LOADED

    # Verify the service is still available (entry was reloaded, not just unloaded)
    assert hass.services.has_service(DOMAIN, SERVICE_FADE_LIGHTS)


async def test_async_setup_auto_import_when_no_entries(
    hass: HomeAssistant,
) -> None:
    """Test async_setup triggers config flow when no entries exist."""
    from custom_components.fade_lights import async_setup

    # Track if flow was initiated
    flow_init_called = False
    original_async_init = hass.config_entries.flow.async_init

    async def mock_async_init(domain, *, context=None):
        nonlocal flow_init_called
        if domain == DOMAIN and context and context.get("source") == "import":
            flow_init_called = True
        return await original_async_init(domain, context=context)

    with patch.object(hass.config_entries.flow, "async_init", side_effect=mock_async_init):
        result = await async_setup(hass, {})

    assert result is True
    # Wait for the task to complete
    await hass.async_block_till_done()
    assert flow_init_called, "Config flow should be initiated when no entries exist"


async def test_async_setup_no_auto_import_when_entry_exists(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Test async_setup does not trigger config flow when entries exist."""
    from custom_components.fade_lights import async_setup

    # Add entry to hass before calling async_setup
    mock_config_entry.add_to_hass(hass)

    flow_init_called = False
    original_async_init = hass.config_entries.flow.async_init

    async def mock_async_init(domain, *, context=None):
        nonlocal flow_init_called
        if domain == DOMAIN and context and context.get("source") == "import":
            flow_init_called = True
        return await original_async_init(domain, context=context)

    with patch.object(hass.config_entries.flow, "async_init", side_effect=mock_async_init):
        result = await async_setup(hass, {})

    assert result is True
    await hass.async_block_till_done()
    assert not flow_init_called, "Config flow should NOT be initiated when entries exist"
