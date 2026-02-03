"""Tests for Fade Lights config flow."""

from __future__ import annotations

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.fade_lights.const import DOMAIN


async def test_user_flow_creates_entry(hass: HomeAssistant) -> None:
    """Test user flow creates config entry with title 'Fade Lights'."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Fade Lights"
    assert result["data"] == {}


async def test_single_instance_only(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Test second setup aborts with 'single_instance_allowed'."""
    mock_config_entry.add_to_hass(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "single_instance_allowed"


async def test_import_flow_creates_entry(hass: HomeAssistant) -> None:
    """Test import flow creates entry."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_IMPORT}, data={}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Fade Lights"
    assert result["data"] == {}


async def test_import_flow_single_instance(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Test import flow aborts if instance exists."""
    mock_config_entry.add_to_hass(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_IMPORT}, data={}
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "single_instance_allowed"
