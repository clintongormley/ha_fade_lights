"""Tests for Fade Lights config flow and options flow."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.fade_lights.const import (
    DEFAULT_BRIGHTNESS_PCT,
    DEFAULT_STEP_DELAY_MS,
    DEFAULT_TRANSITION,
    DOMAIN,
    OPTION_DEFAULT_BRIGHTNESS_PCT,
    OPTION_DEFAULT_TRANSITION,
    OPTION_STEP_DELAY_MS,
)


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


async def test_options_flow_shows_defaults(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Test options flow shows form with default values."""
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    result = await hass.config_entries.options.async_init(mock_config_entry.entry_id)

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"

    # Verify the schema contains our option fields with correct defaults
    schema = result["data_schema"]
    assert schema is not None

    schema_dict = {str(key): key for key in schema.schema}
    assert OPTION_DEFAULT_BRIGHTNESS_PCT in schema_dict
    assert OPTION_DEFAULT_TRANSITION in schema_dict
    assert OPTION_STEP_DELAY_MS in schema_dict

    # Check defaults by accessing the schema key defaults
    for key in schema.schema:
        if str(key) == OPTION_DEFAULT_BRIGHTNESS_PCT:
            assert key.default() == DEFAULT_BRIGHTNESS_PCT
        elif str(key) == OPTION_DEFAULT_TRANSITION:
            assert key.default() == DEFAULT_TRANSITION
        elif str(key) == OPTION_STEP_DELAY_MS:
            assert key.default() == DEFAULT_STEP_DELAY_MS


async def test_options_flow_updates_values(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Test options flow saves updated values."""
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    result = await hass.config_entries.options.async_init(mock_config_entry.entry_id)
    assert result["type"] is FlowResultType.FORM

    # Submit new values
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={
            OPTION_DEFAULT_BRIGHTNESS_PCT: 75,
            OPTION_DEFAULT_TRANSITION: 10,
            OPTION_STEP_DELAY_MS: 200,
        },
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {
        OPTION_DEFAULT_BRIGHTNESS_PCT: 75,
        OPTION_DEFAULT_TRANSITION: 10,
        OPTION_STEP_DELAY_MS: 200,
    }


async def test_options_flow_validates_brightness_range(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Test brightness validated 0-100."""
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    result = await hass.config_entries.options.async_init(mock_config_entry.entry_id)
    assert result["type"] is FlowResultType.FORM

    # Test invalid brightness above 100
    with pytest.raises(vol.Invalid):
        await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={
                OPTION_DEFAULT_BRIGHTNESS_PCT: 150,  # Invalid: > 100
                OPTION_DEFAULT_TRANSITION: DEFAULT_TRANSITION,
                OPTION_STEP_DELAY_MS: DEFAULT_STEP_DELAY_MS,
            },
        )

    # Start a new flow to test negative value
    result = await hass.config_entries.options.async_init(mock_config_entry.entry_id)

    with pytest.raises(vol.Invalid):
        await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={
                OPTION_DEFAULT_BRIGHTNESS_PCT: -10,  # Invalid: < 0
                OPTION_DEFAULT_TRANSITION: DEFAULT_TRANSITION,
                OPTION_STEP_DELAY_MS: DEFAULT_STEP_DELAY_MS,
            },
        )


async def test_options_flow_validates_transition_range(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Test transition validated 0-3600."""
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    result = await hass.config_entries.options.async_init(mock_config_entry.entry_id)
    assert result["type"] is FlowResultType.FORM

    # Test invalid transition above 3600
    with pytest.raises(vol.Invalid):
        await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={
                OPTION_DEFAULT_BRIGHTNESS_PCT: DEFAULT_BRIGHTNESS_PCT,
                OPTION_DEFAULT_TRANSITION: 5000,  # Invalid: > 3600
                OPTION_STEP_DELAY_MS: DEFAULT_STEP_DELAY_MS,
            },
        )

    # Start a new flow to test negative value
    result = await hass.config_entries.options.async_init(mock_config_entry.entry_id)

    with pytest.raises(vol.Invalid):
        await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={
                OPTION_DEFAULT_BRIGHTNESS_PCT: DEFAULT_BRIGHTNESS_PCT,
                OPTION_DEFAULT_TRANSITION: -5,  # Invalid: < 0
                OPTION_STEP_DELAY_MS: DEFAULT_STEP_DELAY_MS,
            },
        )

    # Start a new flow to test float value (should be valid)
    result = await hass.config_entries.options.async_init(mock_config_entry.entry_id)

    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={
            OPTION_DEFAULT_BRIGHTNESS_PCT: DEFAULT_BRIGHTNESS_PCT,
            OPTION_DEFAULT_TRANSITION: 0.5,  # Valid: float value for 500ms
            OPTION_STEP_DELAY_MS: DEFAULT_STEP_DELAY_MS,
        },
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"][OPTION_DEFAULT_TRANSITION] == 0.5


async def test_options_flow_validates_step_delay_range(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Test step_delay_ms validated with MIN_STEP_DELAY_MS (50) to 1000."""
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    result = await hass.config_entries.options.async_init(mock_config_entry.entry_id)
    assert result["type"] is FlowResultType.FORM

    # Test invalid step delay below MIN_STEP_DELAY_MS (50)
    with pytest.raises(vol.Invalid):
        await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={
                OPTION_DEFAULT_BRIGHTNESS_PCT: DEFAULT_BRIGHTNESS_PCT,
                OPTION_DEFAULT_TRANSITION: DEFAULT_TRANSITION,
                OPTION_STEP_DELAY_MS: 30,  # Invalid: < 50 (MIN_STEP_DELAY_MS)
            },
        )

    # Start a new flow to test value above max
    result = await hass.config_entries.options.async_init(mock_config_entry.entry_id)

    with pytest.raises(vol.Invalid):
        await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={
                OPTION_DEFAULT_BRIGHTNESS_PCT: DEFAULT_BRIGHTNESS_PCT,
                OPTION_DEFAULT_TRANSITION: DEFAULT_TRANSITION,
                OPTION_STEP_DELAY_MS: 1500,  # Invalid: > 1000
            },
        )

    # Start a new flow to test valid value at minimum (50)
    result = await hass.config_entries.options.async_init(mock_config_entry.entry_id)

    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={
            OPTION_DEFAULT_BRIGHTNESS_PCT: DEFAULT_BRIGHTNESS_PCT,
            OPTION_DEFAULT_TRANSITION: DEFAULT_TRANSITION,
            OPTION_STEP_DELAY_MS: 50,  # Valid: exactly MIN_STEP_DELAY_MS
        },
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"][OPTION_STEP_DELAY_MS] == 50
