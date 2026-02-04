"""Tests for Fade Lights service parameter handling."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.components.light import ATTR_BRIGHTNESS, ATTR_SUPPORTED_COLOR_MODES
from homeassistant.components.light.const import ColorMode
from homeassistant.const import ATTR_ENTITY_ID, STATE_ON
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.fade_lights.const import (
    ATTR_BRIGHTNESS_PCT,
    ATTR_TRANSITION,
    DEFAULT_TRANSITION,
    DOMAIN,
    SERVICE_FADE_LIGHTS,
)


async def test_service_accepts_single_entity(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
) -> None:
    """Test service works with a single entity_id."""
    with patch(
        "custom_components.fade_lights._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade_light:
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_ENTITY_ID: mock_light_entity,
                ATTR_BRIGHTNESS_PCT: 50,
                ATTR_TRANSITION: 2,
            },
            blocking=True,
        )
        await hass.async_block_till_done()

        # Verify _fade_light was called once for the single entity
        assert mock_fade_light.call_count == 1
        call_args = mock_fade_light.call_args
        assert call_args[0][1] == mock_light_entity  # entity_id
        assert call_args[0][2].brightness_pct == 50  # fade_params.brightness_pct
        assert call_args[0][3] == 2000  # transition_ms (2 seconds * 1000)


async def test_service_accepts_entity_list(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
    mock_light_off: str,
) -> None:
    """Test service works with a list of entity_ids."""
    with patch(
        "custom_components.fade_lights._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade_light:
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_ENTITY_ID: [mock_light_entity, mock_light_off],
                ATTR_BRIGHTNESS_PCT: 75,
                ATTR_TRANSITION: 5,
            },
            blocking=True,
        )
        await hass.async_block_till_done()

        # Verify _fade_light was called once for each entity
        assert mock_fade_light.call_count == 2

        # Collect all entity_ids that were called
        called_entity_ids = {call[0][1] for call in mock_fade_light.call_args_list}
        assert called_entity_ids == {mock_light_entity, mock_light_off}

        # Verify brightness and transition for all calls
        for call in mock_fade_light.call_args_list:
            assert call[0][2].brightness_pct == 75  # fade_params.brightness_pct
            assert call[0][3] == 5000  # transition_ms


async def test_service_accepts_comma_string(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
    mock_light_off: str,
) -> None:
    """Test service works with comma-separated entity_id string."""
    with patch(
        "custom_components.fade_lights._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade_light:
        # Pass entities as comma-separated string
        comma_string = f"{mock_light_entity}, {mock_light_off}"
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_ENTITY_ID: comma_string,
                ATTR_BRIGHTNESS_PCT: 30,
                ATTR_TRANSITION: 1,
            },
            blocking=True,
        )
        await hass.async_block_till_done()

        # Verify _fade_light was called once for each entity
        assert mock_fade_light.call_count == 2

        # Collect all entity_ids that were called
        called_entity_ids = {call[0][1] for call in mock_fade_light.call_args_list}
        assert called_entity_ids == {mock_light_entity, mock_light_off}


async def test_service_expands_light_groups(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_group: str,
    mock_light_entity: str,
    mock_light_off: str,
) -> None:
    """Test service expands light groups to individual lights."""
    with patch(
        "custom_components.fade_lights._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade_light:
        # Call with the group entity
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_ENTITY_ID: mock_light_group,
                ATTR_BRIGHTNESS_PCT: 60,
                ATTR_TRANSITION: 3,
            },
            blocking=True,
        )
        await hass.async_block_till_done()

        # Verify _fade_light was called for the individual lights in the group,
        # not for the group itself
        assert mock_fade_light.call_count == 2

        called_entity_ids = {call[0][1] for call in mock_fade_light.call_args_list}
        # Should have the individual lights, not the group
        assert mock_light_entity in called_entity_ids
        assert mock_light_off in called_entity_ids
        assert mock_light_group not in called_entity_ids


async def test_service_rejects_non_light_entity(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test service raises ServiceValidationError for non-light entities."""
    # Create a non-light entity (e.g., a sensor)
    non_light_entity = "sensor.temperature"
    hass.states.async_set(non_light_entity, "25")

    with pytest.raises(ServiceValidationError, match="is not a light"):
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_ENTITY_ID: non_light_entity,
                ATTR_BRIGHTNESS_PCT: 50,
            },
            blocking=True,
        )


async def test_service_accepts_missing_brightness(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
) -> None:
    """Test service accepts missing brightness_pct (passes None to fade)."""
    with patch(
        "custom_components.fade_lights._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade_light:
        # Call without brightness_pct
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_ENTITY_ID: mock_light_entity,
                ATTR_TRANSITION: 2,
            },
            blocking=True,
        )
        await hass.async_block_till_done()

        # Verify _fade_light was called with None brightness
        assert mock_fade_light.call_count == 1
        call_args = mock_fade_light.call_args
        # fade_params.brightness_pct should be None when not provided
        assert call_args[0][2].brightness_pct is None


async def test_service_uses_default_transition(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
) -> None:
    """Test service uses default transition when transition is not provided."""
    with patch(
        "custom_components.fade_lights._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade_light:
        # Call without transition
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_ENTITY_ID: mock_light_entity,
                ATTR_BRIGHTNESS_PCT: 50,
            },
            blocking=True,
        )
        await hass.async_block_till_done()

        # Verify _fade_light was called with default transition (converted to ms)
        assert mock_fade_light.call_count == 1
        call_args = mock_fade_light.call_args
        assert call_args[0][3] == DEFAULT_TRANSITION * 1000  # transition_ms


async def test_service_deduplicates_entities(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
    mock_light_group: str,
) -> None:
    """Test service deduplicates entities that appear multiple times.

    If an entity is specified directly AND is also part of a group,
    it should only be faded once.
    """
    with patch(
        "custom_components.fade_lights._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade_light:
        # mock_light_entity is both specified directly and is part of mock_light_group
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_ENTITY_ID: [mock_light_entity, mock_light_group],
                ATTR_BRIGHTNESS_PCT: 50,
                ATTR_TRANSITION: 1,
            },
            blocking=True,
        )
        await hass.async_block_till_done()

        # mock_light_group contains mock_light_entity and mock_light_off
        # So we should only have 2 unique entities, not 3
        assert mock_fade_light.call_count == 2

        # Verify each entity was only called once
        called_entity_ids = [call[0][1] for call in mock_fade_light.call_args_list]
        assert len(called_entity_ids) == len(set(called_entity_ids))


async def test_service_expands_nested_groups(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
    mock_light_off: str,
) -> None:
    """Test service expands nested light groups recursively."""
    # Create a nested group structure:
    # outer_group -> inner_group -> mock_light_entity
    #             -> mock_light_off
    inner_group = "light.inner_group"
    hass.states.async_set(
        inner_group,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 150,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
            ATTR_ENTITY_ID: [mock_light_entity],
        },
    )

    outer_group = "light.outer_group"
    hass.states.async_set(
        outer_group,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 150,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
            ATTR_ENTITY_ID: [inner_group, mock_light_off],
        },
    )

    with patch(
        "custom_components.fade_lights._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade_light:
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_ENTITY_ID: outer_group,
                ATTR_BRIGHTNESS_PCT: 50,
                ATTR_TRANSITION: 1,
            },
            blocking=True,
        )
        await hass.async_block_till_done()

        # Should expand to the 2 individual lights
        assert mock_fade_light.call_count == 2

        called_entity_ids = {call[0][1] for call in mock_fade_light.call_args_list}
        assert called_entity_ids == {mock_light_entity, mock_light_off}
        # Groups should not be in the called entities
        assert inner_group not in called_entity_ids
        assert outer_group not in called_entity_ids


async def test_service_handles_comma_string_with_whitespace(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
    mock_light_off: str,
) -> None:
    """Test service handles comma-separated strings with various whitespace."""
    with patch(
        "custom_components.fade_lights._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade_light:
        # Various whitespace patterns
        comma_string = f"  {mock_light_entity}  ,   {mock_light_off}  "
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_ENTITY_ID: comma_string,
                ATTR_BRIGHTNESS_PCT: 50,
                ATTR_TRANSITION: 1,
            },
            blocking=True,
        )
        await hass.async_block_till_done()

        assert mock_fade_light.call_count == 2

        called_entity_ids = {call[0][1] for call in mock_fade_light.call_args_list}
        assert called_entity_ids == {mock_light_entity, mock_light_off}


async def test_service_with_none_entity_id_does_nothing(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test service does nothing when entity_id is None."""
    with patch(
        "custom_components.fade_lights._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade_light:
        # Call service without entity_id (will be None)
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_BRIGHTNESS_PCT: 50,
                ATTR_TRANSITION: 1,
            },
            blocking=True,
        )
        await hass.async_block_till_done()

        # _fade_light should not have been called
        assert mock_fade_light.call_count == 0


async def test_service_expands_group_with_string_entity_id(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
) -> None:
    """Test service handles groups where entity_id attribute is a string (not list)."""
    # Create a group where entity_id is a single string, not a list
    # This covers line 794 in _expand_entity_ids
    group_with_string = "light.string_group"
    hass.states.async_set(
        group_with_string,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 150,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
            ATTR_ENTITY_ID: mock_light_entity,  # String instead of list
        },
    )

    with patch(
        "custom_components.fade_lights._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade_light:
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_ENTITY_ID: group_with_string,
                ATTR_BRIGHTNESS_PCT: 50,
                ATTR_TRANSITION: 1,
            },
            blocking=True,
        )
        await hass.async_block_till_done()

        # Should expand to the individual light
        assert mock_fade_light.call_count == 1
        called_entity_ids = {call[0][1] for call in mock_fade_light.call_args_list}
        assert called_entity_ids == {mock_light_entity}
