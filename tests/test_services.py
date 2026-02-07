"""Tests for Fado service parameter handling."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from homeassistant.components.light import ATTR_BRIGHTNESS, ATTR_SUPPORTED_COLOR_MODES
from homeassistant.components.light.const import ColorMode
from homeassistant.const import ATTR_ENTITY_ID, STATE_ON
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.fado.const import (
    ATTR_BRIGHTNESS_PCT,
    ATTR_FROM,
    ATTR_TRANSITION,
    DEFAULT_TRANSITION,
    DOMAIN,
    SERVICE_FADO,
)


async def test_service_accepts_single_entity(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
) -> None:
    """Test service works with a single entity_id target."""
    with patch(
        "custom_components.fado.coordinator.FadeCoordinator._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade_light:
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADO,
            {
                ATTR_BRIGHTNESS_PCT: 50,
                ATTR_TRANSITION: 2,
            },
            target={"entity_id": mock_light_entity},
            blocking=True,
        )
        await hass.async_block_till_done()

        # Verify _fade_light was called once for the single entity
        assert mock_fade_light.call_count == 1
        call_args = mock_fade_light.call_args
        assert call_args[0][0] == mock_light_entity  # entity_id
        assert call_args[0][1].brightness_pct == 50  # fade_params.brightness_pct
        assert call_args[0][1].transition_ms == 2000  # transition_ms (2 seconds * 1000)


async def test_service_accepts_entity_list(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
    mock_light_off: str,
) -> None:
    """Test service works with a list of entity_ids in target."""
    with patch(
        "custom_components.fado.coordinator.FadeCoordinator._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade_light:
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADO,
            {
                ATTR_BRIGHTNESS_PCT: 75,
                ATTR_TRANSITION: 5,
            },
            target={"entity_id": [mock_light_entity, mock_light_off]},
            blocking=True,
        )
        await hass.async_block_till_done()

        # Verify _fade_light was called once for each entity
        assert mock_fade_light.call_count == 2

        # Collect all entity_ids that were called
        called_entity_ids = {call[0][0] for call in mock_fade_light.call_args_list}
        assert called_entity_ids == {mock_light_entity, mock_light_off}

        # Verify brightness and transition for all calls
        for call in mock_fade_light.call_args_list:
            assert call[0][1].brightness_pct == 75  # fade_params.brightness_pct
            assert call[0][1].transition_ms == 5000


async def test_service_expands_light_groups(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_group: str,
    mock_light_entity: str,
    mock_light_off: str,
) -> None:
    """Test service expands light groups to individual lights."""
    with patch(
        "custom_components.fado.coordinator.FadeCoordinator._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade_light:
        # Call with the group entity
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADO,
            {
                ATTR_BRIGHTNESS_PCT: 60,
                ATTR_TRANSITION: 3,
            },
            target={"entity_id": mock_light_group},
            blocking=True,
        )
        await hass.async_block_till_done()

        # Verify _fade_light was called for the individual lights in the group,
        # not for the group itself
        assert mock_fade_light.call_count == 2

        called_entity_ids = {call[0][0] for call in mock_fade_light.call_args_list}
        # Should have the individual lights, not the group
        assert mock_light_entity in called_entity_ids
        assert mock_light_off in called_entity_ids
        assert mock_light_group not in called_entity_ids


async def test_service_accepts_missing_brightness(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
) -> None:
    """Test service accepts missing brightness_pct (passes None to fade)."""
    with patch(
        "custom_components.fado.coordinator.FadeCoordinator._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade_light:
        # Call with from brightness but no target brightness_pct
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADO,
            {
                ATTR_FROM: {ATTR_BRIGHTNESS_PCT: 20},
                ATTR_TRANSITION: 2,
            },
            target={"entity_id": mock_light_entity},
            blocking=True,
        )
        await hass.async_block_till_done()

        # Verify _fade_light was called with None brightness target
        assert mock_fade_light.call_count == 1
        call_args = mock_fade_light.call_args
        # fade_params.brightness_pct should be None when not provided as target
        assert call_args[0][1].brightness_pct is None
        # But from_brightness_pct should be set
        assert call_args[0][1].from_brightness_pct == 20


async def test_service_uses_default_transition(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
) -> None:
    """Test service uses default transition when transition is not provided."""
    with patch(
        "custom_components.fado.coordinator.FadeCoordinator._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade_light:
        # Call without transition
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADO,
            {
                ATTR_BRIGHTNESS_PCT: 50,
            },
            target={"entity_id": mock_light_entity},
            blocking=True,
        )
        await hass.async_block_till_done()

        # Verify _fade_light was called with default transition (converted to ms)
        assert mock_fade_light.call_count == 1
        call_args = mock_fade_light.call_args
        assert call_args[0][1].transition_ms == DEFAULT_TRANSITION * 1000


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
        "custom_components.fado.coordinator.FadeCoordinator._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade_light:
        # mock_light_entity is both specified directly and is part of mock_light_group
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADO,
            {
                ATTR_BRIGHTNESS_PCT: 50,
                ATTR_TRANSITION: 1,
            },
            target={"entity_id": [mock_light_entity, mock_light_group]},
            blocking=True,
        )
        await hass.async_block_till_done()

        # mock_light_group contains mock_light_entity and mock_light_off
        # So we should only have 2 unique entities, not 3
        assert mock_fade_light.call_count == 2

        # Verify each entity was only called once
        called_entity_ids = [call[0][0] for call in mock_fade_light.call_args_list]
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
        "custom_components.fado.coordinator.FadeCoordinator._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade_light:
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADO,
            {
                ATTR_BRIGHTNESS_PCT: 50,
                ATTR_TRANSITION: 1,
            },
            target={"entity_id": outer_group},
            blocking=True,
        )
        await hass.async_block_till_done()

        # Should expand to the 2 individual lights
        assert mock_fade_light.call_count == 2

        called_entity_ids = {call[0][0] for call in mock_fade_light.call_args_list}
        assert called_entity_ids == {mock_light_entity, mock_light_off}
        # Groups should not be in the called entities
        assert inner_group not in called_entity_ids
        assert outer_group not in called_entity_ids


async def test_service_requires_target(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test service requires at least one target to be specified."""
    import pytest
    from voluptuous.error import MultipleInvalid

    # Call service without target should raise validation error
    with pytest.raises(MultipleInvalid, match="must contain at least one of"):
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADO,
            {
                ATTR_BRIGHTNESS_PCT: 50,
                ATTR_TRANSITION: 1,
            },
            blocking=True,
        )


async def test_service_expands_group_with_string_entity_id(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
) -> None:
    """Test service handles groups where entity_id attribute is a string (not list)."""
    # Create a group where entity_id is a single string, not a list
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
        "custom_components.fado.coordinator.FadeCoordinator._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade_light:
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADO,
            {
                ATTR_BRIGHTNESS_PCT: 50,
                ATTR_TRANSITION: 1,
            },
            target={"entity_id": group_with_string},
            blocking=True,
        )
        await hass.async_block_till_done()

        # Should expand to the individual light
        assert mock_fade_light.call_count == 1
        called_entity_ids = {call[0][0] for call in mock_fade_light.call_args_list}
        assert called_entity_ids == {mock_light_entity}


async def test_service_filters_non_light_entities_from_target(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
) -> None:
    """Test service filters out non-light entities when expanded from groups."""
    # Create a group that (incorrectly) contains a non-light entity
    mixed_group = "light.mixed_group"
    non_light = "sensor.temperature"
    hass.states.async_set(non_light, "25")
    hass.states.async_set(
        mixed_group,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 150,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
            ATTR_ENTITY_ID: [mock_light_entity, non_light],
        },
    )

    with patch(
        "custom_components.fado.coordinator.FadeCoordinator._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade_light:
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADO,
            {
                ATTR_BRIGHTNESS_PCT: 50,
                ATTR_TRANSITION: 1,
            },
            target={"entity_id": mixed_group},
            blocking=True,
        )
        await hass.async_block_till_done()

        # Should only fade the light, not the sensor
        assert mock_fade_light.call_count == 1
        called_entity_ids = {call[0][0] for call in mock_fade_light.call_args_list}
        assert called_entity_ids == {mock_light_entity}


async def test_service_accepts_device_id_target(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test service resolves device_id to light entities."""
    # Set up registries
    device_reg = dr.async_get(hass)
    entity_reg = er.async_get(hass)

    # Create a device
    config_entry = MockConfigEntry(domain="test", data={})
    config_entry.add_to_hass(hass)
    device = device_reg.async_get_or_create(
        config_entry_id=config_entry.entry_id,
        identifiers={("test", "device_1")},
        name="Test Device",
    )

    # Register a light entity associated with the device
    entity_entry = entity_reg.async_get_or_create(
        domain="light",
        platform="test",
        unique_id="light_on_device",
        device_id=device.id,
    )
    entity_id = entity_entry.entity_id

    # Set up the state for the entity
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 128,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    with patch(
        "custom_components.fado.coordinator.FadeCoordinator._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade_light:
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADO,
            {
                ATTR_BRIGHTNESS_PCT: 50,
                ATTR_TRANSITION: 1,
            },
            target={"device_id": device.id},
            blocking=True,
        )
        await hass.async_block_till_done()

        # Should resolve device to its light entity
        assert mock_fade_light.call_count == 1
        called_entity_ids = {call[0][0] for call in mock_fade_light.call_args_list}
        assert entity_id in called_entity_ids


async def test_service_accepts_area_id_target(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test service resolves area_id to light entities."""
    # Set up registries
    area_reg = ar.async_get(hass)
    entity_reg = er.async_get(hass)

    # Create an area
    area = area_reg.async_create("Living Room")

    # Register a light entity associated with the area
    entity_entry = entity_reg.async_get_or_create(
        domain="light",
        platform="test",
        unique_id="light_in_area",
    )
    entity_reg.async_update_entity(entity_entry.entity_id, area_id=area.id)
    entity_id = entity_entry.entity_id

    # Set up the state for the entity
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 128,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    with patch(
        "custom_components.fado.coordinator.FadeCoordinator._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade_light:
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADO,
            {
                ATTR_BRIGHTNESS_PCT: 50,
                ATTR_TRANSITION: 1,
            },
            target={"area_id": area.id},
            blocking=True,
        )
        await hass.async_block_till_done()

        # Should resolve area to its light entity
        assert mock_fade_light.call_count == 1
        called_entity_ids = {call[0][0] for call in mock_fade_light.call_args_list}
        assert entity_id in called_entity_ids


async def test_service_accepts_multiple_target_types(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
) -> None:
    """Test service handles mixed target types (entity_id + area_id)."""
    # Set up registries
    area_reg = ar.async_get(hass)
    entity_reg = er.async_get(hass)

    # Create an area with a different light
    area = area_reg.async_create("Bedroom")
    entity_entry = entity_reg.async_get_or_create(
        domain="light",
        platform="test",
        unique_id="light_in_bedroom",
    )
    entity_reg.async_update_entity(entity_entry.entity_id, area_id=area.id)
    area_light = entity_entry.entity_id

    # Set up the state for the area light
    hass.states.async_set(
        area_light,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 128,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    with patch(
        "custom_components.fado.coordinator.FadeCoordinator._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade_light:
        # Target both a specific entity and an area
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADO,
            {
                ATTR_BRIGHTNESS_PCT: 50,
                ATTR_TRANSITION: 1,
            },
            target={
                "entity_id": mock_light_entity,
                "area_id": area.id,
            },
            blocking=True,
        )
        await hass.async_block_till_done()

        # Should fade both the direct entity and the area's light
        assert mock_fade_light.call_count == 2
        called_entity_ids = {call[0][0] for call in mock_fade_light.call_args_list}
        assert mock_light_entity in called_entity_ids
        assert area_light in called_entity_ids


async def test_service_filters_non_light_entities_from_device(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test service only targets light entities from a device (not sensors etc)."""
    # Set up registries
    device_reg = dr.async_get(hass)
    entity_reg = er.async_get(hass)

    # Create a device
    config_entry = MockConfigEntry(domain="test", data={})
    config_entry.add_to_hass(hass)
    device = device_reg.async_get_or_create(
        config_entry_id=config_entry.entry_id,
        identifiers={("test", "device_2")},
        name="Multi-Entity Device",
    )

    # Register a light entity on the device
    light_entry = entity_reg.async_get_or_create(
        domain="light",
        platform="test",
        unique_id="device_light",
        device_id=device.id,
    )
    light_id = light_entry.entity_id

    # Register a sensor entity on the same device
    sensor_entry = entity_reg.async_get_or_create(
        domain="sensor",
        platform="test",
        unique_id="device_sensor",
        device_id=device.id,
    )
    sensor_id = sensor_entry.entity_id

    # Set up states
    hass.states.async_set(
        light_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 128,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )
    hass.states.async_set(sensor_id, "25")

    with patch(
        "custom_components.fado.coordinator.FadeCoordinator._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade_light:
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADO,
            {
                ATTR_BRIGHTNESS_PCT: 50,
                ATTR_TRANSITION: 1,
            },
            target={"device_id": device.id},
            blocking=True,
        )
        await hass.async_block_till_done()

        # Should only fade the light, not the sensor
        assert mock_fade_light.call_count == 1
        called_entity_ids = {call[0][0] for call in mock_fade_light.call_args_list}
        assert light_id in called_entity_ids
        assert sensor_id not in called_entity_ids


async def test_service_excludes_configured_lights(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    captured_calls: list,
) -> None:
    """Test that lights with exclude=True are filtered from service calls."""
    # Set up two lights
    hass.states.async_set(
        "light.included",
        STATE_ON,
        {ATTR_BRIGHTNESS: 200, ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS]},
    )
    hass.states.async_set(
        "light.excluded",
        STATE_ON,
        {ATTR_BRIGHTNESS: 200, ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS]},
    )
    await hass.async_block_till_done()

    # Configure one light as excluded
    hass.data[DOMAIN].data["light.excluded"] = {"exclude": True}

    # Call service targeting both
    await hass.services.async_call(
        DOMAIN,
        SERVICE_FADO,
        {
            "entity_id": ["light.included", "light.excluded"],
            ATTR_BRIGHTNESS_PCT: 50,
            ATTR_TRANSITION: 0.1,
        },
        blocking=True,
    )

    # Only included light should have changed
    assert hass.states.get("light.included").attributes[ATTR_BRIGHTNESS] == 127
    # Excluded light unchanged
    assert hass.states.get("light.excluded").attributes[ATTR_BRIGHTNESS] == 200
