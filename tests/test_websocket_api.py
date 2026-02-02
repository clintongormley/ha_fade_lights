"""Tests for WebSocket API."""

from unittest.mock import MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.fade_lights.const import DOMAIN


@pytest.fixture
def mock_registries(hass: HomeAssistant):
    """Set up mock registries with test data."""
    # Mock floor registry
    floor_upstairs = MagicMock()
    floor_upstairs.floor_id = "upstairs"
    floor_upstairs.name = "Upstairs"

    # Mock area registry
    area_bedroom = MagicMock()
    area_bedroom.id = "bedroom"
    area_bedroom.name = "Bedroom"
    area_bedroom.floor_id = "upstairs"

    area_kitchen = MagicMock()
    area_kitchen.id = "kitchen"
    area_kitchen.name = "Kitchen"
    area_kitchen.floor_id = None  # No floor

    # Mock entity registry
    entity_bedroom_light = MagicMock()
    entity_bedroom_light.entity_id = "light.bedroom_ceiling"
    entity_bedroom_light.name = "Bedroom Ceiling"
    entity_bedroom_light.original_name = "Bedroom Ceiling"
    entity_bedroom_light.area_id = "bedroom"
    entity_bedroom_light.device_id = None
    entity_bedroom_light.disabled_by = None
    entity_bedroom_light.platform = "hue"

    entity_kitchen_light = MagicMock()
    entity_kitchen_light.entity_id = "light.kitchen_main"
    entity_kitchen_light.name = "Kitchen Main"
    entity_kitchen_light.original_name = "Kitchen Main"
    entity_kitchen_light.area_id = "kitchen"
    entity_kitchen_light.device_id = None
    entity_kitchen_light.disabled_by = None
    entity_kitchen_light.platform = "hue"

    with patch(
        "homeassistant.helpers.floor_registry.async_get",
        return_value=MagicMock(floors={"upstairs": floor_upstairs}),
    ), patch(
        "homeassistant.helpers.area_registry.async_get",
        return_value=MagicMock(
            areas={"bedroom": area_bedroom, "kitchen": area_kitchen},
            async_get_area=lambda aid: {"bedroom": area_bedroom, "kitchen": area_kitchen}.get(aid),
        ),
    ), patch(
        "homeassistant.helpers.entity_registry.async_get",
        return_value=MagicMock(
            entities=MagicMock(
                values=lambda: [entity_bedroom_light, entity_kitchen_light]
            )
        ),
    ), patch(
        "homeassistant.helpers.device_registry.async_get",
        return_value=MagicMock(async_get=lambda did: None),
    ):
        yield


async def test_get_lights_returns_grouped_data(
    hass: HomeAssistant,
    init_integration,
    mock_registries,
) -> None:
    """Test get_lights returns lights grouped by floor and area."""
    from custom_components.fade_lights.websocket_api import async_get_lights

    # Set up storage with config for one light
    hass.data[DOMAIN]["data"]["light.bedroom_ceiling"] = {
        "min_delay_ms": 150,
        "exclude": False,
        "use_native_transition": True,
    }

    result = await async_get_lights(hass)

    assert "floors" in result
    # Find upstairs floor
    upstairs = next((f for f in result["floors"] if f["floor_id"] == "upstairs"), None)
    assert upstairs is not None
    assert upstairs["name"] == "Upstairs"

    # Find bedroom area
    bedroom = next((a for a in upstairs["areas"] if a["area_id"] == "bedroom"), None)
    assert bedroom is not None

    # Find bedroom light
    light = next(
        (l for l in bedroom["lights"] if l["entity_id"] == "light.bedroom_ceiling"),
        None,
    )
    assert light is not None
    assert light["min_delay_ms"] == 150


async def test_get_lights_handles_no_floor(
    hass: HomeAssistant,
    init_integration,
    mock_registries,
) -> None:
    """Test get_lights handles lights with no floor assignment."""
    from custom_components.fade_lights.websocket_api import async_get_lights

    result = await async_get_lights(hass)

    # Find "No Floor" entry
    no_floor = next((f for f in result["floors"] if f["floor_id"] is None), None)
    assert no_floor is not None
    assert no_floor["name"] == "No Floor"

    # Kitchen should be in "No Floor"
    kitchen = next((a for a in no_floor["areas"] if a["area_id"] == "kitchen"), None)
    assert kitchen is not None
    assert kitchen["name"] == "Kitchen"


async def test_get_lights_returns_defaults_for_unconfigured(
    hass: HomeAssistant,
    init_integration,
    mock_registries,
) -> None:
    """Test get_lights returns default values for unconfigured lights."""
    from custom_components.fade_lights.websocket_api import async_get_lights

    result = await async_get_lights(hass)

    # Find kitchen light (unconfigured)
    no_floor = next((f for f in result["floors"] if f["floor_id"] is None), None)
    kitchen = next((a for a in no_floor["areas"] if a["area_id"] == "kitchen"), None)
    light = next(
        (l for l in kitchen["lights"] if l["entity_id"] == "light.kitchen_main"),
        None,
    )

    assert light is not None
    assert light["min_delay_ms"] is None  # No config
    assert light["exclude"] is False  # Default
    assert light["use_native_transition"] is True  # Default


async def test_get_lights_excludes_light_groups(
    hass: HomeAssistant,
    init_integration,
) -> None:
    """Test get_lights excludes light groups (entities with entity_id attribute)."""
    from custom_components.fade_lights.websocket_api import async_get_lights

    # Mock a light group entity
    light_group = MagicMock()
    light_group.entity_id = "light.living_room_group"
    light_group.name = "Living Room Group"
    light_group.area_id = None
    light_group.platform = "group"

    # Set state with entity_id attribute (indicating it's a group)
    hass.states.async_set(
        "light.living_room_group",
        "on",
        {"entity_id": ["light.lamp1", "light.lamp2"]},
    )

    with patch(
        "homeassistant.helpers.floor_registry.async_get",
        return_value=MagicMock(floors={}),
    ), patch(
        "homeassistant.helpers.area_registry.async_get",
        return_value=MagicMock(areas={}, async_get_area=lambda aid: None),
    ), patch(
        "homeassistant.helpers.entity_registry.async_get",
        return_value=MagicMock(
            entities=MagicMock(values=lambda: [light_group])
        ),
    ):
        result = await async_get_lights(hass)

    # Should not include the light group
    all_lights = []
    for floor in result["floors"]:
        for area in floor["areas"]:
            all_lights.extend(area["lights"])

    assert not any(l["entity_id"] == "light.living_room_group" for l in all_lights)


async def test_save_light_config_creates_entry(
    hass: HomeAssistant,
    init_integration,
) -> None:
    """Test save_light_config creates entry for new light."""
    from custom_components.fade_lights.websocket_api import async_save_light_config

    result = await async_save_light_config(
        hass,
        "light.new_light",
        min_delay_ms=200,
        exclude=True,
        use_native_transition=False,
    )

    # Verify data was saved
    assert hass.data[DOMAIN]["data"]["light.new_light"]["min_delay_ms"] == 200
    assert hass.data[DOMAIN]["data"]["light.new_light"]["exclude"] is True
    assert hass.data[DOMAIN]["data"]["light.new_light"]["use_native_transition"] is False

    # Verify result
    assert result == {"success": True}


async def test_save_light_config_updates_existing(
    hass: HomeAssistant,
    init_integration,
) -> None:
    """Test save_light_config updates existing light config."""
    from custom_components.fade_lights.websocket_api import async_save_light_config

    # Set up existing config
    hass.data[DOMAIN]["data"]["light.existing"] = {
        "orig_brightness": 200,
        "min_delay_ms": 100,
        "exclude": False,
    }

    await async_save_light_config(
        hass,
        "light.existing",
        min_delay_ms=150,
    )

    # Verify only min_delay_ms was updated, orig_brightness preserved
    assert hass.data[DOMAIN]["data"]["light.existing"]["min_delay_ms"] == 150
    assert hass.data[DOMAIN]["data"]["light.existing"]["orig_brightness"] == 200
    assert hass.data[DOMAIN]["data"]["light.existing"]["exclude"] is False


async def test_save_light_config_clears_min_delay_with_none(
    hass: HomeAssistant,
    init_integration,
) -> None:
    """Test save_light_config clears min_delay_ms when clear_min_delay is True."""
    from custom_components.fade_lights.websocket_api import async_save_light_config

    # Set up existing config with min_delay_ms
    hass.data[DOMAIN]["data"]["light.test"] = {
        "min_delay_ms": 150,
    }

    await async_save_light_config(
        hass,
        "light.test",
        min_delay_ms=None,
        clear_min_delay=True,
    )

    # Verify min_delay_ms was removed
    assert "min_delay_ms" not in hass.data[DOMAIN]["data"]["light.test"]


async def test_register_websocket_api(
    hass: HomeAssistant,
    init_integration,
) -> None:
    """Test WebSocket commands are registered."""
    from custom_components.fade_lights.websocket_api import async_register_websocket_api

    with patch("homeassistant.components.websocket_api.async_register_command") as mock_register:
        async_register_websocket_api(hass)

        # Verify both commands were registered
        assert mock_register.call_count == 2
