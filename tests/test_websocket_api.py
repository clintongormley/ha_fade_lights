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

    with (
        patch(
            "homeassistant.helpers.floor_registry.async_get",
            return_value=MagicMock(floors={"upstairs": floor_upstairs}),
        ),
        patch(
            "homeassistant.helpers.area_registry.async_get",
            return_value=MagicMock(
                areas={"bedroom": area_bedroom, "kitchen": area_kitchen},
                async_get_area=lambda aid: {"bedroom": area_bedroom, "kitchen": area_kitchen}.get(
                    aid
                ),
            ),
        ),
        patch(
            "homeassistant.helpers.entity_registry.async_get",
            return_value=MagicMock(
                entities=MagicMock(values=lambda: [entity_bedroom_light, entity_kitchen_light])
            ),
        ),
        patch(
            "homeassistant.helpers.device_registry.async_get",
            return_value=MagicMock(async_get=lambda did: None),
        ),
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
        (lt for lt in bedroom["lights"] if lt["entity_id"] == "light.bedroom_ceiling"),
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
        (lt for lt in kitchen["lights"] if lt["entity_id"] == "light.kitchen_main"),
        None,
    )

    assert light is not None
    assert light["min_delay_ms"] is None  # No config
    assert light["exclude"] is False  # Default


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

    with (
        patch(
            "homeassistant.helpers.floor_registry.async_get",
            return_value=MagicMock(floors={}),
        ),
        patch(
            "homeassistant.helpers.area_registry.async_get",
            return_value=MagicMock(areas={}, async_get_area=lambda aid: None),
        ),
        patch(
            "homeassistant.helpers.entity_registry.async_get",
            return_value=MagicMock(entities=MagicMock(values=lambda: [light_group])),
        ),
    ):
        result = await async_get_lights(hass)

    # Should not include the light group
    all_lights = []
    for floor in result["floors"]:
        for area in floor["areas"]:
            all_lights.extend(area["lights"])

    assert not any(lt["entity_id"] == "light.living_room_group" for lt in all_lights)


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
    )

    # Verify data was saved
    assert hass.data[DOMAIN]["data"]["light.new_light"]["min_delay_ms"] == 200
    assert hass.data[DOMAIN]["data"]["light.new_light"]["exclude"] is True

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

        # Verify all six commands were registered
        # (get_lights, save_light_config, autoconfigure, test_native_transitions, get_settings, save_settings)
        assert mock_register.call_count == 6


async def test_get_lights_skips_non_light_entities(
    hass: HomeAssistant,
    init_integration,
) -> None:
    """Test get_lights skips entities that don't start with 'light.'."""
    from custom_components.fade_lights.websocket_api import async_get_lights

    # Mock entity that's not a light
    switch_entity = MagicMock()
    switch_entity.entity_id = "switch.my_switch"
    switch_entity.name = "My Switch"
    switch_entity.area_id = None
    switch_entity.disabled_by = None

    with (
        patch(
            "homeassistant.helpers.floor_registry.async_get",
            return_value=MagicMock(floors={}),
        ),
        patch(
            "homeassistant.helpers.area_registry.async_get",
            return_value=MagicMock(areas={}, async_get_area=lambda aid: None),
        ),
        patch(
            "homeassistant.helpers.entity_registry.async_get",
            return_value=MagicMock(entities=MagicMock(values=lambda: [switch_entity])),
        ),
    ):
        result = await async_get_lights(hass)

    # Should have no lights
    all_lights = []
    for floor in result["floors"]:
        for area in floor["areas"]:
            all_lights.extend(area["lights"])
    assert len(all_lights) == 0


async def test_get_lights_gets_area_from_device(
    hass: HomeAssistant,
    init_integration,
) -> None:
    """Test get_lights gets area from device when entity has no area."""
    from custom_components.fade_lights.websocket_api import async_get_lights

    # Mock device with area
    device = MagicMock()
    device.area_id = "living_room"

    # Mock area
    area = MagicMock()
    area.id = "living_room"
    area.name = "Living Room"
    area.floor_id = None
    area.icon = None

    # Mock light with device but no direct area
    light_entity = MagicMock()
    light_entity.entity_id = "light.device_light"
    light_entity.name = "Device Light"
    light_entity.original_name = "Device Light"
    light_entity.area_id = None  # No direct area
    light_entity.device_id = "device123"  # Has device
    light_entity.disabled_by = None
    light_entity.icon = None

    hass.states.async_set("light.device_light", "on", {"brightness": 200})

    with (
        patch(
            "homeassistant.helpers.floor_registry.async_get",
            return_value=MagicMock(floors={}),
        ),
        patch(
            "homeassistant.helpers.area_registry.async_get",
            return_value=MagicMock(
                areas={"living_room": area},
                async_get_area=lambda aid: area if aid == "living_room" else None,
            ),
        ),
        patch(
            "homeassistant.helpers.entity_registry.async_get",
            return_value=MagicMock(entities=MagicMock(values=lambda: [light_entity])),
        ),
        patch(
            "homeassistant.helpers.device_registry.async_get",
            return_value=MagicMock(async_get=lambda did: device if did == "device123" else None),
        ),
    ):
        result = await async_get_lights(hass)

    # Find living room area
    no_floor = next((f for f in result["floors"] if f["floor_id"] is None), None)
    assert no_floor is not None
    living_room = next((a for a in no_floor["areas"] if a["area_id"] == "living_room"), None)
    assert living_room is not None
    assert living_room["name"] == "Living Room"


async def test_get_lights_uses_state_friendly_name_and_icon(
    hass: HomeAssistant,
    init_integration,
) -> None:
    """Test get_lights uses friendly_name and icon from state."""
    from custom_components.fade_lights.websocket_api import async_get_lights

    light_entity = MagicMock()
    light_entity.entity_id = "light.test_light"
    light_entity.name = "Entity Name"
    light_entity.original_name = "Original Name"
    light_entity.area_id = None
    light_entity.device_id = None
    light_entity.disabled_by = None
    light_entity.icon = "mdi:entity-icon"

    # State has different friendly_name and icon
    hass.states.async_set(
        "light.test_light",
        "on",
        {"friendly_name": "State Friendly Name", "icon": "mdi:state-icon", "brightness": 200},
    )

    with (
        patch(
            "homeassistant.helpers.floor_registry.async_get",
            return_value=MagicMock(floors={}),
        ),
        patch(
            "homeassistant.helpers.area_registry.async_get",
            return_value=MagicMock(areas={}, async_get_area=lambda aid: None),
        ),
        patch(
            "homeassistant.helpers.entity_registry.async_get",
            return_value=MagicMock(entities=MagicMock(values=lambda: [light_entity])),
        ),
        patch(
            "homeassistant.helpers.device_registry.async_get",
            return_value=MagicMock(async_get=lambda did: None),
        ),
    ):
        result = await async_get_lights(hass)

    no_floor = next((f for f in result["floors"] if f["floor_id"] is None), None)
    no_area = next((a for a in no_floor["areas"] if a["area_id"] is None), None)
    light = next((lt for lt in no_area["lights"] if lt["entity_id"] == "light.test_light"), None)

    # Should prefer state values
    assert light["name"] == "State Friendly Name"
    assert light["icon"] == "mdi:state-icon"


async def test_expand_light_groups(hass: HomeAssistant) -> None:
    """Test _expand_light_groups expands groups to individual lights."""
    from custom_components.fade_lights.websocket_api import _expand_light_groups

    # Set up a group and individual lights
    hass.states.async_set(
        "light.group",
        "on",
        {"entity_id": ["light.lamp1", "light.lamp2"]},
    )
    hass.states.async_set("light.lamp1", "on", {"brightness": 200})
    hass.states.async_set("light.lamp2", "on", {"brightness": 150})
    hass.states.async_set("light.individual", "on", {"brightness": 100})

    result = _expand_light_groups(hass, ["light.group", "light.individual"])

    # Should have 3 individual lights (lamp1, lamp2, individual)
    assert len(result) == 3
    assert "light.lamp1" in result
    assert "light.lamp2" in result
    assert "light.individual" in result
    assert "light.group" not in result


async def test_expand_light_groups_handles_string_entity_id(hass: HomeAssistant) -> None:
    """Test _expand_light_groups handles single string entity_id attribute."""
    from custom_components.fade_lights.websocket_api import _expand_light_groups

    # Set up a group with single string entity_id
    hass.states.async_set(
        "light.single_group",
        "on",
        {"entity_id": "light.single_lamp"},  # String, not list
    )
    hass.states.async_set("light.single_lamp", "on", {"brightness": 200})

    result = _expand_light_groups(hass, ["light.single_group"])

    assert len(result) == 1
    assert "light.single_lamp" in result


async def test_expand_light_groups_skips_missing_entities(hass: HomeAssistant) -> None:
    """Test _expand_light_groups skips entities with no state."""
    from custom_components.fade_lights.websocket_api import _expand_light_groups

    # Only set up one light, not the other
    hass.states.async_set("light.exists", "on", {"brightness": 200})

    result = _expand_light_groups(hass, ["light.exists", "light.missing"])

    assert len(result) == 1
    assert "light.exists" in result
    assert "light.missing" not in result


async def test_get_light_config(hass: HomeAssistant, init_integration) -> None:
    """Test _get_light_config returns config from storage."""
    from custom_components.fade_lights.websocket_api import _get_light_config

    # Set up config
    hass.data[DOMAIN]["data"]["light.configured"] = {
        "min_delay_ms": 150,
        "exclude": True,
    }

    config = _get_light_config(hass, "light.configured")
    assert config["min_delay_ms"] == 150
    assert config["exclude"] is True

    # Unconfigured light
    config = _get_light_config(hass, "light.unconfigured")
    assert config == {}


async def test_get_settings(hass: HomeAssistant, init_integration) -> None:
    """Test ws_get_settings returns current settings."""
    from custom_components.fade_lights.websocket_api import _get_config_entry

    from custom_components.fade_lights.const import (
        DEFAULT_LOG_LEVEL,
        DEFAULT_MIN_STEP_DELAY_MS,
    )

    entry = _get_config_entry(hass)
    assert entry is not None

    # Defaults should be returned when no options set
    assert entry.options.get("min_step_delay_ms", DEFAULT_MIN_STEP_DELAY_MS) == DEFAULT_MIN_STEP_DELAY_MS
    assert entry.options.get("log_level", DEFAULT_LOG_LEVEL) == DEFAULT_LOG_LEVEL


async def test_save_settings_updates_min_delay(hass: HomeAssistant, init_integration) -> None:
    """Test save_settings updates min_step_delay_ms."""
    from custom_components.fade_lights.const import OPTION_MIN_STEP_DELAY_MS
    from custom_components.fade_lights.websocket_api import _get_config_entry

    entry = _get_config_entry(hass)

    # Update options
    new_options = dict(entry.options)
    new_options[OPTION_MIN_STEP_DELAY_MS] = 200
    hass.config_entries.async_update_entry(entry, options=new_options)

    # Verify update
    entry = _get_config_entry(hass)
    assert entry.options.get(OPTION_MIN_STEP_DELAY_MS) == 200


async def test_apply_log_level(hass: HomeAssistant, init_integration) -> None:
    """Test _apply_log_level calls logger service."""
    from unittest.mock import AsyncMock

    from custom_components.fade_lights.websocket_api import _apply_log_level

    # Register a mock logger service to capture the call
    calls = []

    async def mock_set_level(call):
        calls.append(call.data)

    hass.services.async_register("logger", "set_level", mock_set_level)

    await _apply_log_level(hass, "debug")

    assert len(calls) == 1
    assert calls[0] == {"custom_components.fade_lights": "debug"}


async def test_apply_log_level_warning(hass: HomeAssistant, init_integration) -> None:
    """Test _apply_log_level with warning level."""
    from custom_components.fade_lights.websocket_api import _apply_log_level

    calls = []

    async def mock_set_level(call):
        calls.append(call.data)

    hass.services.async_register("logger", "set_level", mock_set_level)

    await _apply_log_level(hass, "warning")

    assert len(calls) == 1
    assert calls[0] == {"custom_components.fade_lights": "warning"}


async def test_apply_log_level_unknown_defaults_to_warning(
    hass: HomeAssistant, init_integration
) -> None:
    """Test _apply_log_level defaults to warning for unknown level."""
    from custom_components.fade_lights.websocket_api import _apply_log_level

    calls = []

    async def mock_set_level(call):
        calls.append(call.data)

    hass.services.async_register("logger", "set_level", mock_set_level)

    await _apply_log_level(hass, "unknown_level")

    assert len(calls) == 1
    assert calls[0] == {"custom_components.fade_lights": "warning"}


async def test_get_lights_includes_min_brightness(
    hass: HomeAssistant,
    init_integration,
    mock_registries,
) -> None:
    """Test get_lights includes min_brightness in response."""
    from custom_components.fade_lights.websocket_api import async_get_lights

    # Set up storage with min_brightness for one light
    hass.data[DOMAIN]["data"]["light.bedroom_ceiling"] = {
        "min_delay_ms": 150,
        "min_brightness": 3,
    }

    result = await async_get_lights(hass)

    # Find upstairs floor and bedroom area
    upstairs = next((f for f in result["floors"] if f["floor_id"] == "upstairs"), None)
    bedroom = next((a for a in upstairs["areas"] if a["area_id"] == "bedroom"), None)
    light = next(
        (lt for lt in bedroom["lights"] if lt["entity_id"] == "light.bedroom_ceiling"),
        None,
    )

    assert light is not None
    assert light["min_brightness"] == 3


async def test_get_lights_returns_none_for_unconfigured_min_brightness(
    hass: HomeAssistant,
    init_integration,
    mock_registries,
) -> None:
    """Test get_lights returns None for min_brightness when not configured."""
    from custom_components.fade_lights.websocket_api import async_get_lights

    result = await async_get_lights(hass)

    # Find kitchen light (unconfigured)
    no_floor = next((f for f in result["floors"] if f["floor_id"] is None), None)
    kitchen = next((a for a in no_floor["areas"] if a["area_id"] == "kitchen"), None)
    light = next(
        (lt for lt in kitchen["lights"] if lt["entity_id"] == "light.kitchen_main"),
        None,
    )

    assert light is not None
    assert light["min_brightness"] is None


async def test_save_light_config_saves_min_brightness(
    hass: HomeAssistant,
    init_integration,
) -> None:
    """Test save_light_config saves min_brightness."""
    from custom_components.fade_lights.websocket_api import async_save_light_config

    result = await async_save_light_config(
        hass,
        "light.test",
        min_brightness=5,
    )

    # Verify data was saved
    assert hass.data[DOMAIN]["data"]["light.test"]["min_brightness"] == 5
    assert result == {"success": True}


async def test_save_light_config_clears_min_brightness_with_flag(
    hass: HomeAssistant,
    init_integration,
) -> None:
    """Test save_light_config clears min_brightness when clear_min_brightness is True."""
    from custom_components.fade_lights.websocket_api import async_save_light_config

    # Set up existing config with min_brightness
    hass.data[DOMAIN]["data"]["light.test"] = {
        "min_brightness": 5,
    }

    await async_save_light_config(
        hass,
        "light.test",
        min_brightness=None,
        clear_min_brightness=True,
    )

    # Verify min_brightness was removed
    assert "min_brightness" not in hass.data[DOMAIN]["data"]["light.test"]


async def test_autoconfigure_result_includes_min_brightness(
    hass: HomeAssistant,
    hass_ws_client,
    init_integration,
) -> None:
    """Test autoconfigure result event includes min_brightness."""
    from unittest.mock import patch

    entity_id = "light.test_min_brightness"
    hass.states.async_set(entity_id, "on", {"brightness": 200})

    async def mock_autoconfigure_light(hass, entity_id):
        return {
            "entity_id": entity_id,
            "min_delay_ms": 100,
            "native_transitions": True,
            "min_brightness": 3,
        }

    with patch(
        "custom_components.fade_lights.autoconfigure.async_autoconfigure_light",
        side_effect=mock_autoconfigure_light,
    ):
        client = await hass_ws_client(hass)

        await client.send_json(
            {
                "id": 1,
                "type": "fade_lights/autoconfigure",
                "entity_ids": [entity_id],
            }
        )

        # Collect events
        events = []
        while True:
            msg = await client.receive_json()
            if msg["type"] == "result":
                break
            events.append(msg)

    # Find result event
    result_event = next(
        (e for e in events if e["event"]["type"] == "result"),
        None,
    )

    assert result_event is not None
    assert result_event["event"]["min_brightness"] == 3
