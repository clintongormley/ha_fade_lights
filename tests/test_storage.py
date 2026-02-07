"""Tests for storage helpers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.core import HomeAssistant

from custom_components.fado.const import DOMAIN
from custom_components.fado.coordinator import FadeCoordinator
from custom_components.fado.expected_state import ExpectedState, ExpectedValues


@pytest.fixture
def hass_with_storage(hass: HomeAssistant) -> HomeAssistant:
    """Set up hass with storage data via FadeCoordinator."""
    mock_store = MagicMock()
    mock_store.async_save = AsyncMock()
    storage_data = {
        "light.bedroom": {
            "orig_brightness": 200,
            "min_delay_ms": 150,
            "exclude": True,
        },
        "light.kitchen": {
            "orig_brightness": 255,
        },
    }
    coordinator = FadeCoordinator(
        hass=hass,
        entry=MagicMock(),
        store=mock_store,
        data=storage_data,
        min_step_delay_ms=100,
    )
    hass.data[DOMAIN] = coordinator
    return hass


class TestGetLightConfig:
    """Test get_light_config helper."""

    def test_returns_config_for_configured_light(self, hass_with_storage: HomeAssistant) -> None:
        """Test that config is returned for a configured light."""
        coordinator: FadeCoordinator = hass_with_storage.data[DOMAIN]
        config = coordinator.get_light_config("light.bedroom")

        assert config["min_delay_ms"] == 150
        assert config["exclude"] is True

    def test_returns_empty_dict_for_unconfigured_light(
        self, hass_with_storage: HomeAssistant
    ) -> None:
        """Test that empty dict is returned for unconfigured light."""
        coordinator: FadeCoordinator = hass_with_storage.data[DOMAIN]
        config = coordinator.get_light_config("light.unknown")

        assert config == {}


class TestGetOrigBrightness:
    """Test get_orig_brightness with new storage structure."""

    def test_returns_brightness_from_nested_structure(
        self, hass_with_storage: HomeAssistant
    ) -> None:
        """Test brightness is read from nested dict."""
        coordinator: FadeCoordinator = hass_with_storage.data[DOMAIN]
        brightness = coordinator.get_orig_brightness("light.bedroom")

        assert brightness == 200

    def test_returns_zero_for_unconfigured_light(self, hass_with_storage: HomeAssistant) -> None:
        """Test zero is returned for unconfigured light."""
        coordinator: FadeCoordinator = hass_with_storage.data[DOMAIN]
        brightness = coordinator.get_orig_brightness("light.unknown")

        assert brightness == 0


class TestStoreOrigBrightness:
    """Test store_orig_brightness with new storage structure."""

    def test_stores_brightness_in_nested_structure(self, hass_with_storage: HomeAssistant) -> None:
        """Test brightness is stored in nested dict."""
        coordinator: FadeCoordinator = hass_with_storage.data[DOMAIN]
        coordinator.store_orig_brightness("light.bedroom", 180)

        assert coordinator.data["light.bedroom"]["orig_brightness"] == 180

    def test_creates_entry_for_new_light(self, hass_with_storage: HomeAssistant) -> None:
        """Test new entry is created for unconfigured light."""
        coordinator: FadeCoordinator = hass_with_storage.data[DOMAIN]
        coordinator.store_orig_brightness("light.new", 100)

        assert coordinator.data["light.new"]["orig_brightness"] == 100

    def test_preserves_other_config_when_updating(self, hass_with_storage: HomeAssistant) -> None:
        """Test other config fields are preserved when updating brightness."""
        coordinator: FadeCoordinator = hass_with_storage.data[DOMAIN]
        coordinator.store_orig_brightness("light.bedroom", 180)

        config = coordinator.data["light.bedroom"]
        assert config["orig_brightness"] == 180
        assert config["min_delay_ms"] == 150  # Preserved
        assert config["exclude"] is True  # Preserved


class TestCleanupEntityData:
    """Test cleanup_entity function for entity deletion cleanup."""

    @pytest.fixture
    def hass_with_full_state(self, hass: HomeAssistant) -> HomeAssistant:
        """Set up hass with storage and testing_lights data."""
        mock_store = MagicMock()
        mock_store.async_save = AsyncMock()
        storage_data = {
            "light.test": {
                "orig_brightness": 200,
                "min_delay_ms": 150,
                "exclude": False,
            },
            "light.other": {
                "min_delay_ms": 100,
            },
        }
        coordinator = FadeCoordinator(
            hass=hass,
            entry=MagicMock(),
            store=mock_store,
            data=storage_data,
            min_step_delay_ms=100,
        )
        coordinator.testing_lights = {"light.test", "light.other"}
        hass.data[DOMAIN] = coordinator
        return hass

    async def test_cleanup_removes_persistent_storage(
        self, hass_with_full_state: HomeAssistant
    ) -> None:
        """Test that cleanup removes entity from persistent storage."""
        coordinator: FadeCoordinator = hass_with_full_state.data[DOMAIN]
        await coordinator.cleanup_entity("light.test")

        assert "light.test" not in coordinator.data
        assert "light.other" in coordinator.data
        coordinator.store.async_save.assert_called_once()

    async def test_cleanup_removes_from_testing_lights(
        self, hass_with_full_state: HomeAssistant
    ) -> None:
        """Test that cleanup removes entity from testing_lights set."""
        coordinator: FadeCoordinator = hass_with_full_state.data[DOMAIN]
        await coordinator.cleanup_entity("light.test")

        assert "light.test" not in coordinator.testing_lights
        assert "light.other" in coordinator.testing_lights

    async def test_cleanup_cancels_active_fade(self, hass_with_full_state: HomeAssistant) -> None:
        """Test that cleanup cancels active fade task."""
        coordinator: FadeCoordinator = hass_with_full_state.data[DOMAIN]
        # Create a mock task that will be cancelled
        task = asyncio.create_task(asyncio.sleep(100))
        entity = coordinator.get_or_create_entity("light.test")
        entity.active_task = task

        await coordinator.cleanup_entity("light.test")

        assert coordinator.get_entity("light.test") is None
        assert task.cancelled()

    async def test_cleanup_sets_cancel_event(self, hass_with_full_state: HomeAssistant) -> None:
        """Test that cleanup sets and removes cancel event."""
        coordinator: FadeCoordinator = hass_with_full_state.data[DOMAIN]
        event = asyncio.Event()
        entity = coordinator.get_or_create_entity("light.test")
        entity.cancel_event = event

        await coordinator.cleanup_entity("light.test")

        assert event.is_set()
        assert coordinator.get_entity("light.test") is None

    async def test_cleanup_clears_expected_state(self, hass_with_full_state: HomeAssistant) -> None:
        """Test that cleanup clears expected state."""
        coordinator: FadeCoordinator = hass_with_full_state.data[DOMAIN]
        expected = ExpectedState("light.test")
        expected.add(ExpectedValues(brightness=100))
        entity = coordinator.get_or_create_entity("light.test")
        entity.expected_state = expected

        await coordinator.cleanup_entity("light.test")

        assert coordinator.get_entity("light.test") is None

    async def test_cleanup_removes_completion_condition(
        self, hass_with_full_state: HomeAssistant
    ) -> None:
        """Test that cleanup removes completion condition."""
        coordinator: FadeCoordinator = hass_with_full_state.data[DOMAIN]
        entity = coordinator.get_or_create_entity("light.test")
        entity.complete_condition = asyncio.Condition()

        await coordinator.cleanup_entity("light.test")

        assert coordinator.get_entity("light.test") is None

    async def test_cleanup_clears_intended_state_queue(
        self, hass_with_full_state: HomeAssistant
    ) -> None:
        """Test that cleanup clears intended state queue."""
        coordinator: FadeCoordinator = hass_with_full_state.data[DOMAIN]
        entity = coordinator.get_or_create_entity("light.test")
        entity.intended_queue = [MagicMock()]

        await coordinator.cleanup_entity("light.test")

        assert coordinator.get_entity("light.test") is None

    async def test_cleanup_cancels_restore_task(self, hass_with_full_state: HomeAssistant) -> None:
        """Test that cleanup cancels restore task."""
        coordinator: FadeCoordinator = hass_with_full_state.data[DOMAIN]
        task = asyncio.create_task(asyncio.sleep(100))
        entity = coordinator.get_or_create_entity("light.test")
        entity.restore_task = task

        await coordinator.cleanup_entity("light.test")

        assert coordinator.get_entity("light.test") is None
        assert task.cancelled()

    async def test_cleanup_handles_missing_entity(self, hass: HomeAssistant) -> None:
        """Test that cleanup handles entity not in any data structures."""
        mock_store = MagicMock()
        mock_store.async_save = AsyncMock()
        coordinator = FadeCoordinator(
            hass=hass,
            entry=MagicMock(),
            store=mock_store,
            data={},
            min_step_delay_ms=100,
        )
        hass.data[DOMAIN] = coordinator

        # Should not raise
        await coordinator.cleanup_entity("light.nonexistent")
