"""Tests for storage helpers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.core import HomeAssistant

from custom_components.fado import (
    ACTIVE_FADES,
    FADE_CANCEL_EVENTS,
    FADE_COMPLETE_CONDITIONS,
    FADE_EXPECTED_STATE,
    INTENDED_STATE_QUEUE,
    RESTORE_TASKS,
    _cleanup_entity_data,
    _get_light_config,
    _get_orig_brightness,
    _store_orig_brightness,
)
from custom_components.fado.const import DOMAIN
from custom_components.fado.expected_state import ExpectedState, ExpectedValues


@pytest.fixture
def hass_with_storage(hass: HomeAssistant) -> HomeAssistant:
    """Set up hass with storage data."""
    hass.data[DOMAIN] = {
        "data": {
            "light.bedroom": {
                "orig_brightness": 200,
                "min_delay_ms": 150,
                "exclude": True,
            },
            "light.kitchen": {
                "orig_brightness": 255,
            },
        },
        "store": MagicMock(),
    }
    return hass


class TestGetLightConfig:
    """Test _get_light_config helper."""

    def test_returns_config_for_configured_light(self, hass_with_storage: HomeAssistant) -> None:
        """Test that config is returned for a configured light."""
        config = _get_light_config(hass_with_storage, "light.bedroom")

        assert config["min_delay_ms"] == 150
        assert config["exclude"] is True

    def test_returns_empty_dict_for_unconfigured_light(
        self, hass_with_storage: HomeAssistant
    ) -> None:
        """Test that empty dict is returned for unconfigured light."""
        config = _get_light_config(hass_with_storage, "light.unknown")

        assert config == {}

    def test_returns_empty_dict_when_domain_not_loaded(self, hass: HomeAssistant) -> None:
        """Test that empty dict is returned when domain not loaded."""
        config = _get_light_config(hass, "light.bedroom")

        assert config == {}


class TestGetOrigBrightness:
    """Test _get_orig_brightness with new storage structure."""

    def test_returns_brightness_from_nested_structure(
        self, hass_with_storage: HomeAssistant
    ) -> None:
        """Test brightness is read from nested dict."""
        brightness = _get_orig_brightness(hass_with_storage, "light.bedroom")

        assert brightness == 200

    def test_returns_zero_for_unconfigured_light(self, hass_with_storage: HomeAssistant) -> None:
        """Test zero is returned for unconfigured light."""
        brightness = _get_orig_brightness(hass_with_storage, "light.unknown")

        assert brightness == 0


class TestStoreOrigBrightness:
    """Test _store_orig_brightness with new storage structure."""

    def test_stores_brightness_in_nested_structure(self, hass_with_storage: HomeAssistant) -> None:
        """Test brightness is stored in nested dict."""
        _store_orig_brightness(hass_with_storage, "light.bedroom", 180)

        assert hass_with_storage.data[DOMAIN]["data"]["light.bedroom"]["orig_brightness"] == 180

    def test_creates_entry_for_new_light(self, hass_with_storage: HomeAssistant) -> None:
        """Test new entry is created for unconfigured light."""
        _store_orig_brightness(hass_with_storage, "light.new", 100)

        assert hass_with_storage.data[DOMAIN]["data"]["light.new"]["orig_brightness"] == 100

    def test_preserves_other_config_when_updating(self, hass_with_storage: HomeAssistant) -> None:
        """Test other config fields are preserved when updating brightness."""
        _store_orig_brightness(hass_with_storage, "light.bedroom", 180)

        config = hass_with_storage.data[DOMAIN]["data"]["light.bedroom"]
        assert config["orig_brightness"] == 180
        assert config["min_delay_ms"] == 150  # Preserved
        assert config["exclude"] is True  # Preserved


class TestCleanupEntityData:
    """Test _cleanup_entity_data function for entity deletion cleanup."""

    @pytest.fixture(autouse=True)
    def cleanup_globals(self) -> None:
        """Clear global state before and after each test."""
        ACTIVE_FADES.clear()
        FADE_CANCEL_EVENTS.clear()
        FADE_EXPECTED_STATE.clear()
        FADE_COMPLETE_CONDITIONS.clear()
        INTENDED_STATE_QUEUE.clear()
        RESTORE_TASKS.clear()
        yield
        ACTIVE_FADES.clear()
        FADE_CANCEL_EVENTS.clear()
        FADE_EXPECTED_STATE.clear()
        FADE_COMPLETE_CONDITIONS.clear()
        INTENDED_STATE_QUEUE.clear()
        RESTORE_TASKS.clear()

    @pytest.fixture
    def hass_with_full_state(self, hass: HomeAssistant) -> HomeAssistant:
        """Set up hass with storage and testing_lights data."""
        mock_store = MagicMock()
        mock_store.async_save = AsyncMock()
        hass.data[DOMAIN] = {
            "data": {
                "light.test": {
                    "orig_brightness": 200,
                    "min_delay_ms": 150,
                    "exclude": False,
                },
                "light.other": {
                    "min_delay_ms": 100,
                },
            },
            "store": mock_store,
            "testing_lights": {"light.test", "light.other"},
        }
        return hass

    async def test_cleanup_removes_persistent_storage(
        self, hass_with_full_state: HomeAssistant
    ) -> None:
        """Test that cleanup removes entity from persistent storage."""
        await _cleanup_entity_data(hass_with_full_state, "light.test")

        assert "light.test" not in hass_with_full_state.data[DOMAIN]["data"]
        assert "light.other" in hass_with_full_state.data[DOMAIN]["data"]
        hass_with_full_state.data[DOMAIN]["store"].async_save.assert_called_once()

    async def test_cleanup_removes_from_testing_lights(
        self, hass_with_full_state: HomeAssistant
    ) -> None:
        """Test that cleanup removes entity from testing_lights set."""
        await _cleanup_entity_data(hass_with_full_state, "light.test")

        assert "light.test" not in hass_with_full_state.data[DOMAIN]["testing_lights"]
        assert "light.other" in hass_with_full_state.data[DOMAIN]["testing_lights"]

    async def test_cleanup_cancels_active_fade(self, hass_with_full_state: HomeAssistant) -> None:
        """Test that cleanup cancels active fade task."""
        # Create a mock task that will be cancelled
        task = asyncio.create_task(asyncio.sleep(100))
        ACTIVE_FADES["light.test"] = task

        await _cleanup_entity_data(hass_with_full_state, "light.test")

        assert "light.test" not in ACTIVE_FADES
        assert task.cancelled()

    async def test_cleanup_sets_cancel_event(self, hass_with_full_state: HomeAssistant) -> None:
        """Test that cleanup sets and removes cancel event."""
        event = asyncio.Event()
        FADE_CANCEL_EVENTS["light.test"] = event

        await _cleanup_entity_data(hass_with_full_state, "light.test")

        assert event.is_set()
        assert "light.test" not in FADE_CANCEL_EVENTS

    async def test_cleanup_clears_expected_state(self, hass_with_full_state: HomeAssistant) -> None:
        """Test that cleanup clears expected state."""
        expected = ExpectedState("light.test")
        expected.add(ExpectedValues(brightness=100))
        FADE_EXPECTED_STATE["light.test"] = expected

        await _cleanup_entity_data(hass_with_full_state, "light.test")

        assert "light.test" not in FADE_EXPECTED_STATE

    async def test_cleanup_removes_completion_condition(
        self, hass_with_full_state: HomeAssistant
    ) -> None:
        """Test that cleanup removes completion condition."""
        FADE_COMPLETE_CONDITIONS["light.test"] = asyncio.Condition()

        await _cleanup_entity_data(hass_with_full_state, "light.test")

        assert "light.test" not in FADE_COMPLETE_CONDITIONS

    async def test_cleanup_clears_intended_state_queue(
        self, hass_with_full_state: HomeAssistant
    ) -> None:
        """Test that cleanup clears intended state queue."""
        INTENDED_STATE_QUEUE["light.test"] = [MagicMock()]

        await _cleanup_entity_data(hass_with_full_state, "light.test")

        assert "light.test" not in INTENDED_STATE_QUEUE

    async def test_cleanup_cancels_restore_task(self, hass_with_full_state: HomeAssistant) -> None:
        """Test that cleanup cancels restore task."""
        task = asyncio.create_task(asyncio.sleep(100))
        RESTORE_TASKS["light.test"] = task

        await _cleanup_entity_data(hass_with_full_state, "light.test")

        assert "light.test" not in RESTORE_TASKS
        assert task.cancelled()

    async def test_cleanup_handles_missing_entity(self, hass: HomeAssistant) -> None:
        """Test that cleanup handles entity not in any data structures."""
        hass.data[DOMAIN] = {
            "data": {},
            "store": MagicMock(),
            "testing_lights": set(),
        }

        # Should not raise
        await _cleanup_entity_data(hass, "light.nonexistent")

    async def test_cleanup_handles_domain_not_loaded(self, hass: HomeAssistant) -> None:
        """Test that cleanup handles domain not being loaded."""
        # Should not raise when DOMAIN not in hass.data
        await _cleanup_entity_data(hass, "light.test")
