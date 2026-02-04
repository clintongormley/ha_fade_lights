"""Tests for unconfigured lights notification."""

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components.light import DOMAIN as LIGHT_DOMAIN
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from custom_components.fade_lights import async_setup_entry
from custom_components.fade_lights.const import DOMAIN, NOTIFICATION_ID
from custom_components.fade_lights.notifications import (
    _get_unconfigured_lights,
    _notify_unconfigured_lights,
)


@pytest.fixture
def mock_entity_registry():
    """Create a mock entity registry."""
    registry = MagicMock()
    return registry


class TestGetUnconfiguredLights:
    """Test _get_unconfigured_lights function."""

    def test_returns_empty_when_domain_not_loaded(self, hass: HomeAssistant) -> None:
        """Test returns empty set when domain not in hass.data."""
        with patch(
            "custom_components.fade_lights.notifications.er.async_get"
        ) as mock_er:
            mock_er.return_value.entities = {}
            result = _get_unconfigured_lights(hass)
        assert result == set()

    def test_returns_unconfigured_light(self, hass: HomeAssistant) -> None:
        """Test returns light missing min_delay_ms."""
        hass.data[DOMAIN] = {"data": {}}

        mock_entry = MagicMock()
        mock_entry.entity_id = "light.bedroom"
        mock_entry.domain = LIGHT_DOMAIN
        mock_entry.disabled = False

        with patch(
            "custom_components.fade_lights.notifications.er.async_get"
        ) as mock_er:
            mock_er.return_value.entities.values.return_value = [mock_entry]
            result = _get_unconfigured_lights(hass)

        assert result == {"light.bedroom"}

    def test_excludes_configured_light(self, hass: HomeAssistant) -> None:
        """Test excludes light with all required fields configured."""
        hass.data[DOMAIN] = {"data": {"light.bedroom": {"min_delay_ms": 100, "native_transitions": True}}}

        mock_entry = MagicMock()
        mock_entry.entity_id = "light.bedroom"
        mock_entry.domain = LIGHT_DOMAIN
        mock_entry.disabled = False

        with patch(
            "custom_components.fade_lights.notifications.er.async_get"
        ) as mock_er:
            mock_er.return_value.entities.values.return_value = [mock_entry]
            result = _get_unconfigured_lights(hass)

        assert result == set()

    def test_excludes_disabled_light(self, hass: HomeAssistant) -> None:
        """Test excludes disabled lights."""
        hass.data[DOMAIN] = {"data": {}}

        mock_entry = MagicMock()
        mock_entry.entity_id = "light.bedroom"
        mock_entry.domain = LIGHT_DOMAIN
        mock_entry.disabled = True

        with patch(
            "custom_components.fade_lights.notifications.er.async_get"
        ) as mock_er:
            mock_er.return_value.entities.values.return_value = [mock_entry]
            result = _get_unconfigured_lights(hass)

        assert result == set()

    def test_excludes_excluded_light(self, hass: HomeAssistant) -> None:
        """Test excludes lights marked as excluded."""
        hass.data[DOMAIN] = {"data": {"light.bedroom": {"exclude": True}}}

        mock_entry = MagicMock()
        mock_entry.entity_id = "light.bedroom"
        mock_entry.domain = LIGHT_DOMAIN
        mock_entry.disabled = False

        with patch(
            "custom_components.fade_lights.notifications.er.async_get"
        ) as mock_er:
            mock_er.return_value.entities.values.return_value = [mock_entry]
            result = _get_unconfigured_lights(hass)

        assert result == set()

    def test_excludes_non_light_entities(self, hass: HomeAssistant) -> None:
        """Test excludes non-light domain entities."""
        hass.data[DOMAIN] = {"data": {}}

        mock_entry = MagicMock()
        mock_entry.entity_id = "switch.bedroom"
        mock_entry.domain = "switch"
        mock_entry.disabled = False

        with patch(
            "custom_components.fade_lights.notifications.er.async_get"
        ) as mock_er:
            mock_er.return_value.entities.values.return_value = [mock_entry]
            result = _get_unconfigured_lights(hass)

        assert result == set()


class TestNotifyUnconfiguredLights:
    """Test _notify_unconfigured_lights function."""

    async def test_creates_notification_when_unconfigured(
        self, hass: HomeAssistant
    ) -> None:
        """Test creates notification when lights are unconfigured."""
        hass.data[DOMAIN] = {"data": {}}

        mock_entry = MagicMock()
        mock_entry.entity_id = "light.bedroom"
        mock_entry.domain = LIGHT_DOMAIN
        mock_entry.disabled = False

        with (
            patch(
                "custom_components.fade_lights.notifications.er.async_get"
            ) as mock_er,
            patch(
                "custom_components.fade_lights.notifications.persistent_notification.async_create"
            ) as mock_create,
        ):
            mock_er.return_value.entities.values.return_value = [mock_entry]
            await _notify_unconfigured_lights(hass)

        mock_create.assert_called_once()
        call_args = mock_create.call_args
        assert "1 light" in call_args[0][1]
        assert "/fade-lights" in call_args[0][1]

    async def test_creates_notification_plural(self, hass: HomeAssistant) -> None:
        """Test notification message is plural for multiple lights."""
        hass.data[DOMAIN] = {"data": {}}

        mock_entries = []
        for name in ["bedroom", "kitchen"]:
            entry = MagicMock()
            entry.entity_id = f"light.{name}"
            entry.domain = LIGHT_DOMAIN
            entry.disabled = False
            mock_entries.append(entry)

        with (
            patch(
                "custom_components.fade_lights.notifications.er.async_get"
            ) as mock_er,
            patch(
                "custom_components.fade_lights.notifications.persistent_notification.async_create"
            ) as mock_create,
        ):
            mock_er.return_value.entities.values.return_value = mock_entries
            await _notify_unconfigured_lights(hass)

        call_args = mock_create.call_args
        assert "2 lights" in call_args[0][1]

    async def test_dismisses_notification_when_all_configured(
        self, hass: HomeAssistant
    ) -> None:
        """Test dismisses notification when no unconfigured lights."""
        hass.data[DOMAIN] = {"data": {"light.bedroom": {"min_delay_ms": 100, "native_transitions": True}}}

        mock_entry = MagicMock()
        mock_entry.entity_id = "light.bedroom"
        mock_entry.domain = LIGHT_DOMAIN
        mock_entry.disabled = False

        with (
            patch(
                "custom_components.fade_lights.notifications.er.async_get"
            ) as mock_er,
            patch(
                "custom_components.fade_lights.notifications.persistent_notification.async_dismiss"
            ) as mock_dismiss,
        ):
            mock_er.return_value.entities.values.return_value = [mock_entry]
            await _notify_unconfigured_lights(hass)

        mock_dismiss.assert_called_once_with(hass, NOTIFICATION_ID)

    async def test_dismisses_notification_when_no_lights(
        self, hass: HomeAssistant
    ) -> None:
        """Test dismisses notification when no lights exist."""
        hass.data[DOMAIN] = {"data": {}}

        with (
            patch(
                "custom_components.fade_lights.notifications.er.async_get"
            ) as mock_er,
            patch(
                "custom_components.fade_lights.notifications.persistent_notification.async_dismiss"
            ) as mock_dismiss,
        ):
            mock_er.return_value.entities.values.return_value = []
            await _notify_unconfigured_lights(hass)

        mock_dismiss.assert_called_once_with(hass, NOTIFICATION_ID)


class TestSetupNotification:
    """Test notification on setup."""

    async def test_checks_unconfigured_on_setup(self, hass: HomeAssistant) -> None:
        """Test that setup checks for unconfigured lights."""
        mock_entry = MagicMock(spec=ConfigEntry)
        mock_entry.entry_id = "test_entry"
        mock_entry.options = {}
        mock_entry.async_on_unload = MagicMock()

        with (
            patch(
                "custom_components.fade_lights.async_register_websocket_api"
            ),
            patch(
                "custom_components.fade_lights._notify_unconfigured_lights"
            ) as mock_notify,
            patch(
                "custom_components.fade_lights._apply_stored_log_level"
            ),
        ):
            hass.http = None  # Skip panel registration
            await async_setup_entry(hass, mock_entry)

        mock_notify.assert_called_once_with(hass)


class TestEntityRegistryNotification:
    """Test notification on entity registry events."""

    async def test_notifies_on_light_create(self, hass: HomeAssistant) -> None:
        """Test notification check on light entity creation."""
        from homeassistant.helpers import entity_registry as er

        mock_entry = MagicMock(spec=ConfigEntry)
        mock_entry.entry_id = "test_entry"
        mock_entry.options = {}
        mock_entry.async_on_unload = MagicMock()

        with (
            patch("custom_components.fade_lights.async_register_websocket_api"),
            patch(
                "custom_components.fade_lights._notify_unconfigured_lights"
            ) as mock_notify,
            patch("custom_components.fade_lights._apply_stored_log_level"),
        ):
            hass.http = None
            await async_setup_entry(hass, mock_entry)

            # Reset mock to clear the call from setup
            mock_notify.reset_mock()

            # Simulate entity registry create event
            hass.bus.async_fire(
                er.EVENT_ENTITY_REGISTRY_UPDATED,
                {"action": "create", "entity_id": "light.new_light"},
            )
            await hass.async_block_till_done()

            mock_notify.assert_called()

    async def test_notifies_on_light_reenable(self, hass: HomeAssistant) -> None:
        """Test notification check when light is re-enabled."""
        from homeassistant.helpers import entity_registry as er

        mock_entry = MagicMock(spec=ConfigEntry)
        mock_entry.entry_id = "test_entry"
        mock_entry.options = {}
        mock_entry.async_on_unload = MagicMock()

        with (
            patch("custom_components.fade_lights.async_register_websocket_api"),
            patch(
                "custom_components.fade_lights._notify_unconfigured_lights"
            ) as mock_notify,
            patch("custom_components.fade_lights._apply_stored_log_level"),
        ):
            hass.http = None
            await async_setup_entry(hass, mock_entry)

            # Reset mock to clear the call from setup
            mock_notify.reset_mock()

            # Simulate entity registry update with disabled_by change
            hass.bus.async_fire(
                er.EVENT_ENTITY_REGISTRY_UPDATED,
                {
                    "action": "update",
                    "entity_id": "light.bedroom",
                    "changes": {"disabled_by": None},
                },
            )
            await hass.async_block_till_done()

            mock_notify.assert_called()

    async def test_notifies_on_light_remove(self, hass: HomeAssistant) -> None:
        """Test notification check on light removal (may dismiss)."""
        from homeassistant.helpers import entity_registry as er

        mock_entry = MagicMock(spec=ConfigEntry)
        mock_entry.entry_id = "test_entry"
        mock_entry.options = {}
        mock_entry.async_on_unload = MagicMock()

        with (
            patch("custom_components.fade_lights.async_register_websocket_api"),
            patch(
                "custom_components.fade_lights._notify_unconfigured_lights"
            ) as mock_notify,
            patch("custom_components.fade_lights._apply_stored_log_level"),
        ):
            hass.http = None
            await async_setup_entry(hass, mock_entry)

            # Reset mock to clear the call from setup
            mock_notify.reset_mock()

            # Simulate entity registry remove event
            hass.bus.async_fire(
                er.EVENT_ENTITY_REGISTRY_UPDATED,
                {"action": "remove", "entity_id": "light.old_light"},
            )
            await hass.async_block_till_done()

            mock_notify.assert_called()

    async def test_ignores_non_light_entities(self, hass: HomeAssistant) -> None:
        """Test ignores non-light entity events."""
        from homeassistant.helpers import entity_registry as er

        mock_entry = MagicMock(spec=ConfigEntry)
        mock_entry.entry_id = "test_entry"
        mock_entry.options = {}
        mock_entry.async_on_unload = MagicMock()

        with (
            patch("custom_components.fade_lights.async_register_websocket_api"),
            patch(
                "custom_components.fade_lights._notify_unconfigured_lights"
            ) as mock_notify,
            patch("custom_components.fade_lights._apply_stored_log_level"),
        ):
            hass.http = None
            await async_setup_entry(hass, mock_entry)

            # Reset mock to clear the call from setup
            mock_notify.reset_mock()

            # Simulate switch entity creation
            hass.bus.async_fire(
                er.EVENT_ENTITY_REGISTRY_UPDATED,
                {"action": "create", "entity_id": "switch.new_switch"},
            )
            await hass.async_block_till_done()

            mock_notify.assert_not_called()

    async def test_ignores_update_without_disabled_change(
        self, hass: HomeAssistant
    ) -> None:
        """Test ignores updates that don't change disabled state."""
        from homeassistant.helpers import entity_registry as er

        mock_entry = MagicMock(spec=ConfigEntry)
        mock_entry.entry_id = "test_entry"
        mock_entry.options = {}
        mock_entry.async_on_unload = MagicMock()

        with (
            patch("custom_components.fade_lights.async_register_websocket_api"),
            patch(
                "custom_components.fade_lights._notify_unconfigured_lights"
            ) as mock_notify,
            patch("custom_components.fade_lights._apply_stored_log_level"),
        ):
            hass.http = None
            await async_setup_entry(hass, mock_entry)

            # Reset mock to clear the call from setup
            mock_notify.reset_mock()

            # Simulate entity registry update without disabled_by change
            hass.bus.async_fire(
                er.EVENT_ENTITY_REGISTRY_UPDATED,
                {
                    "action": "update",
                    "entity_id": "light.bedroom",
                    "changes": {"name": "New Name"},
                },
            )
            await hass.async_block_till_done()

            mock_notify.assert_not_called()


class TestDailyNotificationTimer:
    """Test daily notification timer."""

    async def test_registers_daily_timer(self, hass: HomeAssistant) -> None:
        """Test that setup registers a daily timer."""
        mock_entry = MagicMock(spec=ConfigEntry)
        mock_entry.entry_id = "test_entry"
        mock_entry.options = {}
        unload_callbacks = []
        mock_entry.async_on_unload = lambda cb: unload_callbacks.append(cb)

        with (
            patch("custom_components.fade_lights.async_register_websocket_api"),
            patch("custom_components.fade_lights._notify_unconfigured_lights"),
            patch("custom_components.fade_lights._apply_stored_log_level"),
            patch(
                "custom_components.fade_lights.async_track_time_interval"
            ) as mock_timer,
        ):
            hass.http = None
            await async_setup_entry(hass, mock_entry)

        # Verify timer was registered with 24 hour interval
        mock_timer.assert_called_once()
        call_args = mock_timer.call_args
        assert call_args[0][0] is hass  # First arg is hass
        assert call_args[0][2] == timedelta(hours=24)  # Third arg is interval


class TestSaveConfigNotification:
    """Test notification after saving config."""

    async def test_notifies_after_save(self, hass: HomeAssistant) -> None:
        """Test notification check is called after saving config."""
        from custom_components.fade_lights.websocket_api import async_save_light_config

        hass.data[DOMAIN] = {"data": {}, "store": MagicMock()}
        hass.data[DOMAIN]["store"].async_save = AsyncMock()

        with patch(
            "custom_components.fade_lights.websocket_api._notify_unconfigured_lights"
        ) as mock_notify:
            await async_save_light_config(hass, "light.bedroom", min_delay_ms=100)

        mock_notify.assert_called_once_with(hass)
