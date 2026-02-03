"""Tests for unconfigured lights notification."""

from unittest.mock import MagicMock, patch

import pytest
from homeassistant.components.light import DOMAIN as LIGHT_DOMAIN
from homeassistant.core import HomeAssistant

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
        """Test excludes light with min_delay_ms configured."""
        hass.data[DOMAIN] = {"data": {"light.bedroom": {"min_delay_ms": 100}}}

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
        hass.data[DOMAIN] = {"data": {"light.bedroom": {"min_delay_ms": 100}}}

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
