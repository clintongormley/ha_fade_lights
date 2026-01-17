"""Tests for Fade Lights integration initialization."""

from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.fade_lights.const import DOMAIN


async def test_setup_entry(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test the integration is set up correctly."""
    assert DOMAIN in hass.data
    assert hass.services.has_service(DOMAIN, "fade_lights")


async def test_unload_entry(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test the integration can be unloaded."""
    await hass.config_entries.async_unload(init_integration.entry_id)
    await hass.async_block_till_done()

    assert DOMAIN not in hass.data
    assert not hass.services.has_service(DOMAIN, "fade_lights")
