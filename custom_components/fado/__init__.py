"""The Fado integration."""

from __future__ import annotations

import contextlib
import logging
from datetime import datetime, timedelta
from pathlib import Path

import voluptuous as vol
from homeassistant.components import frontend, panel_custom
from homeassistant.components.http import StaticPathConfig
from homeassistant.components.light.const import DOMAIN as LIGHT_DOMAIN
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import (
    Event,
    EventStateChangedData,
    HomeAssistant,
    ServiceCall,
    callback,
)
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.event import (
    TrackStates,
    async_track_state_change_filtered,
    async_track_time_interval,
)
from homeassistant.helpers.storage import Store
from homeassistant.helpers.typing import ConfigType

from .const import (
    DEFAULT_LOG_LEVEL,
    DEFAULT_MIN_STEP_DELAY_MS,
    DOMAIN,
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARNING,
    OPTION_LOG_LEVEL,
    OPTION_MIN_STEP_DELAY_MS,
    SERVICE_FADE_LIGHTS,
    STORAGE_KEY,
    UNCONFIGURED_CHECK_INTERVAL_HOURS,
)
from .coordinator import FadeCoordinator
from .notifications import _notify_unconfigured_lights
from .websocket_api import async_register_websocket_api

_LOGGER = logging.getLogger(__name__)


# =============================================================================
# Integration Setup
# =============================================================================


async def async_setup(hass: HomeAssistant, _config: ConfigType) -> bool:
    """Set up the Fado component."""
    if not hass.config_entries.async_entries(DOMAIN):
        hass.async_create_task(
            hass.config_entries.flow.async_init(DOMAIN, context={"source": "import"})
        )
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Fado from a config entry."""
    store: Store[dict[str, int]] = Store(hass, 1, STORAGE_KEY)
    min_step_delay_ms = entry.options.get(OPTION_MIN_STEP_DELAY_MS, DEFAULT_MIN_STEP_DELAY_MS)

    coordinator = FadeCoordinator(
        hass=hass,
        store=store,
        min_step_delay_ms=min_step_delay_ms,
    )
    await coordinator.async_load()

    hass.data[DOMAIN] = coordinator

    async def handle_fade_lights(call: ServiceCall) -> None:
        """Service handler wrapper."""
        await coordinator.handle_fade_lights(call)

    @callback
    def handle_light_state_change(event: Event[EventStateChangedData]) -> None:
        """Event handler wrapper."""
        coordinator.handle_state_change(event)

    # Valid easing curve names
    valid_easing = [
        "auto",
        "linear",
        "ease_in_quad",
        "ease_in_cubic",
        "ease_out_quad",
        "ease_out_cubic",
        "ease_in_out_sine",
    ]

    hass.services.async_register(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        handle_fade_lights,
        schema=cv.make_entity_service_schema(
            {vol.Optional("easing", default="auto"): vol.In(valid_easing)},
            extra=vol.ALLOW_EXTRA,
        ),
    )

    # Track only light domain state changes (more efficient than listening to all events)
    tracker = async_track_state_change_filtered(
        hass,
        TrackStates(False, set(), {LIGHT_DOMAIN}),
        handle_light_state_change,
    )
    entry.async_on_unload(tracker.async_remove)

    # Listen for entity registry changes to clean up deleted entities and check for new ones
    async def handle_entity_registry_updated(
        event: Event[er.EventEntityRegistryUpdatedData],
    ) -> None:
        """Handle entity registry updates."""
        action = event.data["action"]
        entity_id = event.data["entity_id"]

        # Only handle light entities
        if not entity_id.startswith(f"{LIGHT_DOMAIN}."):
            return

        if action == "remove":
            await coordinator.cleanup_entity(entity_id)
            await _notify_unconfigured_lights(hass)
        elif action == "create":
            await _notify_unconfigured_lights(hass)
        elif action == "update":
            # Check if light was re-enabled (disabled_by changed)
            changes = event.data.get("changes", {})
            if "disabled_by" in changes:
                await _notify_unconfigured_lights(hass)

    entry.async_on_unload(
        hass.bus.async_listen(
            er.EVENT_ENTITY_REGISTRY_UPDATED,
            handle_entity_registry_updated,
        )
    )

    # Register daily timer to check for unconfigured lights
    async def _daily_unconfigured_check(_now: datetime) -> None:
        """Daily check for unconfigured lights."""
        await _notify_unconfigured_lights(hass)

    entry.async_on_unload(
        async_track_time_interval(
            hass,
            _daily_unconfigured_check,
            timedelta(hours=UNCONFIGURED_CHECK_INTERVAL_HOURS),
        )
    )

    # Register WebSocket API
    async_register_websocket_api(hass)

    # Register panel (only if HTTP component is available - won't be in tests)
    if hass.http is not None:
        # Register static path for frontend files
        await hass.http.async_register_static_paths(
            [
                StaticPathConfig(
                    "/fado_panel",
                    str(Path(__file__).parent / "frontend"),
                    cache_headers=False,  # Disable caching during development
                )
            ]
        )

        # Register the panel
        await panel_custom.async_register_panel(
            hass,
            frontend_url_path="fado",
            webcomponent_name="fado-panel",
            sidebar_title="Fado",
            sidebar_icon="mdi:lightbulb-variant",
            module_url="/fado_panel/panel.js",
            require_admin=False,
        )

    # Apply stored log level on startup
    await _apply_stored_log_level(hass, entry)

    # Check for unconfigured lights and notify
    await _notify_unconfigured_lights(hass)

    return True


async def _apply_stored_log_level(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Apply the stored log level setting."""
    log_level = entry.options.get(OPTION_LOG_LEVEL, DEFAULT_LOG_LEVEL)

    # Map our level names to Python logging level names
    level_map = {
        LOG_LEVEL_WARNING: "warning",
        LOG_LEVEL_INFO: "info",
        LOG_LEVEL_DEBUG: "debug",
    }
    python_level = level_map.get(log_level, "warning")

    # Use HA's logger service to set the level
    # Logger service may not be available in tests
    with contextlib.suppress(Exception):
        await hass.services.async_call(
            "logger",
            "set_level",
            {f"custom_components.{DOMAIN}": python_level},
        )


async def async_unload_entry(hass: HomeAssistant, _entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    coordinator: FadeCoordinator = hass.data[DOMAIN]
    await coordinator.shutdown()

    hass.services.async_remove(DOMAIN, SERVICE_FADE_LIGHTS)
    hass.data.pop(DOMAIN, None)

    # Remove the panel
    frontend.async_remove_panel(hass, "fado")

    return True
