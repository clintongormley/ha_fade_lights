"""Config flow for Fade Lights integration."""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigFlow, ConfigFlowResult

from .const import DOMAIN


class FadeLightsConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Fade Lights."""

    VERSION = 1
    MINOR_VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Handle the initial step."""
        # Only allow a single instance
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")

        # Create entry immediately without showing a form
        return self.async_create_entry(
            title="Fade Lights",
            data={},
        )

    async def async_step_import(
        self, _import_config: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle import from configuration.yaml or auto-setup."""
        # Only allow a single instance
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")

        return self.async_create_entry(
            title="Fade Lights",
            data={},
        )
