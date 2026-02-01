"""Config flow for Fade Lights integration."""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry, ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.core import callback

from .const import (
    DEFAULT_MIN_STEP_DELAY_MS,
    DOMAIN,
    MIN_STEP_DELAY_MS,
    OPTION_MIN_STEP_DELAY_MS,
)


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

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> FadeLightsOptionsFlow:
        """Get the options flow for this handler."""
        return FadeLightsOptionsFlow()


class FadeLightsOptionsFlow(OptionsFlow):
    """Handle options flow for Fade Lights."""

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        options = self.config_entry.options

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        OPTION_MIN_STEP_DELAY_MS,
                        default=options.get(OPTION_MIN_STEP_DELAY_MS, DEFAULT_MIN_STEP_DELAY_MS),
                    ): vol.All(vol.Coerce(int), vol.Range(min=MIN_STEP_DELAY_MS, max=1000)),
                }
            ),
        )
