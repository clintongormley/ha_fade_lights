"""Tests for light capability filtering."""

from __future__ import annotations

import logging

import pytest
from homeassistant.components.light import ATTR_SUPPORTED_COLOR_MODES
from homeassistant.components.light.const import ColorMode
from homeassistant.const import STATE_ON
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.fado.coordinator import _can_apply_fade_params
from custom_components.fado.const import (
    ATTR_BRIGHTNESS_PCT,
    ATTR_COLOR_TEMP_KELVIN,
    ATTR_HS_COLOR,
    ATTR_TRANSITION,
    DOMAIN,
    SERVICE_FADO,
)
from custom_components.fado.fade_params import FadeParams


class TestCanFadeColor:
    """Test _can_apply_fade_params capability checks."""

    def _make_state(self, hass: HomeAssistant, modes: list[ColorMode]) -> str:
        """Create a light entity with given color modes and return its entity_id."""
        entity_id = f"light.test_{id(modes)}"
        hass.states.async_set(
            entity_id,
            STATE_ON,
            {ATTR_SUPPORTED_COLOR_MODES: modes},
        )
        return entity_id

    async def test_hs_color_allowed_on_hs_light(self, hass: HomeAssistant) -> None:
        """Test HS fade allowed on light with HS mode."""
        entity_id = self._make_state(hass, [ColorMode.HS])
        state = hass.states.get(entity_id)
        params = FadeParams(hs_color=(200.0, 80.0))
        assert _can_apply_fade_params(state, params) is True

    async def test_hs_color_allowed_on_rgb_light(self, hass: HomeAssistant) -> None:
        """Test HS fade allowed on light with RGB mode (RGB can render HS)."""
        entity_id = self._make_state(hass, [ColorMode.RGB])
        state = hass.states.get(entity_id)
        params = FadeParams(hs_color=(200.0, 80.0))
        assert _can_apply_fade_params(state, params) is True

    async def test_hs_color_allowed_on_rgbw_light(self, hass: HomeAssistant) -> None:
        """Test HS fade allowed on light with RGBW mode."""
        entity_id = self._make_state(hass, [ColorMode.RGBW])
        state = hass.states.get(entity_id)
        params = FadeParams(hs_color=(200.0, 80.0))
        assert _can_apply_fade_params(state, params) is True

    async def test_hs_color_allowed_on_rgbww_light(self, hass: HomeAssistant) -> None:
        """Test HS fade allowed on light with RGBWW mode."""
        entity_id = self._make_state(hass, [ColorMode.RGBWW])
        state = hass.states.get(entity_id)
        params = FadeParams(hs_color=(200.0, 80.0))
        assert _can_apply_fade_params(state, params) is True

    async def test_hs_color_allowed_on_xy_light(self, hass: HomeAssistant) -> None:
        """Test HS fade allowed on light with XY mode."""
        entity_id = self._make_state(hass, [ColorMode.XY])
        state = hass.states.get(entity_id)
        params = FadeParams(hs_color=(200.0, 80.0))
        assert _can_apply_fade_params(state, params) is True

    async def test_hs_color_rejected_on_brightness_only(self, hass: HomeAssistant) -> None:
        """Test HS fade rejected on brightness-only light."""
        entity_id = self._make_state(hass, [ColorMode.BRIGHTNESS])
        state = hass.states.get(entity_id)
        params = FadeParams(hs_color=(200.0, 80.0))
        assert _can_apply_fade_params(state, params) is False

    async def test_hs_color_rejected_on_color_temp_only(self, hass: HomeAssistant) -> None:
        """Test HS fade rejected on color-temp-only light."""
        entity_id = self._make_state(hass, [ColorMode.COLOR_TEMP])
        state = hass.states.get(entity_id)
        params = FadeParams(hs_color=(200.0, 80.0))
        assert _can_apply_fade_params(state, params) is False

    async def test_color_temp_allowed_on_color_temp_light(self, hass: HomeAssistant) -> None:
        """Test color temp fade allowed on COLOR_TEMP light."""
        entity_id = self._make_state(hass, [ColorMode.COLOR_TEMP])
        state = hass.states.get(entity_id)
        params = FadeParams(color_temp_kelvin=4000)
        assert _can_apply_fade_params(state, params) is True

    async def test_color_temp_rejected_on_hs_only(self, hass: HomeAssistant) -> None:
        """Test color temp fade rejected on HS-only light."""
        entity_id = self._make_state(hass, [ColorMode.HS])
        state = hass.states.get(entity_id)
        params = FadeParams(color_temp_kelvin=4000)
        assert _can_apply_fade_params(state, params) is False

    async def test_color_temp_rejected_on_brightness_only(self, hass: HomeAssistant) -> None:
        """Test color temp fade rejected on brightness-only light."""
        entity_id = self._make_state(hass, [ColorMode.BRIGHTNESS])
        state = hass.states.get(entity_id)
        params = FadeParams(color_temp_kelvin=4000)
        assert _can_apply_fade_params(state, params) is False

    async def test_brightness_only_allowed_on_any_dimmable(self, hass: HomeAssistant) -> None:
        """Test brightness-only fade passes (no color filtering needed)."""
        entity_id = self._make_state(hass, [ColorMode.BRIGHTNESS])
        state = hass.states.get(entity_id)
        params = FadeParams(brightness_pct=50)
        assert _can_apply_fade_params(state, params) is True

    async def test_no_params_returns_false(self, hass: HomeAssistant) -> None:
        """Test FadeParams with no targets returns False (nothing to do)."""
        entity_id = self._make_state(hass, [ColorMode.ONOFF])
        state = hass.states.get(entity_id)
        params = FadeParams()
        assert _can_apply_fade_params(state, params) is False

    async def test_multi_mode_light_allows_hs(self, hass: HomeAssistant) -> None:
        """Test light with multiple modes (COLOR_TEMP + HS) allows HS."""
        entity_id = self._make_state(hass, [ColorMode.COLOR_TEMP, ColorMode.HS])
        state = hass.states.get(entity_id)
        params = FadeParams(hs_color=(200.0, 80.0))
        assert _can_apply_fade_params(state, params) is True

    async def test_multi_mode_light_allows_color_temp(self, hass: HomeAssistant) -> None:
        """Test light with multiple modes (COLOR_TEMP + HS) allows color temp."""
        entity_id = self._make_state(hass, [ColorMode.COLOR_TEMP, ColorMode.HS])
        state = hass.states.get(entity_id)
        params = FadeParams(color_temp_kelvin=4000)
        assert _can_apply_fade_params(state, params) is True


class TestServiceHandlerFiltering:
    """Test that the service handler skips incompatible lights."""

    async def test_skips_brightness_only_light_for_hs_fade(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that a brightness-only light is skipped for HS fade."""
        entity_id = "light.brightness_only"
        hass.states.async_set(
            entity_id,
            STATE_ON,
            {ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS]},
        )

        with caplog.at_level(logging.INFO):
            await hass.services.async_call(
                DOMAIN,
                SERVICE_FADO,
                {
                    ATTR_HS_COLOR: [200, 80],
                    ATTR_TRANSITION: 0.1,
                },
                target={"entity_id": entity_id},
                blocking=True,
            )

        assert "Skipping" in caplog.text
        assert entity_id in caplog.text

    async def test_skips_brightness_only_light_for_color_temp_fade(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that a brightness-only light is skipped for color temp fade."""
        entity_id = "light.brightness_only_ct"
        hass.states.async_set(
            entity_id,
            STATE_ON,
            {ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS]},
        )

        with caplog.at_level(logging.INFO):
            await hass.services.async_call(
                DOMAIN,
                SERVICE_FADO,
                {
                    ATTR_COLOR_TEMP_KELVIN: 4000,
                    ATTR_TRANSITION: 0.1,
                },
                target={"entity_id": entity_id},
                blocking=True,
            )

        assert "Skipping" in caplog.text
        assert entity_id in caplog.text

    async def test_does_not_skip_capable_light(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that a capable light is NOT skipped."""
        entity_id = "light.rgb_capable"
        hass.states.async_set(
            entity_id,
            STATE_ON,
            {ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS, ColorMode.COLOR_TEMP]},
        )

        with caplog.at_level(logging.INFO):
            await hass.services.async_call(
                DOMAIN,
                SERVICE_FADO,
                {
                    ATTR_HS_COLOR: [200, 80],
                    ATTR_TRANSITION: 0.1,
                },
                target={"entity_id": entity_id},
                blocking=True,
            )

        assert "Skipping" not in caplog.text

    async def test_group_skips_incompatible_members(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test group fade skips incompatible members, fades compatible ones."""
        capable = "light.capable_member"
        hass.states.async_set(
            capable,
            STATE_ON,
            {ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS]},
        )

        incapable = "light.incapable_member"
        hass.states.async_set(
            incapable,
            STATE_ON,
            {ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS]},
        )

        group_id = "light.mixed_group"
        hass.states.async_set(
            group_id,
            STATE_ON,
            {
                ATTR_SUPPORTED_COLOR_MODES: [ColorMode.HS],
                "entity_id": [capable, incapable],
            },
        )

        with caplog.at_level(logging.INFO):
            await hass.services.async_call(
                DOMAIN,
                SERVICE_FADO,
                {
                    ATTR_HS_COLOR: [200, 80],
                    ATTR_TRANSITION: 0.1,
                },
                target={"entity_id": group_id},
                blocking=True,
            )

        assert "Skipping" in caplog.text
        assert incapable in caplog.text

    async def test_brightness_only_fade_not_skipped(
        self,
        hass: HomeAssistant,
        init_integration: MockConfigEntry,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test brightness-only fade is NOT skipped even on brightness-only light."""
        entity_id = "light.brightness_only_fade"
        hass.states.async_set(
            entity_id,
            STATE_ON,
            {ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS]},
        )

        with caplog.at_level(logging.INFO):
            await hass.services.async_call(
                DOMAIN,
                SERVICE_FADO,
                {
                    ATTR_BRIGHTNESS_PCT: 50,
                    ATTR_TRANSITION: 0.1,
                },
                target={"entity_id": entity_id},
                blocking=True,
            )

        assert "Skipping" not in caplog.text
