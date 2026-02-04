"""Tests for light capability filtering."""

from __future__ import annotations

from homeassistant.components.light import ATTR_SUPPORTED_COLOR_MODES
from homeassistant.components.light.const import ColorMode
from homeassistant.const import STATE_ON
from homeassistant.core import HomeAssistant

from custom_components.fade_lights import _can_fade_color
from custom_components.fade_lights.models import FadeParams


class TestCanFadeColor:
    """Test _can_fade_color capability checks."""

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
        assert _can_fade_color(state, params) is True

    async def test_hs_color_allowed_on_rgb_light(self, hass: HomeAssistant) -> None:
        """Test HS fade allowed on light with RGB mode (RGB can render HS)."""
        entity_id = self._make_state(hass, [ColorMode.RGB])
        state = hass.states.get(entity_id)
        params = FadeParams(hs_color=(200.0, 80.0))
        assert _can_fade_color(state, params) is True

    async def test_hs_color_allowed_on_rgbw_light(self, hass: HomeAssistant) -> None:
        """Test HS fade allowed on light with RGBW mode."""
        entity_id = self._make_state(hass, [ColorMode.RGBW])
        state = hass.states.get(entity_id)
        params = FadeParams(hs_color=(200.0, 80.0))
        assert _can_fade_color(state, params) is True

    async def test_hs_color_allowed_on_rgbww_light(self, hass: HomeAssistant) -> None:
        """Test HS fade allowed on light with RGBWW mode."""
        entity_id = self._make_state(hass, [ColorMode.RGBWW])
        state = hass.states.get(entity_id)
        params = FadeParams(hs_color=(200.0, 80.0))
        assert _can_fade_color(state, params) is True

    async def test_hs_color_allowed_on_xy_light(self, hass: HomeAssistant) -> None:
        """Test HS fade allowed on light with XY mode."""
        entity_id = self._make_state(hass, [ColorMode.XY])
        state = hass.states.get(entity_id)
        params = FadeParams(hs_color=(200.0, 80.0))
        assert _can_fade_color(state, params) is True

    async def test_hs_color_rejected_on_brightness_only(self, hass: HomeAssistant) -> None:
        """Test HS fade rejected on brightness-only light."""
        entity_id = self._make_state(hass, [ColorMode.BRIGHTNESS])
        state = hass.states.get(entity_id)
        params = FadeParams(hs_color=(200.0, 80.0))
        assert _can_fade_color(state, params) is False

    async def test_hs_color_rejected_on_color_temp_only(self, hass: HomeAssistant) -> None:
        """Test HS fade rejected on color-temp-only light."""
        entity_id = self._make_state(hass, [ColorMode.COLOR_TEMP])
        state = hass.states.get(entity_id)
        params = FadeParams(hs_color=(200.0, 80.0))
        assert _can_fade_color(state, params) is False

    async def test_color_temp_allowed_on_color_temp_light(self, hass: HomeAssistant) -> None:
        """Test color temp fade allowed on COLOR_TEMP light."""
        entity_id = self._make_state(hass, [ColorMode.COLOR_TEMP])
        state = hass.states.get(entity_id)
        params = FadeParams(color_temp_mireds=250)
        assert _can_fade_color(state, params) is True

    async def test_color_temp_rejected_on_hs_only(self, hass: HomeAssistant) -> None:
        """Test color temp fade rejected on HS-only light."""
        entity_id = self._make_state(hass, [ColorMode.HS])
        state = hass.states.get(entity_id)
        params = FadeParams(color_temp_mireds=250)
        assert _can_fade_color(state, params) is False

    async def test_color_temp_rejected_on_brightness_only(self, hass: HomeAssistant) -> None:
        """Test color temp fade rejected on brightness-only light."""
        entity_id = self._make_state(hass, [ColorMode.BRIGHTNESS])
        state = hass.states.get(entity_id)
        params = FadeParams(color_temp_mireds=250)
        assert _can_fade_color(state, params) is False

    async def test_brightness_only_allowed_on_any_dimmable(self, hass: HomeAssistant) -> None:
        """Test brightness-only fade passes (no color filtering needed)."""
        entity_id = self._make_state(hass, [ColorMode.BRIGHTNESS])
        state = hass.states.get(entity_id)
        params = FadeParams(brightness_pct=50)
        assert _can_fade_color(state, params) is True

    async def test_no_color_params_always_allowed(self, hass: HomeAssistant) -> None:
        """Test FadeParams with no color targets always passes."""
        entity_id = self._make_state(hass, [ColorMode.ONOFF])
        state = hass.states.get(entity_id)
        params = FadeParams()
        assert _can_fade_color(state, params) is True

    async def test_multi_mode_light_allows_hs(self, hass: HomeAssistant) -> None:
        """Test light with multiple modes (COLOR_TEMP + HS) allows HS."""
        entity_id = self._make_state(hass, [ColorMode.COLOR_TEMP, ColorMode.HS])
        state = hass.states.get(entity_id)
        params = FadeParams(hs_color=(200.0, 80.0))
        assert _can_fade_color(state, params) is True

    async def test_multi_mode_light_allows_color_temp(self, hass: HomeAssistant) -> None:
        """Test light with multiple modes (COLOR_TEMP + HS) allows color temp."""
        entity_id = self._make_state(hass, [ColorMode.COLOR_TEMP, ColorMode.HS])
        state = hass.states.get(entity_id)
        params = FadeParams(color_temp_mireds=250)
        assert _can_fade_color(state, params) is True
