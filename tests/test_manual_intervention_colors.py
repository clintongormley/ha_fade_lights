"""Tests for manual intervention detection with colors."""

from unittest.mock import MagicMock

import pytest
from homeassistant.components.light import ATTR_BRIGHTNESS
from homeassistant.components.light import ATTR_COLOR_TEMP_KELVIN as HA_ATTR_COLOR_TEMP_KELVIN
from homeassistant.components.light import ATTR_HS_COLOR as HA_ATTR_HS_COLOR
from homeassistant.const import STATE_ON

from custom_components.fade_lights import (
    FADE_EXPECTED_STATE,
    ExpectedState,
    _match_and_remove_expected,
)
from custom_components.fade_lights.expected_state import ExpectedValues


@pytest.fixture
def mock_state_on_with_color():
    """Create a mock state that is ON with brightness and HS color."""
    state = MagicMock()
    state.state = STATE_ON
    state.entity_id = "light.test"
    state.domain = "light"
    state.attributes = {
        ATTR_BRIGHTNESS: 200,
        HA_ATTR_HS_COLOR: (180.0, 50.0),
    }
    return state


@pytest.fixture
def mock_state_on_with_kelvin():
    """Create a mock state that is ON with brightness and color temp kelvin."""
    state = MagicMock()
    state.state = STATE_ON
    state.entity_id = "light.test"
    state.domain = "light"
    state.attributes = {
        ATTR_BRIGHTNESS: 200,
        HA_ATTR_COLOR_TEMP_KELVIN: 3003,  # ~333 mireds equivalent
    }
    return state


class TestManualInterventionColors:
    """Test manual intervention detection with color tracking."""

    def setup_method(self):
        """Clear tracking state before each test."""
        FADE_EXPECTED_STATE.clear()

    def test_state_matches_expected_brightness_and_hs_color(self, mock_state_on_with_color):
        """Test state change matches when both brightness and HS color match."""
        entity_id = "light.test"
        FADE_EXPECTED_STATE[entity_id] = ExpectedState(entity_id=entity_id)
        FADE_EXPECTED_STATE[entity_id].add(
            ExpectedValues(brightness=200, hs_color=(180.0, 50.0))
        )

        result = _match_and_remove_expected(entity_id, mock_state_on_with_color)
        assert result is True  # Expected, should not trigger intervention

    def test_state_mismatch_wrong_hs_color_triggers_intervention(self, mock_state_on_with_color):
        """Test state mismatch when HS color is wrong triggers intervention."""
        entity_id = "light.test"
        FADE_EXPECTED_STATE[entity_id] = ExpectedState(entity_id=entity_id)
        # Expecting different color (hue 100 vs actual 180 - outside tolerance)
        FADE_EXPECTED_STATE[entity_id].add(
            ExpectedValues(brightness=200, hs_color=(100.0, 50.0))
        )

        result = _match_and_remove_expected(entity_id, mock_state_on_with_color)
        assert result is False  # Unexpected - should trigger intervention

    def test_state_matches_expected_brightness_and_kelvin(self, mock_state_on_with_kelvin):
        """Test state change matches when both brightness and kelvin match."""
        entity_id = "light.test"
        FADE_EXPECTED_STATE[entity_id] = ExpectedState(entity_id=entity_id)
        FADE_EXPECTED_STATE[entity_id].add(
            ExpectedValues(brightness=200, color_temp_kelvin=3003)
        )

        result = _match_and_remove_expected(entity_id, mock_state_on_with_kelvin)
        assert result is True  # Expected

    def test_state_mismatch_wrong_kelvin_triggers_intervention(self, mock_state_on_with_kelvin):
        """Test state mismatch when kelvin is wrong triggers intervention."""
        entity_id = "light.test"
        FADE_EXPECTED_STATE[entity_id] = ExpectedState(entity_id=entity_id)
        # Expecting different kelvin (4000 vs actual 3003 - outside tolerance of 100)
        FADE_EXPECTED_STATE[entity_id].add(
            ExpectedValues(brightness=200, color_temp_kelvin=4000)
        )

        result = _match_and_remove_expected(entity_id, mock_state_on_with_kelvin)
        assert result is False  # Unexpected - should trigger intervention

    def test_brightness_only_fade_ignores_color_changes(self, mock_state_on_with_color):
        """Test brightness-only fade doesn't care about color changes."""
        entity_id = "light.test"
        FADE_EXPECTED_STATE[entity_id] = ExpectedState(entity_id=entity_id)
        # Only tracking brightness (not color)
        FADE_EXPECTED_STATE[entity_id].add(ExpectedValues(brightness=200))

        # State has color but we weren't tracking it
        result = _match_and_remove_expected(entity_id, mock_state_on_with_color)
        assert result is True  # Match - color is ignored since not tracked
