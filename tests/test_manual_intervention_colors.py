"""Tests for manual intervention detection with colors."""

from unittest.mock import MagicMock

import pytest
from homeassistant.components.light import ATTR_BRIGHTNESS
from homeassistant.components.light import ATTR_HS_COLOR as HA_ATTR_HS_COLOR
from homeassistant.const import STATE_ON

from custom_components.fade_lights import (
    FADE_EXPECTED_BRIGHTNESS,
    ExpectedState,
    _match_and_remove_expected,
)
from custom_components.fade_lights.models import ExpectedValues


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
def mock_state_on_with_mireds():
    """Create a mock state that is ON with brightness and color temp."""
    state = MagicMock()
    state.state = STATE_ON
    state.entity_id = "light.test"
    state.domain = "light"
    state.attributes = {
        ATTR_BRIGHTNESS: 200,
        "color_temp": 333,  # mireds
    }
    return state


class TestManualInterventionColors:
    """Test manual intervention detection with color tracking."""

    def setup_method(self):
        """Clear tracking state before each test."""
        FADE_EXPECTED_BRIGHTNESS.clear()

    def test_state_matches_expected_brightness_and_hs_color(self, mock_state_on_with_color):
        """Test state change matches when both brightness and HS color match."""
        entity_id = "light.test"
        FADE_EXPECTED_BRIGHTNESS[entity_id] = ExpectedState(entity_id=entity_id)
        FADE_EXPECTED_BRIGHTNESS[entity_id].add(
            ExpectedValues(brightness=200, hs_color=(180.0, 50.0))
        )

        result = _match_and_remove_expected(entity_id, mock_state_on_with_color)
        assert result is True  # Expected, should not trigger intervention

    def test_state_mismatch_wrong_hs_color_triggers_intervention(self, mock_state_on_with_color):
        """Test state mismatch when HS color is wrong triggers intervention."""
        entity_id = "light.test"
        FADE_EXPECTED_BRIGHTNESS[entity_id] = ExpectedState(entity_id=entity_id)
        # Expecting different color (hue 100 vs actual 180 - outside tolerance)
        FADE_EXPECTED_BRIGHTNESS[entity_id].add(
            ExpectedValues(brightness=200, hs_color=(100.0, 50.0))
        )

        result = _match_and_remove_expected(entity_id, mock_state_on_with_color)
        assert result is False  # Unexpected - should trigger intervention

    def test_state_matches_expected_brightness_and_mireds(self, mock_state_on_with_mireds):
        """Test state change matches when both brightness and mireds match."""
        entity_id = "light.test"
        FADE_EXPECTED_BRIGHTNESS[entity_id] = ExpectedState(entity_id=entity_id)
        FADE_EXPECTED_BRIGHTNESS[entity_id].add(
            ExpectedValues(brightness=200, color_temp_mireds=333)
        )

        result = _match_and_remove_expected(entity_id, mock_state_on_with_mireds)
        assert result is True  # Expected

    def test_state_mismatch_wrong_mireds_triggers_intervention(self, mock_state_on_with_mireds):
        """Test state mismatch when mireds is wrong triggers intervention."""
        entity_id = "light.test"
        FADE_EXPECTED_BRIGHTNESS[entity_id] = ExpectedState(entity_id=entity_id)
        # Expecting different mireds (250 vs actual 333 - outside tolerance)
        FADE_EXPECTED_BRIGHTNESS[entity_id].add(
            ExpectedValues(brightness=200, color_temp_mireds=250)
        )

        result = _match_and_remove_expected(entity_id, mock_state_on_with_mireds)
        assert result is False  # Unexpected - should trigger intervention

    def test_brightness_only_fade_ignores_color_changes(self, mock_state_on_with_color):
        """Test brightness-only fade doesn't care about color changes."""
        entity_id = "light.test"
        FADE_EXPECTED_BRIGHTNESS[entity_id] = ExpectedState(entity_id=entity_id)
        # Only tracking brightness (not color)
        FADE_EXPECTED_BRIGHTNESS[entity_id].add(ExpectedValues(brightness=200))

        # State has color but we weren't tracking it
        result = _match_and_remove_expected(entity_id, mock_state_on_with_color)
        assert result is True  # Match - color is ignored since not tracked
