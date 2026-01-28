"""Tests for _get_current_color_state function."""

from __future__ import annotations

from unittest.mock import MagicMock

from custom_components.fade_lights import _get_current_color_state


class TestGetCurrentColorState:
    """Test _get_current_color_state function."""

    def test_returns_hs_from_state(self) -> None:
        """Test extracting HS color from state attributes."""
        state = MagicMock()
        state.attributes = {"hs_color": (120.0, 80.0)}

        hs, mireds = _get_current_color_state(state)

        assert hs == (120.0, 80.0)
        assert mireds is None

    def test_returns_color_temp_from_state(self) -> None:
        """Test extracting color temp from state attributes."""
        state = MagicMock()
        state.attributes = {"color_temp": 300}

        hs, mireds = _get_current_color_state(state)

        assert hs is None
        assert mireds == 300

    def test_returns_both_when_present(self) -> None:
        """Test extracting both HS and color temp when both present."""
        state = MagicMock()
        state.attributes = {"hs_color": (50.0, 60.0), "color_temp": 250}

        hs, mireds = _get_current_color_state(state)

        assert hs == (50.0, 60.0)
        assert mireds == 250

    def test_returns_none_when_missing(self) -> None:
        """Test returns None when color attributes not present."""
        state = MagicMock()
        state.attributes = {"brightness": 128}

        hs, mireds = _get_current_color_state(state)

        assert hs is None
        assert mireds is None

    def test_handles_hs_color_as_list(self) -> None:
        """Test handling HS color as list (some integrations return list)."""
        state = MagicMock()
        state.attributes = {"hs_color": [100.0, 75.0]}

        hs, mireds = _get_current_color_state(state)

        assert hs == (100.0, 75.0)
        assert mireds is None
