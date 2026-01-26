"""Tests for condition-based event waiting."""

from __future__ import annotations

import asyncio
import time

import pytest
from homeassistant.components.light import ATTR_BRIGHTNESS, ATTR_SUPPORTED_COLOR_MODES
from homeassistant.components.light.const import ColorMode
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.fade_lights import (
    FADE_EXPECTED_BRIGHTNESS,
    ExpectedState,
    _add_expected_brightness,
    _match_and_remove_expected,
    _prune_expected_brightness,
    _wait_until_stale_events_flushed,
)


async def test_add_expected_brightness_creates_entry(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test _add_expected_brightness creates ExpectedState if not exists."""
    entity_id = "light.test_add"

    # Ensure no entry exists
    FADE_EXPECTED_BRIGHTNESS.pop(entity_id, None)

    _add_expected_brightness(entity_id, 100)

    assert entity_id in FADE_EXPECTED_BRIGHTNESS
    assert 100 in FADE_EXPECTED_BRIGHTNESS[entity_id].values
    assert FADE_EXPECTED_BRIGHTNESS[entity_id]._condition is None

    # Clean up
    FADE_EXPECTED_BRIGHTNESS.pop(entity_id, None)


async def test_add_expected_brightness_updates_timestamp(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test _add_expected_brightness updates timestamp for same brightness."""
    entity_id = "light.test_timestamp"

    FADE_EXPECTED_BRIGHTNESS.pop(entity_id, None)

    _add_expected_brightness(entity_id, 100)
    first_timestamp = FADE_EXPECTED_BRIGHTNESS[entity_id].values[100]

    await asyncio.sleep(0.01)

    _add_expected_brightness(entity_id, 100)
    second_timestamp = FADE_EXPECTED_BRIGHTNESS[entity_id].values[100]

    assert second_timestamp > first_timestamp

    # Clean up
    FADE_EXPECTED_BRIGHTNESS.pop(entity_id, None)


async def test_match_and_remove_expected_removes_matched_value(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test _match_and_remove_expected removes matched brightness."""
    entity_id = "light.test_match"

    hass.states.async_set(
        entity_id,
        STATE_ON,
        {ATTR_BRIGHTNESS: 100, ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS]},
    )

    FADE_EXPECTED_BRIGHTNESS[entity_id] = ExpectedState(values={100: time.monotonic()})

    state = hass.states.get(entity_id)
    result = _match_and_remove_expected(entity_id, state)

    assert result is True
    assert 100 not in FADE_EXPECTED_BRIGHTNESS[entity_id].values

    # Clean up
    FADE_EXPECTED_BRIGHTNESS.pop(entity_id, None)


async def test_match_and_remove_expected_with_tolerance(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test _match_and_remove_expected matches within tolerance."""
    entity_id = "light.test_tolerance"

    hass.states.async_set(
        entity_id,
        STATE_ON,
        {ATTR_BRIGHTNESS: 102, ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS]},
    )

    FADE_EXPECTED_BRIGHTNESS[entity_id] = ExpectedState(values={100: time.monotonic()})

    state = hass.states.get(entity_id)
    result = _match_and_remove_expected(entity_id, state)

    assert result is True
    assert 100 not in FADE_EXPECTED_BRIGHTNESS[entity_id].values

    # Clean up
    FADE_EXPECTED_BRIGHTNESS.pop(entity_id, None)


async def test_match_and_remove_expected_off_state(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test _match_and_remove_expected matches OFF state."""
    entity_id = "light.test_off"

    hass.states.async_set(
        entity_id,
        STATE_OFF,
        {ATTR_BRIGHTNESS: None, ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS]},
    )

    FADE_EXPECTED_BRIGHTNESS[entity_id] = ExpectedState(values={0: time.monotonic()})

    state = hass.states.get(entity_id)
    result = _match_and_remove_expected(entity_id, state)

    assert result is True
    assert 0 not in FADE_EXPECTED_BRIGHTNESS[entity_id].values

    # Clean up
    FADE_EXPECTED_BRIGHTNESS.pop(entity_id, None)


async def test_match_and_remove_expected_notifies_condition(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test _match_and_remove_expected notifies condition when empty."""
    entity_id = "light.test_notify"

    hass.states.async_set(
        entity_id,
        STATE_ON,
        {ATTR_BRIGHTNESS: 100, ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS]},
    )

    expected_state = ExpectedState(values={100: time.monotonic()})
    condition = expected_state.get_condition()  # Create condition via method
    FADE_EXPECTED_BRIGHTNESS[entity_id] = expected_state

    notified = asyncio.Event()

    async def wait_for_notification() -> None:
        async with condition:
            await condition.wait()
            notified.set()

    wait_task = asyncio.create_task(wait_for_notification())
    await asyncio.sleep(0.01)  # Let wait_task start

    state = hass.states.get(entity_id)
    _match_and_remove_expected(entity_id, state)

    await asyncio.sleep(0.05)  # Let notification propagate

    assert notified.is_set()

    wait_task.cancel()
    try:
        await wait_task
    except asyncio.CancelledError:
        pass

    # Clean up
    FADE_EXPECTED_BRIGHTNESS.pop(entity_id, None)


async def test_prune_expected_brightness_removes_stale(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test _prune_expected_brightness removes old values."""
    entity_id = "light.test_prune"

    # Create entry with old timestamp (6 seconds ago)
    old_timestamp = time.monotonic() - 6.0
    FADE_EXPECTED_BRIGHTNESS[entity_id] = ExpectedState(values={100: old_timestamp})

    _prune_expected_brightness(entity_id)

    # Entry should be completely removed since it's now empty
    assert entity_id not in FADE_EXPECTED_BRIGHTNESS


async def test_prune_expected_brightness_keeps_fresh(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test _prune_expected_brightness keeps fresh values."""
    entity_id = "light.test_prune_fresh"

    # Create entry with fresh timestamp
    FADE_EXPECTED_BRIGHTNESS[entity_id] = ExpectedState(values={100: time.monotonic()})

    _prune_expected_brightness(entity_id)

    # Entry should still exist
    assert entity_id in FADE_EXPECTED_BRIGHTNESS
    assert 100 in FADE_EXPECTED_BRIGHTNESS[entity_id].values

    # Clean up
    FADE_EXPECTED_BRIGHTNESS.pop(entity_id, None)


async def test_wait_until_stale_events_flushed_returns_immediately_when_empty(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test _wait_until_stale_events_flushed returns immediately when no expected values."""
    entity_id = "light.test_empty"

    FADE_EXPECTED_BRIGHTNESS.pop(entity_id, None)

    start = time.monotonic()
    await _wait_until_stale_events_flushed(entity_id)
    elapsed = time.monotonic() - start

    # Should return almost immediately
    assert elapsed < 0.1


async def test_wait_until_stale_events_flushed_times_out(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test _wait_until_stale_events_flushed times out when events don't arrive."""
    entity_id = "light.test_timeout"

    FADE_EXPECTED_BRIGHTNESS[entity_id] = ExpectedState(values={100: time.monotonic()})

    start = time.monotonic()
    await _wait_until_stale_events_flushed(entity_id, timeout=0.2)
    elapsed = time.monotonic() - start

    # Should wait approximately the timeout duration
    assert elapsed >= 0.2
    assert elapsed < 0.5

    # Clean up
    FADE_EXPECTED_BRIGHTNESS.pop(entity_id, None)


async def test_wait_until_stale_events_flushed_returns_when_notified(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test _wait_until_stale_events_flushed returns early when condition is notified."""
    entity_id = "light.test_early_return"

    expected_state = ExpectedState(values={100: time.monotonic()})
    condition = expected_state.get_condition()  # Create condition via method
    FADE_EXPECTED_BRIGHTNESS[entity_id] = expected_state

    async def clear_and_notify() -> None:
        await asyncio.sleep(0.1)
        expected_state.values.clear()
        async with condition:
            condition.notify_all()

    asyncio.create_task(clear_and_notify())

    start = time.monotonic()
    await _wait_until_stale_events_flushed(entity_id, timeout=5.0)
    elapsed = time.monotonic() - start

    # Should return well before the 5 second timeout
    assert elapsed < 0.5

    # Clean up
    FADE_EXPECTED_BRIGHTNESS.pop(entity_id, None)