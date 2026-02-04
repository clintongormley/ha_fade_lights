"""Tests for Fade Lights fade execution logic."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
from homeassistant.components.light import (
    ATTR_BRIGHTNESS,
    ATTR_SUPPORTED_COLOR_MODES,
)
from homeassistant.components.light.const import ColorMode
from homeassistant.const import ATTR_ENTITY_ID, STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant, ServiceCall
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.fade_lights.const import (
    ATTR_BRIGHTNESS_PCT,
    ATTR_TRANSITION,
    DOMAIN,
    SERVICE_FADE_LIGHTS,
)


@pytest.fixture
def service_calls(hass: HomeAssistant) -> list[ServiceCall]:
    """Capture light service calls and update light state accordingly.

    This fixture:
    - Registers light.turn_on and light.turn_off services
    - Captures all service calls to a list for assertion
    - Updates the mock light state based on the call so the fade logic can track progress
    - Preserves the context from the service call so fade doesn't detect "manual changes"
    """
    calls: list[ServiceCall] = []

    async def mock_turn_on(call: ServiceCall) -> None:
        """Handle turn_on service call."""
        calls.append(call)

        entity_id = call.data.get(ATTR_ENTITY_ID)
        if entity_id:
            entity_ids = entity_id if isinstance(entity_id, list) else [entity_id]
            for eid in entity_ids:
                current_state = hass.states.get(eid)
                if current_state:
                    current_attrs = dict(current_state.attributes)
                    if ATTR_BRIGHTNESS in call.data:
                        current_attrs[ATTR_BRIGHTNESS] = call.data[ATTR_BRIGHTNESS]
                    elif current_attrs.get(ATTR_BRIGHTNESS) is None:
                        # If turning on without brightness and light was off, set default
                        current_attrs[ATTR_BRIGHTNESS] = 255
                    # Preserve context so the fade logic doesn't think this is a manual change
                    hass.states.async_set(eid, STATE_ON, current_attrs, context=call.context)

    async def mock_turn_off(call: ServiceCall) -> None:
        """Handle turn_off service call."""
        calls.append(call)

        entity_id = call.data.get(ATTR_ENTITY_ID)
        if entity_id:
            entity_ids = entity_id if isinstance(entity_id, list) else [entity_id]
            for eid in entity_ids:
                current_state = hass.states.get(eid)
                if current_state:
                    current_attrs = dict(current_state.attributes)
                    current_attrs[ATTR_BRIGHTNESS] = None
                    # Preserve context so the fade logic doesn't think this is a manual change
                    hass.states.async_set(eid, STATE_OFF, current_attrs, context=call.context)

    hass.services.async_register("light", "turn_on", mock_turn_on)
    hass.services.async_register("light", "turn_off", mock_turn_off)

    return calls


def _get_turn_on_calls(calls: list[ServiceCall]) -> list[ServiceCall]:
    """Filter for turn_on calls only."""
    return [c for c in calls if c.service == "turn_on"]


def _get_turn_off_calls(calls: list[ServiceCall]) -> list[ServiceCall]:
    """Filter for turn_off calls only."""
    return [c for c in calls if c.service == "turn_off"]


def _get_final_brightness(calls: list[ServiceCall]) -> int | None:
    """Get the final brightness from the last turn_on call."""
    turn_on_calls = _get_turn_on_calls(calls)
    if turn_on_calls:
        return turn_on_calls[-1].data.get(ATTR_BRIGHTNESS)
    return None


async def test_fade_down_reaches_target(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test fading down from 78% to 20% reaches brightness 51."""
    entity_id = "light.test_fade_down"
    # 78% of 255 = 199
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 199,  # ~78%
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    await hass.services.async_call(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        {
            ATTR_ENTITY_ID: entity_id,
            ATTR_BRIGHTNESS_PCT: 20,  # 20% of 255 = 51
            ATTR_TRANSITION: 0.5,  # Short transition for faster test
        },
        blocking=True,
    )
    await hass.async_block_till_done()

    # Check the final brightness reached
    turn_on_calls = _get_turn_on_calls(service_calls)
    assert len(turn_on_calls) > 0

    # The last call should have set the target brightness
    final_brightness = _get_final_brightness(service_calls)
    assert final_brightness == 51  # 20% of 255


async def test_fade_up_reaches_target(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test fading up from 20% to 80% reaches brightness 204."""
    entity_id = "light.test_fade_up"
    # 20% of 255 = 51
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 51,  # ~20%
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    await hass.services.async_call(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        {
            ATTR_ENTITY_ID: entity_id,
            ATTR_BRIGHTNESS_PCT: 80,  # 80% of 255 = 204
            ATTR_TRANSITION: 0.5,
        },
        blocking=True,
    )
    await hass.async_block_till_done()

    # Check the final brightness reached
    turn_on_calls = _get_turn_on_calls(service_calls)
    assert len(turn_on_calls) > 0

    final_brightness = _get_final_brightness(service_calls)
    assert final_brightness == 204  # 80% of 255


async def test_fade_to_zero_turns_off(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test fading to 0% calls turn_off."""
    entity_id = "light.test_fade_to_zero"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 100,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    await hass.services.async_call(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        {
            ATTR_ENTITY_ID: entity_id,
            ATTR_BRIGHTNESS_PCT: 0,
            ATTR_TRANSITION: 0.5,
        },
        blocking=True,
    )
    await hass.async_block_till_done()

    # Should have turn_off calls
    turn_off_calls = _get_turn_off_calls(service_calls)
    assert len(turn_off_calls) > 0

    # The last call should be a turn_off
    last_call = service_calls[-1]
    assert last_call.service == "turn_off"
    assert last_call.data.get(ATTR_ENTITY_ID) == entity_id


async def test_fade_already_at_target_no_op(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test no service calls are made if already at target brightness."""
    entity_id = "light.test_already_at_target"
    # 50% of 255 = 127.5, int() = 127
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 127,  # 50%
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    await hass.services.async_call(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        {
            ATTR_ENTITY_ID: entity_id,
            ATTR_BRIGHTNESS_PCT: 50,  # Same as current
            ATTR_TRANSITION: 1,
        },
        blocking=True,
    )
    await hass.async_block_till_done()

    # No calls should be made - already at target
    assert len(service_calls) == 0


async def test_fade_skips_brightness_level_1(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test that brightness level 1 is skipped (goes to 0 or 2 instead)."""
    entity_id = "light.test_skip_level_1"
    # Start at brightness 5 and fade down to 0
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 5,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    await hass.services.async_call(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        {
            ATTR_ENTITY_ID: entity_id,
            ATTR_BRIGHTNESS_PCT: 0,
            ATTR_TRANSITION: 0.5,
        },
        blocking=True,
    )
    await hass.async_block_till_done()

    # Check that no turn_on call had brightness 1
    turn_on_calls = _get_turn_on_calls(service_calls)
    for call in turn_on_calls:
        brightness = call.data.get(ATTR_BRIGHTNESS)
        assert brightness != 1, "Brightness level 1 should be skipped"


async def test_fade_non_dimmable_to_zero(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test that fading a non-dimmable light to 0% turns it off."""
    entity_id = "light.non_dimmable"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.ONOFF],
        },
    )

    await hass.services.async_call(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        {
            ATTR_ENTITY_ID: entity_id,
            ATTR_BRIGHTNESS_PCT: 0,
            ATTR_TRANSITION: 1,
        },
        blocking=True,
    )
    await hass.async_block_till_done()

    # Should have exactly one turn_off call
    turn_off_calls = _get_turn_off_calls(service_calls)
    assert len(turn_off_calls) == 1
    assert turn_off_calls[0].data.get(ATTR_ENTITY_ID) == entity_id

    # Should not have any turn_on calls
    turn_on_calls = _get_turn_on_calls(service_calls)
    assert len(turn_on_calls) == 0


async def test_fade_non_dimmable_to_nonzero(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test that fading a non-dimmable light to >0% turns it on."""
    entity_id = "light.non_dimmable_on"
    hass.states.async_set(
        entity_id,
        STATE_OFF,
        {
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.ONOFF],
        },
    )

    await hass.services.async_call(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        {
            ATTR_ENTITY_ID: entity_id,
            ATTR_BRIGHTNESS_PCT: 50,
            ATTR_TRANSITION: 1,
        },
        blocking=True,
    )
    await hass.async_block_till_done()

    # Should have exactly one turn_on call
    turn_on_calls = _get_turn_on_calls(service_calls)
    assert len(turn_on_calls) == 1
    assert turn_on_calls[0].data.get(ATTR_ENTITY_ID) == entity_id

    # Should not have any turn_off calls
    turn_off_calls = _get_turn_off_calls(service_calls)
    assert len(turn_off_calls) == 0


async def test_fade_unknown_entity_logs_warning(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that fading an unknown entity logs a message but doesn't crash."""
    entity_id = "light.nonexistent_light"
    # Don't create the entity - it should be unknown

    with caplog.at_level(logging.ERROR):
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_ENTITY_ID: entity_id,
                ATTR_BRIGHTNESS_PCT: 50,
                ATTR_TRANSITION: 1,
            },
            blocking=True,
        )
        await hass.async_block_till_done()

    # Should have logged about the unknown light
    assert "Unknown light 'light.nonexistent_light'" in caplog.text

    # No service calls should be made for the unknown entity
    assert len(service_calls) == 0


async def test_fade_stores_orig_brightness(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test that original brightness is stored after fade completes."""
    entity_id = "light.test_store_orig"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 200,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    # Fade to 60% (153 brightness)
    await hass.services.async_call(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        {
            ATTR_ENTITY_ID: entity_id,
            ATTR_BRIGHTNESS_PCT: 60,
            ATTR_TRANSITION: 0.3,
        },
        blocking=True,
    )
    await hass.async_block_till_done()

    # Check that the original brightness was stored (flat map: entity_id -> brightness)
    storage_data = hass.data[DOMAIN]["data"]
    assert entity_id in storage_data
    assert storage_data[entity_id] == 153  # 60% of 255


async def test_fade_from_off_turns_on(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test that fading up from an off light turns it on."""
    entity_id = "light.test_from_off"
    hass.states.async_set(
        entity_id,
        STATE_OFF,
        {
            ATTR_BRIGHTNESS: None,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    await hass.services.async_call(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        {
            ATTR_ENTITY_ID: entity_id,
            ATTR_BRIGHTNESS_PCT: 50,
            ATTR_TRANSITION: 0.3,
        },
        blocking=True,
    )
    await hass.async_block_till_done()

    # Should have turn_on calls to bring the light up
    turn_on_calls = _get_turn_on_calls(service_calls)
    assert len(turn_on_calls) > 0

    # The final state should be at target brightness
    final_brightness = _get_final_brightness(service_calls)
    assert final_brightness == 127  # 50% of 255

    # Verify light state is ON
    state = hass.states.get(entity_id)
    assert state is not None
    assert state.state == STATE_ON


async def test_fade_step_count_limited_by_brightness_levels(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test that number of steps doesn't exceed brightness levels to change.

    When fading from brightness 200 to 190 (10 levels), even with a long
    transition time, we should have at most 10 steps.
    """
    entity_id = "light.test_step_count"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 200,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    # Use a long transition but small brightness change
    # 10 brightness levels, 10 second transition
    # Without the fix, this would try to do many more steps
    await hass.services.async_call(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        {
            ATTR_ENTITY_ID: entity_id,
            ATTR_BRIGHTNESS_PCT: 75,  # 75% = 191, so ~9 levels difference
            ATTR_TRANSITION: 10,
        },
        blocking=True,
    )
    await hass.async_block_till_done()

    # Should have at most ~9 turn_on calls (one per brightness level)
    turn_on_calls = _get_turn_on_calls(service_calls)
    assert len(turn_on_calls) <= 10


async def test_fade_steps_spread_across_transition_time(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test that fade steps are spread evenly across the transition time.

    With default step_delay_ms of 50ms and a 500ms transition,
    we should get approximately 10 steps maximum.
    """
    entity_id = "light.test_transition_spread"
    # 100 brightness levels to change (255 -> 0)
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 255,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    await hass.services.async_call(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        {
            ATTR_ENTITY_ID: entity_id,
            ATTR_BRIGHTNESS_PCT: 0,
            ATTR_TRANSITION: 0.5,  # 500ms transition
        },
        blocking=True,
    )
    await hass.async_block_till_done()

    # With 500ms transition and 50ms step delay, max 10 steps
    # But we have 255 brightness levels, so steps should be ~10
    # Each step changes multiple brightness levels
    total_calls = len(service_calls)
    # Should have roughly 10 steps (500ms / 50ms)
    assert total_calls <= 15  # Allow buffer for rounding


async def test_fade_timing_accounts_for_service_call_duration(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test that fade timing compensates for service call duration.

    When a service call takes time, the sleep duration should be reduced
    so that each step still completes in approximately the target delay time.
    This test mocks time.monotonic() to simulate service call latency.
    """
    entity_id = "light.test_timing"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 255,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    sleep_durations: list[float] = []

    async def mock_sleep(duration: float) -> None:
        """Capture sleep durations."""
        sleep_durations.append(duration)
        # Don't actually sleep to speed up test

    # Simulate service calls taking 30ms each (0.030 seconds)
    # With a target delay of 50ms, sleep should be ~20ms (0.020 seconds)
    call_count = 0
    base_time = 1000.0

    def mock_monotonic() -> float:
        """Return simulated time that advances 30ms per service call."""
        nonlocal call_count
        # Each call to monotonic alternates between start and end of service call
        # Start of step: base_time + (step * 0.050)
        # After service call: start + 0.030 (simulating 30ms service call)
        result = base_time + (call_count * 0.030)
        call_count += 1
        return result

    with (
        patch("asyncio.sleep", side_effect=mock_sleep),
        patch("time.monotonic", side_effect=mock_monotonic),
    ):
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_ENTITY_ID: entity_id,
                ATTR_BRIGHTNESS_PCT: 90,  # Small change for fewer steps
                ATTR_TRANSITION: 0.5,  # 500ms
            },
            blocking=True,
        )
        await hass.async_block_till_done()

    # Verify sleep was called with reduced durations
    # With 30ms service calls and 50ms target delay, sleep should be ~20ms
    # Some sleeps might be skipped if service call took longer than delay
    for duration in sleep_durations:
        # Sleep duration should be less than the full delay (since service call time is subtracted)
        # Duration is in seconds, should be around 0.020 or less
        assert duration <= 0.050, f"Sleep duration {duration} should be <= 0.050s (50ms delay)"
        assert duration >= 0, f"Sleep duration {duration} should be non-negative"


async def test_fade_entity_not_found_logs_warning(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that _execute_fade logs warning when entity doesn't exist.

    This tests lines 514-515 where state is None.
    """
    import asyncio

    from custom_components.fade_lights import _execute_fade

    entity_id = "light.missing_entity"
    cancel_event = asyncio.Event()

    with caplog.at_level(logging.WARNING):
        await _execute_fade(
            hass,
            entity_id,
            50,  # brightness_pct
            1000,  # transition_ms
            50,  # min_step_delay_ms
            cancel_event,
        )

    assert f"Entity {entity_id} not found" in caplog.text
    assert len(service_calls) == 0


async def test_fade_cancel_event_before_brightness_apply(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test that cancel event stops fade before applying brightness.

    This tests line 552 where cancel_event.is_set() at start of loop.
    """
    import asyncio

    from custom_components.fade_lights import _execute_fade

    entity_id = "light.test_cancel_before"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 200,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    cancel_event = asyncio.Event()
    # Set cancel event BEFORE starting fade
    cancel_event.set()

    await _execute_fade(
        hass,
        entity_id,
        50,  # brightness_pct
        5000,  # long transition
        50,  # min_step_delay_ms
        cancel_event,
    )

    # No service calls should have been made since cancel was set
    assert len(service_calls) == 0


async def test_fade_cancel_event_after_brightness_apply(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test that cancel event stops fade after applying brightness.

    This tests line 562 where cancel_event.is_set() after _apply_brightness.
    """
    import asyncio

    from custom_components.fade_lights import _execute_fade

    entity_id = "light.test_cancel_after"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 200,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    cancel_event = asyncio.Event()

    # Create a mock for _apply_brightness that sets cancel event after first call
    original_apply = None
    call_count = 0

    async def cancelling_apply(hass, eid, level):
        nonlocal call_count, original_apply
        call_count += 1
        # After first apply, set the cancel event
        if call_count == 1:
            cancel_event.set()
        # Actually apply the brightness
        if level == 0:
            await hass.services.async_call(
                "light", "turn_off", {ATTR_ENTITY_ID: eid}, blocking=True
            )
        else:
            await hass.services.async_call(
                "light", "turn_on", {ATTR_ENTITY_ID: eid, ATTR_BRIGHTNESS: level}, blocking=True
            )

    with patch("custom_components.fade_lights._apply_brightness", side_effect=cancelling_apply):
        await _execute_fade(
            hass,
            entity_id,
            10,  # brightness_pct - need big change for multiple steps
            5000,  # long transition
            50,  # min_step_delay_ms
            cancel_event,
        )

    # Should have only made 1 service call before cancel took effect
    assert len(service_calls) == 1
