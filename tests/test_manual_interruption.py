"""Tests for manual interruption detection during fades."""

from __future__ import annotations

import asyncio
import contextlib
import time

import pytest
from homeassistant.components.light import (
    ATTR_BRIGHTNESS,
    ATTR_SUPPORTED_COLOR_MODES,
)
from homeassistant.components.light.const import ColorMode
from homeassistant.const import ATTR_ENTITY_ID, STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant, ServiceCall
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.fade_lights import (
    ACTIVE_FADES,
    FADE_CANCEL_EVENTS,
    FADE_EXPECTED_STATE,
    INTENDED_STATE_QUEUE,
    RESTORE_TASKS,
    ExpectedState,
)
from custom_components.fade_lights.const import (
    ATTR_BRIGHTNESS_PCT,
    ATTR_TRANSITION,
    DOMAIN,
    SERVICE_FADE_LIGHTS,
)
from custom_components.fade_lights.expected_state import ExpectedValues


@pytest.fixture
def service_calls(hass: HomeAssistant) -> list[ServiceCall]:
    """Capture light service calls and update light state accordingly.

    This fixture:
    - Registers light.turn_on and light.turn_off services
    - Captures all service calls to a list for assertion
    - Updates the mock light state based on the call
    - Preserves the context from the service call
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
                        current_attrs[ATTR_BRIGHTNESS] = 255
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
                    hass.states.async_set(eid, STATE_OFF, current_attrs, context=call.context)

    hass.services.async_register("light", "turn_on", mock_turn_on)
    hass.services.async_register("light", "turn_off", mock_turn_off)

    return calls


async def test_manual_brightness_change_cancels_fade(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test that an external brightness change stops an active fade."""
    entity_id = "light.test_manual_brightness"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 200,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    # Start a long fade (non-blocking so we can interrupt it)
    fade_task = hass.async_create_task(
        hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_BRIGHTNESS_PCT: 20,  # Fade down to 51 brightness
                ATTR_TRANSITION: 5,  # Long transition so we can interrupt
            },
            target={"entity_id": entity_id},
            blocking=True,
        )
    )

    # Wait for the fade to start
    await asyncio.sleep(0.2)

    # Verify fade is active
    assert entity_id in ACTIVE_FADES

    # Simulate a manual brightness change (external user sets brightness to 100)
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 100,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )
    await hass.async_block_till_done()

    # Give a moment for cancellation to process
    await asyncio.sleep(0.1)

    # Verify fade was cancelled
    assert entity_id not in ACTIVE_FADES

    # Cancel the task to clean up
    fade_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await fade_task


async def test_manual_turn_off_cancels_fade(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test that turning off the light stops an active fade."""
    entity_id = "light.test_manual_off"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 200,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    # Start a long fade
    fade_task = hass.async_create_task(
        hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_BRIGHTNESS_PCT: 50,
                ATTR_TRANSITION: 5,
            },
            target={"entity_id": entity_id},
            blocking=True,
        )
    )

    # Wait for the fade to start
    await asyncio.sleep(0.2)

    # Verify fade is active
    assert entity_id in ACTIVE_FADES

    # Simulate turning off the light manually
    hass.states.async_set(
        entity_id,
        STATE_OFF,
        {
            ATTR_BRIGHTNESS: None,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )
    await hass.async_block_till_done()

    # Give a moment for cancellation to process
    await asyncio.sleep(0.1)

    # Verify fade was cancelled
    assert entity_id not in ACTIVE_FADES

    # Clean up
    fade_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await fade_task


async def test_manual_turn_off_preserves_orig_brightness(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test that turning off during fade preserves pre-fade original brightness."""
    entity_id = "light.test_off_preserves"
    initial_brightness = 200
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: initial_brightness,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    # Start a long fade to lower brightness
    fade_task = hass.async_create_task(
        hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_BRIGHTNESS_PCT: 10,  # Fade to 10%
                ATTR_TRANSITION: 5,
            },
            target={"entity_id": entity_id},
            blocking=True,
        )
    )

    # Wait for the fade to start and make some progress
    await asyncio.sleep(0.3)

    # Verify fade is active and orig brightness was stored
    assert entity_id in ACTIVE_FADES
    storage_data = hass.data[DOMAIN]["data"]
    assert entity_id in storage_data
    assert storage_data[entity_id] == initial_brightness

    # Simulate turning off the light manually (interrupting fade)
    hass.states.async_set(
        entity_id,
        STATE_OFF,
        {
            ATTR_BRIGHTNESS: None,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )
    await hass.async_block_till_done()
    await asyncio.sleep(0.1)

    # Verify fade was cancelled
    assert entity_id not in ACTIVE_FADES

    # Original brightness should still be the pre-fade value
    assert storage_data[entity_id] == initial_brightness

    # Clean up
    fade_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await fade_task


async def test_new_fade_cancels_previous(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test that starting a new fade cancels an in-progress fade."""
    entity_id = "light.test_new_fade"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 200,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    # Start first fade (to 20%)
    first_fade = hass.async_create_task(
        hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_BRIGHTNESS_PCT: 20,
                ATTR_TRANSITION: 5,
            },
            target={"entity_id": entity_id},
            blocking=True,
        )
    )

    # Wait for the fade to start
    await asyncio.sleep(0.2)

    # Verify first fade is active
    assert entity_id in ACTIVE_FADES

    # Start a second fade (to 80%)
    second_fade = hass.async_create_task(
        hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_BRIGHTNESS_PCT: 80,
                ATTR_TRANSITION: 0.5,
            },
            target={"entity_id": entity_id},
            blocking=True,
        )
    )

    # Wait for the second fade to complete
    await second_fade
    await hass.async_block_till_done()

    # Get the final brightness from turn_on calls
    turn_on_calls = [c for c in service_calls if c.service == "turn_on"]
    assert len(turn_on_calls) > 0
    final_brightness = turn_on_calls[-1].data.get(ATTR_BRIGHTNESS)

    # The final brightness should be 80% (204), not 20% (51)
    assert final_brightness == 204

    # Clean up
    first_fade.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await first_fade


async def test_manual_change_during_fade_updates_orig(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test that manual brightness change during fade becomes new original."""
    entity_id = "light.test_update_orig"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 200,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    # Start a fade
    fade_task = hass.async_create_task(
        hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_BRIGHTNESS_PCT: 20,
                ATTR_TRANSITION: 5,
            },
            target={"entity_id": entity_id},
            blocking=True,
        )
    )

    # Wait for the fade to start
    await asyncio.sleep(0.2)

    # Verify orig brightness was stored at fade start
    storage_data = hass.data[DOMAIN]["data"]
    assert entity_id in storage_data
    assert storage_data[entity_id] == 200

    # Simulate a manual brightness change to 150 (interrupting the fade)
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 150,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )
    await hass.async_block_till_done()

    # Give a moment for the state change handler to process
    await asyncio.sleep(0.1)

    # Original brightness should now be 150 (the manual change), not 200
    # This is the user's new intended brightness level
    assert storage_data[entity_id] == 150

    # Clean up
    fade_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await fade_task


async def test_manual_change_without_fade_stores_new_orig(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test that manual brightness change when NOT fading stores as new original."""
    entity_id = "light.test_store_new_orig"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 200,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )
    await hass.async_block_till_done()

    # Simulate a manual brightness change to 150 (no active fade)
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 150,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )
    await hass.async_block_till_done()

    # Check that the new brightness was stored as original (since no fade was active)
    storage_data = hass.data[DOMAIN]["data"]
    assert entity_id in storage_data
    assert storage_data[entity_id] == 150


async def test_group_changes_ignored(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test that state changes for group entities are ignored.

    When a group light (which has entity_id attribute pointing to member lights)
    has a state change, the integration should ignore it and not cancel any fades.
    """
    # Create a regular light
    regular_light = "light.regular_light"
    hass.states.async_set(
        regular_light,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 200,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    # Create a group light (has entity_id attribute) - this makes it a group helper
    group_light = "light.group_light"
    hass.states.async_set(
        group_light,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 200,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
            ATTR_ENTITY_ID: [regular_light],  # This makes it a group
        },
    )

    # Simulate a state change on the group light and verify it doesn't
    # trigger any storage updates or actions that would affect member lights
    # (Groups are detected by having entity_id attribute and are ignored)

    # First, set up storage data for the regular light (flat map: entity_id -> brightness)
    hass.data[DOMAIN]["data"][regular_light] = 200
    original_value = hass.data[DOMAIN]["data"][regular_light]

    # Now simulate a brightness change on the group light
    hass.states.async_set(
        group_light,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 100,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
            ATTR_ENTITY_ID: [regular_light],
        },
    )
    await hass.async_block_till_done()

    # The group's state change should not have stored any brightness for the group
    assert group_light not in hass.data[DOMAIN]["data"]

    # The regular light's stored brightness should be unchanged
    assert hass.data[DOMAIN]["data"][regular_light] == original_value


async def test_brightness_tolerance_allows_rounding(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test that +-3 tolerance is allowed for device rounding.

    When a fade is active and the light reports a brightness that's within
    3 of the expected value (due to device rounding), the fade should continue.
    """
    entity_id = "light.test_tolerance"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 200,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    # Simulate that we're expecting brightness 100 (list of (ExpectedValues, timestamp) tuples)
    FADE_EXPECTED_STATE[entity_id] = ExpectedState(
        entity_id=entity_id,
        values=[(ExpectedValues(brightness=100), time.monotonic())],
    )

    # Create a simple mock task that we can track
    fake_task = asyncio.get_event_loop().create_future()
    ACTIVE_FADES[entity_id] = fake_task  # type: ignore[assignment]
    cancel_event = asyncio.Event()
    FADE_CANCEL_EVENTS[entity_id] = cancel_event

    try:
        # Simulate a state update with brightness within tolerance
        # (expected is 100, so 97-103 should be within tolerance of 3)
        hass.states.async_set(
            entity_id,
            STATE_ON,
            {
                ATTR_BRIGHTNESS: 102,  # Within tolerance (100 + 2)
                ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
            },
        )
        await hass.async_block_till_done()

        # Fade should still be active - the cancel event should not be set
        assert not cancel_event.is_set(), "Cancel should not be set for within-tolerance"

        # Re-add expected brightness since the previous match removed it from tracking
        FADE_EXPECTED_STATE[entity_id] = ExpectedState(
            entity_id=entity_id,
            values=[(ExpectedValues(brightness=100), time.monotonic())],
        )

        # Now test with brightness AT the tolerance boundary
        hass.states.async_set(
            entity_id,
            STATE_ON,
            {
                ATTR_BRIGHTNESS: 103,  # At boundary (100 + 3)
                ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
            },
        )
        await hass.async_block_till_done()

        # Should still not be cancelled (3 is within tolerance)
        assert not cancel_event.is_set(), "Cancel event should not be set at tolerance boundary"

    finally:
        # Clean up
        FADE_EXPECTED_STATE.pop(entity_id, None)
        ACTIVE_FADES.pop(entity_id, None)
        FADE_CANCEL_EVENTS.pop(entity_id, None)
        if not fake_task.done():
            fake_task.cancel()


async def test_brightness_outside_tolerance_cancels_fade(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test that brightness change outside tolerance cancels the fade.

    When a state change has brightness outside the +-3 tolerance, the change
    should be treated as manual intervention and cancel the fade.
    """
    entity_id = "light.test_outside_tolerance"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 200,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    # Simulate that we're expecting brightness 100 (list of (ExpectedValues, timestamp) tuples)
    FADE_EXPECTED_STATE[entity_id] = ExpectedState(
        entity_id=entity_id,
        values=[(ExpectedValues(brightness=100), time.monotonic())],
    )

    # Use an event to control the fake fade task
    stop_fake_fade = asyncio.Event()

    async def fake_fade() -> None:
        await stop_fake_fade.wait()

    fake_task = hass.async_create_task(fake_fade())
    ACTIVE_FADES[entity_id] = fake_task
    cancel_event = asyncio.Event()
    FADE_CANCEL_EVENTS[entity_id] = cancel_event

    try:
        # Simulate a state update with brightness way outside tolerance
        hass.states.async_set(
            entity_id,
            STATE_ON,
            {
                ATTR_BRIGHTNESS: 150,  # Way outside tolerance (100 + 50)
                ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
            },
        )
        await hass.async_block_till_done()
        await asyncio.sleep(0.1)  # Allow event handler to complete

        # The fade should be cancelled because brightness is outside tolerance
        assert cancel_event.is_set(), "Cancel event should be set for out-of-tolerance change"

    finally:
        # Clean up - signal the fake task to stop
        stop_fake_fade.set()
        FADE_EXPECTED_STATE.pop(entity_id, None)
        ACTIVE_FADES.pop(entity_id, None)
        FADE_CANCEL_EVENTS.pop(entity_id, None)
        with contextlib.suppress(asyncio.CancelledError):
            await fake_task


async def test_expected_brightness_changes_ignored(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test that changes matching expected brightness don't cancel the fade.

    When a state change has brightness matching what we expected (within tolerance),
    it's from our own fade operation and should be ignored.
    """
    entity_id = "light.test_expected"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 200,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    # Simulate that we're expecting brightness 100 (list of (ExpectedValues, timestamp) tuples)
    FADE_EXPECTED_STATE[entity_id] = ExpectedState(
        entity_id=entity_id,
        values=[(ExpectedValues(brightness=100), time.monotonic())],
    )

    # Create a simple mock task that we can track
    fake_task = asyncio.get_event_loop().create_future()
    ACTIVE_FADES[entity_id] = fake_task  # type: ignore[assignment]
    cancel_event = asyncio.Event()
    FADE_CANCEL_EVENTS[entity_id] = cancel_event

    try:
        # Simulate a state update with exact expected brightness
        hass.states.async_set(
            entity_id,
            STATE_ON,
            {
                ATTR_BRIGHTNESS: 100,  # Exact expected value
                ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
            },
        )
        await hass.async_block_till_done()

        # Fade should still be active - change matches expected
        assert not cancel_event.is_set(), "Cancel event should not be set for expected brightness"

        # Re-add expected brightness since the previous match removed it from tracking
        FADE_EXPECTED_STATE[entity_id] = ExpectedState(
            entity_id=entity_id,
            values=[(ExpectedValues(brightness=100), time.monotonic())],
        )

        # Also verify with brightness slightly different but within tolerance
        hass.states.async_set(
            entity_id,
            STATE_ON,
            {
                ATTR_BRIGHTNESS: 98,  # Within tolerance (100 - 2)
                ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
            },
        )
        await hass.async_block_till_done()

        # Should still not be cancelled
        assert not cancel_event.is_set(), "Cancel should not be set for within-tolerance"

    finally:
        # Clean up
        FADE_EXPECTED_STATE.pop(entity_id, None)
        ACTIVE_FADES.pop(entity_id, None)
        FADE_CANCEL_EVENTS.pop(entity_id, None)
        if not fake_task.done():
            fake_task.cancel()


async def test_restore_intended_state_turn_off_when_current_is_on(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test _restore_intended_state turns light off when intended is OFF but current is ON.

    This tests the branch at lines 734-742 where intended=0 and current!=0.
    The scenario: user turns light OFF, but a late fade event turns it back ON,
    so we need to restore the intended OFF state.
    """
    from unittest.mock import MagicMock

    from custom_components.fade_lights import INTENDED_STATE_QUEUE, _restore_intended_state

    entity_id = "light.test_restore_off"

    # Set up the light as currently ON (simulating late fade event turned it back on)
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 150,  # Light is ON after late fade event
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    # Store original brightness
    hass.data[DOMAIN]["data"][entity_id] = 200

    # Create mock old_state (was ON) and new_state (user turned OFF)
    old_state = MagicMock()
    old_state.state = STATE_ON
    old_state.attributes = {ATTR_BRIGHTNESS: 150}

    new_state = MagicMock()
    new_state.state = STATE_OFF
    new_state.attributes = {ATTR_BRIGHTNESS: None}

    # Set up intended state queue: [old_state, intended_state]
    INTENDED_STATE_QUEUE[entity_id] = [old_state, new_state]

    try:
        # Call _restore_intended_state directly
        # intended will be 0 (OFF), current will be 150 (ON) - should trigger turn_off
        await _restore_intended_state(hass, entity_id)
        await hass.async_block_till_done()
        await asyncio.sleep(0.2)
        await hass.async_block_till_done()

        # Check that turn_off was called to restore the intended OFF state
        turn_off_calls = [c for c in service_calls if c.service == "turn_off"]
        assert len(turn_off_calls) >= 1, "Light should be turned off to match intended state"
    finally:
        INTENDED_STATE_QUEUE.pop(entity_id, None)


async def test_restore_intended_state_turn_on_when_brightness_differs(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test _restore_intended_state restores brightness when intended differs from current.

    This tests the branch at lines 744-751 where intended>0 and current!=intended.
    """
    entity_id = "light.test_restore_brightness"
    initial_brightness = 200
    intended_brightness = 150

    # Set up the light ON at initial brightness
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: initial_brightness,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    # Store original brightness
    hass.data[DOMAIN]["data"][entity_id] = initial_brightness

    # Simulate that we're expecting brightness 100 (mid-fade) but user set 150
    FADE_EXPECTED_STATE[entity_id] = ExpectedState(
        entity_id=entity_id,
        values=[(ExpectedValues(brightness=100), time.monotonic())],
    )

    # Use an event to control the fake fade task
    stop_fake_fade = asyncio.Event()

    async def fake_fade() -> None:
        await stop_fake_fade.wait()

    fake_task = hass.async_create_task(fake_fade())
    ACTIVE_FADES[entity_id] = fake_task
    cancel_event = asyncio.Event()
    FADE_CANCEL_EVENTS[entity_id] = cancel_event

    try:
        # Simulate manual brightness change during fade (user sets brightness to 150)
        hass.states.async_set(
            entity_id,
            STATE_ON,
            {
                ATTR_BRIGHTNESS: intended_brightness,
                ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
            },
        )
        await hass.async_block_till_done()

        # Allow time for manual intervention detection and _restore_intended_state
        await asyncio.sleep(0.2)

        # Allow fake fade to complete so cleanup happens
        stop_fake_fade.set()
        await asyncio.sleep(0.3)
        await hass.async_block_till_done()

        # The original brightness should be updated to the user's intended brightness
        assert hass.data[DOMAIN]["data"][entity_id] == intended_brightness

    finally:
        # Clean up
        FADE_EXPECTED_STATE.pop(entity_id, None)
        ACTIVE_FADES.pop(entity_id, None)
        FADE_CANCEL_EVENTS.pop(entity_id, None)
        if not fake_task.done():
            fake_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await fake_task


async def test_restore_intended_state_off_to_on_uses_original_brightness(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test _restore_intended_state uses original brightness for OFF->ON transition.

    This tests the _get_intended_brightness OFF->ON path at lines 680-683.
    When user turns light ON from OFF during a fade, we restore to original brightness.
    """
    entity_id = "light.test_off_to_on"
    original_brightness = 200

    # Set up the light in OFF state (as if fade was dimming to off)
    hass.states.async_set(
        entity_id,
        STATE_OFF,
        {
            ATTR_BRIGHTNESS: None,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    # Store original brightness (from before the fade started)
    hass.data[DOMAIN]["data"][entity_id] = original_brightness

    # Simulate that we're expecting brightness 50 (near end of fade to 0%)
    FADE_EXPECTED_STATE[entity_id] = ExpectedState(
        entity_id=entity_id,
        values=[(ExpectedValues(brightness=50), time.monotonic())],
    )

    # Use an event to control the fake fade task
    stop_fake_fade = asyncio.Event()

    async def fake_fade() -> None:
        await stop_fake_fade.wait()

    fake_task = hass.async_create_task(fake_fade())
    ACTIVE_FADES[entity_id] = fake_task
    cancel_event = asyncio.Event()
    FADE_CANCEL_EVENTS[entity_id] = cancel_event

    try:
        # Simulate user turning the light ON from OFF during fade
        # The ON event doesn't have a brightness, so we should restore original
        hass.states.async_set(
            entity_id,
            STATE_ON,
            {
                ATTR_BRIGHTNESS: 255,  # Default ON brightness from switch
                ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
            },
        )
        await hass.async_block_till_done()

        # Allow time for manual intervention detection
        await asyncio.sleep(0.2)

        # Allow fake fade to complete
        stop_fake_fade.set()
        await asyncio.sleep(0.3)
        await hass.async_block_till_done()

        # The original brightness should be preserved
        assert hass.data[DOMAIN]["data"][entity_id] == original_brightness

    finally:
        # Clean up
        FADE_EXPECTED_STATE.pop(entity_id, None)
        ACTIVE_FADES.pop(entity_id, None)
        FADE_CANCEL_EVENTS.pop(entity_id, None)
        if not fake_task.done():
            fake_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await fake_task


async def test_cancel_and_wait_for_fade_task_already_done(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test _cancel_and_wait_for_fade handles task that is already done.

    This tests line 628 where task.done() is True.
    """
    from custom_components.fade_lights import _cancel_and_wait_for_fade

    entity_id = "light.test_already_done"

    # Create a task that is already completed
    async def completed_task() -> None:
        pass

    task = hass.async_create_task(completed_task())
    await task  # Wait for it to complete

    ACTIVE_FADES[entity_id] = task
    cancel_event = asyncio.Event()
    FADE_CANCEL_EVENTS[entity_id] = cancel_event

    try:
        # Call _cancel_and_wait_for_fade with already-done task
        await _cancel_and_wait_for_fade(entity_id)

        # Should complete without error
        # The task was already done, so it should just log and return

    finally:
        # Clean up
        ACTIVE_FADES.pop(entity_id, None)
        FADE_CANCEL_EVENTS.pop(entity_id, None)


async def test_state_change_with_none_new_state_ignored(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test that state change events with None new_state are ignored.

    This tests line 166 where new_state is None.
    """
    from homeassistant.const import EVENT_STATE_CHANGED

    # Fire a state_changed event with no new_state
    # This simulates an entity being removed
    hass.bus.async_fire(
        EVENT_STATE_CHANGED,
        {
            "entity_id": "light.removed_light",
            "old_state": None,
            "new_state": None,
        },
    )
    await hass.async_block_till_done()

    # Should complete without error - the handler returns early for None new_state


async def test_get_intended_brightness_returns_none_when_integration_unloaded(
    hass: HomeAssistant,
) -> None:
    """Test _get_intended_brightness returns None when integration is unloaded.

    This tests line 673 where DOMAIN not in hass.data.
    """
    from unittest.mock import MagicMock

    from custom_components.fade_lights import _get_intended_brightness

    # Ensure DOMAIN is not in hass.data (simulating unloaded integration)
    hass.data.pop(DOMAIN, None)

    entity_id = "light.test_entity"
    old_state = MagicMock()
    old_state.state = STATE_ON
    new_state = MagicMock()
    new_state.state = STATE_ON
    new_state.attributes = {ATTR_BRIGHTNESS: 150}

    # Should return None when integration is unloaded
    result = _get_intended_brightness(hass, entity_id, old_state, new_state)
    assert result is None


async def test_cancel_and_wait_for_fade_timeout(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test _cancel_and_wait_for_fade handles timeout waiting for cleanup.

    This tests lines 768-769 where TimeoutError is caught.
    """
    from unittest.mock import patch

    from custom_components.fade_lights import (
        FADE_COMPLETE_CONDITIONS,
        _cancel_and_wait_for_fade,
    )

    entity_id = "light.test_timeout"

    # Create a task that will never complete naturally
    task_completed = asyncio.Event()

    async def hanging_task() -> None:
        await task_completed.wait()

    task = hass.async_create_task(hanging_task())

    ACTIVE_FADES[entity_id] = task
    cancel_event = asyncio.Event()
    FADE_CANCEL_EVENTS[entity_id] = cancel_event

    # Create condition but make wait_for always time out
    condition = asyncio.Condition()
    FADE_COMPLETE_CONDITIONS[entity_id] = condition

    try:
        # Patch the FADE_CANCEL_TIMEOUT_S to be very short for the test
        with patch("custom_components.fade_lights.FADE_CANCEL_TIMEOUT_S", 0.05):
            # Call should timeout but not raise
            await _cancel_and_wait_for_fade(entity_id)

        # Should complete without raising - timeout is caught and logged

    finally:
        # Clean up
        task_completed.set()
        ACTIVE_FADES.pop(entity_id, None)
        FADE_CANCEL_EVENTS.pop(entity_id, None)
        FADE_COMPLETE_CONDITIONS.pop(entity_id, None)
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_restore_intended_state_when_domain_not_in_hass(
    hass: HomeAssistant,
) -> None:
    """Test _restore_intended_state returns early when DOMAIN not in hass.data.

    This tests line 823 where DOMAIN not in hass.data.
    """
    from unittest.mock import MagicMock

    from custom_components.fade_lights import INTENDED_STATE_QUEUE, _restore_intended_state

    # Ensure DOMAIN is not in hass.data
    hass.data.pop(DOMAIN, None)

    entity_id = "light.test_entity"
    old_state = MagicMock()
    new_state = MagicMock()

    # Set up intended state queue
    INTENDED_STATE_QUEUE[entity_id] = [old_state, new_state]

    try:
        # Should return early without raising
        await _restore_intended_state(hass, entity_id)
    finally:
        INTENDED_STATE_QUEUE.pop(entity_id, None)


async def test_restore_intended_state_when_intended_is_none(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test _restore_intended_state returns early when intended brightness is None.

    This tests line 830 where intended is None.
    """
    from unittest.mock import MagicMock, patch

    from custom_components.fade_lights import INTENDED_STATE_QUEUE, _restore_intended_state

    entity_id = "light.test_entity"

    # Set up the light
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 150,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    old_state = MagicMock()
    old_state.state = STATE_ON
    new_state = MagicMock()
    new_state.state = STATE_ON
    new_state.attributes = {ATTR_BRIGHTNESS: 150}

    # Set up intended state queue
    INTENDED_STATE_QUEUE[entity_id] = [old_state, new_state]

    try:
        # Make _get_intended_brightness return None
        with patch("custom_components.fade_lights._get_intended_brightness", return_value=None):
            # Should return early without raising
            await _restore_intended_state(hass, entity_id)
    finally:
        INTENDED_STATE_QUEUE.pop(entity_id, None)


async def test_restore_intended_state_when_entity_removed(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test _restore_intended_state returns early when entity is removed.

    This tests lines 840-841 where current_state is None.
    """
    from unittest.mock import MagicMock

    from custom_components.fade_lights import INTENDED_STATE_QUEUE, _restore_intended_state

    entity_id = "light.test_removed"

    # Set up then remove the entity
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 150,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )
    hass.states.async_remove(entity_id)
    await hass.async_block_till_done()

    # Store original brightness
    hass.data[DOMAIN]["data"][entity_id] = 200

    old_state = MagicMock()
    old_state.state = STATE_ON
    old_state.attributes = {ATTR_BRIGHTNESS: 150}

    new_state = MagicMock()
    new_state.state = STATE_ON
    new_state.attributes = {ATTR_BRIGHTNESS: 150}

    # Set up intended state queue
    INTENDED_STATE_QUEUE[entity_id] = [old_state, new_state]

    try:
        # Should return early when current_state is None (entity was removed)
        await _restore_intended_state(hass, entity_id)

        # No service calls should have been made
        turn_on_calls = [c for c in service_calls if c.service == "turn_on"]
        turn_off_calls = [c for c in service_calls if c.service == "turn_off"]
        assert len(turn_on_calls) == 0
        assert len(turn_off_calls) == 0
    finally:
        INTENDED_STATE_QUEUE.pop(entity_id, None)


async def test_second_manual_event_during_restore_appends_to_queue(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test that a second manual event during restore wait appends to the queue.

    This tests the queue-based approach where we also check RESTORE_TASKS in the
    event handler to append to INTENDED_STATE_QUEUE even after the fade is cancelled.

    When a restore task is running (entity_id in RESTORE_TASKS), any new state
    change should append to INTENDED_STATE_QUEUE so the restore uses the latest
    user intent rather than a stale first event.
    """
    from unittest.mock import MagicMock

    entity_id = "light.test_second_manual"

    # Set up the light as OFF (simulating after first manual OFF event)
    hass.states.async_set(
        entity_id,
        STATE_OFF,
        {
            ATTR_BRIGHTNESS: None,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )
    await hass.async_block_till_done()

    # Store original brightness
    hass.data[DOMAIN]["data"][entity_id] = 200

    # Create a mock task to simulate a running restore task
    # Using MagicMock avoids async timing issues
    mock_task = MagicMock()

    # Manually set up RESTORE_TASKS to simulate a running restore
    RESTORE_TASKS[entity_id] = mock_task

    # Set an initial intended state queue (simulating the first OFF event)
    initial_off_state = hass.states.get(entity_id)
    # Queue format: [old_state, intended_state]
    # Simulating after first event was processed - queue has just the OFF state
    INTENDED_STATE_QUEUE[entity_id] = [initial_off_state]

    try:
        # Verify initial queue
        assert len(INTENDED_STATE_QUEUE[entity_id]) == 1
        assert INTENDED_STATE_QUEUE[entity_id][0].state == STATE_OFF

        # Second manual event - user turns light ON while restore task is "running"
        # This should append to INTENDED_STATE_QUEUE because entity_id in RESTORE_TASKS
        hass.states.async_set(
            entity_id,
            STATE_ON,
            {
                ATTR_BRIGHTNESS: 180,
                ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
            },
        )
        await hass.async_block_till_done()

        # The key assertion: INTENDED_STATE_QUEUE should have the second event appended
        assert entity_id in INTENDED_STATE_QUEUE
        queue = INTENDED_STATE_QUEUE[entity_id]
        assert len(queue) == 2, "Queue should have 2 entries after second event"
        assert queue[-1].state == STATE_ON, "Latest state should be ON (second event)"
        assert queue[-1].attributes.get(ATTR_BRIGHTNESS) == 180, (
            "Latest brightness should be 180 from second event"
        )

    finally:
        # Clean up
        RESTORE_TASKS.pop(entity_id, None)
        INTENDED_STATE_QUEUE.pop(entity_id, None)
