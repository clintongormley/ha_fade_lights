"""Tests for manual interruption detection during fades."""

from __future__ import annotations

import asyncio
import contextlib

import pytest
from homeassistant.components.light import (
    ATTR_BRIGHTNESS,
    ATTR_SUPPORTED_COLOR_MODES,
)
from homeassistant.components.light.const import ColorMode
from homeassistant.const import ATTR_ENTITY_ID, STATE_OFF, STATE_ON
from homeassistant.core import Context, HomeAssistant, ServiceCall
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.fade_lights import (
    ACTIVE_CONTEXTS,
    ACTIVE_FADES,
    FADE_CANCEL_EVENTS,
    FADE_EXPECTED_BRIGHTNESS,
)
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
                ATTR_ENTITY_ID: entity_id,
                ATTR_BRIGHTNESS_PCT: 20,  # Fade down to 51 brightness
                ATTR_TRANSITION: 5,  # Long transition so we can interrupt
            },
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
                ATTR_ENTITY_ID: entity_id,
                ATTR_BRIGHTNESS_PCT: 50,
                ATTR_TRANSITION: 5,
            },
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
                ATTR_ENTITY_ID: entity_id,
                ATTR_BRIGHTNESS_PCT: 10,  # Fade to 10%
                ATTR_TRANSITION: 5,
            },
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
                ATTR_ENTITY_ID: entity_id,
                ATTR_BRIGHTNESS_PCT: 20,
                ATTR_TRANSITION: 5,
            },
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
                ATTR_ENTITY_ID: entity_id,
                ATTR_BRIGHTNESS_PCT: 80,
                ATTR_TRANSITION: 0.5,
            },
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
                ATTR_ENTITY_ID: entity_id,
                ATTR_BRIGHTNESS_PCT: 20,
                ATTR_TRANSITION: 5,
            },
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
    """Test that +-5 tolerance is allowed for device rounding.

    When a fade is active and the light reports a brightness that's within
    5 of the expected value (due to device rounding), the fade should continue.
    This test verifies the tolerance mechanism by checking that state updates
    with our context and within-tolerance brightness don't cancel the fade.
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

    # Create a context and add it to ACTIVE_CONTEXTS to simulate our fade context
    our_context = Context()
    ACTIVE_CONTEXTS.add(our_context.id)

    # Simulate that we're expecting brightness 100
    FADE_EXPECTED_BRIGHTNESS[entity_id] = 100

    # Create a simple mock task that we can track (not registered with hass)
    fake_task = asyncio.get_event_loop().create_future()
    ACTIVE_FADES[entity_id] = fake_task  # type: ignore[assignment]
    cancel_event = asyncio.Event()
    FADE_CANCEL_EVENTS[entity_id] = cancel_event

    try:
        # Simulate a state update from our context with brightness within tolerance
        # (expected is 100, so 97-103 should be within tolerance of 3)
        hass.states.async_set(
            entity_id,
            STATE_ON,
            {
                ATTR_BRIGHTNESS: 102,  # Within tolerance (100 + 2)
                ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
            },
            context=our_context,
        )
        await hass.async_block_till_done()

        # Fade should still be active - the cancel event should not be set
        assert not cancel_event.is_set(), "Cancel should not be set for within-tolerance"

        # Now test with brightness AT the tolerance boundary
        hass.states.async_set(
            entity_id,
            STATE_ON,
            {
                ATTR_BRIGHTNESS: 103,  # At boundary (100 + 3)
                ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
            },
            context=our_context,
        )
        await hass.async_block_till_done()

        # Should still not be cancelled (3 is within tolerance)
        assert not cancel_event.is_set(), "Cancel event should not be set at tolerance boundary"

    finally:
        # Clean up
        ACTIVE_CONTEXTS.discard(our_context.id)
        FADE_EXPECTED_BRIGHTNESS.pop(entity_id, None)
        ACTIVE_FADES.pop(entity_id, None)
        FADE_CANCEL_EVENTS.pop(entity_id, None)
        # Complete the future if not already done
        if not fake_task.done():
            fake_task.cancel()


async def test_inherited_context_detected_by_brightness(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test that manual change with inherited context is detected via brightness mismatch.

    When a state change has our context (due to context inheritance) but the brightness
    is wildly different from what we expected (outside tolerance), the change should
    be treated as manual and cancel the fade.
    """
    entity_id = "light.test_inherited_context"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 200,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    # Create a context and add it to ACTIVE_CONTEXTS to simulate our fade context
    our_context = Context()
    ACTIVE_CONTEXTS.add(our_context.id)

    # Simulate that we're expecting brightness 100
    FADE_EXPECTED_BRIGHTNESS[entity_id] = 100

    # Use an event to control the fake fade task
    stop_fake_fade = asyncio.Event()

    async def fake_fade() -> None:
        await stop_fake_fade.wait()

    fake_task = hass.async_create_task(fake_fade())
    ACTIVE_FADES[entity_id] = fake_task
    cancel_event = asyncio.Event()
    FADE_CANCEL_EVENTS[entity_id] = cancel_event

    try:
        # Simulate a state update from "our" context but with brightness
        # way outside tolerance - this indicates manual intervention that
        # inherited our context
        hass.states.async_set(
            entity_id,
            STATE_ON,
            {
                ATTR_BRIGHTNESS: 150,  # Way outside tolerance (100 + 50)
                ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
            },
            context=our_context,
        )
        await hass.async_block_till_done()
        await asyncio.sleep(0.1)  # Allow event handler to complete

        # Despite having our context, the fade should be cancelled because
        # brightness is outside tolerance
        assert cancel_event.is_set(), "Cancel event should be set for out-of-tolerance change"

    finally:
        # Clean up - signal the fake task to stop
        stop_fake_fade.set()
        ACTIVE_CONTEXTS.discard(our_context.id)
        FADE_EXPECTED_BRIGHTNESS.pop(entity_id, None)
        ACTIVE_FADES.pop(entity_id, None)
        FADE_CANCEL_EVENTS.pop(entity_id, None)
        # Wait for the fake task to complete (may already be cancelled)
        with contextlib.suppress(asyncio.CancelledError):
            await fake_task


async def test_our_context_changes_ignored(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test that changes from our own context (within tolerance) don't cancel the fade.

    When a state change comes from our context AND the brightness matches what we
    expected (within tolerance), the change should be ignored and not cancel the fade.
    """
    entity_id = "light.test_our_context"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 200,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )

    # Create a context and add it to ACTIVE_CONTEXTS to simulate our fade context
    our_context = Context()
    ACTIVE_CONTEXTS.add(our_context.id)

    # Simulate that we're expecting brightness 100
    FADE_EXPECTED_BRIGHTNESS[entity_id] = 100

    # Create a simple mock task that we can track (not registered with hass)
    fake_task = asyncio.get_event_loop().create_future()
    ACTIVE_FADES[entity_id] = fake_task  # type: ignore[assignment]
    cancel_event = asyncio.Event()
    FADE_CANCEL_EVENTS[entity_id] = cancel_event

    try:
        # Simulate a state update from our context with exact expected brightness
        hass.states.async_set(
            entity_id,
            STATE_ON,
            {
                ATTR_BRIGHTNESS: 100,  # Exact expected value
                ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
            },
            context=our_context,
        )
        await hass.async_block_till_done()

        # Fade should still be active - change was from our context and within tolerance
        assert not cancel_event.is_set(), "Cancel event should not be set for our own context"

        # Also verify with brightness slightly different but within tolerance
        hass.states.async_set(
            entity_id,
            STATE_ON,
            {
                ATTR_BRIGHTNESS: 98,  # Within tolerance (100 - 2)
                ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
            },
            context=our_context,
        )
        await hass.async_block_till_done()

        # Should still not be cancelled
        assert not cancel_event.is_set(), "Cancel should not be set for within-tolerance"

    finally:
        # Clean up
        ACTIVE_CONTEXTS.discard(our_context.id)
        FADE_EXPECTED_BRIGHTNESS.pop(entity_id, None)
        ACTIVE_FADES.pop(entity_id, None)
        FADE_CANCEL_EVENTS.pop(entity_id, None)
        # Complete the future if not already done
        if not fake_task.done():
            fake_task.cancel()
