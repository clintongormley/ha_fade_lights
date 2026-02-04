"""Autoconfigure feature for measuring optimal light delay."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any

from homeassistant.components.light import ATTR_BRIGHTNESS
from homeassistant.components.light.const import DOMAIN as LIGHT_DOMAIN
from homeassistant.const import (
    ATTR_ENTITY_ID,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_ON,
)
from homeassistant.core import Event, EventStateChangedData, HomeAssistant, callback

from .const import (
    AUTOCONFIGURE_ITERATIONS,
    AUTOCONFIGURE_TIMEOUT_S,
    DEFAULT_MIN_STEP_DELAY_MS,
    DOMAIN,
)
from .websocket_api import async_save_light_config

_LOGGER = logging.getLogger(__name__)


async def async_test_light_delay(hass: HomeAssistant, entity_id: str) -> dict[str, Any]:
    """Test a light to determine optimal minimum delay between commands.

    This function measures the response time of a light by toggling its
    brightness between 1 and 255 multiple times and recording how long
    it takes for the state change event to arrive.

    Args:
        hass: Home Assistant instance
        entity_id: The light entity ID to test

    Returns:
        On success: {"entity_id": entity_id, "min_delay_ms": result}
        On failure: {"entity_id": entity_id, "error": "Timeout after retry"}
    """
    _LOGGER.debug("%s: Starting autoconfigure test", entity_id)

    # Capture original state
    original_state = hass.states.get(entity_id)
    if original_state is None:
        _LOGGER.debug("%s: Entity not found", entity_id)
        return {"entity_id": entity_id, "error": "Entity not found"}

    original_on = original_state.state == STATE_ON
    original_brightness = original_state.attributes.get(ATTR_BRIGHTNESS)
    _LOGGER.debug(
        "%s: Original state: on=%s, brightness=%s",
        entity_id,
        original_on,
        original_brightness,
    )

    timings: list[float] = []
    state_changed_event = asyncio.Event()
    retry_count = 0

    @callback
    def _on_state_changed(event: Event[EventStateChangedData]) -> None:
        """Handle state changed events for the test light."""
        new_state = event.data.get("new_state")
        if new_state and new_state.entity_id == entity_id:
            state_changed_event.set()

    # Set up state change listener
    unsub = hass.bus.async_listen("state_changed", _on_state_changed)

    try:
        # Initialize light to a known state (on at brightness 255)
        # This ensures the first test iteration (brightness 1) will be a change
        _LOGGER.debug("%s: Initializing light to brightness 255", entity_id)
        state_changed_event.clear()
        await hass.services.async_call(
            LIGHT_DOMAIN,
            SERVICE_TURN_ON,
            {ATTR_ENTITY_ID: entity_id, ATTR_BRIGHTNESS: 255},
            blocking=True,
        )
        # Wait for state change or short timeout (light may already be at 255)
        try:
            await asyncio.wait_for(state_changed_event.wait(), timeout=2.0)
            _LOGGER.debug("%s: Light initialized (state change detected)", entity_id)
        except TimeoutError:
            _LOGGER.debug(
                "%s: Light initialized (no state change, may already be at 255)",
                entity_id,
            )

        _LOGGER.debug("%s: Starting %d iterations", entity_id, AUTOCONFIGURE_ITERATIONS)
        for i in range(1, AUTOCONFIGURE_ITERATIONS + 1):
            # Clear the event before the test
            state_changed_event.clear()

            # Record start time
            start_time = time.monotonic()

            # Alternate brightness: 10 for odd iterations, 255 for even
            target_brightness = 10 if i % 2 == 1 else 255

            _LOGGER.debug(
                "%s: Iteration %d/%d - setting brightness to %d",
                entity_id,
                i,
                AUTOCONFIGURE_ITERATIONS,
                target_brightness,
            )

            # Call light.turn_on with the target brightness
            await hass.services.async_call(
                LIGHT_DOMAIN,
                SERVICE_TURN_ON,
                {ATTR_ENTITY_ID: entity_id, ATTR_BRIGHTNESS: target_brightness},
                blocking=True,
            )
            _LOGGER.debug("%s: Service call completed, waiting for state change", entity_id)

            # Wait for state_changed event with timeout
            try:
                await asyncio.wait_for(
                    state_changed_event.wait(),
                    timeout=AUTOCONFIGURE_TIMEOUT_S,
                )
            except TimeoutError:
                # Retry once on timeout
                if retry_count == 0:
                    retry_count += 1
                    _LOGGER.warning(
                        "%s: Timeout on iteration %d after %ds, retrying",
                        entity_id,
                        i,
                        AUTOCONFIGURE_TIMEOUT_S,
                    )
                    # Clear and retry
                    state_changed_event.clear()

                    start_time = time.monotonic()
                    await hass.services.async_call(
                        LIGHT_DOMAIN,
                        SERVICE_TURN_ON,
                        {
                            ATTR_ENTITY_ID: entity_id,
                            ATTR_BRIGHTNESS: target_brightness,
                        },
                        blocking=True,
                    )

                    try:
                        await asyncio.wait_for(
                            state_changed_event.wait(),
                            timeout=AUTOCONFIGURE_TIMEOUT_S,
                        )
                    except TimeoutError:
                        # Second timeout - return error
                        _LOGGER.debug("%s: Second timeout, aborting autoconfigure", entity_id)
                        return {"entity_id": entity_id, "error": "Timeout after retry"}
                else:
                    # Already retried once, return error
                    _LOGGER.debug("%s: Timeout and already retried, aborting", entity_id)
                    return {"entity_id": entity_id, "error": "Timeout after retry"}

            # Record elapsed time in milliseconds
            elapsed_ms = (time.monotonic() - start_time) * 1000
            timings.append(elapsed_ms)
            _LOGGER.debug("%s: Iteration %d completed in %.1fms", entity_id, i, elapsed_ms)

    finally:
        # Remove state listener
        unsub()

    # Calculate 90th percentile (value that covers 90% of responses)
    if not timings:
        _LOGGER.debug("%s: No timing data collected", entity_id)
        return {"entity_id": entity_id, "error": "No timing data collected"}

    _LOGGER.debug(
        "%s: All timings (ms): %s",
        entity_id,
        [f"{t:.1f}" for t in timings],
    )

    sorted_timings = sorted(timings)
    min_time = sorted_timings[0]
    max_time = sorted_timings[-1]
    average = sum(timings) / len(timings)

    # Calculate 90th percentile index
    p90_index = int(math.ceil(0.9 * len(sorted_timings))) - 1
    p90_value = sorted_timings[p90_index]

    # Round up to nearest 10ms
    result = math.ceil(p90_value / 10) * 10

    # Enforce global minimum delay
    global_min = hass.data.get(DOMAIN, {}).get("min_step_delay_ms", DEFAULT_MIN_STEP_DELAY_MS)
    if result < global_min:
        _LOGGER.debug(
            "%s: Result %dms is below global minimum %dms, using global minimum",
            entity_id,
            result,
            global_min,
        )
        result = global_min

    _LOGGER.debug(
        "%s: Stats - min=%.1fms, max=%.1fms, avg=%.1fms, p90=%.1fms, result=%dms",
        entity_id,
        min_time,
        max_time,
        average,
        p90_value,
        result,
    )

    # Restore original state
    _LOGGER.debug(
        "%s: Restoring original state (on=%s, brightness=%s)",
        entity_id,
        original_on,
        original_brightness,
    )
    try:
        if original_on:
            if original_brightness is not None:
                await hass.services.async_call(
                    LIGHT_DOMAIN,
                    SERVICE_TURN_ON,
                    {ATTR_ENTITY_ID: entity_id, ATTR_BRIGHTNESS: original_brightness},
                    blocking=True,
                )
            else:
                await hass.services.async_call(
                    LIGHT_DOMAIN,
                    SERVICE_TURN_ON,
                    {ATTR_ENTITY_ID: entity_id},
                    blocking=True,
                )
        else:
            await hass.services.async_call(
                LIGHT_DOMAIN,
                SERVICE_TURN_OFF,
                {ATTR_ENTITY_ID: entity_id},
                blocking=True,
            )
        _LOGGER.debug("%s: State restored successfully", entity_id)
    except Exception:  # noqa: BLE001
        _LOGGER.warning("%s: Failed to restore original state", entity_id)

    # Save min_delay_ms to storage
    _LOGGER.debug("%s: Saving min_delay_ms=%d to storage", entity_id, result)
    await async_save_light_config(hass, entity_id, min_delay_ms=result)

    _LOGGER.info("%s: Measured min_delay_ms=%d (p90=%.1f)", entity_id, result, p90_value)

    return {"entity_id": entity_id, "min_delay_ms": result}


async def async_test_native_transitions(
    hass: HomeAssistant, entity_id: str, transition_s: float = 2.0
) -> dict[str, Any]:
    """Test if a light supports native transitions.

    Sends a brightness command with a transition time and measures how long
    until the state_changed event fires. If the event fires after approximately
    the transition time, the light supports native transitions.

    Args:
        hass: Home Assistant instance
        entity_id: The light entity ID to test
        transition_s: Transition time in seconds to test with

    Returns:
        {
            "entity_id": entity_id,
            "supports_native_transitions": bool,
            "response_time_ms": float,
            "transition_time_ms": float,
        }
    """
    # Capture original state
    original_state = hass.states.get(entity_id)
    if original_state is None:
        return {"entity_id": entity_id, "error": "Entity not found"}

    original_on = original_state.state == STATE_ON
    original_brightness = original_state.attributes.get(ATTR_BRIGHTNESS)
    current_brightness = original_brightness or 0

    # Determine target brightness (opposite end of spectrum)
    target_brightness = 10 if current_brightness > 127 else 255

    state_changed_event = asyncio.Event()

    @callback
    def _on_state_changed(event: Event[EventStateChangedData]) -> None:
        """Handle state changed events for the test light."""
        new_state = event.data.get("new_state")
        if new_state and new_state.entity_id == entity_id:
            state_changed_event.set()

    # Set up listener
    unsub = hass.bus.async_listen("state_changed", _on_state_changed)

    try:
        # First, ensure light is on at a known brightness (without transition)
        await hass.services.async_call(
            LIGHT_DOMAIN,
            SERVICE_TURN_ON,
            {ATTR_ENTITY_ID: entity_id, ATTR_BRIGHTNESS: current_brightness or 128},
            blocking=True,
        )
        await asyncio.sleep(0.5)  # Let state settle

        # Clear event and send command with transition
        state_changed_event.clear()
        start_time = time.monotonic()

        await hass.services.async_call(
            LIGHT_DOMAIN,
            SERVICE_TURN_ON,
            {
                ATTR_ENTITY_ID: entity_id,
                ATTR_BRIGHTNESS: target_brightness,
                "transition": transition_s,
            },
            blocking=True,
        )

        # Wait for state change with generous timeout
        timeout = transition_s + 5.0
        try:
            await asyncio.wait_for(state_changed_event.wait(), timeout=timeout)
        except TimeoutError:
            _LOGGER.warning("%s: No state change received within %.1fs", entity_id, timeout)
            return {"entity_id": entity_id, "error": "Timeout waiting for state change"}

        response_time_ms = (time.monotonic() - start_time) * 1000
        transition_time_ms = transition_s * 1000

        # If response time is within 80% of transition time, consider it native
        # Allow some margin for network latency
        supports_native = response_time_ms > (transition_time_ms * 0.8)

        _LOGGER.info(
            "%s: Native transitions=%s (response %.0fms, transition %.0fms)",
            entity_id,
            supports_native,
            response_time_ms,
            transition_time_ms,
        )

        result = {
            "entity_id": entity_id,
            "supports_native_transitions": supports_native,
            "response_time_ms": round(response_time_ms, 1),
            "transition_time_ms": transition_time_ms,
        }

    finally:
        unsub()

    # Restore original state
    try:
        if original_on:
            if original_brightness is not None:
                await hass.services.async_call(
                    LIGHT_DOMAIN,
                    SERVICE_TURN_ON,
                    {ATTR_ENTITY_ID: entity_id, ATTR_BRIGHTNESS: original_brightness},
                    blocking=True,
                )
            else:
                await hass.services.async_call(
                    LIGHT_DOMAIN,
                    SERVICE_TURN_ON,
                    {ATTR_ENTITY_ID: entity_id},
                    blocking=True,
                )
        else:
            await hass.services.async_call(
                LIGHT_DOMAIN,
                SERVICE_TURN_OFF,
                {ATTR_ENTITY_ID: entity_id},
                blocking=True,
            )
    except Exception:  # noqa: BLE001
        _LOGGER.warning("%s: Failed to restore original state", entity_id)

    return result
