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
    STATE_OFF,
    STATE_ON,
)
from homeassistant.core import Event, EventStateChangedData, HomeAssistant, callback

from .const import AUTOCONFIGURE_ITERATIONS, AUTOCONFIGURE_TIMEOUT_S
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
    # Capture original state
    original_state = hass.states.get(entity_id)
    if original_state is None:
        return {"entity_id": entity_id, "error": "Entity not found"}

    original_on = original_state.state == STATE_ON
    original_brightness = original_state.attributes.get(ATTR_BRIGHTNESS)

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
        for i in range(1, AUTOCONFIGURE_ITERATIONS + 1):
            # Clear the event before the test
            state_changed_event.clear()

            # Record start time
            start_time = time.monotonic()

            # Alternate brightness: 1 for odd iterations, 255 for even
            target_brightness = 1 if i % 2 == 1 else 255

            # Call light.turn_on with the target brightness
            await hass.services.async_call(
                LIGHT_DOMAIN,
                SERVICE_TURN_ON,
                {ATTR_ENTITY_ID: entity_id, ATTR_BRIGHTNESS: target_brightness},
                blocking=True,
            )

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
                        "%s: Timeout on iteration %d, retrying", entity_id, i
                    )
                    # Clear and retry
                    state_changed_event.clear()

                    start_time = time.monotonic()
                    await hass.services.async_call(
                        LIGHT_DOMAIN,
                        SERVICE_TURN_ON,
                        {ATTR_ENTITY_ID: entity_id, ATTR_BRIGHTNESS: target_brightness},
                        blocking=True,
                    )

                    try:
                        await asyncio.wait_for(
                            state_changed_event.wait(),
                            timeout=AUTOCONFIGURE_TIMEOUT_S,
                        )
                    except TimeoutError:
                        # Second timeout - return error
                        return {"entity_id": entity_id, "error": "Timeout after retry"}
                else:
                    # Already retried once, return error
                    return {"entity_id": entity_id, "error": "Timeout after retry"}

            # Record elapsed time in milliseconds
            elapsed_ms = (time.monotonic() - start_time) * 1000
            timings.append(elapsed_ms)

    finally:
        # Remove state listener
        unsub()

    # Calculate average
    if not timings:
        return {"entity_id": entity_id, "error": "No timing data collected"}

    average = sum(timings) / len(timings)

    # Round up to nearest 10ms
    result = math.ceil(average / 10) * 10

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

    # Save min_delay_ms to storage
    await async_save_light_config(hass, entity_id, min_delay_ms=result)

    _LOGGER.info("%s: Measured min_delay_ms=%d (average=%.1f)", entity_id, result, average)

    return {"entity_id": entity_id, "min_delay_ms": result}
