"""Autoconfigure feature for measuring optimal light delay."""

from __future__ import annotations

import asyncio
import contextlib
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
    NATIVE_TRANSITION_MS,
)
from .websocket_api import async_save_light_config

_LOGGER = logging.getLogger(__name__)


async def async_autoconfigure_light(
    hass: HomeAssistant, entity_id: str
) -> dict[str, Any]:
    """Run full autoconfiguration for a light.

    This is the main entry point that:
    1. Captures original state
    2. Runs native transitions test
    3. Runs delay test (with transitions if supported)
    4. Restores original state
    5. Saves results to storage

    Args:
        hass: Home Assistant instance
        entity_id: The light entity ID to test

    Returns:
        {
            "entity_id": entity_id,
            "min_delay_ms": int (or None if error),
            "native_transitions": bool (or None if error),
            "error": str (only if both tests failed),
        }
    """
    # Capture original state
    original_state = hass.states.get(entity_id)
    if original_state is None:
        return {"entity_id": entity_id, "error": "Entity not found"}

    original_on = original_state.state == STATE_ON
    original_brightness = original_state.attributes.get(ATTR_BRIGHTNESS)

    result: dict[str, Any] = {"entity_id": entity_id}

    try:
        # Run native transitions test first
        transition_result = await _async_test_native_transitions(hass, entity_id)
        if "error" not in transition_result:
            result["native_transitions"] = transition_result["supports_native_transitions"]

        # Run delay test with native transitions enabled if supported
        use_transitions = result.get("native_transitions", False)
        delay_result = await _async_test_light_delay(
            hass, entity_id, use_native_transitions=use_transitions
        )
        if "error" in delay_result:
            result["error"] = delay_result["error"]
        else:
            result["min_delay_ms"] = delay_result["min_delay_ms"]

        # Save results to storage
        if "min_delay_ms" in result:
            await async_save_light_config(
                hass, entity_id, min_delay_ms=result["min_delay_ms"]
            )
        if "native_transitions" in result:
            await async_save_light_config(
                hass, entity_id, native_transitions=result["native_transitions"]
            )

    finally:
        # Always restore original state
        await _async_restore_light_state(hass, entity_id, original_on, original_brightness)

    return result


async def _async_restore_light_state(
    hass: HomeAssistant,
    entity_id: str,
    original_on: bool,
    original_brightness: int | None,
) -> None:
    """Restore a light to its original state."""
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


async def _async_set_standard_state(hass: HomeAssistant, entity_id: str) -> None:
    """Set light to standard test state (on at brightness 255).

    Waits for state to settle before returning.
    """
    state_changed_event = asyncio.Event()

    @callback
    def _on_state_changed(event: Event[EventStateChangedData]) -> None:
        """Handle state changed events for the test light."""
        new_state = event.data.get("new_state")
        if new_state and new_state.entity_id == entity_id:
            state_changed_event.set()

    unsub = hass.bus.async_listen("state_changed", _on_state_changed)

    try:
        await hass.services.async_call(
            LIGHT_DOMAIN,
            SERVICE_TURN_ON,
            {ATTR_ENTITY_ID: entity_id, ATTR_BRIGHTNESS: 255},
            blocking=True,
        )
        # Wait for state change or short timeout (light may already be at 255)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(state_changed_event.wait(), timeout=2.0)
        # Let state settle
        await asyncio.sleep(0.5)
    finally:
        unsub()


async def _async_test_light_delay(
    hass: HomeAssistant, entity_id: str, use_native_transitions: bool = False
) -> dict[str, Any]:
    """Test a light to determine optimal minimum delay between commands.

    Internal function - does not capture/restore state or save to storage.
    Use async_autoconfigure_light for the full workflow.

    Args:
        hass: Home Assistant instance
        entity_id: The light entity ID to test
        use_native_transitions: If True, include transition parameter in commands

    Returns:
        On success: {"entity_id": entity_id, "min_delay_ms": result}
        On failure: {"entity_id": entity_id, "error": "..."}
    """
    await _async_set_standard_state(hass, entity_id)

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
        # Build service data with optional transition
        service_data_base: dict[str, Any] = {ATTR_ENTITY_ID: entity_id}
        if use_native_transitions:
            service_data_base["transition"] = NATIVE_TRANSITION_MS / 1000

        for i in range(1, AUTOCONFIGURE_ITERATIONS + 1):
            # Alternate brightness: 10 for odd iterations, 255 for even
            target_brightness = 10 if i % 2 == 1 else 255
            start_time = time.monotonic()

            # Try up to 2 times (initial + one retry)
            for attempt in range(2):
                state_changed_event.clear()
                start_time = time.monotonic()

                await hass.services.async_call(
                    LIGHT_DOMAIN,
                    SERVICE_TURN_ON,
                    {**service_data_base, ATTR_BRIGHTNESS: target_brightness},
                    blocking=True,
                )

                try:
                    await asyncio.wait_for(
                        state_changed_event.wait(),
                        timeout=AUTOCONFIGURE_TIMEOUT_S,
                    )
                    break  # Success, exit retry loop
                except TimeoutError:
                    if attempt == 0 and retry_count == 0:
                        retry_count += 1
                        _LOGGER.warning(
                            "%s: Timeout on iteration %d, retrying",
                            entity_id,
                            i,
                        )
                        continue  # Retry once
                    return {"entity_id": entity_id, "error": "Timeout after retry"}

            elapsed_ms = (time.monotonic() - start_time) * 1000
            timings.append(elapsed_ms)

    finally:
        unsub()

    if not timings:
        return {"entity_id": entity_id, "error": "No timing data collected"}

    sorted_timings = sorted(timings)
    p90_index = int(math.ceil(0.9 * len(sorted_timings))) - 1
    p90_value = sorted_timings[p90_index]

    # Round up to nearest 10ms
    result = math.ceil(p90_value / 10) * 10

    # Enforce global minimum delay
    global_min = hass.data.get(DOMAIN, {}).get("min_step_delay_ms", DEFAULT_MIN_STEP_DELAY_MS)
    if result < global_min:
        result = global_min

    _LOGGER.info("%s: Measured min_delay_ms=%d (p90=%.1f)", entity_id, result, p90_value)

    return {"entity_id": entity_id, "min_delay_ms": result}


async def _async_test_native_transitions(
    hass: HomeAssistant, entity_id: str, transition_s: float = 2.0
) -> dict[str, Any]:
    """Test if a light supports native transitions.

    Internal function - does not capture/restore state.
    Use async_autoconfigure_light for the full workflow.

    Args:
        hass: Home Assistant instance
        entity_id: The light entity ID to test
        transition_s: Transition time in seconds to test with

    Returns:
        On success: {"entity_id": entity_id, "supports_native_transitions": bool, ...}
        On failure: {"entity_id": entity_id, "error": "..."}
    """
    await _async_set_standard_state(hass, entity_id)

    # Target brightness 10 (light starts at 255 from standard state)
    target_brightness = 10

    state_changed_event = asyncio.Event()

    @callback
    def _on_state_changed(event: Event[EventStateChangedData]) -> None:
        """Handle state changed events for the test light."""
        new_state = event.data.get("new_state")
        if new_state and new_state.entity_id == entity_id:
            state_changed_event.set()

    unsub = hass.bus.async_listen("state_changed", _on_state_changed)

    try:
        # Send command with transition (light already at 255)
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
        supports_native = response_time_ms > (transition_time_ms * 0.8)

        _LOGGER.info(
            "%s: Native transitions=%s (response %.0fms, transition %.0fms)",
            entity_id,
            supports_native,
            response_time_ms,
            transition_time_ms,
        )

        return {
            "entity_id": entity_id,
            "supports_native_transitions": supports_native,
            "response_time_ms": round(response_time_ms, 1),
            "transition_time_ms": transition_time_ms,
        }

    finally:
        unsub()
