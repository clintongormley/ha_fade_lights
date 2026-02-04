"""Tests for the autoconfigure feature."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components.light import ATTR_BRIGHTNESS, ATTR_SUPPORTED_COLOR_MODES
from homeassistant.components.light.const import ColorMode
from homeassistant.const import ATTR_ENTITY_ID, STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant, ServiceCall

from custom_components.fade_lights.autoconfigure import async_test_light_delay
from custom_components.fade_lights.const import DOMAIN


@pytest.fixture
def hass_with_storage(hass: HomeAssistant) -> HomeAssistant:
    """Set up hass with storage data."""
    hass.data[DOMAIN] = {
        "data": {},
        "store": MagicMock(),
    }
    hass.data[DOMAIN]["store"].async_save = AsyncMock()
    return hass


@pytest.fixture
def mock_light_on(hass: HomeAssistant) -> str:
    """Create a mock dimmable light entity that is on at brightness 200."""
    entity_id = "light.test_autoconfigure"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 200,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )
    return entity_id


@pytest.fixture
def mock_light_off(hass: HomeAssistant) -> str:
    """Create a mock dimmable light entity that is off."""
    entity_id = "light.test_autoconfigure_off"
    hass.states.async_set(
        entity_id,
        STATE_OFF,
        {
            ATTR_BRIGHTNESS: None,
            ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
        },
    )
    return entity_id


@pytest.fixture
def service_calls_with_state_update(hass: HomeAssistant) -> list[ServiceCall]:
    """Capture light service calls and fire state_changed events.

    This fixture:
    - Registers light.turn_on and light.turn_off services
    - Captures all service calls to a list for assertion
    - Updates the mock light state and fires state_changed events
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
                    # This fires state_changed event automatically
                    hass.states.async_set(eid, STATE_ON, current_attrs)

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
                    # This fires state_changed event automatically
                    hass.states.async_set(eid, STATE_OFF, current_attrs)

    hass.services.async_register("light", "turn_on", mock_turn_on)
    hass.services.async_register("light", "turn_off", mock_turn_off)

    return calls


class TestLightDelay:
    """Tests for test_light_delay function."""

    async def test_normal_measurement_flow(
        self,
        hass_with_storage: HomeAssistant,
        mock_light_on: str,
        service_calls_with_state_update: list[ServiceCall],
    ) -> None:
        """Test normal measurement flow completes successfully."""
        result = await async_test_light_delay(hass_with_storage, mock_light_on)

        assert "error" not in result
        assert result["entity_id"] == mock_light_on
        assert "min_delay_ms" in result
        # Result should be a positive integer, rounded up to nearest 10ms
        assert isinstance(result["min_delay_ms"], int)
        assert result["min_delay_ms"] >= 0
        assert result["min_delay_ms"] % 10 == 0

    async def test_correct_number_of_iterations(
        self,
        hass_with_storage: HomeAssistant,
        mock_light_on: str,
        service_calls_with_state_update: list[ServiceCall],
    ) -> None:
        """Test that correct number of iterations are performed."""
        from custom_components.fade_lights.const import AUTOCONFIGURE_ITERATIONS

        await async_test_light_delay(hass_with_storage, mock_light_on)

        # Should have 1 init call + 10 iteration calls + 1 restore call = 12 calls
        turn_on_calls = [c for c in service_calls_with_state_update if c.service == "turn_on"]
        # 1 init + 10 measurement calls + 1 restore call
        assert len(turn_on_calls) == AUTOCONFIGURE_ITERATIONS + 2

    async def test_alternating_brightness(
        self,
        hass_with_storage: HomeAssistant,
        mock_light_on: str,
        service_calls_with_state_update: list[ServiceCall],
    ) -> None:
        """Test brightness alternates between 10 and 255."""
        from custom_components.fade_lights.const import AUTOCONFIGURE_ITERATIONS

        await async_test_light_delay(hass_with_storage, mock_light_on)

        turn_on_calls = [c for c in service_calls_with_state_update if c.service == "turn_on"]

        # First call is initialization to 255
        assert turn_on_calls[0].data.get(ATTR_BRIGHTNESS) == 255

        # Check the measurement calls (index 1 to ITERATIONS+1, excluding init and restore)
        for i in range(1, AUTOCONFIGURE_ITERATIONS + 1):
            call = turn_on_calls[i]
            # Odd iterations (1, 3, 5...) set brightness to 10, even (2, 4, 6...) to 255
            expected_brightness = 10 if i % 2 == 1 else 255
            assert call.data.get(ATTR_BRIGHTNESS) == expected_brightness

    async def test_restores_original_state_when_on(
        self,
        hass_with_storage: HomeAssistant,
        mock_light_on: str,
        service_calls_with_state_update: list[ServiceCall],
    ) -> None:
        """Test original state is restored after measurement (light was on)."""
        await async_test_light_delay(hass_with_storage, mock_light_on)

        # Last call should be restore call with original brightness
        turn_on_calls = [c for c in service_calls_with_state_update if c.service == "turn_on"]
        restore_call = turn_on_calls[-1]
        assert restore_call.data.get(ATTR_ENTITY_ID) == mock_light_on
        # Original brightness was 200
        assert restore_call.data.get(ATTR_BRIGHTNESS) == 200

    async def test_restores_original_state_when_off(
        self,
        hass_with_storage: HomeAssistant,
        mock_light_off: str,
        service_calls_with_state_update: list[ServiceCall],
    ) -> None:
        """Test original state is restored after measurement (light was off)."""
        await async_test_light_delay(hass_with_storage, mock_light_off)

        # Last call should be turn_off to restore original state
        turn_off_calls = [c for c in service_calls_with_state_update if c.service == "turn_off"]
        assert len(turn_off_calls) == 1
        restore_call = turn_off_calls[-1]
        assert restore_call.data.get(ATTR_ENTITY_ID) == mock_light_off

    async def test_saves_min_delay_to_storage(
        self,
        hass_with_storage: HomeAssistant,
        mock_light_on: str,
        service_calls_with_state_update: list[ServiceCall],
    ) -> None:
        """Test min_delay_ms is saved to storage."""
        result = await async_test_light_delay(hass_with_storage, mock_light_on)

        # Check that storage was updated
        assert mock_light_on in hass_with_storage.data[DOMAIN]["data"]
        stored_config = hass_with_storage.data[DOMAIN]["data"][mock_light_on]
        assert "min_delay_ms" in stored_config
        assert stored_config["min_delay_ms"] == result["min_delay_ms"]

        # Verify store.async_save was called
        hass_with_storage.data[DOMAIN]["store"].async_save.assert_called()

    async def test_round_up_to_nearest_10ms(
        self,
        hass_with_storage: HomeAssistant,
        mock_light_on: str,
        service_calls_with_state_update: list[ServiceCall],
    ) -> None:
        """Test result is rounded up to nearest 10ms."""
        result = await async_test_light_delay(hass_with_storage, mock_light_on)

        # Result should be divisible by 10
        assert result["min_delay_ms"] % 10 == 0

    async def test_entity_not_found(
        self,
        hass_with_storage: HomeAssistant,
    ) -> None:
        """Test error returned when entity not found."""
        result = await async_test_light_delay(hass_with_storage, "light.nonexistent")

        assert "error" in result
        assert result["entity_id"] == "light.nonexistent"
        assert result["error"] == "Entity not found"


class TestLightDelayTimeout:
    """Tests for timeout and retry behavior."""

    async def test_timeout_with_successful_retry(
        self,
        hass_with_storage: HomeAssistant,
        mock_light_on: str,
    ) -> None:
        """Test timeout on first attempt but success on retry."""
        call_count = 0
        calls: list[ServiceCall] = []

        async def mock_turn_on_with_delay(call: ServiceCall) -> None:
            """Mock turn_on that fails first call but succeeds on retry."""
            nonlocal call_count
            calls.append(call)
            call_count += 1

            entity_id = call.data.get(ATTR_ENTITY_ID)
            # First call: don't update state (simulates timeout)
            # Retry call: update state normally
            if entity_id and call_count != 1:
                current_state = hass_with_storage.states.get(entity_id)
                if current_state:
                    current_attrs = dict(current_state.attributes)
                    if ATTR_BRIGHTNESS in call.data:
                        current_attrs[ATTR_BRIGHTNESS] = call.data[ATTR_BRIGHTNESS]
                    hass_with_storage.states.async_set(entity_id, STATE_ON, current_attrs)

        async def mock_turn_off(call: ServiceCall) -> None:
            """Mock turn_off."""
            calls.append(call)

        hass_with_storage.services.async_register("light", "turn_on", mock_turn_on_with_delay)
        hass_with_storage.services.async_register("light", "turn_off", mock_turn_off)

        # Use short timeout for test
        with patch(
            "custom_components.fade_lights.autoconfigure.AUTOCONFIGURE_TIMEOUT_S",
            0.1,
        ):
            result = await async_test_light_delay(hass_with_storage, mock_light_on)

        # Should succeed after retry (first call times out, retry succeeds)
        assert "error" not in result
        assert "min_delay_ms" in result

    async def test_timeout_after_retry_returns_error(
        self,
        hass_with_storage: HomeAssistant,
        mock_light_on: str,
    ) -> None:
        """Test timeout after retry returns error."""
        calls: list[ServiceCall] = []

        async def mock_turn_on_never_updates(call: ServiceCall) -> None:
            """Mock turn_on that never updates state (always times out)."""
            calls.append(call)
            # Don't update state - this simulates device not responding

        hass_with_storage.services.async_register("light", "turn_on", mock_turn_on_never_updates)

        # Use very short timeout for test
        with patch(
            "custom_components.fade_lights.autoconfigure.AUTOCONFIGURE_TIMEOUT_S",
            0.05,
        ):
            result = await async_test_light_delay(hass_with_storage, mock_light_on)

        assert "error" in result
        assert result["entity_id"] == mock_light_on
        assert result["error"] == "Timeout after retry"


class TestLightDelayCalculation:
    """Tests for timing calculation."""

    async def test_timing_calculation_accuracy(
        self,
        hass_with_storage: HomeAssistant,
        mock_light_on: str,
    ) -> None:
        """Test that timing calculation works with known delays."""
        calls: list[ServiceCall] = []
        delay_ms = 25  # Known delay

        async def mock_turn_on_with_known_delay(call: ServiceCall) -> None:
            """Mock turn_on with known delay before state update."""
            calls.append(call)

            entity_id = call.data.get(ATTR_ENTITY_ID)
            if entity_id:
                # Simulate realistic device response time
                await asyncio.sleep(delay_ms / 1000)

                current_state = hass_with_storage.states.get(entity_id)
                if current_state:
                    current_attrs = dict(current_state.attributes)
                    if ATTR_BRIGHTNESS in call.data:
                        current_attrs[ATTR_BRIGHTNESS] = call.data[ATTR_BRIGHTNESS]
                    hass_with_storage.states.async_set(entity_id, STATE_ON, current_attrs)

        async def mock_turn_off(call: ServiceCall) -> None:
            """Mock turn_off."""
            calls.append(call)

        hass_with_storage.services.async_register("light", "turn_on", mock_turn_on_with_known_delay)
        hass_with_storage.services.async_register("light", "turn_off", mock_turn_off)

        result = await async_test_light_delay(hass_with_storage, mock_light_on)

        assert "error" not in result
        # With 25ms delay, the p90 would be around 25-35ms, which rounds up to 30-40ms
        # However, the global minimum is 100ms (DEFAULT_MIN_STEP_DELAY_MS)
        # So the result should be clamped to 100ms
        from custom_components.fade_lights.const import DEFAULT_MIN_STEP_DELAY_MS

        assert result["min_delay_ms"] == DEFAULT_MIN_STEP_DELAY_MS

    async def test_ceil_rounding(
        self,
        hass_with_storage: HomeAssistant,
    ) -> None:
        """Test math.ceil rounding behavior."""
        import math

        # Test various averages and their expected ceiling to nearest 10ms
        test_cases = [
            (5, 10),  # 5ms -> 10ms
            (10, 10),  # 10ms -> 10ms (exact)
            (11, 20),  # 11ms -> 20ms
            (25, 30),  # 25ms -> 30ms
            (100, 100),  # 100ms -> 100ms (exact)
            (101, 110),  # 101ms -> 110ms
        ]

        for average, expected in test_cases:
            result = math.ceil(average / 10) * 10
            assert result == expected, f"Expected {expected} for average {average}, got {result}"


class TestWsAutoconfigure:
    """Tests for ws_autoconfigure WebSocket handler."""

    @pytest.fixture
    def mock_multiple_lights(self, hass: HomeAssistant) -> list[str]:
        """Create multiple mock lights for testing."""
        entity_ids = []
        for i in range(10):
            entity_id = f"light.test_ws_{i}"
            hass.states.async_set(
                entity_id,
                STATE_ON,
                {
                    ATTR_BRIGHTNESS: 200,
                    ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
                },
            )
            entity_ids.append(entity_id)
        return entity_ids

    @pytest.fixture
    def mock_light_group_for_ws(self, hass: HomeAssistant, mock_multiple_lights: list[str]) -> str:
        """Create a mock light group containing lights 0-2."""
        entity_id = "light.ws_group"
        hass.states.async_set(
            entity_id,
            STATE_ON,
            {
                ATTR_BRIGHTNESS: 150,
                ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
                ATTR_ENTITY_ID: mock_multiple_lights[:3],
            },
        )
        return entity_id

    async def test_autoconfigure_respects_parallel_limit(
        self,
        hass: HomeAssistant,
        hass_ws_client,
        init_integration,
        mock_multiple_lights: list[str],
    ) -> None:
        """Verify semaphore limits concurrency to AUTOCONFIGURE_MAX_PARALLEL (5)."""
        from custom_components.fade_lights.const import AUTOCONFIGURE_MAX_PARALLEL

        # Track maximum concurrent tests
        current_concurrent = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        async def mock_autoconfigure_light(hass: HomeAssistant, entity_id: str) -> dict:
            """Mock that tracks concurrent execution."""
            nonlocal current_concurrent, max_concurrent
            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)

            # Simulate some work to allow other tasks to run
            await asyncio.sleep(0.05)

            async with lock:
                current_concurrent -= 1

            return {"entity_id": entity_id, "min_delay_ms": 100, "native_transitions": True}

        with patch(
            "custom_components.fade_lights.autoconfigure.async_autoconfigure_light",
            side_effect=mock_autoconfigure_light,
        ):
            client = await hass_ws_client(hass)

            # Test all 10 lights
            await client.send_json(
                {
                    "id": 1,
                    "type": "fade_lights/autoconfigure",
                    "entity_ids": mock_multiple_lights,
                }
            )

            # Collect all events until we get the final result
            events = []
            while True:
                msg = await client.receive_json()
                if msg["type"] == "result":
                    break
                events.append(msg)

        # Verify the parallel limit was respected
        assert max_concurrent <= AUTOCONFIGURE_MAX_PARALLEL
        # With 10 lights and limit of 5, we should have hit at least some concurrency
        assert max_concurrent >= 1

    async def test_autoconfigure_streams_results(
        self,
        hass: HomeAssistant,
        hass_ws_client,
        init_integration,
        service_calls_with_state_update: list[ServiceCall],
    ) -> None:
        """Verify events sent as lights complete (started, result, error)."""
        # Create test lights
        entity_id_success = "light.stream_test_success"
        entity_id_error = "light.stream_test_error"

        hass.states.async_set(
            entity_id_success,
            STATE_ON,
            {
                ATTR_BRIGHTNESS: 200,
                ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
            },
        )
        hass.states.async_set(
            entity_id_error,
            STATE_ON,
            {
                ATTR_BRIGHTNESS: 200,
                ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
            },
        )

        call_count = 0

        async def mock_autoconfigure_light(hass: HomeAssistant, entity_id: str) -> dict:
            """Mock that returns success or error based on entity."""
            nonlocal call_count
            call_count += 1

            if entity_id == entity_id_success:
                return {"entity_id": entity_id, "min_delay_ms": 100, "native_transitions": True}
            else:
                return {"entity_id": entity_id, "error": "Test error message"}

        with patch(
            "custom_components.fade_lights.autoconfigure.async_autoconfigure_light",
            side_effect=mock_autoconfigure_light,
        ):
            client = await hass_ws_client(hass)

            await client.send_json(
                {
                    "id": 1,
                    "type": "fade_lights/autoconfigure",
                    "entity_ids": [entity_id_success, entity_id_error],
                }
            )

            # Collect events
            events = []
            while True:
                msg = await client.receive_json()
                if msg["type"] == "result":
                    break
                events.append(msg)

        # Verify we got both started and result/error events for each light
        event_types = {}
        for event in events:
            assert event["type"] == "event"
            entity_id = event["event"]["entity_id"]
            event_type = event["event"]["type"]
            if entity_id not in event_types:
                event_types[entity_id] = []
            event_types[entity_id].append(event_type)

        # Check success light
        assert "started" in event_types[entity_id_success]
        assert "result" in event_types[entity_id_success]

        # Check error light
        assert "started" in event_types[entity_id_error]
        assert "error" in event_types[entity_id_error]

        # Verify result event has min_delay_ms
        result_event = next(
            e
            for e in events
            if e["event"]["entity_id"] == entity_id_success and e["event"]["type"] == "result"
        )
        assert result_event["event"]["min_delay_ms"] == 100

        # Verify error event has message
        error_event = next(
            e
            for e in events
            if e["event"]["entity_id"] == entity_id_error and e["event"]["type"] == "error"
        )
        assert error_event["event"]["message"] == "Test error message"

    async def test_autoconfigure_expands_groups(
        self,
        hass: HomeAssistant,
        hass_ws_client,
        init_integration,
        mock_multiple_lights: list[str],
        mock_light_group_for_ws: str,
    ) -> None:
        """Verify light groups are expanded to individual lights."""
        tested_entities: list[str] = []

        async def mock_autoconfigure_light(hass: HomeAssistant, entity_id: str) -> dict:
            """Mock that records tested entity IDs."""
            tested_entities.append(entity_id)
            return {"entity_id": entity_id, "min_delay_ms": 100, "native_transitions": True}

        with patch(
            "custom_components.fade_lights.autoconfigure.async_autoconfigure_light",
            side_effect=mock_autoconfigure_light,
        ):
            client = await hass_ws_client(hass)

            # Send only the group
            await client.send_json(
                {
                    "id": 1,
                    "type": "fade_lights/autoconfigure",
                    "entity_ids": [mock_light_group_for_ws],
                }
            )

            # Wait for completion
            while True:
                msg = await client.receive_json()
                if msg["type"] == "result":
                    break

        # Verify the group was expanded to individual lights
        # The group contains mock_multiple_lights[:3]
        expected_lights = set(mock_multiple_lights[:3])
        actual_lights = set(tested_entities)

        assert actual_lights == expected_lights
        # Group itself should NOT be tested
        assert mock_light_group_for_ws not in tested_entities

    async def test_autoconfigure_filters_excluded(
        self,
        hass: HomeAssistant,
        hass_ws_client,
        init_integration,
    ) -> None:
        """Verify excluded lights are filtered out."""
        # Create lights
        included_light = "light.included"
        excluded_light = "light.excluded"

        hass.states.async_set(
            included_light,
            STATE_ON,
            {
                ATTR_BRIGHTNESS: 200,
                ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
            },
        )
        hass.states.async_set(
            excluded_light,
            STATE_ON,
            {
                ATTR_BRIGHTNESS: 200,
                ATTR_SUPPORTED_COLOR_MODES: [ColorMode.BRIGHTNESS],
            },
        )

        # Mark one light as excluded in storage
        hass.data[DOMAIN]["data"][excluded_light] = {"exclude": True}

        tested_entities: list[str] = []

        async def mock_autoconfigure_light(hass: HomeAssistant, entity_id: str) -> dict:
            """Mock that records tested entity IDs."""
            tested_entities.append(entity_id)
            return {"entity_id": entity_id, "min_delay_ms": 100, "native_transitions": True}

        with patch(
            "custom_components.fade_lights.autoconfigure.async_autoconfigure_light",
            side_effect=mock_autoconfigure_light,
        ):
            client = await hass_ws_client(hass)

            await client.send_json(
                {
                    "id": 1,
                    "type": "fade_lights/autoconfigure",
                    "entity_ids": [included_light, excluded_light],
                }
            )

            # Wait for completion
            while True:
                msg = await client.receive_json()
                if msg["type"] == "result":
                    break

        # Only included light should be tested
        assert included_light in tested_entities
        assert excluded_light not in tested_entities

    async def test_autoconfigure_handles_cancellation(
        self,
        hass: HomeAssistant,
        hass_ws_client,
        init_integration,
        mock_multiple_lights: list[str],
    ) -> None:
        """Verify subscription can be cancelled."""
        started_count = 0
        completed_count = 0
        cancel_after = 2  # Cancel after 2 lights start

        async def mock_autoconfigure_light(hass: HomeAssistant, entity_id: str) -> dict:
            """Mock that simulates work."""
            nonlocal completed_count
            # Longer delay to allow time for cancellation
            await asyncio.sleep(0.5)
            completed_count += 1
            return {"entity_id": entity_id, "min_delay_ms": 100, "native_transitions": True}

        with patch(
            "custom_components.fade_lights.autoconfigure.async_autoconfigure_light",
            side_effect=mock_autoconfigure_light,
        ):
            client = await hass_ws_client(hass)

            # Start autoconfigure with all 10 lights
            await client.send_json(
                {
                    "id": 1,
                    "type": "fade_lights/autoconfigure",
                    "entity_ids": mock_multiple_lights,
                }
            )

            # Wait for some started events, then cancel
            while started_count < cancel_after:
                msg = await asyncio.wait_for(client.receive_json(), timeout=5.0)
                if msg["type"] == "event" and msg["event"]["type"] == "started":
                    started_count += 1

            # Send unsubscribe command
            await client.send_json(
                {
                    "id": 2,
                    "type": "unsubscribe_events",
                    "subscription": 1,
                }
            )

            # Receive unsubscribe confirmation - may receive more events first
            unsubscribe_confirmed = False
            while not unsubscribe_confirmed:
                msg = await asyncio.wait_for(client.receive_json(), timeout=5.0)
                if msg.get("id") == 2:
                    assert msg["success"] is True
                    unsubscribe_confirmed = True
                # Ignore any remaining events from subscription 1

            # Give some time for potential late completions
            await asyncio.sleep(0.2)

        # Due to cancellation, not all lights should have completed
        # The semaphore limit is 5, and we cancel after 2 start
        # Some lights may have completed depending on timing
        assert completed_count < len(mock_multiple_lights)
