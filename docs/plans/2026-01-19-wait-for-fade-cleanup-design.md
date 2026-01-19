# Wait-for-fade-cleanup approach

## Problem

The current implementation uses `USER_INTENDED_BRIGHTNESS` with a 500ms timeout to catch late-arriving fade calls. This is fragile - the timeout is a guess, and the logic is complex.

## Solution

Instead of storing intended state and catching late calls reactively, we:
1. Cancel the fade task
2. Wait for it to actually be removed from `ACTIVE_FADES`
3. Then apply the correct state directly

Since cancelling a task interrupts `asyncio.sleep` immediately, the cleanup happens fast.

## Key insight

When we wait for the fade to fully clean up, there are no more in-flight fade calls. We can directly apply the correct state without needing to catch late arrivals.

## Data structures

### Keep
```python
ACTIVE_FADES: dict[str, asyncio.Task]        # entity_id -> fade task
FADE_EXPECTED_BRIGHTNESS: dict[str, int]      # entity_id -> what fade expects
FADE_CANCEL_EVENTS: dict[str, asyncio.Event]  # entity_id -> cancel signal
```

### Remove
```python
USER_INTENDED_BRIGHTNESS: dict[str, int]      # No longer needed
```

## Helper function

```python
async def _cancel_and_wait_for_fade(entity_id: str) -> None:
    """Cancel any active fade for entity and wait for cleanup."""
    while entity_id in ACTIVE_FADES:
        task = ACTIVE_FADES[entity_id]
        if not task.cancelled():
            task.cancel()
        await asyncio.sleep(0.01)  # Yield to let cleanup happen
```

## State change handler

The callback detects manual changes and spawns an async task to handle them:

```python
@callback
def handle_light_state_change(event: Event) -> None:
    # ... early returns for non-lights, groups, etc ...

    if is_fading and entity_id in FADE_EXPECTED_BRIGHTNESS:
        # Check if state matches expected (our fade, not manual)
        if _state_matches_expected(new_state, expected, tolerance=3):
            return  # Our fade step, ignore

        # Manual change detected - spawn async handler
        hass.async_create_task(
            _handle_state_change(hass, entity_id, old_state, new_state)
        )
        return

    # No active fade - handle normally (spawn task for async work)
    hass.async_create_task(
        _handle_state_change(hass, entity_id, old_state, new_state)
    )
```

## Async state change handler

```python
async def _handle_state_change(
    hass: HomeAssistant,
    entity_id: str,
    old_state: State | None,
    new_state: State,
) -> None:
    """Handle state change after fade cleanup."""
    # Cancel and wait for any active fade
    await _cancel_and_wait_for_fade(entity_id)

    # Determine intended state based on transition
    new_brightness = new_state.attributes.get(ATTR_BRIGHTNESS)

    if new_state.state == STATE_OFF:
        # User turned off - intended state is OFF
        intended_on = False
        intended_brightness = None
    elif old_state and old_state.state == STATE_OFF and new_state.state == STATE_ON:
        # OFF -> ON: restore to original brightness
        orig = _get_orig_brightness(hass, entity_id)
        intended_on = True
        intended_brightness = orig if orig > 0 else new_brightness
    else:
        # ON -> ON or other: use current brightness
        intended_on = True
        intended_brightness = new_brightness

    # Store as original if we have a value
    if intended_brightness:
        _store_orig_brightness(hass, entity_id, intended_brightness)

    # Apply intended state
    current_state = hass.states.get(entity_id)
    if not current_state:
        return

    current_on = current_state.state == STATE_ON
    current_brightness = current_state.attributes.get(ATTR_BRIGHTNESS)

    if intended_on:
        # We want the light ON
        if not current_on or (intended_brightness and current_brightness != intended_brightness):
            await hass.services.async_call(
                LIGHT_DOMAIN, SERVICE_TURN_ON,
                {ATTR_ENTITY_ID: entity_id, ATTR_BRIGHTNESS: intended_brightness}
            )
    else:
        # We want the light OFF
        if current_on:
            await hass.services.async_call(
                LIGHT_DOMAIN, SERVICE_TURN_OFF,
                {ATTR_ENTITY_ID: entity_id}
            )
```

## Edge cases

- **Tolerance**: Allow ±3 brightness difference for device rounding
- **Non-dimmable lights**: Skip brightness restoration (only on/off)
- **No stored original**: Use the new brightness as-is
- **Multiple rapid changes**: Each spawns its own handler, but cancel_and_wait ensures serialization

## Why this is simpler

1. **No timeout guessing**: We wait for actual cleanup, not a fixed 500ms
2. **No intended state tracking**: We apply the correct state directly after cleanup
3. **Single code path**: Both fade interruption and normal changes go through `_handle_state_change`
4. **Deterministic**: No race conditions from late-arriving calls

## Cleanup on unload

```python
async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    for event in FADE_CANCEL_EVENTS.values():
        event.set()
    for task in ACTIVE_FADES.values():
        task.cancel()
    ACTIVE_FADES.clear()
    FADE_CANCEL_EVENTS.clear()
    FADE_EXPECTED_BRIGHTNESS.clear()
    # USER_INTENDED_BRIGHTNESS removed - no longer needed
    ...
```
