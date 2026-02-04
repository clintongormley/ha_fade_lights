# Manual change detection redesign

## Problem

The current context-based detection is unreliable. Context IDs don't consistently identify whether a state change came from our fade operations or from manual user intervention (physical switches, apps, etc.).

Additionally, late-arriving fade service calls can corrupt user intent. When a user manually changes a light during a fade, the in-flight fade call may complete after the manual change, reverting the user's action.

## Solution

Remove context tracking entirely. Instead:
1. Detect manual changes by comparing actual state to expected state during fades
2. Track user's intended state to catch and restore late fade calls

## Data structures

### Keep (existing)
```python
ACTIVE_FADES: dict[str, asyncio.Task]        # entity_id -> fade task
FADE_EXPECTED_BRIGHTNESS: dict[str, int]      # entity_id -> what fade expects
FADE_CANCEL_EVENTS: dict[str, asyncio.Event]  # entity_id -> cancel signal
```

### Add (new)
```python
USER_INTENDED_BRIGHTNESS: dict[str, int]      # entity_id -> user's intent (0 = off)
```

### Remove
```python
ACTIVE_CONTEXTS: set[str]           # No longer needed
RECENTLY_CANCELLED_FADES: set[str]  # No longer needed
```

## Detection logic

In `handle_light_state_change`:

### Step 1: Check for late fade calls (intended state restoration)

```python
if entity_id in USER_INTENDED_BRIGHTNESS:
    intended = USER_INTENDED_BRIGHTNESS.pop(entity_id)  # Clear immediately

    # Check if actual state differs from intent
    actual_off = new_state.state == STATE_OFF
    intended_off = intended == 0

    if actual_off != intended_off:
        # Late call changed on/off state - restore
        if intended_off:
            await hass.services.async_call("light", "turn_off", {ATTR_ENTITY_ID: entity_id})
        else:
            await hass.services.async_call("light", "turn_on", {ATTR_ENTITY_ID: entity_id, ATTR_BRIGHTNESS: intended})
        return

    if not actual_off and new_brightness != intended:
        # Late call changed brightness - restore
        await hass.services.async_call("light", "turn_on", {ATTR_ENTITY_ID: entity_id, ATTR_BRIGHTNESS: intended})
        return
```

### Step 2: Detect manual intervention during fade

```python
if entity_id in ACTIVE_FADES and entity_id in FADE_EXPECTED_BRIGHTNESS:
    expected = FADE_EXPECTED_BRIGHTNESS[entity_id]
    tolerance = 3

    is_manual = False

    # Unexpected OFF (we expected brightness > 0)
    if new_state.state == STATE_OFF and expected > 0:
        is_manual = True
    # Unexpected ON (we expected off / brightness = 0)
    elif new_state.state == STATE_ON and expected == 0:
        is_manual = True
    # Unexpected brightness change
    elif new_brightness is not None and abs(new_brightness - expected) > tolerance:
        is_manual = True

    if is_manual:
        # Cancel the fade
        if entity_id in FADE_CANCEL_EVENTS:
            FADE_CANCEL_EVENTS[entity_id].set()

        # Store user's intended state
        if new_state.state == STATE_OFF:
            USER_INTENDED_BRIGHTNESS[entity_id] = 0
        else:
            USER_INTENDED_BRIGHTNESS[entity_id] = new_brightness or 255

        # Schedule cleanup timeout (500ms fallback)
        async def cleanup_intended():
            await asyncio.sleep(0.5)
            USER_INTENDED_BRIGHTNESS.pop(entity_id, None)

        hass.async_create_task(cleanup_intended())
        return
```

### Step 3: Normal state handling (no active fade)

```python
# OFF -> ON: Restore to stored original brightness
if old_state.state == STATE_OFF and new_state.state == STATE_ON:
    stored = hass.data[DOMAIN]["data"].get(entity_id, 0)
    if stored and new_brightness != stored:
        await hass.services.async_call("light", "turn_on", {ATTR_ENTITY_ID: entity_id, ATTR_BRIGHTNESS: stored})
    return

# ON -> ON with brightness change: Store new brightness as original
if old_state.state == STATE_ON and new_state.state == STATE_ON:
    if new_brightness and new_brightness != old_brightness:
        hass.data[DOMAIN]["data"][entity_id] = new_brightness
        # Persist to storage
```

## Cleanup strategy

USER_INTENDED_BRIGHTNESS entries are cleared:
1. **On restore**: Immediately after detecting mismatch and restoring (Step 1)
2. **On new fade start**: Clear any existing intended state for that entity
3. **Timeout fallback**: 500ms after setting, in case no late call arrives

## Edge cases

- **Tolerance**: Allow ±3 brightness difference to account for device rounding
- **Non-dimmable lights**: Only on/off state matters, no brightness comparison
- **Multiple entities**: Each entity tracked independently
- **Rapid manual changes**: Each change updates the intended state

## Why this works

1. **No context needed**: We detect manual changes by comparing actual vs expected state
2. **Handles late calls**: Intended state catches and reverts fade calls that arrive after user intervention
3. **Self-cleaning**: Multiple cleanup mechanisms prevent stale data
4. **Simple model**: Brightness value captures both on/off state (0 = off) and brightness level
