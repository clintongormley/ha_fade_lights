# Stale Event Suppression Design

## Problem

When a user manually turns off a light during an active fade, a race condition can cause the light to immediately turn back on:

1. Fade is running, light at brightness 135
2. User manually turns off the light
3. Manual OFF detected → `_handle_state_change` spawned to cancel fade
4. **Before fade cleanup completes**, a delayed state event arrives showing brightness=133 (from a previous fade step)
5. Since `is_fading=False` now (fade was cancelled), this looks like a normal OFF→ON transition
6. OFF→ON handler restores brightness to original (232)
7. Light turns back on unexpectedly

## Solution

Add a `FADE_INTERRUPTED` flag that suppresses stale state events during fade cleanup.

### New Tracking Dict

```python
# Maps entity_id -> True when manual intervention was just detected.
# Used to suppress stale state events from the cancelled fade.
# Cleared after _cancel_and_wait_for_fade completes.
FADE_INTERRUPTED: dict[str, bool] = {}
```

### Flow

1. Manual intervention detected in `handle_light_state_change`
2. Set `FADE_INTERRUPTED[entity_id] = True`
3. Spawn `_handle_state_change` async handler
4. Meanwhile, stale events arrive → check flag → flag is set → ignore
5. `_cancel_and_wait_for_fade` returns
6. Clear `FADE_INTERRUPTED[entity_id]`
7. Subsequent events processed normally

### Implementation Changes

**In `handle_light_state_change`:**
- Add early check: if `entity_id in FADE_INTERRUPTED`, log and return
- When manual intervention detected, set `FADE_INTERRUPTED[entity_id] = True` before spawning handler

**In `_handle_state_change`:**
- After `_cancel_and_wait_for_fade` returns, clear `FADE_INTERRUPTED.pop(entity_id, None)`

**In `async_unload_entry`:**
- Add `FADE_INTERRUPTED.clear()`

### Testing

Add test that:
1. Starts a fade
2. Manually turns off during fade
3. Simulates a delayed brightness event arriving
4. Verifies the stale event is ignored
5. Verifies light stays off (doesn't restore to original)