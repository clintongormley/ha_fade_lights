# Physical switch detection during fades

## Problem

When a fade is in progress and the user uses a physical switch to turn off the light or change brightness, the event is ignored because physical switch events don't carry our context ID.

## Solution

During an active fade, check if the light's state matches what we expect - regardless of context. If the state deviates unexpectedly, treat it as manual intervention and cancel the fade.

## Implementation

Modify `handle_light_state_change` in `__init__.py`:

1. Add `is_fading = entity_id in ACTIVE_FADES` check early in the function
2. When fading, compare actual state to expected state from `FADE_EXPECTED_BRIGHTNESS`
3. If unexpected OFF (and not expecting 0) or brightness differs by >5, set `is_our_context = False`
4. Let existing logic handle cancellation and storing new brightness

### Code change

```python
is_fading = entity_id in ACTIVE_FADES

# Check if this is from our own fade operations
is_our_context = event.context.id in ACTIVE_CONTEXTS

# During active fade, check if state matches what we expect
# If not, this is manual intervention regardless of context
if is_fading and entity_id in FADE_EXPECTED_BRIGHTNESS:
    expected = FADE_EXPECTED_BRIGHTNESS[entity_id]
    tolerance = 5

    # Unexpected OFF during fade (only if we're not fading to 0)
    if new_state.state == STATE_OFF and expected != 0:
        is_our_context = False
    # Unexpected brightness during fade
    elif (new_brightness is not None
          and abs(new_brightness - expected) > tolerance):
        is_our_context = False
```

## Edge cases

- **Fade to 0%**: When fading to 0%, we call `turn_off`. The check `expected != 0` ensures our own turn_off isn't treated as unexpected.
- **Non-dimmable lights**: These don't report brightness, so only the OFF state check applies.
- **Brightness tolerance**: Allow ±5 for rounding differences between requested and actual brightness.

## Testing

Existing tests cover the behavior. The change makes detection more robust for physical switches without changing the observable behavior for app-based changes.
