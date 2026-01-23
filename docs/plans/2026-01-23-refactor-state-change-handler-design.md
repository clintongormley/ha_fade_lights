# Refactor _handle_light_state_change

## Overview

Refactor `_handle_light_state_change` from a ~138 line monolithic function into a thin dispatcher with focused helper functions. This improves readability, testability, and maintainability.

## Current Issues

- **Length**: ~138 lines doing multiple things
- **Nesting**: Deep conditionals make flow hard to follow
- **Naming**: Logic could be clearer
- **Structure**: Three responsibilities mixed together

## Design

### Main Function (Thin Dispatcher)

```python
@callback
def _handle_light_state_change(hass: HomeAssistant, event: Event) -> None:
    """Handle light state changes - detects manual intervention and tracks brightness."""
    new_state = event.data.get("new_state")
    old_state = event.data.get("old_state")

    if not _should_process_state_change(new_state):
        return

    entity_id = new_state.entity_id

    if _is_stale_event(entity_id, new_state):
        return

    _log_state_change(entity_id, new_state)

    # During fade: check if this is our fade or manual intervention
    if entity_id in ACTIVE_FADES and entity_id in FADE_EXPECTED_BRIGHTNESS:
        if _is_expected_fade_state(entity_id, new_state):
            _LOGGER.debug("(%s) -> State matches expected during fade, ignoring", entity_id)
            return

        # Manual intervention detected
        _LOGGER.debug(
            "(%s) -> Manual intervention during fade: got=%s/%s",
            entity_id, new_state.state, new_state.attributes.get(ATTR_BRIGHTNESS),
        )
        FADE_INTERRUPTED[entity_id] = True
        hass.async_create_task(_restore_manual_state(hass, entity_id, old_state, new_state))
        return

    # Normal state handling (no active fade)
    if _is_off_to_on_transition(old_state, new_state):
        _handle_off_to_on(hass, entity_id, new_state)
        return

    if _is_brightness_change(old_state, new_state):
        _store_orig_brightness(hass, entity_id, new_state.attributes.get(ATTR_BRIGHTNESS))
```

### Helper Functions

#### Event Filtering

**`_should_process_state_change(new_state)`** - Pure predicate, no logging:
```python
def _should_process_state_change(new_state: State | None) -> bool:
    """Check if this state change should be processed."""
    if not new_state:
        return False
    if new_state.domain != LIGHT_DOMAIN:
        return False
    # Ignore group helpers
    if new_state.attributes.get(ATTR_ENTITY_ID) is not None:
        return False
    return True
```

**`_is_stale_event(entity_id, new_state)`** - Includes existing logging:
```python
def _is_stale_event(entity_id: str, new_state: State) -> bool:
    """Check if event should be suppressed during fade cleanup."""
    if entity_id not in FADE_INTERRUPTED:
        return False
    _LOGGER.debug(
        "(%s) -> Ignoring stale event during fade cleanup (%s/%s)",
        entity_id, new_state.state, new_state.attributes.get(ATTR_BRIGHTNESS),
    )
    return True
```

**`_log_state_change(entity_id, new_state)`** - Main debug log:
```python
def _log_state_change(entity_id: str, new_state: State) -> None:
    """Log state change details."""
    _LOGGER.debug(
        "(%s) -> state=%s, brightness=%s, is_fading=%s",
        entity_id, new_state.state,
        new_state.attributes.get(ATTR_BRIGHTNESS),
        entity_id in ACTIVE_FADES,
    )
```

#### Fade State Detection

**`_is_expected_fade_state(entity_id, new_state)`** - Tolerance check logic:
```python
def _is_expected_fade_state(entity_id: str, new_state: State) -> bool:
    """Check if state matches expected fade values (within tolerance)."""
    expected_values = FADE_EXPECTED_BRIGHTNESS[entity_id]
    new_brightness = new_state.attributes.get(ATTR_BRIGHTNESS)
    tolerance = 3

    if new_state.state == STATE_OFF and 0 in expected_values:
        return True

    if new_state.state == STATE_ON and new_brightness is not None:
        for expected in expected_values:
            if expected > 0 and abs(new_brightness - expected) <= tolerance:
                return True

    return False
```

#### State Transition Predicates

**`_is_off_to_on_transition(old_state, new_state)`**:
```python
def _is_off_to_on_transition(old_state: State | None, new_state: State) -> bool:
    """Check if this is an OFF -> ON transition."""
    return (
        old_state is not None
        and old_state.state == STATE_OFF
        and new_state.state == STATE_ON
    )
```

**`_is_brightness_change(old_state, new_state)`**:
```python
def _is_brightness_change(old_state: State | None, new_state: State) -> bool:
    """Check if this is an ON -> ON brightness change."""
    if not old_state or old_state.state != STATE_ON or new_state.state != STATE_ON:
        return False
    old_brightness = old_state.attributes.get(ATTR_BRIGHTNESS)
    new_brightness = new_state.attributes.get(ATTR_BRIGHTNESS)
    return new_brightness is not None and new_brightness != old_brightness
```

#### State Transition Handlers

**`_handle_off_to_on(hass, entity_id, new_state)`** - Brightness restoration with all logging:
```python
def _handle_off_to_on(hass: HomeAssistant, entity_id: str, new_state: State) -> None:
    """Handle OFF -> ON transition by restoring original brightness."""
    _LOGGER.debug("(%s) -> Light turned OFF->ON", entity_id)

    # Non-dimmable lights can't have brightness restored
    if ColorMode.BRIGHTNESS not in new_state.attributes.get(ATTR_SUPPORTED_COLOR_MODES, []):
        return

    orig_brightness = _get_orig_brightness(hass, entity_id)
    current_brightness = new_state.attributes.get(ATTR_BRIGHTNESS, 0)

    _LOGGER.debug(
        "(%s) -> orig_brightness=%s, current_brightness=%s",
        entity_id, orig_brightness, current_brightness,
    )

    if orig_brightness > 0 and current_brightness != orig_brightness:
        _LOGGER.debug("(%s) -> Restoring to brightness %s", entity_id, orig_brightness)
        hass.async_create_task(
            hass.services.async_call(
                LIGHT_DOMAIN, SERVICE_TURN_ON,
                {ATTR_ENTITY_ID: entity_id, ATTR_BRIGHTNESS: orig_brightness},
            )
        )
```

## File Organization

New helpers placed in existing "State Change Helpers" section (~line 602):

```python
# =============================================================================
# State Change Helpers
# =============================================================================

# --- Event Filtering ---
def _should_process_state_change(new_state: State | None) -> bool: ...
def _is_stale_event(entity_id: str, new_state: State) -> bool: ...
def _log_state_change(entity_id: str, new_state: State) -> None: ...

# --- Fade State Detection ---
def _is_expected_fade_state(entity_id: str, new_state: State) -> bool: ...

# --- State Transition Predicates ---
def _is_off_to_on_transition(old_state: State | None, new_state: State) -> bool: ...
def _is_brightness_change(old_state: State | None, new_state: State) -> bool: ...

# --- State Transition Handlers ---
def _handle_off_to_on(hass: HomeAssistant, entity_id: str, new_state: State) -> None: ...
```

## Summary

- Main function: ~138 lines -> ~35 lines
- 7 focused helper functions extracted
- All debug logging preserved
- Clear separation: filtering -> fade detection -> state transitions
- Each helper independently testable
