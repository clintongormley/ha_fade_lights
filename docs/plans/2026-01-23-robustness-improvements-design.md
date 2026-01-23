# Robustness Improvements

## Overview

Add defensive programming patterns to handle race conditions, missing data, and integration lifecycle edge cases without using locks.

## Issues Identified

### Race Conditions with Shared State

The module uses four global dictionaries that are accessed from both event loop callbacks and async tasks:
- `ACTIVE_FADES`
- `FADE_CANCEL_EVENTS`
- `FADE_EXPECTED_BRIGHTNESS`
- `FADE_INTERRUPTED`

Keys can be removed between a check and subsequent access.

### Missing Guards

Several functions directly index dictionaries without checking if keys exist:
- `_store_orig_brightness` - accesses `hass.data[DOMAIN]["data"]`
- `_save_storage` - accesses `hass.data[DOMAIN]["store"]` and `["data"]`
- `_is_expected_fade_state` - indexes `FADE_EXPECTED_BRIGHTNESS[entity_id]`
- `_track_expected_brightness` - indexes `FADE_EXPECTED_BRIGHTNESS[entity_id]`

### Integration Lifecycle Issues

- `async_unload_entry` cancels tasks but doesn't await them
- Tasks created by callbacks (`_restore_manual_state`, brightness restoration) can outlive the integration
- No guards in callback-created tasks to check if integration is still loaded

## Design

### A. Defensive Access Patterns

**`_store_orig_brightness`** - Add guard at start:
```python
def _store_orig_brightness(hass: HomeAssistant, entity_id: str, level: int) -> None:
    """Store original brightness for an entity."""
    if DOMAIN not in hass.data:
        return
    hass.data[DOMAIN]["data"][entity_id] = level
```

**`_save_storage`** - Add guard at start:
```python
async def _save_storage(hass: HomeAssistant) -> None:
    """Save storage data to disk."""
    if DOMAIN not in hass.data:
        return
    store: Store = hass.data[DOMAIN]["store"]
    await store.async_save(hass.data[DOMAIN]["data"])
```

**`_is_expected_fade_state`** - Use `.get()` and return False if missing:
```python
def _is_expected_fade_state(entity_id: str, new_state: State) -> bool:
    """Check if state matches expected fade values (within tolerance)."""
    expected_values = FADE_EXPECTED_BRIGHTNESS.get(entity_id)
    if expected_values is None:
        return False
    # ... rest unchanged
```

**`_track_expected_brightness`** - Use `.get()` with early return:
```python
def _track_expected_brightness(entity_id: str, new_level: int, delta: int) -> None:
    """Track expected brightness for manual intervention detection."""
    expected_set = FADE_EXPECTED_BRIGHTNESS.get(entity_id)
    if expected_set is None:
        return
    # ... rest unchanged
```

### B. Lifecycle Safety

**`async_unload_entry`** - Await cancelled tasks:
```python
async def async_unload_entry(hass: HomeAssistant, _entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    for event in FADE_CANCEL_EVENTS.values():
        event.set()

    tasks = list(ACTIVE_FADES.values())
    for task in tasks:
        task.cancel()

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    ACTIVE_FADES.clear()
    FADE_CANCEL_EVENTS.clear()
    FADE_EXPECTED_BRIGHTNESS.clear()
    FADE_INTERRUPTED.clear()

    hass.services.async_remove(DOMAIN, SERVICE_FADE_LIGHTS)
    hass.data.pop(DOMAIN, None)

    return True
```

**`_restore_manual_state`** - Add guard at top:
```python
async def _restore_manual_state(...) -> None:
    """Restore intended state after manual intervention during fade."""
    if DOMAIN not in hass.data:
        return
    _LOGGER.debug("(%s) -> in _restore_manual_state", entity_id)
    # ... rest unchanged
```

**`_handle_off_to_on`** - Add guard before creating task:
```python
def _handle_off_to_on(hass: HomeAssistant, entity_id: str, new_state: State) -> None:
    """Handle OFF -> ON transition by restoring original brightness."""
    if DOMAIN not in hass.data:
        return
    _LOGGER.debug("(%s) -> Light turned OFF->ON", entity_id)
    # ... rest unchanged
```

## Summary

- 4 defensive `.get()` or guard additions for missing data
- 3 lifecycle safety guards for callback-created tasks
- 1 await addition for proper task cleanup on unload
