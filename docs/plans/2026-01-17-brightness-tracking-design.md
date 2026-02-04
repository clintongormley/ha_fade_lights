# Brightness Tracking Redesign

## Problem

When fade_lights fades a light to off, it stores the final brightness (e.g., 1%) as the last known value. When the user manually turns the light on, it restores to 1% instead of the original brightness before fading began.

The auto-brightness feature exists as a workaround to boost these dim lights, but this is treating the symptom rather than the cause.

## Solution

Replace the single-value storage with a dual-value system that tracks user intent separately from fade state.

## Storage Schema

**Current (v1):**
```python
storage_data = {
    "light_living_room": 204,
}
```

**New (v2):**
```python
storage_data = {
    "light_living_room": {
        "orig": 204,  # User's intended brightness (0-255)
        "curr": 204,  # Last value set by fade service (0-255)
    },
}
```

**Migration:** On first load, detect old format (integer value) and convert:
```python
# Old: "light_living_room": 204
# New: "light_living_room": {"orig": 204, "curr": 204}
```

## Context Tracking

Track fade service contexts to distinguish our changes from external changes:

```python
ACTIVE_CONTEXTS: set[str] = set()
```

- Create context before fade starts: `context = Context()`
- Register: `ACTIVE_CONTEXTS.add(context.id)`
- Pass context to all `light.turn_on` / `light.turn_off` calls
- On completion/cancel: `ACTIVE_CONTEXTS.discard(context.id)`

## Event Listener

Single unified listener handles all light state changes:

```
state_changed event received
    │
    ├─ Not a light entity? → Ignore
    │
    ├─ event.context.id in ACTIVE_CONTEXTS?
    │       │
    │       └─ Ignore (fade function handles storage internally)
    │
    └─ Not our context (manual/external/other automation)
            │
            ├─ Light turned ON (was OFF)
            │       │
            │       └─ Restore "orig" brightness to light
            │
            ├─ Light turned OFF (was ON)
            │       │
            │       └─ No action
            │
            └─ Brightness changed (was already ON)
                    │
                    ├─ Cancel active fade if any
                    │
                    └─ Store new brightness as "orig"
```

## Fade Function Logic

```python
async def _execute_fade(hass, entity_id, brightness_pct, transition_ms, context, step_delay_ms):
    start_level = _get_current_level(hass, entity_id)
    end_level = int(brightness_pct / 100 * 255)

    # Store starting brightness as "curr"
    _store_curr_brightness(hass, entity_id, start_level)

    # Calculate steps and delta...

    for step in range(num_steps):
        curr = _get_curr_brightness(hass, entity_id)
        new_level = curr + delta  # (delta is negative when fading down)
        new_level = clamp(new_level, 0, 255)

        _store_curr_brightness(hass, entity_id, new_level)
        await set_brightness(hass, entity_id, new_level, context)
        await asyncio.sleep(step_delay_ms / 1000)

    # Fade complete - store "orig" if brightness > 0
    if end_level > 0:
        _store_orig_brightness(hass, entity_id, end_level)
```

## Removals

**Features:**
- Auto-brightness (threshold/target options)
- `force` parameter on fade service
- `_is_automated()` function
- Mid-fade brightness comparison check

**Constants:**
- `AUTO_BRIGHTNESS_THRESHOLD`
- `AUTO_BRIGHTNESS_TARGET`
- `OPTION_AUTO_BRIGHTNESS_THRESHOLD`
- `OPTION_AUTO_BRIGHTNESS_TARGET`
- `ATTR_FORCE`
- `DEFAULT_FORCE`

## Files to Modify

| File | Changes |
|------|---------|
| `__init__.py` | Replace event listener, simplify fade logic, add `ACTIVE_CONTEXTS`, update storage functions, remove `_is_automated()` |
| `const.py` | Remove auto-brightness constants, add `KEY_ORIG_BRIGHTNESS`, `KEY_CURR_BRIGHTNESS`, bump `STORAGE_VERSION` |
| `config_flow.py` | Remove auto-brightness options from UI |
| `strings.json` | Remove auto-brightness translations |
| `services.yaml` | Remove `force` parameter |
| `translations/en.json` | Remove auto-brightness translations |

## New Functions

- `_store_orig_brightness(hass, entity_id, level)` - Store user's intended brightness
- `_store_curr_brightness(hass, entity_id, level)` - Store current fade brightness
- `_get_orig_brightness(hass, entity_id) -> int` - Get user's intended brightness
- `_get_curr_brightness(hass, entity_id) -> int` - Get current fade brightness
- `_migrate_storage(storage_data) -> dict` - One-time migration from v1 to v2

## Replaced Functions

- `_store_current_level()` → replaced by `_store_orig_brightness()` and `_store_curr_brightness()`
- `_is_automated()` → removed entirely
