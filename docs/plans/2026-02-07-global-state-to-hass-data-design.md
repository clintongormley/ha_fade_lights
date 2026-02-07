# Move from Global State to hass.data

## Problem

The integration uses 6 module-level global dictionaries in `__init__.py` to track fade state:
- `ACTIVE_FADES` (entity_id -> asyncio.Task)
- `FADE_CANCEL_EVENTS` (entity_id -> asyncio.Event)
- `FADE_EXPECTED_STATE` (entity_id -> ExpectedState)
- `FADE_COMPLETE_CONDITIONS` (entity_id -> asyncio.Condition)
- `INTENDED_STATE_QUEUE` (entity_id -> list[State])
- `RESTORE_TASKS` (entity_id -> asyncio.Task)

This violates the HA pattern of storing integration state in `hass.data[DOMAIN]`, risks state leaking across reloads, and makes the code harder to test and reason about.

## Solution

Create a `FadeCoordinator` class that holds all state as instance variables. Store it on `hass.data[DOMAIN]`. Move all fade logic from `__init__.py` into `coordinator.py`.

## EntityFadeState dataclass

Replace 6 parallel dicts with one dict of per-entity state objects:

```python
@dataclass
class EntityFadeState:
    """Transient state for a single entity's fade operations."""
    active_task: asyncio.Task | None = None
    cancel_event: asyncio.Event | None = None
    complete_condition: asyncio.Condition | None = None
    expected_state: ExpectedState | None = None
    intended_queue: list[State] = field(default_factory=list)
    restore_task: asyncio.Task | None = None
```

Access pattern:
```python
# Before: 6 dict lookups
if entity_id in ACTIVE_FADES:
    task = ACTIVE_FADES[entity_id]
    FADE_CANCEL_EVENTS[entity_id].set()

# After: 1 lookup, attribute access
entity = self._entities.get(entity_id)
if entity and entity.active_task:
    entity.cancel_event.set()
```

Note: `expected_state` can outlive `active_task` (persists for late event matching after cancellation). Both fields coexist during active fades. The `EntityFadeState` object stays alive as long as any field is non-default.

## FadeCoordinator class

Stored directly on `hass.data[DOMAIN]` (replaces the current dict):

```python
class FadeCoordinator:
    def __init__(self, hass, entry, store, storage_data):
        self.hass = hass
        self.entry = entry
        self.store = store
        self.data = storage_data           # Per-light persistent config
        self.min_step_delay_ms = ...
        self.testing_lights: set[str] = set()
        self._entities: dict[str, EntityFadeState] = {}
```

### Public methods (used by websocket_api, autoconfigure, __init__)
- `handle_fado(call)` - service handler
- `handle_state_change(event)` - state change listener
- `get_light_config(entity_id)` - per-light config access
- `get_orig_brightness(entity_id)` - stored original brightness
- `store_orig_brightness(entity_id, level)` - store original brightness
- `save_storage()` - persist to disk
- `cleanup_entity(entity_id)` - clean up deleted entity
- `shutdown()` - clean shutdown of all operations

### Private methods (internal fade logic)
- `_fade_light(entity_id, params, delay_ms)`
- `_execute_fade(entity_id, params, delay_ms, cancel_event)`
- `_apply_step(entity_id, step, use_transition)`
- `_match_and_remove_expected(entity_id, new_state)`
- `_restore_intended_state(entity_id)`
- `_cancel_and_wait_for_fade(entity_id)`
- `_add_expected_values(entity_id, values)`
- `_wait_until_stale_events_flushed(entity_id, timeout)`
- `_handle_off_to_on(entity_id, new_state)`
- `_restore_original_brightness(entity_id, brightness)`
- `_get_intended_brightness(entity_id, old_state, new_state)`

### Module-level utility functions (stateless, in coordinator.py)
- `_expand_light_groups(hass, entity_ids)` - reads hass.states only
- `_can_apply_fade_params(state, params)` - pure function
- `_should_process_state_change(new_state)` - pure predicate
- `_is_off_to_on_transition(old_state, new_state)` - pure predicate
- `_is_brightness_change(old_state, new_state)` - pure predicate
- `_sleep_remaining_step_time(step_start, delay_ms)` - pure async

Note: `_expand_light_groups` needs access to `coordinator.data` and `coordinator.testing_lights` for exclude/testing checks, so it takes the coordinator as a parameter or becomes a method.

## Thin __init__.py

~100 lines. Responsibilities:
1. `async_setup` - auto-create config entry (unchanged)
2. `async_setup_entry` - create coordinator, register service/listeners/panel, store on `hass.data`
3. `async_unload_entry` - call `coordinator.shutdown()`, remove service/panel, pop `hass.data`
4. `_apply_stored_log_level` - setup concern, stays here

Service handler and state change listener are thin wrappers:
```python
coordinator = FadeCoordinator(hass, entry, store, storage_data)
hass.data[DOMAIN] = coordinator

async def handle_fado(call):
    await coordinator.handle_fado(call)

@callback
def handle_light_state_change(event):
    coordinator.handle_state_change(event)
```

## Impact on other modules

### websocket_api.py - minor changes
```python
# Before
hass.data[DOMAIN]["data"]
hass.data[DOMAIN]["store"]

# After
coordinator: FadeCoordinator = hass.data[DOMAIN]
coordinator.data
coordinator.store
```

### autoconfigure.py - minor changes
Same pattern: `hass.data[DOMAIN]` is now a coordinator, access attributes directly.

### notifications.py - minor changes
Same pattern for accessing `coordinator.data`.

### No changes needed
- `fade_change.py` - pure data/logic, no hass.data access
- `expected_state.py` - pure data/logic
- `fade_params.py` - pure data/logic
- `easing.py` - pure functions
- `config_flow.py` - no hass.data access

## File layout

```
custom_components/fado/
  __init__.py          ~100 lines  (setup, teardown, wrappers)
  coordinator.py       ~900 lines  (FadeCoordinator, EntityFadeState, utilities)
  fade_change.py       unchanged
  expected_state.py    unchanged
  fade_params.py       unchanged
  easing.py            unchanged
  websocket_api.py     minor changes
  autoconfigure.py     minor changes
  notifications.py     minor changes
  config_flow.py       unchanged
  const.py             unchanged
```

## Shutdown

```python
async def shutdown(self) -> None:
    for entity in self._entities.values():
        if entity.cancel_event:
            entity.cancel_event.set()

    tasks = []
    for entity in self._entities.values():
        if entity.active_task:
            entity.active_task.cancel()
            tasks.append(entity.active_task)
        if entity.restore_task:
            entity.restore_task.cancel()
            tasks.append(entity.restore_task)

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    self._entities.clear()
```

## Key behavioral note

`_cancel_and_wait_for_fade` currently waits on `entity_id not in ACTIVE_FADES`. In the new code, this becomes `entity.active_task is None` since the finally block in `fade_light` sets it to None and notifies the condition.

## Implementation approach

This is a mechanical refactor — same logic, different organization. No behavior changes.

1. Create `coordinator.py` with `EntityFadeState` and `FadeCoordinator`
2. Move all functions from `__init__.py` into coordinator methods
3. Replace global dict access with `self._entities` access
4. Replace `hass` parameter with `self.hass`
5. Slim down `__init__.py` to setup/teardown
6. Update `websocket_api.py`, `autoconfigure.py`, `notifications.py` to use coordinator
7. Run existing tests to verify no behavior changes
