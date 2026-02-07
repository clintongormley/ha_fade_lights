# Global State to hass.data Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move 6 module-level global dicts and ~20 functions from `__init__.py` into a `FadeCoordinator` class in `coordinator.py`, stored on `hass.data[DOMAIN]`.

**Architecture:** Create `EntityFadeState` dataclass (replaces 6 parallel dicts) and `FadeCoordinator` class (replaces module-level functions). `__init__.py` becomes a thin setup/teardown wrapper. Other modules (`websocket_api.py`, `autoconfigure.py`, `notifications.py`) update from dict access to coordinator attribute access.

**Tech Stack:** Python 3.13, Home Assistant custom component, asyncio, dataclasses

**Design doc:** `docs/plans/2026-02-07-global-state-to-hass-data-design.md`

---

### Task 1: Create `coordinator.py` with `EntityFadeState` and `FadeCoordinator`

This is the largest task. Create the new file by moving all fade logic from `__init__.py`.

**Files:**
- Create: `custom_components/fado/coordinator.py`
- Reference: `custom_components/fado/__init__.py` (source of all code to move)

**Step 1: Create `coordinator.py` with EntityFadeState dataclass**

At the top of the file, after imports:

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

**Step 2: Create `FadeCoordinator` class with `__init__`**

```python
class FadeCoordinator:
    """Central manager for all fade operations."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        store: Store,
        storage_data: dict[str, Any],
    ) -> None:
        self.hass = hass
        self.entry = entry
        self.store = store
        self.data: dict[str, Any] = storage_data
        self.min_step_delay_ms: int = entry.options.get(
            OPTION_MIN_STEP_DELAY_MS, DEFAULT_MIN_STEP_DELAY_MS
        )
        self.testing_lights: set[str] = set()
        self._entities: dict[str, EntityFadeState] = {}
```

**Step 3: Add entity state access helpers**

```python
    def get_entity(self, entity_id: str) -> EntityFadeState | None:
        """Get entity fade state, or None if not tracked."""
        return self._entities.get(entity_id)

    def get_or_create_entity(self, entity_id: str) -> EntityFadeState:
        """Get or create entity fade state."""
        if entity_id not in self._entities:
            self._entities[entity_id] = EntityFadeState()
        return self._entities[entity_id]
```

**Step 4: Move all functions from `__init__.py` as methods**

Move these functions from `__init__.py` into `FadeCoordinator` as methods. For each function:
- Remove the `hass: HomeAssistant` parameter (use `self.hass` instead)
- Replace global dict access (`ACTIVE_FADES[entity_id]`) with `self._entities` access
- Keep the same logic, just different data access patterns

**Public methods** (remove leading underscore):

| Old function | New method | Key changes |
|---|---|---|
| `_handle_fado(hass, call)` | `async handle_fado(self, call)` | Use `self.min_step_delay_ms` instead of `hass.data[DOMAIN]["min_step_delay_ms"]` |
| `_cleanup_entity_data(hass, entity_id)` | `async cleanup_entity(self, entity_id)` | Use `self._entities`, `self.testing_lights`, `self.data`, `self.store` |
| `_get_light_config(hass, entity_id)` | `get_light_config(self, entity_id)` | Use `self.data` |
| `_get_orig_brightness(hass, entity_id)` | `get_orig_brightness(self, entity_id)` | Use `self.get_light_config()` |
| `_store_orig_brightness(hass, entity_id, level)` | `store_orig_brightness(self, entity_id, level)` | Use `self.data` |
| `_save_storage(hass)` | `async save_storage(self)` | Use `self.store`, `self.data` |

**Private methods** (keep underscore):

| Old function | New method | Key changes |
|---|---|---|
| `_fade_light(hass, entity_id, params, delay_ms)` | `async _fade_light(self, entity_id, params, delay_ms)` | Use `entity = self.get_or_create_entity(entity_id)` then set `entity.cancel_event`, `entity.complete_condition`, `entity.active_task`. In finally: set `entity.active_task = None`, `entity.cancel_event = None`, notify `entity.complete_condition` then set to None. |
| `_execute_fade(hass, entity_id, params, delay_ms, cancel_event)` | `async _execute_fade(self, entity_id, params, delay_ms, cancel_event)` | Use `self.hass.states.get()`, `self.store_orig_brightness()`, `self.get_light_config()`, `self._add_expected_values()`, `self._apply_step()`. Get expected_state from entity: `entity.expected_state` |
| `_apply_step(hass, entity_id, step, *, use_transition)` | `async _apply_step(self, entity_id, step, *, use_transition)` | Use `self.hass.services.async_call()` |
| `_handle_light_state_change(hass, event)` | `handle_state_change(self, event)` | Public (called from __init__). Use `entity = self.get_entity(entity_id)` for checking `entity.active_task`, `entity.expected_state`, `entity.restore_task`. Use `entity.intended_queue` instead of `INTENDED_STATE_QUEUE[entity_id]`. Create restore task on `entity.restore_task`. |
| `_match_and_remove_expected(entity_id, new_state)` | `_match_and_remove_expected(self, entity_id, new_state)` | Get `entity.expected_state` instead of `FADE_EXPECTED_STATE.get()` |
| `_handle_off_to_on(hass, entity_id, new_state)` | `_handle_off_to_on(self, entity_id, new_state)` | Use `self.get_orig_brightness()`, `self.hass.async_create_task()` |
| `_restore_original_brightness(hass, entity_id, brightness)` | `async _restore_original_brightness(self, entity_id, brightness)` | Use `self._add_expected_brightness()`, `self.hass.services.async_call()` |
| `_restore_intended_state(hass, entity_id)` | `async _restore_intended_state(self, entity_id)` | Use `entity.intended_queue` instead of `INTENDED_STATE_QUEUE`. In finally: `entity.restore_task = None` instead of `RESTORE_TASKS.pop()` |
| `_get_intended_brightness(hass, entity_id, old, new)` | `_get_intended_brightness(self, entity_id, old, new)` | Use `self.get_orig_brightness()` |
| `_cancel_and_wait_for_fade(entity_id)` | `async _cancel_and_wait_for_fade(self, entity_id)` | Use `entity.active_task`, `entity.complete_condition`, `entity.cancel_event`. Wait condition: `entity.active_task is None` |
| `_add_expected_values(entity_id, values)` | `_add_expected_values(self, entity_id, values)` | Use `entity = self.get_or_create_entity(entity_id)`, create `ExpectedState` if `entity.expected_state is None` |
| `_add_expected_brightness(entity_id, brightness)` | `_add_expected_brightness(self, entity_id, brightness)` | Calls `self._add_expected_values()` |
| `_wait_until_stale_events_flushed(entity_id, timeout)` | `async _wait_until_stale_events_flushed(self, entity_id, timeout)` | Use `entity.expected_state` instead of `FADE_EXPECTED_STATE.get()` |

**Add shutdown method:**

```python
    async def shutdown(self) -> None:
        """Shut down all fade operations."""
        for entity in self._entities.values():
            if entity.cancel_event:
                entity.cancel_event.set()

        tasks: list[asyncio.Task] = []
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

**Step 5: Move stateless utility functions as module-level functions**

Copy these as-is from `__init__.py` to `coordinator.py` (they don't use `self`):
- `_should_process_state_change(new_state)` — no changes
- `_is_off_to_on_transition(old_state, new_state)` — no changes
- `_is_brightness_change(old_state, new_state)` — no changes
- `_sleep_remaining_step_time(step_start, delay_ms)` — no changes
- `_can_apply_fade_params(state, params)` — no changes

For `_expand_light_groups`: this currently accesses `hass.data[DOMAIN]` for exclude/testing_lights filtering. Make it a method on the coordinator since it needs `self.data` and `self.testing_lights`:

```python
    def _expand_light_groups(self, entity_ids: list[str]) -> list[str]:
        """Expand light groups to individual light entities..."""
        # Same logic, but use self.hass, self.get_light_config(), self.testing_lights
```

**Step 6: Verify `coordinator.py` has correct imports**

The file needs these imports (copied from `__init__.py`):

```python
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from homeassistant.components.light import (
    ATTR_BRIGHTNESS,
    ATTR_SUPPORTED_COLOR_MODES,
)
from homeassistant.components.light import (
    ATTR_COLOR_TEMP_KELVIN as HA_ATTR_COLOR_TEMP_KELVIN,
)
from homeassistant.components.light import (
    ATTR_HS_COLOR as HA_ATTR_HS_COLOR,
)
from homeassistant.components.light.const import DOMAIN as LIGHT_DOMAIN
from homeassistant.components.light.const import ColorMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_ENTITY_ID,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_OFF,
    STATE_ON,
    STATE_UNAVAILABLE,
)
from homeassistant.core import (
    Event,
    EventStateChangedData,
    HomeAssistant,
    ServiceCall,
    State,
    callback,
)
from homeassistant.helpers.service import remove_entity_service_fields
from homeassistant.helpers.storage import Store
from homeassistant.helpers.target import (
    TargetSelection,
    async_extract_referenced_entity_ids,
)

from .const import (
    DEFAULT_MIN_STEP_DELAY_MS,
    DOMAIN,
    FADE_CANCEL_TIMEOUT_S,
    NATIVE_TRANSITION_MS,
    OPTION_MIN_STEP_DELAY_MS,
)
from .expected_state import ExpectedState, ExpectedValues
from .fade_change import FadeChange, FadeStep
from .fade_params import FadeParams
```

**Step 7: Commit**

```bash
git add custom_components/fado/coordinator.py
git commit -m "feat: create FadeCoordinator class in coordinator.py"
```

---

### Task 2: Slim down `__init__.py`

**Files:**
- Modify: `custom_components/fado/__init__.py`

**Step 1: Rewrite `__init__.py`**

Replace the entire file with a thin setup/teardown wrapper. The file should:

1. Remove all 6 global dicts
2. Remove all function definitions (they're now in coordinator.py)
3. Remove imports that are no longer needed
4. Add import of `FadeCoordinator` from `.coordinator`
5. Keep `async_setup` unchanged
6. Rewrite `async_setup_entry` to create coordinator and store on `hass.data[DOMAIN]`
7. Rewrite `async_unload_entry` to call `coordinator.shutdown()`
8. Keep `_apply_stored_log_level` (setup concern)

The new `__init__.py` should look like:

```python
"""The Fado integration."""

from __future__ import annotations

import contextlib
import logging
from datetime import datetime, timedelta
from pathlib import Path

import voluptuous as vol
from homeassistant.components import frontend, panel_custom
from homeassistant.components.http import StaticPathConfig
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import Event, HomeAssistant, ServiceCall, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.event import (
    TrackStates,
    async_track_state_change_filtered,
    async_track_time_interval,
)
from homeassistant.helpers.storage import Store
from homeassistant.helpers.typing import ConfigType

from .const import (
    DEFAULT_LOG_LEVEL,
    DEFAULT_MIN_STEP_DELAY_MS,
    DOMAIN,
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARNING,
    OPTION_LOG_LEVEL,
    OPTION_MIN_STEP_DELAY_MS,
    SERVICE_FADO,
    STORAGE_KEY,
    UNCONFIGURED_CHECK_INTERVAL_HOURS,
)
from .coordinator import FadeCoordinator
from .notifications import _notify_unconfigured_lights
from .websocket_api import async_register_websocket_api

_LOGGER = logging.getLogger(__name__)


async def async_setup(hass: HomeAssistant, _config: ConfigType) -> bool:
    """Set up the Fado component."""
    if not hass.config_entries.async_entries(DOMAIN):
        hass.async_create_task(
            hass.config_entries.flow.async_init(DOMAIN, context={"source": "import"})
        )
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Fado from a config entry."""
    store: Store[dict[str, int]] = Store(hass, 1, STORAGE_KEY)
    storage_data = await store.async_load() or {}

    coordinator = FadeCoordinator(hass, entry, store, storage_data)
    hass.data[DOMAIN] = coordinator

    async def handle_fado(call: ServiceCall) -> None:
        await coordinator.handle_fado(call)

    @callback
    def handle_light_state_change(event):
        coordinator.handle_state_change(event)

    # Valid easing curve names
    valid_easing = [
        "auto", "linear", "ease_in_quad", "ease_in_cubic",
        "ease_out_quad", "ease_out_cubic", "ease_in_out_sine",
    ]

    hass.services.async_register(
        DOMAIN, SERVICE_FADO, handle_fado,
        schema=cv.make_entity_service_schema(
            {vol.Optional("easing", default="auto"): vol.In(valid_easing)},
            extra=vol.ALLOW_EXTRA,
        ),
    )

    tracker = async_track_state_change_filtered(
        hass, TrackStates(False, set(), {"light"}), handle_light_state_change,
    )
    entry.async_on_unload(tracker.async_remove)

    # Entity registry listener
    async def handle_entity_registry_updated(
        event: Event[er.EventEntityRegistryUpdatedData],
    ) -> None:
        action = event.data["action"]
        entity_id = event.data["entity_id"]
        if not entity_id.startswith("light."):
            return
        if action == "remove":
            await coordinator.cleanup_entity(entity_id)
            await _notify_unconfigured_lights(hass)
        elif action == "create":
            await _notify_unconfigured_lights(hass)
        elif action == "update":
            changes = event.data.get("changes", {})
            if "disabled_by" in changes:
                await _notify_unconfigured_lights(hass)

    entry.async_on_unload(
        hass.bus.async_listen(er.EVENT_ENTITY_REGISTRY_UPDATED, handle_entity_registry_updated)
    )

    async def _daily_unconfigured_check(_now: datetime) -> None:
        await _notify_unconfigured_lights(hass)

    entry.async_on_unload(
        async_track_time_interval(
            hass, _daily_unconfigured_check,
            timedelta(hours=UNCONFIGURED_CHECK_INTERVAL_HOURS),
        )
    )

    async_register_websocket_api(hass)

    if hass.http is not None:
        await hass.http.async_register_static_paths(
            [StaticPathConfig("/fado_panel", str(Path(__file__).parent / "frontend"), cache_headers=False)]
        )
        await panel_custom.async_register_panel(
            hass, frontend_url_path="fado", webcomponent_name="fado-panel",
            sidebar_title="Fado", sidebar_icon="mdi:lightbulb-variant",
            module_url="/fado_panel/panel.js", require_admin=False,
        )

    await _apply_stored_log_level(hass, entry)
    await _notify_unconfigured_lights(hass)
    return True


async def async_unload_entry(hass: HomeAssistant, _entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    coordinator: FadeCoordinator = hass.data[DOMAIN]
    await coordinator.shutdown()

    hass.services.async_remove(DOMAIN, SERVICE_FADO)
    hass.data.pop(DOMAIN, None)
    frontend.async_remove_panel(hass, "fado")
    return True


async def _apply_stored_log_level(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Apply the stored log level setting."""
    log_level = entry.options.get(OPTION_LOG_LEVEL, DEFAULT_LOG_LEVEL)
    level_map = {
        LOG_LEVEL_WARNING: "warning",
        LOG_LEVEL_INFO: "info",
        LOG_LEVEL_DEBUG: "debug",
    }
    python_level = level_map.get(log_level, "warning")
    with contextlib.suppress(Exception):
        await hass.services.async_call(
            "logger", "set_level",
            {f"custom_components.{DOMAIN}": python_level},
        )
```

**Step 2: Commit**

```bash
git add custom_components/fado/__init__.py
git commit -m "refactor: slim __init__.py to thin setup/teardown wrapper"
```

---

### Task 3: Update `websocket_api.py`

**Files:**
- Modify: `custom_components/fado/websocket_api.py`

**Step 1: Update imports and data access**

Changes needed:
1. Add import: `from .coordinator import FadeCoordinator`
2. In `async_get_lights` (line 54): change `hass.data.get(DOMAIN, {}).get("data", {})` to:
   ```python
   coordinator: FadeCoordinator = hass.data.get(DOMAIN)
   storage_data = coordinator.data if coordinator else {}
   ```
3. In `async_save_light_config` (line 172): change `hass.data.get(DOMAIN, {}).get("data", {})` to:
   ```python
   coordinator: FadeCoordinator = hass.data[DOMAIN]
   data = coordinator.data
   ```
4. In `async_save_light_config` (line 197): change `store = hass.data[DOMAIN]["store"]` / `await store.async_save(data)` to:
   ```python
   await coordinator.save_storage()
   ```
5. In `ws_autoconfigure` (line 282-293): change `_get_light_config` usage. Since the local `_get_light_config` in websocket_api.py already reads from `hass.data[DOMAIN].get("data", {})`, update it to use coordinator:
   ```python
   coordinator: FadeCoordinator = hass.data[DOMAIN]
   # For _get_light_config calls, use coordinator.get_light_config()
   ```
   Actually, `websocket_api.py` has its own `_get_light_config` function (line 282). Replace its body:
   ```python
   def _get_light_config(hass: HomeAssistant, entity_id: str) -> dict[str, Any]:
       coordinator: FadeCoordinator = hass.data.get(DOMAIN)
       return coordinator.get_light_config(entity_id) if coordinator else {}
   ```
6. In `ws_autoconfigure` (line 344): change `hass.data.get(DOMAIN, {}).get("testing_lights", set())` to:
   ```python
   coordinator: FadeCoordinator = hass.data.get(DOMAIN)
   testing_lights = coordinator.testing_lights if coordinator else set()
   ```
7. In `ws_save_settings` (lines 521-522): change `hass.data[DOMAIN]["min_step_delay_ms"]` to:
   ```python
   coordinator: FadeCoordinator = hass.data.get(DOMAIN)
   if coordinator:
       coordinator.min_step_delay_ms = msg["default_min_delay_ms"]
   ```

**Step 2: Commit**

```bash
git add custom_components/fado/websocket_api.py
git commit -m "refactor: update websocket_api.py to use FadeCoordinator"
```

---

### Task 4: Update `autoconfigure.py`

**Files:**
- Modify: `custom_components/fado/autoconfigure.py`

**Step 1: Update data access**

Changes needed:
1. Add import: `from .coordinator import FadeCoordinator`
2. In `async_autoconfigure_light` (line 93): change
   ```python
   light_config = hass.data.setdefault(DOMAIN, {}).setdefault("data", {}).setdefault(entity_id, {})
   ```
   to:
   ```python
   coordinator: FadeCoordinator = hass.data[DOMAIN]
   if entity_id not in coordinator.data:
       coordinator.data[entity_id] = {}
   light_config = coordinator.data[entity_id]
   ```
3. In `_async_test_light_delay` (line 271): change `hass.data.get(DOMAIN, {}).get("min_step_delay_ms", DEFAULT_MIN_STEP_DELAY_MS)` to:
   ```python
   coordinator: FadeCoordinator = hass.data.get(DOMAIN)
   global_min = coordinator.min_step_delay_ms if coordinator else DEFAULT_MIN_STEP_DELAY_MS
   ```

**Step 2: Commit**

```bash
git add custom_components/fado/autoconfigure.py
git commit -m "refactor: update autoconfigure.py to use FadeCoordinator"
```

---

### Task 5: Update `notifications.py`

**Files:**
- Modify: `custom_components/fado/notifications.py`

**Step 1: Update data access**

Changes needed:
1. Add import: `from .coordinator import FadeCoordinator`
2. In `_get_unconfigured_lights` (line 21-25): change:
   ```python
   if DOMAIN not in hass.data:
       return set()
   ...
   storage_data = hass.data[DOMAIN].get("data", {})
   ```
   to:
   ```python
   coordinator: FadeCoordinator | None = hass.data.get(DOMAIN)
   if coordinator is None:
       return set()
   ...
   storage_data = coordinator.data
   ```

**Step 2: Commit**

```bash
git add custom_components/fado/notifications.py
git commit -m "refactor: update notifications.py to use FadeCoordinator"
```

---

### Task 6: Update `conftest.py`

**Files:**
- Modify: `tests/conftest.py`

**Step 1: Remove `clear_module_state` fixture**

The `clear_module_state` autouse fixture (lines 18-41) clears 4 global dicts that no longer exist. Remove this entire fixture. State is now scoped to the coordinator instance, which is created per test when `init_integration` runs and cleaned up when the test tears down.

**Step 2: Commit**

```bash
git add tests/conftest.py
git commit -m "test: remove global state clearing fixture (state now in coordinator)"
```

---

### Task 7: Update test files that import globals

This is the most tedious task. Each test file that imports from `custom_components.fado` needs updating.

**Files to update:**

**7a: `tests/test_init.py`**

Update imports and assertions:
- Remove imports of `ACTIVE_FADES`, `FADE_CANCEL_EVENTS`, `FADE_EXPECTED_STATE`
- `test_setup_entry_registers_service`: Change `assert DOMAIN in hass.data` (still true - coordinator is stored there)
- `test_setup_entry_loads_storage`: Change `hass.data[DOMAIN]["data"]` to `hass.data[DOMAIN].data` and `hass.data[DOMAIN]["store"]` to `hass.data[DOMAIN].store`
- `test_unload_entry_clears_tracking_dicts`: Rewrite to use coordinator's `_entities` dict. Set up state via coordinator, verify `_entities` is cleared after unload.
- `test_store_orig_brightness_when_domain_not_in_hass`: Import from `custom_components.fado.coordinator` and test directly, or remove (coordinator methods don't work without coordinator)
- `test_save_storage_when_domain_not_in_hass`: Similar — test coordinator method
- `test_handle_off_to_on_when_domain_not_in_hass`: Similar
- `test_fado_skips_unavailable_entities`: Update `_execute_fade` patch path to `custom_components.fado.coordinator._execute_fade` — but since it's now a method, patch the coordinator's method instead: `custom_components.fado.coordinator.FadeCoordinator._execute_fade`

**7b: `tests/test_storage.py`**

Major updates needed:
- Remove all 6 global dict imports
- Remove `cleanup_globals` fixture
- Import `FadeCoordinator` from `custom_components.fado.coordinator`
- `hass_with_storage` fixture: Create a `FadeCoordinator` instance and store on `hass.data[DOMAIN]`
  ```python
  coordinator = FadeCoordinator(hass, mock_entry, mock_store, storage_data)
  hass.data[DOMAIN] = coordinator
  ```
- `TestGetLightConfig`: Call `coordinator.get_light_config()` instead of `_get_light_config(hass, ...)`
- `TestGetOrigBrightness`: Call `coordinator.get_orig_brightness()` instead of `_get_orig_brightness(hass, ...)`
- `TestStoreOrigBrightness`: Call `coordinator.store_orig_brightness()` instead of `_store_orig_brightness(hass, ...)`
- `TestCleanupEntityData`: Call `coordinator.cleanup_entity()` instead of `_cleanup_entity_data(hass, ...)`. Set up state via `coordinator._entities` and `coordinator.data` instead of global dicts. Verify cleanup via coordinator attributes.
  - For `test_cleanup_cancels_active_fade`: Set `coordinator.get_or_create_entity("light.test").active_task = task`
  - For `test_cleanup_sets_cancel_event`: Set `entity.cancel_event = event`
  - Etc. for each global dict that was being tested

**7c: `tests/test_event_waiting.py`**

- Remove `FADE_EXPECTED_STATE` import
- Import `FadeCoordinator` from coordinator
- Need coordinator in scope. Tests that use `init_integration` already have coordinator on `hass.data[DOMAIN]`. Access it:
  ```python
  coordinator: FadeCoordinator = hass.data[DOMAIN]
  ```
- Replace `_add_expected_brightness(entity_id, 100)` with `coordinator._add_expected_brightness(entity_id, 100)`
- Replace `_match_and_remove_expected(entity_id, state)` with `coordinator._match_and_remove_expected(entity_id, state)`
- Replace `_wait_until_stale_events_flushed(entity_id)` with `coordinator._wait_until_stale_events_flushed(entity_id)`
- Replace `FADE_EXPECTED_STATE[entity_id] = ...` with `coordinator.get_or_create_entity(entity_id).expected_state = ...`
- Replace `FADE_EXPECTED_STATE.pop(entity_id, None)` cleanup with accessing coordinator
- Note: `ExpectedState` is still imported from `custom_components.fado.expected_state` (unchanged)

**7d: `tests/test_manual_interruption.py`**

- Remove imports of `ACTIVE_FADES`, `FADE_CANCEL_EVENTS`, `FADE_EXPECTED_STATE`, `INTENDED_STATE_QUEUE`, `RESTORE_TASKS`
- Import `FadeCoordinator` from coordinator
- Replace all direct global dict accesses with coordinator access:
  - `entity_id in ACTIVE_FADES` → `coordinator.get_entity(entity_id) and coordinator.get_entity(entity_id).active_task is not None`
  - `INTENDED_STATE_QUEUE.get(entity_id)` → `coordinator.get_entity(entity_id).intended_queue` (check entity exists first)
- Replace `_restore_intended_state(hass, entity_id)` with `coordinator._restore_intended_state(entity_id)`
- Replace `_cancel_and_wait_for_fade(entity_id)` with `coordinator._cancel_and_wait_for_fade(entity_id)`
- Replace `_get_intended_brightness(hass, ...)` with `coordinator._get_intended_brightness(...)`

**7e: `tests/test_apply_step.py`**

- Change `from custom_components.fado import _apply_step` to import from coordinator:
  ```python
  from custom_components.fado.coordinator import FadeCoordinator
  ```
- Tests use `mock_hass` (MagicMock). Need to create a coordinator with mock_hass instead:
  ```python
  coordinator = FadeCoordinator(mock_hass, mock_entry, mock_store, {})
  await coordinator._apply_step(entity_id, step)
  ```
  Or keep the standalone function approach by patching. Since `_apply_step` is now a method, tests need a coordinator instance. Create a minimal one with mocks.

**7f: `tests/test_execute_fade_colors.py`**

- Change `from custom_components.fado import _execute_fade` to coordinator import
- Tests use `mock_hass` (MagicMock). Create coordinator with mock_hass:
  ```python
  coordinator = FadeCoordinator(mock_hass, mock_entry, mock_store, {})
  await coordinator._execute_fade(entity_id, params, delay_ms, cancel_event)
  ```

**7g: `tests/test_fade_execution.py`**

- Late imports of `_execute_fade` need updating to coordinator
- Tests that use `hass.data[DOMAIN]` dict access need updating to coordinator attribute access

**7h: `tests/test_expected_state_colors.py`**

- `from custom_components.fado import ExpectedState` — This import should still work if `coordinator.py` re-exports it, or change to import from `expected_state.py` directly:
  ```python
  from custom_components.fado.expected_state import ExpectedState
  ```

**7i: `tests/test_capability_filtering.py`**

- `from custom_components.fado import _can_apply_fade_params` — This is now a module-level function in `coordinator.py`:
  ```python
  from custom_components.fado.coordinator import _can_apply_fade_params
  ```

**7j: `tests/test_color_temp_bounds.py`**

- `from custom_components.fado import DOMAIN` — fine (still in const)
- Late import of `_execute_fade` needs updating to coordinator

**7k: `tests/test_mireds_to_hs_fade.py`**

- `from custom_components.fado import _execute_fade` — update to coordinator

**7l: `tests/test_restore_intended_colors.py`**

- Imports from `custom_components.fado` — update to coordinator

**7m: `tests/test_manual_intervention_colors.py`**

- Imports from `custom_components.fado` — update to coordinator

**7n: `tests/test_notifications.py`**

- `from custom_components.fado import async_setup_entry` — this stays since `async_setup_entry` is still in `__init__.py`

**7o: Other test files that access `hass.data[DOMAIN]` as a dict**

Files: `test_services.py`, `test_websocket_api.py`, `test_autoconfigure.py`, `test_brightness_restoration.py`

These set up `hass.data[DOMAIN]` as a dict (e.g., `hass.data[DOMAIN] = {"data": {...}, "store": ...}`). They need to create a `FadeCoordinator` instance instead. The `init_integration` fixture already handles this (since `async_setup_entry` now creates coordinator). For tests that use `hass_with_storage` or manually set up `hass.data[DOMAIN]`, create a coordinator:

```python
from unittest.mock import MagicMock
from custom_components.fado.coordinator import FadeCoordinator

mock_entry = MagicMock()
mock_entry.options = {}
coordinator = FadeCoordinator(hass, mock_entry, mock_store, storage_data)
hass.data[DOMAIN] = coordinator
```

Then change all `hass.data[DOMAIN]["data"]` to `hass.data[DOMAIN].data`, etc.

**Step: Commit all test changes**

```bash
git add tests/
git commit -m "test: update all tests for FadeCoordinator refactor"
```

---

### Task 8: Run tests and fix issues

**Step 1: Run the full test suite**

```bash
python -m pytest tests/ -x -v 2>&1 | head -100
```

Expected: Some failures from missed import updates or subtle behavior differences.

**Step 2: Fix each failure**

Common issues to watch for:
1. **Import errors**: Function moved but test imports old path
2. **AttributeError**: `hass.data[DOMAIN]` is now coordinator, not dict — any remaining `["data"]` access
3. **Coordinator not found**: Tests that don't use `init_integration` and manually set up `hass.data[DOMAIN]` as a dict
4. **`_cancel_and_wait_for_fade` wait condition**: Now checks `entity.active_task is None` instead of `entity_id not in ACTIVE_FADES`
5. **`_restore_intended_state` finally block**: Now sets `entity.restore_task = None` instead of `RESTORE_TASKS.pop()`

**Step 3: Run tests again until all pass**

```bash
python -m pytest tests/ -x -v
```

Expected: 582 tests pass.

**Step 4: Commit fixes**

```bash
git add -A
git commit -m "fix: address test failures from coordinator refactor"
```

---

### Task 9: Lint and type check

**Step 1: Run ruff**

```bash
ruff check . && ruff format .
```

Fix any issues.

**Step 2: Run pyright**

```bash
npx pyright
```

Fix any type errors.

**Step 3: Commit lint fixes**

```bash
git add -A
git commit -m "style: fix lint and type errors"
```

---

### Task 10: Deploy and create PR

**Step 1: Copy to HA config directory**

```bash
rm -Rf /workspaces/homeassistant-core/config/custom_components/fado && cp -r /workspaces/ha-fado/custom_components/fado /workspaces/homeassistant-core/config/custom_components/
```

**Step 2: Run tests one final time**

```bash
python -m pytest tests/ -q
```

Expected: 582 passed

**Step 3: Create PR**

```bash
git push -u origin feature/global-state-to-hass-data
gh pr create --title "Refactor: move global state to FadeCoordinator on hass.data" --body "$(cat <<'EOF'
## Summary
- Created `FadeCoordinator` class in new `coordinator.py` that holds all fade state as instance variables
- Replaced 6 module-level global dicts (`ACTIVE_FADES`, `FADE_CANCEL_EVENTS`, etc.) with per-entity `EntityFadeState` dataclass
- Slimmed `__init__.py` from ~1274 lines to ~100 lines (setup/teardown only)
- Updated `websocket_api.py`, `autoconfigure.py`, `notifications.py` to use coordinator
- Updated all 582 tests

## Test plan
- [ ] All 582 existing tests pass (no behavior changes)
- [ ] `ruff check .` and `ruff format .` clean
- [ ] `npx pyright` clean
- [ ] Integration loads and unloads correctly in HA
- [ ] Fades work correctly after reload

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
