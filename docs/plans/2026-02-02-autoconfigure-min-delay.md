# Autoconfigure min_delay_ms Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Automatically detect optimal min_delay_ms for each light by measuring response times

**Architecture:** WebSocket API streams test results to frontend panel with hierarchical selection UI

**Tech Stack:** Python backend (asyncio, WebSocket), LitElement frontend

---

## Constants

```python
# const.py additions
AUTOCONFIGURE_ITERATIONS = 10
AUTOCONFIGURE_TIMEOUT_S = 5
AUTOCONFIGURE_MAX_PARALLEL = 5
```

---

## Task 1: Add constants to const.py

**Files:**
- Modify: `custom_components/fade_lights/const.py`

Add three new constants for the autoconfigure feature:
- `AUTOCONFIGURE_ITERATIONS = 10`
- `AUTOCONFIGURE_TIMEOUT_S = 5`
- `AUTOCONFIGURE_MAX_PARALLEL = 5`

---

## Task 2: Implement single light test function

**Files:**
- Create: `custom_components/fade_lights/autoconfigure.py`
- Test: `tests/test_autoconfigure.py`

**Function:** `async def test_light_delay(hass, entity_id) -> dict`

**Algorithm:**
1. Capture original state (brightness, on/off state)
2. Set up state change listener for this entity_id using asyncio.Event
3. For i in 1..AUTOCONFIGURE_ITERATIONS:
   a. Record start_time using time.monotonic()
   b. Call light.turn_on with brightness = 1 if i%2==1 else 255
   c. Wait for state_changed event with AUTOCONFIGURE_TIMEOUT_S timeout
   d. If timeout: retry once, if still fails return error
   e. Record elapsed_ms = (now - start_time) * 1000
   f. Append to timings[]
4. Remove state listener
5. Calculate average = sum(timings) / len(timings)
6. Round up to nearest 10ms: `math.ceil(average / 10) * 10`
7. Restore original state
8. Save min_delay_ms to storage via async_save_light_config
9. Return `{"entity_id": entity_id, "min_delay_ms": result}`

**Error return:** `{"entity_id": entity_id, "error": "Timeout after retry"}`

---

## Task 3: Implement WebSocket autoconfigure handler

**Files:**
- Modify: `custom_components/fade_lights/websocket_api.py`

**Function:** `ws_autoconfigure(hass, connection, msg)`

**Input schema:**
```python
{
    "type": "fade_lights/autoconfigure",
    vol.Required("entity_ids"): [str],
}
```

**Flow:**
1. Expand groups to individual entity IDs (reuse _expand_light_groups from __init__)
2. Filter out excluded lights
3. Create semaphore with AUTOCONFIGURE_MAX_PARALLEL permits
4. For each entity_id, spawn task that:
   a. Sends `{"type": "event", "event": {"type": "started", "entity_id": ...}}`
   b. Acquires semaphore
   c. Runs test_light_delay()
   d. Sends result event or error event
   e. Releases semaphore
5. Use asyncio.gather() to run all tasks
6. Send final `{"type": "result"}` to close subscription

**Event types:**
- `{"type": "event", "event": {"type": "started", "entity_id": "..."}}`
- `{"type": "event", "event": {"type": "result", "entity_id": "...", "min_delay_ms": 150}}`
- `{"type": "event", "event": {"type": "error", "entity_id": "...", "message": "..."}}`

---

## Task 4: Add Configure checkbox column to panel

**Files:**
- Modify: `custom_components/fade_lights/frontend/panel.js`

**Changes:**
1. Add new property: `_configureChecked: { type: Object }` - Set of entity_ids
2. Initialize in constructor: `this._configureChecked = new Set()`
3. After fetching lights, populate _configureChecked with lights where min_delay_ms is null
4. Add new table column "Configure" with checkbox
5. Checkbox checked state: `this._configureChecked.has(light.entity_id)`
6. On change: toggle entity_id in _configureChecked Set

**Column widths update:**
- col-light: auto
- col-delay: 120px
- col-exclude: 80px
- col-configure: 80px (new)

---

## Task 5: Add hierarchical checkboxes for floors and areas

**Files:**
- Modify: `custom_components/fade_lights/frontend/panel.js`

**Floor/Area header checkbox logic:**
1. Add checkbox in floor header (before chevron)
2. Add checkbox in area header (before chevron)
3. Floor checkbox state:
   - Checked if ALL lights in ALL areas are checked
   - Unchecked if NONE are checked
   - Indeterminate if SOME are checked
4. Area checkbox state: same logic for lights in that area
5. Clicking floor checkbox: toggle all descendant lights
6. Clicking area checkbox: toggle all lights in that area

**Helper methods:**
- `_getFloorLightIds(floor)` - returns all entity_ids in floor
- `_getAreaLightIds(area)` - returns all entity_ids in area
- `_getCheckboxState(entityIds)` - returns "all" | "none" | "some"

---

## Task 6: Add Autoconfigure button

**Files:**
- Modify: `custom_components/fade_lights/frontend/panel.js`

**Changes:**
1. Add button in header area: `<mwc-button>Autoconfigure (N)</mwc-button>`
2. N = count of checked lights in _configureChecked
3. Disabled when N === 0
4. Disabled when testing in progress
5. Add new properties:
   - `_testing: { type: Object }` - Set of entity_ids currently testing
   - `_testErrors: { type: Object }` - Map of entity_id -> error message

**Button states:**
- Normal: "Autoconfigure (5)" - enabled
- No selection: "Autoconfigure" - disabled, grayed
- Testing: "Testing... (3/5)" - disabled, shows count

---

## Task 7: Implement autoconfigure button click handler

**Files:**
- Modify: `custom_components/fade_lights/frontend/panel.js`

**Method:** `async _runAutoconfigure()`

**Flow:**
1. Get list of checked entity_ids from _configureChecked
2. Clear _testErrors
3. Subscribe to WebSocket: `this.hass.connection.subscribeMessage(callback, {type: "fade_lights/autoconfigure", entity_ids: [...]})`
4. Callback handles events:
   - "started": add entity_id to _testing Set
   - "result": remove from _testing, update local light data with new min_delay_ms, uncheck from _configureChecked
   - "error": remove from _testing, add to _testErrors Map
5. On final result (subscription closes): clear _testing

---

## Task 8: Update light row rendering for test state

**Files:**
- Modify: `custom_components/fade_lights/frontend/panel.js`

**Changes to _renderLight():**
1. If entity_id in _testing:
   - Replace min_delay input with `<ha-circular-progress active></ha-circular-progress>`
   - Disable configure checkbox
2. If entity_id in _testErrors:
   - Show error message in red below the input
3. Normal state: show input as before

**CSS additions:**
- `.testing-spinner` styles
- `.test-error` styles (red text, small font)

---

## Task 9: Write tests for autoconfigure backend

**Files:**
- Create/Modify: `tests/test_autoconfigure.py`

**Test cases:**
1. `test_light_delay_measures_response_time` - verify timing measurement
2. `test_light_delay_handles_timeout_with_retry` - verify retry on first timeout
3. `test_light_delay_fails_after_retry` - verify error return after second timeout
4. `test_light_delay_restores_original_state` - verify light restored after test
5. `test_light_delay_restores_off_light` - verify off light turned back off
6. `test_autoconfigure_respects_parallel_limit` - verify semaphore limits concurrency
7. `test_autoconfigure_streams_results` - verify events sent as lights complete

---

## Task 10: Integration test and cleanup

**Files:**
- All files from previous tasks

**Steps:**
1. Run full test suite: `pytest tests/ -v`
2. Verify no regressions in existing tests
3. Test manually in browser:
   - Check/uncheck lights
   - Click Autoconfigure
   - Verify spinner shows
   - Verify results update
   - Verify errors display
4. Copy to HA installation and verify in real environment

---

## Summary

**Backend:**
- `const.py`: 3 new constants
- `autoconfigure.py`: test_light_delay() function
- `websocket_api.py`: ws_autoconfigure handler

**Frontend:**
- New "Configure" column with checkboxes
- Hierarchical floor/area checkboxes
- Autoconfigure button with count
- Testing state with spinners
- Error display

**Data flow:**
1. User checks lights to configure
2. User clicks Autoconfigure
3. Frontend subscribes to WebSocket
4. Backend tests lights (5 parallel max)
5. Backend streams results as events
6. Frontend updates UI in real-time
7. Results saved to storage automatically
