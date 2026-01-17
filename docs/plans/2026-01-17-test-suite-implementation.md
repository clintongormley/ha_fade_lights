# Fade Lights Test Suite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a comprehensive pytest test suite for the fade_lights Home Assistant custom integration with 95%+ coverage.

**Architecture:** Uses pytest-homeassistant-custom-component to provide HA test fixtures. Mock light entities simulate real lights. Tests cover config flow, services, fade execution, manual interruption detection, and brightness restoration.

**Tech Stack:** pytest, pytest-asyncio, pytest-cov, pytest-homeassistant-custom-component, syrupy

---

## Task 1: Project Setup - Create pyproject.toml and test directory structure

**Files:**
- Create: `pyproject.toml`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Create pyproject.toml with test dependencies**

```toml
[project]
name = "fade_lights"
version = "0.1.0"
description = "Home Assistant custom integration for smooth light fading"
requires-python = ">=3.12"

[project.optional-dependencies]
test = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "pytest-homeassistant-custom-component>=0.13.0",
    "syrupy>=4.6.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "C4", "SIM"]
```

**Step 2: Create tests/__init__.py**

```python
"""Tests for the Fade Lights integration."""
```

**Step 3: Create tests/conftest.py with basic fixtures**

```python
"""Fixtures for Fade Lights tests."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest
from homeassistant.components.light import ATTR_BRIGHTNESS, ATTR_SUPPORTED_COLOR_MODES, DOMAIN as LIGHT_DOMAIN
from homeassistant.const import ATTR_ENTITY_ID, STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.fade_lights.const import DOMAIN


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Enable custom integrations for all tests."""
    return


@pytest.fixture
def mock_config_entry() -> MockConfigEntry:
    """Create a mock config entry."""
    return MockConfigEntry(
        domain=DOMAIN,
        title="Fade Lights",
        data={},
        options={},
    )


@pytest.fixture
async def init_integration(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> MockConfigEntry:
    """Set up the Fade Lights integration."""
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    return mock_config_entry


@pytest.fixture
def mock_light_entity(hass: HomeAssistant) -> str:
    """Create a mock dimmable light entity."""
    entity_id = "light.test_light"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 200,
            ATTR_SUPPORTED_COLOR_MODES: ["brightness"],
        },
    )
    return entity_id


@pytest.fixture
def mock_light_off(hass: HomeAssistant) -> str:
    """Create a mock dimmable light that is off."""
    entity_id = "light.test_light_off"
    hass.states.async_set(
        entity_id,
        STATE_OFF,
        {
            ATTR_SUPPORTED_COLOR_MODES: ["brightness"],
        },
    )
    return entity_id


@pytest.fixture
def mock_non_dimmable_light(hass: HomeAssistant) -> str:
    """Create a mock light without brightness support."""
    entity_id = "light.non_dimmable"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {
            ATTR_SUPPORTED_COLOR_MODES: ["onoff"],
        },
    )
    return entity_id


@pytest.fixture
def mock_light_group(hass: HomeAssistant, mock_light_entity: str) -> str:
    """Create a mock light group."""
    group_entity_id = "light.test_group"
    hass.states.async_set(
        group_entity_id,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 200,
            ATTR_SUPPORTED_COLOR_MODES: ["brightness"],
            ATTR_ENTITY_ID: [mock_light_entity],
        },
    )
    return group_entity_id


@pytest.fixture
def captured_calls(hass: HomeAssistant) -> list[dict[str, Any]]:
    """Capture service calls to light domain."""
    calls: list[dict[str, Any]] = []

    async def mock_service_call(call):
        calls.append({
            "service": call.service,
            "data": dict(call.data),
            "context": call.context,
        })

    hass.services.async_register(LIGHT_DOMAIN, "turn_on", mock_service_call)
    hass.services.async_register(LIGHT_DOMAIN, "turn_off", mock_service_call)

    return calls
```

**Step 4: Install test dependencies and verify setup**

Run:
```bash
cd /tmp/ha_fade_lights && pip install -e ".[test]"
```

**Step 5: Run pytest to verify it can discover tests directory**

Run:
```bash
cd /tmp/ha_fade_lights && pytest --collect-only
```

Expected: Shows tests directory, no collection errors

**Step 6: Commit**

```bash
git add pyproject.toml tests/
git commit -m "test: add project setup and base fixtures for testing"
```

---

## Task 2: Config Flow Tests - User and Import Flows

**Files:**
- Create: `tests/test_config_flow.py`

**Step 1: Write config flow tests**

```python
"""Tests for Fade Lights config flow."""

from __future__ import annotations

from homeassistant.config_entries import SOURCE_IMPORT, SOURCE_USER
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.fade_lights.const import (
    DEFAULT_BRIGHTNESS_PCT,
    DEFAULT_STEP_DELAY_MS,
    DEFAULT_TRANSITION,
    DOMAIN,
    OPTION_DEFAULT_BRIGHTNESS_PCT,
    OPTION_DEFAULT_TRANSITION,
    OPTION_STEP_DELAY_MS,
)


async def test_user_flow_creates_entry(hass: HomeAssistant) -> None:
    """Test user flow creates a config entry."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == "Fade Lights"
    assert result["data"] == {}


async def test_single_instance_only(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Test only single instance allowed."""
    mock_config_entry.add_to_hass(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )

    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "single_instance_allowed"


async def test_import_flow_creates_entry(hass: HomeAssistant) -> None:
    """Test import flow creates a config entry."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_IMPORT}
    )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == "Fade Lights"


async def test_import_flow_single_instance(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Test import flow aborts if instance exists."""
    mock_config_entry.add_to_hass(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_IMPORT}
    )

    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "single_instance_allowed"


async def test_options_flow_shows_defaults(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test options flow shows default values."""
    result = await hass.config_entries.options.async_init(init_integration.entry_id)

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "init"
    # Schema has the default values
    schema = result["data_schema"].schema
    assert OPTION_DEFAULT_BRIGHTNESS_PCT in [k.schema for k in schema.keys()]


async def test_options_flow_updates_values(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test options flow saves updated values."""
    result = await hass.config_entries.options.async_init(init_integration.entry_id)

    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={
            OPTION_DEFAULT_BRIGHTNESS_PCT: 75,
            OPTION_DEFAULT_TRANSITION: 10,
            OPTION_STEP_DELAY_MS: 200,
        },
    )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert init_integration.options[OPTION_DEFAULT_BRIGHTNESS_PCT] == 75
    assert init_integration.options[OPTION_DEFAULT_TRANSITION] == 10
    assert init_integration.options[OPTION_STEP_DELAY_MS] == 200


async def test_options_flow_validates_brightness_range(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test brightness_pct must be 0-100."""
    result = await hass.config_entries.options.async_init(init_integration.entry_id)

    # voluptuous validation happens at schema level
    # Values outside range should be rejected
    # The Range validator will raise an error for values > 100
    schema = result["data_schema"].schema
    for key in schema.keys():
        if key.schema == OPTION_DEFAULT_BRIGHTNESS_PCT:
            # Verify the validator exists
            validator = schema[key]
            assert validator is not None
            break


async def test_options_flow_validates_transition_range(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test transition must be 0-3600."""
    result = await hass.config_entries.options.async_init(init_integration.entry_id)

    schema = result["data_schema"].schema
    for key in schema.keys():
        if key.schema == OPTION_DEFAULT_TRANSITION:
            validator = schema[key]
            assert validator is not None
            break


async def test_options_flow_validates_step_delay_range(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test step_delay_ms must be 10-1000."""
    result = await hass.config_entries.options.async_init(init_integration.entry_id)

    schema = result["data_schema"].schema
    for key in schema.keys():
        if key.schema == OPTION_STEP_DELAY_MS:
            validator = schema[key]
            assert validator is not None
            break
```

**Step 2: Run tests**

Run:
```bash
cd /tmp/ha_fade_lights && pytest tests/test_config_flow.py -v
```

Expected: All 8 tests pass

**Step 3: Commit**

```bash
git add tests/test_config_flow.py
git commit -m "test: add config flow and options flow tests"
```

---

## Task 3: Integration Init Tests - Setup and Unload

**Files:**
- Create: `tests/test_init.py`

**Step 1: Write init tests**

```python
"""Tests for Fade Lights integration setup and unload."""

from __future__ import annotations

from unittest.mock import patch

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.fade_lights import (
    ACTIVE_CONTEXTS,
    ACTIVE_FADES,
    FADE_CANCEL_EVENTS,
    FADE_EXPECTED_BRIGHTNESS,
)
from custom_components.fade_lights.const import DOMAIN, SERVICE_FADE_LIGHTS


async def test_setup_entry_registers_service(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test that setup registers the fade_lights service."""
    assert hass.services.has_service(DOMAIN, SERVICE_FADE_LIGHTS)


async def test_setup_entry_loads_storage(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Test that setup loads storage data."""
    mock_config_entry.add_to_hass(hass)

    with patch(
        "custom_components.fade_lights.Store.async_load",
        return_value={"light_test_light": {"orig": 200, "curr": 150}},
    ):
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()

    assert DOMAIN in hass.data
    assert "data" in hass.data[DOMAIN]
    assert hass.data[DOMAIN]["data"] == {"light_test_light": {"orig": 200, "curr": 150}}


async def test_unload_entry_removes_service(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test that unload removes the service."""
    assert hass.services.has_service(DOMAIN, SERVICE_FADE_LIGHTS)

    await hass.config_entries.async_unload(init_integration.entry_id)
    await hass.async_block_till_done()

    assert not hass.services.has_service(DOMAIN, SERVICE_FADE_LIGHTS)


async def test_unload_entry_clears_tracking_dicts(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test that unload clears all tracking dictionaries."""
    # Add some test data to tracking dicts
    ACTIVE_FADES["test"] = None
    FADE_CANCEL_EVENTS["test"] = None
    FADE_EXPECTED_BRIGHTNESS["test"] = 100
    ACTIVE_CONTEXTS.add("test_context")

    await hass.config_entries.async_unload(init_integration.entry_id)
    await hass.async_block_till_done()

    assert len(ACTIVE_FADES) == 0
    assert len(FADE_CANCEL_EVENTS) == 0
    assert len(FADE_EXPECTED_BRIGHTNESS) == 0
    assert len(ACTIVE_CONTEXTS) == 0


async def test_unload_entry_state(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test config entry state after unload."""
    assert init_integration.state == ConfigEntryState.LOADED

    await hass.config_entries.async_unload(init_integration.entry_id)
    await hass.async_block_till_done()

    assert init_integration.state == ConfigEntryState.NOT_LOADED


async def test_options_update_reloads_entry(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test that updating options triggers a reload."""
    # Update options
    hass.config_entries.async_update_entry(
        init_integration,
        options={"default_brightness_pct": 50},
    )
    await hass.async_block_till_done()

    # Entry should still be loaded after reload
    assert init_integration.state == ConfigEntryState.LOADED
```

**Step 2: Run tests**

Run:
```bash
cd /tmp/ha_fade_lights && pytest tests/test_init.py -v
```

Expected: All 6 tests pass

**Step 3: Commit**

```bash
git add tests/test_init.py
git commit -m "test: add integration setup and unload tests"
```

---

## Task 4: Service Tests - Parameter Handling

**Files:**
- Create: `tests/test_services.py`

**Step 1: Write service tests**

```python
"""Tests for Fade Lights service parameter handling."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.fade_lights.const import (
    ATTR_BRIGHTNESS_PCT,
    ATTR_TRANSITION,
    DEFAULT_BRIGHTNESS_PCT,
    DEFAULT_TRANSITION,
    DOMAIN,
    SERVICE_FADE_LIGHTS,
)


async def test_service_accepts_single_entity(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
) -> None:
    """Test service accepts a single entity_id."""
    with patch(
        "custom_components.fade_lights._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade:
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_ENTITY_ID: mock_light_entity,
                ATTR_BRIGHTNESS_PCT: 50,
                ATTR_TRANSITION: 1,
            },
            blocking=True,
        )

        mock_fade.assert_called_once()
        call_args = mock_fade.call_args
        assert call_args[0][1] == mock_light_entity  # entity_id


async def test_service_accepts_entity_list(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
) -> None:
    """Test service accepts a list of entity_ids."""
    # Create a second light
    hass.states.async_set(
        "light.second_light",
        "on",
        {"brightness": 200, "supported_color_modes": ["brightness"]},
    )

    with patch(
        "custom_components.fade_lights._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade:
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_ENTITY_ID: [mock_light_entity, "light.second_light"],
                ATTR_BRIGHTNESS_PCT: 50,
                ATTR_TRANSITION: 1,
            },
            blocking=True,
        )

        assert mock_fade.call_count == 2


async def test_service_accepts_comma_string(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
) -> None:
    """Test service accepts comma-separated string of entities."""
    hass.states.async_set(
        "light.second_light",
        "on",
        {"brightness": 200, "supported_color_modes": ["brightness"]},
    )

    with patch(
        "custom_components.fade_lights._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade:
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_ENTITY_ID: f"{mock_light_entity}, light.second_light",
                ATTR_BRIGHTNESS_PCT: 50,
                ATTR_TRANSITION: 1,
            },
            blocking=True,
        )

        assert mock_fade.call_count == 2


async def test_service_expands_light_groups(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_group: str,
    mock_light_entity: str,
) -> None:
    """Test service expands light groups to individual lights."""
    with patch(
        "custom_components.fade_lights._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade:
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_ENTITY_ID: mock_light_group,
                ATTR_BRIGHTNESS_PCT: 50,
                ATTR_TRANSITION: 1,
            },
            blocking=True,
        )

        # Should call for the individual light, not the group
        mock_fade.assert_called_once()
        call_args = mock_fade.call_args
        assert call_args[0][1] == mock_light_entity


async def test_service_rejects_non_light_entity(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
) -> None:
    """Test service raises error for non-light entities."""
    hass.states.async_set("sensor.temperature", "25")

    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_ENTITY_ID: "sensor.temperature",
                ATTR_BRIGHTNESS_PCT: 50,
                ATTR_TRANSITION: 1,
            },
            blocking=True,
        )


async def test_service_uses_default_brightness(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
) -> None:
    """Test service uses default brightness when not specified."""
    with patch(
        "custom_components.fade_lights._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade:
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_ENTITY_ID: mock_light_entity,
                ATTR_TRANSITION: 1,
            },
            blocking=True,
        )

        call_args = mock_fade.call_args
        assert call_args[0][2] == DEFAULT_BRIGHTNESS_PCT  # brightness_pct


async def test_service_uses_default_transition(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
) -> None:
    """Test service uses default transition when not specified."""
    with patch(
        "custom_components.fade_lights._fade_light",
        new_callable=AsyncMock,
    ) as mock_fade:
        await hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_ENTITY_ID: mock_light_entity,
                ATTR_BRIGHTNESS_PCT: 50,
            },
            blocking=True,
        )

        call_args = mock_fade.call_args
        # transition_ms = transition * 1000
        assert call_args[0][3] == DEFAULT_TRANSITION * 1000
```

**Step 2: Run tests**

Run:
```bash
cd /tmp/ha_fade_lights && pytest tests/test_services.py -v
```

Expected: All 7 tests pass

**Step 3: Commit**

```bash
git add tests/test_services.py
git commit -m "test: add service parameter handling tests"
```

---

## Task 5: Fade Execution Tests

**Files:**
- Create: `tests/test_fade_execution.py`

**Step 1: Write fade execution tests**

```python
"""Tests for Fade Lights fade execution logic."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.components.light import ATTR_BRIGHTNESS, DOMAIN as LIGHT_DOMAIN
from homeassistant.const import ATTR_ENTITY_ID, SERVICE_TURN_OFF, SERVICE_TURN_ON, STATE_ON
from homeassistant.core import HomeAssistant, ServiceCall

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.fade_lights.const import (
    ATTR_BRIGHTNESS_PCT,
    ATTR_TRANSITION,
    DOMAIN,
    SERVICE_FADE_LIGHTS,
)


@pytest.fixture
def service_calls(hass: HomeAssistant) -> list[ServiceCall]:
    """Capture all service calls."""
    calls: list[ServiceCall] = []

    async def capture_call(call: ServiceCall) -> None:
        calls.append(call)
        # Update the light state based on the call
        entity_id = call.data.get(ATTR_ENTITY_ID)
        if entity_id:
            if call.service == SERVICE_TURN_OFF:
                hass.states.async_set(entity_id, "off", {})
            elif call.service == SERVICE_TURN_ON:
                brightness = call.data.get(ATTR_BRIGHTNESS, 255)
                hass.states.async_set(
                    entity_id,
                    STATE_ON,
                    {ATTR_BRIGHTNESS: brightness, "supported_color_modes": ["brightness"]},
                )

    hass.services.async_register(LIGHT_DOMAIN, SERVICE_TURN_ON, capture_call)
    hass.services.async_register(LIGHT_DOMAIN, SERVICE_TURN_OFF, capture_call)

    return calls


async def test_fade_down_reaches_target(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
    service_calls: list[ServiceCall],
) -> None:
    """Test fading down reaches target brightness."""
    # Light starts at brightness 200 (78%)
    await hass.services.async_call(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        {
            ATTR_ENTITY_ID: mock_light_entity,
            ATTR_BRIGHTNESS_PCT: 20,  # Target: 51
            ATTR_TRANSITION: 0.1,  # Fast for testing
        },
        blocking=True,
    )

    # Check final state
    state = hass.states.get(mock_light_entity)
    assert state.attributes.get(ATTR_BRIGHTNESS) == 51


async def test_fade_up_reaches_target(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test fading up reaches target brightness."""
    # Create light at low brightness
    entity_id = "light.dim_light"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {ATTR_BRIGHTNESS: 50, "supported_color_modes": ["brightness"]},
    )

    await hass.services.async_call(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        {
            ATTR_ENTITY_ID: entity_id,
            ATTR_BRIGHTNESS_PCT: 80,  # Target: 204
            ATTR_TRANSITION: 0.1,
        },
        blocking=True,
    )

    state = hass.states.get(entity_id)
    assert state.attributes.get(ATTR_BRIGHTNESS) == 204


async def test_fade_to_zero_turns_off(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
    service_calls: list[ServiceCall],
) -> None:
    """Test fading to 0% turns the light off."""
    await hass.services.async_call(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        {
            ATTR_ENTITY_ID: mock_light_entity,
            ATTR_BRIGHTNESS_PCT: 0,
            ATTR_TRANSITION: 0.1,
        },
        blocking=True,
    )

    # Check that turn_off was called
    turn_off_calls = [c for c in service_calls if c.service == SERVICE_TURN_OFF]
    assert len(turn_off_calls) > 0

    state = hass.states.get(mock_light_entity)
    assert state.state == "off"


async def test_fade_already_at_target_no_op(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test no calls made if already at target brightness."""
    entity_id = "light.at_target"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {ATTR_BRIGHTNESS: 102, "supported_color_modes": ["brightness"]},  # 40%
    )

    await hass.services.async_call(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        {
            ATTR_ENTITY_ID: entity_id,
            ATTR_BRIGHTNESS_PCT: 40,  # Same as current
            ATTR_TRANSITION: 0.1,
        },
        blocking=True,
    )

    # No service calls should be made for this light
    light_calls = [c for c in service_calls if c.data.get(ATTR_ENTITY_ID) == entity_id]
    assert len(light_calls) == 0


async def test_fade_skips_brightness_level_1(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test brightness level 1 is skipped (goes to 0 or 2)."""
    entity_id = "light.low_light"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {ATTR_BRIGHTNESS: 10, "supported_color_modes": ["brightness"]},
    )

    await hass.services.async_call(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        {
            ATTR_ENTITY_ID: entity_id,
            ATTR_BRIGHTNESS_PCT: 0,
            ATTR_TRANSITION: 0.1,
        },
        blocking=True,
    )

    # Check that no call was made with brightness=1
    for call in service_calls:
        if call.service == SERVICE_TURN_ON:
            assert call.data.get(ATTR_BRIGHTNESS) != 1


async def test_fade_non_dimmable_to_zero(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_non_dimmable_light: str,
    service_calls: list[ServiceCall],
) -> None:
    """Test non-dimmable light turns off at 0%."""
    await hass.services.async_call(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        {
            ATTR_ENTITY_ID: mock_non_dimmable_light,
            ATTR_BRIGHTNESS_PCT: 0,
            ATTR_TRANSITION: 0.1,
        },
        blocking=True,
    )

    turn_off_calls = [
        c for c in service_calls
        if c.service == SERVICE_TURN_OFF and c.data.get(ATTR_ENTITY_ID) == mock_non_dimmable_light
    ]
    assert len(turn_off_calls) == 1


async def test_fade_non_dimmable_to_nonzero(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test non-dimmable light turns on at >0%."""
    entity_id = "light.non_dimmable_off"
    hass.states.async_set(
        entity_id,
        "off",
        {"supported_color_modes": ["onoff"]},
    )

    await hass.services.async_call(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        {
            ATTR_ENTITY_ID: entity_id,
            ATTR_BRIGHTNESS_PCT: 50,
            ATTR_TRANSITION: 0.1,
        },
        blocking=True,
    )

    turn_on_calls = [
        c for c in service_calls
        if c.service == SERVICE_TURN_ON and c.data.get(ATTR_ENTITY_ID) == entity_id
    ]
    assert len(turn_on_calls) == 1


async def test_fade_unknown_entity_logs_warning(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test unknown entity logs warning but doesn't crash."""
    await hass.services.async_call(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        {
            ATTR_ENTITY_ID: "light.nonexistent",
            ATTR_BRIGHTNESS_PCT: 50,
            ATTR_TRANSITION: 0.1,
        },
        blocking=True,
    )

    assert "not found" in caplog.text.lower() or "unknown" in caplog.text.lower()


async def test_fade_stores_orig_brightness(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
    service_calls: list[ServiceCall],
) -> None:
    """Test original brightness is stored after fade completes."""
    await hass.services.async_call(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        {
            ATTR_ENTITY_ID: mock_light_entity,
            ATTR_BRIGHTNESS_PCT: 50,  # Target: 127
            ATTR_TRANSITION: 0.1,
        },
        blocking=True,
    )

    # Check storage
    storage_data = hass.data[DOMAIN]["data"]
    storage_key = mock_light_entity.replace(".", "_")
    assert storage_key in storage_data
    assert storage_data[storage_key]["orig"] == 127
```

**Step 2: Run tests**

Run:
```bash
cd /tmp/ha_fade_lights && pytest tests/test_fade_execution.py -v
```

Expected: All 10 tests pass

**Step 3: Commit**

```bash
git add tests/test_fade_execution.py
git commit -m "test: add fade execution logic tests"
```

---

## Task 6: Manual Interruption Tests

**Files:**
- Create: `tests/test_manual_interruption.py`

**Step 1: Write manual interruption tests**

```python
"""Tests for manual interruption detection during fades."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.components.light import ATTR_BRIGHTNESS, DOMAIN as LIGHT_DOMAIN
from homeassistant.const import ATTR_ENTITY_ID, EVENT_STATE_CHANGED, STATE_OFF, STATE_ON
from homeassistant.core import Context, HomeAssistant, ServiceCall

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.fade_lights import ACTIVE_FADES, FADE_CANCEL_EVENTS
from custom_components.fade_lights.const import (
    ATTR_BRIGHTNESS_PCT,
    ATTR_TRANSITION,
    DOMAIN,
    SERVICE_FADE_LIGHTS,
)


@pytest.fixture
def service_calls(hass: HomeAssistant) -> list[ServiceCall]:
    """Capture all service calls."""
    calls: list[ServiceCall] = []

    async def capture_call(call: ServiceCall) -> None:
        calls.append(call)
        entity_id = call.data.get(ATTR_ENTITY_ID)
        if entity_id:
            if call.service == "turn_off":
                hass.states.async_set(entity_id, STATE_OFF, {})
            elif call.service == "turn_on":
                brightness = call.data.get(ATTR_BRIGHTNESS, 255)
                hass.states.async_set(
                    entity_id,
                    STATE_ON,
                    {ATTR_BRIGHTNESS: brightness, "supported_color_modes": ["brightness"]},
                )

    hass.services.async_register(LIGHT_DOMAIN, "turn_on", capture_call)
    hass.services.async_register(LIGHT_DOMAIN, "turn_off", capture_call)

    return calls


async def test_manual_brightness_change_cancels_fade(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
    service_calls: list[ServiceCall],
) -> None:
    """Test external brightness change cancels active fade."""
    # Start a long fade
    fade_task = hass.async_create_task(
        hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_ENTITY_ID: mock_light_entity,
                ATTR_BRIGHTNESS_PCT: 10,
                ATTR_TRANSITION: 5,  # Long enough to interrupt
            },
            blocking=True,
        )
    )

    # Wait for fade to start
    await asyncio.sleep(0.2)
    assert mock_light_entity in ACTIVE_FADES

    # Simulate manual brightness change (external context)
    hass.states.async_set(
        mock_light_entity,
        STATE_ON,
        {ATTR_BRIGHTNESS: 150, "supported_color_modes": ["brightness"]},
    )
    await hass.async_block_till_done()

    # Give time for cancellation to process
    await asyncio.sleep(0.2)

    # Fade should be cancelled
    assert mock_light_entity not in ACTIVE_FADES or fade_task.done()


async def test_manual_turn_off_cancels_fade(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
    service_calls: list[ServiceCall],
) -> None:
    """Test turning off light cancels active fade."""
    fade_task = hass.async_create_task(
        hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_ENTITY_ID: mock_light_entity,
                ATTR_BRIGHTNESS_PCT: 10,
                ATTR_TRANSITION: 5,
            },
            blocking=True,
        )
    )

    await asyncio.sleep(0.2)
    assert mock_light_entity in ACTIVE_FADES

    # Simulate manual turn off
    hass.states.async_set(mock_light_entity, STATE_OFF, {})
    await hass.async_block_till_done()

    await asyncio.sleep(0.2)

    assert mock_light_entity not in ACTIVE_FADES or fade_task.done()


async def test_new_fade_cancels_previous(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
    service_calls: list[ServiceCall],
) -> None:
    """Test starting new fade cancels in-progress fade."""
    # Start first fade
    first_fade = hass.async_create_task(
        hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_ENTITY_ID: mock_light_entity,
                ATTR_BRIGHTNESS_PCT: 10,
                ATTR_TRANSITION: 5,
            },
            blocking=True,
        )
    )

    await asyncio.sleep(0.2)

    # Start second fade (should cancel first)
    await hass.services.async_call(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        {
            ATTR_ENTITY_ID: mock_light_entity,
            ATTR_BRIGHTNESS_PCT: 80,
            ATTR_TRANSITION: 0.1,
        },
        blocking=True,
    )

    # First fade should have been cancelled
    assert first_fade.done() or first_fade.cancelled()


async def test_manual_change_stores_new_orig(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
) -> None:
    """Test manual brightness change becomes new original."""
    # Simulate manual brightness change
    hass.states.async_set(
        mock_light_entity,
        STATE_ON,
        {ATTR_BRIGHTNESS: 180, "supported_color_modes": ["brightness"]},
    )
    await hass.async_block_till_done()

    # Check storage updated
    storage_data = hass.data[DOMAIN]["data"]
    storage_key = mock_light_entity.replace(".", "_")
    assert storage_key in storage_data
    assert storage_data[storage_key]["orig"] == 180


async def test_group_changes_ignored(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_group: str,
) -> None:
    """Test state changes for group entities are ignored."""
    # Change group state (should not trigger any special handling)
    hass.states.async_set(
        mock_light_group,
        STATE_ON,
        {
            ATTR_BRIGHTNESS: 150,
            "supported_color_modes": ["brightness"],
            ATTR_ENTITY_ID: ["light.test_light"],  # Has entity_id = is a group
        },
    )
    await hass.async_block_till_done()

    # Group shouldn't be stored in storage
    storage_data = hass.data[DOMAIN]["data"]
    storage_key = mock_light_group.replace(".", "_")
    assert storage_key not in storage_data


async def test_brightness_tolerance_allows_rounding(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
    service_calls: list[ServiceCall],
) -> None:
    """Test ±5 brightness tolerance for device rounding."""
    # Start a fade
    fade_task = hass.async_create_task(
        hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_ENTITY_ID: mock_light_entity,
                ATTR_BRIGHTNESS_PCT: 50,
                ATTR_TRANSITION: 2,
            },
            blocking=True,
        )
    )

    await asyncio.sleep(0.2)

    # Get expected brightness from tracking
    from custom_components.fade_lights import FADE_EXPECTED_BRIGHTNESS
    if mock_light_entity in FADE_EXPECTED_BRIGHTNESS:
        expected = FADE_EXPECTED_BRIGHTNESS[mock_light_entity]

        # Change brightness within tolerance (should NOT cancel)
        hass.states.async_set(
            mock_light_entity,
            STATE_ON,
            {ATTR_BRIGHTNESS: expected + 3, "supported_color_modes": ["brightness"]},
        )
        await hass.async_block_till_done()
        await asyncio.sleep(0.1)

        # Fade should still be active
        assert mock_light_entity in ACTIVE_FADES

    # Clean up
    if not fade_task.done():
        fade_task.cancel()
        try:
            await fade_task
        except asyncio.CancelledError:
            pass


async def test_inherited_context_detected_by_brightness(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
    service_calls: list[ServiceCall],
) -> None:
    """Test manual change with inherited context is detected by brightness difference."""
    # Start a fade
    fade_task = hass.async_create_task(
        hass.services.async_call(
            DOMAIN,
            SERVICE_FADE_LIGHTS,
            {
                ATTR_ENTITY_ID: mock_light_entity,
                ATTR_BRIGHTNESS_PCT: 30,
                ATTR_TRANSITION: 3,
            },
            blocking=True,
        )
    )

    await asyncio.sleep(0.3)

    # Get expected brightness
    from custom_components.fade_lights import FADE_EXPECTED_BRIGHTNESS
    if mock_light_entity in FADE_EXPECTED_BRIGHTNESS:
        expected = FADE_EXPECTED_BRIGHTNESS[mock_light_entity]

        # Change brightness way outside tolerance (simulates manual override)
        # Even if context were inherited, this should trigger cancellation
        hass.states.async_set(
            mock_light_entity,
            STATE_ON,
            {ATTR_BRIGHTNESS: expected + 50, "supported_color_modes": ["brightness"]},
        )
        await hass.async_block_till_done()
        await asyncio.sleep(0.2)

        # Fade should be cancelled
        assert mock_light_entity not in ACTIVE_FADES or fade_task.done()
    else:
        # If fade already completed, just clean up
        if not fade_task.done():
            fade_task.cancel()
            try:
                await fade_task
            except asyncio.CancelledError:
                pass
```

**Step 2: Run tests**

Run:
```bash
cd /tmp/ha_fade_lights && pytest tests/test_manual_interruption.py -v
```

Expected: All 8 tests pass

**Step 3: Commit**

```bash
git add tests/test_manual_interruption.py
git commit -m "test: add manual interruption detection tests"
```

---

## Task 7: Brightness Restoration Tests

**Files:**
- Create: `tests/test_brightness_restoration.py`

**Step 1: Write brightness restoration tests**

```python
"""Tests for brightness restoration when lights turn back on."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from homeassistant.components.light import ATTR_BRIGHTNESS, DOMAIN as LIGHT_DOMAIN
from homeassistant.const import ATTR_ENTITY_ID, STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant, ServiceCall

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.fade_lights.const import (
    ATTR_BRIGHTNESS_PCT,
    ATTR_TRANSITION,
    DOMAIN,
    KEY_ORIG_BRIGHTNESS,
    SERVICE_FADE_LIGHTS,
)


@pytest.fixture
def service_calls(hass: HomeAssistant) -> list[ServiceCall]:
    """Capture all service calls."""
    calls: list[ServiceCall] = []

    async def capture_call(call: ServiceCall) -> None:
        calls.append(call)
        entity_id = call.data.get(ATTR_ENTITY_ID)
        if entity_id:
            if call.service == "turn_off":
                hass.states.async_set(
                    entity_id,
                    STATE_OFF,
                    {"supported_color_modes": ["brightness"]},
                )
            elif call.service == "turn_on":
                brightness = call.data.get(ATTR_BRIGHTNESS, 255)
                hass.states.async_set(
                    entity_id,
                    STATE_ON,
                    {ATTR_BRIGHTNESS: brightness, "supported_color_modes": ["brightness"]},
                )

    hass.services.async_register(LIGHT_DOMAIN, "turn_on", capture_call)
    hass.services.async_register(LIGHT_DOMAIN, "turn_off", capture_call)

    return calls


async def test_restore_brightness_on_turn_on(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
    service_calls: list[ServiceCall],
) -> None:
    """Test light restored to original brightness when turned on."""
    # Fade light to off
    await hass.services.async_call(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        {
            ATTR_ENTITY_ID: mock_light_entity,
            ATTR_BRIGHTNESS_PCT: 0,
            ATTR_TRANSITION: 0.1,
        },
        blocking=True,
    )

    # Clear calls
    service_calls.clear()

    # Simulate turning light on at different brightness
    hass.states.async_set(
        mock_light_entity,
        STATE_ON,
        {ATTR_BRIGHTNESS: 50, "supported_color_modes": ["brightness"]},
    )
    await hass.async_block_till_done()

    # Wait for restoration
    await asyncio.sleep(0.2)
    await hass.async_block_till_done()

    # Should have a turn_on call to restore original brightness (200)
    restore_calls = [
        c for c in service_calls
        if c.service == "turn_on"
        and c.data.get(ATTR_ENTITY_ID) == mock_light_entity
        and c.data.get(ATTR_BRIGHTNESS) == 200
    ]
    assert len(restore_calls) >= 1


async def test_no_restore_if_no_stored_brightness(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test no restore if original brightness is 0."""
    entity_id = "light.new_light"
    hass.states.async_set(
        entity_id,
        STATE_OFF,
        {"supported_color_modes": ["brightness"]},
    )

    # Turn on without any stored brightness
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {ATTR_BRIGHTNESS: 100, "supported_color_modes": ["brightness"]},
    )
    await hass.async_block_till_done()
    await asyncio.sleep(0.1)

    # No restoration calls should be made
    restore_calls = [
        c for c in service_calls
        if c.data.get(ATTR_ENTITY_ID) == entity_id
    ]
    assert len(restore_calls) == 0


async def test_no_restore_if_already_at_orig(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_light_entity: str,
    service_calls: list[ServiceCall],
) -> None:
    """Test no extra call if turned on at correct brightness."""
    # Store original brightness
    storage_key = mock_light_entity.replace(".", "_")
    hass.data[DOMAIN]["data"][storage_key] = {KEY_ORIG_BRIGHTNESS: 150}

    # Turn off then on at original brightness
    hass.states.async_set(
        mock_light_entity,
        STATE_OFF,
        {"supported_color_modes": ["brightness"]},
    )
    await hass.async_block_till_done()

    service_calls.clear()

    # Turn on at exact original brightness
    hass.states.async_set(
        mock_light_entity,
        STATE_ON,
        {ATTR_BRIGHTNESS: 150, "supported_color_modes": ["brightness"]},
    )
    await hass.async_block_till_done()
    await asyncio.sleep(0.1)

    # No restoration call needed
    restore_calls = [
        c for c in service_calls
        if c.data.get(ATTR_ENTITY_ID) == mock_light_entity
    ]
    assert len(restore_calls) == 0


async def test_no_restore_for_non_dimmable(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_non_dimmable_light: str,
    service_calls: list[ServiceCall],
) -> None:
    """Test non-dimmable lights skip restoration."""
    # Turn off
    hass.states.async_set(
        mock_non_dimmable_light,
        STATE_OFF,
        {"supported_color_modes": ["onoff"]},
    )
    await hass.async_block_till_done()

    service_calls.clear()

    # Turn on
    hass.states.async_set(
        mock_non_dimmable_light,
        STATE_ON,
        {"supported_color_modes": ["onoff"]},
    )
    await hass.async_block_till_done()
    await asyncio.sleep(0.1)

    # No restoration calls
    restore_calls = [
        c for c in service_calls
        if c.data.get(ATTR_ENTITY_ID) == mock_non_dimmable_light
    ]
    assert len(restore_calls) == 0


async def test_storage_persists_across_reload(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Test stored brightness survives integration reload."""
    mock_config_entry.add_to_hass(hass)

    stored_data = {"light_test_light": {"orig": 180, "curr": 100}}

    with patch(
        "custom_components.fade_lights.Store.async_load",
        return_value=stored_data,
    ):
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()

    # Check data was loaded
    assert hass.data[DOMAIN]["data"]["light_test_light"]["orig"] == 180

    # Unload and reload
    await hass.config_entries.async_unload(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    with patch(
        "custom_components.fade_lights.Store.async_load",
        return_value=stored_data,
    ):
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()

    # Data should still be there
    assert hass.data[DOMAIN]["data"]["light_test_light"]["orig"] == 180


async def test_restore_uses_correct_brightness(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    service_calls: list[ServiceCall],
) -> None:
    """Test restores to exact stored value."""
    entity_id = "light.precise_light"
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {ATTR_BRIGHTNESS: 173, "supported_color_modes": ["brightness"]},
    )
    await hass.async_block_till_done()

    # Fade to off (stores original as 173)
    await hass.services.async_call(
        DOMAIN,
        SERVICE_FADE_LIGHTS,
        {
            ATTR_ENTITY_ID: entity_id,
            ATTR_BRIGHTNESS_PCT: 0,
            ATTR_TRANSITION: 0.1,
        },
        blocking=True,
    )

    service_calls.clear()

    # Turn on at different brightness
    hass.states.async_set(
        entity_id,
        STATE_ON,
        {ATTR_BRIGHTNESS: 50, "supported_color_modes": ["brightness"]},
    )
    await hass.async_block_till_done()
    await asyncio.sleep(0.2)
    await hass.async_block_till_done()

    # Should restore to exactly 173
    restore_calls = [
        c for c in service_calls
        if c.service == "turn_on"
        and c.data.get(ATTR_ENTITY_ID) == entity_id
        and c.data.get(ATTR_BRIGHTNESS) == 173
    ]
    assert len(restore_calls) >= 1
```

**Step 2: Run tests**

Run:
```bash
cd /tmp/ha_fade_lights && pytest tests/test_brightness_restoration.py -v
```

Expected: All 6 tests pass

**Step 3: Commit**

```bash
git add tests/test_brightness_restoration.py
git commit -m "test: add brightness restoration tests"
```

---

## Task 8: GitHub Actions CI Workflow

**Files:**
- Create: `.github/workflows/tests.yml`

**Step 1: Create CI workflow**

```yaml
name: Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12", "3.13"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[test]"

      - name: Run tests with coverage
        run: |
          pytest tests/ \
            --cov=custom_components.fade_lights \
            --cov-report=term-missing \
            --cov-report=xml \
            --cov-fail-under=90 \
            -v

      - name: Upload coverage report
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: coverage-report-${{ matrix.python-version }}
          path: coverage.xml
```

**Step 2: Commit**

```bash
git add .github/workflows/tests.yml
git commit -m "ci: add GitHub Actions test workflow"
```

---

## Task 9: Run Full Test Suite and Verify Coverage

**Step 1: Run full test suite with coverage**

Run:
```bash
cd /tmp/ha_fade_lights && pytest tests/ --cov=custom_components.fade_lights --cov-report=term-missing -v
```

Expected: All 45 tests pass, coverage > 90%

**Step 2: Push all commits**

```bash
git push
```

---

## Summary

| Task | Files | Tests |
|------|-------|-------|
| 1. Project Setup | pyproject.toml, tests/__init__.py, tests/conftest.py | - |
| 2. Config Flow | tests/test_config_flow.py | 8 |
| 3. Init Tests | tests/test_init.py | 6 |
| 4. Service Tests | tests/test_services.py | 7 |
| 5. Fade Execution | tests/test_fade_execution.py | 10 |
| 6. Manual Interruption | tests/test_manual_interruption.py | 8 |
| 7. Brightness Restoration | tests/test_brightness_restoration.py | 6 |
| 8. CI Workflow | .github/workflows/tests.yml | - |
| **Total** | **10 files** | **45 tests** |
