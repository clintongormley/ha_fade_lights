# Fade Lights Test Suite Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a comprehensive pytest test suite for the fade_lights integration with 95%+ coverage.

**Architecture:** Uses pytest-homeassistant-custom-component for HA test fixtures, with mock light entities to test fade execution, manual interruption, and brightness restoration behaviors.

**Tech Stack:** pytest, pytest-asyncio, pytest-cov, pytest-homeassistant-custom-component, syrupy

---

## 1. Test Structure

```
/tmp/ha_fade_lights/
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures
│   ├── test_config_flow.py      # Config flow & options flow tests
│   ├── test_init.py             # Integration setup/unload tests
│   ├── test_services.py         # Service parameter handling tests
│   ├── test_fade_execution.py   # Fade logic & timing tests
│   ├── test_manual_interruption.py  # Manual change detection tests
│   └── test_brightness_restoration.py  # Restore behavior tests
├── pyproject.toml               # Test dependencies & pytest config
└── .github/
    └── workflows/
        └── tests.yml            # CI workflow
```

## 2. Core Fixtures (conftest.py)

### Mock Light Entity
```python
@pytest.fixture
async def mock_light(hass):
    """Create a mock dimmable light entity."""
    # Registers light.test_light with:
    # - ATTR_BRIGHTNESS support
    # - ColorMode.BRIGHTNESS in supported modes
    # - Initial state ON at brightness 200 (78%)
```

### Mock Light Group
```python
@pytest.fixture
async def mock_light_group(hass, mock_light):
    """Create a light group containing mock lights."""
    # Registers light.test_group with entity_id attribute
```

### Mock Non-Dimmable Light
```python
@pytest.fixture
async def mock_non_dimmable_light(hass):
    """Create a light without brightness support."""
```

### Integration Setup
```python
@pytest.fixture
async def init_integration(hass, enable_custom_integrations):
    """Set up the fade_lights integration."""
    # Creates MockConfigEntry, calls async_setup_entry
    # Returns the config entry
```

### Service Call Capture
```python
@pytest.fixture
def captured_calls(hass):
    """Capture light.turn_on/turn_off service calls."""
```

## 3. Test Cases

### test_config_flow.py (8 tests)

| Test | Description |
|------|-------------|
| `test_user_flow_creates_entry` | User setup creates config entry |
| `test_single_instance_only` | Second setup aborts |
| `test_import_flow` | Auto-import creates entry |
| `test_options_flow_shows_defaults` | Options form shows current values |
| `test_options_flow_updates_values` | Changed options are saved |
| `test_options_flow_validates_brightness_range` | 0-100 enforced |
| `test_options_flow_validates_transition_range` | 0-3600 enforced |
| `test_options_flow_validates_step_delay_range` | 10-1000 enforced |

### test_init.py (6 tests)

| Test | Description |
|------|-------------|
| `test_setup_entry_registers_service` | Service available after setup |
| `test_setup_entry_loads_storage` | Stored data loaded |
| `test_unload_entry_removes_service` | Service removed on unload |
| `test_unload_entry_cancels_active_fades` | Active fades cancelled |
| `test_unload_entry_clears_tracking_dicts` | Tracking dicts cleared |
| `test_options_update_reloads_entry` | Options change triggers reload |

### test_services.py (7 tests)

| Test | Description |
|------|-------------|
| `test_service_accepts_single_entity` | Single entity_id works |
| `test_service_accepts_entity_list` | List of entity_ids works |
| `test_service_accepts_comma_string` | Comma-separated string works |
| `test_service_expands_light_groups` | Groups expanded |
| `test_service_rejects_non_light_entity` | Non-light raises error |
| `test_service_uses_default_brightness` | Missing param uses default |
| `test_service_uses_default_transition` | Missing param uses default |

### test_fade_execution.py (10 tests)

| Test | Description |
|------|-------------|
| `test_fade_down_reaches_target` | Fade down hits target brightness |
| `test_fade_up_reaches_target` | Fade up hits target brightness |
| `test_fade_to_zero_turns_off` | 0% calls turn_off |
| `test_fade_from_off_turns_on` | From off state turns on |
| `test_fade_skips_brightness_level_1` | Level 1 skipped |
| `test_fade_already_at_target_no_op` | No calls if at target |
| `test_fade_non_dimmable_to_zero` | Non-dimmable turns off |
| `test_fade_non_dimmable_to_nonzero` | Non-dimmable turns on |
| `test_fade_unknown_entity_logs_warning` | Unknown entity logged |
| `test_fade_stores_orig_brightness` | Orig brightness stored |

### test_manual_interruption.py (8 tests)

| Test | Description |
|------|-------------|
| `test_manual_brightness_change_cancels_fade` | External change stops fade |
| `test_manual_turn_off_cancels_fade` | Turn off stops fade |
| `test_our_context_changes_ignored` | Our context not cancelled |
| `test_inherited_context_detected_by_brightness` | Inherited context detected |
| `test_brightness_tolerance_allows_rounding` | ±5 tolerance |
| `test_new_fade_cancels_previous` | New fade cancels old |
| `test_manual_change_stores_new_orig` | Manual becomes new orig |
| `test_group_changes_ignored` | Group entities ignored |

### test_brightness_restoration.py (6 tests)

| Test | Description |
|------|-------------|
| `test_restore_brightness_on_turn_on` | Restored on turn on |
| `test_no_restore_if_no_stored_brightness` | No restore if orig=0 |
| `test_no_restore_if_already_at_orig` | No extra call if correct |
| `test_no_restore_for_non_dimmable` | Non-dimmable skipped |
| `test_storage_persists_across_reload` | Storage survives reload |
| `test_restore_uses_correct_brightness` | Exact value restored |

## 4. Dependencies

### pyproject.toml additions
```toml
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
testpaths = ["tests"]
```

## 5. CI/CD Workflow

### .github/workflows/tests.yml
- Triggers: push/PR to main
- Matrix: Python 3.12, 3.13
- Steps: checkout, install deps, pytest with coverage
- Coverage threshold: 95%
- Artifact: coverage report

## 6. Test Summary

| File | Tests | Target |
|------|-------|--------|
| test_config_flow.py | 8 | config_flow.py 100% |
| test_init.py | 6 | Setup/unload paths |
| test_services.py | 7 | Service handling |
| test_fade_execution.py | 10 | Fade logic |
| test_manual_interruption.py | 8 | Cancellation |
| test_brightness_restoration.py | 6 | Restore behavior |
| **Total** | **45** | **95%+ overall** |

## Sources

- [Home Assistant Testing Docs](https://developers.home-assistant.io/docs/development_testing/)
- [pytest-homeassistant-custom-component](https://github.com/MatthewFlamm/pytest-homeassistant-custom-component)
