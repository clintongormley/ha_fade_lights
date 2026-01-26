# Fade Lights Custom Integration

[![HACS Default](https://img.shields.io/badge/HACS-Default-blue.svg)](https://github.com/hacs/integration)
[![GitHub Release](https://img.shields.io/github/v/release/clintongormley/ha_fade_lights)](https://github.com/clintongormley/ha_fade_lights/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Home Assistant custom integration that provides smooth light fading with automatic brightness restoration.

## Compatibility

- **Home Assistant:** 2024.1.0 or newer
- **Python:** 3.12 or newer

## Features

### Smooth Light Fading Service

- Fade lights to any brightness level (0-100%) over a specified transition period
- Supports individual lights and light groups
- Automatically expands light groups
- Cancels fade when lights are manually adjusted

### Automatic Brightness Restoration

When you fade a light down to off and then manually turn it back on, the integration automatically restores the light to its original brightness (before the fade started).

**Example:**

1. Light is at 80% brightness
2. You fade it to 0% (off) over 5 seconds
3. Later, you turn the light on manually
4. Light automatically restores to 80%

## Installation

### HACS (Recommended)

1. Open HACS in your Home Assistant instance
2. Click the 3 dots in the top right corner
3. Select "Custom repositories"
4. Add `https://github.com/clintongormley/ha_fade_lights` as an integration
5. Click "Explore & Download Repositories"
6. Search for "Fade Lights"
7. Click "Download"
8. Restart Home Assistant

### Manual Installation

1. Copy the `custom_components/fade_lights` folder to your Home Assistant installation:
   ```
   <config_directory>/custom_components/fade_lights/
   ```
2. Restart Home Assistant

## Configuration

After installation and restart, add the integration via the Home Assistant UI:

1. Go to **Settings** → **Devices & Services**
2. Click **+ Add Integration**
3. Search for "Fade Lights"
4. Click to add it

Once configured, the `fade_lights.fade_lights` service will be available in **Developer Tools** → **Actions**.

### Configuration Options

To adjust the integration settings, go to **Settings** → **Devices & Services** → **Fade Lights** → **Configure**.

| Option                          | Description                                                                                                          | Default | Range   |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------- | ------- |
| **Minimum delay between steps** | Minimum delay between brightness steps in milliseconds. Lower values create smoother fades but increase system load. | 100ms   | 50-1000 |

## Usage

### Service: `fade_lights.fade_lights`

Fades one or more lights to a target brightness over a transition period.

#### Parameters:

- **entity_id** (required): One or more light entities. Accepts:
  - A single entity: `light.bedroom`
  - A comma-separated string: `light.bedroom, light.kitchen`
  - A YAML list: `[light.bedroom, light.kitchen]`
  - Light groups are automatically expanded to their individual lights
  - Duplicate entities are automatically deduplicated
- **brightness_pct** (optional, default: 40): Target brightness percentage (0-100)
- **transition** (optional, default: 3): Transition duration in seconds (supports decimals, e.g., `0.5` for 500ms)

#### Examples:

**Basic fade:**

```yaml
service: fade_lights.fade_lights
data:
  entity_id: light.bedroom
  brightness_pct: 50
  transition: 5
```

**Fade multiple lights:**

```yaml
service: fade_lights.fade_lights
data:
  entity_id:
    - light.bedroom
    - light.living_room
  brightness_pct: 80
  transition: 10
```

**Fade a light group:**

```yaml
service: fade_lights.fade_lights
data:
  entity_id: light.all_downstairs
  brightness_pct: 30
  transition: 60
```

### Automation Example

```yaml
automation:
  - alias: "Sunset fade"
    trigger:
      - platform: sun
        event: sunset
        offset: "-00:30:00"
    action:
      - service: fade_lights.fade_lights
        data:
          entity_id: light.living_room
          brightness_pct: 20
          transition: 1800 # 30 minutes
```

## How It Works

### Brightness Tracking

The integration maintains two brightness values for each light:

- **Original brightness**: The brightness level before the fade started or before the light was turned off
- **Current brightness**: The brightness being set during fade operations

When you fade a light to off, the original brightness is preserved. When the light is manually turned on again, it's automatically restored to that original brightness.

### Manual Change Detection

The integration detects manual changes by comparing state (off vs on) and actual brightness to expected brightness during fades. If the brightness differs by more than ±3 (tolerance for device rounding), the change is treated as manual intervention and the fade is cancelled.

When a fade is cancelled due to manual intervention, the integration stores the user's intended state. If a late-arriving fade service call reverts the user's change, the integration automatically restores the user's intended state.

#### Behavior During Fades

| Action                | Result                                                                     |
| --------------------- | -------------------------------------------------------------------------- |
| **Turn off**          | Fade cancelled, light turns off, original brightness before fade preserved |
| **Change brightness** | Fade cancelled, original brightness updated to new brightness              |

#### Behavior When Light Is Off

| Action                             | Result                            |
| ---------------------------------- | --------------------------------- |
| **Turn on**                        | Brightness restored to original   |
| **Turn on and set new brightness** | Brightness restored to original\* |

_\*Limitation: When turning on a light and simultaneously setting a brightness (via app or some smart switches), the integration cannot distinguish this from a simple turn-on. The original brightness will be restored, overriding the requested brightness._

#### Behavior When Light Is On (No Active Fade)

If you manually adjust a light's brightness while it's on (and no fade is active), that becomes the new "original" brightness for future restoration.

### Non-Dimmable Lights

Lights that do not support brightness will turn off when brightness is set to 0, or turn on when brightness is greater than 0.

## Troubleshooting

### Logging Levels

The integration provides two levels of logging:

| Level | What it shows |
|-------|---------------|
| **INFO** | High-level overview: fade start/complete, manual intervention detected, brightness restoration |
| **DEBUG** | Low-level details: every brightness step, expected state tracking, task cancellation internals |

For most troubleshooting, **INFO** level is sufficient and easier to follow.

### Enable Logging via UI

1. Go to **Settings** > **Devices & Services** > **Fade Lights**
2. Click **Enable debug logging**
3. Reproduce the issue
4. Click **Disable debug logging** to download the log file

This enables DEBUG level logging temporarily.

### Enable Logging via configuration.yaml

Add to your `configuration.yaml`:

```yaml
# INFO level - recommended for general troubleshooting
logger:
  logs:
    custom_components.fade_lights: info
```

```yaml
# DEBUG level - for detailed investigation
logger:
  logs:
    custom_components.fade_lights: debug
```

After editing, restart Home Assistant or call the `logger.set_level` action.

### Enable Logging via Action Call

You can also enable logging temporarily via **Developer Tools** > **Actions**:

```yaml
action: logger.set_level
data:
  custom_components.fade_lights: info
```

Or for debug level:

```yaml
action: logger.set_level
data:
  custom_components.fade_lights: debug
```

To view logs, go to **Settings** > **System** > **Logs**, click the three-dot menu, and select **Load full logs**.

### Reporting Issues

If you encounter a bug, please [open an issue](https://github.com/clintongormley/ha_fade_lights/issues/new/choose) with:

- Your Home Assistant version
- The integration version
- Debug logs showing the problem
- Steps to reproduce

## Development

### Running Tests

The integration includes a comprehensive test suite with 85 tests covering config flow, service handling, fade execution, manual interruption detection, and brightness restoration.

#### Prerequisites

Install the test dependencies:

```bash
pip install pytest pytest-asyncio pytest-cov pytest-homeassistant-custom-component syrupy
```

> **Note:** Do not use `pip install -e .` (editable install) as it conflicts with `pytest-homeassistant-custom-component`'s custom component discovery mechanism.

#### Running Tests

Run all tests:

```bash
pytest tests/ -v
```

Run tests with coverage report:

```bash
pytest tests/ --cov=custom_components.fade_lights --cov-report=term-missing -v
```

Run a specific test file:

```bash
pytest tests/test_fade_execution.py -v
```

#### Test Coverage

The test suite achieves 100% code coverage and includes tests for:

- **Config flow** (`test_config_flow.py`): User setup, import flow, options validation
- **Integration setup** (`test_init.py`): Service registration, storage loading, unload cleanup
- **Service handling** (`test_services.py`): Entity ID formats, group expansion, default parameters
- **Fade execution** (`test_fade_execution.py`): Fade up/down, turn off at 0%, non-dimmable lights
- **Manual interruption** (`test_manual_interruption.py`): Brightness change detection, fade cancellation
- **Brightness restoration** (`test_brightness_restoration.py`): Restore on turn-on, storage persistence
- **Event waiting** (`test_event_waiting.py`): Condition-based event waiting, stale value pruning

### Continuous Integration

Tests run automatically on push and pull requests via GitHub Actions. The workflow tests against Python 3.12 and 3.13.

## License

MIT License - feel free to modify and redistribute
