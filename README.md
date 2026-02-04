# Fade Lights Custom Integration

A Home Assistant custom integration that provides smooth light fading with automatic brightness restoration.

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
2. You fade it to 0% (off) over 30 minutes
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

## Usage

### Service: `fade_lights.fade_lights`

Fades one or more lights to a target brightness over a transition period.

#### Parameters:

- **entity_id** (required): Light entity ID, light group, or list of light entities
- **brightness_pct** (optional, default: 40): Target brightness percentage (0-100)
- **transition** (optional, default: 3): Transition duration in seconds

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
- **Original brightness**: The user's intended brightness level
- **Current brightness**: The brightness being set during fade operations

When you fade a light to off, the original brightness is preserved. When the light is manually turned on again, it's automatically restored to that original brightness.

### Manual Change Detection

The integration uses Home Assistant's context system to distinguish between:
- Changes made by the fade service (ignored for restoration)
- Changes made manually or by other automations (triggers brightness restoration)

If you manually adjust a light's brightness while it's on, that becomes the new "original" brightness.

### Non-Dimmable Lights

Lights that do not support brightness will turn off when brightness is set to 0, or turn on when brightness is greater than 0.

## Development

### Running Tests

The integration includes a comprehensive test suite with 45 tests covering config flow, service handling, fade execution, manual interruption detection, and brightness restoration.

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

The test suite targets 90%+ code coverage and includes tests for:

- **Config flow** (`test_config_flow.py`): User setup, import flow, options validation
- **Integration setup** (`test_init.py`): Service registration, storage loading, unload cleanup
- **Service handling** (`test_services.py`): Entity ID formats, group expansion, default parameters
- **Fade execution** (`test_fade_execution.py`): Fade up/down, turn off at 0%, non-dimmable lights
- **Manual interruption** (`test_manual_interruption.py`): Brightness change detection, fade cancellation
- **Brightness restoration** (`test_brightness_restoration.py`): Restore on turn-on, storage persistence

### Continuous Integration

Tests run automatically on push and pull requests via GitHub Actions. The workflow tests against Python 3.12 and 3.13.

## License

MIT License - feel free to modify and redistribute
