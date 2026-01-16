# Fade Lights Custom Integration

A Home Assistant custom integration that provides smooth light fading with intelligent manual change detection.

## Features

### 1. Smooth Light Fading Service

- Fade lights to any brightness level (0-100%) over a specified transition period
- Supports individual lights and light groups
- Automatically expands light groups
- Detects and aborts when lights are manually adjusted during fade
- Can be forced to override manual change detection

### 2. Auto-Brightness Correction

- Automatically detects when lights are turned on very dim (default brightness < 10)
- Boosts brightness to default 40% to prevent "nearly off" lights
- Only triggers on manual light activation (not automation-triggered)

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
- **force** (optional, default: false): Force fade even if light was manually changed

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

**Force fade (ignore manual changes):**

```yaml
service: fade_lights.fade_lights
data:
  entity_id: light.bedroom
  brightness_pct: 100
  transition: 2
  force: true
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

### Manual Change Detection

The integration tracks the last brightness level set by automation. If a light's brightness doesn't match the stored value, it's considered "manually changed" and fading is skipped (unless `force: true`).

### Fade Algorithm

- Calculates optimal step size based on transition duration
- Minimum 100ms delay between steps to prevent overwhelming devices
- Monitors brightness during fade and aborts if external changes detected
- Handles edge cases (brightness = 1, non-dimmable lights)

### Auto-Brightness Feature

Listens for state changes and automatically corrects lights that turn on very dim:

- Triggers only on manual activation (no parent context)
- Only for dimmable lights with brightness < 10
- Sets brightness to 40% automatically
- Ignores light groups

### Non-Dimmable Lights

Lights that do not support brightness will turn off when brightness is set to 0, or turn on when brightness is greater than 0.

## License

MIT License - feel free to modify and redistribute
