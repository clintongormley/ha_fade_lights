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
- Automatically detects when lights are turned on very dim (brightness < 10)
- Boosts brightness to 40% to prevent "nearly off" lights
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

The integration automatically configures itself when Home Assistant starts. Optionally, you can add it to your `configuration.yaml` to ensure it loads:

```yaml
fade_lights:
```

After restart, the `fade_lights.fade_lights` service will be available in **Developer Tools** → **Actions**.

## Usage

### Service: `fade_lights.fade_lights`

Fades one or more lights to a target brightness over a transition period.

#### Parameters:
- **entity_id** (required): Light entity ID or list of light entities
- **brightness_pct** (optional, default: 40): Target brightness percentage (0-100)
- **transition** (optional, default: 3): Transition duration in seconds or HH:MM:SS format
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
  transition: "00:00:10"
```

**Fade a light group:**
```yaml
service: fade_lights.fade_lights
data:
  entity_id: light.all_downstairs
  brightness_pct: 30
  transition: 60
```

**Long transition format:**
```yaml
service: fade_lights.fade_lights
data:
  entity_id: light.bedroom
  brightness_pct: 0
  transition: "00:05:30"  # 5 minutes 30 seconds
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
          transition: "00:30:00"  # 30 minutes
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

## Differences from Pyscript Version

### Improvements:
- ✅ Native Home Assistant integration (no pyscript dependency)
- ✅ Proper async/await patterns
- ✅ Uses Home Assistant storage API
- ✅ Cancellable fade tasks
- ✅ Better error handling
- ✅ UI configuration via config flow
- ✅ HACS compatible

### Maintained Features:
- ✅ Smooth gradual fading
- ✅ Manual change detection
- ✅ Light group expansion
- ✅ Auto-brightness correction
- ✅ Persistent brightness storage
- ✅ Context-aware (doesn't trigger on automations)

## Troubleshooting

### Fade doesn't work
- Check that the light supports brightness
- Ensure the entity ID is correct
- Check logs for error messages

### Fade aborts immediately
- Light may have been manually changed
- Try using `force: true` to override

### Auto-brightness not triggering
- Check that brightness is actually < 10 when light turns on
- Verify the light wasn't turned on by an automation
- Check that it's not a light group

## Development

This integration follows Home Assistant development best practices:
- Async/await for all I/O operations
- Proper service registration
- Config flow for UI setup
- Persistent storage via Store
- Event listener cleanup on unload

## License

MIT License - feel free to modify and redistribute
