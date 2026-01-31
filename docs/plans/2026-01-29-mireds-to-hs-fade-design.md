# Mireds to HS hybrid fade transition

## Overview

Add support for smooth color transitions from color temperature mode to HS color mode. When a light is in `COLOR_TEMP` hardware mode and the user requests a fade to an HS color, the integration should transition smoothly through the Planckian locus rather than jumping directly to the target color.

This is the symmetric counterpart to the existing HS-to-mireds hybrid fade.

## Detection logic

Trigger condition:
```python
color_mode = state.attributes.get("color_mode")

if color_mode == ColorMode.COLOR_TEMP and end_hs is not None:
    # Use hybrid mireds -> HS transition
```

The check uses the light's actual hardware `color_mode` attribute, not a saturation-based heuristic.

## Step builder function

New function `_build_mireds_to_hs_steps()`:

```python
def _build_mireds_to_hs_steps(
    start_mireds: int,
    end_hs: tuple[float, float],
    transition_ms: int,
    min_step_delay_ms: int,
) -> list[FadeStep]:
```

### Phase 1: Mireds along locus (~30% of steps)

1. Find the locus point closest to the target hue using `_hs_to_mireds(end_hs)`
2. Generate mireds steps from `start_mireds` toward `target_locus_mireds`
3. Each step contains only `color_temp_mireds`

### Phase 2: HS to final target (~70% of steps)

1. Convert `target_locus_mireds` to HS using `_mireds_to_hs()` to get the starting HS point on the locus
2. Generate HS steps from that locus HS to `end_hs`
3. Each step contains only `hs_color`

The 30/70 split mirrors the existing HS-to-mireds function (which uses 70/30) because the "main event" is different: for mireds-to-HS, the saturation increase is the visually significant change.

## Integration into _execute_fade()

Update the step builder selection logic:

```python
color_mode = state.attributes.get("color_mode")

if start_hs is not None and end_mireds is not None and not _is_on_planckian_locus(start_hs):
    # Existing: Hybrid HS -> mireds
    steps = _build_hs_to_mireds_steps(start_hs, end_mireds, transition_ms, min_step_delay_ms)
elif color_mode == ColorMode.COLOR_TEMP and end_hs is not None:
    # New: Hybrid mireds -> HS
    steps = _build_mireds_to_hs_steps(start_mireds, end_hs, transition_ms, min_step_delay_ms)
else:
    # Standard fade
    steps = _build_fade_steps(...)
```

Brightness is interpolated across all steps using the same pattern as the existing HS-to-mireds code:

```python
if brightness_changing and end_brightness is not None:
    num_steps = len(steps)
    for i, step in enumerate(steps):
        t = (i + 1) / num_steps
        step.brightness = round(start_brightness + (end_brightness - start_brightness) * t)
```

## Required imports

```python
from homeassistant.components.light import ATTR_COLOR_MODE
```

## Testing strategy

New test file: `tests/test_mireds_to_hs_fade.py`

### Integration tests

1. **Basic mireds to HS transition** - Light in COLOR_TEMP mode at 3000K, fade to red (0, 100). Verify steps transition from mireds to HS.

2. **Mireds to HS with brightness** - Same as above but also fading brightness 50% to 100%. Verify brightness interpolates across all steps.

3. **Target on locus** - Light in COLOR_TEMP mode, fade to low-saturation HS (30, 5). Verify we still end in HS mode.

4. **Light not in COLOR_TEMP mode** - Light in HS mode, fade to different HS. Verify standard HS fade used (not hybrid).

5. **Step count verification** - Verify approximately 30% mireds steps, 70% HS steps.

### Unit tests for _build_mireds_to_hs_steps()

- Correct phase split (30/70)
- Mireds steps fade toward target hue's locus point
- HS steps start from locus and end at target
- Edge cases: target already on locus, extreme color temperatures

## Files to modify

- `custom_components/fade_lights/__init__.py`
  - Add `ATTR_COLOR_MODE` import
  - Add `_build_mireds_to_hs_steps()` function
  - Update step builder selection in `_execute_fade()`

## Files to create

- `tests/test_mireds_to_hs_fade.py` - Test cases for the new functionality
