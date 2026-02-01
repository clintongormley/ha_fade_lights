# FadeChange refactoring design

## Overview

Refactor FadeChange to handle hybrid HS↔color temp transitions internally as a single object with flat step generation, and consolidate resolution logic into a new `resolve_fade()` function.

## Goals

1. Single FadeChange object for all fade types (including hybrid transitions)
2. Flat step list for easier ease-in/ease-out implementation later
3. Consolidate resolution logic (currently in `_execute_fade` lines 293-351 and `_calculate_changes`)
4. Handle capability filtering in one place
5. Simplify `_execute_fade` to just: resolve, iterate, done

## Section 1: resolve_fade() function

New function that consolidates all resolution and filtering logic:

```python
def resolve_fade(
    params: FadeParams,
    state_attributes: dict,
    supported_color_modes: set[ColorMode],
    min_step_delay_ms: int,
    min_mireds: int | None = None,
    max_mireds: int | None = None,
) -> FadeChange | None:
    """Resolve fade parameters against light state, returning configured FadeChange."""
```

Responsibilities:
- Resolve start values from params or state
- Convert kelvin to mireds (with bounds clamping)
- Detect hybrid transition scenarios
- Filter/convert based on light capabilities
- Return `None` if nothing to fade

## Section 2: Flat step generation with mode crossover

Instead of internal phase objects, FadeChange tracks a crossover point and generates a flat step sequence:

```python
@dataclass
class FadeChange:
    """A fade operation with flat step generation."""

    # Target values
    end_brightness: int | None = None
    end_hs: tuple[float, float] | None = None
    end_mireds: int | None = None

    # Start values
    start_brightness: int | None = None
    start_hs: tuple[float, float] | None = None
    start_mireds: int | None = None

    # Timing
    transition_ms: int = 0
    min_step_delay_ms: int = 100

    # Hybrid transition tracking (private)
    _hybrid_direction: str | None = field(default=None, repr=False)  # "hs_to_mireds" | "mireds_to_hs" | None
    _crossover_step: int | None = field(default=None, repr=False)
    _crossover_hs: tuple[float, float] | None = field(default=None, repr=False)
    _crossover_mireds: int | None = field(default=None, repr=False)
```

### Hybrid directions

**HS → mireds** (current state is saturated HS, target is color temp):
- Steps 1 to crossover: emit `hs_color` (fading saturation toward 0)
- Steps after crossover: emit `color_temp_kelvin` (fading to target mireds)
- Crossover point: Planckian locus HS (low saturation white)

**mireds → HS** (current state is color temp, target is saturated HS):
- Steps 1 to crossover: emit `color_temp_kelvin` (fading to crossover mireds)
- Steps after crossover: emit `hs_color` (fading from white to target HS)
- Crossover point: same Planckian locus

The iterator checks `_hybrid_direction` to know which attribute to emit before/after the crossover. For non-hybrid fades, `_hybrid_direction` is `None` and it just interpolates normally.

## Section 3: Capability filtering in resolve_fade()

All light type handling happens in `resolve_fade()` before creating FadeChange:

1. **Non-dimmable light** (only `ONOFF`):
   - Create FadeChange with single step: brightness 0 or 255
   - `transition_ms=0` so `delay_ms()` returns 0
   - Iterator yields one step and is done

2. **Color temp requested but not supported** (only HS mode):
   - Convert target kelvin → equivalent HS on Planckian locus
   - Set `end_hs` instead of `end_mireds`

3. **HS requested but not supported** (only color_temp mode):
   - Convert target HS → nearest mireds (if low saturation)
   - Or skip the HS component if high saturation

4. **Brightness only light**: Strip any color attributes, keep brightness only

5. **Nothing to fade**: Return `None` (empty iterator case)

## Section 4: Simplified _execute_fade

With `resolve_fade()` handling all the complexity:

```python
async def _execute_fade(
    hass: HomeAssistant,
    entity_id: str,
    params: FadeParams,
    min_step_delay_ms: int,
    cancel_event: asyncio.Event,
) -> None:
    state = hass.states.get(entity_id)
    if state is None:
        return

    supported_modes = _get_supported_color_modes(state)
    min_mireds, max_mireds = _get_mireds_bounds(state)

    fade = resolve_fade(
        params, state.attributes, supported_modes,
        min_step_delay_ms, min_mireds, max_mireds
    )

    if fade is None:
        return  # Nothing to fade

    delay_s = fade.delay_ms() / 1000

    while fade.has_next():
        if cancel_event.is_set():
            return
        step = fade.next_step()
        await _apply_step(hass, entity_id, step)
        if fade.has_next():
            await asyncio.sleep(delay_s)
```

No more:
- Special case for non-dimmable lights
- Looping over `list[FadeChange]`
- `_calculate_changes()` dispatcher function

## Section 5: Summary

### Architecture changes

1. `resolve_fade()` - new function that:
   - Resolves params + state → configured FadeChange
   - Filters/converts based on light capabilities
   - Handles non-dimmable lights (single step, zero delay)
   - Handles hybrid detection and setup
   - Returns `None` if nothing to fade

2. `FadeChange` - refactored to:
   - Handle hybrid transitions internally (flat step list with crossover)
   - Track `_hybrid_direction`, `_crossover_step`, `_crossover_hs`, `_crossover_mireds`
   - Iterator generates steps seamlessly across mode switch

3. `_execute_fade` - simplified to:
   - Call `resolve_fade()`
   - Iterate if not None

### Code to remove

- `_calculate_changes()` dispatcher
- `_calculate_hs_to_mireds_changes()`
- `_calculate_mireds_to_hs_changes()`
- Special case handling in `_execute_fade`
- `list[FadeChange]` return pattern

### Tests to update

- Tests checking for `len(changes) == 2` need to test crossover behavior instead
- Tests calling `_calculate_changes` need to call `resolve_fade`
