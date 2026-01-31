# FadeChange Refactor Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor step-building code to separate change calculation from step generation, using a self-contained FadeChange dataclass that calculates its own step count and generates steps on demand via an iterator pattern.

**Architecture:** FadeChange dataclass holds start/end values for each dimension plus transition timing. It calculates optimal step count based on change magnitude, then yields steps via has_next()/next_step(). Hybrid transitions return a list of FadeChange phases with brightness split proportionally.

**Tech Stack:** Python, Home Assistant integration patterns

---

## Section 1: Data Structures

### FadeChange Dataclass

Add to `models.py`:

```python
@dataclass
class FadeChange:
    """A single phase of a fade operation with iterator-based step generation.

    Holds start/end values for each dimension being faded, plus timing info.
    Calculates optimal step count and generates steps on demand.
    """

    # Brightness (0-255 scale)
    start_brightness: int | None = None
    end_brightness: int | None = None

    # HS color (hue 0-360, saturation 0-100)
    start_hs: tuple[float, float] | None = None
    end_hs: tuple[float, float] | None = None

    # Color temperature (mireds)
    start_mireds: int | None = None
    end_mireds: int | None = None

    # Timing
    transition_ms: int = 0
    min_step_delay_ms: int = 100

    # Iterator state
    _current_step: int = field(default=0, repr=False)
    _step_count: int | None = field(default=None, repr=False)

    def step_count(self) -> int:
        """Calculate optimal step count based on change magnitude and time."""
        if self._step_count is not None:
            return self._step_count

        # Use _calculate_step_count logic from const.py imports
        brightness_change = None
        if self.start_brightness is not None and self.end_brightness is not None:
            brightness_change = abs(self.end_brightness - self.start_brightness)

        hue_change = None
        sat_change = None
        if self.start_hs is not None and self.end_hs is not None:
            hue_diff = abs(self.end_hs[0] - self.start_hs[0])
            if hue_diff > 180:
                hue_diff = 360 - hue_diff
            hue_change = hue_diff
            sat_change = abs(self.end_hs[1] - self.start_hs[1])

        mireds_change = None
        if self.start_mireds is not None and self.end_mireds is not None:
            mireds_change = abs(self.end_mireds - self.start_mireds)

        self._step_count = _calculate_step_count(
            brightness_change,
            hue_change,
            sat_change,
            mireds_change,
            self.transition_ms,
            self.min_step_delay_ms,
        )
        return self._step_count

    def delay_ms(self) -> float:
        """Calculate delay between steps in milliseconds."""
        count = self.step_count()
        if count <= 1:
            return 0.0
        return self.transition_ms / count

    def reset(self) -> None:
        """Reset iterator to beginning."""
        self._current_step = 0

    def has_next(self) -> bool:
        """Check if more steps remain."""
        return self._current_step < self.step_count()

    def next_step(self) -> FadeStep:
        """Generate and return the next step in the fade sequence."""
        if not self.has_next():
            raise StopIteration("No more steps")

        count = self.step_count()
        step_idx = self._current_step
        self._current_step += 1

        # Calculate interpolation factor (0.0 to 1.0)
        # For last step, always use 1.0 to ensure we hit target exactly
        if step_idx >= count - 1:
            t = 1.0
        else:
            t = (step_idx + 1) / count

        return FadeStep(
            brightness=self._interpolate_brightness(t),
            hs_color=self._interpolate_hs(t),
            color_temp_mireds=self._interpolate_mireds(t),
        )

    def _interpolate_brightness(self, t: float) -> int | None:
        """Interpolate brightness at factor t."""
        if self.start_brightness is None or self.end_brightness is None:
            return None
        return round(self.start_brightness + t * (self.end_brightness - self.start_brightness))

    def _interpolate_hs(self, t: float) -> tuple[float, float] | None:
        """Interpolate HS color at factor t, handling hue wraparound."""
        if self.start_hs is None or self.end_hs is None:
            return None

        start_hue, start_sat = self.start_hs
        end_hue, end_sat = self.end_hs

        # Handle hue wraparound (take shortest path)
        hue_diff = end_hue - start_hue
        if hue_diff > 180:
            hue_diff -= 360
        elif hue_diff < -180:
            hue_diff += 360

        new_hue = (start_hue + t * hue_diff) % 360
        new_sat = start_sat + t * (end_sat - start_sat)

        return (round(new_hue, 2), round(new_sat, 2))

    def _interpolate_mireds(self, t: float) -> int | None:
        """Interpolate mireds at factor t."""
        if self.start_mireds is None or self.end_mireds is None:
            return None
        return round(self.start_mireds + t * (self.end_mireds - self.start_mireds))
```

---

## Section 2: Change Calculators

### Main Dispatcher

Add to `__init__.py`:

```python
def _calculate_changes(
    params: FadeParams,
    current_state: dict[str, Any],
    min_step_delay_ms: int,
) -> list[FadeChange]:
    """Calculate fade changes, dispatching to hybrid calculators if needed.

    Returns a list of FadeChange phases:
    - Simple fades: single FadeChange
    - Hybrid transitions (HS↔mireds): two FadeChange phases
    """
    # Resolve start values from state or params.from_*
    start_brightness = _resolve_start_brightness(params, current_state)
    start_hs = _resolve_start_hs(params, current_state)
    start_mireds = _resolve_start_mireds(params, current_state)

    # Resolve end values from params
    end_brightness = _resolve_end_brightness(params, current_state)
    end_hs = params.hs_color
    end_mireds = params.color_temp_mireds

    # Detect hybrid transitions
    if start_hs is not None and end_mireds is not None and end_hs is None:
        # HS → mireds transition
        return _calculate_hs_to_mireds_changes(
            start_brightness, end_brightness,
            start_hs, end_mireds,
            params.transition_ms, min_step_delay_ms,
        )

    if start_mireds is not None and end_hs is not None and end_mireds is None:
        # mireds → HS transition
        return _calculate_mireds_to_hs_changes(
            start_brightness, end_brightness,
            start_mireds, end_hs,
            params.transition_ms, min_step_delay_ms,
        )

    # Simple fade (single phase)
    return [FadeChange(
        start_brightness=start_brightness,
        end_brightness=end_brightness,
        start_hs=start_hs if end_hs is not None else None,
        end_hs=end_hs,
        start_mireds=start_mireds if end_mireds is not None else None,
        end_mireds=end_mireds,
        transition_ms=params.transition_ms,
        min_step_delay_ms=min_step_delay_ms,
    )]
```

### Hybrid Calculators

```python
def _calculate_hs_to_mireds_changes(
    start_brightness: int | None,
    end_brightness: int | None,
    start_hs: tuple[float, float],
    end_mireds: int,
    transition_ms: int,
    min_step_delay_ms: int,
) -> list[FadeChange]:
    """Calculate HS → mireds hybrid transition (two phases).

    Phase 1 (70%): Fade HS toward Planckian locus
    Phase 2 (30%): Fade mireds to target
    """
    # Find intersection point on Planckian locus
    intersection_hs, intersection_mireds = _find_planckian_intersection(start_hs)

    # Split timing 70/30
    phase1_ms = int(transition_ms * 0.7)
    phase2_ms = transition_ms - phase1_ms

    # Split brightness proportionally
    mid_brightness = None
    if start_brightness is not None and end_brightness is not None:
        brightness_change = end_brightness - start_brightness
        mid_brightness = start_brightness + int(brightness_change * 0.7)

    return [
        FadeChange(
            start_brightness=start_brightness,
            end_brightness=mid_brightness,
            start_hs=start_hs,
            end_hs=intersection_hs,
            transition_ms=phase1_ms,
            min_step_delay_ms=min_step_delay_ms,
        ),
        FadeChange(
            start_brightness=mid_brightness,
            end_brightness=end_brightness,
            start_mireds=intersection_mireds,
            end_mireds=end_mireds,
            transition_ms=phase2_ms,
            min_step_delay_ms=min_step_delay_ms,
        ),
    ]


def _calculate_mireds_to_hs_changes(
    start_brightness: int | None,
    end_brightness: int | None,
    start_mireds: int,
    end_hs: tuple[float, float],
    transition_ms: int,
    min_step_delay_ms: int,
) -> list[FadeChange]:
    """Calculate mireds → HS hybrid transition (two phases).

    Phase 1 (30%): Fade mireds to Planckian intersection
    Phase 2 (70%): Fade HS to target
    """
    # Find intersection point on Planckian locus
    intersection_hs, intersection_mireds = _find_planckian_intersection_from_mireds(
        start_mireds, end_hs
    )

    # Split timing 30/70
    phase1_ms = int(transition_ms * 0.3)
    phase2_ms = transition_ms - phase1_ms

    # Split brightness proportionally
    mid_brightness = None
    if start_brightness is not None and end_brightness is not None:
        brightness_change = end_brightness - start_brightness
        mid_brightness = start_brightness + int(brightness_change * 0.3)

    return [
        FadeChange(
            start_brightness=start_brightness,
            end_brightness=mid_brightness,
            start_mireds=start_mireds,
            end_mireds=intersection_mireds,
            transition_ms=phase1_ms,
            min_step_delay_ms=min_step_delay_ms,
        ),
        FadeChange(
            start_brightness=mid_brightness,
            end_brightness=end_brightness,
            start_hs=intersection_hs,
            end_hs=end_hs,
            transition_ms=phase2_ms,
            min_step_delay_ms=min_step_delay_ms,
        ),
    ]
```

### Resolver Helpers

```python
def _resolve_start_brightness(params: FadeParams, state: dict) -> int | None:
    """Resolve starting brightness from params.from_brightness_pct or current state."""
    if params.from_brightness_pct is not None:
        return int(params.from_brightness_pct * 255 / 100)
    return state.get(ATTR_BRIGHTNESS)


def _resolve_end_brightness(params: FadeParams, state: dict) -> int | None:
    """Resolve ending brightness from params.brightness_pct."""
    if params.brightness_pct is not None:
        return int(params.brightness_pct * 255 / 100)
    return None


def _resolve_start_hs(params: FadeParams, state: dict) -> tuple[float, float] | None:
    """Resolve starting HS from params.from_hs_color or current state."""
    if params.from_hs_color is not None:
        return params.from_hs_color
    return state.get(ATTR_HS_COLOR)


def _resolve_start_mireds(params: FadeParams, state: dict) -> int | None:
    """Resolve starting mireds from params.from_color_temp_mireds or current state."""
    if params.from_color_temp_mireds is not None:
        return params.from_color_temp_mireds
    return state.get(ATTR_COLOR_TEMP)
```

---

## Section 3: Step Generation (Iterator Pattern)

FadeChange uses an iterator pattern instead of pre-building a list of steps:

```python
# Usage in _execute_fade:
for phase in phases:
    phase.reset()  # Ensure iterator starts at beginning
    while phase.has_next():
        step = phase.next_step()
        # Send step to light
        await _send_step(hass, entity_id, step, expected_state)
        # Wait before next step
        await asyncio.sleep(phase.delay_ms() / 1000)
```

Benefits:
- Memory efficient (no list allocation)
- Self-contained (FadeChange knows its own step count and timing)
- Simple interface (has_next/next_step)

---

## Section 4: Orchestration

### Simplified _execute_fade

```python
async def _execute_fade(
    hass: HomeAssistant,
    entity_id: str,
    params: FadeParams,
    expected_state: ExpectedState,
    min_step_delay_ms: int,
) -> None:
    """Execute a fade operation on a single light."""
    # Get current state
    state = hass.states.get(entity_id)
    if state is None:
        return
    current_state = state.attributes

    # Calculate changes (returns list of phases)
    phases = _calculate_changes(params, current_state, min_step_delay_ms)

    # Execute each phase
    for phase in phases:
        phase.reset()
        while phase.has_next():
            # Check for cancellation
            if _is_fade_cancelled(entity_id):
                return

            step = phase.next_step()

            # Track expected values for manual intervention detection
            expected = ExpectedValues(
                brightness=step.brightness,
                hs_color=step.hs_color,
                color_temp_mireds=step.color_temp_mireds,
            )
            expected_state.add(expected)

            # Send step to light
            await _send_light_command(hass, entity_id, step)

            # Wait before next step (skip delay after last step of last phase)
            if phase.has_next() or phase is not phases[-1]:
                await asyncio.sleep(phase.delay_ms() / 1000)
```

---

## Section 5: Summary

### Code to Add

**models.py:**
- `FadeChange` dataclass with:
  - Start/end fields for brightness, hs, mireds
  - `transition_ms` and `min_step_delay_ms` timing fields
  - `step_count()` - calculates optimal steps from change magnitude
  - `delay_ms()` - calculates inter-step delay
  - `reset()`, `has_next()`, `next_step()` - iterator interface
  - `_interpolate_*()` helpers for each dimension

**__init__.py:**
- `_calculate_changes()` - main dispatcher
- `_calculate_hs_to_mireds_changes()` - hybrid HS→mireds (70/30 split)
- `_calculate_mireds_to_hs_changes()` - hybrid mireds→HS (30/70 split)
- `_resolve_start_brightness()`, `_resolve_end_brightness()`
- `_resolve_start_hs()`, `_resolve_start_mireds()`

### Code to Remove

**__init__.py:**
- `_build_fade_steps()` - replaced by FadeChange iterator
- `_build_hs_to_mireds_steps()` - replaced by `_calculate_hs_to_mireds_changes()`
- `_build_mireds_to_hs_steps()` - replaced by `_calculate_mireds_to_hs_changes()`
- `_calculate_fade_steps()` - logic moved into FadeChange.step_count()

### Benefits

1. **Separation of concerns**: Change calculation separate from step generation
2. **Self-contained phases**: FadeChange knows its timing and step count
3. **Memory efficient**: Iterator pattern avoids pre-building step lists
4. **Simpler orchestration**: _execute_fade just iterates phases
5. **Proportional brightness**: Hybrid transitions split brightness correctly across phases
