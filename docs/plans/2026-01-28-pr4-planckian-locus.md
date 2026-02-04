# PR4: Planckian Locus Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable smooth transitions between arbitrary HS colors and color temperature by implementing Planckian locus detection and hybrid HS→mireds step generation.

**Architecture:** A lookup table maps color temperatures (in mireds) to approximate HS values on the Planckian locus. Helper functions detect when an HS color is "on" the locus (low saturation warm/cool white), find the closest locus point to an arbitrary HS, and generate hybrid step sequences that transition from HS to mireds smoothly.

**Tech Stack:** Python 3.13, dataclasses, pytest

---

## Context

**Working directory:** `/tmp/ha_fade_lights/.worktrees/pr4-planckian-locus`
**Branch:** `pr4-planckian-locus`
**Test command:** `cd /tmp/ha_fade_lights/.worktrees/pr4-planckian-locus && python -m pytest tests/ -x -q`
**Lint command:** `cd /tmp/ha_fade_lights/.worktrees/pr4-planckian-locus && ruff check custom_components/ tests/ && ruff format --check custom_components/ tests/`

**Key files:**
- `custom_components/fade_lights/__init__.py` — will contain Planckian locus functions
- `custom_components/fade_lights/const.py` — will contain the locus lookup table
- `tests/test_planckian_locus.py` — new test file

**Design doc reference:** `/tmp/ha_fade_lights/docs/plans/2026-01-28-color-fading-design.md`, section "HS ↔ Color Temp Transitions (Planckian Locus)"

**From the design doc:**

The Planckian locus is a curved line through HS color space representing blackbody radiation (color temperatures). Key requirements:
1. Detect if starting HS is on/near the Planckian locus (saturation < 10 threshold)
2. If HS is NOT on locus: fade HS toward closest point on locus, then switch to mireds
3. If HS IS on locus: convert directly to mireds and fade using mireds only
4. Generate hybrid step sequences for HS→mireds transitions

**Planckian Locus HS Approximation (derived from blackbody radiation):**

| Mireds | Kelvin | Approx Hue | Approx Saturation |
|--------|--------|------------|-------------------|
| 154    | 6500K  | 220        | 5                 | (Cool daylight)
| 182    | 5500K  | 210        | 4                 | (Noon daylight)
| 222    | 4500K  | 45         | 6                 | (Cool white)
| 286    | 3500K  | 38         | 12                | (Neutral white)
| 333    | 3000K  | 35         | 18                | (Warm white)
| 400    | 2500K  | 32         | 30                | (Soft white)
| 500    | 2000K  | 28         | 45                | (Candlelight)

Note: The locus has very low saturation at cool temps (bluish-white) and increasing saturation at warm temps (amber/orange).

---

### Task 1: Add Planckian locus lookup table to const.py

**Files:**
- Modify: `custom_components/fade_lights/const.py`

**Step 1: Add the lookup table**

Add this at the end of the file:

```python
# Planckian locus lookup table: mireds -> (hue, saturation)
# Approximates the curve of blackbody radiation through HS color space.
# Used for HS ↔ color temperature transitions.
PLANCKIAN_LOCUS_HS: tuple[tuple[int, tuple[float, float]], ...] = (
    (154, (220.0, 5.0)),   # 6500K - Cool daylight (bluish)
    (167, (215.0, 4.5)),   # 6000K
    (182, (210.0, 4.0)),   # 5500K - Noon daylight
    (200, (55.0, 5.0)),    # 5000K - Horizon daylight
    (222, (45.0, 6.0)),    # 4500K - Cool white
    (250, (42.0, 8.0)),    # 4000K
    (286, (38.0, 12.0)),   # 3500K - Neutral white
    (303, (36.0, 15.0)),   # 3300K
    (333, (35.0, 18.0)),   # 3000K - Warm white
    (370, (33.0, 24.0)),   # 2700K - Soft white
    (400, (32.0, 30.0)),   # 2500K
    (435, (30.0, 38.0)),   # 2300K
    (500, (28.0, 45.0)),   # 2000K - Candlelight
)

# Maximum saturation to consider a color "on" the Planckian locus
PLANCKIAN_LOCUS_SATURATION_THRESHOLD = 15.0
```

**Step 2: Commit**

```bash
cd /tmp/ha_fade_lights/.worktrees/pr4-planckian-locus && git add custom_components/fade_lights/const.py && git commit -m "feat: add Planckian locus lookup table constants"
```

---

### Task 2: Write failing tests for `_is_on_planckian_locus`

**Files:**
- Create: `tests/test_planckian_locus.py`

**Step 1: Write initial tests**

```python
"""Tests for Planckian locus functions."""

from __future__ import annotations

from custom_components.fade_lights import _is_on_planckian_locus


class TestIsOnPlanckianLocus:
    """Test detection of HS colors on the Planckian locus."""

    def test_pure_white_is_on_locus(self) -> None:
        """Test that pure white (0 saturation) is on the locus."""
        assert _is_on_planckian_locus((0.0, 0.0)) is True
        assert _is_on_planckian_locus((180.0, 0.0)) is True

    def test_low_saturation_warm_white_is_on_locus(self) -> None:
        """Test that low saturation warm white is on the locus."""
        # Warm white hue ~35, low saturation
        assert _is_on_planckian_locus((35.0, 10.0)) is True
        assert _is_on_planckian_locus((32.0, 14.0)) is True

    def test_low_saturation_cool_white_is_on_locus(self) -> None:
        """Test that low saturation cool white is on the locus."""
        # Cool white hue ~210-220, low saturation
        assert _is_on_planckian_locus((210.0, 5.0)) is True
        assert _is_on_planckian_locus((220.0, 8.0)) is True

    def test_high_saturation_is_not_on_locus(self) -> None:
        """Test that high saturation colors are NOT on the locus."""
        assert _is_on_planckian_locus((35.0, 50.0)) is False  # Saturated warm
        assert _is_on_planckian_locus((210.0, 50.0)) is False  # Saturated cool
        assert _is_on_planckian_locus((120.0, 80.0)) is False  # Green

    def test_saturated_colors_not_on_locus(self) -> None:
        """Test that saturated colors like red, green, blue are NOT on locus."""
        assert _is_on_planckian_locus((0.0, 100.0)) is False   # Red
        assert _is_on_planckian_locus((120.0, 100.0)) is False # Green
        assert _is_on_planckian_locus((240.0, 100.0)) is False # Blue

    def test_threshold_boundary(self) -> None:
        """Test behavior at saturation threshold boundary."""
        # At threshold (15.0) should be on locus
        assert _is_on_planckian_locus((35.0, 15.0)) is True
        # Just above threshold should not be on locus
        assert _is_on_planckian_locus((35.0, 16.0)) is False
```

**Step 2: Run tests to verify they fail**

Run: `cd /tmp/ha_fade_lights/.worktrees/pr4-planckian-locus && python -m pytest tests/test_planckian_locus.py -x -q`
Expected: FAIL with `ImportError: cannot import name '_is_on_planckian_locus'`

**Step 3: Commit**

```bash
cd /tmp/ha_fade_lights/.worktrees/pr4-planckian-locus && git add tests/test_planckian_locus.py && git commit -m "test: add failing tests for _is_on_planckian_locus"
```

---

### Task 3: Implement `_is_on_planckian_locus`

**Files:**
- Modify: `custom_components/fade_lights/__init__.py`

**Step 1: Add import for the constant**

Find the existing imports from `.const` and add `PLANCKIAN_LOCUS_SATURATION_THRESHOLD`:

```python
from .const import (
    ...existing imports...,
    PLANCKIAN_LOCUS_SATURATION_THRESHOLD,
)
```

**Step 2: Add the function**

Add this function after `_interpolate_hue` (around line 771):

```python
def _is_on_planckian_locus(hs_color: tuple[float, float]) -> bool:
    """Check if an HS color is on or near the Planckian locus.

    The Planckian locus represents the colors of blackbody radiation
    (color temperatures). Colors on the locus have low saturation
    (white/off-white appearance).

    Args:
        hs_color: Tuple of (hue 0-360, saturation 0-100)

    Returns:
        True if the color is close enough to the locus to transition
        directly to mireds-based fading.
    """
    _, saturation = hs_color
    return saturation <= PLANCKIAN_LOCUS_SATURATION_THRESHOLD
```

**Step 3: Run tests to verify they pass**

Run: `cd /tmp/ha_fade_lights/.worktrees/pr4-planckian-locus && python -m pytest tests/test_planckian_locus.py::TestIsOnPlanckianLocus -x -q`
Expected: All passed

**Step 4: Commit**

```bash
cd /tmp/ha_fade_lights/.worktrees/pr4-planckian-locus && git add custom_components/fade_lights/__init__.py && git commit -m "feat: add _is_on_planckian_locus detection function"
```

---

### Task 4: Write failing tests for `_hs_to_mireds`

**Files:**
- Modify: `tests/test_planckian_locus.py`

**Step 1: Add tests for HS to mireds conversion**

Append to the test file:

```python
from custom_components.fade_lights import _hs_to_mireds


class TestHsToMireds:
    """Test conversion from HS to approximate mireds using lookup table."""

    def test_cool_daylight_hue(self) -> None:
        """Test cool daylight hue maps to cool mireds."""
        # Hue ~220 is cool daylight (~6500K = 154 mireds)
        mireds = _hs_to_mireds((220.0, 5.0))
        assert 140 <= mireds <= 180

    def test_warm_white_hue(self) -> None:
        """Test warm white hue maps to warm mireds."""
        # Hue ~35 is warm white (~3000K = 333 mireds)
        mireds = _hs_to_mireds((35.0, 18.0))
        assert 300 <= mireds <= 370

    def test_neutral_white_hue(self) -> None:
        """Test neutral white maps to middle mireds."""
        # Hue ~42 is neutral (~4000K = 250 mireds)
        mireds = _hs_to_mireds((42.0, 8.0))
        assert 220 <= mireds <= 290

    def test_candlelight_hue(self) -> None:
        """Test very warm hue maps to high mireds."""
        # Hue ~28 is candlelight (~2000K = 500 mireds)
        mireds = _hs_to_mireds((28.0, 45.0))
        assert 450 <= mireds <= 550

    def test_pure_white_defaults_to_neutral(self) -> None:
        """Test pure white (0 saturation) returns neutral mireds."""
        mireds = _hs_to_mireds((0.0, 0.0))
        # Should return something reasonable in the middle range
        assert 200 <= mireds <= 400

    def test_returns_int(self) -> None:
        """Test that result is an integer."""
        mireds = _hs_to_mireds((35.0, 10.0))
        assert isinstance(mireds, int)
```

**Step 2: Run tests to verify they fail**

Run: `cd /tmp/ha_fade_lights/.worktrees/pr4-planckian-locus && python -m pytest tests/test_planckian_locus.py::TestHsToMireds -x -q`
Expected: FAIL with `ImportError: cannot import name '_hs_to_mireds'`

**Step 3: Commit**

```bash
cd /tmp/ha_fade_lights/.worktrees/pr4-planckian-locus && git add tests/test_planckian_locus.py && git commit -m "test: add failing tests for _hs_to_mireds"
```

---

### Task 5: Implement `_hs_to_mireds`

**Files:**
- Modify: `custom_components/fade_lights/__init__.py`

**Step 1: Add import for PLANCKIAN_LOCUS_HS**

Add `PLANCKIAN_LOCUS_HS` to the imports from `.const`.

**Step 2: Add the function**

Add this after `_is_on_planckian_locus`:

```python
def _hs_to_mireds(hs_color: tuple[float, float]) -> int:
    """Convert an HS color to approximate mireds using Planckian locus lookup.

    Finds the closest matching color temperature on the Planckian locus
    based on hue matching. Used when transitioning from HS to color temp.

    Args:
        hs_color: Tuple of (hue 0-360, saturation 0-100)

    Returns:
        Approximate color temperature in mireds
    """
    hue, saturation = hs_color

    # For very low saturation, return neutral white
    if saturation < 3:
        return 286  # ~3500K neutral white

    # Find closest match in the lookup table based on hue
    best_mireds = 286  # Default to neutral white
    best_distance = float("inf")

    for mireds, (locus_hue, _) in PLANCKIAN_LOCUS_HS:
        # Calculate hue distance (circular)
        distance = abs(hue - locus_hue)
        if distance > 180:
            distance = 360 - distance

        if distance < best_distance:
            best_distance = distance
            best_mireds = mireds

    return best_mireds
```

**Step 3: Run tests to verify they pass**

Run: `cd /tmp/ha_fade_lights/.worktrees/pr4-planckian-locus && python -m pytest tests/test_planckian_locus.py::TestHsToMireds -x -q`
Expected: All passed

**Step 4: Commit**

```bash
cd /tmp/ha_fade_lights/.worktrees/pr4-planckian-locus && git add custom_components/fade_lights/__init__.py && git commit -m "feat: add _hs_to_mireds conversion using Planckian locus lookup"
```

---

### Task 6: Write failing tests for `_mireds_to_hs`

**Files:**
- Modify: `tests/test_planckian_locus.py`

**Step 1: Add tests for mireds to HS conversion**

Append to the test file:

```python
from custom_components.fade_lights import _mireds_to_hs


class TestMiredsToHs:
    """Test conversion from mireds to HS using Planckian locus lookup."""

    def test_cool_daylight_mireds(self) -> None:
        """Test cool daylight mireds maps to cool hue."""
        # 154 mireds = 6500K = cool daylight
        hs = _mireds_to_hs(154)
        hue, sat = hs
        assert 200 <= hue <= 230  # Cool blue-ish hue
        assert sat < 15  # Low saturation

    def test_warm_white_mireds(self) -> None:
        """Test warm white mireds maps to warm hue."""
        # 333 mireds = 3000K = warm white
        hs = _mireds_to_hs(333)
        hue, sat = hs
        assert 30 <= hue <= 45  # Warm amber hue
        assert 10 <= sat <= 25

    def test_neutral_white_mireds(self) -> None:
        """Test neutral mireds maps to neutral hue."""
        # 286 mireds = 3500K = neutral
        hs = _mireds_to_hs(286)
        hue, sat = hs
        assert 35 <= hue <= 45
        assert 8 <= sat <= 18

    def test_candlelight_mireds(self) -> None:
        """Test candlelight mireds maps to very warm hue."""
        # 500 mireds = 2000K = candlelight
        hs = _mireds_to_hs(500)
        hue, sat = hs
        assert 25 <= hue <= 35  # Very warm amber
        assert sat >= 40  # Higher saturation

    def test_interpolation_between_points(self) -> None:
        """Test that values between lookup points are interpolated."""
        # 310 is between 303 (36, 15) and 333 (35, 18)
        hs = _mireds_to_hs(310)
        hue, sat = hs
        assert 35 <= hue <= 36
        assert 15 <= sat <= 18

    def test_extrapolation_below_range(self) -> None:
        """Test mireds below lookup range returns coolest value."""
        hs = _mireds_to_hs(100)  # Below 154
        hue, _ = hs
        assert 210 <= hue <= 230  # Should be cool

    def test_extrapolation_above_range(self) -> None:
        """Test mireds above lookup range returns warmest value."""
        hs = _mireds_to_hs(600)  # Above 500
        hue, sat = hs
        assert 25 <= hue <= 30  # Very warm
        assert sat >= 40

    def test_returns_tuple(self) -> None:
        """Test that result is a tuple of two floats."""
        hs = _mireds_to_hs(300)
        assert isinstance(hs, tuple)
        assert len(hs) == 2
        assert isinstance(hs[0], float)
        assert isinstance(hs[1], float)
```

**Step 2: Run tests to verify they fail**

Run: `cd /tmp/ha_fade_lights/.worktrees/pr4-planckian-locus && python -m pytest tests/test_planckian_locus.py::TestMiredsToHs -x -q`
Expected: FAIL with `ImportError: cannot import name '_mireds_to_hs'`

**Step 3: Commit**

```bash
cd /tmp/ha_fade_lights/.worktrees/pr4-planckian-locus && git add tests/test_planckian_locus.py && git commit -m "test: add failing tests for _mireds_to_hs"
```

---

### Task 7: Implement `_mireds_to_hs`

**Files:**
- Modify: `custom_components/fade_lights/__init__.py`

**Step 1: Add the function**

Add this after `_hs_to_mireds`:

```python
def _mireds_to_hs(mireds: int) -> tuple[float, float]:
    """Convert mireds to approximate HS using Planckian locus lookup.

    Interpolates between lookup table entries to find the HS color
    that corresponds to the given color temperature.

    Args:
        mireds: Color temperature in mireds

    Returns:
        Tuple of (hue 0-360, saturation 0-100)
    """
    # Handle values outside the lookup range
    if mireds <= PLANCKIAN_LOCUS_HS[0][0]:
        return PLANCKIAN_LOCUS_HS[0][1]
    if mireds >= PLANCKIAN_LOCUS_HS[-1][0]:
        return PLANCKIAN_LOCUS_HS[-1][1]

    # Find the two bracketing entries
    for i in range(len(PLANCKIAN_LOCUS_HS) - 1):
        lower_mireds, lower_hs = PLANCKIAN_LOCUS_HS[i]
        upper_mireds, upper_hs = PLANCKIAN_LOCUS_HS[i + 1]

        if lower_mireds <= mireds <= upper_mireds:
            # Interpolate between the two entries
            t = (mireds - lower_mireds) / (upper_mireds - lower_mireds)
            hue = lower_hs[0] + (upper_hs[0] - lower_hs[0]) * t
            sat = lower_hs[1] + (upper_hs[1] - lower_hs[1]) * t
            return (round(hue, 2), round(sat, 2))

    # Fallback (should not reach here)
    return (38.0, 12.0)  # Neutral white
```

**Step 2: Run tests to verify they pass**

Run: `cd /tmp/ha_fade_lights/.worktrees/pr4-planckian-locus && python -m pytest tests/test_planckian_locus.py::TestMiredsToHs -x -q`
Expected: All passed

**Step 3: Commit**

```bash
cd /tmp/ha_fade_lights/.worktrees/pr4-planckian-locus && git add custom_components/fade_lights/__init__.py && git commit -m "feat: add _mireds_to_hs conversion with interpolation"
```

---

### Task 8: Write failing tests for `_build_hs_to_mireds_steps`

**Files:**
- Modify: `tests/test_planckian_locus.py`

**Step 1: Add tests for hybrid step generation**

Append to the test file:

```python
from custom_components.fade_lights import _build_hs_to_mireds_steps
from custom_components.fade_lights.models import FadeStep


class TestBuildHsToMiredsSteps:
    """Test hybrid HS→mireds step generation."""

    def test_on_locus_hs_goes_straight_to_mireds(self) -> None:
        """Test that HS already on locus generates only mireds steps."""
        # Starting with low saturation (on locus) going to mireds
        steps = _build_hs_to_mireds_steps(
            start_hs=(35.0, 10.0),  # Warm white, on locus
            end_mireds=400,
            transition_ms=1000,
            min_step_delay_ms=100,
        )

        # All steps should have mireds, no HS
        for step in steps:
            assert step.color_temp_mireds is not None
            assert step.hs_color is None

        # Last step should be target mireds
        assert steps[-1].color_temp_mireds == 400

    def test_off_locus_hs_transitions_through_locus(self) -> None:
        """Test that HS off locus first fades toward locus, then to mireds."""
        # Starting with saturated red, going to warm white mireds
        steps = _build_hs_to_mireds_steps(
            start_hs=(0.0, 80.0),  # Saturated red, NOT on locus
            end_mireds=333,  # Warm white
            transition_ms=1000,
            min_step_delay_ms=100,
        )

        # Should have some steps with HS (fading toward locus)
        hs_steps = [s for s in steps if s.hs_color is not None]
        assert len(hs_steps) > 0

        # Should end with mireds steps
        mireds_steps = [s for s in steps if s.color_temp_mireds is not None]
        assert len(mireds_steps) > 0

        # Last step should be target mireds
        assert steps[-1].color_temp_mireds == 333
        assert steps[-1].hs_color is None

    def test_hs_saturation_decreases_toward_locus(self) -> None:
        """Test that HS saturation decreases as we approach the locus."""
        steps = _build_hs_to_mireds_steps(
            start_hs=(120.0, 100.0),  # Saturated green
            end_mireds=286,  # Neutral white
            transition_ms=1000,
            min_step_delay_ms=100,
        )

        # Get the HS steps
        hs_steps = [s for s in steps if s.hs_color is not None]
        if len(hs_steps) > 1:
            saturations = [s.hs_color[1] for s in hs_steps]
            # Saturation should generally decrease
            assert saturations[-1] < saturations[0]

    def test_minimum_one_step(self) -> None:
        """Test at least one step is generated."""
        steps = _build_hs_to_mireds_steps(
            start_hs=(35.0, 5.0),  # Already near target
            end_mireds=333,
            transition_ms=100,
            min_step_delay_ms=100,
        )

        assert len(steps) >= 1

    def test_step_count_respects_timing(self) -> None:
        """Test step count is limited by transition time."""
        steps = _build_hs_to_mireds_steps(
            start_hs=(0.0, 80.0),
            end_mireds=333,
            transition_ms=300,
            min_step_delay_ms=100,
        )

        # Max 3 steps (300ms / 100ms)
        assert len(steps) <= 3

    def test_returns_list_of_fade_steps(self) -> None:
        """Test return type is list of FadeStep."""
        steps = _build_hs_to_mireds_steps(
            start_hs=(35.0, 10.0),
            end_mireds=333,
            transition_ms=1000,
            min_step_delay_ms=100,
        )

        assert isinstance(steps, list)
        for step in steps:
            assert isinstance(step, FadeStep)
```

**Step 2: Run tests to verify they fail**

Run: `cd /tmp/ha_fade_lights/.worktrees/pr4-planckian-locus && python -m pytest tests/test_planckian_locus.py::TestBuildHsToMiredsSteps -x -q`
Expected: FAIL with `ImportError: cannot import name '_build_hs_to_mireds_steps'`

**Step 3: Commit**

```bash
cd /tmp/ha_fade_lights/.worktrees/pr4-planckian-locus && git add tests/test_planckian_locus.py && git commit -m "test: add failing tests for _build_hs_to_mireds_steps"
```

---

### Task 9: Implement `_build_hs_to_mireds_steps`

**Files:**
- Modify: `custom_components/fade_lights/__init__.py`

**Step 1: Add the function**

Add this after `_mireds_to_hs`:

```python
def _build_hs_to_mireds_steps(
    start_hs: tuple[float, float],
    end_mireds: int,
    transition_ms: int,
    min_step_delay_ms: int,
) -> list[FadeStep]:
    """Build hybrid step sequence from HS color to mireds.

    If the starting HS is already on the Planckian locus, generates
    pure mireds-based steps. Otherwise, first fades the HS toward
    the locus (reducing saturation), then switches to mireds.

    Args:
        start_hs: Starting (hue, saturation)
        end_mireds: Target color temperature in mireds
        transition_ms: Total transition time in milliseconds
        min_step_delay_ms: Minimum delay between steps

    Returns:
        List of FadeStep objects transitioning from HS to mireds
    """
    max_steps = max(1, transition_ms // min_step_delay_ms)

    # If already on locus, just do mireds-based fading
    if _is_on_planckian_locus(start_hs):
        start_mireds = _hs_to_mireds(start_hs)
        return _build_fade_steps(
            start_brightness=None,
            end_brightness=None,
            start_hs=None,
            end_hs=None,
            start_mireds=start_mireds,
            end_mireds=end_mireds,
            transition_ms=transition_ms,
            min_step_delay_ms=min_step_delay_ms,
        )

    # Off locus: need to transition HS toward locus first, then to mireds
    # Get the target HS on the locus (what the end_mireds looks like in HS)
    target_locus_hs = _mireds_to_hs(end_mireds)

    # Calculate how much of the transition is HS->locus vs locus->mireds
    # Use 70% of steps for HS transition, 30% for final mireds adjustment
    hs_steps_count = max(1, int(max_steps * 0.7))
    mireds_steps_count = max(1, max_steps - hs_steps_count)

    steps = []

    # Phase 1: HS toward the locus target
    for i in range(1, hs_steps_count + 1):
        t = i / hs_steps_count
        hue = _interpolate_hue(start_hs[0], target_locus_hs[0], t)
        sat = start_hs[1] + (target_locus_hs[1] - start_hs[1]) * t
        steps.append(FadeStep(hs_color=(round(hue, 2), round(sat, 2))))

    # Phase 2: Mireds from locus point to target
    locus_mireds = _hs_to_mireds(target_locus_hs)
    for i in range(1, mireds_steps_count + 1):
        t = i / mireds_steps_count
        mireds = round(locus_mireds + (end_mireds - locus_mireds) * t)
        steps.append(FadeStep(color_temp_mireds=mireds))

    return steps
```

**Step 2: Run tests to verify they pass**

Run: `cd /tmp/ha_fade_lights/.worktrees/pr4-planckian-locus && python -m pytest tests/test_planckian_locus.py::TestBuildHsToMiredsSteps -x -q`
Expected: All passed

**Step 3: Commit**

```bash
cd /tmp/ha_fade_lights/.worktrees/pr4-planckian-locus && git add custom_components/fade_lights/__init__.py && git commit -m "feat: add _build_hs_to_mireds_steps for hybrid transitions"
```

---

### Task 10: Run full test suite and lint

**Step 1: Run all tests**

Run: `cd /tmp/ha_fade_lights/.worktrees/pr4-planckian-locus && python -m pytest tests/ -q`
Expected: All 160+ tests pass

**Step 2: Run linting**

Run: `cd /tmp/ha_fade_lights/.worktrees/pr4-planckian-locus && ruff check custom_components/ tests/ && ruff format --check custom_components/ tests/`
Expected: No errors

**Step 3: Fix any lint issues if needed**

If ruff format fails:
```bash
cd /tmp/ha_fade_lights/.worktrees/pr4-planckian-locus && ruff format custom_components/ tests/
git add -A && git commit -m "style: fix formatting issues"
```

---

### Task 11: Push and verify

**Step 1: Run full test suite one final time**

Run: `cd /tmp/ha_fade_lights/.worktrees/pr4-planckian-locus && python -m pytest tests/ -q`
Expected: All tests pass

**Step 2: Push**

```bash
cd /tmp/ha_fade_lights/.worktrees/pr4-planckian-locus && git push -u origin pr4-planckian-locus
```

---

## Summary

This PR implements:

1. **`PLANCKIAN_LOCUS_HS`** - Lookup table mapping mireds to HS colors on the Planckian locus
2. **`_is_on_planckian_locus(hs)`** - Detects if an HS color is on/near the locus (low saturation)
3. **`_hs_to_mireds(hs)`** - Converts HS to approximate mireds using hue matching
4. **`_mireds_to_hs(mireds)`** - Converts mireds to HS with interpolation between table entries
5. **`_build_hs_to_mireds_steps(...)`** - Generates hybrid step sequences for HS→mireds transitions

These functions will be used by PR5 (Execute Fade with Colors) to handle the complex case of fading from an arbitrary color to a color temperature.
