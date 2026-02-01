# Mireds-to-HS Hybrid Fade Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add smooth color transitions from color temperature mode to HS color mode by fading through the Planckian locus.

**Architecture:** When light is in COLOR_TEMP hardware mode and target is HS color, first fade mireds along the locus toward target hue (~30% of steps), then switch to HS and fade to final target (~70% of steps). Mirrors existing HS-to-mireds hybrid fade.

**Tech Stack:** Python, pytest, Home Assistant light component

---

### Task 1: Add ATTR_COLOR_MODE import

**Files:**
- Modify: `custom_components/fade_lights/__init__.py:9-12`

**Step 1: Add the import**

Add `ATTR_COLOR_MODE` to the imports from `homeassistant.components.light`:

```python
from homeassistant.components.light import (
    ATTR_BRIGHTNESS,
    ATTR_COLOR_MODE,
    ATTR_SUPPORTED_COLOR_MODES,
)
```

**Step 2: Verify no import errors**

Run: `python -c "from custom_components.fade_lights import *"`
Expected: No errors

**Step 3: Commit**

```bash
git add custom_components/fade_lights/__init__.py
git commit -m "feat: add ATTR_COLOR_MODE import for mireds-to-HS detection"
```

---

### Task 2: Write unit tests for _build_mireds_to_hs_steps

**Files:**
- Create: `tests/test_mireds_to_hs_steps.py`

**Step 1: Write the test file**

```python
"""Tests for _build_mireds_to_hs_steps function."""

import pytest

from custom_components.fade_lights import _build_mireds_to_hs_steps


class TestBuildMiredsToHsSteps:
    """Tests for the mireds-to-HS step builder."""

    def test_basic_transition_generates_steps(self):
        """Test that basic transition generates both mireds and HS steps."""
        # 3000K (333 mireds) to red (0, 100)
        steps = _build_mireds_to_hs_steps(
            start_mireds=333,
            end_hs=(0.0, 100.0),
            transition_ms=10000,
            min_step_delay_ms=100,
        )

        assert len(steps) > 0

        # First steps should have mireds, no HS
        assert steps[0].color_temp_mireds is not None
        assert steps[0].hs_color is None

        # Last steps should have HS, no mireds
        assert steps[-1].hs_color is not None
        assert steps[-1].color_temp_mireds is None

        # Final step should be the target
        assert steps[-1].hs_color == (0.0, 100.0)

    def test_phase_split_approximately_30_70(self):
        """Test that steps are split roughly 30% mireds, 70% HS."""
        steps = _build_mireds_to_hs_steps(
            start_mireds=333,
            end_hs=(0.0, 100.0),
            transition_ms=10000,
            min_step_delay_ms=100,
        )

        mireds_steps = [s for s in steps if s.color_temp_mireds is not None]
        hs_steps = [s for s in steps if s.hs_color is not None]

        total = len(steps)
        mireds_ratio = len(mireds_steps) / total
        hs_ratio = len(hs_steps) / total

        # Allow some tolerance: 20-40% mireds, 60-80% HS
        assert 0.2 <= mireds_ratio <= 0.4, f"Mireds ratio {mireds_ratio} out of range"
        assert 0.6 <= hs_ratio <= 0.8, f"HS ratio {hs_ratio} out of range"

    def test_mireds_steps_move_toward_target_hue(self):
        """Test that mireds steps move along locus toward target."""
        # Start at warm (333 mireds = 3000K), target blue-ish hue
        steps = _build_mireds_to_hs_steps(
            start_mireds=333,
            end_hs=(240.0, 80.0),  # Blue
            transition_ms=10000,
            min_step_delay_ms=100,
        )

        mireds_steps = [s for s in steps if s.color_temp_mireds is not None]
        assert len(mireds_steps) >= 2

        # Mireds should be changing (moving along locus)
        first_mireds = mireds_steps[0].color_temp_mireds
        last_mireds = mireds_steps[-1].color_temp_mireds
        assert first_mireds != last_mireds

    def test_hs_steps_end_at_target(self):
        """Test that HS steps interpolate toward target."""
        target_hs = (120.0, 75.0)  # Green
        steps = _build_mireds_to_hs_steps(
            start_mireds=250,
            end_hs=target_hs,
            transition_ms=5000,
            min_step_delay_ms=100,
        )

        hs_steps = [s for s in steps if s.hs_color is not None]
        assert len(hs_steps) >= 2

        # Last HS step should match target
        final_hs = hs_steps[-1].hs_color
        assert abs(final_hs[0] - target_hs[0]) < 0.1
        assert abs(final_hs[1] - target_hs[1]) < 0.1

    def test_no_brightness_in_steps(self):
        """Test that steps don't include brightness (handled separately)."""
        steps = _build_mireds_to_hs_steps(
            start_mireds=333,
            end_hs=(0.0, 100.0),
            transition_ms=5000,
            min_step_delay_ms=100,
        )

        for step in steps:
            assert step.brightness is None
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mireds_to_hs_steps.py -v`
Expected: FAIL with "cannot import name '_build_mireds_to_hs_steps'"

**Step 3: Commit the tests**

```bash
git add tests/test_mireds_to_hs_steps.py
git commit -m "test: add unit tests for _build_mireds_to_hs_steps"
```

---

### Task 3: Implement _build_mireds_to_hs_steps function

**Files:**
- Modify: `custom_components/fade_lights/__init__.py` (add function after `_build_hs_to_mireds_steps`)

**Step 1: Add the function**

Add after `_build_hs_to_mireds_steps` (around line 947):

```python
def _build_mireds_to_hs_steps(
    start_mireds: int,
    end_hs: tuple[float, float],
    transition_ms: int,
    min_step_delay_ms: int,
) -> list[FadeStep]:
    """Build hybrid step sequence from mireds to HS color.

    When a light is in color temp mode and needs to transition to an HS color,
    this function creates a smooth transition:
    1. First, fade mireds along the Planckian locus toward the target hue (~30%)
    2. Then, switch to HS and fade to the final target (~70%)

    This is the symmetric counterpart to _build_hs_to_mireds_steps.

    Args:
        start_mireds: Starting color temperature in mireds
        end_hs: Target (hue, saturation)
        transition_ms: Total transition time in milliseconds
        min_step_delay_ms: Minimum delay between steps

    Returns:
        List of FadeStep objects transitioning from mireds to HS
    """
    max_steps = max(1, transition_ms // min_step_delay_ms)

    # Find the locus point closest to the target hue
    target_locus_mireds = _hs_to_mireds(end_hs)

    # If already at or very close to target mireds, skip mireds phase
    if abs(start_mireds - target_locus_mireds) < 10:
        # Just do HS fade from locus point
        start_hs = _mireds_to_hs(start_mireds)
        return _build_fade_steps(
            start_brightness=None,
            end_brightness=None,
            start_hs=start_hs,
            end_hs=end_hs,
            start_mireds=None,
            end_mireds=None,
            transition_ms=transition_ms,
            min_step_delay_ms=min_step_delay_ms,
        )

    # Split: ~30% mireds, ~70% HS (opposite of HS->mireds which is 70/30)
    mireds_steps_count = max(1, int(max_steps * 0.3))
    hs_steps_count = max(1, max_steps - mireds_steps_count)

    steps = []

    # Phase 1: Mireds along locus toward target hue
    for i in range(1, mireds_steps_count + 1):
        t = i / mireds_steps_count
        mireds = round(start_mireds + (target_locus_mireds - start_mireds) * t)
        steps.append(FadeStep(color_temp_mireds=mireds))

    # Phase 2: HS from locus point to final target
    locus_hs = _mireds_to_hs(target_locus_mireds)
    for i in range(1, hs_steps_count + 1):
        t = i / hs_steps_count
        hue = _interpolate_hue(locus_hs[0], end_hs[0], t)
        sat = locus_hs[1] + (end_hs[1] - locus_hs[1]) * t
        steps.append(FadeStep(hs_color=(round(hue, 2), round(sat, 2))))

    return steps
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/test_mireds_to_hs_steps.py -v`
Expected: All 5 tests PASS

**Step 3: Commit**

```bash
git add custom_components/fade_lights/__init__.py
git commit -m "feat: implement _build_mireds_to_hs_steps function"
```

---

### Task 4: Write integration tests for COLOR_TEMP to HS fade

**Files:**
- Create: `tests/test_mireds_to_hs_fade.py`

**Step 1: Write the integration test file**

```python
"""Integration tests for mireds-to-HS hybrid fade transitions."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.components.light.const import ColorMode
from homeassistant.const import STATE_ON

from custom_components.fade_lights import _execute_fade, FadeParams
from custom_components.fade_lights.models import FadeStep


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    hass = MagicMock()
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    return hass


@pytest.fixture
def color_temp_light_state():
    """Create a light state in COLOR_TEMP mode."""
    state = MagicMock()
    state.state = STATE_ON
    state.attributes = {
        "brightness": 200,
        "color_mode": ColorMode.COLOR_TEMP,
        "color_temp": 333,  # ~3000K in mireds
        "supported_color_modes": [ColorMode.COLOR_TEMP, ColorMode.HS],
    }
    return state


class TestMiredsToHsFade:
    """Integration tests for mireds-to-HS transitions."""

    @pytest.mark.asyncio
    async def test_color_temp_to_hs_uses_hybrid_fade(
        self, mock_hass, color_temp_light_state
    ):
        """Test that COLOR_TEMP mode light fading to HS uses hybrid transition."""
        mock_hass.states.get = MagicMock(return_value=color_temp_light_state)

        cancel_event = MagicMock()
        cancel_event.is_set = MagicMock(return_value=False)

        fade_params = FadeParams(
            brightness_pct=None,
            hs_color=(0.0, 100.0),  # Red
        )

        with patch(
            "custom_components.fade_lights._build_mireds_to_hs_steps"
        ) as mock_builder:
            mock_builder.return_value = [
                FadeStep(color_temp_mireds=300),
                FadeStep(hs_color=(30.0, 20.0)),
                FadeStep(hs_color=(0.0, 100.0)),
            ]

            await _execute_fade(
                mock_hass,
                "light.test",
                fade_params,
                transition_ms=3000,
                min_step_delay_ms=100,
                cancel_event=cancel_event,
            )

            # Verify hybrid builder was called
            mock_builder.assert_called_once()
            call_args = mock_builder.call_args
            assert call_args[0][0] == 333  # start_mireds
            assert call_args[0][1] == (0.0, 100.0)  # end_hs

    @pytest.mark.asyncio
    async def test_hs_mode_light_does_not_use_hybrid(self, mock_hass):
        """Test that HS mode light uses standard fade, not hybrid."""
        hs_state = MagicMock()
        hs_state.state = STATE_ON
        hs_state.attributes = {
            "brightness": 200,
            "color_mode": ColorMode.HS,
            "hs_color": (200.0, 50.0),
            "supported_color_modes": [ColorMode.COLOR_TEMP, ColorMode.HS],
        }
        mock_hass.states.get = MagicMock(return_value=hs_state)

        cancel_event = MagicMock()
        cancel_event.is_set = MagicMock(return_value=False)

        fade_params = FadeParams(
            brightness_pct=None,
            hs_color=(0.0, 100.0),
        )

        with patch(
            "custom_components.fade_lights._build_mireds_to_hs_steps"
        ) as mock_hybrid, patch(
            "custom_components.fade_lights._build_fade_steps"
        ) as mock_standard:
            mock_standard.return_value = [FadeStep(hs_color=(0.0, 100.0))]

            await _execute_fade(
                mock_hass,
                "light.test",
                fade_params,
                transition_ms=3000,
                min_step_delay_ms=100,
                cancel_event=cancel_event,
            )

            # Hybrid should NOT be called
            mock_hybrid.assert_not_called()

    @pytest.mark.asyncio
    async def test_color_temp_to_mireds_uses_standard_fade(self, mock_hass, color_temp_light_state):
        """Test that COLOR_TEMP to mireds uses standard fade (no mode switch needed)."""
        mock_hass.states.get = MagicMock(return_value=color_temp_light_state)

        cancel_event = MagicMock()
        cancel_event.is_set = MagicMock(return_value=False)

        fade_params = FadeParams(
            brightness_pct=None,
            color_temp_mireds=200,  # Target is mireds, not HS
        )

        with patch(
            "custom_components.fade_lights._build_mireds_to_hs_steps"
        ) as mock_hybrid:
            await _execute_fade(
                mock_hass,
                "light.test",
                fade_params,
                transition_ms=3000,
                min_step_delay_ms=100,
                cancel_event=cancel_event,
            )

            # Hybrid should NOT be called (staying in mireds mode)
            mock_hybrid.assert_not_called()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mireds_to_hs_fade.py -v`
Expected: FAIL (hybrid builder not called because integration not complete)

**Step 3: Commit the tests**

```bash
git add tests/test_mireds_to_hs_fade.py
git commit -m "test: add integration tests for mireds-to-HS fade"
```

---

### Task 5: Integrate _build_mireds_to_hs_steps into _execute_fade

**Files:**
- Modify: `custom_components/fade_lights/__init__.py` (update `_execute_fade` step builder selection)

**Step 1: Update the step builder selection logic**

Find the step builder selection block in `_execute_fade` (around line 536-557) and update it:

```python
    # Get current color mode for hybrid transition detection
    color_mode = state.attributes.get("color_mode")

    # Determine which step builder to use
    if start_hs is not None and end_mireds is not None and not _is_on_planckian_locus(start_hs):
        # Hybrid HS -> mireds transition
        steps = _build_hs_to_mireds_steps(start_hs, end_mireds, transition_ms, min_step_delay_ms)
        # If also fading brightness, add it to each step
        if brightness_changing and end_brightness is not None:
            num_steps = len(steps)
            for i, step in enumerate(steps):
                t = (i + 1) / num_steps
                step.brightness = round(start_brightness + (end_brightness - start_brightness) * t)
    elif color_mode == ColorMode.COLOR_TEMP and end_hs is not None:
        # Hybrid mireds -> HS transition
        steps = _build_mireds_to_hs_steps(
            start_mireds if start_mireds is not None else 333,  # Default to ~3000K
            end_hs,
            transition_ms,
            min_step_delay_ms,
        )
        # If also fading brightness, add it to each step
        if brightness_changing and end_brightness is not None:
            num_steps = len(steps)
            for i, step in enumerate(steps):
                t = (i + 1) / num_steps
                step.brightness = round(start_brightness + (end_brightness - start_brightness) * t)
    else:
        # Standard fade using _build_fade_steps
        steps = _build_fade_steps(
            start_brightness=start_brightness if brightness_changing else None,
            end_brightness=end_brightness if brightness_changing else None,
            start_hs=start_hs if hs_changing else None,
            end_hs=end_hs if hs_changing else None,
            start_mireds=start_mireds if mireds_changing else None,
            end_mireds=end_mireds if mireds_changing else None,
            transition_ms=transition_ms,
            min_step_delay_ms=min_step_delay_ms,
        )
```

**Step 2: Add ColorMode import if not already present**

Verify this import exists (should already be there):
```python
from homeassistant.components.light.const import ColorMode
```

**Step 3: Run integration tests**

Run: `pytest tests/test_mireds_to_hs_fade.py -v`
Expected: All 3 tests PASS

**Step 4: Run all tests**

Run: `pytest tests/ -x -q`
Expected: All tests pass (209 + new tests)

**Step 5: Commit**

```bash
git add custom_components/fade_lights/__init__.py
git commit -m "feat: integrate mireds-to-HS hybrid fade into _execute_fade"
```

---

### Task 6: Run full test suite and fix any issues

**Files:**
- All test files

**Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass

**Step 2: Run linting**

Run: `python -m ruff check custom_components/fade_lights/`
Expected: No errors (or fix any that appear)

**Step 3: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address any linting or test issues"
```

---

### Task 7: Create summary commit

**Step 1: Review all changes**

Run: `git log --oneline color_support..HEAD`

**Step 2: Verify test count**

Run: `pytest tests/ -q`
Expected: ~217 tests pass (209 base + ~8 new)

**Step 3: Push branch**

```bash
git push -u origin feature/mireds-to-hs
```