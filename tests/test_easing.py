"""Tests for the easing module."""

from __future__ import annotations

import math

import pytest

from custom_components.fado.easing import (
    EASING_FUNCTIONS,
    auto_select_easing,
    ease_in_cubic,
    ease_in_out_sine,
    ease_in_quad,
    ease_out_cubic,
    ease_out_quad,
    get_easing_func,
    linear,
)


class TestLinear:
    """Test linear easing function."""

    def test_at_zero(self) -> None:
        """Test linear returns 0 at t=0."""
        assert linear(0.0) == 0.0

    def test_at_one(self) -> None:
        """Test linear returns 1 at t=1."""
        assert linear(1.0) == 1.0

    def test_at_midpoint(self) -> None:
        """Test linear returns 0.5 at t=0.5."""
        assert linear(0.5) == 0.5

    def test_passthrough(self) -> None:
        """Test linear is identity function."""
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert linear(t) == t


class TestEaseInQuad:
    """Test quadratic ease-in function."""

    def test_at_zero(self) -> None:
        """Test ease_in_quad returns 0 at t=0."""
        assert ease_in_quad(0.0) == 0.0

    def test_at_one(self) -> None:
        """Test ease_in_quad returns 1 at t=1."""
        assert ease_in_quad(1.0) == 1.0

    def test_at_midpoint(self) -> None:
        """Test ease_in_quad returns 0.25 at t=0.5 (t^2)."""
        assert ease_in_quad(0.5) == 0.25

    def test_slower_than_linear_at_start(self) -> None:
        """Test ease_in_quad is slower than linear at start."""
        for t in [0.1, 0.2, 0.3, 0.4]:
            assert ease_in_quad(t) < linear(t)

    def test_faster_than_linear_at_end(self) -> None:
        """Test ease_in_quad catches up to linear near end."""
        # At t=0.9, ease_in = 0.81 vs linear = 0.9
        assert ease_in_quad(0.9) < linear(0.9)


class TestEaseInCubic:
    """Test cubic ease-in function."""

    def test_at_zero(self) -> None:
        """Test ease_in_cubic returns 0 at t=0."""
        assert ease_in_cubic(0.0) == 0.0

    def test_at_one(self) -> None:
        """Test ease_in_cubic returns 1 at t=1."""
        assert ease_in_cubic(1.0) == 1.0

    def test_at_midpoint(self) -> None:
        """Test ease_in_cubic returns 0.125 at t=0.5 (t^3)."""
        assert ease_in_cubic(0.5) == 0.125

    def test_slower_than_quad_at_start(self) -> None:
        """Test ease_in_cubic is slower than ease_in_quad at start."""
        for t in [0.2, 0.3, 0.4, 0.5]:
            assert ease_in_cubic(t) < ease_in_quad(t)


class TestEaseOutQuad:
    """Test quadratic ease-out function."""

    def test_at_zero(self) -> None:
        """Test ease_out_quad returns 0 at t=0."""
        assert ease_out_quad(0.0) == 0.0

    def test_at_one(self) -> None:
        """Test ease_out_quad returns 1 at t=1."""
        assert ease_out_quad(1.0) == 1.0

    def test_at_midpoint(self) -> None:
        """Test ease_out_quad returns 0.75 at t=0.5."""
        assert ease_out_quad(0.5) == 0.75

    def test_faster_than_linear_at_start(self) -> None:
        """Test ease_out_quad is faster than linear at start."""
        for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
            assert ease_out_quad(t) > linear(t)


class TestEaseOutCubic:
    """Test cubic ease-out function."""

    def test_at_zero(self) -> None:
        """Test ease_out_cubic returns 0 at t=0."""
        assert ease_out_cubic(0.0) == 0.0

    def test_at_one(self) -> None:
        """Test ease_out_cubic returns 1 at t=1."""
        assert ease_out_cubic(1.0) == 1.0

    def test_at_midpoint(self) -> None:
        """Test ease_out_cubic returns 0.875 at t=0.5."""
        assert ease_out_cubic(0.5) == 0.875

    def test_faster_than_quad_at_start(self) -> None:
        """Test ease_out_cubic is faster than ease_out_quad at start."""
        for t in [0.1, 0.2, 0.3, 0.4]:
            assert ease_out_cubic(t) > ease_out_quad(t)


class TestEaseInOutSine:
    """Test sinusoidal ease-in-out function."""

    def test_at_zero(self) -> None:
        """Test ease_in_out_sine returns 0 at t=0."""
        assert ease_in_out_sine(0.0) == pytest.approx(0.0)

    def test_at_one(self) -> None:
        """Test ease_in_out_sine returns 1 at t=1."""
        assert ease_in_out_sine(1.0) == pytest.approx(1.0)

    def test_at_midpoint(self) -> None:
        """Test ease_in_out_sine returns 0.5 at t=0.5."""
        assert ease_in_out_sine(0.5) == pytest.approx(0.5)

    def test_s_curve_shape(self) -> None:
        """Test ease_in_out_sine has S-curve shape."""
        # Slower than linear at start (easing in)
        assert ease_in_out_sine(0.25) < linear(0.25)
        # Faster than linear at end (easing out)
        assert ease_in_out_sine(0.75) > linear(0.75)

    def test_symmetric(self) -> None:
        """Test ease_in_out_sine is symmetric around midpoint."""
        for t in [0.1, 0.2, 0.3, 0.4]:
            # f(t) + f(1-t) should equal 1 for symmetric curves
            assert ease_in_out_sine(t) + ease_in_out_sine(1 - t) == pytest.approx(1.0)


class TestGetEasingFunc:
    """Test get_easing_func lookup function."""

    def test_returns_linear(self) -> None:
        """Test returns linear function for 'linear' name."""
        func = get_easing_func("linear")
        assert func is linear

    def test_returns_ease_in_quad(self) -> None:
        """Test returns ease_in_quad function."""
        func = get_easing_func("ease_in_quad")
        assert func is ease_in_quad

    def test_returns_ease_in_cubic(self) -> None:
        """Test returns ease_in_cubic function."""
        func = get_easing_func("ease_in_cubic")
        assert func is ease_in_cubic

    def test_returns_ease_out_quad(self) -> None:
        """Test returns ease_out_quad function."""
        func = get_easing_func("ease_out_quad")
        assert func is ease_out_quad

    def test_returns_ease_out_cubic(self) -> None:
        """Test returns ease_out_cubic function."""
        func = get_easing_func("ease_out_cubic")
        assert func is ease_out_cubic

    def test_returns_ease_in_out_sine(self) -> None:
        """Test returns ease_in_out_sine function."""
        func = get_easing_func("ease_in_out_sine")
        assert func is ease_in_out_sine

    def test_unknown_name_returns_linear(self) -> None:
        """Test unknown curve name defaults to linear."""
        func = get_easing_func("unknown_curve")
        assert func is linear

    def test_empty_string_returns_linear(self) -> None:
        """Test empty string defaults to linear."""
        func = get_easing_func("")
        assert func is linear


class TestAutoSelectEasing:
    """Test auto_select_easing direction-based selection."""

    def test_fading_up_from_zero(self) -> None:
        """Test fading up from 0 selects ease_in_quad (slow start)."""
        result = auto_select_easing(0, 100)
        assert result == "ease_in_quad"

    def test_fading_up_from_zero_to_max(self) -> None:
        """Test fading from 0 to 255 selects ease_in_quad (slow start)."""
        result = auto_select_easing(0, 255)
        assert result == "ease_in_quad"

    def test_fading_down_to_zero(self) -> None:
        """Test fading down to 0 selects ease_out_quad (slow end)."""
        result = auto_select_easing(100, 0)
        assert result == "ease_out_quad"

    def test_fading_down_from_max_to_zero(self) -> None:
        """Test fading from 255 to 0 selects ease_out_quad (slow end)."""
        result = auto_select_easing(255, 0)
        assert result == "ease_out_quad"

    def test_mid_range_fade_up(self) -> None:
        """Test mid-range fade up selects ease_in_quad (slow start)."""
        result = auto_select_easing(50, 200)
        assert result == "ease_in_quad"

    def test_mid_range_fade_down(self) -> None:
        """Test mid-range fade down selects ease_out_quad (slow end)."""
        result = auto_select_easing(200, 50)
        assert result == "ease_out_quad"

    def test_same_brightness(self) -> None:
        """Test same start/end brightness selects ease_in_out_sine."""
        result = auto_select_easing(100, 100)
        assert result == "ease_in_out_sine"

    def test_small_non_zero_start(self) -> None:
        """Test fading up from 1 to 255 selects ease_in_quad (slow start)."""
        result = auto_select_easing(1, 255)
        assert result == "ease_in_quad"

    def test_small_non_zero_end(self) -> None:
        """Test fading down to 1 selects ease_out_quad (slow end)."""
        result = auto_select_easing(255, 1)
        assert result == "ease_out_quad"


class TestEasingFunctionsDict:
    """Test EASING_FUNCTIONS dictionary."""

    def test_contains_all_curves(self) -> None:
        """Test EASING_FUNCTIONS contains all expected curves."""
        expected = {
            "linear",
            "ease_in_quad",
            "ease_in_cubic",
            "ease_out_quad",
            "ease_out_cubic",
            "ease_in_out_sine",
        }
        assert set(EASING_FUNCTIONS.keys()) == expected

    def test_all_functions_callable(self) -> None:
        """Test all functions in EASING_FUNCTIONS are callable."""
        for name, func in EASING_FUNCTIONS.items():
            assert callable(func), f"{name} is not callable"

    def test_all_functions_return_float(self) -> None:
        """Test all functions return float for float input."""
        for name, func in EASING_FUNCTIONS.items():
            result = func(0.5)
            assert isinstance(result, float), f"{name} did not return float"


class TestEasingCurveProperties:
    """Test mathematical properties of easing curves."""

    @pytest.mark.parametrize("func_name", EASING_FUNCTIONS.keys())
    def test_boundary_zero(self, func_name: str) -> None:
        """Test all curves return 0 at t=0."""
        func = EASING_FUNCTIONS[func_name]
        assert func(0.0) == pytest.approx(0.0)

    @pytest.mark.parametrize("func_name", EASING_FUNCTIONS.keys())
    def test_boundary_one(self, func_name: str) -> None:
        """Test all curves return 1 at t=1."""
        func = EASING_FUNCTIONS[func_name]
        assert func(1.0) == pytest.approx(1.0)

    @pytest.mark.parametrize("func_name", EASING_FUNCTIONS.keys())
    def test_monotonic_increasing(self, func_name: str) -> None:
        """Test all curves are monotonically increasing."""
        func = EASING_FUNCTIONS[func_name]
        prev = 0.0
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            current = func(t)
            assert current >= prev, f"{func_name} not monotonic at t={t}"
            prev = current

    @pytest.mark.parametrize("func_name", EASING_FUNCTIONS.keys())
    def test_bounded_zero_to_one(self, func_name: str) -> None:
        """Test all curves stay in [0, 1] range for t in [0, 1]."""
        func = EASING_FUNCTIONS[func_name]
        for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            result = func(t)
            assert 0.0 <= result <= 1.0, f"{func_name} out of bounds at t={t}: {result}"
