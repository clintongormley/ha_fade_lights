"""Easing functions for smooth light fading transitions."""

from __future__ import annotations

import math
from collections.abc import Callable

# Type alias for easing functions
EasingFunc = Callable[[float], float]


def linear(t: float) -> float:
    """Linear easing - no acceleration."""
    return t


def ease_in_quad(t: float) -> float:
    """Quadratic ease-in - accelerating from zero velocity."""
    return t * t


def ease_in_cubic(t: float) -> float:
    """Cubic ease-in - accelerating from zero velocity."""
    return t * t * t


def ease_out_quad(t: float) -> float:
    """Quadratic ease-out - decelerating to zero velocity."""
    return 1 - (1 - t) ** 2


def ease_out_cubic(t: float) -> float:
    """Cubic ease-out - decelerating to zero velocity."""
    return 1 - (1 - t) ** 3


def ease_in_out_sine(t: float) -> float:
    """Sinusoidal ease-in-out - smooth acceleration and deceleration."""
    return -(math.cos(math.pi * t) - 1) / 2


# Mapping of easing function names to their implementations
EASING_FUNCTIONS: dict[str, EasingFunc] = {
    "linear": linear,
    "ease_in_quad": ease_in_quad,
    "ease_in_cubic": ease_in_cubic,
    "ease_out_quad": ease_out_quad,
    "ease_out_cubic": ease_out_cubic,
    "ease_in_out_sine": ease_in_out_sine,
}


def get_easing_func(name: str) -> EasingFunc:
    """Get an easing function by name.

    Args:
        name: The name of the easing function.

    Returns:
        The easing function, or linear if the name is not found.
    """
    return EASING_FUNCTIONS.get(name, linear)


def auto_select_easing(start_brightness: int, end_brightness: int) -> str:
    """Automatically select the best easing function based on fade direction.

    Args:
        start_brightness: Starting brightness value (0-255).
        end_brightness: Ending brightness value (0-255).

    Returns:
        The name of the recommended easing function.
    """
    if start_brightness == 0:
        return "ease_out_quad"
    if end_brightness == 0:
        return "ease_in_quad"
    return "ease_in_out_sine"
