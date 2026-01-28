"""Data models for the Fade Lights integration."""

from __future__ import annotations

from dataclasses import dataclass

from .const import DEFAULT_TRANSITION


@dataclass
class FadeParams:
    """Normalized parameters for a fade operation.

    All color inputs are converted to internal representations:
    - RGB/RGBW/RGBWW/XY colors -> hs_color (hue 0-360, saturation 0-100)
    - color_temp_kelvin -> color_temp_mireds
    """

    brightness_pct: int | None = None
    hs_color: tuple[float, float] | None = None  # (hue, saturation)
    color_temp_mireds: int | None = None
    transition_ms: int = DEFAULT_TRANSITION * 1000

    # Starting values (from: parameter)
    from_brightness_pct: int | None = None
    from_hs_color: tuple[float, float] | None = None
    from_color_temp_mireds: int | None = None


@dataclass
class FadeStep:
    """A single step in a fade sequence.

    All values are optional - only include attributes being faded.
    """

    brightness: int | None = None
    hs_color: tuple[float, float] | None = None
    color_temp_mireds: int | None = None
