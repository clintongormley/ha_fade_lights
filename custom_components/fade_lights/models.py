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

    def has_target(self) -> bool:
        """Check if any fade target values are specified."""
        return (
            self.brightness_pct is not None
            or self.hs_color is not None
            or self.color_temp_mireds is not None
        )

    def has_from_target(self) -> bool:
        """Check if any from values are specified."""
        return (
            self.from_brightness_pct is not None
            or self.from_hs_color is not None
            or self.from_color_temp_mireds is not None
        )
