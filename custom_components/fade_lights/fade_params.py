"""FadeParams model for the Fade Lights integration."""

from __future__ import annotations

from dataclasses import dataclass

from homeassistant.exceptions import ServiceValidationError
from homeassistant.util.color import (
    color_RGB_to_hs,
    color_rgbw_to_rgb,
    color_rgbww_to_rgb,
    color_xy_to_hs,
)

from .const import (
    ATTR_BRIGHTNESS_PCT,
    ATTR_COLOR_TEMP_KELVIN,
    ATTR_FROM,
    ATTR_HS_COLOR,
    ATTR_RGB_COLOR,
    ATTR_RGBW_COLOR,
    ATTR_RGBWW_COLOR,
    ATTR_XY_COLOR,
    COLOR_PARAMS,
    DEFAULT_TRANSITION,
)


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

    @classmethod
    def from_service_data(cls, data: dict) -> FadeParams:
        """Create FadeParams from service call data with validation.

        Validates:
        - At most one color parameter is specified
        - Color parameter values are within valid ranges

        Converts:
        - rgb_color, rgbw_color, rgbww_color, xy_color -> hs_color
        - color_temp_kelvin -> color_temp_mireds

        Also handles the 'from:' parameter for starting values.

        Args:
            data: Service call data dictionary

        Returns:
            FadeParams with normalized color values

        Raises:
            ServiceValidationError: If validation fails
        """
        cls._validate_color_params(data)
        cls._validate_color_ranges(data)

        brightness_pct, hs_color, color_temp_mireds = cls._extract_fade_values(data)

        from_brightness_pct = None
        from_hs_color = None
        from_color_temp_mireds = None

        from_data = data.get(ATTR_FROM, {})
        if from_data:
            (
                from_brightness_pct,
                from_hs_color,
                from_color_temp_mireds,
            ) = cls._extract_fade_values(from_data)

        return cls(
            brightness_pct=brightness_pct,
            hs_color=hs_color,
            color_temp_mireds=color_temp_mireds,
            from_brightness_pct=from_brightness_pct,
            from_hs_color=from_hs_color,
            from_color_temp_mireds=from_color_temp_mireds,
        )

    @classmethod
    def _validate_color_params(cls, data: dict) -> None:
        """Validate that at most one color parameter is specified.

        Also validates the from: parameter if present.

        Raises:
            ServiceValidationError: If multiple color parameters are provided.
        """
        cls._validate_single_color_param(data)

        from_data = data.get(ATTR_FROM, {})
        if from_data:
            cls._validate_single_color_param(from_data, "in 'from:'")

    @staticmethod
    def _validate_single_color_param(data: dict, context: str = "") -> None:
        """Validate that at most one color parameter is specified in data.

        Args:
            data: Dictionary to check for color parameters.
            context: Optional context string for error message (e.g., "in 'from:'").

        Raises:
            ServiceValidationError: If multiple color parameters are provided.
        """
        specified = [param for param in COLOR_PARAMS if param in data]
        if len(specified) > 1:
            ctx = f" {context}" if context else ""
            raise ServiceValidationError(
                f"Only one color parameter allowed{ctx}, got: {', '.join(sorted(specified))}"
            )

    @classmethod
    def _validate_color_ranges(cls, data: dict) -> None:
        """Validate color parameter value ranges.

        Raises:
            ServiceValidationError: If values are out of valid ranges.
        """
        cls._validate_color_ranges_dict(data, "")

        from_data = data.get(ATTR_FROM, {})
        if from_data:
            cls._validate_color_ranges_dict(from_data, "from: ")

    @staticmethod
    def _validate_color_ranges_dict(data: dict, prefix: str) -> None:
        """Validate color and brightness ranges in a single dict."""
        if ATTR_BRIGHTNESS_PCT in data:
            brightness = data[ATTR_BRIGHTNESS_PCT]
            if not (0 <= brightness <= 100):
                raise ServiceValidationError(
                    f"{prefix}Brightness must be between 0 and 100, got {brightness}"
                )

        if ATTR_HS_COLOR in data:
            hs = data[ATTR_HS_COLOR]
            if not (0 <= hs[0] <= 360):
                raise ServiceValidationError(
                    f"{prefix}Hue must be between 0 and 360, got {hs[0]}"
                )
            if not (0 <= hs[1] <= 100):
                raise ServiceValidationError(
                    f"{prefix}Saturation must be between 0 and 100, got {hs[1]}"
                )

        if ATTR_RGB_COLOR in data:
            rgb = data[ATTR_RGB_COLOR]
            for val in rgb[:3]:
                if not (0 <= val <= 255):
                    raise ServiceValidationError(
                        f"{prefix}RGB values must be between 0 and 255, got {val}"
                    )

        if ATTR_RGBW_COLOR in data:
            rgbw = data[ATTR_RGBW_COLOR]
            for val in rgbw[:4]:
                if not (0 <= val <= 255):
                    raise ServiceValidationError(
                        f"{prefix}RGBW values must be between 0 and 255, got {val}"
                    )

        if ATTR_RGBWW_COLOR in data:
            rgbww = data[ATTR_RGBWW_COLOR]
            for val in rgbww[:5]:
                if not (0 <= val <= 255):
                    raise ServiceValidationError(
                        f"{prefix}RGBWW values must be between 0 and 255, got {val}"
                    )

        if ATTR_XY_COLOR in data:
            xy = data[ATTR_XY_COLOR]
            for val in xy[:2]:
                if not (0 <= val <= 1):
                    raise ServiceValidationError(
                        f"{prefix}XY values must be between 0 and 1, got {val}"
                    )

        if ATTR_COLOR_TEMP_KELVIN in data:
            kelvin = data[ATTR_COLOR_TEMP_KELVIN]
            if not (1000 <= kelvin <= 40000):
                raise ServiceValidationError(
                    f"{prefix}Color temp must be between 1000K and 40000K, got {kelvin}K"
                )

    @classmethod
    def _extract_fade_values(
        cls, data: dict
    ) -> tuple[int | None, tuple[float, float] | None, int | None]:
        """Extract brightness, HS color, and mireds from data dict.

        Returns:
            Tuple of (brightness_pct, hs_color, color_temp_mireds)
        """
        brightness_pct = (
            int(data[ATTR_BRIGHTNESS_PCT]) if ATTR_BRIGHTNESS_PCT in data else None
        )
        hs, mireds = cls._extract_color(data)
        return brightness_pct, hs, mireds

    @staticmethod
    def _extract_color(data: dict) -> tuple[tuple[float, float] | None, int | None]:
        """Extract color from data dict, converting to HS or mireds.

        Returns:
            Tuple of (hs_color, color_temp_mireds) - one will be None
        """
        # Handle HS color (pass through)
        if ATTR_HS_COLOR in data:
            hs = data[ATTR_HS_COLOR]
            return (float(hs[0]), float(hs[1])), None

        # Handle RGB -> HS
        if ATTR_RGB_COLOR in data:
            rgb = data[ATTR_RGB_COLOR]
            hs = color_RGB_to_hs(rgb[0], rgb[1], rgb[2])
            return hs, None

        # Handle RGBW -> RGB -> HS
        if ATTR_RGBW_COLOR in data:
            rgbw = data[ATTR_RGBW_COLOR]
            rgb = color_rgbw_to_rgb(rgbw[0], rgbw[1], rgbw[2], rgbw[3])
            hs = color_RGB_to_hs(rgb[0], rgb[1], rgb[2])
            return hs, None

        # Handle RGBWW -> RGB -> HS
        if ATTR_RGBWW_COLOR in data:
            rgbww = data[ATTR_RGBWW_COLOR]
            rgb = color_rgbww_to_rgb(
                rgbww[0],
                rgbww[1],
                rgbww[2],
                rgbww[3],
                rgbww[4],
                min_kelvin=2700,
                max_kelvin=6500,
            )
            hs = color_RGB_to_hs(rgb[0], rgb[1], rgb[2])
            return hs, None

        # Handle XY -> HS
        if ATTR_XY_COLOR in data:
            xy = data[ATTR_XY_COLOR]
            hs = color_xy_to_hs(xy[0], xy[1])
            return hs, None

        # Handle color temperature
        if ATTR_COLOR_TEMP_KELVIN in data:
            kelvin = data[ATTR_COLOR_TEMP_KELVIN]
            return None, int(1_000_000 / kelvin)

        return None, None
