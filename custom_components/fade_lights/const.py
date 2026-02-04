"""Constants for the Fade Lights integration."""

DOMAIN = "fade_lights"

# Services
SERVICE_FADE_LIGHTS = "fade_lights"

# Service attributes
ATTR_BRIGHTNESS_PCT = "brightness_pct"
ATTR_TRANSITION = "transition"

# Color attributes (input parameters)
ATTR_COLOR_TEMP_KELVIN = "color_temp_kelvin"
ATTR_HS_COLOR = "hs_color"
ATTR_RGB_COLOR = "rgb_color"
ATTR_RGBW_COLOR = "rgbw_color"
ATTR_RGBWW_COLOR = "rgbww_color"
ATTR_XY_COLOR = "xy_color"
ATTR_FROM = "from"

# All color parameter names (for validation)
COLOR_PARAMS = frozenset(
    {
        ATTR_COLOR_TEMP_KELVIN,
        ATTR_HS_COLOR,
        ATTR_RGB_COLOR,
        ATTR_RGBW_COLOR,
        ATTR_RGBWW_COLOR,
        ATTR_XY_COLOR,
    }
)

# Storage
STORAGE_KEY = f"{DOMAIN}.last_brightness"

# Storage keys
KEY_ORIG_BRIGHTNESS = "orig"
KEY_CURR_BRIGHTNESS = "curr"

# Option keys
OPTION_MIN_STEP_DELAY_MS = "min_step_delay_ms"

# Defaults (used when options are not set)
DEFAULT_BRIGHTNESS_PCT = 40
DEFAULT_TRANSITION = 3  # seconds
DEFAULT_MIN_STEP_DELAY_MS = 100  # milliseconds

# Hard minimum for step delay (service call overhead)
MIN_STEP_DELAY_MS = 50

# Brightness tolerance for detecting manual intervention (accounts for device rounding)
BRIGHTNESS_TOLERANCE = 3

# Timeout for waiting on fade cancellation (seconds)
FADE_CANCEL_TIMEOUT_S = 2.0

# Time before an expected brightness value is considered stale (seconds)
STALE_THRESHOLD = 5.0

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
