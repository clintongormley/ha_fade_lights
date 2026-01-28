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
