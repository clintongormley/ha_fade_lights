"""Constants for the Fado integration."""

DOMAIN = "fado"

# Services
SERVICE_FADE_LIGHTS = "fade_lights"

# Service attributes
ATTR_BRIGHTNESS = "brightness"
ATTR_BRIGHTNESS_PCT = "brightness_pct"
ATTR_TRANSITION = "transition"
ATTR_EASING = "easing"

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

# Valid parameters for from: block
FROM_PARAMS = frozenset(
    {
        ATTR_BRIGHTNESS,
        ATTR_BRIGHTNESS_PCT,
        ATTR_COLOR_TEMP_KELVIN,
        ATTR_HS_COLOR,
        ATTR_RGB_COLOR,
        ATTR_RGBW_COLOR,
        ATTR_RGBWW_COLOR,
        ATTR_XY_COLOR,
    }
)

# Valid parameters for main service call (entity_id handled separately by HA)
MAIN_PARAMS = frozenset(
    {
        "entity_id",
        ATTR_BRIGHTNESS,
        ATTR_BRIGHTNESS_PCT,
        ATTR_TRANSITION,
        ATTR_EASING,
        ATTR_FROM,
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
OPTION_LOG_LEVEL = "log_level"

# Log levels (matching Python logging module)
LOG_LEVEL_WARNING = "warning"
LOG_LEVEL_INFO = "info"
LOG_LEVEL_DEBUG = "debug"
DEFAULT_LOG_LEVEL = LOG_LEVEL_WARNING

# Defaults (used when options are not set)
DEFAULT_TRANSITION = 3  # seconds
DEFAULT_MIN_STEP_DELAY_MS = 100  # milliseconds

# Hard minimum for step delay (service call overhead)
MIN_STEP_DELAY_MS = 50

# Brightness tolerance for detecting manual intervention (accounts for device rounding)
BRIGHTNESS_TOLERANCE = 3

# Color tolerances for detecting manual intervention (accounts for device rounding)
HUE_TOLERANCE = 5.0  # degrees (0-360 scale)
SATURATION_TOLERANCE = 3.0  # percentage points (0-100 scale)
MIREDS_TOLERANCE = 10  # mireds (internal use only)
KELVIN_TOLERANCE = 100  # kelvin

# Timeout for waiting on fade cancellation (seconds)
FADE_CANCEL_TIMEOUT_S = 2.0

# Time before an expected brightness value is considered stale (seconds)
STALE_THRESHOLD = 5.0

# Planckian locus lookup table: mireds -> (hue, saturation)
# Approximates the curve of blackbody radiation through HS color space.
# Used for HS ↔ color temperature transitions.
PLANCKIAN_LOCUS_HS: tuple[tuple[int, tuple[float, float]], ...] = (
    (154, (220.0, 5.0)),  # 6500K - Cool daylight (bluish)
    (167, (215.0, 4.5)),  # 6000K
    (182, (210.0, 4.0)),  # 5500K - Noon daylight
    (200, (55.0, 5.0)),  # 5000K - Horizon daylight
    (222, (45.0, 6.0)),  # 4500K - Cool white
    (250, (42.0, 8.0)),  # 4000K
    (286, (38.0, 12.0)),  # 3500K - Neutral white
    (303, (36.0, 15.0)),  # 3300K
    (333, (35.0, 18.0)),  # 3000K - Warm white
    (370, (33.0, 24.0)),  # 2700K - Soft white
    (400, (32.0, 30.0)),  # 2500K
    (435, (30.0, 38.0)),  # 2300K
    (500, (28.0, 45.0)),  # 2000K - Candlelight
)

# Maximum saturation to consider a color "on" the Planckian locus
PLANCKIAN_LOCUS_SATURATION_THRESHOLD = 15.0

# Minimum deltas for step calculation (smoothest possible fade)
MIN_BRIGHTNESS_DELTA = 1  # 0-255 scale
MIN_HUE_DELTA = 1.0  # degrees (0-360)
MIN_SATURATION_DELTA = 1.0  # percentage (0-100)
MIN_MIREDS_DELTA = 5  # mireds (~150-500 typical range)

# Hybrid transition crossover ratio (HS phase proportion for hs_to_mireds)
# For hs_to_mireds: 70% of steps in HS phase, 30% in mireds phase
# For mireds_to_hs: inverted (30% mireds, 70% HS)
HYBRID_HS_PHASE_RATIO = 0.7

# Native transitions feature
NATIVE_TRANSITION_MS = 10  # Transition time added to turn_on for native_transitions lights

# Autoconfigure feature settings (for measuring optimal min_delay_ms)
AUTOCONFIGURE_ITERATIONS = 10  # Number of times to test each light
AUTOCONFIGURE_TIMEOUT_S = 5  # Timeout for each state change wait (seconds)
AUTOCONFIGURE_MAX_PARALLEL = 5  # Maximum lights to test in parallel

# Notification for unconfigured lights
NOTIFICATION_ID = "fado_unconfigured"
REQUIRED_CONFIG_FIELDS = frozenset({"min_delay_ms", "min_brightness", "native_transitions"})
UNCONFIGURED_CHECK_INTERVAL_HOURS = 24
