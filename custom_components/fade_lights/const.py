"""Constants for the Fade Lights integration."""

DOMAIN = "fade_lights"

# Services
SERVICE_FADE_LIGHTS = "fade_lights"

# Service attributes
ATTR_BRIGHTNESS_PCT = "brightness_pct"
ATTR_TRANSITION = "transition"
ATTR_FORCE = "force"

# Storage
STORAGE_KEY = f"{DOMAIN}.last_brightness"
STORAGE_VERSION = 1

# Option keys
OPTION_DEFAULT_BRIGHTNESS_PCT = "default_brightness_pct"
OPTION_DEFAULT_TRANSITION = "default_transition"
OPTION_AUTO_BRIGHTNESS_THRESHOLD = "auto_brightness_threshold"
OPTION_AUTO_BRIGHTNESS_TARGET = "auto_brightness_target"

# Defaults (used when options are not set)
DEFAULT_BRIGHTNESS_PCT = 40
DEFAULT_TRANSITION = 3
DEFAULT_FORCE = False

# Auto-brightness settings (used when options are not set)
AUTO_BRIGHTNESS_THRESHOLD = 10
AUTO_BRIGHTNESS_TARGET = 40
