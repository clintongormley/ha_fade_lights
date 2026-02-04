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

# Defaults
DEFAULT_BRIGHTNESS_PCT = 40
DEFAULT_TRANSITION = 3
DEFAULT_FORCE = False

# Auto-brightness settings
AUTO_BRIGHTNESS_THRESHOLD = 10
AUTO_BRIGHTNESS_TARGET = 40
