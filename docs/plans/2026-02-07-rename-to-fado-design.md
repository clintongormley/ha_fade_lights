# Design: Rename integration from Fade Lights to Fado

## Context

The integration is being renamed from "Fade Lights" / `fade_lights` to "Fado" / `fado`. This is a clean rename with no migration logic needed for existing users. The GitHub repository will also be renamed from `ha_fade_lights` to `ha-fado`.

## Scope

~427 references across 45 files. All mechanical find-and-replace with no logic changes.

## Rename mappings

| Pattern | From | To |
|---|---|---|
| Domain (snake_case) | `fade_lights` | `fado` |
| Display name | `Fade Lights` | `Fado` |
| Kebab-case | `fade-lights` | `fado` |
| Class: panel | `FadeLightsPanel` | `FadoPanel` |
| Class: config flow | `FadeLightsConfigFlow` | `FadoConfigFlow` |
| Service constant | `SERVICE_FADE_LIGHTS` | `SERVICE_FADO` |
| localStorage key | `fade_lights_collapsed` | `fado_collapsed` |
| WS command prefix | `fade_lights/` | `fado/` |
| Panel URL | `/fade-lights` | `/fado` |
| Panel element tag | `fade-lights-panel` | `fado-panel` |
| Notification ID | `fade_lights_unconfigured` | `fado_unconfigured` |
| Directory | `custom_components/fade_lights/` | `custom_components/fado/` |
| Repo | `ha_fade_lights` | `ha-fado` |

## Execution order

### 1. Rename directory
```
custom_components/fade_lights/ -> custom_components/fado/
```

### 2. Update const.py
- `DOMAIN = "fade_lights"` -> `DOMAIN = "fado"`
- `SERVICE_FADE_LIGHTS = "fade_lights"` -> `SERVICE_FADO = "fado"`
- `NOTIFICATION_ID = "fade_lights_unconfigured"` -> `NOTIFICATION_ID = "fado_unconfigured"`

### 3. Update all Python source files
Apply rename mappings to:
- `__init__.py` (service refs, panel registration, docstrings)
- `config_flow.py` (class name, entry title, docstrings)
- `websocket_api.py` (WS command type strings, docstrings)
- `autoconfigure.py` (docstrings only)
- `notifications.py` (notification text, URL, title)
- `fade_params.py`, `fade_change.py`, `expected_state.py`, `easing.py` (docstrings)

### 4. Update config/translation files
- `manifest.json` (domain, name, URLs)
- `strings.json` (title, service key)
- `translations/en.json` (title, service key)
- `services.yaml` (service key)

### 5. Update frontend
In `panel.js`:
- Class: `FadeLightsPanel` -> `FadoPanel`
- Element: `fade-lights-panel` -> `fado-panel`
- WS types: `fade_lights/*` -> `fado/*`
- localStorage: `fade_lights_collapsed` -> `fado_collapsed`
- Title: `<h1>Fade Lights</h1>` -> `<h1>Fado</h1>`

### 6. Update test files
All test files under `tests/`:
- Imports: `custom_components.fade_lights` -> `custom_components.fado`
- Service constant: `SERVICE_FADE_LIGHTS` -> `SERVICE_FADO`
- Service name strings: `"fade_lights"` -> `"fado"`
- Display name strings: `"Fade Lights"` -> `"Fado"`
- Config flow references

### 7. Update manifest URLs
- `documentation` -> `https://github.com/clintongormley/ha-fado`
- `issue_tracker` -> `https://github.com/clintongormley/ha-fado/issues`

### 8. Final verification
- `grep -r "fade_lights" .` to catch stragglers
- `grep -r "Fade Lights" .` to catch stragglers
- `grep -r "fade-lights" .` to catch stragglers
- Run `ruff check` on all Python files
- Run `python -m pytest tests/ -x --timeout=60 --numprocesses=auto`

### 9. Sync and deploy
- Copy `custom_components/fado/` to HA workdir
- Commit, push, create PR, merge

## What does NOT change
- Internal class names like `FadeParams`, `FadeChange`, `FadeStep`, `ExpectedState` (these describe fade behavior, not the integration name)
- The `easing.py` module name
- The fade execution logic
- Any stored data format or structure

## Post-rename: GitHub repo
Rename the repository on GitHub from `ha_fade_lights` to `ha-fado` (done manually by the user).
