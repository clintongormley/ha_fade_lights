# Claude Code Instructions

## Pre-approved Commands (NEVER ask for permission)

The following commands are pre-approved and must be run WITHOUT asking for permission:

```
rm -Rf /workspaces/homeassistant-core/config/custom_components/fado && cp -r /tmp/ha-fado/custom_components/fado /workspaces/homeassistant-core/config/custom_components/
```

## Git Workflow

- **NEVER commit directly to main** - always create a feature branch first
- All changes must go through a Pull Request (PR)
- Do NOT merge PRs automatically - wait for user approval before merging
- When merging a PR (after approval), delete the feature branch

## Code Quality

- Before creating a PR, always run `ruff check .` and `ruff format .` to fix any linting issues
- Before creating a PR, run `npx pyright` to check for Pylance/type errors and fix any that can be fixed

## Deployment

- After making changes, always copy the integration to the HA config directory for testing:
  ```
  rm -Rf /workspaces/homeassistant-core/config/custom_components/fado && cp -r /workspaces/ha-fado/custom_components/fado /workspaces/homeassistant-core/config/custom_components/
  ```
- This is a **custom component**, NOT a core integration - never copy to `homeassistant/components/`
- Run the copy command automatically without asking for permission

## Shortcuts

- `cp to ha` - Remove existing and copy the integration to `/workspaces/homeassistant-core/config/custom_components/fado/` for testing:
  ```
  rm -Rf /workspaces/homeassistant-core/config/custom_components/fado && cp -r /workspaces/ha-fado/custom_components/fado /workspaces/homeassistant-core/config/custom_components/
  ```
- `pr` - Commit any changes in separate commits if that makes sense, and create a PR (do not merge)
- `merge` - Merge the current PR (only after explicit user approval)
