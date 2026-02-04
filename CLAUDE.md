# Claude Code Instructions

## Git Workflow

- All changes should be made via a PR (do not push directly to main), and never merge a PR without asking
- Do NOT merge PRs automatically - wait for user approval before merging
- When merging a PR (after approval), delete the branch that the PR was on

## Code Quality

- Before creating a PR, always run `ruff check .` and `ruff format .` to fix any linting issues
- Before creating a PR, run `npx pyright` to check for Pylance/type errors and fix any that can be fixed

## Shortcuts

- `cp to ha` - Copy the integration to `/workspaces/homeassistant-core/config/custom_components/fade_lights/` for testing
- `pr` - Commit any changes in separate commits if that makes sense, and create a PR (do not merge)
- `merge` - Merge the current PR (only after explicit user approval)
