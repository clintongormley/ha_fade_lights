# Claude Code Instructions

## Git Workflow

- All changes should be made via a PR (do not push directly to main)
- Do NOT merge PRs automatically - wait for user approval before merging
- When merging a PR (after approval), delete the branch that the PR was on

## Code Quality

- Before creating a PR, always run `ruff check .` and `ruff format .` to fix any linting issues
- Before creating a PR, run `npx pyright` to check for Pylance/type errors and fix any that can be fixed
