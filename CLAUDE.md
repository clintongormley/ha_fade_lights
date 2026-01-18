# Claude Code Instructions

## Git Workflow

- Always ask before merging a PR
- When merging a PR, delete the branch that the PR was on

## Code Quality

- Before creating a PR, always run `ruff check .` and `ruff format .` to fix any linting issues
- Before creating a PR, run `npx pyright` to check for Pylance/type errors and fix any that can be fixed
