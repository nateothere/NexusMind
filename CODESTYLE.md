# Coding Style

NexusMind code is typed, linted, and formatted. `make lint` runs all of
this locally; CI enforces it.

## Toolchain

| Tool    | Config                    | Notes                     |
| ------- | ------------------------- | ------------------------- |
| ruff    | `pyproject.toml [tool.ruff]` | lint + format          |
| mypy    | `pyproject.toml [tool.mypy]` | `--strict`            |
| pytest  | `pyproject.toml [tool.pytest]` | coverage >= 85%    |

## Rules

- **Typing:** every public function is fully annotated. No `Any` leaks —
  if you need `Any`, justify it in a comment.
- **Docstrings:** Google style, imperative mood ("Return the fused
  rankings", not "Returns the fused rankings").
- **Imports:** stdlib → third-party → first-party, alphabetical within
  groups. `isort` profile for black-style compatibility.
- **Naming:** `snake_case` for functions/variables, `PascalCase` for
  classes, `UPPER_CASE` for module constants.
- **Errors:** raise `NexusMindError` subclasses from `nexusmind/utils.py`;
  never leak raw `Exception` messages to API clients.
- **Line length:** 100 chars (ruff default for this project).
- **No dead code:** unreachable branches fail lint. Remove unused imports
  on sight.
- **Tests:** unit tests never touch the network. Integration tests are
  tagged `@pytest.mark.integration` and skipped by default.

## Pre-commit

`.pre-commit-config.yaml` runs ruff, ruff-format, mypy, markdownlint and
the end-of-file fixer on every commit. `make setup` installs the hook.