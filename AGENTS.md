# AGENTS.md

This file is the source of truth for agent instructions in this repository.

## Project Overview
- Name: `eta-ctrl`
- Purpose: Framework for digital twins of factories/machines with rolling-horizon optimization, simulation, and live system interaction.
- Status: Beta. APIs may change; don't bother with backward-compatible changes unless explicitly requested.
- based on Gymnasium and stable_baselines3

## Primary Components
- `eta_ctrl/core.py`: Main `EtaCtrl` orchestration.
- `eta_ctrl/config/`: Runtime and path/config abstractions (`Config`, `RunInfo`).
- `eta_ctrl/envs/`: Environment base classes and implementations (`BaseEnv`, `LiveEnv`, `PyomoSimEnv`, `SimEnv`). Subclasses override `_step()`/`_reset()`, not the public gymnasium methods.
- `eta_ctrl/simulators/`: FMU simulation helpers.
- `eta_ctrl/timeseries/`: CSV import + resample/interpolation utilities.
- `eta_ctrl/agents/`: Agent integrations.
- `scripts/`: CLI entry points.
- `test/`: Test suite.
- `docs/`: Sphinx documentation.

## How Experiments Run
Experiments are driven by `EtaCtrl` (`eta_ctrl/core.py`), which takes a `Config` file. From the config, everything required is loaded — including `StateConfig` and `ScenarioManager`. Environments and agents are then prepared and run via `EtaCtrl.play()` or `EtaCtrl.learn()`.

## Runtime And Tooling Constraints
- Python: `>=3.10.16,<3.13`
- Package manager/build: Poetry (`poetry-core` backend)
- Typing: mypy (strict for main package; tests excluded)
- Lint/format: Ruff (line length 120; configured in `pyproject.toml`)
- Tests: pytest (+ coverage config in `pyproject.toml`)

## Canonical Commands
- Install deps: `poetry sync`
- Run tests: `poetry run pytest`
- Run tests (quiet): `poetry run pytest -q`
- Lint: `poetry run ruff check`
- Format check: `poetry run ruff format --check .`
- Type check: `poetry run mypy --config-file pyproject.toml`
- Pre-commit: `poetry run pre-commit run --all-files`
- Docs build: `cd docs && make clean && make ci-html`

## CLI Entry Points
- `create-sim-env` -> `scripts.fmu_cli:create_sim_env`
- `export-fmu-data` -> `scripts.fmu_cli:export_fmu_data`
- `export-pyomo-data` -> `scripts.pyomo_export_cli:export_pyomo_data`

## Coding Standards
- Prefer explicit type hints for production code.
- Keep public interfaces mostly stable; explicitly document breaking changes.
- Follow existing Ruff and mypy settings; do not add local style exceptions unless justified.
- Keep changes minimal and scoped to the request.
- Preserve existing module structure and naming patterns.
- Propose structural improvements to the user if adequate.

## Testing Expectations
- For behavior changes, add/update tests in `test/`.
- At minimum, run targeted tests for touched modules.
- Before finalizing substantial changes, run pre-commit and pytest with CI settings (or explain why not run).

## Change Safety Rules
- Be careful in these areas:
  - `eta_ctrl/envs/`: Step/reset semantics and Gymnasium compatibility.
  - `eta_ctrl/timeseries/`: Time index alignment, resampling frequency, interpolation correctness.
  - `eta_ctrl/simulators/`: FMU I/O contracts and simulation timing.
  - `eta_ctrl/config/`: Path handling and run metadata persistence.
- Preserve backward compatibility for script entry points and config behavior unless explicitly asked to break.

## Preferred Patterns
- Do:
  - Reuse existing helpers/utilities before introducing new abstractions.
  - Fail with clear exceptions for invalid configuration/state.
  - Keep side effects explicit.
- Avoid:
  - Silent behavior changes.
  - Broad refactors unrelated to the request.
  - New dependencies without clear need.

## PR/Change Checklist
- Code passes CI and imports cleanly.
- Relevant tests updated/passing.
- Ruff, mypy, and pytest pass locally (or deviations documented).
- Docs/examples updated when public behavior changes.

## Agent Operating Rules
- Ask before:
  - Introducing breaking API changes.
  - Adding/removing dependencies.
  - Editing CI/release pipeline behavior.
- Never:
  - Commit secrets.
  - Disable checks globally to make CI pass.
- When unsure:
  - Prefer smallest safe change and leave clear notes in the final summary.
