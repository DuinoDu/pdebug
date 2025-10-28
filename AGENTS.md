# Repository Guidelines

## Project Structure & Module Organization
- `pdebug/algo/`: streaming, dejitter, and SOT algorithms; keep optimizer-heavy logic here.
- `pdebug/debug/`: model inspection suites (depth, pose, SAM); add new analyzers or viz.
- `pdebug/geometry/`: geometric primitives (vectors, boxes, meshes); extend reusable math only.
- `pdebug/otn/`: orchestration for training/infer pipelines; register node managers and runners.
- `pdebug/pdag/`: Kedro-inspired DAG tooling; place pipeline graphs, IO bridges, and runners.
- `pdebug/piata/`: dataset handlers, registries, and type casts; manage data loading contracts.
- `pdebug/runnb/`: notebook and batch runners (`runnb.py`, `submit.py`); automation lives here.
- `pdebug/templates/`: CLI and submission templates (Typer, HTML, YAML); tweak UI payloads.
- `pdebug/tests/`: central pytest entry; mirror module paths with `test_*.py` coverage.
- `pdebug/utils/`: infrastructure helpers (IO, GPU, profiling, env); prefer reusing existing utils.
- `pdebug/visp/`: visualization glue (plotly, rerun, vispy); add renderers and color maps here.
- `scripts/` hosts CLI entrypoints; `requirements/`, `setup.cfg`, and `pyproject.toml` track deps.

## Build, Test, and Development Commands
- `make env`: install build/test/dev requirements and configure pre-commit.
- `make lint`: run Black, isort, Flake8, and docstyle hooks across the tree.
- `make test`: execute `pytest pdebug -s` with doctests for parity with CI.
- `make watch-test`: use `ptw` to re-run focused pytest suites during iteration.

## Coding Style & Naming Conventions
- Target Python 3.8+, 4-space indents, and 79-char lines; format with Black and isort (`profile=black`).
- Use snake_case for modules/functions, PascalCase for classes, and SCREAMING_SNAKE_CASE for constants.
- Flake8 (E/W/F, ignoring `E203`/`E731`) and Pydocstyle enforce code/documentation hygiene—fix warnings rather than silencing them.

## Testing Guidelines
- Place new coverage in `pdebug/tests/` or package-level `tests/` folders, naming files `test_*.py`.
- Pytest runs doctests, so keep README-style snippets executable and importable.
- Favor parametrized tests and shared fixtures (`conftest.py`) for datasets or config reuse.

## Commit & Pull Request Guidelines
- Write short, imperative commit subjects (e.g., `fix bug`, `update scripts/use_with_lance.py`).
- PRs should summarize intent, list core modules touched, link issues, and attach `make lint` / `make test` output.
- Provide screenshots or logs when behavior changes and request reviewers for affected subpackages.
