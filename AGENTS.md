# AGENTS.md

**Purpose**
This repository is a small Python 3.12 `uv` project for a local pt-BR voice assistant.
Use this file as the primary repository guide for coding agents working in this tree.
Favor small, direct changes that preserve the current architecture and runtime model.

**Repository Snapshot**
- Package layout uses `src/` and the import root is `jarvis`.
- Main package lives in `src/jarvis/`.
- CLI entrypoint is `jarvis = "jarvis.cli:main"` from `pyproject.toml`.
- Packaging/build backend is `uv_build`.
- Required Python version is `>=3.12`.
- Dependency management is via `uv` and `uv.lock`.
- Linting/formatting tool in repo is `ruff`.
- Ruff is configured in `pyproject.toml`.
- Active Ruff lint rules are `E`, `F`, and `I`.
- Ruff line length is `100`.
- There is no committed `tests/` directory right now.

**Editor Rule Files**
- No `.cursor/rules/` directory was found.
- No `.cursorrules` file was found.
- No `.github/copilot-instructions.md` file was found.
- Treat this file plus the checked-in source as the authoritative local guidance.

**Setup And Environment**
- Create or update the environment with `uv sync --dev`.
- The repo includes `.python-version` set to `3.12`.
- Runtime configuration is loaded from environment variables, optionally from `--env-file`.
- Environment variable names use the `JARVIS_` prefix.
- `.env.example` is the source of truth for documented config knobs.
- `load_dotenv(..., override=False)` is used, so explicit environment variables keep winning.

**Build, Lint, Format, And Verification Commands**
- Lint whole repo: `uv run ruff check .`
- Lint one file: `uv run ruff check src/jarvis/pipeline.py`
- Format check whole repo: `uv run ruff format --check .`
- Format whole repo: `uv run ruff format .`
- Build distributions: `uv build`
- Cheap Python smoke check: `uv run python -m compileall src`
- CLI help smoke test: `uv run jarvis --help`
- Devices help smoke test: `uv run jarvis devices --help`
- List local audio devices: `uv run jarvis devices`

**Testing Reality In This Repo**
- There is currently no configured test suite in the repository.
- There is no `pytest` dependency in `pyproject.toml`.
- Today, the practical validation baseline is: lint, format check, `compileall`, and `uv build`.
- For runtime audio/model changes, full end-to-end verification may require local hardware and models.
- If those dependencies are unavailable, say so explicitly instead of faking coverage.

**Single-Test Guidance**
- There is no single-test command available today because no test runner is configured.
- If your task adds a pytest-based suite, use `uv run pytest` for all tests.
- If your task adds a pytest-based suite, run one test with `uv run pytest tests/test_file.py::test_name`.
- If you add tests, also add the test dependency in the dev group instead of assuming it exists.

**Code Map**
- `src/jarvis/cli.py`: argparse CLI, logging setup, command dispatch, and exit-code handling.
- `src/jarvis/config.py`: immutable app configuration and environment parsing.
- `src/jarvis/pipeline.py`: Pipecat pipeline construction, task wiring, observers, and warmup integration.
- `src/jarvis/turn_gate.py`: turn-gating heuristics and message normalization.
- `src/jarvis/turn_presets.py`: named timing presets for turn-taking behavior.
- `src/jarvis/warmup.py`: local Whisper path resolution and local model prewarming.
- `src/jarvis/ollama.py`: Ollama model discovery, fallback resolution, and request extras.
- `src/jarvis/audio_gate.py`: local echo-suppression gate.

**Repository Conventions**
- Prefer keeping new code inside the existing modules unless a new file is clearly warranted.
- Follow the existing `src/jarvis/*.py` flat module structure.
- Keep the local/offline architecture intact unless the task explicitly asks for a new integration.
- Avoid introducing cloud-only assumptions into the default path.
- Preserve pt-BR user-facing behavior unless the task explicitly changes it.

**Imports And Module Structure**
- Match the existing file header style: `from __future__ import annotations` at the top of normal modules.
- Group imports as stdlib, third-party, then local package imports.
- Let Ruff handle import sorting; do not hand-tune import order against Ruff.
- Use relative imports inside the package, e.g. `from .config import AppConfig`.
- Keep module-level constants in `UPPER_SNAKE_CASE`.

**Formatting**
- Keep lines within Ruff's configured `100` characters.
- Use standard Ruff formatting rather than custom style choices.
- Keep comments rare and purposeful.
- Do not add docstrings everywhere; the current codebase mostly relies on clear naming instead.

**Types**
- Add type hints for public functions and most non-trivial helpers.
- Use Python 3.12 built-in generics like `list[str]`, `dict[str, Any]`, and `tuple[float, str]`.
- Use `| None` instead of `Optional[...]`.
- Prefer `Path` over raw path strings when a filesystem path is part of domain state.
- Use frozen dataclasses for configuration-like value objects.
- When evolving immutable config objects, prefer `dataclasses.replace(...)` over mutation.

**Naming**
- Classes use `PascalCase`.
- Functions and variables use `snake_case`.
- Private helpers use a leading underscore.
- Constants use `UPPER_SNAKE_CASE`.
- Builder helpers are commonly named `_build_*`.
- Event-hook helpers are commonly named `_attach_*`.

**Control Flow And Architecture**
- Keep CLI orchestration in `cli.py` and pipeline construction in `pipeline.py`.
- Keep config parsing in `config.py`; do not scatter `os.getenv` calls across the codebase.
- If you add a new configuration flag, update `AppConfig`, env parsing, defaults, and `.env.example` together.
- If a config value is user-tunable from CLI, thread it through argparse and `replace(...)` overrides.
- Keep pure heuristics pure when possible, as in `compute_turn_gate_delay(...)`.

**Async And Pipecat Patterns**
- This codebase uses async processors and event handlers heavily.
- Preserve the current async style when extending frame processors.
- In overridden `process_frame(...)` methods, follow the existing push/return structure carefully.
- Avoid double-pushing frames or swallowing frames accidentally.
- Keep event-handler definitions close to the registration site for readability.

**Logging**
- Use `loguru` for runtime logging.
- Use Loguru brace formatting like `logger.info("{}", value)` instead of `%s` formatting.
- Avoid f-strings in logging calls unless you need them before the logger boundary.
- Follow the existing prefix style such as `[chat]`, `[turn]`, `[stt]`, `[tts]`, `[ollama]`, and `[context]`.
- Existing logs and user-facing runtime strings are in Portuguese; keep that language consistent.

**Error Handling**
- Raise `ValueError` for invalid user/config values when the input itself is wrong.
- Raise `RuntimeError` for operational failures such as unreachable local services or warmup failures.
- When wrapping lower-level exceptions, preserve context with `raise ... from exc`.
- Keep broad exception handling at process boundaries, not deep in helpers.
- `cli.py` is the current boundary that logs exceptions and converts them into exit codes.

**I/O And Side Effects**
- Use `print(...)` only for intentional CLI stdout output like device listing.
- Use logging for operational/runtime information elsewhere.
- Use `Path(...).expanduser()` when resolving filesystem paths from env vars.

**When Adding Tests Later**
- Prefer `pytest` if you need to introduce a test suite.
- Put tests under a top-level `tests/` directory.
- Mirror module names where practical, e.g. `tests/test_turn_gate.py`.
- Start with pure-function coverage for `turn_gate.py`, `ollama.py`, and config parsing helpers.
- Keep end-to-end audio tests optional and clearly marked, since they depend on local devices and models.

**Change-Specific Verification Expectations**
- Config parsing change: run Ruff, `compileall`, and at least one CLI help command.
- Packaging/build change: run `uv build`.
- Pure helper change: run Ruff and `compileall`; add tests if you introduced a test framework.
- Pipeline or warmup change: run Ruff, `compileall`, and explain any runtime verification blockers.

**Practical Agent Guidance**
- Keep edits small and local.
- Do not add new abstraction layers unless the current file is clearly becoming harder to follow.
- Do not silently change user-facing language away from Portuguese.
- State clearly when something cannot be fully verified because the repo lacks tests or requires local runtime dependencies.
