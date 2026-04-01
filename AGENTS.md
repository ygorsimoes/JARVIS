# AGENTS.md

## Purpose
Guidance for coding agents working in `/Users/ygorsimoes/projects/J.A.R.V.I.S`.
Stay aligned with the repo's actual build, lint, test, and style conventions.
Prefer existing patterns over inventing new abstractions.

## Repo Snapshot
- Project: J.A.R.V.I.S., a local-first assistant runtime.
- Architecture reference: `docs/architecture/JARVIS.md`.
- Python app code: `src/jarvis/`.
- Python tests: `tests/python/` with `unit`, `integration`, and `e2e` splits.
- Swift bridges: `bridges/apple/SpeechAnalyzerCLI/` and `bridges/apple/FoundationModelsBridge/`.
- Target platform: Apple Silicon, macOS 26+.
- Toolchain: Python `3.12`, Swift tools `6.2`, package/task runner `uv`.

## Editor Rules
No `.cursorrules`, `.cursor/rules/`, or `.github/copilot-instructions.md` files were found.
If any of those files are added later, treat them as higher-priority local instructions.

## Important Paths
- `docs/architecture/JARVIS.md`: source-of-truth architecture document.
- `src/jarvis/config.py`: Pydantic settings model.
- `src/jarvis/runtime.py`: main runtime orchestration.
- `src/jarvis/adapters/interfaces.py`: adapter `Protocol` contracts.
- `src/jarvis/models/`: dataclasses, enums, and payload models.
- `src/jarvis/observability/`: structlog-based logging and tracing.
- `scripts/setup-macos.sh`: bootstrap script.
- `scripts/build-bridges.sh`: release Swift bridge builds.
- `scripts/test-all.sh`: full repo quality gate.
- `pyproject.toml`: Python dependencies, pytest config, Ruff config.
- `.swiftlint.yml`: Swift lint rules.

## Setup And Quality
```bash
bash scripts/setup-macos.sh
uv sync --frozen
uv sync --extra macos-runtime
bash scripts/build-bridges.sh
bash scripts/test-all.sh
uv run jarvis --doctor
uv run jarvis --interactive
uv run jarvis --voice
```

`scripts/test-all.sh` runs `ruff check`, `ruff format --check`, `pytest tests/python`, `swiftlint`, and both Swift package test suites.

## Python Commands

```bash
uv run ruff check .
uv run ruff check --fix .
uv run ruff format --check .
uv run ruff format .
uv run pytest tests/python
uv run pytest -m unit
uv run pytest -m integration
uv run pytest -m e2e
uv run pytest -k runtime
uv run pytest --cov=jarvis --cov-branch --cov-report=term-missing --cov-report=xml:coverage.xml --cov-fail-under=80
```

```bash
uv run pytest tests/python/unit/test_config.py
uv run pytest tests/python/unit/test_config.py::TestJarvisConfig::test_defaults_follow_canonical_local_runtime
uv run pytest tests/python/unit/test_config.py --collect-only -q
```

- `pytest` config lives in `pyproject.toml`.
- `asyncio_mode = strict`.
- Markers are `unit`, `integration`, and `e2e`; `tests/python/conftest.py` auto-tags tests by directory.
- There is no separate Python type-check step configured.

## Swift Commands

```bash
swiftlint lint --config .swiftlint.yml
swift build -c release --package-path bridges/apple/SpeechAnalyzerCLI
swift build -c release --package-path bridges/apple/FoundationModelsBridge
swift build -Xswiftc -warnings-as-errors --package-path bridges/apple/SpeechAnalyzerCLI
swift build -Xswiftc -warnings-as-errors --package-path bridges/apple/FoundationModelsBridge
swift test --package-path bridges/apple/SpeechAnalyzerCLI
swift test --package-path bridges/apple/FoundationModelsBridge
```

```bash
swift test list --package-path bridges/apple/SpeechAnalyzerCLI
swift test list --package-path bridges/apple/FoundationModelsBridge
swift test --package-path bridges/apple/SpeechAnalyzerCLI --filter 'SpeechAnalyzerCoreTests.SpeechAnalyzerCoreTests/parseOptionsForLiveMode()'
swift test --package-path bridges/apple/FoundationModelsBridge --filter 'FoundationModelsBridgeCoreTests.FoundationModelsBridgeCoreTests/jsonValueRoundTripsThroughJSON()'
```

- Both packages use Swift Testing, not XCTest.
- Both manifests target `macOS(.v26)`.
- `scripts/build-bridges.sh` builds both packages in release mode.

## Python Style
- Default to `from __future__ import annotations`; most modules do.
- Group imports as standard library, third-party, then local package imports.
- Ruff enforces import order and unused import cleanup; do not leave dead imports behind.
- Let Ruff own formatting instead of hand-formatting around it.
- Match the typing style already used in the file you touch.
- The repo mixes `typing.List/Optional/Dict` and built-in generics; keep consistency within a file.
- Add explicit type hints for public functions, methods, dataclasses, and important locals.
- Use dataclasses for lightweight domain state and internal runtime records.
- Use `field(default_factory=...)` for mutable dataclass defaults.
- Use Pydantic `BaseSettings` for config and `BaseModel` for validated action payloads.
- Preserve the interface-first pattern built around `Protocol`s in `src/jarvis/adapters/interfaces.py`.
- Enums commonly subclass `str, Enum` when values are serialized.
- Use `snake_case` for modules, functions, and variables.
- Use `PascalCase` for classes, dataclasses, and enums.
- Use `UPPER_SNAKE_CASE` for constants.
- Tool names and capabilities use dotted identifiers like `system.get_time`.

## Python Runtime Conventions
- Prefer `Path` over raw string path manipulation.
- Use `asyncio.to_thread(...)` for blocking filesystem, subprocess, network, or model-loading work.
- Prefer structured logging via `get_logger(__name__)` from `jarvis.observability`.
- Include context as keyword fields on log calls when useful.
- Raise specific domain errors when callers need to branch on failure modes.
- The codebase also uses direct `RuntimeError` for unsupported backend selections or violated runtime preconditions; keep those messages explicit.
- Do not swallow `asyncio.CancelledError` unless the code is intentionally handling cancellation.
- Broad `except Exception` is only appropriate for defensive cleanup, degraded behavior, or log-and-continue boundaries.
- Keep shutdown paths explicit; adapters commonly expose `shutdown`, `close_session`, or `close`.
- Preserve existing Portuguese user-facing strings unless there is a clear reason to change them.

## Swift Style
- Keep code compatible with Swift `6.2` package manifests.
- Prefer small `struct`s and `enum`s with explicit conformances such as `Sendable`, `Equatable`, and `Codable`.
- Use `LocalizedError` for stable human-readable errors.
- Keep access control explicit when package APIs are meant to be public.
- Import only the modules you use.
- Use `UpperCamelCase` for types and `lowerCamelCase` for properties and functions.
- Keep wire-format keys stable through explicit `CodingKeys`; snake_case JSON keys are intentional in bridge payloads.
- SwiftLint is strict and checks `force_cast`, `force_try`, `line_length`, `trailing_whitespace`, `unused_import`, `vertical_whitespace`, and `empty_count`.
- Respect `.swiftlint.yml` line length thresholds: warning at `120`, error at `140`.

## Testing Conventions
- Prefer focused unit tests for pure logic and integration tests for orchestration.
- Python async tests use `@pytest.mark.asyncio` and `pytest_asyncio.fixture` where needed.
- Existing Python tests usually prefer local fakes and direct assertions over heavy mocking.
- Group related Python tests in `Test...` classes when it improves readability.
- Swift tests use `import Testing`, `@Test`, `@Suite`, `#expect`, and `Issue.record(...)`.
- Keep Swift test names stable enough that `swift test list` output remains useful for `--filter` runs.
- Verify changes with the narrowest useful command first, then broaden if needed.
- For Python-only changes, run Ruff plus the most relevant pytest target.
- For Swift-only changes, run SwiftLint plus the relevant package test or filtered test.
- If a change crosses Python and Swift boundaries, run checks on both sides.

## Agent Working Norms
- When behavior or boundaries are unclear, read `docs/architecture/JARVIS.md` before improvising.
- Prefer the smallest correct change.
- Preserve existing subsystem vocabulary and tone.
- Do not edit generated or cache directories such as `.build/`, `.venv/`, `.pytest_cache/`, `.ruff_cache/`, `.coverage`, or `coverage.xml` unless the task is specifically about them.
- Verify with the narrowest useful command first; if a change crosses the Python/Swift boundary, run both sides.
