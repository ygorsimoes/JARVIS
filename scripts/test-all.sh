#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

uv run ruff check .
uv run ruff format --check .
uv run pytest tests/python
swift test --package-path "$ROOT_DIR/bridges/apple/SpeechAnalyzerCLI"
swift test --package-path "$ROOT_DIR/bridges/apple/FoundationModelsBridge"
