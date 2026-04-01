#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

uv run ruff check .
uv run ruff format --check .
uv run pytest tests/python

if ! command -v swiftlint >/dev/null 2>&1; then
  echo "swiftlint nao encontrado. Instale com: brew install swiftlint" >&2
  exit 1
fi

swiftlint lint --config "$ROOT_DIR/.swiftlint.yml"

swift test --package-path "$ROOT_DIR/bridges/apple/SpeechAnalyzerCLI"
swift test --package-path "$ROOT_DIR/bridges/apple/FoundationModelsBridge"
