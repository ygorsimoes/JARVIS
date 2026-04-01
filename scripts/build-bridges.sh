#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

swift build -c release --package-path "$ROOT_DIR/bridges/apple/SpeechAnalyzerCLI"
swift build -c release --package-path "$ROOT_DIR/bridges/apple/FoundationModelsBridge"
