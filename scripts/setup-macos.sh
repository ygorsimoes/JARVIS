#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "J.A.R.V.I.S. setup suporta apenas macOS." >&2
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "J.A.R.V.I.S. requer Apple Silicon (arm64)." >&2
  exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew nao encontrado. Instale em https://brew.sh e rode o setup novamente." >&2
  exit 1
fi

if ! xcode-select -p >/dev/null 2>&1; then
  echo "Xcode Command Line Tools nao encontrados. Rode 'xcode-select --install'." >&2
  exit 1
fi

brew bundle --file "$ROOT_DIR/Brewfile"
uv python install 3.12
uv sync --extra macos-runtime

if [[ ! -f "$ROOT_DIR/.env" ]]; then
  cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
fi

swiftlint lint --config "$ROOT_DIR/.swiftlint.yml"
"$ROOT_DIR/scripts/build-bridges.sh"
uv run jarvis --doctor

cat <<'EOF'
Setup concluido.

Para conversar com o J.A.R.V.I.S.:
  uv run jarvis --interactive
  uv run jarvis --voice
EOF
