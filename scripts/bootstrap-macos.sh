#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

uv sync

if [[ ! -f "$ROOT_DIR/.env" ]]; then
  cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
fi

"$ROOT_DIR/scripts/build-bridges.sh"

cat <<'EOF'
Bootstrap concluido.

Proximos passos:
1. Revise o arquivo .env.
2. Garanta as permissoes de Microfone e Acessibilidade no macOS.
3. Rode `uv run jarvis --interactive` ou `uv run jarvis --voice`.
EOF
