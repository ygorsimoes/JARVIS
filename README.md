# J.A.R.V.I.S.

Assistente local-first, macOS-first, com control-plane em Python e bridges nativos Apple em Swift.

## Estado atual

- Foco atual: `macOS 26+` e `Apple Silicon`
- Core Python em `src/jarvis`
- Bridges nativos em `bridges/apple`
- Testes Python em `tests/python`
- Concepção ampla em `docs/architecture/JARVIS.md`

## Quickstart

### Pré-requisitos

- `uv`
- `Xcode Command Line Tools`
- `ffmpeg`
- `espeak-ng`
- `swiftlint`

### Bootstrap

```bash
bash scripts/bootstrap-macos.sh
```

Se quiser fazer manualmente:

```bash
uv sync
cp .env.example .env
swiftlint lint --config .swiftlint.yml
swift build -c release --package-path bridges/apple/SpeechAnalyzerCLI
swift build -c release --package-path bridges/apple/FoundationModelsBridge
```

## Rodando

Modo texto:

```bash
uv run jarvis --interactive
```

Modo voz:

```bash
uv run jarvis --voice
```

Turno único:

```bash
uv run jarvis --demo "Que horas sao agora?"
```

Modo desenvolvimento com backends fake/noop:

```bash
uv run jarvis --mock-backends --interactive
```

## Testes e qualidade

Tudo:

```bash
bash scripts/test-all.sh
```

Somente Python:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/python
```

Somente Swift:

```bash
swiftlint lint --config .swiftlint.yml
swift test --package-path bridges/apple/SpeechAnalyzerCLI
swift test --package-path bridges/apple/FoundationModelsBridge
```

## Estrutura

```text
src/jarvis/                    # control-plane Python
bridges/apple/                # bridges Swift nativos do macOS
tests/python/                 # suíte pytest por camada
docs/architecture/JARVIS.md   # especificação ampla do projeto
docs/guides/                  # setup e operação
scripts/                      # bootstrap, build e validação
```

## Documentação

- `docs/architecture/JARVIS.md`
- `docs/guides/setup-macos.md`
- `docs/testing/strategy.md`
