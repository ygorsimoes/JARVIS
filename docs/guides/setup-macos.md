# Setup macOS

## Pré-requisitos

- `macOS 26+`
- `Apple Silicon`
- `uv`
- `Xcode Command Line Tools`
- `ffmpeg`
- `espeak-ng`

## Instalação

```bash
uv sync
cp .env.example .env
swift build -c release --package-path bridges/apple/SpeechAnalyzerCLI
swift build -c release --package-path bridges/apple/FoundationModelsBridge
```

## Permissões

- Microfone
- Acessibilidade, se usar hotkey global
- Apple Intelligence, para `Foundation Models`

## Primeiro uso

```bash
uv run jarvis --interactive
uv run jarvis --voice
uv run jarvis --demo "Que horas sao agora?"
```

## Modo de desenvolvimento

```bash
uv run jarvis --mock-backends --interactive
```
