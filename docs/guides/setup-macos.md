# Setup macOS

## Pré-requisitos

- `macOS 26+`
- `Apple Silicon`
- `uv`
- `Xcode Command Line Tools`
- `ffmpeg`
- `espeak-ng`
- `swiftlint`

## Instalação

```bash
brew bundle
uv sync --extra macos-runtime
cp .env.example .env
swiftlint lint --config .swiftlint.yml
swift build -c release --package-path bridges/apple/SpeechAnalyzerCLI
swift build -c release --package-path bridges/apple/FoundationModelsBridge
uv run jarvis --doctor
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

`Foundation Models` continua opcional. Se o hot path nativo estiver indisponivel, o runtime deve degradar para o fallback local configurado.
