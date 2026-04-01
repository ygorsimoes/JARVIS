# Testing Strategy

## Python

- `tests/python/unit`: lógica pura e módulos pequenos
- `tests/python/integration`: integração entre runtime, adapters e tools
- `tests/python/e2e`: smoke tests de ponta a ponta

Runner padrão:

```bash
uv run pytest tests/python
```

## Swift

- Os testes Swift ficam dentro de cada package em `bridges/apple/*/Tests`
- Framework padrão: `swift-testing`

Execução:

```bash
swift test --package-path bridges/apple/SpeechAnalyzerCLI
swift test --package-path bridges/apple/FoundationModelsBridge
```

## Qualidade

```bash
uv run ruff check .
uv run ruff format --check .
```
