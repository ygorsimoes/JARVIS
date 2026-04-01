# J.A.R.V.I.S.

**Just A Rather Very Intelligent System**

`MacBook Air M4` · `16 GB` · `macOS Tahoe 26.3` · `arm64`  
Stack local-first, performance-first, escalável por design.

---

## Índice

| #   | Seção                                                        |
| --- | ------------------------------------------------------------ |
| 1   | [Visão e Princípios](#1-visão-e-princípios)                  |
| 2   | [Hardware](#2-hardware)                                      |
| 3   | [Arquitetura](#3-arquitetura)                                |
| 4   | [Máquina de Estados](#4-máquina-de-estados)                  |
| 5   | [Turn Manager](#5-turn-manager)                              |
| 6   | [Sentence Streamer](#6-sentence-streamer)                    |
| 7   | [Camada de Entrada](#7-camada-de-entrada)                    |
| 8   | [Control-Plane](#8-control-plane)                            |
| 9   | [Memory System](#9-memory-system)                            |
| 10  | [Ferramentas e Capabilities](#10-ferramentas-e-capabilities) |
| 11  | [Camada de Saída](#11-camada-de-saída)                       |
| 12  | [Contratos e Adapters](#12-contratos-e-adapters)             |
| 13  | [Configuração](#13-configuração)                             |
| 14  | [Estrutura do Projeto](#14-estrutura-do-projeto)             |
| 15  | [Dependências](#15-dependências)                             |
| 16  | [SLOs Operacionais](#16-slos-operacionais)                   |
| 17  | [Tabela de Decisões](#17-tabela-de-decisões)                 |
| 18  | [System Prompt](#18-system-prompt)                           |

---

## 1. Visão e Princípios

O objetivo não é construir um agente agentic com voz. É construir **uma presença local fluida** — que entende bem, responde rápido, age com autonomia e soa natural, espelhando a conversa do Tony Stark com o JARVIS nos filmes.

O centro de gravidade do sistema não é o LLM. É a **qualidade da conversa**. O LLM é o motor de raciocínio dentro de um sistema maior que governa escuta, turno, interrupção, memória e ação.

**Python como control-plane portável.** Orquestração, memória, ferramentas e raciocínio vivem em Python. Swift entra como adapter de voz nativo no Mac — encapsulado como CLI e microserviço — não como control-plane. Quando o JARVIS rodar em Linux, Windows ou VPS, o core Python não muda; apenas os adapters de voz são trocados.

**Adapters, não dependências.** Nenhum provider é o contrato do sistema. STT, LLM, TTS e memória são interfaces com implementações substituíveis. Trocar um backend é uma linha no `.env`.

---

## 2. Hardware

### Especificações

| Componente       | Especificação                                      |
| ---------------- | -------------------------------------------------- |
| Chip             | Apple M4 — 10 cores CPU · 8 cores GPU              |
| Neural Engine    | 38 TOPS — separado do pool Metal                   |
| Memória          | 16 GB unificada (CPU + GPU + ANE compartilham)     |
| Metal disponível | ~10.6–12 GB (kernel limita a ~66–75% da RAM total) |
| Largura de banda | ~120 GB/s — principal limitador de tokens/s        |
| OS               | macOS Tahoe 26.3 arm64                             |

### Papéis intencionais dos aceleradores

| Acelerador        | Papel intencional no JARVIS                             | Framework       |
| ----------------- | ------------------------------------------------------- | --------------- |
| **ANE**           | STT (SpeechAnalyzer) + LLM hot path (Foundation Models) | Swift / Core ML |
| **GPU via Metal** | LLM deliberativo (mlx-lm) + TTS (mlx-audio)             | MLX             |
| **CPU**           | VAD · turn detection · asyncio · ferramentas            | Python / stdlib |

O JARVIS separa o pool MLX/Metal dos runtimes nativos da Apple usando políticas de orçamento para evitar contenção no caminho quente da conversa. O escalonamento interno de aceleradores nos frameworks nativos (SpeechAnalyzer, Foundation Models) pertence ao runtime da plataforma — o sistema não gerencia qual unidade de silício é usada internamente. O que o JARVIS controla é que os componentes MLX rodam no pool Metal governado pelo ResourceGovernor, enquanto os componentes Swift nativos rodam fora desse pool.

### Orçamento Metal

| Componente                    | Metal       | Quando ativo        |
| ----------------------------- | ----------- | ------------------- |
| mlx-lm Qwen3 8B Q4 — pesos    | ~5.2 GB     | THINKING → SPEAKING |
| mlx-lm KV cache (4096 tokens) | ~0.8 GB     | Durante THINKING    |
| mlx-audio Kokoro-82M          | ~400 MB     | Durante SPEAKING    |
| Qwen3 Embedding               | ~600 MB     | Busca de memória    |
| **Pico total**                | **~7.0 GB** | —                   |
| **Limite configurado**        | **9.5 GB**  | ResourceGovernor    |
| **Margem de segurança**       | **~2.5 GB** | —                   |

### Escala por hardware

| Parâmetro        | M4 16 GB    | M4 Pro 24 GB | M4 Pro 48 GB |
| ---------------- | ----------- | ------------ | ------------ |
| `memory_limit`   | 9.5 GB      | 18 GB        | 42 GB        |
| `wired_limit`    | 8.5 GB      | 16 GB        | 38 GB        |
| `cache_limit`    | 512 MB      | 1 GB         | 2 GB         |
| `max_kv_size`    | 4096 tokens | 8192 tokens  | 32768 tokens |
| LLM deliberativo | Qwen3 8B Q4 | Qwen3 14B Q4 | Qwen3 32B Q4 |

---

## 3. Arquitetura

```
                           Você (voz)
                                │
          ┌─────────────────────▼───────────────────────┐
          │              CAMADA DE ENTRADA              │
          │  Ativação → VAD → STT (runtime nativo)      │
          └─────────────────────┬───────────────────────┘
                                │ texto + metadata de áudio
          ┌─────────────────────▼───────────────────────┐
          │               TURN MANAGER                  │
          │   fim de turno · barge-in · yield policy    │
          └─────────────────────┬───────────────────────┘
                                │ turno validado
 ┌──────────────────────────────▼──────────────────────────────┐
 │                    CONTROL-PLANE (Python)                   │
 │                                                             │
 │   Dialogue Manager                                          │
 │         │                                                   │
 │   Complexity Router                                         │
 │   ├── Hot Path    ──► Foundation Models (runtime nativo)    │
 │   └── Deliberativo ──► Qwen3 8B Q4 (Metal)                  │
 │                                │                            │
 │   Resource Governor ◄──────────┘                            │
 │         │                                                   │
 │   ┌─────▼───────────────────────────────────────────────┐   │
 │   │                  Memory System                      │   │
 │   │   Working · Episodic · Profile · Procedural         │   │
 │   │   SQLite + FTS5 + sqlite-vec · Provenance           │   │
 │   └─────────────────────────────────────────────────────┘   │
 │         │                                                   │
 │   Capability Broker ──► Action Broker ──► Tool Registry     │
 │         │                                                   │
 │   Sentence Streamer — buffer LLM→TTS por sentença           │
 └──────────────────────────────┬──────────────────────────────┘
                                │ sentenças em streaming
          ┌─────────────────────▼───────────────────────┐
          │               CAMADA DE SAÍDA               │
          │       mlx-audio + Kokoro-82M (Metal)        │
          └─────────────────────┬───────────────────────┘
                                │ áudio
                            Você ouve
```

### Componentes de primeira classe

| Componente            | Responsabilidade                                                        |
| --------------------- | ----------------------------------------------------------------------- |
| **Turn Manager**      | Barge-in · fim de turno · yield — o coração da naturalidade             |
| **Sentence Streamer** | Buffer LLM→TTS por sentença — o coração da fluência                     |
| **ResourceGovernor**  | Limites Metal/MLX — previne OOM e kernel panic                          |
| **ComplexityRouter**  | Decide hot path vs. deliberativo — invisível ao usuário                 |
| **DialogueManager**   | Contexto de conversa · composição de prompt                             |
| **ActionBroker**      | Contratos tipados para ações — nunca parsing livre de texto             |
| **CapabilityBroker**  | Ledger explícito: quais tools · em quais escopos · com que confirmações |
| **MemorySystem**      | 4 classes + provenance                                                  |
| **ToolRegistry**      | Registro e execução com fronteiras de segurança                         |

---

## 4. Máquina de Estados

```
IDLE
  │  ← hotkey / wake word
  ▼
ARMED
  │  ← VAD detecta início de fala  (SpeechDetector no Mac · Silero cross-platform)
  ▼
LISTENING
  │  ← Turn Manager valida fim de turno
  ▼
TRANSCRIBING
  │  ← SpeechAnalyzer retorna transcrição final
  ▼
THINKING       ← ComplexityRouter: hot path ou deliberativo
  │  ← LLM em streaming + Sentence Streamer despachando TTS
  ├──► ACTING  ← tool solicitada · CapabilityBroker autoriza
  │       │  ← tool concluída
  ▼       ▼
SPEAKING       ← TTS tocando · Sentence Streamer bufferando próximas sentenças
  │
  ▼
IDLE

Transições de exceção:
  INTERRUPTED  ←  barge-in durante SPEAKING ou THINKING
  FAILED       ←  timeout · erro de tool destrutiva · OOM
```

---

## 5. Turn Manager

A diferença entre um assistente que parece natural e um que parece robótico não está no LLM nem no TTS. Está no **turn-taking**: quando o sistema para de ouvir, quando começa a falar, como reage a uma interrupção.

### VAD — dois sinais combinados

O Turn Manager combina dois sinais para decidir o fim de turno com precisão.

**VAD primário no Mac — SpeechDetector (Swift nativo).** A Apple lançou o `SpeechDetector` como módulo do SpeechAnalyzer para responder "há fala?" e gatear a transcrição com eficiência. No Mac, o adapter padrão é `SpeechDetectorAdapter` — roda fora do pool Metal junto ao SpeechAnalyzer.

**VAD fallback cross-platform — Silero (CPU only).** Para Linux, Windows e VPS. Roda exclusivamente em CPU, sem consumir Metal. Parâmetros de referência:

| Parâmetro                 | Valor | Razão                                  |
| ------------------------- | ----- | -------------------------------------- |
| `threshold`               | 0.5   | Equilíbrio entre ruído e sensibilidade |
| `min_silence_duration_ms` | 700   | Preserva pausas naturais da fala       |
| `speech_pad_ms`           | 100   | Preserva o início de palavras          |

**Texto parcial do STT — sinal secundário para ambos.** Se a transcrição parcial termina gramaticalmente (ponto, frase completa), o turno pode encerrar mesmo com pausa curta. Se termina no meio de uma locução ("e então eu..."), aguarda mais.

### Políticas de turno

| Situação                           | Política                                                |
| ---------------------------------- | ------------------------------------------------------- |
| Pausa natural < 700ms durante fala | Aguarda — não é fim de turno                            |
| Silêncio > 800ms após fala         | Fim de turno — envia ao LLM                             |
| Fala detectada durante SPEAKING    | Barge-in — cancela TTS em < 150ms · retorna a LISTENING |
| Fala detectada durante THINKING    | Enfileira interrupção — cancela geração quando seguro   |
| Turno muito longo (> 30s)          | Corta e processa o que tem                              |

### SLOs

| Métrica                             | Target   |
| ----------------------------------- | -------- |
| Barge-in → TTS parado               | < 150 ms |
| Taxa de falso corte de turno        | < 3%     |
| Taxa de yield correto após barge-in | > 97%    |

---

## 6. Sentence Streamer

O Sentence Streamer senta entre o LLM (que emite tokens em streaming) e o TTS (que precisa de uma sentença completa para sintetizar). Ele dispara o TTS assim que detecta o fim de uma sentença — sem esperar a resposta completa.

### Por que isso importa

```
Abordagem ingênua:
  STT (80ms) → LLM completo (800ms) → TTS (300ms) → áudio
  Silêncio percebido: ~1.2s  ← inaceitável

Com Sentence Streamer:
  STT (80ms) → LLM inicia → 1ª sentença (420ms) → TTS dispara
  🔊 Áudio começa em: ~580ms  ← LLM ainda gera o resto
```

### Estratégia de segmentação

Segmentação por pontuação simples (`.!?`) é insuficiente em português: `:` frequentemente abre continuação, não encerra pensamento; frases curtas truncadas prejudicam a prosódia do TTS. A implementação canônica combina quatro critérios:

1. **Pontuação terminal** — `.` `!` `?` confirmam fim de sentença
2. **Comprimento mínimo** — buffer < 8 tokens não é despachado mesmo com pontuação (evita "Ok." isolado)
3. **Cláusula semântica** — `:` e `\n` só encerram se o buffer tiver >= 40 chars
4. **Backpressure do TTS** — se a fila tiver > 2 itens pendentes, o streamer aguarda antes de despachar mais

```python
async def sentence_stream(llm_token_stream, tts_queue: asyncio.Queue):
    buffer = ""
    HARD_ENDS = {'.', '!', '?'}
    SOFT_ENDS = {':', '\n'}

    async for token in llm_token_stream:
        buffer += token
        stripped = buffer.rstrip()

        is_hard_end = any(stripped.endswith(c) for c in HARD_ENDS)
        is_soft_end = any(stripped.endswith(c) for c in SOFT_ENDS)
        long_enough = len(buffer.strip()) > 8

        if long_enough and (is_hard_end or (is_soft_end and len(buffer) >= 40)):
            while tts_queue.qsize() > 2:
                await asyncio.sleep(0.01)
            await tts_queue.put(buffer.strip())
            buffer = ""

    if buffer.strip():
        await tts_queue.put(buffer.strip())

    await tts_queue.put(None)  # sentinel — fim da resposta
```

### Timeline de referência

Os valores abaixo são observações no ambiente de referência (M4 16 GB, modo deliberativo) — targets de projeto, não garantias contratuais.

```
t=0ms    fim de fala detectado pelo VAD
t=80ms   SpeechAnalyzer retorna texto
t=90ms   Turn Manager valida fim de turno
t=100ms  ComplexityRouter → deliberativo
t=110ms  Qwen3 8B começa streaming de tokens
t=530ms  1ª sentença completa → tts_queue
t=580ms  🔊 JARVIS começa a falar
t=580ms  LLM ainda gera a 2ª sentença
t=950ms  2ª sentença → tts_queue → sem pausa perceptível
```

---

## 7. Camada de Entrada

### Ativação

| Mecanismo                    | Status              | Implementação                                              |
| ---------------------------- | ------------------- | ---------------------------------------------------------- |
| Push-to-talk / hotkey global | **Padrão canônico** | `pynput`                                                   |
| Wake word                    | Opcional            | Porcupine — arm64 oficial · modelos customizáveis em pt-BR |
| Click-to-speak               | Opcional            | Status bar item Swift                                      |

> **Wake word em pt-BR:** OpenWakeWord não suporta português — o problema é adequação linguística, não plataforma. Porcupine tem suporte oficial a arm64 e permite modelos customizados treinados na sua voz dizendo "Hey JARVIS".

### STT — SpeechAnalyzer

Benchmarks comunitários em macOS Tahoe reportam o SpeechAnalyzer significativamente mais rápido que o Whisper Large V3 na mesma máquina. Roda fora do pool Metal, liberando recursos para o LLM. Suporta `pt_BR` na lista oficial de locales. Produz transcrição ao vivo com `.progressiveLiveTranscription`. Latência e WER devem ser revalidados por benchmark no hardware-alvo.

**Integração com Python** via Swift CLI subprocess com protocolo NDJSON. Cada evento emitido tem um campo `type` explícito:

```json
{"type": "partial_transcript", "text": "que horas são", "confidence": 0.87}
{"type": "final_transcript",   "text": "Que horas são?", "confidence": 0.96}
{"type": "speech_started"}
{"type": "speech_ended"}
{"type": "error", "message": "model_not_ready"}
```

```python
class SpeechAnalyzerSTT(STTAdapter):
    async def transcribe_stream(self) -> AsyncIterator[str]:
        proc = await asyncio.create_subprocess_exec(
            "./bridges/apple/SpeechAnalyzerCLI/.build/release/speechanalyzer-cli",
            "--live", "--locale", "pt-BR", "--format", "ndjson",
            stdout=asyncio.subprocess.PIPE,
        )
        async for line in proc.stdout:
            event = json.loads(line)
            if event["type"] == "final_transcript":
                yield event["text"]
```

### Fallbacks STT

| Backend                     | Plataforma       | Acelerador           | Observação                                |
| --------------------------- | ---------------- | -------------------- | ----------------------------------------- |
| **SpeechAnalyzer** (padrão) | macOS 26+        | Runtime nativo Apple | Latência ref. ~80ms · validar no hardware |
| mlx-whisper                 | macOS (fallback) | Metal                | Apple Silicon only                        |
| faster-whisper              | Linux · Windows  | CPU / CUDA           | CTranslate2 — multiplataforma real        |
| whisper.cpp                 | Todos            | CPU · Metal · CUDA   | Build com Core ML no Mac para ANE         |

---

## 8. Control-Plane

### Resource Governor

Inicializado em `main.py` antes de qualquer carregamento de modelo. Os três knobs do `mlx.core` governam o pool Metal compartilhado por mlx-lm e mlx-audio.

```python
import mlx.core as mx

def initialize_resource_governor(config: JarvisConfig) -> None:
    mx.set_memory_limit(config.metal_memory_limit)   # 9.5 GB
    mx.set_wired_limit(config.metal_wired_limit)     # 8.5 GB
    mx.set_cache_limit(config.metal_cache_limit)     # 512 MB
    # max_kv_size é passado em TODA chamada ao LLM — nunca omitido
```

> **Por que esses três knobs são obrigatórios:** sem eles, o MLX pode fiar toda a RAM disponível. Com contextos longos, o KV cache cresce sem teto e causa kernel panic — comportamento documentado em issue ativo do mlx-lm, inclusive em máquinas de 96 GB.

### Complexity Router

A maioria das interações é trivial. Rotear tudo para o Qwen3 8B desperdiça latência.

```
"Que horas são?"              → tool direta            (~30ms)
"Abre o Spotify"              → tool direta            (~50ms)
"Define um timer de 20min"    → hot path (FM)          (~200ms)
"Resume esse email"           → deliberativo (8B)      (~700ms)
"Explica esse erro de código" → deliberativo (8B)      (~900ms)
"Cria um plano para..."       → deliberativo ou nuvem
```

**Critérios de roteamento:**

| Sinal                                                    | Destino        |
| -------------------------------------------------------- | -------------- |
| Intent é tool direta (timer, app, volume)                | Tool — sem LLM |
| Frase < 12 tokens · sem subordinadas · contexto simples  | Hot path       |
| Contexto rico de memória · múltiplos passos              | Deliberativo   |
| Palavras de raciocínio ("analisa", "explica", "por que") | Deliberativo   |
| Tool calling encadeado                                   | Deliberativo   |

### Hot Path — Foundation Models (macOS 26)

Modelo ~3B rodando no runtime nativo da Apple · tool calling · sessões stateful · streaming · grátis · sem consumo Metal.

| Restrição                                           | Status                                                                                                         |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Requer Apple Intelligence em System Settings        | Obrigatório                                                                                                    |
| MacBook Air M4 é device elegível                    | ✅                                                                                                             |
| Suporta Português (Brasil)                          | ✅                                                                                                             |
| API Swift-only — acesso via microserviço HTTP local | Ver bridge abaixo                                                                                              |
| Escopo de uso                                       | Tarefas estruturadas e simples — Apple documenta: _"not designed to be a chatbot for general world knowledge"_ |

O Foundation Models bridge expõe sessões stateful, não requests avulsos. O protocolo SSE usa os mesmos tipos de evento do SpeechAnalyzer CLI:

```json
{"type": "response_chunk",  "text": "São 14h32."}
{"type": "tool_call",       "name": "get_time", "args": {}}
{"type": "tool_result",     "name": "get_time", "result": "14:32"}
{"type": "response_end"}
{"type": "error",           "message": "model_not_ready"}
```

```swift
// bridges/apple/FoundationModelsBridge/Sources/FoundationModelsBridge/main.swift
let session = LanguageModelSession(
    instructions: "Assistente conciso em pt-BR. Responda em no máximo 2 frases."
)
app.get("chat") { req async throws -> Response in
    let prompt = try req.query.get(String.self, at: "q")
    return req.eventLoop.makeSSEResponse(session.streamResponse(to: prompt))
}
```

**Fallback:** Qwen3-4B non-thinking mode via mlx-lm, se Foundation Models não disponível.

### Deliberativo — Qwen3 8B Q4_K_M

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen3-8B-4bit")

def chat_stream(messages: list, tools: list, max_kv_size: int = 4096):
    # max_kv_size OBRIGATÓRIO — nunca omitir
    yield from generate(
        model, tokenizer,
        prompt=format_prompt(messages, tools),
        max_tokens=1024,
        max_kv_size=max_kv_size,
        stream=True,
    )
```

> **Nunca usar `mlx_lm.server`:** o servidor fia ~75% da RAM ao iniciar sem controle de `max_kv_size`. KV cache sem teto causa kernel panic — issue ativo documentado no mlx-lm, inclusive em máquinas de 96 GB.

**LLM remoto como fallback:** Anthropic / OpenAI via adapter. Ativado pelo PolicyEngine quando a tarefa excede a capacidade local ou o dispositivo está sob pressão de memória.

### Dialogue Manager

Decide _como_ responder: tom, comprimento, se confirmar antes de agir, se pedir esclarecimento. Compõe o contexto a partir de três fontes:

1. Working memory — deque dos últimos N turnos em RAM
2. Memórias recuperadas do Memory System por similaridade semântica
3. System prompt base + instruções da sessão atual

**Política de recuperação de memória:** busca lexical via FTS5 primeiro (zero Metal); busca vetorial semântica apenas no modo deliberativo. Embedding de novas memórias é sempre assíncrono — nunca bloqueia o caminho de resposta. Orçamento de latência explícito: < 50ms por ciclo de recuperação.

### Action Broker

O LLM nunca é integrado por parsing livre de texto quando a intenção for operacional. Toda ação passa por contrato tipado.

```python
# ✗ Errado — parsing livre
if "abrir" in response and "spotify" in response.lower():
    open_app("Spotify")

# ✓ Correto — structured output via tool schema
class OpenAppTool(BaseModel):
    tool: Literal["open_app"]
    app_name: str

action = OpenAppTool.model_validate_json(llm_tool_call)
tool_registry.execute(action)
```

---

## 9. Memory System

### Quatro classes

| Classe         | O que armazena                                    | Acesso             |
| -------------- | ------------------------------------------------- | ------------------ |
| **Working**    | Contexto ativo do turno/sessão — deque em RAM     | Zero I/O           |
| **Episodic**   | Eventos e fatos de sessões passadas               | SQLite + embedding |
| **Profile**    | Preferências estáveis · hábitos · projetos ativos | SQLite + embedding |
| **Procedural** | Como executar rotinas e playbooks recorrentes     | SQLite + embedding |

### Provenance

Cada memória carrega metadados de confiança para evitar **alucinação de familiaridade** — o JARVIS tratar uma memória inferida com o mesmo peso de uma afirmação explícita.

```python
@dataclass
class Memory:
    content: str
    category: Literal["working", "episodic", "profile", "procedural"]
    source: Literal["explicit", "inferred", "system"]
    confidence: float       # 0.0–1.0
    recency_weight: float   # decai com o tempo
    scope: str              # "session" · "project:nome" · "global"
    created_at: datetime
    last_accessed: datetime
```

### Política de escrita

| Categoria                                     | Política                       |
| --------------------------------------------- | ------------------------------ |
| Fatos pessoais · preferências · rotinas       | Sempre grava — alta prioridade |
| Projetos e tarefas em andamento               | Sempre grava                   |
| Decisões e instruções explícitas              | Sempre grava                   |
| Respostas informativas relevantes             | Grava com score de relevância  |
| Small talk · saudações · confirmações de tool | Não grava                      |
| Sessões longas sem fatos novos                | Grava resumo comprimido        |

### Storage e embedding

**SQLite** para estado estruturado · **FTS5** para busca lexical · **sqlite-vec** para busca semântica.

`sqlite-vec` é pré-v1 e brute-force — aceitável para bases pessoais. Versão pinada no `pyproject.toml`. A interface `MemoryAdapter` permite trocar para pgvector sem mudar o Memory System.

| Modelo de embedding               | Idiomas | Contexto    | Metal   |
| --------------------------------- | ------- | ----------- | ------- |
| **Qwen3-Embedding-0.6B** (padrão) | 100+    | 32K tokens  | ~600 MB |
| BGE-M3                            | 100+    | 8192 tokens | ~600 MB |
| nomic-embed-text-v2-moe           | ~100    | 2048 tokens | ~280 MB |

---

## 10. Ferramentas e Capabilities

### Capability Broker

O CapabilityBroker é o registro explícito do que o JARVIS pode fazer, em que condições, e com que restrições. Ele responde a uma pergunta antes de qualquer execução de tool: _"esta ação está habilitada para este escopo com este nível de risco?"_

```python
@dataclass
class Capability:
    tool_name: str
    enabled: bool
    scope: str                  # "global" · "session" · "project:nome"
    risk_level: Literal["read_only", "write_safe", "destructive"]
    requires_confirmation: bool
    side_effects: list[str]     # descritivo — para auditoria
    audit_log: bool
```

Capabilities podem ser habilitadas/desabilitadas em runtime via configuração ou via instrução explícita do usuário — nunca via conteúdo externo.

### Fronteiras de segurança (imutáveis)

| Princípio                 | Definição                                                                  |
| ------------------------- | -------------------------------------------------------------------------- |
| **Source trust boundary** | Texto da web · email · arquivos externos é dado — nunca instrução          |
| **Tool intent isolation** | Conteúdo externo nunca escolhe ou aciona tools; apenas o usuário, via fala |
| **Compensating actions**  | Se tool falhar no meio, Action Broker desfaz ou pede confirmação           |

### Hierarquia de integração macOS

| Prioridade | Mecanismo                     | Razão                                 |
| ---------- | ----------------------------- | ------------------------------------- |
| 1          | App Intents / Shortcuts       | Forma oficial da Apple · mais estável |
| 2          | API do framework via `pyobjc` | EventKit, Contacts, etc.              |
| 3          | x-callback-url / URL scheme   | Para apps que suportam                |
| 4          | AppleScript / JXA             | Funcional · segunda escolha           |
| 5          | Accessibility / UI scripting  | Último recurso                        |

### Níveis de risco

| Nível      | Exemplos                                         | Comportamento                    |
| ---------- | ------------------------------------------------ | -------------------------------- |
| Read-only  | Ler calendário · buscar na web · listar arquivos | Execução direta                  |
| Write-safe | Criar evento · setar timer · escrever arquivo    | Execução com log                 |
| Destrutivo | Deletar arquivo · enviar email · executar shell  | Confirmação explícita do usuário |

### Ferramentas canônicas

| Ferramenta | Capacidade                               | Implementação                            |
| ---------- | ---------------------------------------- | ---------------------------------------- |
| `timer`    | Criar · listar · cancelar timers         | Python stdlib                            |
| `shell`    | Comandos macOS — allowlist explícita     | `subprocess` com allowlist               |
| `system`   | Abrir apps · volume · brilho             | App Intents / `pyobjc`                   |
| `browser`  | Busca na web · fetch de páginas          | DuckDuckGo / Brave Search API            |
| `calendar` | Ler e criar eventos                      | EventKit via `pyobjc-framework-EventKit` |
| `files`    | Ler · listar · mover — paths autorizados | Python stdlib                            |

---

## 11. Camada de Saída

### TTS — mlx-audio + Kokoro-82M

```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/Kokoro-82M-bf16")

async def synthesize_stream(text: str) -> AsyncIterator[bytes]:
    for result in model.generate(text, voice="pm_santa", lang_code="p"):
        yield result.audio.tobytes()
```

`lang_code="p"` é Português Brasileiro. Vozes: `pm_santa` (masculino) · `pf_dora` (feminino).

AVSpeechSynthesizer foi avaliado e rejeitado como padrão — qualidade claramente inferior ao Kokoro, incompatível com o objetivo de soar natural como nos filmes.

### Modelos disponíveis

| Modelo                                | Metal   | Qualidade                             | Latência 1ª sílaba |
| ------------------------------------- | ------- | ------------------------------------- | ------------------ |
| **Kokoro-82M-bf16** (padrão macOS)    | ~400 MB | Boa — **validar pt-BR auditivamente** | < 200 ms (ref.)    |
| Kokoro-82M-4bit                       | ~200 MB | Boa                                   | < 150 ms (ref.)    |
| Qwen3-TTS-0.6B (voz clonada)          | ~1.2 GB | Superior                              | ~300 ms (ref.)     |
| Kokoro-ONNX / Piper (Linux · Windows) | —       | Boa                                   | variável           |
| AVSpeechSynthesizer (fallback macOS)  | —       | Aceitável                             | ~50 ms             |

> **Qualidade em pt-BR:** a qualidade real do Kokoro depende do modelo G2P e dos dados de treino para português. Validar por teste auditivo e taxa de compreensão no hardware-alvo antes de tratar como "pronto".

> **Upgrade path — voz clonada:** Qwen3-TTS-0.6B via mlx-audio clona qualquer voz com ~3 segundos de áudio de referência. Suporta português. Requer 1.2 GB Metal adicionais.

---

## 12. Contratos e Adapters

### Interfaces

```python
from typing import Protocol, AsyncIterator, runtime_checkable

@runtime_checkable
class STTAdapter(Protocol):
    async def transcribe_stream(self) -> AsyncIterator[str]: ...

@runtime_checkable
class LLMAdapter(Protocol):
    async def chat_stream(
        self, messages: list, tools: list, max_kv_size: int
    ) -> AsyncIterator[str]: ...

@runtime_checkable
class TTSAdapter(Protocol):
    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]: ...

@runtime_checkable
class WakeWordAdapter(Protocol):
    async def listen(self) -> bool: ...

@runtime_checkable
class MemoryAdapter(Protocol):
    def search(self, query: str, top_k: int) -> list[Memory]: ...
    def save(self, content: str, metadata: dict) -> None: ...
    def should_persist(self, turn: Turn) -> bool: ...
```

### Backends por plataforma

| Componente            | macOS 26                                      | Linux / VPS                 | Windows                     |
| --------------------- | --------------------------------------------- | --------------------------- | --------------------------- |
| Ativação              | Push-to-talk · Porcupine                      | Push-to-talk                | Push-to-talk                |
| VAD                   | SpeechDetector (runtime nativo)               | Silero VAD (CPU)            | Silero VAD (CPU)            |
| STT                   | SpeechAnalyzer (runtime nativo · NDJSON)      | faster-whisper              | faster-whisper              |
| STT fallback          | mlx-whisper (Metal — Apple Silicon only)      | whisper.cpp                 | whisper.cpp                 |
| LLM hot path          | Foundation Models (runtime nativo · Swift)    | —                           | —                           |
| LLM hot path fallback | Qwen3-4B non-thinking (mlx-lm)                | Qwen3-4B (Ollama)           | Qwen3-4B (Ollama)           |
| LLM deliberativo      | mlx-lm Qwen3 8B Q4                            | Ollama Qwen3 8B             | Ollama Qwen3 8B             |
| LLM remoto            | Anthropic / OpenAI                            | idem                        | idem                        |
| TTS                   | mlx-audio Kokoro (Metal — Apple Silicon only) | Kokoro-ONNX / Piper         | Kokoro-ONNX / Piper         |
| TTS fallback          | AVSpeechSynthesizer                           | pyttsx3                     | pyttsx3                     |
| Embedding             | Qwen3 Embedding (mlx)                         | sentence-transformers (CPU) | sentence-transformers (CPU) |
| Memória               | SQLite + sqlite-vec                           | idem                        | idem                        |
| Event bus             | asyncio                                       | asyncio → Redis             | asyncio → Redis             |

> **mlx-audio e mlx-whisper são Apple Silicon only** — não rodam em Linux/Windows. Para essas plataformas, TTS usa Kokoro via runtime ONNX ou Piper; STT usa faster-whisper ou whisper.cpp.

STT e TTS são sempre locais — latência de áudio não tolera round-trip de rede.

---

## 13. Configuração

```bash
# .env — MacBook Air M4 16 GB

# ── Metal Governor ─────────────────────────────────────────────
JARVIS_METAL_MEMORY_LIMIT_GB=9.5
JARVIS_METAL_WIRED_LIMIT_GB=8.5
JARVIS_METAL_CACHE_LIMIT_GB=0.5
JARVIS_LLM_MAX_KV_SIZE=4096          # obrigatório em toda chamada ao LLM

# ── STT ────────────────────────────────────────────────────────
JARVIS_STT_BACKEND=speech_analyzer   # ou: mlx_whisper · faster_whisper
JARVIS_STT_LOCALE=pt-BR

# ── LLM ────────────────────────────────────────────────────────
JARVIS_LLM_HOT_PATH=foundation_models
JARVIS_LLM_DELIBERATIVE=mlx_lm
JARVIS_LLM_DELIBERATIVE_MODEL=mlx-community/Qwen3-8B-4bit

# ── TTS ────────────────────────────────────────────────────────
JARVIS_TTS_BACKEND=mlx_audio_kokoro
JARVIS_TTS_MODEL=mlx-community/Kokoro-82M-bf16
JARVIS_TTS_VOICE=pm_santa            # pt-BR masculino
JARVIS_TTS_LANG_CODE=p

# ── Ativação ───────────────────────────────────────────────────
JARVIS_WAKE_WORD_BACKEND=push_to_talk  # ou: porcupine

# ── Memória ────────────────────────────────────────────────────
JARVIS_MEMORY_SQLITE_VEC_VERSION=0.1.3
JARVIS_EMBEDDING_MODEL=mlx-community/Qwen3-Embedding-0.6B-bf16
```

---

## 14. Estrutura do Projeto

A estrutura abaixo assume o JARVIS como um monorepo macOS-first: Python fica responsável pelo control-plane, Swift abriga apenas os bridges nativos Apple e os testes Python são organizados por camada. Ela continua evolutiva, mas já resolve a ambiguidade entre `jarvis/`, `swift/` e `tests/` achatados na raiz.

```
J.A.R.V.I.S/
│
├── pyproject.toml                   # uv + setuptools + pytest + ruff
├── .env.example                     # configuração inicial mínima
├── README.md                        # onboarding curto e operacional
│
├── docs/
│   ├── architecture/
│   │   └── JARVIS.md                # concepção geral e arquitetura alvo
│   ├── guides/                      # setup, uso e operação no macOS
│   ├── testing/                     # estratégia de testes e convenções
│   └── decisions/                   # ADRs / decisões importantes
│
├── scripts/                         # bootstrap, doctor, helpers operacionais
│
├── src/
│   └── jarvis/
│       ├── main.py                  # CLI foreground · entrypoint
│       ├── config.py                # pydantic-settings · carrega .env
│       ├── bus.py                   # event bus asyncio → redis depois
│       ├── runtime.py               # orquestração do runtime conversacional
│       ├── prompts.py               # prompt base e perfis de sistema
│       ├── observability/
│       │   └── logging.py           # structlog · contexto · renderers
│       │
│       ├── core/
│       │   ├── state_machine.py     # estados e transições
│       │   ├── turn_manager.py      # ★ barge-in · fim de turno · yield
│       │   ├── sentence_streamer.py # ★ buffer LLM→TTS por sentença
│       │   ├── resource_governor.py # Metal limits + max_kv_size
│       │   ├── complexity_router.py # ★ hot path vs. deliberativo
│       │   ├── dialogue_manager.py  # contexto · prompt · orçamento memória
│       │   ├── action_broker.py     # ★ contratos tipados para ações
│       │   ├── capability_broker.py # ★ ledger de capabilities · escopos
│       │   └── policy_engine.py     # degradação automática de modelo
│       │
│       ├── adapters/
│       │   ├── interfaces.py        # Protocols @runtime_checkable
│       │   ├── activation/
│       │   │   ├── push_to_talk.py
│       │   │   └── porcupine.py
│       │   ├── vad/
│       │   │   ├── speech_detector.py
│       │   │   └── silero_vad.py
│       │   ├── stt/
│       │   │   ├── speech_analyzer.py
│       │   │   ├── mlx_whisper.py
│       │   │   ├── faster_whisper.py
│       │   │   └── whisper_cpp.py
│       │   ├── llm/
│       │   │   ├── foundation_models.py
│       │   │   ├── mlx_lm.py
│       │   │   ├── ollama.py
│       │   │   ├── anthropic.py
│       │   │   └── openai.py
│       │   ├── tts/
│       │   │   ├── mlx_audio_kokoro.py
│       │   │   ├── mlx_audio_qwen3.py
│       │   │   ├── kokoro_onnx.py
│       │   │   ├── piper.py
│       │   │   ├── avspeech.py
│       │   │   └── elevenlabs.py
│       │   └── memory/
│       │       ├── sqlite_vec.py
│       │       └── postgres.py
│       │
│       ├── memory/
│       │   ├── store.py             # working · episodic · profile · procedural
│       │   ├── relevance.py         # classificador — quando gravar
│       │   ├── provenance.py        # ★ origin · confidence · recency · scope
│       │   └── embedding.py         # Qwen3 Embedding · BGE-M3
│       │
│       ├── audio/
│       │   └── playback.py          # playback local e fallbacks
│       │
│       ├── tools/
│       │   ├── __init__.py          # registry e despacho
│       │   ├── security.py          # trust boundary · intent isolation
│       │   ├── system.py            # App Intents · AppleScript
│       │   ├── browser.py
│       │   ├── calendar.py
│       │   ├── shell.py
│       │   ├── files.py
│       │   └── timer.py
│       │
│       └── models/                  # eventos, ações, estado, memória, conversa
│
├── bridges/
│   └── apple/
│       ├── SpeechAnalyzerCLI/       # STT + VAD nativos — NDJSON via subprocess
│       │   ├── Package.swift
│       │   ├── Sources/
│       │   └── Tests/               # swift-testing
│       └── FoundationModelsBridge/  # hot path LLM — sessão stateful via HTTP/SSE
│           ├── Package.swift
│           ├── Sources/
│           └── Tests/               # swift-testing
│
└── tests/
    └── python/
        ├── unit/                    # regras puras e módulos pequenos
        ├── integration/             # adapters e fluxos entre componentes
        ├── e2e/                     # smoke tests de ponta a ponta
        ├── fixtures/                # fixtures compartilhadas
        └── conftest.py              # bootstrap pytest + marcação por camada
```

---

## 15. Dependências

```toml
[project]
name = "jarvis"
requires-python = ">=3.12"

dependencies = [
    # ML — pool Metal compartilhado (Apple Silicon only)
    "mlx-lm>=0.24.0",               # LLM deliberativo — Qwen3 8B Q4
    "mlx-audio>=0.4.0",             # TTS — Kokoro-82M (macOS · Apple Silicon only)

    # Áudio
    "sounddevice",                  # captura + playback
    "pynput",                       # hotkey global
    "numpy",
    "torch",                        # Silero VAD fallback — CPU only, não Metal
    "torchaudio",

    # Memória e embedding
    "sqlite-vec==0.1.3",            # PINADA — pré-v1
    "sentence-transformers",        # Qwen3 Embedding · BGE-M3

    # Configuração
    "pydantic-settings",

    # Ferramentas macOS
    "pyobjc-framework-EventKit",
    "pyobjc-framework-AppKit",
]

[project.optional-dependencies]
cloud              = ["anthropic", "openai"]
wake_word          = ["pvporcupine"]
cross_platform_tts = ["piper-tts"]
dev                = ["pytest", "pytest-asyncio", "ruff", "mypy"]
```

**Instalados fora do pip:**

| Componente             | Instalação                                                  | Papel                          |
| ---------------------- | ----------------------------------------------------------- | ------------------------------ |
| SpeechAnalyzerCLI      | `swift build -c release` em `bridges/apple/SpeechAnalyzerCLI/`      | STT + VAD nativo · NDJSON      |
| FoundationModelsBridge | `swift build -c release` em `bridges/apple/FoundationModelsBridge/` | Hot path LLM · sessão stateful |
| espeak-ng              | `brew install espeak-ng`                                    | Kokoro phonemes pt-BR          |
| ffmpeg                 | `brew install ffmpeg`                                       | mlx-audio encoding             |

---

## 16. SLOs Operacionais

> Os valores abaixo são **targets de projeto** baseados em observações no ambiente de referência (M4 16 GB). Devem ser revalidados por benchmark no hardware-alvo antes de serem tratados como garantias.

### Latência

| Etapa                             | Target       |
| --------------------------------- | ------------ |
| IDLE → LISTENING após hotkey      | < 100 ms     |
| STT — SpeechAnalyzer              | < 100 ms     |
| Turn Manager — validação de turno | < 150 ms     |
| Recuperação semântica de memória  | < 50 ms      |
| **Ponta a ponta — hot path**      | **< 500 ms** |
| **Ponta a ponta — deliberativo**  | **< 1.2 s**  |
| **Barge-in → TTS parado**         | **< 150 ms** |

### Qualidade conversacional

| Métrica                                | Target                |
| -------------------------------------- | --------------------- |
| Taxa de falso corte de turno           | < 3%                  |
| Taxa de yield correto após barge-in    | > 97%                 |
| Taxa de ação correta no primeiro turno | > 93%                 |
| WER pt-BR — SpeechAnalyzer             | < 5% (revalidar)      |
| WER pt-BR — mlx-whisper fallback       | < 7% (revalidar)      |
| Qualidade TTS pt-BR — Kokoro           | Validar auditivamente |

### Recursos e estabilidade

| Métrica                 | Target          |
| ----------------------- | --------------- |
| Pico de memória Metal   | < 9.5 GB        |
| Thermal pressure macOS  | Nunca "Serious" |
| Uptime por sessão de 8h | > 99.5%         |

---

## 17. Tabela de Decisões

| Decisão                                          | Razão                                                                                                  |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| Python como control-plane                        | Cross-platform é objetivo declarado; Swift não porta para Linux/Windows/VPS                            |
| Swift como adapter no Mac                        | SpeechAnalyzer e Foundation Models são Swift-only — encapsulados como CLIs com protocolo NDJSON        |
| Turn Manager como primeiro-classe                | É o mecanismo da naturalidade — não pode ser detalhe de implementação                                  |
| Sentence Streamer como primeiro-classe           | Diferença entre ~580ms e ~1.2s de silêncio percebido; segmentação robusta para pt-BR                   |
| Papéis de acelerador como intenção, não contrato | O escalonamento interno dos frameworks nativos Apple pertence ao runtime da plataforma                 |
| SpeechDetector como VAD primário no Mac          | API nativa Apple para "há fala?" — consistente com o pipeline SpeechAnalyzer; Silero como fallback     |
| SpeechAnalyzer como STT padrão                   | Significativamente mais rápido que Whisper (benchmarks comunitários); fora do pool Metal; pt_BR nativo |
| Protocolo NDJSON nos bridges Swift               | Carrega metadados (tipo, confiança, timestamps); Foundation Models como sessão, não request avulso     |
| Foundation Models no hot path                    | Grátis · runtime nativo · tool calling · sessões stateful — ideal para tarefas simples                 |
| Foundation Models não como cérebro principal     | Apple documenta: "not designed to be a chatbot" — ~3B insuficiente para raciocínio complexo            |
| Qwen3 8B Q4 no deliberativo                      | Melhor all-rounder para 16 GB; salto de qualidade real sobre 4B                                        |
| `max_kv_size` obrigatório em toda chamada        | KV sem limite causa kernel panic em qualquer hardware — issue ativo mlx-lm                             |
| Nunca `mlx_lm.server`                            | Fia 75% da RAM sem `max_kv_size` — kernel panic documentado                                            |
| Kokoro como TTS padrão no Mac                    | AVSpeechSynthesizer rejeitado por qualidade; Kokoro candidato mais forte — validar pt-BR               |
| Kokoro-ONNX / Piper como TTS cross-platform      | mlx-audio é Apple Silicon only — Linux/Windows precisam de runtime real                                |
| CapabilityBroker como primeiro-classe            | Torna explícito o que está habilitado, em que escopo e com que confirmações                            |
| Política de memória: FTS5 primeiro, vetor async  | Recuperação semântica não bloqueia o hot path; orçamento < 50ms                                        |
| System Prompt alinhado ao regime de risco        | "Age sem confirmação" só para read-only e write-safe — não para ações destrutivas                      |
| 4 classes de memória + provenance                | Working/Episodic/Profile/Procedural mais preciso; provenance evita alucinação de familiaridade         |
| Action Broker com contratos tipados              | Parsing livre de texto para ações operacionais é frágil e inseguro                                     |
| App Intents sobre AppleScript                    | Mais estável · oficial · bem suportado pela Apple                                                      |
| Wake word: Porcupine sobre OpenWakeWord          | OpenWakeWord não suporta pt-BR — razão linguística, não de plataforma                                  |
| ResourceGovernor com três knobs mlx.core         | `set_memory_limit` · `set_wired_limit` · `set_cache_limit` — controle completo do pool Metal           |
| sqlite-vec pinado com upgrade path               | Pré-v1 · brute-force — aceitável para uso pessoal; pgvector quando necessário                          |
| STT e TTS sempre locais                          | Latência de áudio não tolera round-trip de rede                                                        |
| Event bus asyncio → Redis                        | Começa simples; escala sem mudar os módulos                                                            |

---

## 18. System Prompt

```
Você é J.A.R.V.I.S., assistente pessoal de [seu nome].

IDENTIDADE
Específico para [seu nome]. Não é um assistente genérico.
Conhece as preferências, o trabalho e a rotina de [seu nome].
Tem iniciativa e bom julgamento.

TOM
Direto. Sem rodeios. Sem "claro!", "ótimo!", "com certeza!".
Culto, preciso, levemente irônico quando o contexto permite.
Age sem pedir confirmação em leitura e em ações write-safe autorizadas.
Em ações destrutivas, externas ou irreversíveis, confirma antes de executar.
Fala sempre em português do Brasil. Natural, não formal.

FORMATO  (você está sendo lido em voz alta)
Sem markdown. Sem asteriscos, hífens de lista ou hashtags.
Frases curtas. Máximo 3 frases por turno em conversa casual.
Extenso apenas quando a tarefa exige.
Prefira "confirmado" a "ok".
Prefira "não encontrei" a "infelizmente não foi possível localizar".

COMPORTAMENTO
Tem informação: usa. Não pergunta o que já sabe.
Falta contexto crítico: uma pergunta, a mais específica possível.
Executou tool: confirma o resultado — não o processo.
Proativo: se perceber algo relevante (prazo, conflito, contexto), menciona.
```

---

## Resumo

O J.A.R.V.I.S. é um sistema local-first de conversa em tempo real com **Turn Manager** e **Sentence Streamer** como componentes centrais da naturalidade, **Python como control-plane portável** com adapters Swift nativos no Mac para STT (SpeechAnalyzer) e LLM hot path (Foundation Models) — ambos via protocolo NDJSON fora do pool Metal —, **LLM deliberativo** Qwen3 8B Q4 com KV cache governado pelo ResourceGovernor, **TTS** via mlx-audio Kokoro no Mac e Kokoro-ONNX/Piper em outras plataformas, **memória** em 4 classes com provenance e recuperação assíncrona, **CapabilityBroker** como ledger explícito de permissões, e **Action Broker** com contratos tipados.

Projetado para latência de referência abaixo de 500ms no hot path e 1.2s no deliberativo, pool Metal governado, thermal seguro no MacBook Air M4, escalável por substituição de adapters sem refatorar o core.
