# J.A.R.V.I.S — A Just Really Versatile Interactive System

> **Objetivo:** fluidez, clareza, naturalidade e elegância — sem travar, com delay mínimo percebido.
> **Plataforma:** MacBook Air M4 · 16 GB RAM · macOS Tahoe

---

## Visão geral: duas trilhas simultâneas

O princípio central desta arquitetura é separar o sistema em duas trilhas que rodam em
paralelo e se comunicam por eventos, nunca em sequência bloqueante:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  TRILHA PERCEPTIVA          reação quase imediata, sem LLM               │
│                                                                          │
│  Porcupine  ·  Silero VAD  ·  AEC  ·  barge-in  ·  reparo cache          │
│  score de commit  ·  classificação especulativa  ·  acknowledgment       │
└──────────────────────────────────────────┬───────────────────────────────┘
                                           │
                               eventos assíncronos
                                           │
┌──────────────────────────────────────────┴───────────────────────────────┐
│  TRILHA SEMÂNTICA           pode ser um pouco mais cuidadosa             │
│                                                                          │
│  ASR estável  ·  resolução de ato  ·  planner  ·  realizer  ·  TTS       │
└──────────────────────────────────────────────────────────────────────────┘
```

Quando as duas trilhas ficam acopladas sequencialmente, o sistema
"pensa demais antes de respirar". O usuário percebe isso como travada,
mesmo que a latência real não seja alta.

---

## Fluxo principal — máquina de estados

```
╔══════════════════════════════════════════════════════════════════════════╗
║   ORQUESTRADOR — ciclo de conversa                                       ║
╚══════════════════════════════════════════════════════════════════════════╝

             ┌───────────────────────────────────────────────────────┐
             │                                                       │
             ▼                                                       │
          ┌──────┐                                                   │
          │ IDLE │   apenas Porcupine ativo, < 1% CPU                │
          └──┬───┘                                                   │
             │ wake word detectada                                   │
             ▼                                                       │
          ┌────────┐   silêncio > 3s                                 │
          │ ARMED  │ ─────────────────► cancela silenciosamente ─────┤
          └──┬─────┘                                                 │
             │ energia + VAD confirmam fala                          │
             │ pre-roll buffer já ativo                              │
             ▼                                                       │
    ┌─────────────────────────────────────────┐                      │
    │  CAPTURE_PREVIEW                        │                      │
    │                                         │                      │
    │  VAD contínuo (Silero, < 1ms/chunk)     │                      │
    │  ASR parcial em streaming               │                      │
    │                                         │                      │
    │  score contínuo de commit:              │                      │
    │    estabilidade acústica                │                      │
    │    estabilidade lexical                 │                      │
    │    probabilidade de continuação         │                      │
    │    probabilidade de correção            │                      │
    │                                         │                      │
    │  classificação especulativa do ato:     │                      │
    │    "não, na verdade…" → corretivo       │                      │
    │    "explica melhor…"  → expansão        │                      │
    │    "sim" / "aham"     → confirmação     │                      │
    │                                         │                      │
    │  ato especulativo → DISCOURSE_STATE     │                      │
    └──────────────────┬──────────────────────┘                      │
                       │ score cruza limiar                          │
                       ▼                                             │
                   ┌────────┐                                        │
                   │ COMMIT │  decisão que já vinha sendo            │
                   └───┬────┘  construída durante CAPTURE_PREVIEW    │
                       │                                             │
          ┌────────────┼──────────────────────┐                      │
          │            │                      │                      │
          ▼            ▼                      ▼                      │
    ruído / curto   fragmento /         confirmação /                │
    < 300ms         vazio / incerto     ato determinístico           │
          │            │                      │                      │
          ▼            ▼                      ▼                      │
     descarte      REPAIR_POLICY          FAST_ACT                   │
     silencioso    (sem LLM)              (cache PCM)                │
          │            │                      │                      │
          └────────────┴──────────────────────┘                      │
                       │                                             │
                       └─────────────────────────────────────────────┤
                                                                     │
    fala válida                                                      │
         │                                                           │
         ▼                                                           │
    DISCOURSE_STATE                                                  │
         │                                                           │
         ▼                                                           │
    RESPONSE_PLANNER                                                 │
         │                                                           │
         ▼                                                           │
    ┌──────────────────────────────────────────────────────┐         │
    │  SEMANTIC_RESPONSE                                   │         │
    │                                                      │         │
    │  ASR estável                                         │         │
    │       │                                              │         │
    │       ▼                                              │         │
    │  RESPONSE_PLANNER                                    │         │
    │       │                                              │         │
    │       ▼                                              │         │
    │  SPEECH_SURFACE_REALIZER                             │         │
    │       │                                              │         │
    │       ▼                                              │         │
    │  TTS por cláusula estável                            │         │
    │  (não por token — por bloco prosódico)               │         │
    └──────────────────────┬───────────────────────────────┘         │
                           │                                         │
                           ▼                                         │
    ┌──────────────────────────────────────────────────────┐         │
    │  SPEAK                                               │         │
    │                                                      │         │
    │  Kokoro pt-BR / Piper pt-BR / XTTS-v2                │         │
    │  wake word DESLIGADA durante fala                    │         │
    │                                                      │         │
    │  barge-in SEMPRE ativo (trilha perceptiva):          │         │
    │    AEC filtra eco do próprio TTS                     │         │
    │    energia + VAD + duração mínima >= 400ms           │         │
    │    janela de supressão 200ms no início do TTS        │         │
    │    corte imediato ao detectar barge-in real          │         │
    │                                                      │         │
    │  intenção da interrupção → DISCOURSE_STATE:          │         │
    │    complemento  "e também…"   → continua contexto    │         │
    │    correção     "não, era…"   → ato corretivo        │         │
    │    expansão     "explica…"    → aprofunda            │         │
    │    mudança      novo assunto  → reseta estado        │         │
    └──────────────────────┬───────────────────────────────┘         │
                           │                                         │
                           ▼                                         │
    ┌──────────────────────────────────────────────────────┐         │ 
    │  FOLLOWUP_WINDOW                                     │         │ 
    │                                                      │         │
    │  sem wake word — janela conversacional aberta        │         │
    │                                                      │         │
    │  duração pelo fechamento do turno anterior:          │         │
    │    conclusivo           →  curta       (~4s)         │         │
    │    aberto / consultivo  →  longa       (~14s)        │         │
    │    corretivo            →  média       (~8s)         │         │
    │    suspenso             →  longa + tom acolhedor     │         │
    │    pós-interrupção      →  curta (já engajado)       │         │
    │                                                      │         │
    │  nova fala → CAPTURE_PREVIEW (sem wake word)         │         │
    │  silêncio longo → IDLE silencioso                    │         │
    └──────────────────────────────────────────────────────┘         │
                           │                                         │
                           └─────────────────────────────────────────┘
```

---

## Estados em detalhe

### IDLE
```
  · apenas Porcupine ativo
  · < 1% CPU
  · nenhum modelo quente além do wake word
  · retorno ao IDLE sempre silencioso — sem bip, sem voz
```

### ARMED
```
  · pre-roll buffer ligado (captura os primeiros 200–400ms de fala)
  · timeout 3s sem fala → cancela → IDLE
  · nenhuma carga extra de modelo
```

### CAPTURE_PREVIEW  *(substitui CAPTURING + FINALIZING)*
```
  · VAD contínuo (Silero, sempre quente)
  · ASR parcial em streaming → texto provisório atualizado a cada 200ms
  · score de commit calculado continuamente:

      [ estabilidade acústica ]  × peso_a
    + [ estabilidade lexical   ]  × peso_l
    + [ prob. de continuação   ]  × peso_c  (invertido)
    + [ prob. de correção      ]  × peso_k  (invertido)
      ─────────────────────────────────────────────────
    = commit_score  ∈  [0, 1]

  · endpoint adaptativo (threshold, não estado):
      turno anterior conclusivo  → threshold mais baixo
      turno anterior aberto      → threshold mais alto
      pitch descendente + pausa  → peso extra para commit

  · FINALIZING eliminado como estado separado
    (economiza 100–200ms de hesitação percebida)
```

### COMMIT
```
  · momento em que commit_score cruza o limiar
  · não é espera — é a confirmação de uma decisão progressiva

  classificação do resultado:
    vazio / ruído curto    → REPAIR: descarte silencioso
    fragmento              → REPAIR: "não peguei o final"
    incerteza              → REPAIR: "acho que entendi, confirma:"
    ausência               → REPAIR: "pode repetir?"
    ato ambíguo            → interpreta por DISCOURSE_STATE (sem LLM)
    confirmação simples    → FAST_ACT
    fala válida            → DISCOURSE_STATE → RESPONSE_PLANNER
```

### REPAIR_POLICY  *(trilha perceptiva — sem LLM)*
```
  política de latência:
    0 – 900ms              silêncio limpo
    900ms – 1.8s           micro acknowledgment (pré-sintetizado)
    > 1.8s                 filler verbal contextual (pool rotativo)
    nunca repetir filler em turnos consecutivos

  pool rotativo por tipo (3–5 variantes, pré-sintetizadas em PCM):
    ausência     →  "pode repetir?"  /  "não captei"  /  "me fala de novo"
    fragmento    →  "não peguei o final"  /  "repete o final?"
    incerteza    →  "acho que entendi — confirma:"  /  "quer dizer…?"
    descarte     →  silêncio limpo (sem voz)
    ato ambíguo  →  resolve por DISCOURSE_STATE
```

### FAST_ACT
```
  para atos determinísticos: confirmação, fechamento, ack simples
  usa exclusivamente áudio pré-sintetizado em cache PCM
  latência: ~0ms de TTS (só playback)

  exemplos:
    "entendido"  /  "certo"  /  "pode ser"  /  "claro"
    "um momento" /  "tá bom" /  "com certeza"
```

### DISCOURSE_STATE  *(4 eixos contínuos)*
```
  eixo 1 — continuidade   [0.0 novo assunto  ←────────→  1.0 continuação  ]
  eixo 2 — urgência       [0.0 calmo         ←────────→  1.0 com pressa   ]
  eixo 3 — profundidade   [0.0 breve         ←────────→  1.0 detalhado    ]
  eixo 4 — tom            [0.0 factual       ←────────→  1.0 acolhedor    ]

  atualizado após cada turno completo
  peso de turnos antigos decai progressivamente por turno
  persistência parcial: retorno < 5min restaura com 50% de peso
  descartado: IDLE por silêncio longo (> 5min)

  influencia: REPAIR_POLICY · RESPONSE_PLANNER · FOLLOWUP · prosódia TTS
```

### RESPONSE_PLANNER
```
  decide o ato de fala antes de gerar qualquer conteúdo:
    responder     → gera conteúdo
    confirmar     → FAST_ACT direto
    acolher       → tom antes de conteúdo  (eixo 4 > 0.6)
    corrigir      → abre correção suave
    pedir detalhe → pergunta específica e curta
    continuar     → retoma sem reiniciar contexto
    fechar        → encerra com clareza

  calibração de tamanho por DISCOURSE_STATE:
    urgência > 0.7           → máx 1–2 frases
    pergunta simples         → 1–2 frases
    pergunta aberta          → 3–4 frases
    pedido de explicação     → parágrafos, sinalizado ("então…", "primeiro…")
    N >= 2 interrupções      → encurta agressivamente

  metadados de prosódia para o TTS:
    confirmação / fechamento  → leve, ascendente
    explicação / continuação  → pausado, deliberado
    correção                  → neutro, firme
    acolhimento               → suave, volume ligeiramente mais baixo
    reparo                    → ascendente, levemente mais lento
```

### SPEECH_SURFACE_REALIZER  *(camada entre planner e TTS)*
```
  transforma resposta semântica em fala boa — não apenas texto correto

  responsabilidades:
    encurtar frases longas
    quebrar ideias em cláusulas faláveis (uma ideia por frase)
    expandir siglas       VAD   → "detector de atividade de voz"
    normalizar números    3.5   → "três vírgula cinco"
    normalizar datas      01/04 → "primeiro de abril"
    normalizar unidades   400ms → "quatrocentos milissegundos"
    remover marcas de escrita   **, ##, -, listas, títulos
    ajustar pontuação para pausa natural
    anotar metadados de prosódia por cláusula

  exemplo de transformação:

    ENTRADA (texto de tela):
      "O ideal é usar VAD + ASR streaming e, se necessário,
       fallback para repair com 3–5 variantes pré-sintetizadas."

    SAÍDA (3 blocos prosódicos):
      [bloco 1 · pausado ]  "O ideal é usar detector de atividade de voz
                             com transcrição em tempo real."
      [bloco 2 · leve    ]  "E, se precisar, cair para um reparo curto."
      [bloco 3 · neutro  ]  "Com algumas variações pré-gravadas."
```

### SEMANTIC_RESPONSE  *(pipeline de geração)*
```
  faster-whisper small  → transcrição estável pt-BR
  Qwen3 4B /no_think    → executa ato decidido pelo planner
  SPEECH_SURFACE_REALIZER → transforma para fala
  TTS por cláusula estável → não por token, não por resposta completa

  política de latência de geração:
    < 400ms    → silêncio limpo
    400–900ms  → silêncio ainda (não mascarar o que não é longo)
    900ms–1.8s → micro acknowledgment pré-sintetizado
    > 1.8s     → filler verbal contextual (pool rotativo, sem repetição)
    > 3s       → "um momento…" + aborta se necessário

  chunking do TTS:
    aguarda primeira cláusula estável (não o primeiro token)
    começa a falar ao fim da primeira cláusula
    continua em blocos prosódicos
    nunca corta no meio de uma ideia
```

---

## Tratamento de falha

```
  evento                           ação                      próximo estado
  ────────────────────────────────────────────────────────────────────────
  armed + silêncio > 3s            cancela silenciosamente   IDLE
  fala < 300ms / ruído             descarte silencioso       FOLLOWUP
  ASR vazio                        REPAIR: ausência          FOLLOWUP
  LLM < 400ms                      silêncio limpo            aguarda
  LLM 400ms – 900ms                silêncio limpo            aguarda
  LLM 900ms – 1.8s                 micro acknowledgment      aguarda
  LLM > 1.8s                       filler contextual         aguarda
  LLM timeout > 3s                 "um momento…" + aborta    FOLLOWUP
  TTS erro                         loga silencioso           FOLLOWUP
  barge-in < 400ms                 ignora (eco residual)     continua
  barge-in >= 400ms                corte TTS + captura       CAPTURE_PREVIEW
  ────────────────────────────────────────────────────────────────────────
```

---

## Disciplina de carga — MacBook Air M4

```
  estado             carga ativa
  ────────────────────────────────────────────────────────────────────
  IDLE               Porcupine apenas                   (< 1% CPU)
  ARMED              + pre-roll buffer
  CAPTURE_PREVIEW    + Silero VAD + faster-whisper       (Metal / ANE)
  COMMIT             leve — score já calculado
  REPAIR / FAST      playback PCM pré-carregado          (~0ms latência)
  RESPONDING         Qwen3 4B via llama.cpp              (GPU Metal M4)
  SPEAK              Kokoro / Piper via MPS + barge-in mínimo
  ────────────────────────────────────────────────────────────────────

  regra de ouro: transições de estado não implicam reinicialização
    modelos sempre quentes entre estados
    áudio não reconfigurado entre CAPTURE e SPEAK
    AVAudioEngine mantido ativo em toda a sessão
```

### Aceleração por componente

```
  componente            backend ideal no M4
  ──────────────────────────────────────────────────────────────────
  Porcupine             CPU            já leve, sem necessidade
  Silero VAD            CPU            < 1ms/chunk, já é suficiente
  AEC                   AVAudioEngine  voice processing mode nativo
  faster-whisper small  MLX / CoreML   3–5× sobre CPU puro
  Qwen3 4B              llama.cpp      Metal, GPU integrada M4
  Kokoro / Piper        MPS            Metal Performance Shaders
  ──────────────────────────────────────────────────────────────────
```

---

## Política de degradação elegante sob carga

```
  nível   memória livre    sacrifica
  ──────────────────────────────────────────────────────────────────────
    1       > 3 GB          nada — operação normal
    2       2 – 3 GB        TTS expressivo → voz rápida (Piper)
    3       1 – 2 GB        + comprimento máximo de resposta reduzido
                            + contexto conversacional encurtado
    4       < 1 GB          + expansões espontâneas suspensas
                            + respostas fixas em todos os atos simples
  ──────────────────────────────────────────────────────────────────────

  nunca sacrificado em nenhum nível:
    wake word (Porcupine)  ·  VAD (Silero)  ·  AEC
    barge-in acústico      ·  reparo curto pré-sintetizado
```

---

## Stack recomendada

```
  função               componente               observação
  ──────────────────────────────────────────────────────────────────────
  Wake word            Porcupine                modelo pt-BR se disponível
  VAD                  Silero VAD               sempre quente, < 1ms/chunk
  AEC                  AVAudioEngine            voice processing mode nativo
  ASR (principal)      faster-whisper small     fine-tuned pt-BR
  ASR (alternativa)    WhisperKit               API nativa Apple Silicon
  ASR (alternativa)    whisper.cpp              Core ML / ANE encoder
  LLM                  Qwen3 4B via llama.cpp   /no_think por padrão
  LLM (alternativa)    MLX-LM                   para modelos MLX nativos
  TTS padrão           Kokoro pt-BR             verificar treino pt-BR
  TTS leve / nível 2   Piper pt-BR              rápido, robusto
  TTS expressivo       XTTS-v2                  para momentos de riqueza
  Cache de reparo      PCM/WAV pré-carregado    ~20–30 frases em memória
  Orquestração         Python asyncio           ou Rust para latência crítica
  ──────────────────────────────────────────────────────────────────────
```

> **Nota TTS pt-BR:** verificar se o modelo foi treinado com dados de português
> brasileiro, não europeu. Um modelo treinado em pt-PT soa artificial no Brasil
> mesmo com texto correto — prosódia, ritmo e entoação são distintos.

> **Nota ASR pt-BR:** Whisper `base` tem taxa de erro notável em pt-BR.
> faster-whisper `small` corta o problema pela metade. Erros de transcrição
> geram REPAIRs desnecessários que o usuário sente como "o assistente não me
> entende" — e isso mata a naturalidade antes de qualquer refinamento de estado.

---

## Cadeia completa — visão de latência

```
  evento                          latência alvo    responsável
  ──────────────────────────────────────────────────────────────────────
  Wake word detectada              < 50ms           Porcupine
  VAD confirma início              < 5ms            Silero
  Primeira cláusula transcrita     < 200ms          faster-whisper stream
  Commit de turno                  ~0ms extra       score já calculado
  Classificação de ato             ~0ms extra       especulativa já feita
  FAST_ACT (cache)                 < 5ms            playback PCM
  REPAIR (cache)                   < 5ms            playback PCM
  Primeira cláusula LLM            50 – 200ms       Qwen3 4B + Metal
  Realizer processa cláusula       < 20ms           regras determinísticas
  Primeira cláusula TTS            50 – 150ms       Kokoro + Metal
  ──────────────────────────────────────────────────────────────────────
  Latência percebida total         ~300 – 500ms     resposta simples
  (do fim da fala do usuário ao início da fala do assistente)
  ──────────────────────────────────────────────────────────────────────
```

---

## Frases de cache — biblioteca mínima

Pré-sintetizar na inicialização e manter em memória como PCM:

```
  tipo              variantes (mínimo 3 por tipo)
  ──────────────────────────────────────────────────────────────────────
  Ausência          "pode repetir?"
                    "não captei"
                    "me fala de novo"

  Fragmento         "não peguei o final"
                    "repete o final?"
                    "pode terminar?"

  Incerteza         "acho que entendi — confirma:"
                    "quer dizer…?"
                    "entendi como… é isso?"

  Confirmação       "entendido"  ·  "certo"  ·  "pode ser"
                    "claro"  ·  "com certeza"  ·  "tá bom"

  Espera            "um momento"  ·  "só um segundo"  ·  "já vejo"

  Acolhimento       "entendo"  ·  "faz sentido"  ·  "imagino"

  Fechamento        "qualquer coisa é só falar"  ·  "pode chamar"
  ──────────────────────────────────────────────────────────────────────
  Total estimado    ~25–30 arquivos PCM  ·  ~5–10 MB em memória
  ──────────────────────────────────────────────────────────────────────
```

> **Regra de elegância:** nunca usar a mesma variante duas vezes seguidas.
> Controle de repetição obrigatório no pool rotativo.

---

## Resumo — 13 melhorias priorizadas

```
  FASE 1 — Fundação perceptiva         maior ganho, menor complexidade
  ──────────────────────────────────────────────────────────────────────
  1   AEC nativo via AVAudioEngine voice processing
      → elimina barge-in falso causado pelo próprio TTS

  2   Reparo e acknowledgments pré-sintetizados em cache PCM
      → latência zero nos atos mais frequentes

  3   Barge-in por energia + VAD + duração mínima >= 400ms
      → sem ASR; corte imediato do TTS

  4   Política de filler restrita
      → silêncio até 900ms · gesto leve até 1.8s · verbal acima disso


  FASE 2 — Reestruturação do pipeline de decisão
  ──────────────────────────────────────────────────────────────────────
  5   Colapsar FINALIZING em score contínuo de commit
      → elimina microvácuo de hesitação percebida (~100–200ms)

  6   Classificação especulativa do ato durante CAPTURE_PREVIEW
      → sistema "acompanha" em vez de "reanalisar do zero"

  7   DISCOURSE_STATE reduzido a 4 eixos contínuos com decay
      → comportamento mais fluido em casos de borda


  FASE 3 — Qualidade da resposta falada
  ──────────────────────────────────────────────────────────────────────
  8   Speech Surface Realizer como camada explícita entre planner e TTS
      → assistente soa como fala, não como texto lido

  9   TTS por cláusula estável
      → começa rápido sem sacrificar naturalidade prosódica

  10  ASR: faster-whisper small com dados pt-BR
      → reduz REPAIR desnecessário · melhora clareza percebida


  FASE 4 — Otimização sistêmica
  ──────────────────────────────────────────────────────────────────────
  11  Aceleração MLX/ANE para Whisper · MPS para Kokoro/Piper
      → headroom para streaming paralelo sem aquecer o Mac

  12  Modelos sempre quentes · sem cold start · AVAudioEngine contínuo
      → transições de estado imperceptíveis

  13  Política de degradação elegante em 4 níveis
      → nunca colapso total · responsividade sempre preservada
  ──────────────────────────────────────────────────────────────────────
```

---

## Princípio central de design

> **O próximo salto não é mais IA. É mais ritmo, mais timing e mais disciplina de conversa.**
>
> A arquitetura já está madura o suficiente para que refinamentos de orquestração
> produzam mais ganho percebido do que qualquer troca de modelo.
> Elegância conversacional vem de poucos controles bem calibrados,
> não de taxonomias extensas.