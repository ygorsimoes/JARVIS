from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import replace

from loguru import logger

from .config import AppConfig, load_config


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.env_file)
    if args.log_level:
        config = replace(config, log_level=args.log_level.upper())

    _configure_logging(config.log_level)

    try:
        if args.command == "devices":
            return _run_devices()
        if args.command == "transcribe":
            return asyncio.run(_run_transcribe(_apply_device_overrides(config, args)))
        if args.command == "chat":
            return asyncio.run(_run_chat(_apply_chat_overrides(config, args)))
    except KeyboardInterrupt:
        logger.info("Encerrado pelo usuario")
        return 130
    except RuntimeError as exc:
        logger.error("{}", exc)
        return 1
    except Exception as exc:
        logger.exception("Falha ao executar a CLI: {}", exc)
        return 1

    parser.print_help()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jarvis")
    parser.add_argument("--env-file", help="Arquivo .env opcional")
    parser.add_argument(
        "--log-level",
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de log da CLI",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("devices", help="Lista os devices de audio locais")

    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Escuta o microfone e mostra as transcricoes no terminal",
    )
    transcribe_parser.add_argument("--input-device", type=int, help="Indice do device de entrada")

    chat_parser = subparsers.add_parser(
        "chat",
        help="Sobe o assistente local completo com STT, LLM e TTS",
    )
    chat_parser.add_argument("--input-device", type=int, help="Indice do device de entrada")
    chat_parser.add_argument("--output-device", type=int, help="Indice do device de saida")
    chat_parser.add_argument("--model", help="Modelo principal do Ollama")
    chat_parser.add_argument("--fallback-model", help="Modelo de fallback do Ollama")
    chat_parser.add_argument("--voice", help="Voice id do Kokoro")

    return parser


def _configure_logging(level: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
        ),
    )


def _run_devices() -> int:
    import pyaudio

    audio = pyaudio.PyAudio()
    try:
        default_input_index = _default_device_index(audio, is_input=True)
        default_output_index = _default_device_index(audio, is_input=False)

        print("Devices de audio locais:\n")
        for index in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(index)
            roles: list[str] = []
            if int(info.get("maxInputChannels", 0)) > 0:
                roles.append("in")
            if int(info.get("maxOutputChannels", 0)) > 0:
                roles.append("out")

            markers: list[str] = []
            if index == default_input_index:
                markers.append("default-in")
            if index == default_output_index:
                markers.append("default-out")

            role_label = ", ".join(roles) or "n/a"
            marker_label = f" [{' '.join(markers)}]" if markers else ""
            print(
                f"{index:>2} | {info.get('name', 'desconhecido')} | {role_label}"
                f" | {int(float(info.get('defaultSampleRate', 0)))}Hz{marker_label}"
            )
    finally:
        audio.terminate()

    return 0


async def _run_transcribe(config: AppConfig) -> int:
    from pipecat.pipeline.runner import PipelineRunner

    from .pipeline import build_transcribe_task

    logger.info(
        "[transcribe] ouvindo microfone | whisper={} | input_device={}",
        config.whisper_model,
        _device_label(config.input_device_index),
    )
    logger.info("[transcribe] pressione Ctrl+C para encerrar")

    task = build_transcribe_task(config)
    runner = PipelineRunner(handle_sigint=sys.platform != "win32")
    await runner.run(task)
    return 0


async def _run_chat(config: AppConfig) -> int:
    from pipecat.pipeline.runner import PipelineRunner

    from .pipeline import build_chat_task

    logger.info(
        "[chat] iniciando sessao local | input_device={} | output_device={}",
        _device_label(config.input_device_index),
        _device_label(config.output_device_index),
    )
    logger.info("[chat] aguardando fala do usuario | pressione Ctrl+C para encerrar")

    task = build_chat_task(config)
    runner = PipelineRunner(handle_sigint=sys.platform != "win32")
    await runner.run(task)
    return 0


def _apply_device_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    input_device = args.input_device if args.input_device is not None else config.input_device_index
    return replace(config, input_device_index=input_device)


def _apply_chat_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    return replace(
        config,
        input_device_index=(
            args.input_device if args.input_device is not None else config.input_device_index
        ),
        output_device_index=(
            args.output_device if args.output_device is not None else config.output_device_index
        ),
        ollama_model=args.model or config.ollama_model,
        ollama_fallback_model=args.fallback_model or config.ollama_fallback_model,
        kokoro_voice=args.voice or config.kokoro_voice,
    )


def _default_device_index(audio, *, is_input: bool) -> int | None:
    try:
        info = (
            audio.get_default_input_device_info()
            if is_input
            else audio.get_default_output_device_info()
        )
    except OSError:
        return None

    return int(info["index"])


def _device_label(index: int | None) -> str:
    return str(index) if index is not None else "default"
