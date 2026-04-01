from __future__ import annotations

import argparse
import asyncio
from typing import Optional

from .config import load_config
from .doctor import run_doctor
from .observability import configure_logging
from .runtime import JarvisRuntime


async def _run_demo(
    prompt: Optional[str],
    interactive: bool,
    voice: bool,
    use_native_backends: bool,
) -> None:
    config = load_config()
    configure_logging(config.log_level, config.log_format)
    runtime = JarvisRuntime.from_config(
        config, enable_native_backends=use_native_backends
    )
    try:
        if prompt:
            response = await runtime.respond_text(prompt)
            print(response.full_text)
            return

        if interactive:
            print("J.A.R.V.I.S. em modo texto. Digite 'sair' para encerrar.")
            while True:
                user_input = input("voce> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in {"sair", "exit", "quit"}:
                    return
                response = await runtime.respond_text(user_input)
                print("jarvis> %s" % response.full_text)
            return

        if voice:
            print(
                "J.A.R.V.I.S. em modo voz. Use a ativacao configurada para iniciar um turno."
            )
            await runtime.run_voice_foreground()
            return

        raise SystemExit("use --demo, --interactive ou --voice")
    finally:
        await runtime.shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. foreground runtime")
    parser.add_argument("--demo", help="executa um turno de texto unico")
    parser.add_argument(
        "--interactive", action="store_true", help="inicia o modo texto interativo"
    )
    parser.add_argument(
        "--voice", action="store_true", help="inicia o loop foreground de voz"
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="valida se o ambiente macOS esta pronto para o runtime real",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.doctor:
        config = load_config()
        configure_logging(config.log_level, config.log_format)
        report = asyncio.run(run_doctor(config))
        raise SystemExit(1 if report.has_blockers else 0)

    asyncio.run(
        _run_demo(
            prompt=args.demo,
            interactive=args.interactive,
            voice=args.voice,
            use_native_backends=True,
        )
    )


if __name__ == "__main__":
    main()
