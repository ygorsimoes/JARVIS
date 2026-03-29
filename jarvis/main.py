from __future__ import annotations

import argparse
import asyncio
from typing import Optional

from .config import load_config
from .runtime import JarvisRuntime


async def _run_demo(
    prompt: Optional[str],
    interactive: bool,
    voice: bool,
    use_native_backends: bool,
) -> None:
    config = load_config()
    runtime = JarvisRuntime.from_config(config, enable_native_backends=use_native_backends)
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
            print("J.A.R.V.I.S. em modo voz. Use a ativacao configurada para iniciar um turno.")
            await runtime.run_voice_foreground()
            return

        raise SystemExit("use --demo, --interactive ou --voice")
    finally:
        await runtime.shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. foreground runtime")
    parser.add_argument("--demo", help="executa um turno de texto unico")
    parser.add_argument("--interactive", action="store_true", help="inicia o modo texto interativo")
    parser.add_argument("--voice", action="store_true", help="inicia o loop foreground de voz")
    parser.add_argument(
        "--use-native-backends",
        action="store_true",
        help="usa os adapters reais de Foundation Models e MLX quando disponiveis",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(
        _run_demo(
            prompt=args.demo,
            interactive=args.interactive,
            voice=args.voice,
            use_native_backends=args.use_native_backends,
        )
    )


if __name__ == "__main__":
    main()
