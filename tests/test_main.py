import argparse
import types
import unittest
from unittest.mock import AsyncMock, patch

from jarvis import main as main_module


class MainModuleTests(unittest.TestCase):
    def test_build_parser_supports_expected_flags(self):
        parser = main_module.build_parser()

        args = parser.parse_args(["--demo", "oi", "--use-native-backends"])

        self.assertEqual(args.demo, "oi")
        self.assertTrue(args.use_native_backends)
        self.assertFalse(args.interactive)
        self.assertFalse(args.voice)

    def test_main_invokes_asyncio_run_with_parsed_arguments(self):
        parsed_args = argparse.Namespace(
            demo="ola",
            interactive=False,
            voice=False,
            use_native_backends=True,
        )

        with patch.object(main_module, "build_parser") as build_parser:
            with patch("jarvis.main.asyncio.run") as run_asyncio:
                build_parser.return_value.parse_args.return_value = parsed_args
                main_module.main()

        run_asyncio.assert_called_once()
        run_asyncio.call_args.args[0].close()

    def test_run_demo_executes_prompt_mode_and_shuts_down_runtime(self):
        runtime = AsyncMock()
        runtime.respond_text.return_value = types.SimpleNamespace(
            full_text="resposta final"
        )

        with patch("jarvis.main.load_config", return_value=object()):
            with patch("jarvis.main.JarvisRuntime.from_config", return_value=runtime):
                result = main_module.asyncio.run(
                    main_module._run_demo(
                        prompt="teste",
                        interactive=False,
                        voice=False,
                        use_native_backends=False,
                    )
                )

        self.assertIsNone(result)
        runtime.respond_text.assert_awaited_once_with("teste")
        runtime.shutdown.assert_awaited_once()

    def test_run_demo_executes_interactive_mode_until_exit(self):
        runtime = AsyncMock()
        runtime.respond_text.return_value = types.SimpleNamespace(
            full_text="resposta interativa"
        )

        with patch("jarvis.main.load_config", return_value=object()):
            with patch("jarvis.main.JarvisRuntime.from_config", return_value=runtime):
                with patch("builtins.input", side_effect=["Oi", "sair"]):
                    with patch("builtins.print") as print_mock:
                        result = main_module.asyncio.run(
                            main_module._run_demo(
                                prompt=None,
                                interactive=True,
                                voice=False,
                                use_native_backends=False,
                            )
                        )

        self.assertIsNone(result)
        runtime.respond_text.assert_awaited_once_with("Oi")
        runtime.shutdown.assert_awaited_once()
        printed_lines = [call.args[0] for call in print_mock.call_args_list]
        self.assertIn(
            "J.A.R.V.I.S. em modo texto. Digite 'sair' para encerrar.", printed_lines
        )
        self.assertIn("jarvis> resposta interativa", printed_lines)

    def test_run_demo_executes_voice_mode(self):
        runtime = AsyncMock()

        with patch("jarvis.main.load_config", return_value=object()):
            with patch("jarvis.main.JarvisRuntime.from_config", return_value=runtime):
                with patch("builtins.print"):
                    result = main_module.asyncio.run(
                        main_module._run_demo(
                            prompt=None,
                            interactive=False,
                            voice=True,
                            use_native_backends=False,
                        )
                    )

        self.assertIsNone(result)
        runtime.run_voice_foreground.assert_awaited_once()
        runtime.shutdown.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
