import unittest
from typing import Any, cast

from jarvis.config import JarvisConfig


class JarvisConfigTests(unittest.TestCase):
    def test_defaults_follow_canonical_local_runtime(self):
        config = JarvisConfig()

        self.assertEqual(config.activation_backend, "push_to_talk")
        self.assertEqual(config.stt_backend, "speech_analyzer")
        self.assertEqual(config.llm_hot_path, "foundation_models")
        self.assertEqual(config.llm_deliberative, "mlx_lm")
        self.assertEqual(config.tts_backend, "mlx_audio_kokoro")
        self.assertEqual(config.tts_model, "mlx-community/Kokoro-82M-bf16")
        self.assertEqual(config.tts_voice, "pm_santa")
        self.assertEqual(config.tts_lang_code, "p")

    def test_string_lists_are_parsed_from_csv(self):
        config = JarvisConfig(
            allowed_file_roots=cast(Any, "/tmp,/var/tmp"),
            system_allowed_apps=cast(Any, "Safari, Notes ,Terminal"),
        )

        self.assertEqual(config.allowed_file_roots, ["/tmp", "/var/tmp"])
        self.assertEqual(config.system_allowed_apps, ["Safari", "Notes", "Terminal"])

    def test_prompt_override_takes_precedence(self):
        config = JarvisConfig(system_prompt_override="Use respostas curtas")
        self.assertEqual(config.system_prompt, "Use respostas curtas")

    def test_memory_limits_are_converted_to_bytes(self):
        config = JarvisConfig(
            metal_memory_limit_gb=1.5,
            metal_wired_limit_gb=2.0,
            metal_cache_limit_gb=0.25,
        )

        self.assertEqual(config.metal_memory_limit_bytes, int(1.5 * 1024**3))
        self.assertEqual(config.metal_wired_limit_bytes, int(2.0 * 1024**3))
        self.assertEqual(config.metal_cache_limit_bytes, int(0.25 * 1024**3))


if __name__ == "__main__":
    unittest.main()
