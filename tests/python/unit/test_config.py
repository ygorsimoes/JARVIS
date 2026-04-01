from typing import Any, cast

from jarvis.config import JarvisConfig


class TestJarvisConfig:
    def test_defaults_follow_canonical_local_runtime(self):
        config = JarvisConfig()

        assert config.activation_backend == "push_to_talk"
        assert config.stt_backend == "speech_analyzer"
        assert config.llm_hot_path == "foundation_models"
        assert config.llm_deliberative == "mlx_lm"
        assert config.tts_backend == "mlx_audio_kokoro"
        assert config.tts_model == "mlx-community/Kokoro-82M-bf16"
        assert config.tts_voice == "pm_santa"
        assert config.tts_lang_code == "p"

    def test_string_lists_are_parsed_from_csv(self):
        config = JarvisConfig(
            allowed_file_roots=cast(Any, "/tmp,/var/tmp"),
            system_allowed_apps=cast(Any, "Safari, Notes ,Terminal"),
        )

        assert config.allowed_file_roots == ["/tmp", "/var/tmp"]
        assert config.system_allowed_apps == ["Safari", "Notes", "Terminal"]

    def test_prompt_override_takes_precedence(self):
        config = JarvisConfig(system_prompt_override="Use respostas curtas")
        assert config.system_prompt == "Use respostas curtas"

    def test_memory_limits_are_converted_to_bytes(self):
        config = JarvisConfig(
            metal_memory_limit_gb=1.5,
            metal_wired_limit_gb=2.0,
            metal_cache_limit_gb=0.25,
        )

        assert config.metal_memory_limit_bytes == int(1.5 * 1024**3)
        assert config.metal_wired_limit_bytes == int(2.0 * 1024**3)
        assert config.metal_cache_limit_bytes == int(0.25 * 1024**3)
