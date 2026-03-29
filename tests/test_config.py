import unittest

from jarvis.config import JarvisConfig


class JarvisConfigTests(unittest.TestCase):
    def test_string_lists_are_parsed_from_csv(self):
        config = JarvisConfig(
            allowed_file_roots="/tmp,/var/tmp",
            system_allowed_apps="Safari, Notes ,Terminal",
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
