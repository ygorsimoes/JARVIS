from __future__ import annotations

import importlib
from dataclasses import dataclass

from ..config import JarvisConfig


@dataclass
class ResourceGovernorStatus:
    applied: bool
    backend: str
    limits: dict


class ResourceGovernor:
    def __init__(self, config: JarvisConfig) -> None:
        self.config = config

    def limits(self) -> dict:
        return {
            "memory_limit_bytes": self.config.metal_memory_limit_bytes,
            "wired_limit_bytes": self.config.metal_wired_limit_bytes,
            "cache_limit_bytes": self.config.metal_cache_limit_bytes,
            "max_kv_size": self.config.llm_max_kv_size,
        }

    def apply(self) -> ResourceGovernorStatus:
        try:
            mx = importlib.import_module("mlx.core")
        except ImportError:
            return ResourceGovernorStatus(applied=False, backend="unavailable", limits=self.limits())

        mx.set_memory_limit(self.config.metal_memory_limit_bytes)
        mx.set_wired_limit(self.config.metal_wired_limit_bytes)
        mx.set_cache_limit(self.config.metal_cache_limit_bytes)
        return ResourceGovernorStatus(applied=True, backend="mlx", limits=self.limits())
