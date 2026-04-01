from __future__ import annotations

import os
from pathlib import Path


def resolve_cached_model_reference(model_ref: str) -> str:
    candidate = Path(model_ref).expanduser()
    if candidate.exists():
        return str(candidate)

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return model_ref

    try:
        return snapshot_download(model_ref, local_files_only=True)
    except Exception:
        return model_ref
