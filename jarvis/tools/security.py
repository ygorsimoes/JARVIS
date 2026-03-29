from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def normalize_roots(paths: Iterable[str]) -> List[Path]:
    return [Path(path).expanduser().resolve() for path in paths if str(path).strip()]


def ensure_path_within_roots(target: str, roots: Iterable[Path]) -> Path:
    candidate = Path(target).expanduser().resolve()
    for root in roots:
        try:
            candidate.relative_to(root)
            return candidate
        except ValueError:
            continue
    raise PermissionError("path %s is outside the allowed roots" % candidate)
