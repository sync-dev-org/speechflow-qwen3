from __future__ import annotations

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - direct source import before a build
    __version__ = "0.1.0"

from ._engine import Qwen3CloneTTSEngine, Qwen3CustomVoiceTTSEngine, Qwen3ModelSource

__all__ = [
    "Qwen3CloneTTSEngine",
    "Qwen3CustomVoiceTTSEngine",
    "Qwen3ModelSource",
    "__version__",
]
