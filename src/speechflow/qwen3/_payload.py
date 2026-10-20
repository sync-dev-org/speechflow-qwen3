from __future__ import annotations

import ctypes
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from speechflow.core import EnvironmentUnavailableError

NATIVE_PAYLOAD_FILENAMES = ("libqwen.so",)
_CUDA_LIBRARIES = (
    (
        "nvidia-cuda-runtime",
        "nvidia/cu13/lib/libcudart.so.13",
    ),
    (
        "nvidia-cublas",
        "nvidia/cu13/lib/libcublasLt.so.13",
    ),
    (
        "nvidia-cublas",
        "nvidia/cu13/lib/libcublas.so.13",
    ),
)
_LOADED_HANDLES: list[Any] = []


def _environment_error(*, why: str, what: str, how: str, cause: Exception | None = None) -> EnvironmentUnavailableError:
    error = EnvironmentUnavailableError(f"why={why}; what={what}; how={how}")
    if cause is not None:
        error.__cause__ = cause
    return error


def _native_directory() -> Path:
    return Path(__file__).with_name("_native")


def _validate_platform() -> None:
    machine = platform.machine().lower()
    if sys.platform != "linux" or machine not in {"x86_64", "amd64"}:
        raise _environment_error(
            why="the qwentts.cpp native payload supports Linux x86_64 only",
            what=f"platform={sys.platform}, machine={platform.machine()}",
            how="use a Linux x86_64 host; editable imports remain available for GPU-free tests",
        )


def _distribution_library(distribution_name: str, relative_path: str) -> Path:
    try:
        distribution = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError as exc:
        raise _environment_error(
            why="a CUDA 13.3 user-space runtime dependency is not installed",
            what=f"distribution={distribution_name}, path={relative_path}",
            how="install speechflow-qwen3 with dependencies and consult README.md",
            cause=exc,
        ) from exc

    path = Path(str(distribution.locate_file(relative_path))).resolve()
    if not path.is_file():
        raise _environment_error(
            why="a CUDA 13.3 user-space runtime library cannot be resolved",
            what=f"distribution={distribution_name}, path={path}",
            how="reinstall speechflow-qwen3 with dependencies and consult README.md",
        )
    return path


def _load_library(path: Path, *, mode: int) -> Any:
    try:
        return ctypes.CDLL(str(path), mode=mode)
    except OSError as exc:
        raise _environment_error(
            why="a required native library could not be loaded",
            what=f"path={path}, error={exc}",
            how="verify the CUDA 13.3 driver/runtime boundary and consult README.md",
            cause=exc,
        ) from exc


def load_native(*, native_directory: Path | None = None) -> Any:
    """Load CUDA runtime libraries before the locally scoped qwentts.cpp library."""
    _validate_platform()
    native_root = native_directory if native_directory is not None else _native_directory()

    missing = [
        native_root / filename for filename in NATIVE_PAYLOAD_FILENAMES if not (native_root / filename).is_file()
    ]
    if missing:
        paths = ", ".join(str(path) for path in missing)
        raise _environment_error(
            why="the package-owned qwentts.cpp native payload is missing",
            what=f"missing paths={paths}",
            how="run scripts/build_native.sh for editable development; see README.md",
        )

    loaded_handles: list[Any] = []
    for distribution_name, relative_path in _CUDA_LIBRARIES:
        loaded_handles.append(
            _load_library(_distribution_library(distribution_name, relative_path), mode=ctypes.RTLD_GLOBAL)
        )

    handle = _load_library(native_root / "libqwen.so", mode=ctypes.RTLD_LOCAL)
    loaded_handles.append(handle)
    _LOADED_HANDLES.extend(loaded_handles)
    return handle
