from __future__ import annotations

import ctypes
import gc
from importlib import metadata
from pathlib import Path

import pytest
from speechflow.core import EnvironmentUnavailableError

from speechflow.qwen3 import _payload


@pytest.mark.req("SP-SFQWEN3-018")
def test_missing_payload_is_a_recoverable_environment_error(tmp_path: Path) -> None:
    """A missing native payload raises EnvironmentUnavailableError with its location and build guidance."""
    with pytest.raises(EnvironmentUnavailableError) as caught:
        _payload.load_native(native_directory=tmp_path)

    message = str(caught.value)
    assert str(tmp_path / "libqwen.so") in message
    assert "scripts/build_native.sh" in message
    assert "README" in message


@pytest.mark.req("SP-SFQWEN3-019")
@pytest.mark.parametrize("missing", ["distribution", "library"])
def test_missing_cuda_dependency_reports_the_supported_runtime_range(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing: str
) -> None:
    """A missing CUDA dependency identifies the CUDA 13.2 runtime range and how to reinstall it."""
    (tmp_path / "libqwen.so").write_bytes(b"native")

    def distribution(name: str) -> metadata.PathDistribution:
        if missing == "distribution":
            raise metadata.PackageNotFoundError(name)
        return metadata.PathDistribution(tmp_path / "missing.dist-info")

    monkeypatch.setattr("speechflow.qwen3._payload.metadata.distribution", distribution)
    with pytest.raises(EnvironmentUnavailableError) as caught:
        _payload.load_native(native_directory=tmp_path)

    message = str(caught.value)
    assert "CUDA 13.2" in message
    assert ">=13.2,<14" in message
    assert "nvidia-cuda-runtime" in message
    assert "libcudart.so.13" in message
    assert "install" in message
    assert "README.md" in message


@pytest.mark.req("SP-SFQWEN3-019")
def test_unsupported_platform_is_an_environment_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Qwen3 rejects native loading on an unsupported platform."""
    monkeypatch.setattr("speechflow.qwen3._payload.sys.platform", "darwin")

    with pytest.raises(EnvironmentUnavailableError, match="Linux x86_64"):
        _payload.load_native(native_directory=tmp_path)


@pytest.mark.req("SP-SFQWEN3-034")
@pytest.mark.req("SP-SF-068")
def test_cuda_runtime_loads_before_local_qwen_library(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Native loading opens CUDA libraries globally before a locally scoped qwen library."""
    assert _payload.NATIVE_PAYLOAD_FILENAMES == ("libqwen.so",)
    for filename in _payload.NATIVE_PAYLOAD_FILENAMES:
        (tmp_path / filename).write_bytes(b"native")

    cuda_runtime = tmp_path / "site/nvidia/cu13/lib/libcudart.so.13"
    cublas = tmp_path / "site/nvidia/cu13/lib/libcublas.so.13"
    cublas_lt = tmp_path / "site/nvidia/cu13/lib/libcublasLt.so.13"
    for path in (cuda_runtime, cublas, cublas_lt):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"cuda")

    locations = {
        "nvidia-cuda-runtime": cuda_runtime,
        "nvidia-cublas": cublas,
    }

    class FakeDistribution:
        def __init__(self, name: str) -> None:
            self.name = name

        def locate_file(self, relative: str) -> Path:
            if "cudart" in relative:
                return locations[self.name]
            if "cublasLt" in relative:
                return cublas_lt
            return cublas

    monkeypatch.setattr("speechflow.qwen3._payload.metadata.distribution", FakeDistribution)
    loaded: list[tuple[Path, int]] = []
    finalized: list[str] = []

    class FakeHandle:
        def __init__(self, name: str) -> None:
            self.name = name

        def __del__(self) -> None:
            finalized.append(self.name)

    qwen_handle = FakeHandle("libqwen.so")

    def fake_cdll(path: str, *, mode: int) -> object:
        loaded.append((Path(path), mode))
        return qwen_handle if path.endswith("libqwen.so") else FakeHandle(Path(path).name)

    monkeypatch.setattr("speechflow.qwen3._payload.ctypes.CDLL", fake_cdll)

    assert _payload.load_native(native_directory=tmp_path) is qwen_handle
    assert [path.name for path, _ in loaded] == [
        "libcudart.so.13",
        "libcublasLt.so.13",
        "libcublas.so.13",
        "libqwen.so",
    ]
    assert [mode for _, mode in loaded] == [ctypes.RTLD_GLOBAL] * 3 + [ctypes.RTLD_LOCAL]
    gc.collect()
    assert finalized == []
