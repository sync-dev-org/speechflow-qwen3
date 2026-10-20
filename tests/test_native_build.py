from __future__ import annotations

import ctypes
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
from scripts.check_wheel import _read_glibc_versions
from scripts.vendor_source import load_sources

from speechflow.qwen3 import _binding, _payload

ROOT = Path(__file__).parents[1]
REQUIRED_TOOLS = ("cmake", "ninja", "gcc", "g++")
MISSING_TOOLS = [tool for tool in REQUIRED_TOOLS if shutil.which(tool) is None]
NVCC = Path("/usr/local/cuda/bin/nvcc")
CAN_BUILD = platform.system() == "Linux" and platform.machine() == "x86_64" and not MISSING_TOOLS and NVCC.is_file()


def _assert_cuda_13_3(cuda_toolkit_version: str) -> None:
    assert cuda_toolkit_version.startswith("13.3.")


@pytest.mark.req("SP-SFQWEN3-018")
def test_native_build_without_source_explains_how_to_fetch_it(tmp_path: Path) -> None:
    """A checkout without native source fails before compilation and explains the fetch command."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    builder = scripts / "build_native.sh"
    shutil.copy2(ROOT / "scripts/build_native.sh", builder)
    result = subprocess.run([str(builder)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "python3 scripts/fetch_vendor.py" in result.stderr
    assert "vendor/qwentts.cpp" in result.stderr
    assert not (tmp_path / "build").exists()


@pytest.mark.req("SP-SF-068")
def test_native_build_configures_reproducible_cpu_and_runtime_baselines() -> None:
    """The build script requests fixed CPU, local runtime, and CUDA architecture settings."""
    script = (ROOT / "scripts/build_native.sh").read_text()
    assert "-DBUILD_SHARED_LIBS=OFF" in script
    assert "-DCMAKE_POSITION_INDEPENDENT_CODE=ON" in script

    for option in (
        "-DGGML_NATIVE=OFF",
        "-DGGML_SSE42=ON",
        "-DGGML_AVX=ON",
        "-DGGML_AVX2=ON",
        "-DGGML_BMI2=ON",
        "-DGGML_FMA=ON",
        "-DGGML_F16C=ON",
        "-DGGML_AVX_VNNI=OFF",
        "-DGGML_AVX512=OFF",
        "-DGGML_AVX512_VBMI=OFF",
        "-DGGML_AVX512_VNNI=OFF",
        "-DGGML_AVX512_BF16=OFF",
        "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
        "-DCMAKE_INSTALL_RPATH=$ORIGIN",
        "-DCMAKE_CUDA_ARCHITECTURES=86-real;89-real;120a-real",
    ):
        assert option in script


@pytest.mark.req("SP-SFQWEN3-019")
@pytest.mark.parametrize(
    ("compiler", "language", "variable", "shared_flags"),
    [
        ("gcc", "c", "C", ("-fPIC",)),
        ("g++", "c++", "CXX", ("-fPIC",)),
        pytest.param(
            str(NVCC),
            "cu",
            "CUDA",
            ("-Xcompiler=-fPIC",),
            marks=pytest.mark.skipif(not NVCC.is_file(), reason="CUDA compiler required"),
        ),
    ],
)
@pytest.mark.skipif(
    platform.system() != "Linux" or platform.machine() != "x86_64" or bool(MISSING_TOOLS),
    reason="Linux compiler required",
)
def test_native_math_build_preserves_square_roots_without_new_glibc_symbols(
    tmp_path: Path, compiler: str, language: str, variable: str, shared_flags: tuple[str, ...]
) -> None:
    """Native compiler settings preserve square roots while requiring glibc 2.38 or older."""
    arguments = shlex.split((ROOT / "scripts/build_native.sh").read_text(), comments=True)
    prefix = f"-DCMAKE_{variable}_FLAGS="
    flags = shlex.split(
        next((argument.removeprefix(prefix) for argument in arguments if argument.startswith(prefix)), "")
    )
    source = tmp_path / "sqrt.c"
    source.write_text(
        "#include <math.h>\n"
        '#ifdef __cplusplus\nextern "C"\n#endif\n'
        '__attribute__((visibility("default"))) float square_root(float value) { return sqrtf(value); }\n'
    )
    library_path = tmp_path / "sqrt.so"
    subprocess.run(
        [
            compiler,
            "-x",
            language,
            "-shared",
            *shared_flags,
            "-O3",
            *flags,
            str(source),
            "-lm",
            "-o",
            str(library_path),
        ],
        check=True,
    )
    assert max(_read_glibc_versions(library_path), default=(0,)) <= (2, 38)
    library = ctypes.CDLL(str(library_path))
    library.square_root.argtypes = [ctypes.c_float]
    library.square_root.restype = ctypes.c_float
    assert library.square_root(9.0) == 3.0
    assert library.square_root(0.0) == 0.0
    assert math.copysign(1.0, library.square_root(-0.0)) == -1.0
    assert library.square_root(math.inf) == math.inf
    assert math.isnan(library.square_root(-1.0))


@pytest.mark.req("SP-SFQWEN3-019")
@pytest.mark.req("SP-SFQWEN3-020")
@pytest.mark.req("SP-SF-068")
@pytest.mark.native_build
@pytest.mark.skipif(
    not CAN_BUILD,
    reason=(
        "native build requires Linux x86_64, cmake/ninja/gcc/g++, and "
        "/usr/local/cuda/bin/nvcc; missing=" + ",".join(MISSING_TOOLS)
    ),
)
def test_native_build_places_payload_and_exposes_upstream_version(native_source: Path) -> None:
    """The built qwen library carries CUDA code and exports only the public C ABI.

    library が外に出す symbol は、自分の C ABI に限る。
    """
    subprocess.run(
        [ROOT / "scripts/build_native.sh"],
        cwd=ROOT,
        check=True,
        env=os.environ.copy(),
    )

    native_root = ROOT / "src/speechflow/qwen3/_native"
    assert {path.name for path in native_root.iterdir() if path.is_file()} == {
        *_payload.NATIVE_PAYLOAD_FILENAMES,
        "BUILD-METADATA.json",
    }
    assert all(not path.is_symlink() for path in native_root.iterdir())
    libqwen = native_root / "libqwen.so"
    needed = subprocess.run(["readelf", "-d", libqwen], check=True, capture_output=True, text=True).stdout
    assert "libggml" not in needed
    symbols = subprocess.run(["nm", "-D", "--defined-only", libqwen], check=True, capture_output=True, text=True).stdout
    assert symbols
    assert all(line.split()[-1].startswith("qt_") for line in symbols.splitlines())
    build_metadata = json.loads((native_root / "BUILD-METADATA.json").read_text())
    assert build_metadata["cpu_baseline"] == "x86-64-v3"
    assert build_metadata["cmake_cuda_architectures"] == "86-real;89-real;120a-real"
    _assert_cuda_13_3(build_metadata["cuda_toolkit_version"])

    compile_commands = json.loads((ROOT / "build/native/compile_commands.json").read_text())
    cpu_commands = [
        entry["command"] for entry in compile_commands if "ggml/src/CMakeFiles/ggml-cpu.dir/" in entry["command"]
    ]
    assert cpu_commands
    assert all("-march=native" not in command for command in cpu_commands)
    assert all(all(flag in command for flag in ("-mavx2", "-mfma", "-mf16c")) for command in cpu_commands)
    assert all("-mavx512" not in command for command in cpu_commands)

    codegen_flags = {
        flag
        for entry in compile_commands
        for flag in re.findall(r"--generate-code=arch=compute_[^ \"']+", entry["command"])
    }
    assert codegen_flags == {
        "--generate-code=arch=compute_86,code=[sm_86]",
        "--generate-code=arch=compute_89,code=[sm_89]",
        "--generate-code=arch=compute_120a,code=[sm_120a]",
    }

    previous = os.environ.get("LD_LIBRARY_PATH")
    try:
        os.environ["LD_LIBRARY_PATH"] = str(native_root)
        version = _binding.configure_library(_payload.load_native()).qt_version().decode("utf-8")
        assert version.startswith(f"{load_sources()['qwentts'].revision[:7]} (")
    finally:
        if previous is None:
            os.environ.pop("LD_LIBRARY_PATH", None)
        else:
            os.environ["LD_LIBRARY_PATH"] = previous
