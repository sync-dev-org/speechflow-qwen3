from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pytest
from scripts.check_wheel import (
    NATIVE_PAYLOAD_FILENAMES,
    WheelCheckError,
    _parse_cuobjdump_architectures,
    inspect_wheel,
)
from scripts.vendor_source import load_sources

DIST_INFO = "speechflow_qwen3-0.1.0.dist-info"
EXPECTED_ARCHITECTURES = "86-real;89-real;120a-real"
EXPECTED_CUBINS = {"sm_86", "sm_89", "sm_120a"}
CUDA_PLATFORM_MARKER = "sys_platform == 'linux' and platform_machine == 'x86_64'"
CUDA_MINIMUMS = {"nvidia-cuda-runtime": "13.2", "nvidia-cublas": "13.3"}
CUDA_REQUIREMENTS = tuple(f"{name}>={minimum},<14; {CUDA_PLATFORM_MARKER}" for name, minimum in CUDA_MINIMUMS.items())


def _record_entry(name: str, data: bytes) -> tuple[str, str, str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return name, f"sha256={digest}", str(len(data))


def _wheel(
    tmp_path: Path,
    *,
    tag: str = "py3-none-manylinux_2_38_x86_64",
    filename_version: str = "0.1.0",
    dist_info: str = DIST_INFO,
    metadata_name: str = "speechflow-qwen3",
    metadata_version: str = "0.1.0",
    core_requirement: str = "speechflow>=0.19.2",
    cuda_requirements: tuple[str, ...] = CUDA_REQUIREMENTS,
    extra_file: str | None = None,
    omit_payload: str | None = None,
    omit_license: bool = False,
    metadata_architectures: str = EXPECTED_ARCHITECTURES,
    metadata_cpu_baseline: str = "x86-64-v3",
    metadata_toolkit_version: str = "13.2.51",
) -> Path:
    files: dict[str, bytes] = {
        "speechflow/qwen3/__init__.py": b"",
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.4\n"
            f"Name: {metadata_name}\n"
            f"Version: {metadata_version}\n"
            "Requires-Python: >=3.12,<3.14\n"
            f"Requires-Dist: {core_requirement}\n"
            "Requires-Dist: numpy>=1.26.4\n"
            + "".join(f"Requires-Dist: {requirement}\n" for requirement in cuda_requirements)
            + "\n"
        ).encode(),
        f"{dist_info}/WHEEL": (f"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: false\nTag: {tag}\n\n").encode(),
    }
    for filename in NATIVE_PAYLOAD_FILENAMES:
        if filename != omit_payload:
            if filename == "BUILD-METADATA.json":
                files[f"speechflow/qwen3/_native/{filename}"] = json.dumps(
                    {
                        "build_type": "Release",
                        "cmake_cuda_architectures": metadata_architectures,
                        "cpu_baseline": metadata_cpu_baseline,
                        "cuda_toolkit_version": metadata_toolkit_version,
                        "ggml_revision": load_sources()["ggml"].revision,
                        "qwentts_revision": load_sources()["qwentts"].revision,
                        "schema_version": 1,
                    }
                ).encode()
            else:
                files[f"speechflow/qwen3/_native/{filename}"] = b"ELF"
    if not omit_license:
        files["speechflow/qwen3/_licenses/qwentts.cpp-LICENSE.txt"] = b"MIT"
        files["speechflow/qwen3/_licenses/ggml-LICENSE.txt"] = b"MIT"
        files["speechflow/qwen3/_licenses/cpp-httplib-LICENSE.txt"] = b"MIT"
        files["speechflow/qwen3/_licenses/yyjson-LICENSE.txt"] = b"MIT"
    if extra_file:
        files[extra_file] = b"owned elsewhere"

    rows = [_record_entry(name, data) for name, data in files.items()]
    rows.append((f"{dist_info}/RECORD", "", ""))
    record_buffer = io.StringIO(newline="")
    csv.writer(record_buffer).writerows(rows)
    files[f"{dist_info}/RECORD"] = record_buffer.getvalue().encode()

    wheel = tmp_path / f"speechflow_qwen3-{filename_version}-{tag}.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return wheel


@pytest.mark.req("SP-SFQWEN3-019")
@pytest.mark.req("SP-SFQWEN3-024")
@pytest.mark.parametrize("lower_first", [True, False])
def test_wheel_accepts_cuda_13_2_dependency_ranges(tmp_path: Path, lower_first: bool) -> None:
    """Wheel metadata accepts the CUDA 13.2 runtime and matching cuBLAS package ranges on Linux x86_64."""
    requirements = tuple(
        f"{name}{f'>={minimum},<14' if lower_first else f'<14,>={minimum}'}; {CUDA_PLATFORM_MARKER}"
        for name, minimum in CUDA_MINIMUMS.items()
    )
    wheel = _wheel(tmp_path, cuda_requirements=requirements)

    summary = inspect_wheel(
        wheel,
        needed_reader=lambda _: {"libcudart.so.13", "libcublas.so.13"},
        glibc_version_reader=lambda _: {(2, 38)},
        runtime_paths_reader=lambda _: ("$ORIGIN",),
        symbol_reader=lambda _: {"qt_version"},
        cuda_architecture_reader=lambda _: (EXPECTED_CUBINS, False),
    )

    assert summary["external_dependencies"] == ["libcublas.so.13", "libcudart.so.13"]


@pytest.mark.req("SP-SFQWEN3-019")
@pytest.mark.req("SP-SFQWEN3-024")
@pytest.mark.parametrize("dependency", ["nvidia-cuda-runtime", "nvidia-cublas"])
@pytest.mark.parametrize(
    "requirement_suffix",
    [
        f"==13.2.51; {CUDA_PLATFORM_MARKER}",
        ">={minimum}; " + CUDA_PLATFORM_MARKER,
        f">=13.0,<14; {CUDA_PLATFORM_MARKER}",
        ">={minimum},<14.1; " + CUDA_PLATFORM_MARKER,
        ">={minimum},<14",
        ">={minimum},<14; sys_platform == 'win32' and platform_machine == 'x86_64'",
        ">={minimum},<14; sys_platform == 'linux' and platform_machine == 'aarch64'",
        ">={minimum},<14; sys_platform == 'linux' or platform_machine == 'x86_64'",
    ],
)
def test_wheel_rejects_incorrect_cuda_dependency_ranges_or_platforms(
    tmp_path: Path, dependency: str, requirement_suffix: str
) -> None:
    """Wheel inspection rejects CUDA pins, wrong bounds, and dependencies outside Linux x86_64."""
    requirement_suffix = requirement_suffix.format(minimum=CUDA_MINIMUMS[dependency])
    requirements = tuple(
        f"{dependency}{requirement_suffix}" if requirement.startswith(dependency) else requirement
        for requirement in CUDA_REQUIREMENTS
    )
    wheel = _wheel(tmp_path, cuda_requirements=requirements)

    with pytest.raises(WheelCheckError, match=f"dependency metadata mismatch for {dependency}"):
        inspect_wheel(
            wheel,
            needed_reader=lambda _: set(),
            glibc_version_reader=lambda _: {(2, 38)},
            runtime_paths_reader=lambda _: ("$ORIGIN",),
            symbol_reader=lambda _: {"qt_version"},
            cuda_architecture_reader=lambda _: (EXPECTED_CUBINS, False),
        )


@pytest.mark.req("SP-SFQWEN3-019")
@pytest.mark.parametrize("toolkit_version", ["13.1.80", "13.3.73", "14.0.0"])
def test_wheel_rejects_a_build_from_another_cuda_toolkit(tmp_path: Path, toolkit_version: str) -> None:
    """The wheel must be built with CUDA 13.2 even when runtime dependency ranges are correct."""
    wheel = _wheel(tmp_path, metadata_toolkit_version=toolkit_version)
    with pytest.raises(WheelCheckError, match="cuda_toolkit_version"):
        inspect_wheel(
            wheel,
            needed_reader=lambda _: set(),
            glibc_version_reader=lambda _: {(2, 38)},
            runtime_paths_reader=lambda _: ("$ORIGIN",),
            symbol_reader=lambda _: {"qt_version"},
            cuda_architecture_reader=lambda _: (EXPECTED_CUBINS, False),
        )


@pytest.mark.req("SP-SFQWEN3-010")
def test_wheel_rejects_an_upper_bound_on_the_core_dependency(tmp_path: Path) -> None:
    """SP-SFQWEN3-010: Published module metadata has no upper bound on core."""
    wheel = _wheel(tmp_path, core_requirement="speechflow<0.99,>=0.19.2")

    with pytest.raises(WheelCheckError, match="dependency metadata mismatch for speechflow"):
        inspect_wheel(
            wheel,
            needed_reader=lambda _: set(),
            glibc_version_reader=lambda _: {(2, 38)},
            runtime_paths_reader=lambda _: ("$ORIGIN",),
            symbol_reader=lambda _: {"qt_version"},
            cuda_architecture_reader=lambda _: (EXPECTED_CUBINS, False),
        )


@pytest.mark.req("SP-SFQWEN3-010")
def test_wheel_rejects_a_core_dependency_below_the_supported_version(tmp_path: Path) -> None:
    """SP-SFQWEN3-010: Published module metadata requires the compatible core release."""
    wheel = _wheel(tmp_path, core_requirement="speechflow>=0.18.0")

    with pytest.raises(WheelCheckError, match="dependency metadata mismatch for speechflow"):
        inspect_wheel(
            wheel,
            needed_reader=lambda _: set(),
            glibc_version_reader=lambda _: {(2, 38)},
            runtime_paths_reader=lambda _: ("$ORIGIN",),
            symbol_reader=lambda _: {"qt_version"},
            cuda_architecture_reader=lambda _: (EXPECTED_CUBINS, False),
        )


@pytest.mark.req("SP-SFQWEN3-024")
@pytest.mark.req("SP-SFQWEN3-020")
@pytest.mark.req("SP-SFQWEN3-010")
def test_valid_wheel_reports_tag_payload_and_external_dependencies(tmp_path: Path) -> None:
    """Inspection reports a synthetic wheel's tag, payload, dependencies, and CUDA reader result."""
    assert NATIVE_PAYLOAD_FILENAMES == ("libqwen.so", "BUILD-METADATA.json")
    wheel = _wheel(tmp_path)
    inspected_cubin_paths: list[str] = []

    def read_cuda_architectures(path: Path) -> tuple[set[str], bool]:
        inspected_cubin_paths.append(path.name)
        return EXPECTED_CUBINS, False

    summary = inspect_wheel(
        wheel,
        needed_reader=lambda _: {"libc.so.6", "libcudart.so.13"},
        glibc_version_reader=lambda _: {(2, 38)},
        runtime_paths_reader=lambda _: ("$ORIGIN",),
        symbol_reader=lambda _: {"qt_version"},
        cuda_architecture_reader=read_cuda_architectures,
    )

    assert summary["tag"] == "py3-none-manylinux_2_38_x86_64"
    assert summary["payload"] == sorted(NATIVE_PAYLOAD_FILENAMES)
    assert summary["external_dependencies"] == ["libc.so.6", "libcudart.so.13"]
    assert summary["cuda_cubins"] == ["sm_86", "sm_89", "sm_120a"]
    assert summary["cuda_ptx_present"] is False
    assert summary["size"] == wheel.stat().st_size
    assert inspected_cubin_paths == ["libqwen.so"]


@pytest.mark.req("SP-SFQWEN3-034")
@pytest.mark.req("SP-SF-068")
@pytest.mark.parametrize("symbol", ["ggml_backend_init_by_name", "unrelated_export"])
def test_wheel_rejects_exports_outside_qwen_c_abi(tmp_path: Path, symbol: str) -> None:
    """Wheel inspection rejects a forbidden symbol supplied by its symbol reader.

    library が外に出す symbol は、自分の C ABI に限る。
    """
    wheel = _wheel(tmp_path)

    with pytest.raises(WheelCheckError, match=symbol):
        inspect_wheel(
            wheel,
            needed_reader=lambda _: set(),
            glibc_version_reader=lambda _: {(2, 38)},
            runtime_paths_reader=lambda _: ("$ORIGIN",),
            symbol_reader=lambda _: {"qt_version", symbol},
            cuda_architecture_reader=lambda _: (EXPECTED_CUBINS, False),
        )


@pytest.mark.req("SP-SFQWEN3-034")
@pytest.mark.req("SP-SF-068")
def test_wheel_rejects_a_ggml_dynamic_dependency(tmp_path: Path) -> None:
    """Wheel inspection rejects a shared ggml dependency supplied by its dependency reader."""
    wheel = _wheel(tmp_path)

    with pytest.raises(WheelCheckError, match=r"libggml-cuda\.so\.0"):
        inspect_wheel(
            wheel,
            needed_reader=lambda _: {"libggml-cuda.so.0"},
            glibc_version_reader=lambda _: {(2, 38)},
            runtime_paths_reader=lambda _: ("$ORIGIN",),
            symbol_reader=lambda _: {"qt_version"},
            cuda_architecture_reader=lambda _: (EXPECTED_CUBINS, False),
        )


@pytest.mark.parametrize(
    ("wheel_options", "message"),
    [
        ({"tag": "py3-none-any"}, "platform tag"),
        ({"tag": "py3-none-linux_x86_64"}, "platform tag"),
        ({"extra_file": "speechflow/other.py"}, "namespace ownership"),
        ({"extra_file": "speechflow/qwen3/_native/libggml-cpu.so.0"}, "native payload"),
        ({"omit_payload": "libqwen.so"}, "native payload"),
        ({"omit_license": True}, "license notice"),
    ],
)
@pytest.mark.req("SP-SFQWEN3-024")
def test_invalid_archive_contract_is_rejected(tmp_path: Path, wheel_options: dict[str, Any], message: str) -> None:
    wheel = _wheel(tmp_path, **wheel_options)

    with pytest.raises(WheelCheckError, match=message):
        inspect_wheel(
            wheel,
            needed_reader=lambda _: set(),
            glibc_version_reader=lambda _: {(2, 38)},
            runtime_paths_reader=lambda _: ("$ORIGIN",),
            symbol_reader=lambda _: {"qt_version"},
            cuda_architecture_reader=lambda _: (EXPECTED_CUBINS, False),
        )


def test_forbidden_external_native_dependency_is_rejected(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)

    with pytest.raises(WheelCheckError, match=r"libforbidden\.so"):
        inspect_wheel(
            wheel,
            needed_reader=lambda _: {"libforbidden.so"},
            glibc_version_reader=lambda _: {(2, 38)},
            runtime_paths_reader=lambda _: ("$ORIGIN",),
            symbol_reader=lambda _: {"qt_version"},
            cuda_architecture_reader=lambda _: (EXPECTED_CUBINS, False),
        )


@pytest.mark.req("SP-SFQWEN3-020")
def test_incorrect_build_architecture_metadata_is_rejected(tmp_path: Path) -> None:
    """Wheel inspection rejects CUDA architecture metadata outside the required three real targets."""
    wheel = _wheel(tmp_path, metadata_architectures="120a-real")

    with pytest.raises(WheelCheckError, match="architecture mismatch"):
        inspect_wheel(
            wheel,
            needed_reader=lambda _: set(),
            glibc_version_reader=lambda _: {(2, 38)},
            runtime_paths_reader=lambda _: ("$ORIGIN",),
            symbol_reader=lambda _: {"qt_version"},
            cuda_architecture_reader=lambda _: (EXPECTED_CUBINS, False),
        )


def test_incorrect_cpu_baseline_metadata_is_rejected(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, metadata_cpu_baseline="native")

    with pytest.raises(WheelCheckError, match="cpu_baseline"):
        inspect_wheel(
            wheel,
            needed_reader=lambda _: set(),
            glibc_version_reader=lambda _: {(2, 38)},
            runtime_paths_reader=lambda _: ("$ORIGIN",),
            symbol_reader=lambda _: {"qt_version"},
            cuda_architecture_reader=lambda _: (EXPECTED_CUBINS, False),
        )


@pytest.mark.parametrize(
    "runtime_paths",
    [
        ("/producer/build",),
        ("",),
        ("$ORIGIN", "relative/path"),
    ],
)
def test_non_origin_native_runtime_paths_are_rejected(tmp_path: Path, runtime_paths: tuple[str, ...]) -> None:
    wheel = _wheel(tmp_path)

    with pytest.raises(WheelCheckError, match=r"RPATH|RUNPATH"):
        inspect_wheel(
            wheel,
            needed_reader=lambda _: set(),
            glibc_version_reader=lambda _: {(2, 38)},
            runtime_paths_reader=lambda _: runtime_paths,
            symbol_reader=lambda _: {"qt_version"},
            cuda_architecture_reader=lambda _: (EXPECTED_CUBINS, False),
        )


@pytest.mark.parametrize(
    ("filename_version", "dist_info"),
    [
        ("9.9.9", DIST_INFO),
        ("0.1.0", "speechflow_qwen3-9.9.9.dist-info"),
    ],
)
def test_wheel_identity_components_must_match_metadata(tmp_path: Path, filename_version: str, dist_info: str) -> None:
    wheel = _wheel(tmp_path, filename_version=filename_version, dist_info=dist_info)

    with pytest.raises(WheelCheckError, match="identity"):
        inspect_wheel(
            wheel,
            needed_reader=lambda _: set(),
            glibc_version_reader=lambda _: {(2, 38)},
            runtime_paths_reader=lambda _: ("$ORIGIN",),
            symbol_reader=lambda _: {"qt_version"},
            cuda_architecture_reader=lambda _: (EXPECTED_CUBINS, False),
        )


def test_wheel_identity_comparison_normalizes_distribution_names(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, metadata_name="SpeechFlow.Qwen3")

    summary = inspect_wheel(
        wheel,
        needed_reader=lambda _: set(),
        glibc_version_reader=lambda _: {(2, 38)},
        runtime_paths_reader=lambda _: ("$ORIGIN",),
        symbol_reader=lambda _: {"qt_version"},
        cuda_architecture_reader=lambda _: (EXPECTED_CUBINS, False),
    )

    assert summary["version"] == "0.1.0"


def test_a_copied_wheel_with_a_different_filename_is_rejected(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)
    renamed = wheel.with_name("speechflow_qwen3-9.9.9-py3-none-manylinux_2_38_x86_64.whl")
    shutil.copyfile(wheel, renamed)

    with pytest.raises(WheelCheckError, match="identity"):
        inspect_wheel(
            renamed,
            needed_reader=lambda _: set(),
            glibc_version_reader=lambda _: {(2, 38)},
            runtime_paths_reader=lambda _: ("$ORIGIN",),
            symbol_reader=lambda _: {"qt_version"},
            cuda_architecture_reader=lambda _: (EXPECTED_CUBINS, False),
        )


@pytest.mark.req("SP-SFQWEN3-020")
@pytest.mark.parametrize(
    ("cubins", "ptx_present"),
    [
        ({"sm_120a"}, False),
        ({"sm_86", "sm_120a"}, False),
        ({"sm_86", "sm_89", "sm_120a", "sm_90"}, False),
        (EXPECTED_CUBINS, True),
    ],
)
def test_cuda_payload_architecture_must_match_exact_real_cubin_set(
    tmp_path: Path,
    cubins: set[str],
    ptx_present: bool,
) -> None:
    """Wheel inspection requires exactly sm_86, sm_89, and sm_120a cubins without PTX."""
    wheel = _wheel(tmp_path)

    with pytest.raises(WheelCheckError, match="architecture mismatch"):
        inspect_wheel(
            wheel,
            needed_reader=lambda _: set(),
            glibc_version_reader=lambda _: {(2, 38)},
            runtime_paths_reader=lambda _: ("$ORIGIN",),
            symbol_reader=lambda _: {"qt_version"},
            cuda_architecture_reader=lambda _: (cubins, ptx_present),
        )


def test_cuda_payload_architecture_comparison_is_order_independent(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)

    summary = inspect_wheel(
        wheel,
        needed_reader=lambda _: set(),
        glibc_version_reader=lambda _: {(2, 38)},
        runtime_paths_reader=lambda _: ("$ORIGIN",),
        symbol_reader=lambda _: {"qt_version"},
        cuda_architecture_reader=lambda _: ({"sm_120a", "sm_89", "sm_86"}, False),
    )

    assert summary["cuda_cubins"] == ["sm_86", "sm_89", "sm_120a"]


@pytest.mark.req("SP-SFQWEN3-019")
@pytest.mark.parametrize("required_version", [(2, 9), (2, 38)])
def test_wheel_accepts_glibc_symbol_versions_at_or_below_the_supported_minimum(
    tmp_path: Path, required_version: tuple[int, ...]
) -> None:
    """A wheel may require glibc symbols through the supported 2.38 boundary."""
    wheel = _wheel(tmp_path)
    summary = inspect_wheel(
        wheel,
        needed_reader=lambda _: set(),
        runtime_paths_reader=lambda _: ("$ORIGIN",),
        symbol_reader=lambda _: {"qt_version"},
        cuda_architecture_reader=lambda _: (EXPECTED_CUBINS, False),
        glibc_version_reader=lambda _: {required_version},
    )
    assert summary["glibc_max_required"] == "GLIBC_" + ".".join(map(str, required_version))


@pytest.mark.req("SP-SFQWEN3-019")
@pytest.mark.parametrize("required_version", [(2, 39), (2, 43), (2, 44)])
def test_wheel_rejects_glibc_symbol_version_above_the_supported_minimum(
    tmp_path: Path, required_version: tuple[int, ...]
) -> None:
    """A wheel requiring a newer glibc symbol fails before distribution."""
    wheel = _wheel(tmp_path)
    with pytest.raises(WheelCheckError, match=r"GLIBC_2\.\d+.*2\.38"):
        inspect_wheel(
            wheel,
            needed_reader=lambda _: set(),
            runtime_paths_reader=lambda _: ("$ORIGIN",),
            symbol_reader=lambda _: {"qt_version"},
            cuda_architecture_reader=lambda _: (EXPECTED_CUBINS, False),
            glibc_version_reader=lambda _: {(2, 9), required_version},
        )


@pytest.mark.req("SP-SFQWEN3-019")
def test_glibc_version_reader_uses_required_versions_only() -> None:
    """Wheel inspection uses required glibc symbol versions for its compatibility boundary."""
    from scripts import check_wheel

    output = """\
Version definition section '.gnu.version_d' contains 1 entry:
  0x0010: Name: GLIBC_9.99
Version needs section '.gnu.version_r' contains 2 entries:
  0x0010: Name: GLIBC_2.9  Flags: none  Version: 2
  0x0020: Name: GLIBC_2.43  Flags: none  Version: 3
  0x0030: Name: GLIBCXX_3.4.32  Flags: none  Version: 4
"""
    assert check_wheel._parse_glibc_versions(output) == {(2, 9), (2, 43)}


@pytest.mark.req("SP-SFQWEN3-020")
def test_cuobjdump_parser_distinguishes_no_ptx_diagnostic_from_a_ptx_entry() -> None:
    """Wheel inspection distinguishes a PTX entry from a no-PTX diagnostic."""
    elf_output = "\n".join(
        [
            "ELF file    1: libqwen.so.1.sm_120a.cubin",
            "ELF file    2: libqwen.so.2.sm_86.cubin",
            "ELF file    3: libqwen.so.3.sm_89.cubin",
        ]
    )

    assert _parse_cuobjdump_architectures(
        elf_output,
        "cuobjdump info    : No PTX file found to extract from 'libqwen.so'.",
    ) == ({"sm_86", "sm_89", "sm_120a"}, False)
    assert _parse_cuobjdump_architectures(
        elf_output,
        "PTX file    1: libqwen.so.1.compute_89.ptx",
    ) == ({"sm_86", "sm_89", "sm_120a"}, True)
