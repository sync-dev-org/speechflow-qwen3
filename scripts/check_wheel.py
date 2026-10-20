#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable, Iterable
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any

from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.vendor_source import load_sources

EXPECTED_DISTRIBUTION = "speechflow-qwen3"
EXPECTED_TAG = "py3-none-manylinux_2_38_x86_64"
EXPECTED_ARCHITECTURES = "86-real;89-real;120a-real"
EXPECTED_CUBIN_ORDER = ("sm_86", "sm_89", "sm_120a")
EXPECTED_CUBINS = set(EXPECTED_CUBIN_ORDER)
MINIMUM_GLIBC_VERSION = (2, 38)
NATIVE_LIBRARY_FILENAMES = ("libqwen.so",)
NATIVE_PAYLOAD_FILENAMES = (*NATIVE_LIBRARY_FILENAMES, "BUILD-METADATA.json")
LICENSE_FILENAMES = (
    "qwentts.cpp-LICENSE.txt",
    "ggml-LICENSE.txt",
    "cpp-httplib-LICENSE.txt",
    "yyjson-LICENSE.txt",
)
ALLOWED_EXTERNAL_DEPENDENCIES = {
    "libc.so.6",
    "libm.so.6",
    "libdl.so.2",
    "libpthread.so.0",
    "librt.so.1",
    "ld-linux-x86-64.so.2",
    "libstdc++.so.6",
    "libgcc_s.so.1",
    "libgomp.so.1",
    "libcuda.so.1",
    "libcudart.so.13",
    "libcublas.so.13",
    "libcublasLt.so.13",
}


class WheelCheckError(ValueError):
    """Raised when a wheel violates the speechflow-qwen3 archive contract."""


def _single(names: Iterable[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise WheelCheckError(f"wheel must contain exactly one {suffix}: {matches}")
    return matches[0]


def _verify_record(archive: zipfile.ZipFile, names: list[str], record_name: str) -> None:
    rows = list(csv.reader(archive.read(record_name).decode("utf-8").splitlines()))
    entries = {row[0]: row[1:] for row in rows}
    if set(entries) != set(names):
        raise WheelCheckError("RECORD does not enumerate the complete wheel archive")

    for name in names:
        hash_and_size = entries[name]
        if name == record_name:
            if hash_and_size != ["", ""]:
                raise WheelCheckError("RECORD must leave its own hash and size empty")
            continue
        if len(hash_and_size) != 2 or not hash_and_size[0].startswith("sha256="):
            raise WheelCheckError(f"RECORD lacks a SHA-256 entry for {name}")
        data = archive.read(name)
        expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        if hash_and_size != [f"sha256={expected}", str(len(data))]:
            raise WheelCheckError(f"RECORD hash or size mismatch for {name}")


def _check_metadata(archive: zipfile.ZipFile, names: list[str]) -> tuple[str, str]:
    metadata_name = _single(names, ".dist-info/METADATA")
    metadata = BytesParser(policy=policy.default).parsebytes(archive.read(metadata_name))
    distribution = str(metadata["Name"])
    version = str(metadata["Version"])
    if canonicalize_name(distribution) != EXPECTED_DISTRIBUTION:
        raise WheelCheckError(f"distribution identity mismatch: {distribution}")
    try:
        Version(version)
    except InvalidVersion as exc:
        raise WheelCheckError(f"distribution version is invalid: {version}") from exc
    python_specifiers = {part.strip() for part in str(metadata["Requires-Python"]).split(",")}
    if python_specifiers != {">=3.12", "<3.14"}:
        raise WheelCheckError(f"Python requirement mismatch: {metadata['Requires-Python']}")

    dependencies = [str(value) for value in metadata.get_all("Requires-Dist", [])]
    required_fragments = {
        "numpy": (">=1.26.4",),
        "nvidia-cuda-runtime": ("==13.3.29", "sys_platform", "platform_machine"),
        "nvidia-cublas": ("==13.6.0.2", "sys_platform", "platform_machine"),
    }
    for dependency, fragments in required_fragments.items():
        matches = [value for value in dependencies if value.lower().startswith(dependency)]
        if len(matches) != 1 or not all(fragment in matches[0] for fragment in fragments):
            raise WheelCheckError(f"dependency metadata mismatch for {dependency}: {matches}")
    core_matches = [value for value in dependencies if value.lower().startswith("speechflow")]
    if core_matches != ["speechflow>=0.19.2"]:
        raise WheelCheckError(f"dependency metadata mismatch for speechflow: {core_matches}")
    return distribution, version


def _check_identity(wheel: Path, names: list[str], distribution: str, version: str) -> None:
    try:
        filename_distribution, filename_version, build, _ = parse_wheel_filename(wheel.name)
    except ValueError as exc:
        raise WheelCheckError(f"wheel identity is invalid: filename={wheel.name}") from exc

    expected_distribution = canonicalize_name(distribution)
    expected_version = Version(version)
    if filename_distribution != expected_distribution or filename_version != expected_version or build:
        raise WheelCheckError(f"wheel identity mismatch: filename={wheel.name}, metadata={distribution} {version}")

    metadata_name = _single(names, ".dist-info/METADATA")
    dist_info_root = metadata_name.split("/", 1)[0]
    identity = dist_info_root.removesuffix(".dist-info")
    dist_info_distribution, separator, dist_info_version = identity.rpartition("-")
    try:
        normalized_dist_info_version = Version(dist_info_version)
    except InvalidVersion as exc:
        raise WheelCheckError(f"wheel identity is invalid: dist-info={dist_info_root}") from exc
    if (
        not separator
        or canonicalize_name(dist_info_distribution) != expected_distribution
        or normalized_dist_info_version != expected_version
    ):
        raise WheelCheckError(f"wheel identity mismatch: dist-info={dist_info_root}, metadata={distribution} {version}")


def _check_tag(archive: zipfile.ZipFile, wheel: Path, names: list[str]) -> str:
    wheel_name = _single(names, ".dist-info/WHEEL")
    headers = BytesParser(policy=policy.default).parsebytes(archive.read(wheel_name))
    tags = headers.get_all("Tag", [])
    if tags != [EXPECTED_TAG] or not wheel.name.endswith(f"-{EXPECTED_TAG}.whl"):
        raise WheelCheckError(f"platform tag must be {EXPECTED_TAG}: filename={wheel.name}, tags={tags}")
    if str(headers["Root-Is-Purelib"]).lower() != "false":
        raise WheelCheckError("platform wheel must declare Root-Is-Purelib: false")
    return EXPECTED_TAG


def _check_ownership(names: list[str]) -> None:
    dist_info_roots = {name.split("/", 1)[0] for name in names if ".dist-info/" in name}
    if len(dist_info_roots) != 1:
        raise WheelCheckError(f"expected one dist-info directory: {sorted(dist_info_roots)}")
    dist_info_root = next(iter(dist_info_roots))
    invalid = [
        name for name in names if not name.startswith("speechflow/qwen3/") and not name.startswith(f"{dist_info_root}/")
    ]
    if invalid:
        raise WheelCheckError(f"namespace ownership violation: {invalid}")
    if any(name.lower().endswith(".gguf") for name in names):
        raise WheelCheckError("namespace ownership violation: wheel must not contain GGUF weights")


def _check_payload_and_licenses(names: list[str]) -> None:
    payload_prefix = "speechflow/qwen3/_native/"
    payload = {
        name.removeprefix(payload_prefix)
        for name in names
        if name.startswith(payload_prefix) and not name.endswith("/")
    }
    if payload != set(NATIVE_PAYLOAD_FILENAMES):
        raise WheelCheckError(
            f"native payload mismatch: expected={sorted(NATIVE_PAYLOAD_FILENAMES)}, actual={sorted(payload)}"
        )

    license_prefix = "speechflow/qwen3/_licenses/"
    licenses = {
        name.removeprefix(license_prefix)
        for name in names
        if name.startswith(license_prefix) and not name.endswith("/")
    }
    if licenses != set(LICENSE_FILENAMES):
        raise WheelCheckError(
            f"license notice mismatch: expected={sorted(LICENSE_FILENAMES)}, actual={sorted(licenses)}"
        )


def _check_build_metadata(archive: zipfile.ZipFile) -> None:
    member = "speechflow/qwen3/_native/BUILD-METADATA.json"
    try:
        build_metadata = json.loads(archive.read(member))
    except (KeyError, json.JSONDecodeError) as exc:
        raise WheelCheckError(f"build metadata is missing or invalid: {member}") from exc

    sources = load_sources()
    expected = {
        "schema_version": 1,
        "build_type": "Release",
        "cpu_baseline": "x86-64-v3",
        "qwentts_revision": sources["qwentts"].revision,
        "ggml_revision": sources["ggml"].revision,
    }
    for key, value in expected.items():
        if build_metadata.get(key) != value:
            raise WheelCheckError(f"build metadata mismatch for {key}: {build_metadata.get(key)!r}")
    if not str(build_metadata.get("cuda_toolkit_version", "")).startswith("13.3."):
        raise WheelCheckError(
            f"build metadata mismatch for cuda_toolkit_version: {build_metadata.get('cuda_toolkit_version')!r}"
        )
    architectures = build_metadata.get("cmake_cuda_architectures")
    if architectures != EXPECTED_ARCHITECTURES:
        raise WheelCheckError(
            f"CUDA architecture mismatch: metadata={architectures!r}, expected={EXPECTED_ARCHITECTURES!r}"
        )


def _read_needed(path: Path) -> set[str]:
    try:
        result = subprocess.run(
            ["readelf", "-d", path],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise WheelCheckError(f"readelf failed for native payload {path}: {exc}") from exc
    return set(re.findall(r"\(NEEDED\).*Shared library: \[([^]]+)]", result.stdout))


def _parse_glibc_versions(output: str) -> set[tuple[int, ...]]:
    _, separator, version_needs = output.partition("Version needs section")
    if not separator:
        return set()
    names = re.findall(r"\bName:\s+GLIBC_([0-9]+(?:\.[0-9]+)+)\b", version_needs)
    return {tuple(int(component) for component in name.split(".")) for name in names}


def _read_glibc_versions(path: Path) -> set[tuple[int, ...]]:
    try:
        result = subprocess.run(
            ["readelf", "--version-info", path],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise WheelCheckError(f"readelf failed for native payload {path}: {exc}") from exc
    return _parse_glibc_versions(result.stdout)


def _read_runtime_paths(path: Path) -> tuple[str, ...]:
    try:
        result = subprocess.run(
            ["readelf", "-d", path],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise WheelCheckError(f"readelf failed for native payload {path}: {exc}") from exc
    fields = re.findall(r"\((?:RPATH|RUNPATH)\).*Library (?:rpath|runpath): \[([^]]*)]", result.stdout)
    return tuple(element for field in fields for element in field.split(":"))


def _read_exported_symbols(path: Path) -> set[str]:
    try:
        result = subprocess.run(
            ["nm", "-D", "--defined-only", path],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise WheelCheckError(f"nm failed for native payload {path}: {exc}") from exc
    return {line.split()[-1] for line in result.stdout.splitlines() if line.split()}


def _parse_cuobjdump_architectures(elf_output: str, ptx_output: str) -> tuple[set[str], bool]:
    cubins = set(re.findall(r"\bsm_[0-9]+a?\b", elf_output))
    ptx_present = bool(re.search(r"(?m)^PTX file\s+\d+:|\bcompute_[0-9]+a?\b", ptx_output))
    return cubins, ptx_present


def _read_cuda_architectures(path: Path) -> tuple[set[str], bool]:
    try:
        elf = subprocess.run(
            ["cuobjdump", "--list-elf", path],
            check=True,
            capture_output=True,
            text=True,
        )
        ptx = subprocess.run(
            ["cuobjdump", "--list-ptx", path],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise WheelCheckError(f"cuobjdump failed for CUDA payload {path.name}: {exc}") from exc
    return _parse_cuobjdump_architectures(
        elf.stdout + elf.stderr,
        ptx.stdout + ptx.stderr,
    )


def inspect_wheel(
    wheel: Path,
    *,
    needed_reader: Callable[[Path], set[str]] = _read_needed,
    runtime_paths_reader: Callable[[Path], tuple[str, ...]] = _read_runtime_paths,
    symbol_reader: Callable[[Path], set[str]] = _read_exported_symbols,
    cuda_architecture_reader: Callable[[Path], tuple[set[str], bool]] = _read_cuda_architectures,
    glibc_version_reader: Callable[[Path], set[tuple[int, ...]]] = _read_glibc_versions,
) -> dict[str, Any]:
    """Validate a wheel and return its artifact summary."""
    if not wheel.is_file():
        raise WheelCheckError(f"wheel does not exist: {wheel}")

    try:
        with zipfile.ZipFile(wheel) as archive:
            corrupt = archive.testzip()
            if corrupt is not None:
                raise WheelCheckError(f"wheel CRC check failed: {corrupt}")
            names = [info.filename for info in archive.infolist() if not info.is_dir()]
            tag = _check_tag(archive, wheel, names)
            distribution, version = _check_metadata(archive, names)
            _check_identity(wheel, names, distribution, version)
            _check_ownership(names)
            _check_payload_and_licenses(names)
            _check_build_metadata(archive)
            record_name = _single(names, ".dist-info/RECORD")
            _verify_record(archive, names, record_name)

            needed: set[str] = set()
            exported_symbols: set[str] = set()
            cuda_cubins: set[str] = set()
            cuda_ptx_present = False
            glibc_versions: set[tuple[int, ...]] = set()
            with tempfile.TemporaryDirectory(prefix="speechflow-qwen3-wheel-") as temporary:
                temporary_root = Path(temporary)
                for filename in NATIVE_LIBRARY_FILENAMES:
                    member = f"speechflow/qwen3/_native/{filename}"
                    destination = temporary_root / filename
                    destination.write_bytes(archive.read(member))
                    needed.update(needed_reader(destination))
                    runtime_paths = runtime_paths_reader(destination)
                    if not runtime_paths or any(path != "$ORIGIN" for path in runtime_paths):
                        raise WheelCheckError(
                            f"native payload RPATH/RUNPATH must contain only $ORIGIN: "
                            f"filename={filename}, paths={list(runtime_paths)}"
                        )
                    exported_symbols = symbol_reader(destination)
                    cuda_cubins, cuda_ptx_present = cuda_architecture_reader(destination)
                    glibc_versions.update(glibc_version_reader(destination))
    except zipfile.BadZipFile as exc:
        raise WheelCheckError(f"invalid wheel ZIP archive: {wheel}") from exc

    ggml_dependencies = sorted(name for name in needed if name.startswith("libggml"))
    if ggml_dependencies:
        raise WheelCheckError(f"qwen native payload must not need shared ggml libraries: {ggml_dependencies}")
    if not exported_symbols or any(not name.startswith("qt_") for name in exported_symbols):
        unexpected = sorted(name for name in exported_symbols if not name.startswith("qt_"))
        raise WheelCheckError(f"qwen native payload exports symbols outside qt_*: {unexpected}")

    external = needed - set(NATIVE_LIBRARY_FILENAMES)
    forbidden = external - ALLOWED_EXTERNAL_DEPENDENCIES
    if forbidden:
        raise WheelCheckError(f"forbidden external native dependencies: {sorted(forbidden)}")
    if cuda_cubins != EXPECTED_CUBINS or cuda_ptx_present:
        raise WheelCheckError(
            "CUDA architecture mismatch: "
            f"expected_cubins={sorted(EXPECTED_CUBINS)}, actual_cubins={sorted(cuda_cubins)}, "
            f"ptx_present={cuda_ptx_present}"
        )
    glibc_max_required = max(glibc_versions, default=None)
    if glibc_max_required is not None and glibc_max_required > MINIMUM_GLIBC_VERSION:
        required = ".".join(map(str, glibc_max_required))
        supported = ".".join(map(str, MINIMUM_GLIBC_VERSION))
        raise WheelCheckError(f"native payload requires GLIBC_{required} above supported glibc {supported}")

    return {
        "filename": wheel.name,
        "size": wheel.stat().st_size,
        "version": version,
        "tag": tag,
        "payload": sorted(NATIVE_PAYLOAD_FILENAMES),
        "external_dependencies": sorted(external),
        "cuda_cubins": list(EXPECTED_CUBIN_ORDER),
        "cuda_ptx_present": cuda_ptx_present,
        "glibc_max_required": (
            "GLIBC_" + ".".join(map(str, glibc_max_required)) if glibc_max_required is not None else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a speechflow-qwen3 platform wheel")
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args(argv)
    try:
        summary = inspect_wheel(args.wheel)
    except WheelCheckError as exc:
        print(f"wheel check failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
