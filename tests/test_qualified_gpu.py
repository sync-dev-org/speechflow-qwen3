from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import time
import tomllib
import zipfile
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from urllib.parse import urlsplit

import numpy as np
import pytest
from scripts import qualified_gpu
from scripts.vendor_source import load_sources
from speechflow.core import AudioData, ConfigurationError

MODULE_VERSION = "0.5.0"
PUBLIC_REPOSITORY_URL = "https://github.com/sync-dev-org/speechflow-qwen3"
PUBLIC_DOCUMENTS_URL = f"{PUBLIC_REPOSITORY_URL}/blob/main/docs"


def _module_wheel_name(version: str = MODULE_VERSION) -> str:
    return f"speechflow_qwen3-{version}-py3-none-manylinux_2_38_x86_64.whl"


def _checks(profile: str, *, failing: str | None = None) -> dict[str, dict[str, str]]:
    required = qualified_gpu.PROFILE_CHECKS[profile]
    checks: dict[str, dict[str, str]] = {}
    failed = False
    for name in required:
        if name == failing:
            checks[name] = {"status": "fail", "detail": "injected failure"}
            failed = True
        elif failed:
            checks[name] = {"status": "not_run", "detail": "not reached after injected failure"}
        else:
            checks[name] = {"status": "pass", "detail": f"{name} passed"}
    return checks


def _gpu_memory() -> dict[str, object]:
    return {
        "baseline_mib": 3_000.0,
        "peak_total_mib": 4_250.0,
        "peak_delta_mib": 1_250.0,
        "poll_interval_seconds": 0.1,
        "sample_count": 12,
        "expected_sample_count": 10,
        "errors": [],
    }


def _full_measurements(*, base: bool = False) -> dict[str, object]:
    return {
        "cold_load_seconds": 1.25,
        "warm_get": {
            "wall_seconds": 0.5,
            "audio_seconds": 1.0,
            "wall_rtf": 0.5,
            "samples": 24_000,
            "trial_wall_seconds": [0.5, 0.49],
        },
        "streaming": {
            "ttfa_seconds": 0.1,
            "wall_seconds": 0.6,
            "audio_seconds": 1.0,
            "wall_rtf": 0.6,
            "samples": 24_000,
        },
        "peak_rss_bytes": 123_000_000,
        "gpu_memory": _gpu_memory(),
        "base_conditioning_seconds": [0.2, 0.19] if base else None,
    }


def _smoke_measurements() -> dict[str, object]:
    return {
        "warmup_seconds": 1.25,
        "stream_trials": [
            {
                "trial": trial,
                "status": "pass",
                "chunk_count": 2,
                "samples": 24_000,
                "audio_seconds": 1.0,
                "ttfa_seconds": 0.1,
                "wall_seconds": 0.6,
                "failure": None,
            }
            for trial in (1, 2, 3)
        ],
        "peak_rss_bytes": 123_000_000,
        "gpu_memory": _gpu_memory(),
    }


def _environment(*, cuda_architecture: str = "sm_120a") -> dict[str, object]:
    capability = qualified_gpu.ARCHITECTURE_TO_CAPABILITY[cuda_architecture]
    return {
        "hardware": "NVIDIA RTX PRO 4000 Blackwell",
        "os": "Linux x86_64; Ubuntu 24.04 / WSL2",
        "glibc": "2.43",
        "python_version": "3.12.10",
        "nvidia_driver": "616.92",
        "cuda_runtime": "13.2.51",
        "gpu_compute_capability": capability,
        "cuda_architecture": cuda_architecture,
        "physical_gpu_index": 0,
        "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
        "gpu_count": 1,
        "qwentts_revision": qualified_gpu.QUALIFIED_QWENTTS_REVISION,
        "co_resident_gpu_note": "",
    }


def _build_metadata() -> dict[str, object]:
    return {
        "schema_version": 1,
        "build_type": "Release",
        "cpu_baseline": "x86-64-v3",
        "qwentts_revision": qualified_gpu.QUALIFIED_QWENTTS_REVISION,
        "ggml_revision": load_sources()["ggml"].revision,
        "cuda_toolkit_version": "13.2.51",
        "cmake_cuda_architectures": "86-real;89-real;120a-real",
    }


def _artifact(version: str = MODULE_VERSION) -> dict[str, object]:
    return {
        "repository": "sync-dev-org/speechflow-qwen3",
        "release_tag": f"v{version}",
        "module_wheel": _module_wheel_name(version),
        "module_wheel_sha256": "b" * 64,
        "core_wheel": "speechflow-0.16.2-py3-none-any.whl",
        "core_wheel_sha256": "c" * 64,
        "module_import_path": "lib/python3.12/site-packages/speechflow/qwen3/__init__.py",
        "build_metadata": _build_metadata(),
    }


def _full_run_input(*, base: bool = False) -> dict[str, object]:
    return {
        "text": "Qualified synthesis.",
        "language": "en",
        "speaker": None if base else "Ryan",
        "reference": "qwen3-tts-0.6b-customvoice-Q4_K_M" if base else None,
        "reference_text_modes": ["present", "absent"] if base else [],
        "options": {"max_new_tokens": 64, "seed": 7},
        "warm_up": "model load completed before measured synthesis",
        "trial_count": 2,
        "concurrency": 1,
        "build_type": "Release",
    }


def _smoke_run_input() -> dict[str, object]:
    return {
        "text": qualified_gpu.ARCHITECTURE_SMOKE_TEXT,
        "language": "en",
        "speaker": "ryan",
        "options": {"seed": 7, "max_new_tokens": 64},
        "trial_count": 3,
        "concurrency": 1,
        "build_type": "Release",
    }


def _record(
    model_id: str = "qwen3-tts-0.6b-customvoice-Q4_K_M",
    *,
    profile: str = "full-pair",
    cuda_architecture: str = "sm_120a",
    failing: str | None = None,
    version: str = MODULE_VERSION,
) -> dict[str, object]:
    entry = qualified_gpu.CATALOG_BY_ID[model_id]
    smoke = profile == "architecture-smoke"
    return qualified_gpu.make_record(
        entry,
        profile=profile,
        checks=_checks(profile, failing=failing),
        measurements=_smoke_measurements() if smoke else _full_measurements(base=entry.mode == "Base"),
        environment=_environment(cuda_architecture=cuda_architecture),
        artifact=_artifact(version),
        run_input=_smoke_run_input() if smoke else _full_run_input(base=entry.mode == "Base"),
        executed_at="2026-09-09T12:00:00+09:00",
        failure=(
            {"check": failing, "type": "InjectedFailure", "message": "failure kept in the record"}
            if failing is not None
            else None
        ),
    )


@pytest.mark.req("SP-SFQWEN3-017")
def test_catalog_is_the_required_eight_pair_cross_product() -> None:
    """The qualification catalog identifies all eight model and quantization pairs."""
    assert len(qualified_gpu.CATALOG) == 8
    assert {(entry.mode, entry.size, entry.quantization) for entry in qualified_gpu.CATALOG} == {
        (mode, size, quantization)
        for mode in ("CustomVoice", "Base")
        for size in ("0.6B", "1.7B")
        for quantization in ("Q4_K_M", "Q8_0")
    }


@pytest.mark.req("SP-SFQWEN3-017")
@pytest.mark.parametrize("profile", ["full-pair", "architecture-smoke"])
def test_schema_v3_accepts_the_exact_profile_shape(profile: str) -> None:
    """Qualification records retain their profile evidence and identify wheels without a commit."""
    record = _record(profile=profile)
    assert record["schema_version"] == 3
    assert "tag_commit" not in cast(Mapping[str, object], record["artifact"])
    qualified_gpu.validate_record(record)
    assert record["profile"] == profile
    evidence = cast(Mapping[str, object], record["evidence"])
    assert set(evidence) == {"run_input", "checks", "measurements"}
    assert set(cast(Mapping[str, object], evidence["checks"])) == set(qualified_gpu.PROFILE_CHECKS[profile])
    assert (
        set(cast(Mapping[str, object], evidence["measurements"])) == qualified_gpu.PROFILE_MEASUREMENT_FIELDS[profile]
    )


@pytest.mark.req("SP-SFQWEN3-017")
def test_record_rejects_schema_version_2() -> None:
    """New qualification evidence rejects the earlier schema even with its original artifact fields."""
    record = _record()
    record["schema_version"] = 2
    cast(dict[str, object], record["artifact"])["tag_commit"] = "a" * 40
    with pytest.raises(qualified_gpu.RecordValidationError, match="schema_version must be 3"):
        qualified_gpu.validate_record(record)


@pytest.mark.req("SP-SFQWEN3-017")
def test_record_rejects_tag_commit_in_schema_version_3() -> None:
    """Qualification evidence rejects a commit field instead of silently accepting an ambiguous source."""
    record = _record()
    record["schema_version"] = 3
    cast(dict[str, object], record["artifact"])["tag_commit"] = "a" * 40
    with pytest.raises(qualified_gpu.RecordValidationError, match=r"artifact fields mismatch:.*tag_commit"):
        qualified_gpu.validate_record(record)


@pytest.mark.req("SP-SFQWEN3-019")
@pytest.mark.parametrize("runtime", ["13.2.51", "13.4.0", "13.9.1"])
def test_qualified_record_accepts_cuda_runtimes_from_13_2(runtime: str) -> None:
    """Qualified evidence accepts CUDA 13 user-space libraries starting at 13.2."""
    record = _record()
    cast(dict[str, object], record["environment"])["cuda_runtime"] = runtime
    qualified_gpu.validate_record(record)
    assert record["status"] == "pass"


@pytest.mark.req("SP-SFQWEN3-019")
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cuda_runtime", "13.1.80", "cuda_runtime"),
        ("cuda_runtime", "14.0.0", "cuda_runtime"),
        ("nvidia_driver", "575.51.03", "nvidia_driver"),
    ],
)
def test_cuda_13_qualification_rejects_an_unsupported_runtime_or_driver(field: str, value: str, message: str) -> None:
    """Qualified evidence rejects runtimes outside >=13.2,<14 and drivers below the CUDA 13.2 minimum."""
    record = _record()
    cast(dict[str, object], record["environment"])[field] = value
    with pytest.raises(qualified_gpu.RecordValidationError, match=message):
        qualified_gpu.validate_record(record)


@pytest.mark.req("SP-SFQWEN3-019")
@pytest.mark.parametrize(
    ("os_name", "minimum", "accepted", "rejected"),
    [
        pytest.param(
            "Linux x86_64; Ubuntu 24.04",
            "595.45.04",
            ("595.45.04", "610.43.02"),
            ("595.45.03", "580.65.06"),
            id="linux",
        ),
        pytest.param(
            "Linux x86_64; Ubuntu 24.04 / WSL2",
            "595",
            ("595.0", "595.45", "616.92"),
            ("594.99", "580.65.06"),
            id="wsl2",
        ),
    ],
)
def test_cuda_13_2_qualification_requires_the_driver_floor_for_its_os(
    os_name: str, minimum: str, accepted: tuple[str, ...], rejected: tuple[str, ...]
) -> None:
    """Qualified evidence requires a CUDA 13.2 compatible driver for its host OS."""
    record = _record()
    environment = cast(dict[str, object], record["environment"])
    environment["os"] = os_name

    for driver in accepted:
        environment["nvidia_driver"] = driver
        qualified_gpu.validate_record(record)

    for driver in rejected:
        environment["nvidia_driver"] = driver
        with pytest.raises(qualified_gpu.RecordValidationError) as exc_info:
            qualified_gpu.validate_record(record)
        message = str(exc_info.value)
        assert "environment.nvidia_driver" in message
        assert f">={minimum}" in message
        assert repr(driver) in message


@pytest.mark.req("SP-SFQWEN3-019")
def test_qualified_record_requires_glibc_2_38_or_newer() -> None:
    """SP-SFQWEN3-019: A qualified record needs the minimum supported glibc version."""
    record = _record()
    cast(dict[str, object], record["environment"])["glibc"] = "2.37"
    with pytest.raises(qualified_gpu.RecordValidationError, match=r"environment\.glibc must be >=2\.38"):
        qualified_gpu.validate_record(record)

    cast(dict[str, object], record["environment"])["glibc"] = "2.38"
    qualified_gpu.validate_record(record)


@pytest.mark.req("SP-SFQWEN3-019")
@pytest.mark.parametrize("toolkit_version", ["12.9.86", "13.1.80", "13.3.73", "14.0.0"])
def test_cuda_13_2_qualification_rejects_other_toolkit_build_metadata(toolkit_version: str) -> None:
    """Qualified evidence requires a native build from the CUDA 13.2 toolkit series."""
    record = _record()
    artifact = cast(dict[str, object], record["artifact"])
    cast(dict[str, object], artifact["build_metadata"])["cuda_toolkit_version"] = toolkit_version
    with pytest.raises(qualified_gpu.RecordValidationError, match="cuda_toolkit_version"):
        qualified_gpu.validate_record(record)


@pytest.mark.parametrize(
    ("profile", "section", "field"),
    [
        ("full-pair", "checks", "stream_complete"),
        ("full-pair", "measurements", "cold_load_seconds"),
        ("architecture-smoke", "checks", "selected_gpu"),
        ("architecture-smoke", "measurements", "stream_trials"),
    ],
)
def test_profile_shape_rejects_missing_and_extra_fields(profile: str, section: str, field: str) -> None:
    record = _record(profile=profile)
    evidence = cast(dict[str, Any], record["evidence"])
    changed = cast(dict[str, Any], evidence[section])
    changed.pop(field)
    changed["foreign_profile_field"] = None
    with pytest.raises(qualified_gpu.RecordValidationError, match=f"evidence.{section} fields mismatch"):
        qualified_gpu.validate_record(record)


def test_profile_payloads_cannot_be_mixed() -> None:
    record = _record(profile="architecture-smoke")
    evidence = cast(dict[str, Any], record["evidence"])
    evidence["checks"] = _checks("full-pair")
    evidence["measurements"] = _full_measurements()
    with pytest.raises(qualified_gpu.RecordValidationError, match=r"evidence\.checks fields mismatch"):
        qualified_gpu.validate_record(record)


@pytest.mark.req("SP-SFQWEN3-017")
@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("gpu_compute_capability", "8.9", "architecture.*compute capability"),
        ("cuda_architecture", "sm_90", "cuda_architecture"),
        ("module_wheel_sha256", "short", "module_wheel_sha256"),
        ("core_wheel_sha256", "", "core_wheel_sha256"),
    ],
)
def test_record_rejects_architecture_and_digest_mismatches(field: str, value: str, match: str) -> None:
    """Qualified evidence rejects architecture and wheel digest mismatches."""
    record = _record()
    if field.startswith(("module_", "core_")):
        cast(dict[str, object], record["artifact"])[field] = value
    else:
        cast(dict[str, object], record["environment"])[field] = value
    with pytest.raises(qualified_gpu.RecordValidationError, match=match):
        qualified_gpu.validate_record(record)


@pytest.mark.req("SP-SFQWEN3-017")
@pytest.mark.parametrize("version", [MODULE_VERSION, "1.2.3"])
def test_record_accepts_its_own_matching_release_tag_and_wheel_version(version: str) -> None:
    """Qualification records bind to their own release rather than a fixed module version."""
    record = _record(version=version)
    qualified_gpu.validate_record(record)
    artifact = cast(Mapping[str, object], record["artifact"])
    assert artifact["release_tag"] == f"v{version}"
    assert artifact["module_wheel"] == _module_wheel_name(version)


@pytest.mark.req("SP-SFQWEN3-024")
@pytest.mark.parametrize(
    "wheel_name",
    [
        f"other_module-{MODULE_VERSION}-py3-none-manylinux_2_38_x86_64.whl",
        f"speechflow_qwen3-{MODULE_VERSION}-cp312-none-linux_x86_64.whl",
        f"speechflow_qwen3-{MODULE_VERSION}-py3-cp312-linux_x86_64.whl",
        f"speechflow_qwen3-{MODULE_VERSION}-py3-none-any.whl",
        f"speechflow_qwen3-{MODULE_VERSION}-1-py3-none-manylinux_2_38_x86_64.whl",
        "module.whl",
    ],
)
def test_record_preserves_the_module_wheel_distribution_and_platform_contract(wheel_name: str) -> None:
    """Matching versions still require the existing module distribution and wheel tags."""
    record = _record()
    cast(dict[str, object], record["artifact"])["module_wheel"] = wheel_name
    with pytest.raises(qualified_gpu.RecordValidationError, match=r"module[ _]wheel"):
        qualified_gpu.validate_record(record)


@pytest.mark.req("SP-SFQWEN3-037")
@pytest.mark.req("SP-SF-028")
def test_record_requires_the_version_tag_prefix() -> None:
    """The release tag retains its v prefix when its version matches the wheel."""
    record = _record()
    cast(dict[str, object], record["artifact"])["release_tag"] = MODULE_VERSION
    with pytest.raises(qualified_gpu.RecordValidationError, match="release_tag"):
        qualified_gpu.validate_record(record)


@pytest.mark.req("SP-SFQWEN3-037")
@pytest.mark.req("SP-SF-028")
@pytest.mark.parametrize("version", ["", "garbage", "0.5", "0.5.0rc1", "0.5.0.dev1"])
def test_record_rejects_matching_versions_outside_the_release_tag_format(version: str) -> None:
    """Record release tags use vX.Y.Z even when a malformed version matches the wheel."""
    record = _record(version=version)
    with pytest.raises(qualified_gpu.RecordValidationError, match=r"release_tag.*vX.Y.Z"):
        qualified_gpu.validate_record(record)


@pytest.mark.req("SP-SFQWEN3-017")
def test_release_binding_rejects_wrong_wheel_version_python_and_full_pair_architecture() -> None:
    """Qualified records reject version, Python, and full-pair architecture mismatches."""
    record = _record()
    cast(dict[str, object], record["artifact"])["module_wheel"] = _module_wheel_name("0.5.1")
    with pytest.raises(qualified_gpu.RecordValidationError, match="release_tag"):
        qualified_gpu.validate_record(record)

    record = _record()
    cast(dict[str, object], record["environment"])["python_version"] = "3.13.1"
    with pytest.raises(qualified_gpu.RecordValidationError, match="python_version"):
        qualified_gpu.validate_record(record)

    record = _record(cuda_architecture="sm_89")
    with pytest.raises(qualified_gpu.RecordValidationError, match=r"full-pair.*sm_120a"):
        qualified_gpu.validate_record(record)


@pytest.mark.req("SP-SFQWEN3-017")
def test_status_and_failure_are_derived_without_rounding_failure_to_success() -> None:
    """A failed qualification result cannot be reported as a passing record."""
    record = _record(profile="architecture-smoke", cuda_architecture="sm_86", failing="warmup")
    qualified_gpu.validate_record(record)
    assert record["status"] == "fail"
    assert record["failure"] == {
        "check": "warmup",
        "type": "InjectedFailure",
        "message": "failure kept in the record",
    }
    evidence = cast(Mapping[str, object], record["evidence"])
    checks = cast(Mapping[str, Mapping[str, str]], evidence["checks"])
    assert checks["stream_trials"]["status"] == "not_run"
    record["status"] = "pass"
    with pytest.raises(qualified_gpu.RecordValidationError, match="derived"):
        qualified_gpu.validate_record(record)


@pytest.mark.req("SP-SFQWEN3-017")
def test_passing_architecture_smoke_rejects_a_failed_stream_trial() -> None:
    """A failed stream trial cannot create a passing architecture record."""
    record = _record(profile="architecture-smoke", cuda_architecture="sm_86")
    evidence = cast(dict[str, object], record["evidence"])
    measurements = cast(dict[str, object], evidence["measurements"])
    trials = cast(list[dict[str, object]], measurements["stream_trials"])
    trials[1].update(
        {
            "status": "fail",
            "chunk_count": 0,
            "samples": 0,
            "audio_seconds": None,
            "failure": {"type": "InjectedFailure", "message": "trial failed"},
        }
    )

    with pytest.raises(qualified_gpu.RecordValidationError, match="passing architecture-smoke"):
        qualified_gpu.validate_record(record)


def test_record_rejects_machine_local_path_and_non_portable_json() -> None:
    record = _record()
    cast(dict[str, object], record["artifact"])["module_import_path"] = (
        "/home/producer/venv/site-packages/speechflow/qwen3/__init__.py"
    )
    with pytest.raises(qualified_gpu.RecordValidationError, match=r"machine-local|venv-relative"):
        qualified_gpu.validate_record(record)

    record = _record()
    measurements = cast(dict[str, object], cast(dict[str, object], record["evidence"])["measurements"])
    measurements["cold_load_seconds"] = float("nan")
    with pytest.raises(qualified_gpu.RecordValidationError, match=r"finite|portable"):
        qualified_gpu.validate_record(record)


def test_architecture_smoke_is_fixed_to_one_model_and_run_input() -> None:
    wrong_model = _record("qwen3-tts-1.7b-customvoice-Q4_K_M", profile="architecture-smoke", cuda_architecture="sm_86")
    with pytest.raises(qualified_gpu.RecordValidationError, match="architecture-smoke model"):
        qualified_gpu.validate_record(wrong_model)

    wrong_input = _record(profile="architecture-smoke", cuda_architecture="sm_86")
    run_input = cast(dict[str, object], cast(dict[str, object], wrong_input["evidence"])["run_input"])
    run_input["trial_count"] = 2
    with pytest.raises(qualified_gpu.RecordValidationError, match="trial_count"):
        qualified_gpu.validate_record(wrong_input)


def _direct_url(wheel: Path, digest: str) -> dict[str, object]:
    return {
        "url": wheel.as_uri(),
        "archive_info": {
            "hash": f"sha256={digest}",
            "hashes": {"sha256": digest},
        },
    }


@pytest.mark.parametrize(
    ("distribution_name", "wheel_name", "version", "import_suffix"),
    [
        (
            "speechflow-qwen3",
            _module_wheel_name(),
            MODULE_VERSION,
            Path("speechflow/qwen3/__init__.py"),
        ),
        (
            "speechflow",
            "speechflow-0.16.2-py3-none-any.whl",
            "0.16.2",
            Path("speechflow/__init__.py"),
        ),
    ],
)
def test_installed_wheel_identity_is_bound_to_supplied_archive_bytes(
    tmp_path: Path,
    distribution_name: str,
    wheel_name: str,
    version: str,
    import_suffix: Path,
) -> None:
    prefix = tmp_path / "runtime"
    origin = prefix / "lib/python3.12/site-packages" / import_suffix
    wheel = tmp_path / wheel_name
    wheel.write_bytes(b"release wheel bytes")
    digest = qualified_gpu._sha256_file(wheel)
    identity = qualified_gpu._validate_installed_wheel_identity(
        origin=origin,
        prefix=prefix,
        distribution_name=distribution_name,
        installed_distribution_name=distribution_name,
        distribution_version=version,
        wheel=wheel,
        wheel_sha256=digest,
        direct_url=_direct_url(wheel, digest),
    )
    assert identity.relative_origin == Path("lib/python3.12/site-packages") / import_suffix
    assert identity.module_wheel == wheel_name

    other_wheel = tmp_path / "other" / f"different-{wheel_name}"
    other_wheel.parent.mkdir()
    other_wheel.write_bytes(b"different same-version wheel bytes")
    with pytest.raises(qualified_gpu._GateFailure, match="editable"):
        qualified_gpu._validate_installed_wheel_identity(
            origin=origin,
            prefix=prefix,
            distribution_name=distribution_name,
            installed_distribution_name=distribution_name,
            distribution_version=version,
            wheel=wheel,
            wheel_sha256=digest,
            direct_url={**_direct_url(wheel, digest), "dir_info": {"editable": True}},
        )
    with pytest.raises(qualified_gpu._GateFailure, match=r"source|filename|URL"):
        qualified_gpu._validate_installed_wheel_identity(
            origin=origin,
            prefix=prefix,
            distribution_name=distribution_name,
            installed_distribution_name=distribution_name,
            distribution_version=version,
            wheel=wheel,
            wheel_sha256=digest,
            direct_url=_direct_url(other_wheel, digest),
        )
    with pytest.raises(qualified_gpu._GateFailure, match=r"digest|SHA-256|hash"):
        qualified_gpu._validate_installed_wheel_identity(
            origin=origin,
            prefix=prefix,
            distribution_name=distribution_name,
            installed_distribution_name=distribution_name,
            distribution_version=version,
            wheel=wheel,
            wheel_sha256=digest,
            direct_url=_direct_url(wheel, "f" * 64),
        )
    with pytest.raises(qualified_gpu._GateFailure, match="distribution name"):
        qualified_gpu._validate_installed_wheel_identity(
            origin=origin,
            prefix=prefix,
            distribution_name=distribution_name,
            installed_distribution_name="different-distribution",
            distribution_version=version,
            wheel=wheel,
            wheel_sha256=digest,
            direct_url=_direct_url(wheel, digest),
        )
    with pytest.raises(qualified_gpu._GateFailure, match="provenance"):
        qualified_gpu._validate_installed_wheel_identity(
            origin=origin,
            prefix=prefix,
            distribution_name=distribution_name,
            installed_distribution_name=distribution_name,
            distribution_version=version,
            wheel=wheel,
            wheel_sha256=digest,
            direct_url=None,
        )


def test_uv_file_provenance_without_archive_hash_requires_exact_installed_payload(tmp_path: Path) -> None:
    prefix = tmp_path / "runtime"
    site_packages = prefix / "lib/python3.12/site-packages"
    origin = site_packages / "speechflow/qwen3/__init__.py"
    wheel = tmp_path / _module_wheel_name()
    dist_info = f"speechflow_qwen3-{MODULE_VERSION}.dist-info"
    archive_files = {
        "speechflow/qwen3/__init__.py": b"module package",
        f"{dist_info}/METADATA": f"Name: speechflow-qwen3\nVersion: {MODULE_VERSION}\n".encode(),
        f"{dist_info}/RECORD": b"archive record",
    }
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in archive_files.items():
            archive.writestr(name, content)
    for name, content in archive_files.items():
        installed = site_packages / name
        installed.parent.mkdir(parents=True, exist_ok=True)
        installed.write_bytes(b"installed record" if name.endswith("/RECORD") else content)
    installed_files = [*archive_files, f"{dist_info}/direct_url.json"]
    direct_url = {"url": wheel.as_uri(), "archive_info": {}}
    digest = qualified_gpu._sha256_file(wheel)

    qualified_gpu._validate_installed_wheel_identity(
        origin=origin,
        prefix=prefix,
        distribution_name="speechflow-qwen3",
        installed_distribution_name="speechflow-qwen3",
        distribution_version=MODULE_VERSION,
        wheel=wheel,
        wheel_sha256=digest,
        direct_url=direct_url,
        installed_files=installed_files,
    )

    origin.write_bytes(b"different installed module bytes")
    with pytest.raises(qualified_gpu._GateFailure, match=r"installed payload|content"):
        qualified_gpu._validate_installed_wheel_identity(
            origin=origin,
            prefix=prefix,
            distribution_name="speechflow-qwen3",
            installed_distribution_name="speechflow-qwen3",
            distribution_version=MODULE_VERSION,
            wheel=wheel,
            wheel_sha256=digest,
            direct_url=direct_url,
            installed_files=installed_files,
        )


@pytest.mark.parametrize(
    "machine_path",
    [
        "/home/producer/private/model.gguf",
        "/var/lib/private-host/model.gguf",
        "/mnt/c/private/model.gguf",
        "/Users/producer/private/model.gguf",
        r"C:\Users\producer\private\model.gguf",
        r"\\private-server\share\model.gguf",
        "file:///var/lib/private-host/model.gguf",
        "//private-server/share/model.gguf",
    ],
)
def test_record_rejects_machine_local_absolute_paths_in_every_string(machine_path: str) -> None:
    record = _record(profile="architecture-smoke", cuda_architecture="sm_86")
    cast(dict[str, object], record["environment"])["co_resident_gpu_note"] = f"observed:{machine_path}"
    with pytest.raises(qualified_gpu.RecordValidationError, match=r"machine-local absolute path"):
        qualified_gpu.validate_record(record)


def test_record_rejects_nested_trial_failure_absolute_path() -> None:
    record = _record(profile="architecture-smoke", cuda_architecture="sm_86", failing="stream_trials")
    measurements = cast(dict[str, object], cast(dict[str, object], record["evidence"])["measurements"])
    trials = cast(list[dict[str, object]], measurements["stream_trials"])
    trials[0].update(
        {
            "status": "fail",
            "failure": {"type": "OSError", "message": r"cannot read C:\private\weights\talker.gguf"},
        }
    )
    with pytest.raises(qualified_gpu.RecordValidationError, match=r"machine-local absolute path"):
        qualified_gpu.validate_record(record)


@pytest.mark.parametrize(
    "machine_path",
    ["file:///var/lib/private-host/model.gguf", "//private-server/share/model.gguf"],
)
def test_record_rejects_nested_trial_failure_uri_and_authority_paths(machine_path: str) -> None:
    record = _record(profile="architecture-smoke", cuda_architecture="sm_86", failing="stream_trials")
    measurements = cast(dict[str, object], cast(dict[str, object], record["evidence"])["measurements"])
    trials = cast(list[dict[str, object]], measurements["stream_trials"])
    trials[0].update(
        {
            "status": "fail",
            "failure": {"type": "OSError", "message": f"cannot read {machine_path}"},
        }
    )
    with pytest.raises(qualified_gpu.RecordValidationError, match=r"machine-local absolute path"):
        qualified_gpu.validate_record(record)


def test_record_writer_redacts_file_uri_and_authority_paths_recursively() -> None:
    value = {
        "note": "observed:file:///var/lib/private-host/model.gguf",
        "trials": [{"failure": {"message": "cannot read //private-server/share/model.gguf"}}],
        "remote": "https://example.invalid/asset.whl",
    }
    redacted = qualified_gpu._redact_machine_local_paths(value)
    assert redacted["note"] == "observed:<machine-local-path>"
    assert redacted["trials"][0]["failure"]["message"] == "cannot read <machine-local-path>"
    assert redacted["remote"] == "https://example.invalid/asset.whl"


def test_record_writer_redacts_absolute_paths_recursively() -> None:
    entry = qualified_gpu.CATALOG_BY_ID[qualified_gpu.ARCHITECTURE_SMOKE_MODEL_ID]
    measurements = _smoke_measurements()
    trials = cast(list[dict[str, object]], measurements["stream_trials"])
    trials[0].update(
        {
            "status": "fail",
            "failure": {"type": "OSError", "message": r"cannot read C:\private\weights\talker.gguf"},
        }
    )
    environment = _environment(cuda_architecture="sm_86")
    environment["co_resident_gpu_note"] = "shared workload log: /var/log/private-gpu.log"
    record = qualified_gpu.make_record(
        entry,
        profile="architecture-smoke",
        checks=_checks("architecture-smoke", failing="stream_trials"),
        measurements=measurements,
        environment=environment,
        artifact=_artifact(),
        run_input=_smoke_run_input(),
        executed_at="2026-09-09T12:00:00+09:00",
        failure={
            "check": "stream_trials",
            "type": "OSError",
            "message": r"server \\private-server\share\weights failed",
        },
    )

    encoded = json.dumps(record)
    assert "/var/log/private-gpu.log" not in encoded
    assert r"C:\\private\\weights\\talker.gguf" not in encoded
    assert r"\\\\private-server\\share\\weights" not in encoded
    assert encoded.count("<machine-local-path>") >= 3
    qualified_gpu.validate_record(record)


class _ExpectedNativeError(RuntimeError):
    pass


class _FailingNativeBinding:
    def load_model(self, talker_path: str, codec_path: str, *, codec_chunk_sec: float | None = None) -> object:
        raise _ExpectedNativeError("native operation=qt_init; status=null; diagnostic=model file does not exist")


def test_native_diagnostic_exercises_binding_error_with_operation_status_and_diagnostic(tmp_path: Path) -> None:
    binding_module = SimpleNamespace(CtypesNativeBinding=_FailingNativeBinding, NativeCallError=_ExpectedNativeError)
    detail = qualified_gpu._exercise_native_diagnostic(binding_module, prefix=tmp_path)
    assert detail == "native operation=qt_init; status=null; diagnostic was non-empty"


@pytest.mark.req("SP-SFQWEN3-023")
async def test_full_pair_counting_binding_forwards_codec_chunk_sec_during_prepare(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Full-pair prepare passes the engine's codec chunk setting through the counting binding."""
    engine_module = importlib.import_module("speechflow.qwen3._engine")
    manifest_module = importlib.import_module("speechflow.qwen3._manifest")
    entry = qualified_gpu.CATALOG_BY_ID[qualified_gpu.ARCHITECTURE_SMOKE_MODEL_ID]
    pair = next(pair for pair in manifest_module.load_manifest() if pair.model_id == entry.model_id)

    class RecordingBinding:
        def __init__(self) -> None:
            self.loaded: list[tuple[str, str, float | None]] = []
            self.context = object()

        def load_model(self, talker_path: str, codec_path: str, *, codec_chunk_sec: float | None = None) -> object:
            self.loaded.append((talker_path, codec_path, codec_chunk_sec))
            return self.context

        def list_speakers(self, context: object) -> tuple[str, ...]:
            assert context is self.context
            return ("Ryan",)

        def free_context(self, context: object) -> None:
            assert context is self.context

        def synthesize(self, context: object, request: object, *, on_chunk: object, cancelled: object) -> Any:
            assert context is self.context
            return engine_module.NativeAudio(np.array([0.0], dtype=np.float32), 24000, 1)

    inner = RecordingBinding()
    binding = qualified_gpu._CountingBinding(inner)
    monkeypatch.setattr(engine_module, "binding_factory", lambda: binding)
    monkeypatch.setattr(engine_module, "validate_model_source", lambda *_args, **_kwargs: pair)
    talker_path = str(tmp_path / entry.talker)
    codec_path = str(tmp_path / entry.codec)
    source = engine_module.Qwen3ModelSource(talker_path, codec_path)
    engine = engine_module.Qwen3CustomVoiceTTSEngine(source, default_voice="Ryan", codec_chunk_sec=3.5)
    try:
        await engine.prepare()
        await engine.prepare()
        assert inner.loaded == [(talker_path, codec_path, 3.5)]
        assert binding.load_calls == 1
    finally:
        await engine.aclose()


def test_gpu_memory_sampler_preserves_poll_failure_and_physical_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples: list[float | BaseException] = [3_000.0, OSError("transient nvidia-smi failure"), 3_250.0]

    def read_sample(physical_gpu_index: int) -> float:
        assert physical_gpu_index == 2
        value = samples.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(qualified_gpu._GpuMemorySampler, "_read_total_mib", staticmethod(read_sample))
    sampler = qualified_gpu._GpuMemorySampler(interval=0.1, physical_gpu_index=2)
    sampler._sample_once()
    measurement = sampler.stop()
    assert measurement["errors"] == ["OSError: transient nvidia-smi failure"]
    with pytest.raises(qualified_gpu._GateFailure, match="errors=1"):
        qualified_gpu._validate_gpu_memory_measurement(measurement)


def test_gpu_memory_sampler_completes_due_samples_when_a_poll_runs_late(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def read_sample(physical_gpu_index: int) -> float:
        nonlocal calls
        calls += 1
        if calls == 2:
            time.sleep(0.35)
        return 4_000.0 + calls

    monkeypatch.setattr(qualified_gpu._GpuMemorySampler, "_read_total_mib", staticmethod(read_sample))
    sampler = qualified_gpu._GpuMemorySampler(interval=0.1, physical_gpu_index=0)
    sampler.start()
    time.sleep(0.15)
    measurement = sampler.stop()
    assert measurement["errors"] == []
    assert measurement["sample_count"] >= measurement["expected_sample_count"] >= 5
    assert measurement["peak_total_mib"] == 4_000.0 + calls
    qualified_gpu._validate_gpu_memory_measurement(measurement)


class _FakeDriver:
    def __init__(self, *, profile: str, failing: str | None = None) -> None:
        self.profile = profile
        self.failing = failing
        self.seen: list[str] = []

    async def execute(self, entry: qualified_gpu.CatalogEntry) -> qualified_gpu.PairOutcome:
        self.seen.append(entry.model_id)
        smoke = self.profile == "architecture-smoke"
        return qualified_gpu.PairOutcome(
            checks=_checks(self.profile, failing=self.failing),
            measurements=_smoke_measurements() if smoke else _full_measurements(base=entry.mode == "Base"),
            run_input=_smoke_run_input() if smoke else _full_run_input(base=entry.mode == "Base"),
            failure=(
                {"check": self.failing, "type": "FakeFailure", "message": "fake driver failure"}
                if self.failing is not None
                else None
            ),
        )


@pytest.mark.req("SP-SFQWEN3-017")
@pytest.mark.parametrize("profile", ["full-pair", "architecture-smoke"])
async def test_runner_writes_schema_v3_once_and_preserves_gate_failure(tmp_path: Path, profile: str) -> None:
    """The runner writes one schema-3 record without a commit and preserves the failed qualification."""
    driver = _FakeDriver(profile=profile, failing="warmup")
    destination = tmp_path / "record.json"
    record = await qualified_gpu.run_qualified_pair(
        "qwen3-tts-0.6b-customvoice-Q4_K_M",
        profile=profile,
        driver=driver,
        environment=_environment(cuda_architecture="sm_86" if profile == "architecture-smoke" else "sm_120a"),
        artifact=_artifact(),
        executed_at="2026-09-09T12:00:00+09:00",
        destination=destination,
    )
    assert driver.seen == ["qwen3-tts-0.6b-customvoice-Q4_K_M"]
    assert record["status"] == "fail"
    written = json.loads(destination.read_text(encoding="utf-8"))
    assert written == record
    assert written["schema_version"] == 3
    assert "tag_commit" not in written["artifact"]
    qualified_gpu.validate_record(record)


async def test_runner_redacts_machine_local_paths_from_failure_diagnostics(tmp_path: Path) -> None:
    driver = _FakeDriver(profile="architecture-smoke", failing="warmup")
    original_execute = driver.execute

    async def execute_with_path(entry: qualified_gpu.CatalogEntry) -> qualified_gpu.PairOutcome:
        outcome = await original_execute(entry)
        outcome.checks["warmup"]["detail"] = "missing /home/producer/models/talker.gguf"
        outcome.failure = {
            "check": "warmup",
            "type": "FileNotFoundError",
            "message": "missing /home/producer/models/talker.gguf",
        }
        return outcome

    driver.execute = execute_with_path  # type: ignore[method-assign]
    destination = tmp_path / "record.json"
    record = await qualified_gpu.run_qualified_pair(
        qualified_gpu.ARCHITECTURE_SMOKE_MODEL_ID,
        profile="architecture-smoke",
        driver=driver,
        environment=_environment(cuda_architecture="sm_86"),
        artifact=_artifact(),
        executed_at="2026-09-09T12:00:00+09:00",
        destination=destination,
    )

    encoded = json.dumps(record)
    assert "/home/" not in encoded
    assert "<machine-local-path>" in encoded
    assert destination.is_file()


def _record_path(root: Path, record: Mapping[str, object]) -> Path:
    model = cast(Mapping[str, str], record["model"])
    environment = cast(Mapping[str, str], record["environment"])
    path = root / f"{model['model_id']}--{environment['cuda_architecture']}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _readme_pending() -> str:
    architecture_rows = [
        "| `sm_86` | 8.6 | NVIDIA RTX A6000 | included | pending | — |",
        "| `sm_89` | 8.9 | NVIDIA RTX 6000 Ada | included | pending | — |",
        "| `sm_120a` | 12.0 | NVIDIA RTX PRO 4000 Blackwell | included | pending | — |",
    ]
    model_rows = [f"| `{entry.model_id}` | {entry.mode} | pending | — |" for entry in qualified_gpu.CATALOG]
    return "\n".join([*architecture_rows, *model_rows]) + "\n"


@pytest.mark.req("SP-SFQWEN3-017")
def test_pending_documentation_is_consistent_with_an_empty_record_directory(tmp_path: Path) -> None:
    """Models without adopted pass records remain pending in documentation."""
    evidence = tmp_path / "qualified-evidence.md"
    rendered = qualified_gpu.render_evidence([])
    assert "schema-v3" in rendered
    evidence.write_text(rendered, encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(_readme_pending(), encoding="utf-8")
    qualified_gpu.assert_documentation_consistent(readme, evidence, [])


@pytest.mark.req("SP-SFQWEN3-017")
def test_documentation_never_presents_failed_evidence_as_qualified(tmp_path: Path) -> None:
    """Failed model evidence cannot be displayed as qualified."""
    record = _record(profile="architecture-smoke", cuda_architecture="sm_86", failing="warmup")
    record_path = _record_path(tmp_path, record)
    evidence = tmp_path / "qualified-evidence.md"
    evidence.write_text(qualified_gpu.render_evidence([record]), encoding="utf-8")
    readme = tmp_path / "README.md"
    pending_row = "| `sm_86` | 8.6 | NVIDIA RTX A6000 | included | pending | — |"
    failed_row = (
        "| `sm_86` | 8.6 | NVIDIA RTX A6000 | included | failed | "
        f"[record]({PUBLIC_DOCUMENTS_URL}/qualified-records/qwen3-tts-0.6b-customvoice-Q4_K_M--sm_86.json) |"
    )
    readme_text = _readme_pending().replace(pending_row, failed_row)
    readme.write_text(readme_text, encoding="utf-8")
    qualified_gpu.assert_documentation_consistent(readme, evidence, [record_path])
    readme.write_text(readme_text.replace("| failed |", "| qualified |"), encoding="utf-8")
    with pytest.raises(qualified_gpu.DocumentationContractError, match="failed"):
        qualified_gpu.assert_documentation_consistent(readme, evidence, [record_path])


@pytest.mark.req("SP-SFQWEN3-017")
def test_documentation_rejects_relative_readme_record_links(tmp_path: Path) -> None:
    """README qualification records must remain reachable from the PyPI description."""
    record = _record(profile="architecture-smoke", cuda_architecture="sm_86", failing="warmup")
    record_path = _record_path(tmp_path, record)
    evidence = tmp_path / "qualified-evidence.md"
    evidence.write_text(qualified_gpu.render_evidence([record]), encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(
        _readme_pending().replace(
            "| `sm_86` | 8.6 | NVIDIA RTX A6000 | included | pending | — |",
            f"| `sm_86` | 8.6 | NVIDIA RTX A6000 | included | failed | "
            f"[record](docs/qualified-records/{record_path.name}) |",
        ),
        encoding="utf-8",
    )

    with pytest.raises(qualified_gpu.DocumentationContractError, match="README architecture sm_86 must link"):
        qualified_gpu.assert_documentation_consistent(readme, evidence, [record_path])


@pytest.mark.req("SP-SFQWEN3-017")
def test_documentation_rejects_relative_readme_model_links(tmp_path: Path) -> None:
    """Each model pair links to its public qualification record in the PyPI description."""
    record = _record()
    record_path = _record_path(tmp_path, record)
    model = cast(Mapping[str, str], record["model"])
    entry = qualified_gpu.CATALOG_BY_ID[model["model_id"]]
    evidence = tmp_path / "qualified-evidence.md"
    evidence.write_text(qualified_gpu.render_evidence([record]), encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(
        _readme_pending()
        .replace(
            "| `sm_120a` | 12.0 | NVIDIA RTX PRO 4000 Blackwell | included | pending | — |",
            "| `sm_120a` | 12.0 | NVIDIA RTX PRO 4000 Blackwell | included | pending | "
            f"[record]({PUBLIC_DOCUMENTS_URL}/qualified-records/{record_path.name}) |",
        )
        .replace(
            f"| `{entry.model_id}` | {entry.mode} | pending | — |",
            f"| `{entry.model_id}` | {entry.mode} | qualified | [record](docs/qualified-records/{record_path.name}) |",
        ),
        encoding="utf-8",
    )

    with pytest.raises(qualified_gpu.DocumentationContractError, match=f"README model pair {entry.model_id} must link"):
        qualified_gpu.assert_documentation_consistent(readme, evidence, [record_path])


@pytest.mark.req("SP-SFQWEN3-017")
@pytest.mark.parametrize("operation", ["render", "check"])
def test_documentation_rejects_pass_records_from_different_releases(tmp_path: Path, operation: str) -> None:
    """Architecture qualification uses evidence from one release and names any mixed releases."""
    records = [
        _record(entry.model_id, version="0.4.0" if index == 0 else MODULE_VERSION)
        for index, entry in enumerate(qualified_gpu.CATALOG)
    ]
    record_paths = [_record_path(tmp_path, record) for record in records]
    readme = tmp_path / "README.md"
    readme.write_text(_readme_pending(), encoding="utf-8")
    evidence = tmp_path / "qualified-evidence.md"
    evidence.write_text(qualified_gpu.render_evidence([]), encoding="utf-8")

    with pytest.raises(qualified_gpu.DocumentationContractError, match="same release") as error:
        if operation == "render":
            qualified_gpu.render_evidence(records)
        else:
            qualified_gpu.assert_documentation_consistent(readme, evidence, record_paths)
    assert "v0.4.0" in str(error.value)
    assert f"v{MODULE_VERSION}" in str(error.value)


@pytest.mark.req("SP-SFQWEN3-017")
def test_repository_documents_match_adopted_records() -> None:
    """README links to public adopted records while the evidence document keeps relative links."""
    repository = Path(__file__).resolve().parents[1]
    record_paths = sorted((repository / "docs" / "qualified-records").glob("*.json"))
    # Adopted records describe their original wheel, so current build validation does not apply.
    expected_names = [f"{entry.model_id}--sm_120a.json" for entry in qualified_gpu.CATALOG]
    assert [path.name for path in record_paths] == sorted(expected_names)
    readme = (repository / "README.md").read_text(encoding="utf-8")
    evidence = (repository / "docs/qualified-evidence.md").read_text(encoding="utf-8")

    def row(document: str, identity: str) -> list[str]:
        matches = [line for line in document.splitlines() if line.startswith(f"| `{identity}` |")]
        assert len(matches) == 1
        return [cell.strip() for cell in matches[0].split("|")[1:-1]]

    states: list[str] = []
    for path in record_paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["schema_version"] == 3
        model_id = record["model"]["model_id"]
        architecture = record["environment"]["cuda_architecture"]
        assert path.name == f"{model_id}--{architecture}.json"
        assert record["profile"] == ("full-pair" if architecture == "sm_120a" else "architecture-smoke")
        assert record["status"] in {"pass", "fail"}
        state = "qualified" if record["status"] == "pass" else "failed"
        states.append(state)
        assert row(readme, model_id)[4:] == [
            state,
            f"[{path.name}]({PUBLIC_DOCUMENTS_URL}/qualified-records/{path.name})",
        ]
        measurements = record["evidence"]["measurements"]
        performance = ["—"] * 4
        if model_id in qualified_gpu.MANDATORY_MEASUREMENT_MODELS:
            performance = [
                f"{measurements['warm_get']['wall_seconds']:.3f}",
                f"{measurements['streaming']['ttfa_seconds']:.3f}",
                str(measurements["peak_rss_bytes"]),
                f"{measurements['gpu_memory']['peak_delta_mib']:.3f}",
            ]
        assert row(evidence, model_id)[1:] == [
            state,
            record["executed_at"],
            architecture,
            f"[{path.name}](qualified-records/{path.name})",
            *performance,
        ]
    architecture_state = "failed" if "failed" in states else "qualified"
    for document, prefix in ((readme, f"{PUBLIC_DOCUMENTS_URL}/"), (evidence, "")):
        assert row(document, "sm_120a")[4] == architecture_state
        assert row(document, "sm_120a")[5] == ", ".join(
            f"[{path.name}]({prefix}qualified-records/{path.name})" for path in record_paths
        )
        for architecture in ("sm_86", "sm_89"):
            assert row(document, architecture)[4:] == ["pending", "—"]


@pytest.mark.req("SP-SFQWEN3-024")
def test_readme_links_are_absolute_for_pypi() -> None:
    """Every README link except a section anchor has an absolute destination for PyPI readers."""
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\[[^\]]*\]\(([^)]+)\)", readme)
    assert targets
    assert all(urlsplit(target).scheme for target in targets if not target.startswith("#"))
    assert f"{PUBLIC_DOCUMENTS_URL}/qualified-evidence.md" in targets


@pytest.mark.req("SP-SFQWEN3-024")
def test_pypi_project_links_to_public_repository() -> None:
    """Package metadata directs PyPI readers to the public source repository."""
    project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"].get("urls", {}).get("Repository") == PUBLIC_REPOSITORY_URL


def test_main_returns_one_for_a_written_fail_record_and_two_before_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = [
        "--profile",
        "architecture-smoke",
        "--cuda-architecture",
        "sm_86",
        "--physical-gpu-index",
        "0",
        "--model-id",
        "qwen3-tts-0.6b-customvoice-Q4_K_M",
        "--weights-dir",
        str(tmp_path),
        "--output",
        str(tmp_path / "record.json"),
        "--core-wheel",
        str(tmp_path / "core.whl"),
        "--module-wheel",
        str(tmp_path / _module_wheel_name()),
        "--sha256sums",
        str(tmp_path / "SHA256SUMS"),
        "--release-tag",
        f"v{MODULE_VERSION}",
    ]
    monkeypatch.setenv("SPEECHFLOW_QWEN3_RUN_GPU", "1")
    monkeypatch.setattr(
        qualified_gpu,
        "_execute_cli",
        lambda _: _record(profile="architecture-smoke", cuda_architecture="sm_86", failing="warmup"),
    )
    assert qualified_gpu.main(argv) == 1

    def fail_before_record(_: object) -> dict[str, object]:
        raise qualified_gpu._GateFailure("pre-record failure")

    monkeypatch.setattr(qualified_gpu, "_execute_cli", fail_before_record)
    assert qualified_gpu.main(argv) == 2


def test_failed_record_keeps_all_measurement_keys_even_when_values_are_unreached() -> None:
    record = _record(profile="architecture-smoke", cuda_architecture="sm_89", failing="warmup")
    measurements = cast(dict[str, object], cast(dict[str, object], record["evidence"])["measurements"])
    measurements.update({"warmup_seconds": None, "stream_trials": [], "peak_rss_bytes": None, "gpu_memory": None})
    qualified_gpu.validate_record(record)
    assert set(measurements) == qualified_gpu.PROFILE_MEASUREMENT_FIELDS["architecture-smoke"]


def test_artifact_build_metadata_must_match_environment_revision_and_architectures() -> None:
    record = _record()
    metadata = cast(dict[str, object], cast(dict[str, object], record["artifact"])["build_metadata"])
    metadata["qwentts_revision"] = "d" * 40
    with pytest.raises(qualified_gpu.RecordValidationError, match="qwentts_revision"):
        qualified_gpu.validate_record(record)

    record = _record()
    metadata = cast(dict[str, object], cast(dict[str, object], record["artifact"])["build_metadata"])
    metadata["cmake_cuda_architectures"] = "120a-real"
    with pytest.raises(qualified_gpu.RecordValidationError, match="cmake_cuda_architectures"):
        qualified_gpu.validate_record(record)


def test_sha256sum_parser_requires_one_matching_wheel_entry(tmp_path: Path) -> None:
    wheel = tmp_path / _module_wheel_name()
    wheel.write_bytes(b"wheel bytes")
    digest = qualified_gpu._sha256_file(wheel)
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"{digest}  {wheel.name}\n", encoding="utf-8")
    assert qualified_gpu._validate_sha256sums(wheel, sums) == digest
    sums.write_text(f"{'f' * 64}  {wheel.name}\n", encoding="utf-8")
    with pytest.raises(qualified_gpu._GateFailure, match="SHA256SUMS"):
        qualified_gpu._validate_sha256sums(wheel, sums)


def test_profile_cli_defaults_match_architecture_smoke_contract(tmp_path: Path) -> None:
    args = qualified_gpu._parse_args(
        [
            "--profile",
            "architecture-smoke",
            "--cuda-architecture",
            "sm_86",
            "--physical-gpu-index",
            "2",
            "--model-id",
            "qwen3-tts-0.6b-customvoice-Q4_K_M",
            "--weights-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "record.json"),
            "--core-wheel",
            str(tmp_path / "core.whl"),
            "--module-wheel",
            str(tmp_path / _module_wheel_name()),
            "--sha256sums",
            str(tmp_path / "SHA256SUMS"),
            "--release-tag",
            f"v{MODULE_VERSION}",
        ]
    )
    assert args.text == qualified_gpu.ARCHITECTURE_SMOKE_TEXT
    assert args.language == "en"
    assert args.speaker == "ryan"
    assert args.seed == 7
    assert args.max_new_tokens == 64
    assert args.trial_count == 3
    assert args.physical_gpu_index == 2


class _DriverState:
    def __init__(self, scenario: str = "success") -> None:
        self.scenario = scenario
        self.prepare_calls = 0
        self.get_calls = 0
        self.stream_calls = 0
        self.close_calls = 0
        self.sampler_indexes: list[int] = []


class _DriverNativeBinding:
    def load_model(self, talker_path: str, codec_path: str, *, codec_chunk_sec: float | None = None) -> object:
        del talker_path, codec_path
        return object()

    def free_context(self, context: object) -> None:
        del context

    def list_speakers(self, context: object) -> tuple[str, ...]:
        del context
        return ("ryan",)


def _driver_audio() -> AudioData:
    return AudioData(np.array([0.25, -0.25], dtype=np.float32), 24_000)


def _driver_modules(state: _DriverState) -> dict[str, object]:
    entry = qualified_gpu.CATALOG_BY_ID[qualified_gpu.ARCHITECTURE_SMOKE_MODEL_ID]
    engine_module = SimpleNamespace(binding_factory=_DriverNativeBinding)

    class ModelSource:
        def __init__(self, talker_path: str, codec_path: str) -> None:
            self.talker_path = talker_path
            self.codec_path = codec_path

    class Engine:
        def __init__(self, source: object, default_voice: str | None = None) -> None:
            del source, default_voice
            self.binding = engine_module.binding_factory()
            self.model_identifier = entry.model_id
            self._loaded: SimpleNamespace | None = None

        async def prepare(self) -> None:
            state.prepare_calls += 1
            if state.scenario == "warmup":
                raise RuntimeError("prepare failed")
            if self._loaded is None:
                context = self.binding.load_model(entry.talker, entry.codec)
                self._loaded = SimpleNamespace(context=context)

        def list_voices(self) -> list[dict[str, str]]:
            return [{"name": "ryan"}]

        async def get(self, text: str, **options: object) -> AudioData:
            del text
            state.get_calls += 1
            if options.get("seed") is True:
                if state.scenario == "post_stream":
                    return _driver_audio()
                raise ConfigurationError("seed must be an integer")
            return _driver_audio()

        def stream(self, text: str, **options: object) -> AsyncIterator[AudioData]:
            del text, options
            state.stream_calls += 1
            call_number = state.stream_calls

            async def generate() -> AsyncIterator[AudioData]:
                completed = False
                try:
                    if state.scenario == "before_stream" and call_number == 1:
                        raise RuntimeError("stream failed before the first chunk")
                    yield _driver_audio()
                    if state.scenario == "during_stream" and call_number == 1:
                        raise RuntimeError("stream failed after the first chunk")
                    if state.scenario == "smoke_stream" and call_number == 2:
                        raise RuntimeError("second smoke trial failed after the first chunk")
                    yield _driver_audio()
                    completed = True
                finally:
                    if not completed and hasattr(self.binding, "cancel_observations"):
                        self.binding.cancel_observations += 1

            return generate()

        async def aclose(self) -> None:
            state.close_calls += 1
            if state.scenario == "cleanup":
                raise RuntimeError("cleanup failed")

    engine_module.Qwen3ModelSource = ModelSource
    engine_module.Qwen3CustomVoiceTTSEngine = Engine
    engine_module.Qwen3CloneTTSEngine = Engine
    manifest_module = SimpleNamespace(
        load_manifest=lambda: [
            SimpleNamespace(
                model_id=entry.model_id,
                talker=SimpleNamespace(basename=entry.talker),
                codec=SimpleNamespace(basename=entry.codec),
            )
        ]
    )
    binding_module = SimpleNamespace(CtypesNativeBinding=_DriverNativeBinding, NativeCallError=RuntimeError)
    return {
        "speechflow.qwen3._engine": engine_module,
        "speechflow.qwen3._manifest": manifest_module,
        "speechflow.qwen3._binding": binding_module,
    }


def _driver_identity(version: str = MODULE_VERSION) -> qualified_gpu.InstalledWheelIdentity:
    return qualified_gpu.InstalledWheelIdentity(
        relative_origin=Path("lib/python3.12/site-packages/speechflow/qwen3/__init__.py"),
        distribution_version=version,
        module_wheel=_module_wheel_name(version),
    )


def _install_driver_boundaries(monkeypatch: pytest.MonkeyPatch, state: _DriverState) -> None:
    modules = _driver_modules(state)
    original_import = importlib.import_module

    def import_module(name: str) -> Any:
        return modules[name] if name in modules else original_import(name)

    class Sampler:
        def __init__(self, interval: float, *, physical_gpu_index: int) -> None:
            del interval
            state.sampler_indexes.append(physical_gpu_index)
            if state.scenario == "sampler_init":
                raise RuntimeError("sampler initialization failed")

        def start(self) -> None:
            return None

        def stop(self) -> dict[str, object]:
            if state.scenario == "sampler_stop":
                raise RuntimeError("sampler cleanup failed")
            return _gpu_memory()

    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setattr(qualified_gpu, "_exercise_native_diagnostic", lambda *_args, **_kwargs: "native diagnostic")
    monkeypatch.setattr(qualified_gpu, "_GpuMemorySampler", Sampler)
    monkeypatch.setattr(qualified_gpu, "_rss_bytes", lambda: 123_000_000)
    clock_ticks = iter(index / 100 for index in range(1_000))
    monkeypatch.setattr(time, "perf_counter", lambda: next(clock_ticks))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")


def _native_driver(state: _DriverState) -> qualified_gpu.NativePairDriver:
    return qualified_gpu.NativePairDriver(
        qualified_gpu.NativeDriverConfig(
            weights_dir=Path("weights"),
            text="Qualified synthesis.",
            language="en",
            speaker="ryan",
            reference_audio_npy=None,
            reference_source=None,
            reference_text=None,
            reference_output_npy=None,
            max_new_tokens=64,
            seed=7,
            trial_count=2,
            vram_poll_interval=0.1,
            identity=_driver_identity(),
            environment=_environment(),
            physical_gpu_index=3,
        )
    )


def _smoke_driver(state: _DriverState) -> qualified_gpu.ArchitectureSmokeDriver:
    return qualified_gpu.ArchitectureSmokeDriver(
        qualified_gpu.ArchitectureSmokeConfig(
            weights_dir=Path("weights"),
            text=qualified_gpu.ARCHITECTURE_SMOKE_TEXT,
            language="en",
            speaker="ryan",
            max_new_tokens=64,
            seed=7,
            trial_count=3,
            vram_poll_interval=0.1,
            physical_gpu_index=3,
            identity=_driver_identity(),
            environment=_environment(cuda_architecture="sm_86"),
            artifact=_artifact(),
        )
    )


async def test_native_pair_driver_executes_the_success_state_machine_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _DriverState()
    _install_driver_boundaries(monkeypatch, state)

    outcome = await _native_driver(state).execute(
        qualified_gpu.CATALOG_BY_ID[qualified_gpu.ARCHITECTURE_SMOKE_MODEL_ID]
    )

    assert {name: value["status"] for name, value in outcome.checks.items()} == dict.fromkeys(
        qualified_gpu.FULL_PAIR_CHECKS, "pass"
    )
    assert outcome.failure is None
    assert state.prepare_calls == 2
    assert state.get_calls == 3
    assert state.stream_calls == 2
    assert state.close_calls == 2
    assert state.sampler_indexes == [3]


@pytest.mark.parametrize(
    ("scenario", "failed_check", "last_pass", "first_not_run"),
    [
        ("warmup", "warmup", "manifest_load", "warmup_idempotent"),
        ("before_stream", "stream_first_chunk", "get_audio", "stream_complete"),
        ("post_stream", "configuration_diagnostic", "cancellation", None),
        ("sampler_stop", "gpu_memory_measurement", "close", None),
        ("cleanup", "close", "configuration_diagnostic", None),
    ],
)
async def test_native_pair_driver_preserves_failure_transitions_and_measurements(
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    failed_check: str,
    last_pass: str,
    first_not_run: str | None,
) -> None:
    state = _DriverState(scenario)
    _install_driver_boundaries(monkeypatch, state)

    outcome = await _native_driver(state).execute(
        qualified_gpu.CATALOG_BY_ID[qualified_gpu.ARCHITECTURE_SMOKE_MODEL_ID]
    )

    assert outcome.checks[failed_check]["status"] == "fail"
    assert outcome.checks[last_pass]["status"] == "pass"
    if first_not_run is not None:
        assert outcome.checks[first_not_run]["status"] == "not_run"
    assert outcome.failure is not None
    assert outcome.failure["check"] == failed_check
    assert outcome.measurements["peak_rss_bytes"] == 123_000_000
    if scenario in {"before_stream", "post_stream", "sampler_stop", "cleanup"}:
        assert outcome.measurements["warm_get"] is not None


async def test_native_pair_driver_keeps_partial_stream_measurement_after_midstream_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _DriverState("during_stream")
    _install_driver_boundaries(monkeypatch, state)

    outcome = await _native_driver(state).execute(
        qualified_gpu.CATALOG_BY_ID[qualified_gpu.ARCHITECTURE_SMOKE_MODEL_ID]
    )

    assert outcome.checks["stream_first_chunk"]["status"] == "pass"
    assert outcome.checks["stream_complete"]["status"] == "fail"
    streaming = cast(Mapping[str, object], outcome.measurements["streaming"])
    assert streaming["samples"] == 2
    assert state.stream_calls == 1
    assert outcome.checks["audio_contract"]["status"] == "not_run"
    assert outcome.checks["close"]["status"] == "pass"
    assert outcome.checks["gpu_memory_measurement"]["status"] == "pass"


async def test_architecture_smoke_driver_executes_three_independent_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _DriverState()
    _install_driver_boundaries(monkeypatch, state)

    outcome = await _smoke_driver(state).execute(qualified_gpu.CATALOG_BY_ID[qualified_gpu.ARCHITECTURE_SMOKE_MODEL_ID])

    assert {name: value["status"] for name, value in outcome.checks.items()} == dict.fromkeys(
        qualified_gpu.ARCHITECTURE_SMOKE_CHECKS, "pass"
    )
    assert len(cast(list[object], outcome.measurements["stream_trials"])) == 3
    assert state.stream_calls == 3
    assert state.close_calls == 2
    assert state.sampler_indexes == [3]


async def test_architecture_smoke_driver_stops_after_a_midstream_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _DriverState("smoke_stream")
    _install_driver_boundaries(monkeypatch, state)

    outcome = await _smoke_driver(state).execute(qualified_gpu.CATALOG_BY_ID[qualified_gpu.ARCHITECTURE_SMOKE_MODEL_ID])

    trials = cast(list[Mapping[str, object]], outcome.measurements["stream_trials"])
    assert [trial["status"] for trial in trials] == ["pass", "fail"]
    assert trials[1]["samples"] == 2
    assert state.stream_calls == 2
    assert outcome.checks["stream_trials"]["status"] == "fail"
    assert outcome.checks["audio_contract"]["status"] == "not_run"
    assert outcome.checks["close"]["status"] == "pass"
    assert outcome.checks["gpu_memory_measurement"]["status"] == "pass"


async def test_architecture_smoke_driver_attributes_sampler_initialization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _DriverState("sampler_init")
    _install_driver_boundaries(monkeypatch, state)

    outcome = await _smoke_driver(state).execute(qualified_gpu.CATALOG_BY_ID[qualified_gpu.ARCHITECTURE_SMOKE_MODEL_ID])

    assert outcome.failure is not None
    assert outcome.failure["check"] == "gpu_memory_measurement"
    assert outcome.checks["gpu_memory_measurement"]["status"] == "fail"
    assert outcome.checks["warmup"]["status"] == "not_run"
    assert outcome.checks["close"]["status"] == "pass"


def _real_cli_argv(tmp_path: Path, *, version: str = MODULE_VERSION) -> tuple[list[str], Path, Path, Path]:
    module_wheel = tmp_path / _module_wheel_name(version)
    core_wheel = tmp_path / "speechflow-0.16.2-py3-none-any.whl"
    module_wheel.write_bytes(b"module release wheel")
    core_wheel.write_bytes(b"core release wheel")
    sha256sums = tmp_path / "SHA256SUMS"
    sha256sums.write_text(f"{qualified_gpu._sha256_file(module_wheel)}  {module_wheel.name}\n", encoding="utf-8")
    output = tmp_path / "qwen3-tts-0.6b-customvoice-Q4_K_M--sm_86.json"
    return (
        [
            "--profile",
            "architecture-smoke",
            "--cuda-architecture",
            "sm_86",
            "--physical-gpu-index",
            "3",
            "--model-id",
            qualified_gpu.ARCHITECTURE_SMOKE_MODEL_ID,
            "--weights-dir",
            str(tmp_path / "weights"),
            "--output",
            str(output),
            "--core-wheel",
            str(core_wheel),
            "--module-wheel",
            str(module_wheel),
            "--sha256sums",
            str(sha256sums),
            "--release-tag",
            f"v{version}",
        ],
        output,
        module_wheel,
        core_wheel,
    )


@pytest.mark.req("SP-SFQWEN3-017")
@pytest.mark.parametrize("version", [MODULE_VERSION, "1.2.3"])
def test_cli_accepts_a_release_tag_matching_the_module_wheel(tmp_path: Path, version: str) -> None:
    """The runner accepts matching release and wheel versions without a release-specific code change."""
    argv, _, module_wheel, _ = _real_cli_argv(tmp_path, version=version)
    args = qualified_gpu._parse_args(argv)
    assert args.release_tag == f"v{version}"
    assert args.module_wheel == module_wheel


@pytest.mark.req("SP-SFQWEN3-037")
@pytest.mark.req("SP-SFQWEN3-017")
@pytest.mark.req("SP-SF-028")
@pytest.mark.parametrize(
    ("version", "release_tag"),
    [(MODULE_VERSION, "v0.4.0"), ("0.5.1", f"v{MODULE_VERSION}"), (MODULE_VERSION, MODULE_VERSION)],
)
def test_cli_rejects_a_release_tag_that_does_not_match_the_module_wheel(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    version: str,
    release_tag: str,
) -> None:
    """The runner rejects mismatched versions and missing tag prefixes before any GPU work."""
    argv, output, module_wheel, _ = _real_cli_argv(tmp_path, version=version)
    argv[argv.index("--release-tag") + 1] = release_tag
    with pytest.raises(SystemExit) as error:
        qualified_gpu.main(argv)
    assert error.value.code == 2
    diagnostic = capsys.readouterr().err
    assert release_tag in diagnostic
    assert module_wheel.name in diagnostic
    assert not output.exists()


@pytest.mark.req("SP-SFQWEN3-037")
@pytest.mark.req("SP-SF-028")
@pytest.mark.parametrize("version", ["", "garbage", "0.5", "0.5.0rc1", "0.5.0.dev1"])
def test_cli_rejects_matching_versions_outside_the_release_tag_format(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    version: str,
) -> None:
    """The runner requires vX.Y.Z rather than accepting any matching tag and wheel token."""
    argv, output, _, _ = _real_cli_argv(tmp_path, version=version)
    with pytest.raises(SystemExit) as error:
        qualified_gpu.main(argv)
    assert error.value.code == 2
    diagnostic = capsys.readouterr().err
    assert "vX.Y.Z" in diagnostic
    assert f"v{version}" in diagnostic
    assert not output.exists()


@pytest.mark.req("SP-SFQWEN3-024")
def test_cli_rejects_a_matching_version_with_the_wrong_wheel_platform(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The runner retains the Linux x86_64 wheel requirement when the release version matches."""
    argv, output, _, _ = _real_cli_argv(tmp_path)
    wheel_name = f"speechflow_qwen3-{MODULE_VERSION}-py3-none-any.whl"
    argv[argv.index("--module-wheel") + 1] = str(tmp_path / wheel_name)
    with pytest.raises(SystemExit) as error:
        qualified_gpu.main(argv)
    assert error.value.code == 2
    diagnostic = capsys.readouterr().err
    assert wheel_name in diagnostic
    assert "manylinux_2_38_x86_64" in diagnostic
    assert not output.exists()


@pytest.mark.req("SP-SFQWEN3-017")
@pytest.mark.parametrize(("scenario", "exit_code"), [("success", 0), ("smoke_stream", 1)])
def test_real_cli_runs_real_smoke_driver_and_binds_both_installed_wheels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    exit_code: int,
) -> None:
    """A matching future release produces a pass or fail record bound to both installed wheels."""
    state = _DriverState(scenario)
    _install_driver_boundaries(monkeypatch, state)
    argv, output, module_wheel, core_wheel = _real_cli_argv(tmp_path)
    identity_calls: list[tuple[str, Path, str]] = []

    def module_identity(wheel: Path, digest: str) -> qualified_gpu.InstalledWheelIdentity:
        identity_calls.append(("speechflow-qwen3", wheel, digest))
        return _driver_identity()

    def core_identity(wheel: Path, digest: str) -> qualified_gpu.InstalledWheelIdentity:
        identity_calls.append(("speechflow", wheel, digest))
        return qualified_gpu.InstalledWheelIdentity(
            relative_origin=Path("lib/python3.12/site-packages/speechflow/__init__.py"),
            distribution_version="0.16.2",
            module_wheel=wheel.name,
        )

    monkeypatch.setattr(qualified_gpu, "_site_packages_origin", module_identity)
    monkeypatch.setattr(qualified_gpu, "_core_site_packages_origin", core_identity, raising=False)
    monkeypatch.setattr(qualified_gpu, "_build_metadata", _build_metadata)
    monkeypatch.setattr(
        qualified_gpu,
        "collect_environment",
        lambda *, cuda_architecture, physical_gpu_index, co_resident_gpu_note: {
            **_environment(cuda_architecture=cuda_architecture),
            "physical_gpu_index": physical_gpu_index,
            "co_resident_gpu_note": co_resident_gpu_note,
        },
    )
    monkeypatch.setenv("SPEECHFLOW_QWEN3_RUN_GPU", "1")

    assert qualified_gpu.main(argv) == exit_code
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["schema_version"] == 3
    assert "tag_commit" not in record["artifact"]
    assert record["status"] == ("pass" if exit_code == 0 else "fail")
    assert [name for name, _, _ in identity_calls] == ["speechflow-qwen3", "speechflow"]
    assert identity_calls[0][1] == module_wheel
    assert identity_calls[0][2] == qualified_gpu._sha256_file(module_wheel)
    assert identity_calls[1][1] == core_wheel
    assert identity_calls[1][2] == qualified_gpu._sha256_file(core_wheel)
    assert state.sampler_indexes == [3]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    if scenario == "smoke_stream":
        assert state.stream_calls == 2


def test_real_cli_returns_two_before_writing_a_record_when_core_wheel_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _DriverState()
    _install_driver_boundaries(monkeypatch, state)
    argv, output, _, core_wheel = _real_cli_argv(tmp_path)
    core_wheel.unlink()
    monkeypatch.setattr(qualified_gpu, "_site_packages_origin", lambda _wheel, _digest: _driver_identity())
    monkeypatch.setenv("SPEECHFLOW_QWEN3_RUN_GPU", "1")

    assert qualified_gpu.main(argv) == 2
    assert not output.exists()
    assert state.prepare_calls == 0


def test_nvidia_environment_uses_the_selected_physical_gpu_index(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        commands.append(command)
        if "--query-gpu=index" in command:
            return subprocess.CompletedProcess(command, 0, stdout="0\n3\n", stderr="")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="NVIDIA RTX A6000, 596.72, 8.6, GPU-selected\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", run)

    environment = qualified_gpu._nvidia_environment(3)

    assert environment == ("NVIDIA RTX A6000", "596.72", "8.6", "GPU-selected", 2)
    assert commands[1][:3] == ["nvidia-smi", "-i", "3"]
