#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import resource
import subprocess
import sys
import threading
import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast
from urllib.parse import unquote, urlsplit

import numpy as np
from speechflow.core import AudioData, ConfigurationError

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.vendor_source import load_sources


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    model_id: str
    mode: str
    size: str
    quantization: str
    talker: str
    codec: str


def _entry(mode: str, size: str, quantization: str) -> CatalogEntry:
    lowered_mode = mode.lower()
    lowered_size = size.lower()
    return CatalogEntry(
        model_id=f"qwen3-tts-{lowered_size}-{lowered_mode}-{quantization}",
        mode=mode,
        size=size,
        quantization=quantization,
        talker=f"qwen-talker-{lowered_size}-{lowered_mode}-{quantization}.gguf",
        codec=f"qwen-tokenizer-12hz-{quantization}.gguf",
    )


CATALOG = (
    _entry("CustomVoice", "0.6B", "Q4_K_M"),
    _entry("CustomVoice", "0.6B", "Q8_0"),
    _entry("Base", "0.6B", "Q4_K_M"),
    _entry("Base", "0.6B", "Q8_0"),
    _entry("CustomVoice", "1.7B", "Q4_K_M"),
    _entry("CustomVoice", "1.7B", "Q8_0"),
    _entry("Base", "1.7B", "Q4_K_M"),
    _entry("Base", "1.7B", "Q8_0"),
)
CATALOG_BY_ID = {entry.model_id: entry for entry in CATALOG}
FULL_PAIR_CHECKS = (
    "isolated_import",
    "host_boundary",
    "manifest_load",
    "native_diagnostic",
    "warmup",
    "warmup_idempotent",
    "get_audio",
    "stream_first_chunk",
    "stream_complete",
    "audio_contract",
    "voice_contract",
    "cancellation",
    "close",
    "configuration_diagnostic",
    "gpu_memory_measurement",
)
ARCHITECTURE_SMOKE_CHECKS = (
    "release_identity",
    "host_boundary",
    "selected_gpu",
    "wheel_digest",
    "warmup",
    "stream_trials",
    "audio_contract",
    "gpu_memory_measurement",
    "close",
)
PROFILE_CHECKS = {
    "full-pair": FULL_PAIR_CHECKS,
    "architecture-smoke": ARCHITECTURE_SMOKE_CHECKS,
}
# The full-pair driver predates profiles and uses this name internally.
REQUIRED_CHECKS = FULL_PAIR_CHECKS
PROFILE_MEASUREMENT_FIELDS = {
    "full-pair": {
        "cold_load_seconds",
        "warm_get",
        "streaming",
        "peak_rss_bytes",
        "gpu_memory",
        "base_conditioning_seconds",
    },
    "architecture-smoke": {"warmup_seconds", "stream_trials", "peak_rss_bytes", "gpu_memory"},
}
ARCHITECTURE_TO_CAPABILITY = {"sm_86": "8.6", "sm_89": "8.9", "sm_120a": "12.0"}
ARCHITECTURE_SMOKE_MODEL_ID = "qwen3-tts-0.6b-customvoice-Q4_K_M"
ARCHITECTURE_SMOKE_TEXT = "SpeechFlow multi-architecture qualified synthesis."
QUALIFIED_QWENTTS_REVISION = load_sources()["qwentts"].revision
EXPECTED_CMAKE_CUDA_ARCHITECTURES = "86-real;89-real;120a-real"
CANONICAL_REPOSITORY = "sync-dev-org/speechflow-qwen3"
# Match scripts/release.py without importing its wheel inspector's packaging dependency.
# This runner executes in a venv containing the release wheels and their runtime dependencies.
_RELEASE_TAG_PATTERN = re.compile(r"v(?P<version>\d+\.\d+\.\d+)")
MANDATORY_MEASUREMENT_MODELS = {
    "qwen3-tts-0.6b-customvoice-Q4_K_M",
    "qwen3-tts-0.6b-base-Q4_K_M",
}
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "profile",
    "executed_at",
    "status",
    "model",
    "environment",
    "artifact",
    "evidence",
    "failure",
}
_ENVIRONMENT_FIELDS = {
    "hardware",
    "os",
    "glibc",
    "python_version",
    "nvidia_driver",
    "cuda_runtime",
    "gpu_compute_capability",
    "cuda_architecture",
    "physical_gpu_index",
    "gpu_uuid",
    "gpu_count",
    "qwentts_revision",
    "co_resident_gpu_note",
}
_ARTIFACT_FIELDS = {
    "repository",
    "release_tag",
    "module_wheel",
    "module_wheel_sha256",
    "core_wheel",
    "core_wheel_sha256",
    "module_import_path",
    "build_metadata",
}
_BUILD_METADATA_FIELDS = {
    "schema_version",
    "build_type",
    "cpu_baseline",
    "qwentts_revision",
    "ggml_revision",
    "cuda_toolkit_version",
    "cmake_cuda_architectures",
}
_FULL_PAIR_RUN_INPUT_FIELDS = {
    "text",
    "language",
    "speaker",
    "reference",
    "reference_text_modes",
    "options",
    "warm_up",
    "trial_count",
    "concurrency",
    "build_type",
}
_ARCHITECTURE_SMOKE_RUN_INPUT_FIELDS = {
    "text",
    "language",
    "speaker",
    "options",
    "trial_count",
    "concurrency",
    "build_type",
}
_MACHINE_LOCAL_ABSOLUTE_PATH = re.compile(
    r"""(?x)
    (?:
        (?<![A-Za-z0-9])file://[^\s,;'\"<>|?*()\[\]{}]+
        |
        (?<![A-Za-z0-9._~/\\-])(?<!https:)(?<!http:)(?<!ftps:)(?<!sftp:)(?<!ftp:)(?<!ssh:)(?<!git:)(?<!s3:)(?<!gs:)
        //[^\s/,;'\"<>|?*()\[\]{}]+/[^\s,;'\"<>|?*()\[\]{}]+
        |
        (?<![A-Za-z0-9._~/\\-])/(?!/)[^\s,;'\"<>|?*()\[\]{}]+
        |
        (?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s,;'\"<>|?*()\[\]{}]+
        |
        (?<![A-Za-z0-9\\])\\\\[^\s\\/,;'\"<>|?*()\[\]{}]+[\\/][^\s,;'\"<>|?*()\[\]{}]+
    )
    """
)
_DOCUMENT_MACHINE_LOCAL_PATH_FRAGMENTS = ("/home/", "/.nf/", "/tmp/")


class RecordValidationError(ValueError):
    """Raised when a qualified result is not a complete schema-v3 record."""


class DocumentationContractError(ValueError):
    """Raised when README, evidence, and JSON records disagree."""


@dataclass(slots=True)
class PairOutcome:
    checks: dict[str, dict[str, str]]
    measurements: dict[str, Any]
    run_input: dict[str, Any]
    failure: dict[str, str] | None = None


class PairDriver(Protocol):
    async def execute(self, entry: CatalogEntry) -> PairOutcome: ...


@dataclass(frozen=True, slots=True)
class InstalledWheelIdentity:
    relative_origin: Path
    distribution_version: str
    module_wheel: str


def _portable_diagnostic(value: str) -> str:
    return _MACHINE_LOCAL_ABSOLUTE_PATH.sub("<machine-local-path>", value)


def _redact_machine_local_paths(value: Any) -> Any:
    if isinstance(value, str):
        return _portable_diagnostic(value)
    if isinstance(value, Mapping):
        return {key: _redact_machine_local_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_machine_local_paths(item) for item in value]
    return value


def _reject_machine_local_record_paths(value: Any, *, field: str = "record") -> None:
    if isinstance(value, str):
        match = _MACHINE_LOCAL_ABSOLUTE_PATH.search(value)
        if match is not None:
            raise RecordValidationError(f"{field} contains machine-local absolute path {match.group(0)!r}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_machine_local_record_paths(item, field=f"{field}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_machine_local_record_paths(item, field=f"{field}[{index}]")


def make_record(
    entry: CatalogEntry,
    *,
    profile: str,
    checks: Mapping[str, Mapping[str, str]],
    measurements: Mapping[str, Any],
    environment: Mapping[str, Any],
    artifact: Mapping[str, Any],
    run_input: Mapping[str, Any],
    executed_at: str,
    failure: Mapping[str, str] | None,
) -> dict[str, Any]:
    try:
        required_checks = PROFILE_CHECKS[profile]
    except KeyError as exc:
        raise RecordValidationError(f"unknown profile: {profile!r}") from exc
    copied_checks = {
        name: {
            **value,
            "detail": _portable_diagnostic(value.get("detail", "")),
        }
        for name, value in checks.items()
    }
    status = "pass" if all(copied_checks.get(name, {}).get("status") == "pass" for name in required_checks) else "fail"
    record = {
        "schema_version": 3,
        "profile": profile,
        "executed_at": executed_at,
        "status": status,
        "model": {
            "model_id": entry.model_id,
            "mode": entry.mode,
            "size": entry.size,
            "quantization": entry.quantization,
            "talker": entry.talker,
            "codec": entry.codec,
        },
        "environment": dict(environment),
        "artifact": dict(artifact),
        "evidence": {
            "run_input": dict(run_input),
            "checks": copied_checks,
            "measurements": dict(measurements),
        },
        "failure": (
            None
            if failure is None
            else {
                **failure,
                "message": _portable_diagnostic(failure.get("message", "")),
            }
        ),
    }
    return cast(dict[str, Any], _redact_machine_local_paths(record))


async def run_qualified_pair(
    model_id: str,
    *,
    profile: str,
    driver: PairDriver,
    environment: Mapping[str, Any],
    artifact: Mapping[str, Any],
    executed_at: str,
    destination: Path,
) -> dict[str, Any]:
    try:
        entry = CATALOG_BY_ID[model_id]
    except KeyError as exc:
        raise RecordValidationError(f"unknown qualified model_id={model_id!r}") from exc
    outcome = await driver.execute(entry)
    record = make_record(
        entry,
        profile=profile,
        checks=outcome.checks,
        measurements=outcome.measurements,
        environment=environment,
        artifact=artifact,
        run_input=outcome.run_input,
        executed_at=executed_at,
        failure=outcome.failure,
    )
    validate_record(record)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return record


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecordValidationError(f"{field} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _require_exact_fields(value: Mapping[str, Any], fields: set[str], *, field: str) -> None:
    actual = set(value)
    if actual != fields:
        missing = sorted(fields - actual)
        extra = sorted(actual - fields)
        raise RecordValidationError(f"{field} fields mismatch: missing={missing}, extra={extra}")


def _require_non_empty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RecordValidationError(f"{field} must be a non-empty string")
    return value


def _require_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise RecordValidationError(f"{field} must be a non-negative finite number")
    return float(value)


def _version_components(value: str, *, field: str) -> tuple[int, ...]:
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", value) is None:
        raise RecordValidationError(f"environment.{field} must be a numeric dotted version")
    return tuple(int(component) for component in value.split("."))


def _qualified_environment_failure(environment: Mapping[str, Any]) -> str | None:
    os_name = cast(str, environment["os"])
    if "Linux" not in os_name or "x86_64" not in os_name:
        return f"environment.os must identify Linux x86_64, got {os_name!r}"

    version_fields = ("glibc", "python_version", "nvidia_driver", "cuda_runtime")
    versions: dict[str, tuple[int, ...]] = {}
    for field in version_fields:
        try:
            versions[field] = _version_components(cast(str, environment[field]), field=field)
        except RecordValidationError as exc:
            return str(exc)
    if versions["glibc"] < (2, 38):
        return f"environment.glibc must be >=2.38, got {environment['glibc']!r}"
    if versions["python_version"][:2] != (3, 12):
        return f"environment.python_version must be in the 3.12 series, got {environment['python_version']!r}"
    # NVIDIA CUDA 13.2 GA Release Notes, Table 3: Linux >=595.45.04; Windows N/A.
    # https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/index.html
    minimum_driver = "595" if "WSL2" in os_name else "595.45.04"
    if versions["nvidia_driver"] < _version_components(minimum_driver, field="nvidia_driver"):
        return (
            f"environment.nvidia_driver must be >={minimum_driver} for {os_name}, got {environment['nvidia_driver']!r}"
        )
    if not (13, 2) <= versions["cuda_runtime"] < (14,):
        return f"environment.cuda_runtime must be >=13.2,<14, got {environment['cuda_runtime']!r}"
    return None


def _validate_qualified_environment(environment: Mapping[str, Any]) -> None:
    failure = _qualified_environment_failure(environment)
    if failure is not None:
        raise _GateFailure(f"qualified host boundary failed: {failure}")


def _validate_audio_measurement(value: object, *, field: str) -> None:
    measurement = _require_mapping(value, field=field)
    required = {"wall_seconds", "audio_seconds", "wall_rtf", "samples"}
    if field.endswith(".streaming"):
        required.add("ttfa_seconds")
    else:
        required.add("trial_wall_seconds")
    _require_exact_fields(measurement, required, field=field)
    for name in required - {"samples"}:
        if name == "trial_wall_seconds":
            continue
        _require_number(measurement[name], field=f"{field}.{name}")
    samples = measurement["samples"]
    if type(samples) is not int or samples <= 0:
        raise RecordValidationError(f"{field}.samples must be a positive integer")
    trials = measurement.get("trial_wall_seconds")
    if trials is not None:
        if not isinstance(trials, list) or not trials:
            raise RecordValidationError(f"{field}.trial_wall_seconds must be a non-empty array")
        for index, trial in enumerate(trials):
            _require_number(trial, field=f"{field}.trial_wall_seconds[{index}]")


def _validate_gpu_measurement(value: object, *, field: str, reliable: bool) -> None:
    gpu = _require_mapping(value, field=field)
    _require_exact_fields(
        gpu,
        {
            "baseline_mib",
            "peak_total_mib",
            "peak_delta_mib",
            "poll_interval_seconds",
            "sample_count",
            "expected_sample_count",
            "errors",
        },
        field=field,
    )
    for name in ("baseline_mib", "peak_total_mib", "peak_delta_mib", "poll_interval_seconds"):
        _require_number(gpu[name], field=f"{field}.{name}")
    for name in ("sample_count", "expected_sample_count"):
        if type(gpu[name]) is not int or gpu[name] <= 0:
            raise RecordValidationError(f"{field}.{name} must be a positive integer")
    errors = gpu["errors"]
    if not isinstance(errors, list) or any(not isinstance(error, str) or not error for error in errors):
        raise RecordValidationError(f"{field}.errors must be an array of non-empty strings")
    if reliable and (gpu["sample_count"] < gpu["expected_sample_count"] or errors):
        raise RecordValidationError(f"{field} is not reliable")


def _validate_stream_trial(value: object, *, field: str) -> None:
    trial = _require_mapping(value, field=field)
    _require_exact_fields(
        trial,
        {
            "trial",
            "status",
            "chunk_count",
            "samples",
            "audio_seconds",
            "ttfa_seconds",
            "wall_seconds",
            "failure",
        },
        field=field,
    )
    if type(trial["trial"]) is not int or trial["trial"] <= 0:
        raise RecordValidationError(f"{field}.trial must be a positive integer")
    if trial["status"] not in {"pass", "fail"}:
        raise RecordValidationError(f"{field}.status must be pass or fail")
    if trial["status"] == "pass":
        for name in ("chunk_count", "samples"):
            if type(trial[name]) is not int or trial[name] <= 0:
                raise RecordValidationError(f"{field}.{name} must be a positive integer")
        for name in ("audio_seconds", "ttfa_seconds", "wall_seconds"):
            _require_number(trial[name], field=f"{field}.{name}")
        if trial["failure"] is not None:
            raise RecordValidationError(f"{field}.failure must be null for a passing trial")
    else:
        for name in ("chunk_count", "samples"):
            if trial[name] is not None and (type(trial[name]) is not int or trial[name] < 0):
                raise RecordValidationError(f"{field}.{name} must be null or a non-negative integer")
        for name in ("audio_seconds", "ttfa_seconds", "wall_seconds"):
            if trial[name] is not None:
                _require_number(trial[name], field=f"{field}.{name}")
        failure = _require_mapping(trial["failure"], field=f"{field}.failure")
        _require_exact_fields(failure, {"type", "message"}, field=f"{field}.failure")
        _require_non_empty_string(failure["type"], field=f"{field}.failure.type")
        _require_non_empty_string(failure["message"], field=f"{field}.failure.message")


def _validate_full_pair_run_input(run_input: Mapping[str, Any], entry: CatalogEntry) -> None:
    _require_exact_fields(run_input, _FULL_PAIR_RUN_INPUT_FIELDS, field="evidence.run_input")
    for name in ("text", "language", "warm_up", "build_type"):
        _require_non_empty_string(run_input[name], field=f"evidence.run_input.{name}")
    if run_input["build_type"] != "Release":
        raise RecordValidationError("evidence.run_input.build_type must be Release")
    if type(run_input["trial_count"]) is not int or run_input["trial_count"] <= 0:
        raise RecordValidationError("evidence.run_input.trial_count must be a positive integer")
    if run_input["concurrency"] != 1:
        raise RecordValidationError("evidence.run_input.concurrency must be 1")
    options = _require_mapping(run_input["options"], field="evidence.run_input.options")
    _require_exact_fields(options, {"max_new_tokens", "seed"}, field="evidence.run_input.options")
    if type(options["max_new_tokens"]) is not int or options["max_new_tokens"] <= 0:
        raise RecordValidationError("evidence.run_input.options.max_new_tokens must be a positive integer")
    if type(options["seed"]) is not int:
        raise RecordValidationError("evidence.run_input.options.seed must be an integer")
    if entry.mode == "CustomVoice":
        _require_non_empty_string(run_input["speaker"], field="evidence.run_input.speaker")
        if run_input["reference"] is not None or run_input["reference_text_modes"] != []:
            raise RecordValidationError("CustomVoice run_input must not contain reference conditioning")
    else:
        if run_input["speaker"] is not None:
            raise RecordValidationError("Base run_input.speaker must be null")
        _require_non_empty_string(run_input["reference"], field="evidence.run_input.reference")
        if run_input["reference_text_modes"] != ["present", "absent"]:
            raise RecordValidationError("Base run_input.reference_text_modes must be present then absent")


def _validate_smoke_run_input(run_input: Mapping[str, Any]) -> None:
    _require_exact_fields(run_input, _ARCHITECTURE_SMOKE_RUN_INPUT_FIELDS, field="evidence.run_input")
    expected = {
        "text": ARCHITECTURE_SMOKE_TEXT,
        "language": "en",
        "speaker": "ryan",
        "options": {"seed": 7, "max_new_tokens": 64},
        "trial_count": 3,
        "concurrency": 1,
        "build_type": "Release",
    }
    if run_input != expected:
        differences = sorted(name for name in expected if run_input.get(name) != expected[name])
        raise RecordValidationError(f"architecture-smoke run_input mismatch: {differences}")


def _validate_full_pair_measurements(
    measurements: Mapping[str, Any], *, entry: CatalogEntry, passing: bool, checks: Mapping[str, Any]
) -> None:
    _require_exact_fields(measurements, PROFILE_MEASUREMENT_FIELDS["full-pair"], field="evidence.measurements")
    if passing or measurements["cold_load_seconds"] is not None:
        _require_number(measurements["cold_load_seconds"], field="evidence.measurements.cold_load_seconds")
    for name in ("warm_get", "streaming"):
        if passing or measurements[name] is not None:
            _validate_audio_measurement(measurements[name], field=f"evidence.measurements.{name}")
    if (passing or measurements["peak_rss_bytes"] is not None) and (
        type(measurements["peak_rss_bytes"]) is not int or measurements["peak_rss_bytes"] <= 0
    ):
        raise RecordValidationError("evidence.measurements.peak_rss_bytes must be a positive integer")
    if passing or measurements["gpu_memory"] is not None:
        _validate_gpu_measurement(
            measurements["gpu_memory"],
            field="evidence.measurements.gpu_memory",
            reliable=checks["gpu_memory_measurement"]["status"] == "pass",
        )
    conditioning = measurements["base_conditioning_seconds"]
    if entry.mode == "Base" and (passing or conditioning is not None):
        if not isinstance(conditioning, list) or not conditioning:
            raise RecordValidationError("evidence.measurements.base_conditioning_seconds must be non-empty for Base")
        for index, duration in enumerate(conditioning):
            _require_number(duration, field=f"evidence.measurements.base_conditioning_seconds[{index}]")
    elif entry.mode == "CustomVoice" and conditioning is not None:
        raise RecordValidationError("evidence.measurements.base_conditioning_seconds must be null for CustomVoice")


def _validate_smoke_measurements(measurements: Mapping[str, Any], *, passing: bool, checks: Mapping[str, Any]) -> None:
    _require_exact_fields(
        measurements,
        PROFILE_MEASUREMENT_FIELDS["architecture-smoke"],
        field="evidence.measurements",
    )
    if passing or measurements["warmup_seconds"] is not None:
        _require_number(measurements["warmup_seconds"], field="evidence.measurements.warmup_seconds")
    trials = measurements["stream_trials"]
    if not isinstance(trials, list):
        raise RecordValidationError("evidence.measurements.stream_trials must be an array")
    if passing and len(trials) != 3:
        raise RecordValidationError("evidence.measurements.stream_trials must contain three trials")
    if len(trials) > 3:
        raise RecordValidationError("evidence.measurements.stream_trials cannot contain more than three trials")
    for index, trial in enumerate(trials, start=1):
        _validate_stream_trial(trial, field=f"evidence.measurements.stream_trials[{index - 1}]")
        if cast(Mapping[str, Any], trial)["trial"] != index:
            raise RecordValidationError("evidence.measurements.stream_trials must be ordered 1, 2, 3")
    if passing and any(cast(Mapping[str, Any], trial)["status"] != "pass" for trial in trials):
        raise RecordValidationError("passing architecture-smoke record must contain three passing stream trials")
    if (passing or measurements["peak_rss_bytes"] is not None) and (
        type(measurements["peak_rss_bytes"]) is not int or measurements["peak_rss_bytes"] <= 0
    ):
        raise RecordValidationError("evidence.measurements.peak_rss_bytes must be a positive integer")
    if passing or measurements["gpu_memory"] is not None:
        _validate_gpu_measurement(
            measurements["gpu_memory"],
            field="evidence.measurements.gpu_memory",
            reliable=checks["gpu_memory_measurement"]["status"] == "pass",
        )


def validate_record(record: object) -> None:
    document = _require_mapping(record, field="record")
    _require_exact_fields(document, _TOP_LEVEL_FIELDS, field="record")
    if document["schema_version"] != 3:
        raise RecordValidationError("schema_version must be 3")
    profile = document["profile"]
    if profile not in PROFILE_CHECKS:
        raise RecordValidationError("profile must be full-pair or architecture-smoke")
    profile = cast(str, profile)
    executed_at = _require_non_empty_string(document["executed_at"], field="executed_at")
    try:
        parsed_time = datetime.fromisoformat(executed_at)
    except ValueError as exc:
        raise RecordValidationError("executed_at must be extended ISO 8601") from exc
    if parsed_time.utcoffset() is None:
        raise RecordValidationError("executed_at must include a timezone offset")

    model = _require_mapping(document["model"], field="model")
    _require_exact_fields(model, {"model_id", "mode", "size", "quantization", "talker", "codec"}, field="model")
    model_id = _require_non_empty_string(model["model_id"], field="model.model_id")
    entry = CATALOG_BY_ID.get(model_id)
    if entry is None or model != {
        "model_id": entry.model_id,
        "mode": entry.mode,
        "size": entry.size,
        "quantization": entry.quantization,
        "talker": entry.talker,
        "codec": entry.codec,
    }:
        raise RecordValidationError(f"model does not match the qualified catalog: {model_id}")
    if profile == "architecture-smoke" and model_id != ARCHITECTURE_SMOKE_MODEL_ID:
        raise RecordValidationError(f"architecture-smoke model must be {ARCHITECTURE_SMOKE_MODEL_ID}")

    evidence = _require_mapping(document["evidence"], field="evidence")
    _require_exact_fields(evidence, {"run_input", "checks", "measurements"}, field="evidence")
    required_checks = PROFILE_CHECKS[profile]
    checks = _require_mapping(evidence["checks"], field="evidence.checks")
    _require_exact_fields(checks, set(required_checks), field="evidence.checks")
    for name in required_checks:
        check = _require_mapping(checks[name], field=f"evidence.checks.{name}")
        _require_exact_fields(check, {"status", "detail"}, field=f"evidence.checks.{name}")
        if check["status"] not in {"pass", "fail", "not_run"}:
            raise RecordValidationError(f"evidence.checks.{name}.status must be pass, fail, or not_run")
        _require_non_empty_string(check["detail"], field=f"evidence.checks.{name}.detail")
    derived_status = "pass" if all(checks[name]["status"] == "pass" for name in required_checks) else "fail"
    if document["status"] != derived_status:
        raise RecordValidationError(f"status must be derived from required checks: expected={derived_status}")

    failure = document["failure"]
    if derived_status == "pass":
        if failure is not None:
            raise RecordValidationError("failure must be null for a passing record")
    else:
        failure_value = _require_mapping(failure, field="failure")
        _require_exact_fields(failure_value, {"check", "type", "message"}, field="failure")
        failure_check = _require_non_empty_string(failure_value["check"], field="failure.check")
        if failure_check not in checks or checks[failure_check]["status"] != "fail":
            raise RecordValidationError("failure.check must identify a failed required check")
        _require_non_empty_string(failure_value["type"], field="failure.type")
        _require_non_empty_string(failure_value["message"], field="failure.message")

    environment = _require_mapping(document["environment"], field="environment")
    _require_exact_fields(environment, _ENVIRONMENT_FIELDS, field="environment")
    for name in _ENVIRONMENT_FIELDS:
        if name == "co_resident_gpu_note":
            if not isinstance(environment[name], str):
                raise RecordValidationError("environment.co_resident_gpu_note must be a string")
        elif name == "gpu_uuid":
            if environment[name] is not None and (not isinstance(environment[name], str) or not environment[name]):
                raise RecordValidationError("environment.gpu_uuid must be null or a non-empty string")
        elif name in {"physical_gpu_index", "gpu_count"}:
            minimum = 0 if name == "physical_gpu_index" else 1
            if type(environment[name]) is not int or environment[name] < minimum:
                raise RecordValidationError(f"environment.{name} must be an integer >= {minimum}")
        else:
            _require_non_empty_string(environment[name], field=f"environment.{name}")
    architecture = cast(str, environment["cuda_architecture"])
    capability = ARCHITECTURE_TO_CAPABILITY.get(architecture)
    if capability is None:
        raise RecordValidationError(f"environment.cuda_architecture is not qualified: {architecture!r}")
    if environment["gpu_compute_capability"] != capability:
        raise RecordValidationError(
            "environment architecture and compute capability mismatch: "
            f"architecture={architecture}, capability={environment['gpu_compute_capability']}"
        )
    if profile == "full-pair" and architecture != "sm_120a":
        raise RecordValidationError("full-pair records must use environment.cuda_architecture sm_120a")
    environment_failure = _qualified_environment_failure(environment)
    if environment_failure is not None and checks["host_boundary"]["status"] != "fail":
        raise RecordValidationError(environment_failure)

    artifact = _require_mapping(document["artifact"], field="artifact")
    _require_exact_fields(artifact, _ARTIFACT_FIELDS, field="artifact")
    for name in _ARTIFACT_FIELDS - {"build_metadata"}:
        _require_non_empty_string(artifact[name], field=f"artifact.{name}")
    if artifact["repository"] != CANONICAL_REPOSITORY:
        raise RecordValidationError(f"artifact.repository must be {CANONICAL_REPOSITORY}")
    import_path = Path(cast(str, artifact["module_import_path"]))
    expected_import_suffix = ("speechflow", "qwen3", "__init__.py")
    if (
        import_path.is_absolute()
        or not any(part in {"site-packages", "dist-packages"} for part in import_path.parts)
        or import_path.parts[-3:] != expected_import_suffix
    ):
        raise RecordValidationError(
            "artifact.module_import_path must be a venv-relative site-packages or dist-packages path"
        )
    for name in ("core_wheel", "module_wheel"):
        value = cast(str, artifact[name])
        if Path(value).name != value:
            raise RecordValidationError(f"artifact.{name} must be a wheel filename without a host path")
    try:
        _release_wheel_version(
            cast(str, artifact["release_tag"]),
            cast(str, artifact["module_wheel"]),
        )
    except _GateFailure as exc:
        raise RecordValidationError(f"artifact.{exc}") from exc
    for name in ("module_wheel_sha256", "core_wheel_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", cast(str, artifact[name])) is None:
            raise RecordValidationError(f"artifact.{name} must be a lowercase SHA-256 digest")
    metadata = _require_mapping(artifact["build_metadata"], field="artifact.build_metadata")
    _require_exact_fields(metadata, _BUILD_METADATA_FIELDS, field="artifact.build_metadata")
    expected_metadata = {
        "schema_version": 1,
        "build_type": "Release",
        "cpu_baseline": "x86-64-v3",
        "cmake_cuda_architectures": EXPECTED_CMAKE_CUDA_ARCHITECTURES,
    }
    for name, expected in expected_metadata.items():
        if metadata[name] != expected:
            raise RecordValidationError(f"artifact.build_metadata.{name} must be {expected!r}")
    cuda_toolkit_version = _require_non_empty_string(
        metadata["cuda_toolkit_version"], field="artifact.build_metadata.cuda_toolkit_version"
    )
    if not cuda_toolkit_version.startswith("13.2."):
        raise RecordValidationError("artifact.build_metadata.cuda_toolkit_version must be in the 13.2 series")
    if metadata["qwentts_revision"] != environment["qwentts_revision"]:
        raise RecordValidationError("artifact.build_metadata.qwentts_revision must match environment.qwentts_revision")
    _require_non_empty_string(metadata["ggml_revision"], field="artifact.build_metadata.ggml_revision")
    run_input = _require_mapping(evidence["run_input"], field="evidence.run_input")
    measurements = _require_mapping(evidence["measurements"], field="evidence.measurements")
    if profile == "full-pair":
        _validate_full_pair_run_input(run_input, entry)
        _validate_full_pair_measurements(measurements, entry=entry, passing=derived_status == "pass", checks=checks)
    else:
        _validate_smoke_run_input(run_input)
        _validate_smoke_measurements(measurements, passing=derived_status == "pass", checks=checks)

    try:
        encoded = json.dumps(document, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RecordValidationError(f"record must contain portable JSON values: {exc}") from exc
    del encoded
    _reject_machine_local_record_paths(document)


def _format_measurement(value: object) -> str:
    return "—" if value is None else f"{float(cast(float, value)):.3f}"


def _record_identity(record: Mapping[str, Any]) -> tuple[str, str]:
    model = cast(Mapping[str, str], record["model"])
    environment = cast(Mapping[str, str], record["environment"])
    return model["model_id"], environment["cuda_architecture"]


def _record_link(record: Mapping[str, Any], *, from_readme: bool) -> str:
    model_id, architecture = _record_identity(record)
    return f"[{model_id}--{architecture}.json{_record_link_target(record, from_readme=from_readme)}"


def _record_link_target(record: Mapping[str, Any], *, from_readme: bool) -> str:
    model_id, architecture = _record_identity(record)
    prefix = (
        f"https://github.com/{CANONICAL_REPOSITORY}/blob/main/docs/qualified-records"
        if from_readme
        else "qualified-records"
    )
    return f"]({prefix}/{model_id}--{architecture}.json)"


def _documentation_states(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, str], dict[tuple[str, str], Mapping[str, Any]]]:
    release_tags: set[str] = set()
    for record in records:
        validate_record(record)
        release_tags.add(cast(str, record["artifact"]["release_tag"]))
    if len(release_tags) > 1:
        raise DocumentationContractError(
            "adopted qualification records must use the same release; "
            f"found release tags: {', '.join(sorted(release_tags))}"
        )

    by_identity: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in records:
        identity = _record_identity(record)
        if identity in by_identity:
            raise DocumentationContractError(f"duplicate qualified record identity: {identity}")
        profile = cast(str, record["profile"])
        if profile == "full-pair" and identity[1] != "sm_120a":
            raise DocumentationContractError("full-pair records must use sm_120a")
        if profile == "architecture-smoke" and identity[1] not in {"sm_86", "sm_89"}:
            raise DocumentationContractError("adopted architecture-smoke records must use sm_86 or sm_89")
        by_identity[identity] = record

    architecture_states = dict.fromkeys(ARCHITECTURE_TO_CAPABILITY, "pending")
    for architecture in ("sm_86", "sm_89"):
        architecture_record = by_identity.get((ARCHITECTURE_SMOKE_MODEL_ID, architecture))
        if architecture_record is not None:
            architecture_states[architecture] = "qualified" if architecture_record["status"] == "pass" else "failed"
    sm120a = [
        record
        for (model_id, architecture), record in by_identity.items()
        if architecture == "sm_120a" and model_id in CATALOG_BY_ID
    ]
    if any(record["status"] == "fail" for record in sm120a):
        architecture_states["sm_120a"] = "failed"
    elif len(sm120a) == len(CATALOG) and all(record["status"] == "pass" for record in sm120a):
        architecture_states["sm_120a"] = "qualified"

    pair_states: dict[str, str] = {}
    for entry in CATALOG:
        pair_record = by_identity.get((entry.model_id, "sm_120a"))
        pair_states[entry.model_id] = (
            "pending" if pair_record is None else "qualified" if pair_record["status"] == "pass" else "failed"
        )
    return architecture_states, pair_states, by_identity


def render_evidence(records: Sequence[Mapping[str, Any]]) -> str:
    architecture_states, pair_states, by_identity = _documentation_states(records)
    gpu_names = {
        "sm_86": "NVIDIA RTX A6000",
        "sm_89": "NVIDIA RTX 6000 Ada",
        "sm_120a": "NVIDIA RTX PRO 4000 Blackwell",
    }
    lines = [
        "# Qualified GPU evidence",
        "",
        "この文書は `docs/qualified-records/*.json` の schema-v3 record と一致する公開状態です。",
        "native target の同梱と実機 runtime evidence を区別し、性能値に合否閾値を設けません。",
        "",
        "## Architecture evidence",
        "",
        "| Architecture | Compute capability | Representative GPU | Native target | Runtime evidence | Record |",
        "|---|---:|---|---|---|---|",
    ]
    for architecture, capability in ARCHITECTURE_TO_CAPABILITY.items():
        state = architecture_states[architecture]
        architecture_records = [
            record for (_, record_architecture), record in by_identity.items() if record_architecture == architecture
        ]
        links = ", ".join(_record_link(record, from_readme=False) for record in architecture_records) or "—"
        lines.append(f"| `{architecture}` | {capability} | {gpu_names[architecture]} | included | {state} | {links} |")

    lines.extend(
        [
            "",
            "## Model-pair results",
            "",
            "| Model identifier | Gate | Executed at | Architecture | Record | Warm get (s) | "
            "Streaming TTFA (s) | Peak RSS (bytes) | Peak VRAM delta (MiB) |",
            "|---|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for entry in CATALOG:
        record = by_identity.get((entry.model_id, "sm_120a"))
        state = pair_states[entry.model_id]
        executed_at: object = "—"
        link = "—"
        warm_get: object = None
        ttfa: object = None
        peak_rss: object = None
        gpu_delta: object = None
        if record is not None:
            executed_at = record["executed_at"]
            link = _record_link(record, from_readme=False)
            if entry.model_id in MANDATORY_MEASUREMENT_MODELS:
                evidence = cast(Mapping[str, Any], record["evidence"])
                measurements = cast(Mapping[str, Any], evidence["measurements"])
                warm = measurements.get("warm_get")
                stream = measurements.get("streaming")
                gpu = measurements.get("gpu_memory")
                warm_get = warm.get("wall_seconds") if isinstance(warm, Mapping) else None
                ttfa = stream.get("ttfa_seconds") if isinstance(stream, Mapping) else None
                peak_rss = measurements.get("peak_rss_bytes")
                gpu_delta = gpu.get("peak_delta_mib") if isinstance(gpu, Mapping) else None
        lines.append(
            f"| `{entry.model_id}` | {state} | {executed_at} | sm_120a | {link} | "
            f"{_format_measurement(warm_get)} | {_format_measurement(ttfa)} | {peak_rss or '—'} | "
            f"{_format_measurement(gpu_delta)} |"
        )
    lines.extend(
        [
            "",
            "`pending` は record 未採用、`qualified` は pass record 採用、`failed` は fail record 採用を示します。",
            "fail record を `qualified` または `supported` と表示しません。",
            "",
        ]
    )
    return "\n".join(lines)


def _find_table_row(text: str, identity: str) -> str:
    rows = [line for line in text.splitlines() if line.startswith("|") and f"`{identity}`" in line]
    if len(rows) != 1:
        raise DocumentationContractError(f"expected one table row for {identity}, got {len(rows)}")
    return rows[0]


def assert_documentation_consistent(readme: Path, evidence: Path, record_paths: Sequence[Path]) -> None:
    if not readme.is_file():
        raise DocumentationContractError(f"README is missing: {readme}")
    if not evidence.is_file():
        raise DocumentationContractError(f"qualified evidence is missing: {evidence}")
    records: list[Mapping[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for path in record_paths:
        try:
            record_text = path.read_text(encoding="utf-8")
            _reject_machine_local_paths(record_text, field=str(path))
            record = json.loads(record_text)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DocumentationContractError(f"qualified record cannot be read: {path}: {exc}") from exc
        validate_record(record)
        identity = _record_identity(record)
        if path.name != f"{identity[0]}--{identity[1]}.json" or identity in identities:
            raise DocumentationContractError(f"qualified record filename or identity mismatch: {path}")
        identities.add(identity)
        records.append(record)

    readme_text = readme.read_text(encoding="utf-8")
    evidence_text = evidence.read_text(encoding="utf-8")
    _reject_document_machine_local_paths(readme_text, field=str(readme))
    _reject_document_machine_local_paths(evidence_text, field=str(evidence))
    expected_evidence = render_evidence(records)
    if evidence_text != expected_evidence:
        raise DocumentationContractError("qualified evidence must equal the generated content from adopted records")
    architecture_states, pair_states, by_identity = _documentation_states(records)
    for architecture, state in architecture_states.items():
        row = _find_table_row(readme_text, architecture)
        if f"| {state} |" not in row:
            raise DocumentationContractError(
                f"README architecture {architecture} must show {state}; failed evidence remains failed"
            )
        architecture_records = [
            record for (_, record_architecture), record in by_identity.items() if record_architecture == architecture
        ]
        for record in architecture_records:
            link = _record_link_target(record, from_readme=True)
            if link not in row:
                raise DocumentationContractError(f"README architecture {architecture} must link {link}")
        if not architecture_records and "| — |" not in row:
            raise DocumentationContractError(f"README pending architecture {architecture} must not link a record")
    for model_id, state in pair_states.items():
        row = _find_table_row(readme_text, model_id)
        if f"| {state} |" not in row:
            raise DocumentationContractError(f"README model pair {model_id} must show {state}")
        record = by_identity.get((model_id, "sm_120a"))
        if record is not None and _record_link_target(record, from_readme=True) not in row:
            raise DocumentationContractError(f"README model pair {model_id} must link its sm_120a record")
        if record is None and "| — |" not in row:
            raise DocumentationContractError(f"README pending model pair {model_id} must not link a record")


def _reject_machine_local_paths(text: str, *, field: str) -> None:
    match = _MACHINE_LOCAL_ABSOLUTE_PATH.search(text)
    if match is not None:
        raise DocumentationContractError(f"{field} contains machine-local absolute path {match.group(0)!r}")


def _reject_document_machine_local_paths(text: str, *, field: str) -> None:
    for fragment in _DOCUMENT_MACHINE_LOCAL_PATH_FRAGMENTS:
        if fragment in text:
            raise DocumentationContractError(f"{field} contains machine-local path fragment {fragment!r}")


def _empty_checks() -> dict[str, dict[str, str]]:
    return {name: {"status": "not_run", "detail": "not reached"} for name in REQUIRED_CHECKS}


def _empty_measurements() -> dict[str, Any]:
    return {
        "cold_load_seconds": None,
        "warm_get": None,
        "streaming": None,
        "peak_rss_bytes": None,
        "gpu_memory": None,
        "base_conditioning_seconds": None,
    }


class _GateFailure(RuntimeError):
    pass


class _CountingBinding:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.load_calls = 0
        self.cancel_observations = 0

    def load_model(self, talker_path: str, codec_path: str, *, codec_chunk_sec: float | None = None) -> object:
        self.load_calls += 1
        return cast(object, self.inner.load_model(talker_path, codec_path, codec_chunk_sec=codec_chunk_sec))

    def free_context(self, context: object) -> None:
        self.inner.free_context(context)

    def list_speakers(self, context: object) -> tuple[str, ...]:
        return cast(tuple[str, ...], self.inner.list_speakers(context))

    def synthesize(self, context: object, request: object, *, on_chunk: Any, cancelled: Any) -> Any:
        def observed_cancelled() -> bool:
            value = bool(cancelled())
            if value:
                self.cancel_observations += 1
            return value

        return self.inner.synthesize(context, request, on_chunk=on_chunk, cancelled=observed_cancelled)

    def conditioning_seconds(self, context: object, samples: np.ndarray, *, trials: int) -> list[float]:
        durations: list[float] = []
        for _ in range(trials):
            started = time.perf_counter()
            self.inner.extract_voice_ref(context, samples)
            durations.append(time.perf_counter() - started)
        return durations


_SAMPLER_STOP_TIMEOUT_SECONDS = 30.0


class _GpuMemorySampler:
    def __init__(self, interval: float, *, physical_gpu_index: int) -> None:
        self.interval = interval
        self.physical_gpu_index = physical_gpu_index
        self.baseline_mib = self._read_total_mib(physical_gpu_index)
        self.peak_total_mib = self.baseline_mib
        self.sample_count = 1
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None
        self._stopped_elapsed: float | None = None

    @staticmethod
    def _read_total_mib(physical_gpu_index: int) -> float:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(physical_gpu_index),
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        values = [float(line.strip()) for line in completed.stdout.splitlines() if line.strip()]
        if len(values) != 1:
            raise _GateFailure(f"qualified gate requires exactly one GPU, found {len(values)}")
        return values[0]

    def start(self) -> None:
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="qwen3-vram-sampler", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        # Every poll slot that fell due before the stop request is sampled, even when a
        # slow nvidia-smi call pushed the schedule behind. The elapsed time recorded at
        # exit therefore always agrees with the number of slots this thread sampled.
        started_at = self._started_at if self._started_at is not None else time.monotonic()
        next_sample_at = started_at + self.interval
        while True:
            now = time.monotonic()
            remaining = next_sample_at - now
            if remaining > 0 and self._stop.wait(remaining):
                self._stopped_elapsed = now - started_at
                return
            self._sample_once()
            next_sample_at += self.interval

    def _sample_once(self) -> None:
        try:
            value = self._read_total_mib(self.physical_gpu_index)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
            return
        self.sample_count += 1
        self.peak_total_mib = max(self.peak_total_mib, value)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=_SAMPLER_STOP_TIMEOUT_SECONDS)
            if self._thread.is_alive():
                self.errors.append(f"sampler thread did not stop within {_SAMPLER_STOP_TIMEOUT_SECONDS:g} s")
        if self._started_at is None:
            elapsed = 0.0
        elif self._stopped_elapsed is not None:
            elapsed = self._stopped_elapsed
        else:
            elapsed = max(0.0, time.monotonic() - self._started_at)
        self._sample_once()
        expected_sample_count = 2 + math.floor(elapsed / self.interval)
        return {
            "baseline_mib": self.baseline_mib,
            "peak_total_mib": self.peak_total_mib,
            "peak_delta_mib": max(0.0, self.peak_total_mib - self.baseline_mib),
            "poll_interval_seconds": self.interval,
            "sample_count": self.sample_count,
            "expected_sample_count": expected_sample_count,
            "errors": list(self.errors),
        }


def _validate_gpu_memory_measurement(measurement: Mapping[str, Any]) -> None:
    sample_count = measurement.get("sample_count")
    expected_sample_count = measurement.get("expected_sample_count")
    errors = measurement.get("errors")
    if type(sample_count) is not int or type(expected_sample_count) is not int or not isinstance(errors, list):
        raise _GateFailure("GPU memory sampler returned malformed reliability metadata")
    if sample_count < expected_sample_count or errors:
        raise _GateFailure(
            "unreliable GPU memory sampling: "
            f"samples={sample_count}, expected>={expected_sample_count}; errors={len(errors)}; details={errors}"
        )


def _audio_summary(audio: object) -> dict[str, float | int]:
    if not isinstance(audio, AudioData):
        raise _GateFailure(f"expected AudioData, got {type(audio).__name__}")
    if audio.sample_rate != 24_000 or audio.channels != 1 or audio.format != "pcm":
        raise _GateFailure(
            f"audio metadata mismatch: sample_rate={audio.sample_rate}, "
            f"channels={audio.channels}, format={audio.format}"
        )
    if audio.data.dtype != np.float32 or audio.data.ndim != 1 or not audio.data.flags.writeable:
        raise _GateFailure(
            f"audio array mismatch: dtype={audio.data.dtype}, shape={audio.data.shape}, "
            f"writeable={audio.data.flags.writeable}"
        )
    if audio.data.size == 0 or not np.isfinite(audio.data).all():
        raise _GateFailure("audio samples must be non-empty and finite")
    minimum = float(np.min(audio.data))
    maximum = float(np.max(audio.data))
    if minimum < -1.0 or maximum > 1.0:
        raise _GateFailure(f"audio samples outside [-1, 1]: min={minimum}, max={maximum}")
    return {
        "samples": int(audio.data.size),
        "duration_seconds": float(audio.data.size / 24_000),
        "minimum": minimum,
        "maximum": maximum,
    }


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if sys.platform.startswith("linux") else value


def _wheel_distribution_version(module_wheel: str | Path) -> tuple[str, str]:
    wheel_name = Path(module_wheel).name
    if not wheel_name.endswith(".whl"):
        raise _GateFailure(f"module wheel must be a wheel filename: {wheel_name!r}")
    parts = wheel_name[:-4].split("-")
    if len(parts) < 5:
        raise _GateFailure(f"module wheel filename is malformed: {wheel_name!r}")
    distribution = re.sub(r"[-_.]+", "-", parts[0]).lower()
    if distribution != "speechflow-qwen3":
        raise _GateFailure(f"module wheel distribution must be speechflow-qwen3, got {parts[0]!r}")
    return wheel_name, parts[1]


def _release_wheel_version(release_tag: str, module_wheel: str | Path) -> str:
    match = _RELEASE_TAG_PATTERN.fullmatch(release_tag)
    if match is None:
        raise _GateFailure(
            f"release_tag must be in vX.Y.Z form: tag={release_tag!r}; module_wheel={Path(module_wheel).name!r}"
        )
    version = match.group("version")
    wheel_name, wheel_version = _wheel_distribution_version(module_wheel)
    if wheel_version != version:
        raise _GateFailure(f"release_tag must match module_wheel version: tag={release_tag!r}; wheel={wheel_name!r}")
    expected_wheel = f"speechflow_qwen3-{version}-py3-none-manylinux_2_38_x86_64.whl"
    if wheel_name != expected_wheel:
        raise _GateFailure(f"module_wheel must be {expected_wheel!r}; got {wheel_name!r}")
    return version


def _validate_installed_wheel_identity(
    *,
    origin: Path,
    prefix: Path,
    distribution_name: str,
    installed_distribution_name: str,
    distribution_version: str,
    wheel: Path,
    wheel_sha256: str,
    direct_url: Mapping[str, Any] | None,
    installed_files: Sequence[str] | None = None,
) -> InstalledWheelIdentity:
    resolved_origin = origin.resolve()
    resolved_prefix = prefix.resolve()
    try:
        relative_origin = resolved_origin.relative_to(resolved_prefix)
    except ValueError as exc:
        raise _GateFailure(
            f"{distribution_name} import origin must be beneath the active interpreter prefix; "
            f"origin={resolved_origin}; prefix={resolved_prefix}"
        ) from exc
    if not any(part in {"site-packages", "dist-packages"} for part in relative_origin.parts):
        raise _GateFailure(
            f"{distribution_name} import origin must be beneath the active interpreter prefix in site-packages; "
            f"relative_origin={relative_origin}"
        )

    wheel_name = wheel.name
    if not wheel_name.endswith(".whl"):
        raise _GateFailure(f"supplied {distribution_name} artifact is not a wheel: {wheel_name!r}")
    wheel_parts = wheel_name[:-4].split("-")
    if len(wheel_parts) < 5:
        raise _GateFailure(f"supplied {distribution_name} wheel filename is malformed: {wheel_name!r}")
    normalized_distribution = re.sub(r"[-_.]+", "-", distribution_name).lower()
    normalized_installed_distribution = re.sub(r"[-_.]+", "-", installed_distribution_name).lower()
    if normalized_installed_distribution != normalized_distribution:
        raise _GateFailure(
            f"installed distribution name mismatch: expected={distribution_name!r}; "
            f"actual={installed_distribution_name!r}"
        )
    wheel_distribution = re.sub(r"[-_.]+", "-", wheel_parts[0]).lower()
    if wheel_distribution != normalized_distribution:
        raise _GateFailure(
            f"supplied wheel distribution mismatch: expected={distribution_name!r}; actual={wheel_parts[0]!r}"
        )
    expected_version = wheel_parts[1]
    if distribution_version != expected_version:
        raise _GateFailure(
            f"installed {distribution_name} distribution version does not match the supplied wheel filename: "
            f"installed={distribution_version!r}; wheel={wheel_name!r}; expected={expected_version!r}"
        )
    expected_import_suffixes = {
        "speechflow-qwen3": ("speechflow", "qwen3", "__init__.py"),
        "speechflow": ("speechflow", "__init__.py"),
    }
    expected_import_suffix = expected_import_suffixes.get(normalized_distribution)
    if (
        expected_import_suffix is None
        or relative_origin.parts[-len(expected_import_suffix) :] != expected_import_suffix
    ):
        raise _GateFailure(
            f"installed {distribution_name} import origin does not match its owned package: {relative_origin}"
        )
    if not wheel.is_file():
        raise _GateFailure(f"supplied {distribution_name} wheel file does not exist: {wheel}")
    actual_wheel_sha256 = _sha256_file(wheel)
    if re.fullmatch(r"[0-9a-f]{64}", wheel_sha256) is None or actual_wheel_sha256 != wheel_sha256:
        raise _GateFailure(
            f"supplied {distribution_name} wheel SHA-256 mismatch: "
            f"expected={wheel_sha256}; actual={actual_wheel_sha256}"
        )
    if direct_url is None:
        raise _GateFailure(f"installed {distribution_name} distribution has no direct_url.json archive provenance")
    directory = direct_url.get("dir_info")
    if isinstance(directory, Mapping) and directory.get("editable") is True:
        raise _GateFailure(
            f"installed {distribution_name} distribution is editable; an isolated wheel install is required"
        )
    raw_url = direct_url.get("url")
    if not isinstance(raw_url, str) or not raw_url:
        raise _GateFailure(f"installed {distribution_name} direct_url.json has no archive URL")
    parsed_url = urlsplit(raw_url)
    installed_wheel_name = Path(unquote(parsed_url.path)).name
    if installed_wheel_name != wheel_name:
        raise _GateFailure(
            f"installed {distribution_name} wheel source filename does not match the supplied wheel: "
            f"installed={installed_wheel_name!r}; supplied={wheel_name!r}"
        )
    archive_info = direct_url.get("archive_info")
    if not isinstance(archive_info, Mapping):
        raise _GateFailure(f"installed {distribution_name} direct_url.json has no archive_info")
    provenance_digests: list[str] = []
    hashes = archive_info.get("hashes")
    if isinstance(hashes, Mapping) and isinstance(hashes.get("sha256"), str):
        provenance_digests.append(cast(str, hashes["sha256"]))
    legacy_hash = archive_info.get("hash")
    if isinstance(legacy_hash, str) and legacy_hash.startswith("sha256="):
        provenance_digests.append(legacy_hash.removeprefix("sha256="))
    if provenance_digests and any(digest != wheel_sha256 for digest in provenance_digests):
        raise _GateFailure(
            f"installed {distribution_name} archive SHA-256 digest does not match the supplied wheel: "
            f"installed={provenance_digests}; supplied={wheel_sha256}"
        )
    if not provenance_digests:
        if (
            parsed_url.scheme != "file"
            or parsed_url.netloc
            or Path(unquote(parsed_url.path)).resolve() != wheel.resolve()
        ):
            raise _GateFailure(
                f"installed {distribution_name} archive provenance has no SHA-256 hash and its file URL "
                "does not identify the supplied wheel"
            )
        if installed_files is None:
            raise _GateFailure(
                f"installed {distribution_name} archive provenance has no SHA-256 hash or installed payload listing"
            )
    if installed_files is not None:
        _validate_installed_wheel_payload(
            wheel=wheel,
            prefix=resolved_prefix,
            relative_origin=relative_origin,
            installed_files=installed_files,
            distribution_name=distribution_name,
        )
    return InstalledWheelIdentity(
        relative_origin=relative_origin,
        distribution_version=distribution_version,
        module_wheel=wheel_name,
    )


def _installed_relative_path(member: PurePosixPath) -> PurePosixPath:
    if member.is_absolute() or ".." in member.parts:
        raise _GateFailure(f"wheel contains an unsafe member path: {member}")
    if (
        len(member.parts) >= 3
        and member.parts[0].endswith(".data")
        and member.parts[1]
        in {
            "purelib",
            "platlib",
        }
    ):
        return PurePosixPath(*member.parts[2:])
    return member


def _is_installer_generated_path(path: str) -> bool:
    member = PurePosixPath(path)
    if member.name.endswith(".pyc") and "__pycache__" in member.parts:
        return True
    return bool(
        len(member.parts) >= 2
        and member.parts[0].endswith(".dist-info")
        and member.name in {"INSTALLER", "REQUESTED", "direct_url.json", "uv_cache.json"}
    )


def _validate_installed_wheel_payload(
    *,
    wheel: Path,
    prefix: Path,
    relative_origin: Path,
    installed_files: Sequence[str],
    distribution_name: str,
) -> None:
    origin_parts = relative_origin.parts
    try:
        site_index = next(
            index for index, part in enumerate(origin_parts) if part in {"site-packages", "dist-packages"}
        )
    except StopIteration as exc:
        raise _GateFailure(f"installed {distribution_name} origin has no site-packages component") from exc
    site_root = prefix.joinpath(*origin_parts[: site_index + 1])
    expected_files: set[str] = set()
    try:
        with zipfile.ZipFile(wheel) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                member = _installed_relative_path(PurePosixPath(info.filename))
                relative_name = member.as_posix()
                expected_files.add(relative_name)
                if member.name == "RECORD" and member.parent.name.endswith(".dist-info"):
                    continue
                installed_path = site_root.joinpath(*member.parts)
                if not installed_path.is_file():
                    raise _GateFailure(
                        f"installed {distribution_name} payload is missing supplied wheel member {relative_name!r}"
                    )
                archive_digest = hashlib.sha256()
                with archive.open(info) as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        archive_digest.update(chunk)
                installed_digest = _sha256_file(installed_path)
                if installed_digest != archive_digest.hexdigest():
                    raise _GateFailure(
                        f"installed {distribution_name} payload content differs from supplied wheel member "
                        f"{relative_name!r}"
                    )
    except (OSError, zipfile.BadZipFile) as exc:
        raise _GateFailure(f"supplied {distribution_name} wheel payload cannot be inspected: {wheel}: {exc}") from exc

    actual_files = {PurePosixPath(path).as_posix() for path in installed_files}
    missing = sorted(expected_files - actual_files)
    unexpected = sorted(path for path in actual_files - expected_files if not _is_installer_generated_path(path))
    if missing or unexpected:
        raise _GateFailure(
            f"installed {distribution_name} payload file set differs from the supplied wheel: "
            f"missing={missing}; unexpected={unexpected}"
        )


def _active_installed_wheel_identity(
    *,
    distribution_name: str,
    import_name: str,
    wheel: Path,
    wheel_sha256: str,
) -> InstalledWheelIdentity:
    module = importlib.import_module(import_name)
    raw = getattr(module, "__file__", None)
    if not isinstance(raw, str):
        raise _GateFailure(f"{import_name} does not have a filesystem import origin")
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise _GateFailure(f"active interpreter has no {distribution_name} distribution metadata") from exc
    direct_url_text = distribution.read_text("direct_url.json")
    direct_url: Mapping[str, Any] | None = None
    if direct_url_text is not None:
        try:
            parsed_direct_url = json.loads(direct_url_text)
        except json.JSONDecodeError as exc:
            raise _GateFailure(f"{distribution_name} direct_url.json is not valid JSON") from exc
        if not isinstance(parsed_direct_url, Mapping):
            raise _GateFailure(f"{distribution_name} direct_url.json must be a JSON object")
        direct_url = cast(Mapping[str, Any], parsed_direct_url)
    return _validate_installed_wheel_identity(
        origin=Path(raw),
        prefix=Path(sys.prefix),
        distribution_name=distribution_name,
        installed_distribution_name=distribution.metadata.get("Name", ""),
        distribution_version=distribution.version,
        wheel=wheel,
        wheel_sha256=wheel_sha256,
        direct_url=direct_url,
        installed_files=(None if distribution.files is None else [str(path) for path in distribution.files]),
    )


def _site_packages_origin(module_wheel: Path, module_wheel_sha256: str) -> InstalledWheelIdentity:
    return _active_installed_wheel_identity(
        distribution_name="speechflow-qwen3",
        import_name="speechflow.qwen3",
        wheel=module_wheel,
        wheel_sha256=module_wheel_sha256,
    )


def _core_site_packages_origin(core_wheel: Path, core_wheel_sha256: str) -> InstalledWheelIdentity:
    return _active_installed_wheel_identity(
        distribution_name="speechflow",
        import_name="speechflow",
        wheel=core_wheel,
        wheel_sha256=core_wheel_sha256,
    )


def _exercise_native_diagnostic(binding_module: Any, *, prefix: Path) -> str:
    missing_root = prefix / "__speechflow_qwen3_qualified_missing__"
    talker_path = missing_root / "talker.gguf"
    codec_path = missing_root / "codec.gguf"
    if talker_path.exists() or codec_path.exists():
        raise _GateFailure("controlled native diagnostic paths unexpectedly exist")
    binding = binding_module.CtypesNativeBinding()
    try:
        context = binding.load_model(str(talker_path), str(codec_path))
    except binding_module.NativeCallError as exc:
        message = str(exc)
        if "operation=qt_init;" not in message:
            raise _GateFailure(f"native diagnostic did not identify qt_init: {message}") from exc
        if "; status=" not in message:
            raise _GateFailure(f"native diagnostic did not include status: {message}") from exc
        diagnostic = message.partition("; diagnostic=")[2]
        if not diagnostic or diagnostic == "<empty>":
            raise _GateFailure(f"native diagnostic did not include qt_last_error detail: {message}") from exc
        status = message.partition("; status=")[2].partition(";")[0]
        return f"native operation=qt_init; status={status}; diagnostic was non-empty"
    except Exception as exc:
        raise _GateFailure(f"controlled qt_init failure leaked {type(exc).__name__}: {exc}") from exc
    else:
        try:
            binding.free_context(context)
        finally:
            raise _GateFailure("controlled qt_init with nonexistent model paths unexpectedly succeeded")


@dataclass(slots=True)
class NativeDriverConfig:
    weights_dir: Path
    text: str
    language: str
    speaker: str | None
    reference_audio_npy: Path | None
    reference_source: str | None
    reference_text: str | None
    reference_output_npy: Path | None
    max_new_tokens: int
    seed: int
    trial_count: int
    vram_poll_interval: float
    identity: InstalledWheelIdentity
    environment: Mapping[str, Any]
    physical_gpu_index: int


class NativePairDriver:
    def __init__(self, config: NativeDriverConfig) -> None:
        self.config = config

    def _run_input(self, entry: CatalogEntry, *, speaker: str | None = None) -> dict[str, Any]:
        is_base = entry.mode == "Base"
        return {
            "text": self.config.text,
            "language": self.config.language,
            "speaker": speaker if not is_base else None,
            "reference": self.config.reference_source if is_base else None,
            "reference_text_modes": ["present", "absent"] if is_base else [],
            "options": {"max_new_tokens": self.config.max_new_tokens, "seed": self.config.seed},
            "warm_up": "model load completed before measured synthesis",
            "trial_count": self.config.trial_count,
            "concurrency": 1,
            "build_type": "Release",
        }

    async def execute(self, entry: CatalogEntry) -> PairOutcome:
        checks = _empty_checks()
        measurements = _empty_measurements()
        run_input = self._run_input(entry, speaker=self.config.speaker)
        failure: dict[str, str] | None = None
        active_check = "isolated_import"
        engine: Any | None = None
        sampler: _GpuMemorySampler | None = None
        binding: _CountingBinding | None = None

        def passed(name: str, detail: str) -> None:
            checks[name] = {"status": "pass", "detail": detail}

        try:
            identity = self.config.identity
            passed(
                "isolated_import",
                f"module import={identity.relative_origin.as_posix()}; distribution=speechflow-qwen3 "
                f"{identity.distribution_version}; wheel={identity.module_wheel}; prefix=active runtime",
            )

            active_check = "host_boundary"
            _validate_qualified_environment(self.config.environment)
            passed(
                "host_boundary",
                "qualified host/build boundary matched Linux x86_64, "
                f"{self.config.environment['hardware']}, architecture "
                f"{self.config.environment['cuda_architecture']}, driver/CUDA/glibc minima, "
                "and wheel qwentts revision",
            )

            active_check = "manifest_load"
            manifest_module = importlib.import_module("speechflow.qwen3._manifest")
            manifest_pairs = {pair.model_id: pair for pair in manifest_module.load_manifest()}
            pair = manifest_pairs.get(entry.model_id)
            if pair is None or pair.talker.basename != entry.talker or pair.codec.basename != entry.codec:
                raise _GateFailure(f"wheel-owned manifest identity mismatch for {entry.model_id}")

            talker_path = (self.config.weights_dir / entry.talker).resolve()
            codec_path = (self.config.weights_dir / entry.codec).resolve()
            binding_module = importlib.import_module("speechflow.qwen3._binding")
            engine_module: Any = importlib.import_module("speechflow.qwen3._engine")

            active_check = "native_diagnostic"
            native_diagnostic = await asyncio.to_thread(
                _exercise_native_diagnostic,
                binding_module,
                prefix=Path(sys.prefix),
            )
            passed("native_diagnostic", native_diagnostic)

            active_check = "manifest_load"
            binding = _CountingBinding(binding_module.CtypesNativeBinding())
            previous_factory = engine_module.binding_factory
            engine_module.binding_factory = lambda: binding
            try:
                source = engine_module.Qwen3ModelSource(str(talker_path), str(codec_path))
                if entry.mode == "CustomVoice":
                    engine = engine_module.Qwen3CustomVoiceTTSEngine(source, default_voice=self.config.speaker)
                else:
                    engine = engine_module.Qwen3CloneTTSEngine(source)
            finally:
                engine_module.binding_factory = previous_factory
            passed("manifest_load", f"manifest pair resolved talker={talker_path.name}, codec={codec_path.name}")

            active_check = "gpu_memory_measurement"
            sampler = _GpuMemorySampler(
                self.config.vram_poll_interval,
                physical_gpu_index=self.config.physical_gpu_index,
            )
            sampler.start()
            active_check = "warmup"
            started = time.perf_counter()
            await engine.prepare()
            measurements["cold_load_seconds"] = time.perf_counter() - started
            if engine.model_identifier != entry.model_id or binding.load_calls != 1:
                raise _GateFailure(
                    f"loaded identity/count mismatch: model={engine.model_identifier}, load_calls={binding.load_calls}"
                )
            passed("warmup", f"cold load completed once as {engine.model_identifier}")

            active_check = "warmup_idempotent"
            await engine.prepare()
            if binding.load_calls != 1:
                raise _GateFailure(f"second prepare caused load_calls={binding.load_calls}")
            passed("warmup_idempotent", "second prepare completed without another native load")

            reference: AudioData | None = None
            speaker: str | None = None
            if entry.mode == "CustomVoice":
                active_check = "voice_contract"
                voices = engine.list_voices()
                names = [value["name"] for value in voices]
                if not names:
                    raise _GateFailure("CustomVoice list_voices returned no built-in speakers")
                requested = self.config.speaker
                speaker = (
                    names[0]
                    if requested is None
                    else next(
                        (name for name in names if name.casefold() == requested.casefold()),
                        None,
                    )
                )
                if speaker is None:
                    raise _GateFailure(f"requested built-in speaker is unavailable: {requested}; available={names}")
                run_input = self._run_input(entry, speaker=speaker)
                passed("voice_contract", f"selected built-in speaker={speaker}; available={names}")
            else:
                active_check = "voice_contract"
                reference = self._load_reference()
                summary = _audio_summary(reference)
                passed(
                    "voice_contract",
                    f"one-shot reference accepted with/without reference_text: samples={summary['samples']}",
                )

            active_check = "get_audio"
            get_durations: list[float] = []
            get_audio: AudioData | None = None
            for _ in range(self.config.trial_count):
                started = time.perf_counter()
                if entry.mode == "CustomVoice":
                    get_audio = await engine.get(
                        self.config.text,
                        voice=speaker,
                        language=self.config.language,
                        seed=self.config.seed,
                        max_new_tokens=self.config.max_new_tokens,
                    )
                else:
                    get_audio = await engine.get(
                        self.config.text,
                        reference_audio=reference,
                        reference_text=self.config.reference_text,
                        language=self.config.language,
                        seed=self.config.seed,
                        max_new_tokens=self.config.max_new_tokens,
                    )
                get_durations.append(time.perf_counter() - started)
            assert get_audio is not None
            get_summary = _audio_summary(get_audio)
            get_wall = sum(get_durations) / len(get_durations)
            measurements["warm_get"] = {
                "wall_seconds": get_wall,
                "audio_seconds": get_summary["duration_seconds"],
                "wall_rtf": get_wall / cast(float, get_summary["duration_seconds"]),
                "samples": get_summary["samples"],
                "trial_wall_seconds": get_durations,
            }
            passed("get_audio", f"get returned {get_summary['samples']} normalized samples")
            if entry.mode == "CustomVoice" and self.config.reference_output_npy is not None:
                self.config.reference_output_npy.parent.mkdir(parents=True, exist_ok=True)
                np.save(self.config.reference_output_npy, get_audio.data, allow_pickle=False)

            if entry.mode == "Base":
                assert reference is not None and binding is not None
                loaded = getattr(engine, "_loaded", None)
                if loaded is None:
                    raise _GateFailure("Base model context was not retained after prepare")
                measurements["base_conditioning_seconds"] = await asyncio.to_thread(
                    binding.conditioning_seconds,
                    loaded.context,
                    reference.data,
                    trials=self.config.trial_count,
                )

            active_check = "stream_first_chunk"
            stream_started = time.perf_counter()
            if entry.mode == "CustomVoice":
                stream = engine.stream(
                    self.config.text,
                    voice=speaker,
                    language=self.config.language,
                    seed=self.config.seed,
                    max_new_tokens=self.config.max_new_tokens,
                )
            else:
                stream = engine.stream(
                    self.config.text,
                    reference_audio=reference,
                    reference_text=None,
                    language=self.config.language,
                    seed=self.config.seed,
                    max_new_tokens=self.config.max_new_tokens,
                )
            first = await anext(stream)
            ttfa = time.perf_counter() - stream_started
            first_summary = _audio_summary(first)
            passed("stream_first_chunk", f"first chunk arrived with {first_summary['samples']} normalized samples")

            active_check = "stream_complete"
            chunks = [first]
            try:
                async for chunk in stream:
                    _audio_summary(chunk)
                    chunks.append(chunk)
            finally:
                stream_wall = time.perf_counter() - stream_started
                stream_samples = sum(chunk.data.size for chunk in chunks)
                stream_audio_seconds = stream_samples / 24_000
                measurements["streaming"] = {
                    "ttfa_seconds": ttfa,
                    "wall_seconds": stream_wall,
                    "audio_seconds": stream_audio_seconds,
                    "wall_rtf": stream_wall / stream_audio_seconds,
                    "samples": stream_samples,
                }
            passed("stream_complete", f"stream terminated normally after {len(chunks)} chunks")
            passed("audio_contract", "get and every stream chunk were 24 kHz mono writable finite float32 in [-1, 1]")

            active_check = "cancellation"
            before = binding.cancel_observations
            if entry.mode == "CustomVoice":
                early = engine.stream(
                    "This synthesis is intentionally closed after the first native chunk.",
                    voice=speaker,
                    language=self.config.language,
                    seed=self.config.seed + 1,
                    max_new_tokens=max(256, self.config.max_new_tokens),
                )
            else:
                early = engine.stream(
                    "This cloned synthesis is intentionally closed after the first native chunk.",
                    reference_audio=reference,
                    reference_text=self.config.reference_text,
                    language=self.config.language,
                    seed=self.config.seed + 1,
                    max_new_tokens=max(256, self.config.max_new_tokens),
                )
            _audio_summary(await anext(early))
            await early.aclose()
            try:
                await anext(early)
            except StopAsyncIteration:
                pass
            else:
                raise _GateFailure("stream yielded a chunk after early close")
            if binding.cancel_observations <= before:
                raise _GateFailure("native cancellation callback did not observe consumer early close")
            passed("cancellation", "consumer early close reached native cancellation and no later chunk was yielded")

            active_check = "configuration_diagnostic"
            try:
                if entry.mode == "CustomVoice":
                    await engine.get(self.config.text, voice=speaker, seed=True)
                else:
                    await engine.get(self.config.text, reference_audio=reference, seed=True)
            except ConfigurationError as exc:
                if "seed" not in str(exc):
                    raise _GateFailure(f"ConfigurationError did not identify seed: {exc}") from exc
                passed("configuration_diagnostic", f"invalid option identified by ConfigurationError: {exc}")
            else:
                raise _GateFailure("invalid seed option was accepted")

            active_check = "close"
            await engine.aclose()
            await engine.aclose()
            passed("close", "aclose completed twice without error")
            engine = None
        except Exception as exc:
            checks[active_check] = {"status": "fail", "detail": f"{type(exc).__name__}: {exc}"}
            failure = {"check": active_check, "type": type(exc).__name__, "message": str(exc)}
        finally:
            if engine is not None:
                try:
                    await engine.aclose()
                    if checks["close"]["status"] == "not_run":
                        passed("close", "aclose completed during failure cleanup")
                except Exception as close_error:
                    checks["close"] = {
                        "status": "fail",
                        "detail": f"{type(close_error).__name__}: {close_error}",
                    }
                    if failure is None:
                        failure = {"check": "close", "type": type(close_error).__name__, "message": str(close_error)}
            if sampler is not None:
                try:
                    gpu_memory = sampler.stop()
                    measurements["gpu_memory"] = gpu_memory
                    _validate_gpu_memory_measurement(gpu_memory)
                except Exception as sampler_error:
                    checks["gpu_memory_measurement"] = {
                        "status": "fail",
                        "detail": f"GPU sampler failure: {sampler_error}",
                    }
                    if failure is None:
                        failure = {
                            "check": "gpu_memory_measurement",
                            "type": type(sampler_error).__name__,
                            "message": str(sampler_error),
                        }
                else:
                    passed(
                        "gpu_memory_measurement",
                        f"VRAM polling samples={gpu_memory['sample_count']}, "
                        f"expected>={gpu_memory['expected_sample_count']}, errors=0",
                    )
            measurements["peak_rss_bytes"] = _rss_bytes()

        return PairOutcome(
            checks=checks,
            measurements=measurements,
            run_input=run_input,
            failure=failure,
        )

    def _load_reference(self) -> AudioData:
        path = self.config.reference_audio_npy
        if path is None:
            raise _GateFailure("Base qualified gate requires --reference-audio-npy")
        try:
            samples = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise _GateFailure(f"reference NumPy audio cannot be read: {path}: {exc}") from exc
        return AudioData(np.asarray(samples), sample_rate=24_000, channels=1, format="pcm")


@dataclass(slots=True)
class ArchitectureSmokeConfig:
    weights_dir: Path
    text: str
    language: str
    speaker: str
    max_new_tokens: int
    seed: int
    trial_count: int
    vram_poll_interval: float
    physical_gpu_index: int
    identity: InstalledWheelIdentity
    environment: Mapping[str, Any]
    artifact: Mapping[str, Any]


class ArchitectureSmokeDriver:
    def __init__(self, config: ArchitectureSmokeConfig) -> None:
        self.config = config

    async def execute(self, entry: CatalogEntry) -> PairOutcome:
        checks = {name: {"status": "not_run", "detail": "not reached"} for name in ARCHITECTURE_SMOKE_CHECKS}
        measurements: dict[str, Any] = {
            "warmup_seconds": None,
            "stream_trials": [],
            "peak_rss_bytes": None,
            "gpu_memory": None,
        }
        run_input = {
            "text": self.config.text,
            "language": self.config.language,
            "speaker": self.config.speaker,
            "options": {"seed": self.config.seed, "max_new_tokens": self.config.max_new_tokens},
            "trial_count": self.config.trial_count,
            "concurrency": 1,
            "build_type": "Release",
        }
        active_check = "release_identity"
        failure: dict[str, str] | None = None
        engine: Any | None = None
        sampler: _GpuMemorySampler | None = None

        def passed(name: str, detail: str) -> None:
            checks[name] = {"status": "pass", "detail": detail}

        try:
            artifact = self.config.artifact
            expected_version = _release_wheel_version(
                cast(str, artifact["release_tag"]), cast(str, artifact["module_wheel"])
            )
            if self.config.identity.distribution_version != expected_version:
                raise _GateFailure(
                    "release wheel and installed distribution version must match the release tag: "
                    f"wheel={expected_version}, installed={self.config.identity.distribution_version}, "
                    f"tag={artifact['release_tag']}"
                )
            passed(
                "release_identity",
                f"repository={artifact['repository']}; tag={artifact['release_tag']}; wheel={artifact['module_wheel']}",
            )

            active_check = "host_boundary"
            _validate_qualified_environment(self.config.environment)
            architecture = cast(str, self.config.environment["cuda_architecture"])
            capability = cast(str, self.config.environment["gpu_compute_capability"])
            if ARCHITECTURE_TO_CAPABILITY.get(architecture) != capability:
                raise _GateFailure(f"CUDA architecture/compute capability mismatch: {architecture}/{capability}")
            passed(
                "host_boundary",
                f"Linux x86_64 host matched {architecture}/CC {capability}, driver/CUDA/glibc minima, "
                "and wheel qwentts revision",
            )

            active_check = "selected_gpu"
            selector = os.environ.get("CUDA_VISIBLE_DEVICES")
            if selector != str(self.config.physical_gpu_index):
                raise _GateFailure(
                    "CUDA_VISIBLE_DEVICES must equal the physical GPU index: "
                    f"expected={self.config.physical_gpu_index}, actual={selector!r}"
                )
            passed(
                "selected_gpu",
                f"physical GPU {self.config.physical_gpu_index} selected by nvidia-smi -i and "
                "CUDA_VISIBLE_DEVICES; CUDA logical device is 0",
            )

            active_check = "wheel_digest"
            passed(
                "wheel_digest",
                f"module wheel SHA-256 matched SHA256SUMS: {artifact['module_wheel_sha256']}",
            )

            active_check = "warmup"
            manifest_module = importlib.import_module("speechflow.qwen3._manifest")
            manifest_pairs = {pair.model_id: pair for pair in manifest_module.load_manifest()}
            pair = manifest_pairs.get(entry.model_id)
            if pair is None or pair.talker.basename != entry.talker or pair.codec.basename != entry.codec:
                raise _GateFailure(f"wheel-owned manifest identity mismatch for {entry.model_id}")
            talker_path = (self.config.weights_dir / entry.talker).resolve()
            codec_path = (self.config.weights_dir / entry.codec).resolve()
            engine_module: Any = importlib.import_module("speechflow.qwen3._engine")
            source = engine_module.Qwen3ModelSource(str(talker_path), str(codec_path))
            engine = engine_module.Qwen3CustomVoiceTTSEngine(source, default_voice=self.config.speaker)
            active_check = "gpu_memory_measurement"
            sampler = _GpuMemorySampler(
                self.config.vram_poll_interval,
                physical_gpu_index=self.config.physical_gpu_index,
            )
            sampler.start()
            active_check = "warmup"
            started = time.perf_counter()
            await engine.prepare()
            measurements["warmup_seconds"] = time.perf_counter() - started
            passed("warmup", "prepare completed once")

            active_check = "stream_trials"
            voices = engine.list_voices()
            speaker = next(
                (
                    value["name"]
                    for value in voices
                    if cast(str, value["name"]).casefold() == self.config.speaker.casefold()
                ),
                None,
            )
            if speaker is None:
                raise _GateFailure(f"requested built-in speaker is unavailable: {self.config.speaker}")
            for trial_number in range(1, self.config.trial_count + 1):
                started = time.perf_counter()
                chunk_count = 0
                samples = 0
                ttfa: float | None = None
                try:
                    stream = engine.stream(
                        self.config.text,
                        voice=speaker,
                        language=self.config.language,
                        seed=self.config.seed,
                        max_new_tokens=self.config.max_new_tokens,
                    )
                    async for chunk in stream:
                        summary = _audio_summary(chunk)
                        if ttfa is None:
                            ttfa = time.perf_counter() - started
                        chunk_count += 1
                        samples += cast(int, summary["samples"])
                    if chunk_count == 0 or samples == 0 or ttfa is None:
                        raise _GateFailure("stream trial returned no audio chunks")
                except Exception as exc:
                    elapsed = time.perf_counter() - started
                    cast(list[dict[str, Any]], measurements["stream_trials"]).append(
                        {
                            "trial": trial_number,
                            "status": "fail",
                            "chunk_count": chunk_count,
                            "samples": samples,
                            "audio_seconds": samples / 24_000 if samples else None,
                            "ttfa_seconds": ttfa,
                            "wall_seconds": elapsed,
                            "failure": {"type": type(exc).__name__, "message": str(exc)},
                        }
                    )
                    raise
                elapsed = time.perf_counter() - started
                cast(list[dict[str, Any]], measurements["stream_trials"]).append(
                    {
                        "trial": trial_number,
                        "status": "pass",
                        "chunk_count": chunk_count,
                        "samples": samples,
                        "audio_seconds": samples / 24_000,
                        "ttfa_seconds": ttfa,
                        "wall_seconds": elapsed,
                        "failure": None,
                    }
                )
            passed("stream_trials", f"{self.config.trial_count} independent stream iterators completed")
            passed("audio_contract", "all chunks were writable finite mono float32 at 24 kHz in [-1, 1]")

            active_check = "close"
            await engine.aclose()
            await engine.aclose()
            passed("close", "aclose completed twice without error")
            engine = None
        except Exception as exc:
            checks[active_check] = {"status": "fail", "detail": f"{type(exc).__name__}: {exc}"}
            failure = {"check": active_check, "type": type(exc).__name__, "message": str(exc)}
        finally:
            if engine is not None:
                try:
                    await engine.aclose()
                    if checks["close"]["status"] == "not_run":
                        passed("close", "aclose completed during failure cleanup")
                except Exception as close_error:
                    checks["close"] = {"status": "fail", "detail": f"{type(close_error).__name__}: {close_error}"}
                    if failure is None:
                        failure = {"check": "close", "type": type(close_error).__name__, "message": str(close_error)}
            if sampler is not None:
                try:
                    gpu_memory = sampler.stop()
                    measurements["gpu_memory"] = gpu_memory
                    _validate_gpu_memory_measurement(gpu_memory)
                except Exception as sampler_error:
                    checks["gpu_memory_measurement"] = {
                        "status": "fail",
                        "detail": f"GPU sampler failure: {sampler_error}",
                    }
                    if failure is None:
                        failure = {
                            "check": "gpu_memory_measurement",
                            "type": type(sampler_error).__name__,
                            "message": str(sampler_error),
                        }
                else:
                    passed(
                        "gpu_memory_measurement",
                        f"VRAM polling samples={gpu_memory['sample_count']}, "
                        f"expected>={gpu_memory['expected_sample_count']}, errors=0",
                    )
            measurements["peak_rss_bytes"] = _rss_bytes()

        return PairOutcome(checks=checks, measurements=measurements, run_input=run_input, failure=failure)


def _nvidia_environment(physical_gpu_index: int) -> tuple[str, str, str, str | None, int]:
    listed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    gpu_count = len([line for line in listed.stdout.splitlines() if line.strip()])
    if gpu_count < 1:
        raise _GateFailure("nvidia-smi did not enumerate a GPU")
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(physical_gpu_index),
            "--query-gpu=name,driver_version,compute_cap,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise _GateFailure(f"physical GPU selector returned {len(rows)} rows for index {physical_gpu_index}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 4:
        raise _GateFailure(f"unexpected nvidia-smi environment output: {rows[0]!r}")
    uuid = fields[3] or None
    return fields[0], fields[1], fields[2], uuid, gpu_count


def _build_metadata() -> Mapping[str, Any]:
    module = importlib.import_module("speechflow.qwen3")
    raw = getattr(module, "__file__", None)
    if not isinstance(raw, str):
        raise _GateFailure("speechflow.qwen3 import has no file path")
    path = Path(raw).with_name("_native") / "BUILD-METADATA.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _GateFailure(f"installed native build metadata cannot be read: {path}: {exc}") from exc
    return _require_mapping(value, field="build metadata")


def collect_environment(
    *,
    cuda_architecture: str,
    physical_gpu_index: int,
    co_resident_gpu_note: str,
) -> dict[str, Any]:
    hardware, driver, capability, uuid, gpu_count = _nvidia_environment(physical_gpu_index)
    release = platform.freedesktop_os_release()
    os_name = (
        f"{platform.system()} {platform.machine()}; "
        f"{release.get('NAME', platform.system())} {release.get('VERSION_ID', platform.release())}"
    )
    if "microsoft" in platform.release().lower():
        os_name += " / WSL2"
    libc_name, libc_version = platform.libc_ver()
    metadata = _build_metadata()
    try:
        cuda_runtime = importlib.metadata.version("nvidia-cuda-runtime")
    except importlib.metadata.PackageNotFoundError as exc:
        raise _GateFailure("installed wheel environment lacks nvidia-cuda-runtime") from exc
    return {
        "hardware": hardware,
        "os": os_name,
        "glibc": libc_version if libc_name == "glibc" else f"{libc_name} {libc_version}",
        "python_version": platform.python_version(),
        "nvidia_driver": driver,
        "cuda_runtime": cuda_runtime,
        "gpu_compute_capability": capability,
        "cuda_architecture": cuda_architecture,
        "physical_gpu_index": physical_gpu_index,
        "gpu_uuid": uuid,
        "gpu_count": gpu_count,
        "qwentts_revision": str(metadata.get("qwentts_revision", "")),
        "co_resident_gpu_note": co_resident_gpu_note,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256sums(wheel: Path, sha256sums: Path) -> str:
    if not wheel.is_file():
        raise _GateFailure(f"wheel file does not exist: {wheel}")
    try:
        lines = sha256sums.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise _GateFailure(f"SHA256SUMS cannot be read: {sha256sums}: {exc}") from exc
    entries: list[tuple[str, str]] = []
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/]+)", line)
        if match is None:
            raise _GateFailure(f"SHA256SUMS contains an invalid entry: {line!r}")
        entries.append((match.group(2), match.group(1)))
    if len(entries) != 1:
        raise _GateFailure(f"SHA256SUMS must enumerate exactly one Release wheel, got {len(entries)} entries")
    matches = [digest for filename, digest in entries if filename == wheel.name]
    if len(matches) != 1:
        raise _GateFailure(f"SHA256SUMS must enumerate {wheel.name} exactly once")
    actual = _sha256_file(wheel)
    if matches[0] != actual:
        raise _GateFailure(f"SHA256SUMS digest mismatch for {wheel.name}: expected={matches[0]}, actual={actual}")
    return actual


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one speechflow-qwen3 qualified GPU pair from installed wheels")
    parser.add_argument("--profile", choices=tuple(PROFILE_CHECKS), required=True)
    parser.add_argument("--cuda-architecture", choices=tuple(ARCHITECTURE_TO_CAPABILITY), required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--model-id", choices=tuple(CATALOG_BY_ID), required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core-wheel", type=Path, required=True)
    parser.add_argument("--module-wheel", type=Path, required=True)
    parser.add_argument("--sha256sums", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--text", default=ARCHITECTURE_SMOKE_TEXT)
    parser.add_argument("--language", default="en")
    parser.add_argument("--speaker", default="ryan")
    parser.add_argument("--reference-audio-npy", type=Path)
    parser.add_argument("--reference-source")
    parser.add_argument("--reference-text", default="Reference audio for the qualified one-shot voice clone.")
    parser.add_argument("--write-reference-audio-npy", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trial-count", type=int, default=3)
    parser.add_argument("--vram-poll-interval", type=float, default=0.1)
    parser.add_argument("--co-resident-gpu-note", default="")
    args = parser.parse_args(argv)
    if args.trial_count <= 0:
        parser.error("--trial-count must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.vram_poll_interval <= 0:
        parser.error("--vram-poll-interval must be positive")
    if args.physical_gpu_index < 0:
        parser.error("--physical-gpu-index must be non-negative")
    try:
        _release_wheel_version(args.release_tag, args.module_wheel)
    except _GateFailure as exc:
        parser.error(str(exc))
    entry = CATALOG_BY_ID[args.model_id]
    if args.profile == "full-pair" and args.cuda_architecture != "sm_120a":
        parser.error("full-pair requires --cuda-architecture sm_120a")
    if args.profile == "full-pair" and entry.mode == "Base" and args.reference_audio_npy is None:
        parser.error("Base models require --reference-audio-npy")
    if args.profile == "architecture-smoke":
        expected = {
            "model_id": ARCHITECTURE_SMOKE_MODEL_ID,
            "text": ARCHITECTURE_SMOKE_TEXT,
            "language": "en",
            "speaker": "ryan",
            "seed": 7,
            "max_new_tokens": 64,
            "trial_count": 3,
        }
        mismatches = [name for name, value in expected.items() if getattr(args, name) != value]
        if mismatches:
            parser.error(f"architecture-smoke fixed input mismatch: {', '.join(mismatches)}")
    return args


def _execute_cli(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu_index)
    module_digest = _validate_sha256sums(args.module_wheel, args.sha256sums)
    if not args.core_wheel.is_file():
        raise _GateFailure(f"core wheel file does not exist: {args.core_wheel}")
    core_digest = _sha256_file(args.core_wheel)
    identity = _site_packages_origin(args.module_wheel, module_digest)
    _core_site_packages_origin(args.core_wheel, core_digest)
    build_metadata = dict(_build_metadata())
    expected_output = f"{args.model_id}--{args.cuda_architecture}.json"
    if args.output.name != expected_output:
        raise _GateFailure(f"record output filename must be {expected_output}")
    artifact = {
        "repository": CANONICAL_REPOSITORY,
        "release_tag": args.release_tag,
        "module_wheel": args.module_wheel.name,
        "module_wheel_sha256": module_digest,
        "core_wheel": args.core_wheel.name,
        "core_wheel_sha256": core_digest,
        "module_import_path": identity.relative_origin.as_posix(),
        "build_metadata": build_metadata,
    }
    environment = collect_environment(
        cuda_architecture=args.cuda_architecture,
        physical_gpu_index=args.physical_gpu_index,
        co_resident_gpu_note=cast(str, args.co_resident_gpu_note),
    )
    if args.profile == "full-pair":
        config = NativeDriverConfig(
            weights_dir=args.weights_dir,
            text=args.text,
            language=args.language,
            speaker=args.speaker,
            reference_audio_npy=args.reference_audio_npy,
            reference_source=args.reference_source,
            reference_text=args.reference_text,
            reference_output_npy=args.write_reference_audio_npy,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            trial_count=args.trial_count,
            vram_poll_interval=args.vram_poll_interval,
            identity=identity,
            environment=environment,
            physical_gpu_index=args.physical_gpu_index,
        )
        driver: PairDriver = NativePairDriver(config)
    else:
        smoke_config = ArchitectureSmokeConfig(
            weights_dir=args.weights_dir,
            text=args.text,
            language=args.language,
            speaker=args.speaker,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            trial_count=args.trial_count,
            vram_poll_interval=args.vram_poll_interval,
            physical_gpu_index=args.physical_gpu_index,
            identity=identity,
            environment=environment,
            artifact=artifact,
        )
        driver = ArchitectureSmokeDriver(smoke_config)
    executed_at = datetime.now().astimezone().isoformat(timespec="seconds")
    return asyncio.run(
        run_qualified_pair(
            args.model_id,
            profile=args.profile,
            driver=driver,
            environment=environment,
            artifact=artifact,
            executed_at=executed_at,
            destination=args.output,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if os.environ.get("SPEECHFLOW_QWEN3_RUN_GPU") != "1":
        print("qualified GPU gate is disabled; set SPEECHFLOW_QWEN3_RUN_GPU=1", file=sys.stderr)
        return 2
    try:
        record = _execute_cli(args)
    except Exception as exc:
        print(
            f"qualified GPU runner failed before it could write a record: {type(exc).__name__}: {exc}", file=sys.stderr
        )
        return 2
    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "profile": args.profile,
                "status": record["status"],
                "output": str(args.output),
            }
        )
    )
    return 0 if record["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
