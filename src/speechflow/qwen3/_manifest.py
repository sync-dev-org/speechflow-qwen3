from __future__ import annotations

import hashlib
import json
import re
import stat
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from speechflow.core import ConfigurationError, EnvironmentUnavailableError

_TOP_LEVEL_FIELDS = {"manifest_version", "models"}
_MODEL_FIELDS = {"model_id", "mode", "size", "quantization", "talker", "codec"}
_FILE_FIELDS = {"basename", "size_bytes", "sha256"}
_MODES = {"CustomVoice", "Base"}
_SIZES = {"0.6B", "1.7B"}
_QUANTIZATIONS = {"Q4_K_M", "Q8_0"}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_HASH_BLOCK_BYTES = 1024 * 1024


class _DuplicateField(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class WeightFile:
    basename: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ModelPair:
    model_id: str
    mode: str
    size: str
    quantization: str
    talker: WeightFile
    codec: WeightFile


def load_manifest() -> tuple[ModelPair, ...]:
    """Load and validate the package-owned weights catalog."""
    try:
        text = files("speechflow.qwen3").joinpath("weights-manifest-v1.json").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise _manifest_error("wheel-owned weights manifest cannot be read", cause=exc) from exc
    try:
        document = json.loads(text, object_pairs_hook=_unique_object)
    except (_DuplicateField, json.JSONDecodeError) as exc:
        raise _manifest_error("wheel-owned weights manifest is not strict JSON", cause=exc) from exc
    return _parse_document(document)


def validate_model_source(talker_path: str, codec_path: str, *, expected_mode: str) -> ModelPair:
    """Verify one talker/codec pair before it reaches the native runtime."""
    talker = Path(talker_path)
    codec = Path(codec_path)
    _require_regular_file(talker, label="talker_path")
    _require_regular_file(codec, label="codec_path")
    pairs = load_manifest()

    pair = next((item for item in pairs if item.talker.basename == talker.name), None)
    if pair is None:
        raise _configuration_error("talker basename is outside the wheel-owned catalog", f"talker_path={talker}")
    if codec.name != pair.codec.basename:
        if not any(item.codec.basename == codec.name for item in pairs):
            raise _configuration_error("codec basename is outside the wheel-owned catalog", f"codec_path={codec}")
        raise _configuration_error(
            "talker and codec do not identify one approved pair",
            f"talker_path={talker}, codec_path={codec}",
        )
    if pair.mode != expected_mode:
        alternative = "Qwen3CloneTTSEngine" if pair.mode == "Base" else "Qwen3CustomVoiceTTSEngine"
        raise _configuration_error(
            "engine class and weights mode do not match",
            f"class mode={expected_mode}, manifest mode={pair.mode}",
            how=f"use {alternative} for this model pair",
        )

    _verify_file(talker, pair.talker, label="talker_path")
    _verify_file(codec, pair.codec, label="codec_path")
    return pair


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateField(key)
        result[key] = value
    return result


def _parse_document(document: object) -> tuple[ModelPair, ...]:
    if not isinstance(document, dict) or set(document) != _TOP_LEVEL_FIELDS:
        raise _manifest_error("weights manifest must contain exactly manifest_version and models")
    if type(document["manifest_version"]) is not int or document["manifest_version"] != 1:
        raise _manifest_error("weights manifest_version must be the integer 1")
    models = document["models"]
    if not isinstance(models, list) or len(models) != 8:
        raise _manifest_error("weights manifest must contain exactly eight model pairs")

    parsed = tuple(_parse_pair(value) for value in models)
    model_ids = [pair.model_id for pair in parsed]
    talkers = [pair.talker.basename for pair in parsed]
    if len(set(model_ids)) != len(model_ids):
        raise _manifest_error("weights manifest contains a duplicate model identifier")
    if len(set(talkers)) != len(talkers):
        raise _manifest_error("weights manifest contains a duplicate talker basename")
    expected = {(mode, size, quantization) for mode in _MODES for size in _SIZES for quantization in _QUANTIZATIONS}
    if {(pair.mode, pair.size, pair.quantization) for pair in parsed} != expected:
        raise _manifest_error("weights manifest does not contain the supported model cross-product")
    return parsed


def _parse_pair(value: object) -> ModelPair:
    if not isinstance(value, dict) or set(value) != _MODEL_FIELDS:
        raise _manifest_error("each model pair must contain exactly the v1 fields")
    for field in ("model_id", "mode", "size", "quantization"):
        if type(value[field]) is not str or not value[field]:
            raise _manifest_error(f"model pair {field} must be a non-empty string")
    mode = value["mode"]
    size = value["size"]
    quantization = value["quantization"]
    if mode not in _MODES or size not in _SIZES or quantization not in _QUANTIZATIONS:
        raise _manifest_error("model pair mode, size, or quantization is unsupported")
    expected_id = f"qwen3-tts-{size.lower()}-{mode.lower()}-{quantization}"
    if value["model_id"] != expected_id:
        raise _manifest_error("model identifier does not match its pair fields")
    return ModelPair(
        model_id=value["model_id"],
        mode=mode,
        size=size,
        quantization=quantization,
        talker=_parse_weight_file(value["talker"], label="talker"),
        codec=_parse_weight_file(value["codec"], label="codec"),
    )


def _parse_weight_file(value: object, *, label: str) -> WeightFile:
    if not isinstance(value, dict) or set(value) != _FILE_FIELDS:
        raise _manifest_error(f"{label} entry must contain exactly the v1 file fields")
    basename = value["basename"]
    if type(basename) is not str or not basename or basename in {".", ".."}:
        raise _manifest_error(f"{label} basename must be a non-empty safe filename")
    if "/" in basename or "\\" in basename or "\x00" in basename:
        raise _manifest_error(f"{label} basename must be a safe filename")
    size_bytes = value["size_bytes"]
    if type(size_bytes) is not int or size_bytes <= 0:
        raise _manifest_error(f"{label} size_bytes must be a positive integer")
    sha256 = value["sha256"]
    if type(sha256) is not str or _SHA256_PATTERN.fullmatch(sha256) is None:
        raise _manifest_error(f"{label} SHA-256 must be 64 lowercase hexadecimal digits")
    return WeightFile(basename=basename, size_bytes=size_bytes, sha256=sha256)


def _require_regular_file(path: Path, *, label: str) -> None:
    if not path.is_absolute():
        raise _configuration_error("weights path must be absolute", f"{label}={path}")
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise _configuration_error(
            "weights path must identify an existing regular file", f"{label}={path}", cause=exc
        ) from exc
    if not stat.S_ISREG(mode):
        raise _configuration_error("weights path must identify an existing regular file", f"{label}={path}")


def _verify_file(path: Path, expected: WeightFile, *, label: str) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise _configuration_error("weights byte size cannot be read", f"{label}={path}", cause=exc) from exc
    if size != expected.size_bytes:
        raise _configuration_error(
            "weights byte size does not match the wheel-owned manifest",
            f"{label}={path}, expected={expected.size_bytes}, actual={size}",
        )
    if _sha256(path) != expected.sha256:
        raise _configuration_error("weights SHA-256 does not match the wheel-owned manifest", f"{label}={path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(_HASH_BLOCK_BYTES), b""):
                digest.update(block)
    except OSError as exc:
        raise _configuration_error("weights file cannot be hashed", f"path={path}", cause=exc) from exc
    return digest.hexdigest()


def _configuration_error(
    why: str,
    what: str,
    *,
    how: str = "use one approved absolute talker/codec pair from weights-manifest-v1.json",
    cause: Exception | None = None,
) -> ConfigurationError:
    error = ConfigurationError(f"why={why}; what={what}; how={how}")
    if cause is not None:
        error.__cause__ = cause
    return error


def _manifest_error(why: str, *, cause: Exception | None = None) -> EnvironmentUnavailableError:
    error = EnvironmentUnavailableError(
        f"why={why}; what=manifest; how=reinstall speechflow-qwen3 with its package-owned weights-manifest-v1.json"
    )
    if cause is not None:
        error.__cause__ = cause
    return error
