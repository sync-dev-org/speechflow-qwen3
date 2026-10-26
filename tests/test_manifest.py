from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import pytest
from speechflow.core import ConfigurationError, EnvironmentUnavailableError


def _manifest_module() -> Any:
    return importlib.import_module("speechflow.qwen3._manifest")


def _entry(module: Any, *, mode: str = "CustomVoice", payload: bytes = b"talker", codec: bytes = b"codec") -> Any:
    return module.ModelPair(
        model_id=f"qwen3-tts-0.6b-{mode.lower()}-Q4_K_M",
        mode=mode,
        size="0.6B",
        quantization="Q4_K_M",
        talker=module.WeightFile(
            basename="talker.gguf",
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        ),
        codec=module.WeightFile(
            basename="codec.gguf",
            size_bytes=len(codec),
            sha256=hashlib.sha256(codec).hexdigest(),
        ),
    )


@pytest.mark.req("SP-SFQWEN3-021")
def test_packaged_manifest_contains_the_exact_eight_model_pairs() -> None:
    """The packaged manifest identifies eight CustomVoice and Base weight pairs."""
    manifest = _manifest_module()

    pairs = manifest.load_manifest()

    assert len(pairs) == 8
    assert {(pair.mode, pair.size, pair.quantization) for pair in pairs} == {
        (mode, size, quantization)
        for mode in ("CustomVoice", "Base")
        for size in ("0.6B", "1.7B")
        for quantization in ("Q4_K_M", "Q8_0")
    }
    assert len({pair.model_id for pair in pairs}) == 8
    assert all(len(pair.talker.sha256) == 64 and pair.talker.sha256 == pair.talker.sha256.lower() for pair in pairs)
    assert all(len(pair.codec.sha256) == 64 and pair.codec.sha256 == pair.codec.sha256.lower() for pair in pairs)


@pytest.mark.req("SP-SFQWEN3-021")
def test_manifest_resource_is_strict_v1_json_and_has_no_override_surface() -> None:
    """The packaged weight manifest has no replacement API."""
    manifest = _manifest_module()
    resource = Path(manifest.__file__).with_name("weights-manifest-v1.json")
    document = json.loads(resource.read_text(encoding="utf-8"))

    assert document["manifest_version"] == 1
    assert len(document["models"]) == 8
    assert not hasattr(manifest, "set_manifest")


@pytest.mark.req("SP-SFQWEN3-021")
def test_valid_pair_is_hashed_in_streaming_blocks_and_identified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Qwen3 accepts a weight pair whose manifest identity and digest match."""
    manifest = _manifest_module()
    talker_payload = b"talker"
    codec_payload = b"codec"
    talker = tmp_path / "talker.gguf"
    codec = tmp_path / "codec.gguf"
    talker.write_bytes(talker_payload)
    codec.write_bytes(codec_payload)
    pair = _entry(manifest, payload=talker_payload, codec=codec_payload)
    monkeypatch.setattr(manifest, "load_manifest", lambda: (pair,))
    monkeypatch.setattr(Path, "read_bytes", lambda _self: (_ for _ in ()).throw(AssertionError("read_bytes used")))

    result = manifest.validate_model_source(str(talker), str(codec), expected_mode="CustomVoice")

    assert result == pair


@pytest.mark.req("SP-SFQWEN3-021")
@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing", "path"),
        ("non-regular", "regular"),
        ("wrong-basename", "basename"),
        ("wrong-size", "size"),
        ("wrong-digest", "SHA-256"),
        ("wrong-pair", "pair"),
        ("wrong-mode", "class|mode"),
    ],
)
def test_invalid_weight_identity_is_rejected_without_modifying_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    """Qwen3 rejects a missing or mismatched weight pair without changing input files."""
    manifest = _manifest_module()
    talker = tmp_path / "talker.gguf"
    codec = tmp_path / "codec.gguf"
    talker.write_bytes(b"talker")
    codec.write_bytes(b"codec")
    pair = _entry(manifest)
    other = _entry(manifest, mode="Base")
    expected_mode = "CustomVoice"
    if mutation == "missing":
        talker.unlink()
    elif mutation == "non-regular":
        talker.unlink()
        talker.mkdir()
    elif mutation == "wrong-basename":
        talker = tmp_path / "unknown.gguf"
        talker.write_bytes(b"talker")
    elif mutation == "wrong-size":
        talker.write_bytes(b"larger-talk")
    elif mutation == "wrong-digest":
        talker.write_bytes(b"xxxxxx")
    elif mutation == "wrong-pair":
        other_codec = tmp_path / "other-codec.gguf"
        other_codec.write_bytes(b"other")
        other = manifest.ModelPair(
            model_id=other.model_id,
            mode=other.mode,
            size=other.size,
            quantization=other.quantization,
            talker=other.talker,
            codec=manifest.WeightFile(
                basename="other-codec.gguf",
                size_bytes=5,
                sha256=hashlib.sha256(b"other").hexdigest(),
            ),
        )
        codec = other_codec
    else:
        expected_mode = "Base"
    monkeypatch.setattr(manifest, "load_manifest", lambda: (pair, other))
    before = {path: path.read_bytes() for path in (talker, codec) if path.is_file()}

    with pytest.raises(ConfigurationError, match=match):
        manifest.validate_model_source(str(talker), str(codec), expected_mode=expected_mode)

    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.req("SP-SFQWEN3-036")
@pytest.mark.req("SP-SF-060")
@pytest.mark.parametrize("contents", [None, "invalid JSON", '{"manifest_version": 1, "models": []}'])
def test_unreadable_or_invalid_package_manifest_reports_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, contents: str | None
) -> None:
    """A missing or corrupt package-owned manifest reports an unavailable native environment."""
    manifest = _manifest_module()
    if contents is not None:
        (tmp_path / "weights-manifest-v1.json").write_text(contents, encoding="utf-8")
    monkeypatch.setattr(manifest, "files", lambda _: tmp_path)
    with pytest.raises(EnvironmentUnavailableError, match="manifest"):
        manifest.load_manifest()
