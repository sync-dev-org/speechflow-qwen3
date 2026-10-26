from __future__ import annotations

import asyncio
import gc
import hashlib
import importlib
import inspect
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from speechflow.core import (
    AudioData,
    ConfigurationError,
    EmptyOutputError,
    EngineClosedError,
    EngineStateError,
    EnvironmentUnavailableError,
    InferenceError,
)


@dataclass
class FakeBinding:
    samples: np.ndarray = field(default_factory=lambda: np.array([-2.0, 0.25, np.nan, np.inf, 2.0], dtype=np.float64))
    stream_samples: list[np.ndarray] = field(
        default_factory=lambda: [np.array([0.1], dtype=np.float32), np.array([0.2, 0.3], dtype=np.float32)]
    )
    speakers: tuple[str, ...] = ("Ryan", "Vivian")
    load_errors: list[BaseException] = field(default_factory=list)
    synth_error: BaseException | None = None
    load_entered: threading.Event | None = None
    load_release: threading.Event | None = None
    synth_entered: threading.Event | None = None
    synth_release: threading.Event | None = None
    load_calls: int = 0
    free_calls: int = 0
    synth_calls: int = 0
    streaming_calls: list[bool] = field(default_factory=list)
    loaded_paths: list[tuple[str, str]] = field(default_factory=list)
    loaded_codec_chunk_sec: list[float | None] = field(default_factory=list)
    requests: list[Any] = field(default_factory=list)
    thread_ids: list[int] = field(default_factory=list)
    active_calls: int = 0
    max_active_calls: int = 0

    def load_model(self, talker_path: str, codec_path: str, *, codec_chunk_sec: float | None = None) -> object:
        assert Path(talker_path).is_absolute()
        assert Path(codec_path).is_absolute()
        self.load_calls += 1
        self.loaded_paths.append((talker_path, codec_path))
        self.loaded_codec_chunk_sec.append(codec_chunk_sec)
        self.thread_ids.append(threading.get_ident())
        if self.load_entered is not None:
            self.load_entered.set()
        if self.load_release is not None:
            self.load_release.wait(2)
        if self.load_errors:
            raise self.load_errors.pop(0)
        return object()

    def free_context(self, context: object) -> None:
        del context
        self.free_calls += 1
        self.thread_ids.append(threading.get_ident())

    def list_speakers(self, context: object) -> tuple[str, ...]:
        del context
        self.thread_ids.append(threading.get_ident())
        return self.speakers

    def synthesize(
        self,
        context: object,
        request: object,
        *,
        on_chunk: Callable[[object], bool] | None,
        cancelled: Callable[[], bool],
    ) -> object:
        del context
        self.synth_calls += 1
        self.streaming_calls.append(on_chunk is not None)
        self.requests.append(request)
        self.thread_ids.append(threading.get_ident())
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            if self.synth_entered is not None:
                self.synth_entered.set()
            if self.synth_release is not None:
                while not self.synth_release.wait(0.005):
                    if cancelled():
                        raise asyncio.CancelledError
            if self.synth_error is not None:
                raise self.synth_error
            module = importlib.import_module("speechflow.qwen3._binding")
            if on_chunk is not None:
                for values in self.stream_samples:
                    if cancelled() or not on_chunk(module.NativeAudio(values, 24000, 1)):
                        raise asyncio.CancelledError
                return None
            return module.NativeAudio(self.samples, 24000, 1)
        finally:
            self.active_calls -= 1


@pytest.fixture
def engine_setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Any, FakeBinding, Any]:
    engine = importlib.import_module("speechflow.qwen3._engine")
    manifest = importlib.import_module("speechflow.qwen3._manifest")
    binding = FakeBinding()
    pair = manifest.ModelPair(
        model_id="qwen3-tts-0.6b-customvoice-Q4_K_M",
        mode="CustomVoice",
        size="0.6B",
        quantization="Q4_K_M",
        talker=manifest.WeightFile("talker.gguf", 1, "0" * 64),
        codec=manifest.WeightFile("codec.gguf", 1, "0" * 64),
    )
    monkeypatch.setattr(engine, "binding_factory", lambda: binding)
    monkeypatch.setattr(engine, "validate_model_source", lambda *_args, **_kwargs: pair)
    source = engine.Qwen3ModelSource(str(tmp_path / "talker.gguf"), str(tmp_path / "codec.gguf"))
    return engine, binding, source


async def _prepare_for_synthesis(instance: Any, binding: FakeBinding) -> None:
    await instance.prepare()
    binding.requests.clear()
    binding.synth_calls = 0
    binding.streaming_calls.clear()


def _model_pair(
    manifest: Any,
    *,
    talker_name: str,
    talker_payload: bytes,
    codec_name: str,
    codec_payload: bytes,
    mode: str = "CustomVoice",
    talker_size: int | None = None,
    talker_digest: str | None = None,
) -> Any:
    return manifest.ModelPair(
        model_id=f"qwen3-tts-0.6b-{mode.lower()}-Q4_K_M",
        mode=mode,
        size="0.6B",
        quantization="Q4_K_M",
        talker=manifest.WeightFile(
            talker_name,
            len(talker_payload) if talker_size is None else talker_size,
            hashlib.sha256(talker_payload).hexdigest() if talker_digest is None else talker_digest,
        ),
        codec=manifest.WeightFile(
            codec_name,
            len(codec_payload),
            hashlib.sha256(codec_payload).hexdigest(),
        ),
    )


def _tuning_engine(
    engine_setup: tuple[Any, FakeBinding, Any],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    *,
    codec_chunk_sec: float | None = None,
) -> tuple[Any, FakeBinding, dict[str, AudioData]]:
    engine, binding, source = engine_setup
    if kind == "clone":
        manifest = importlib.import_module("speechflow.qwen3._manifest")
        pair = _model_pair(
            manifest,
            talker_name="talker.gguf",
            talker_payload=b"x",
            codec_name="codec.gguf",
            codec_payload=b"y",
            mode="Base",
        )
        monkeypatch.setattr(engine, "validate_model_source", lambda *_args, **_kwargs: pair)
        reference = AudioData(np.array([0.2], dtype=np.float32), 24000)
        return (
            engine.Qwen3CloneTTSEngine(source, codec_chunk_sec=codec_chunk_sec),
            binding,
            {"reference_audio": reference},
        )
    return engine.Qwen3CustomVoiceTTSEngine(source, default_voice="Ryan", codec_chunk_sec=codec_chunk_sec), binding, {}


@pytest.mark.req("SP-SFQWEN3-023")
def test_constructors_preserve_engine_options(engine_setup: tuple[Any, FakeBinding, Any]) -> None:
    """Both engines accept their model source and fixed codec width, and CustomVoice accepts a default speaker."""
    engine, _, _ = engine_setup
    assert list(inspect.signature(engine.Qwen3CustomVoiceTTSEngine).parameters) == [
        "model_source",
        "default_voice",
        "codec_chunk_sec",
    ]
    assert list(inspect.signature(engine.Qwen3CloneTTSEngine).parameters) == ["model_source", "codec_chunk_sec"]


@pytest.mark.req("SP-SFQWEN3-023")
@pytest.mark.parametrize("kind", ["custom", "clone"])
@pytest.mark.parametrize("method", ["get", "stream"])
async def test_tuning_options_reach_both_engines_and_all_synthesis_methods(
    engine_setup: tuple[Any, FakeBinding, Any], monkeypatch: pytest.MonkeyPatch, kind: str, method: str
) -> None:
    """Each Qwen3 synthesis method passes call-local sampling values and the constructor's codec width."""
    instance, binding, reference = _tuning_engine(engine_setup, monkeypatch, kind, codec_chunk_sec=3.5)
    await _prepare_for_synthesis(instance, binding)
    options = {
        "do_sample": False,
        "temperature": 1.25,
        "top_k": 0,
        "top_p": 0.5,
        "repetition_penalty": 1.5,
    }

    if method == "get":
        await instance.get("Tuning", **reference, **options)
    elif method == "stream":
        _ = [chunk async for chunk in instance.stream("Tuning", **reference, **options)]

    assert binding.loaded_codec_chunk_sec == [3.5]
    assert len(binding.requests) == 1
    assert {name: getattr(binding.requests[0], name) for name in options} == options
    await instance.aclose()


@pytest.mark.req("SP-SFQWEN3-023")
@pytest.mark.parametrize("kind", ["custom", "clone"])
async def test_tuning_options_are_unset_when_omitted(
    engine_setup: tuple[Any, FakeBinding, Any], monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Omitted tuning arguments leave native defaults to the binding for both Qwen3 engines."""
    instance, binding, reference = _tuning_engine(engine_setup, monkeypatch, kind)
    await _prepare_for_synthesis(instance, binding)

    await instance.get("Defaults", **reference)

    assert binding.loaded_codec_chunk_sec == [None]
    request = binding.requests[0]
    assert (request.do_sample, request.temperature, request.top_k, request.top_p, request.repetition_penalty) == (
        None,
        None,
        None,
        None,
        None,
    )
    await instance.aclose()


@pytest.mark.req("SP-SFQWEN3-027")
@pytest.mark.parametrize("kind", ["custom", "clone"])
@pytest.mark.parametrize("method", ["get", "stream"])
@pytest.mark.parametrize(
    ("text", "options", "expected_limit"),
    [
        ("Hey!", {"language": "en"}, 28),
        ("はい。", {"language": "ja"}, 35),
        ("你好。", {"language": "zh"}, 35),
        ("안녕!", {"language": "ko"}, 35),
        *[(" Hi! ", {"language": code}, 30) for code in ("de", "it", "pt", "es", "fr", "ru")],
        ("Hi! \n", {}, 45),
        ("はい。", {"language": None}, 35),
        ("Hey!", {"language": "en", "max_new_tokens": None}, 28),
    ],
)
async def test_omitted_token_limit_caps_synthesis_by_text_length(
    engine_setup: tuple[Any, FakeBinding, Any],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    method: str,
    text: str,
    options: dict[str, object],
    expected_limit: int,
) -> None:
    """Both engines cap get and stream by the original text length and the requested language."""
    instance, binding, reference = _tuning_engine(engine_setup, monkeypatch, kind)
    await _prepare_for_synthesis(instance, binding)
    try:
        if method == "get":
            await instance.get(text, **reference, **options)
        else:
            _ = [chunk async for chunk in instance.stream(text, **reference, **options)]

        assert len(binding.requests) == 1
        assert binding.requests[0].max_new_tokens == expected_limit
    finally:
        await instance.aclose()


@pytest.mark.req("SP-SFQWEN3-027")
@pytest.mark.parametrize(
    ("language", "text", "ordinary_max_frames"),
    [
        ("en", "Hey!", 21),
        ("en", "Hi!", 17),
        ("en", "Yes.", 14),
        ("en", "Okay.", 13),
        ("en", "Hello there!", 27),
        ("en", "I'm Yakultman!", 25),
        ("en", "Nice to meet you.", 21),
        ("en", "Sorry, could you say that again?", 54),
        ("en", "That was fun! Let's talk again soon!", 50),
        ("en", "Thanks for stopping by. Have a great day!", 43),
        ("en", "Did you know that your gut is home to trillions of tiny helpers?", 88),
        (
            "en",
            "Yakult contains billions of live bacteria that help keep your gut healthy, "
            "so try one with breakfast every morning.",
            130,
        ),
        (
            "en",
            "I was made to tell people about good bacteria, and honestly, it is the best job in the world "
            "because everyone I meet leaves a little healthier and a little happier than before.",
            211,
        ),
        (
            "en",
            "If you have any questions about how probiotics work, what makes our drink special, "
            "or how to build healthy habits for your family, just ask me and I will do my best "
            "to explain everything step by step.",
            227,
        ),
        ("ja", "はい。", 26),
        ("ja", "こんにちは!", 21),
        ("ja", "ありがとう。", 42),
        ("ja", "もう一度お願いします。", 39),
        ("ja", "はじめまして、ヤクルトマンです。", 69),
        ("ja", "お腹の中には、たくさんの小さな仲間が住んでいます。", 83),
        ("ja", "毎朝の朝ごはんと一緒に一本飲むと、腸の調子を整えるのに役立ちます。", 86),
        (
            "ja",
            "わたしは善い菌のことをみんなに伝えるために生まれました。"
            "会う人がみんな少し元気になって帰ってくれるのが、何よりうれしいんです。",
            227,
        ),
    ],
)
async def test_default_token_limit_preserves_measured_ordinary_speech(
    engine_setup: tuple[Any, FakeBinding, Any], language: str, text: str, ordinary_max_frames: int
) -> None:
    """The default cap accommodates the measured maximum ordinary duration of each of 22 utterances."""
    # Measured with the 1.7B Q8_0 CustomVoice model, voice Vivian, default sampling, and seeds 0-19.
    # Ordinary durations exclude frames above three times each utterance's median across seeds 0-19.
    engine, binding, source = engine_setup
    instance = engine.Qwen3CustomVoiceTTSEngine(source, default_voice="Vivian")
    await _prepare_for_synthesis(instance, binding)
    try:
        await instance.get(text, language=language)

        limit = binding.requests[0].max_new_tokens
        assert limit is not None
        assert limit >= ordinary_max_frames
    finally:
        await instance.aclose()


@pytest.mark.req("SP-SFQWEN3-027")
@pytest.mark.parametrize("kind", ["custom", "clone"])
@pytest.mark.parametrize("method", ["get", "stream"])
@pytest.mark.parametrize("limit", [1, 2048, 2_147_483_647])
async def test_explicit_token_limit_overrides_text_length_cap(
    engine_setup: tuple[Any, FakeBinding, Any],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    method: str,
    limit: int,
) -> None:
    """An explicit token limit reaches native synthesis unchanged, including both accepted bounds."""
    instance, binding, reference = _tuning_engine(engine_setup, monkeypatch, kind)
    await _prepare_for_synthesis(instance, binding)
    try:
        options = {"language": "ja", "max_new_tokens": limit}
        if method == "get":
            await instance.get("はい。", **reference, **options)
        else:
            _ = [chunk async for chunk in instance.stream("はい。", **reference, **options)]

        assert len(binding.requests) == 1
        assert binding.requests[0].max_new_tokens == limit
    finally:
        await instance.aclose()


@pytest.mark.req("SP-SFQWEN3-027")
@pytest.mark.parametrize("kind", ["custom", "clone"])
@pytest.mark.parametrize("method", ["get", "stream"])
@pytest.mark.parametrize("limit", [True, 1.5, "32", -1])
async def test_invalid_explicit_token_limit_is_rejected_before_native_synthesis(
    engine_setup: tuple[Any, FakeBinding, Any],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    method: str,
    limit: object,
) -> None:
    """Both synthesis methods reject invalid explicit token limits instead of replacing them with a text cap."""
    instance, binding, reference = _tuning_engine(engine_setup, monkeypatch, kind)
    await _prepare_for_synthesis(instance, binding)
    try:
        with pytest.raises(ConfigurationError, match="max_new_tokens"):
            if method == "get":
                await instance.get("はい。", **reference, max_new_tokens=limit)
            else:
                _ = [chunk async for chunk in instance.stream("はい。", **reference, max_new_tokens=limit)]

        assert binding.synth_calls == 0
    finally:
        await instance.aclose()


@pytest.mark.req("SP-SFQWEN3-014")
@pytest.mark.parametrize("kind", ["custom", "clone"])
@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("do_sample", 1),
        ("temperature", True),
        ("temperature", "1"),
        ("temperature", 0),
        ("temperature", -0.5),
        ("temperature", float("nan")),
        ("temperature", float("inf")),
        ("temperature", 1e-50),
        ("temperature", 1e40),
        ("top_k", True),
        ("top_k", 1.5),
        ("top_k", -1),
        ("top_k", 2_147_483_648),
        ("top_p", False),
        ("top_p", 0),
        ("top_p", 1.1),
        ("top_p", float("nan")),
        ("repetition_penalty", "1"),
        ("repetition_penalty", 0),
        ("repetition_penalty", float("inf")),
    ],
)
async def test_invalid_tuning_option_does_not_start_native_synthesis(
    engine_setup: tuple[Any, FakeBinding, Any],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    option: str,
    value: object,
) -> None:
    """Invalid sampling types and ranges raise ConfigurationError before either engine starts native synthesis."""
    instance, binding, reference = _tuning_engine(engine_setup, monkeypatch, kind)
    await _prepare_for_synthesis(instance, binding)

    with pytest.raises(ConfigurationError, match=option):
        await instance.get("Invalid tuning", **reference, **{option: value})

    assert binding.load_calls == 1
    assert binding.synth_calls == 0
    await instance.aclose()


@pytest.mark.req("SP-SFQWEN3-014")
@pytest.mark.parametrize("kind", ["custom", "clone"])
@pytest.mark.parametrize("method", ["stream"])
async def test_invalid_tuning_option_precedes_native_synthesis_in_other_methods(
    engine_setup: tuple[Any, FakeBinding, Any], monkeypatch: pytest.MonkeyPatch, kind: str, method: str
) -> None:
    """Streaming rejects an invalid sampling option before native synthesis."""
    instance, binding, reference = _tuning_engine(engine_setup, monkeypatch, kind)
    await _prepare_for_synthesis(instance, binding)

    with pytest.raises(ConfigurationError, match="do_sample"):
        if method == "stream":
            _ = [chunk async for chunk in instance.stream("Invalid tuning", **reference, do_sample=1)]

    assert binding.load_calls == 1
    assert binding.synth_calls == 0
    await instance.aclose()


@pytest.mark.req("SP-SFQWEN3-014")
@pytest.mark.parametrize("kind", ["custom", "clone"])
@pytest.mark.parametrize("value", [True, "3", 0, -1, float("nan"), float("inf"), 1e-50, 1e40])
def test_invalid_codec_chunk_sec_is_rejected_at_construction(
    engine_setup: tuple[Any, FakeBinding, Any], monkeypatch: pytest.MonkeyPatch, kind: str, value: object
) -> None:
    """An invalid codec chunk width raises ConfigurationError during construction, before native synthesis."""
    engine, binding, source = engine_setup
    constructor = engine.Qwen3CloneTTSEngine if kind == "clone" else engine.Qwen3CustomVoiceTTSEngine

    with pytest.raises(ConfigurationError, match="codec_chunk_sec"):
        constructor(source, codec_chunk_sec=value)

    assert binding.load_calls == 0


@pytest.mark.req("SP-SFQWEN3-023")
@pytest.mark.parametrize("kind", ["custom", "clone"])
def test_codec_chunk_sec_cannot_change_after_construction(
    engine_setup: tuple[Any, FakeBinding, Any], monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Each Qwen3 engine holds its codec chunk width fixed after construction."""
    instance, binding, _ = _tuning_engine(engine_setup, monkeypatch, kind, codec_chunk_sec=3.5)

    assert instance.codec_chunk_sec == 3.5
    with pytest.raises(AttributeError):
        instance.codec_chunk_sec = 4.0
    assert binding.load_calls == 0


@pytest.mark.req("SP-SFQWEN3-036")
@pytest.mark.req("SP-SF-055")
def test_constructor_is_structural_lazy_and_model_source_is_immutable(
    engine_setup: tuple[Any, FakeBinding, Any],
) -> None:
    """Constructing an engine does not load the model, and its model source cannot change afterwards."""
    engine, binding, source = engine_setup

    instance = engine.Qwen3CustomVoiceTTSEngine(source, default_voice="Ryan")

    assert binding.load_calls == 0
    assert instance.model_source == source
    assert instance.default_voice == "Ryan"
    with pytest.raises((AttributeError, TypeError)):
        source.talker_path = "/changed"


@pytest.mark.req("SP-SFQWEN3-014")
@pytest.mark.parametrize(
    ("talker", "codec", "match"),
    [
        (1, "/codec.gguf", "talker_path"),
        ("", "/codec.gguf", "talker_path"),
        ("relative.gguf", "/codec.gguf", "absolute"),
        ("/talker.gguf", 1, "codec_path"),
        ("/talker.gguf", "", "codec_path"),
        ("/talker.gguf", "relative.gguf", "absolute"),
    ],
)
def test_model_source_rejects_invalid_structure(talker: object, codec: object, match: str) -> None:
    """Qwen3 rejects model sources without absolute talker and codec paths."""
    engine = importlib.import_module("speechflow.qwen3._engine")

    with pytest.raises(ConfigurationError, match=match):
        engine.Qwen3ModelSource(talker, codec)


@pytest.mark.req("SP-SFQWEN3-014")
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"text": ""}, "text"),
        ({"text": "unsafe <|token|>"}, r"<\|"),
        ({"text": "ok", "model": ""}, "model"),
        ({"text": "ok", "voice": ""}, "voice"),
        ({"text": "ok", "language": ""}, "language"),
        ({"text": "ok", "seed": True}, "seed"),
        ({"text": "ok", "seed": 1.5}, "seed"),
        ({"text": "ok", "max_new_tokens": 0}, "max_new_tokens"),
        ({"text": "ok", "max_new_tokens": 2_147_483_648}, "max_new_tokens"),
        ({"text": "ok", "instruct": ""}, "instruct"),
    ],
)
async def test_custom_validation_precedes_native_synthesis(
    engine_setup: tuple[Any, FakeBinding, Any], kwargs: dict[str, object], match: str
) -> None:
    """Invalid text or options raise ConfigurationError before native synthesis."""
    engine, binding, source = engine_setup
    instance = engine.Qwen3CustomVoiceTTSEngine(source, default_voice="Ryan")
    await _prepare_for_synthesis(instance, binding)

    with pytest.raises(ConfigurationError, match=match):
        await instance.get(**kwargs)

    assert binding.load_calls == 1
    assert binding.synth_calls == 0


@pytest.mark.req("SP-SFQWEN3-011")
@pytest.mark.req("SP-SFQWEN3-012")
@pytest.mark.req("SP-SFQWEN3-022")
async def test_custom_get_normalizes_audio_and_maps_all_request_options(
    engine_setup: tuple[Any, FakeBinding, Any],
) -> None:
    """CustomVoice returns normalized mono float32 audio at 24 kHz for its requested options."""
    engine, binding, source = engine_setup
    instance = engine.Qwen3CustomVoiceTTSEngine(source, default_voice="ryan")
    await _prepare_for_synthesis(instance, binding)

    audio = await instance.get(
        "Hello",
        model="qwen3-tts-0.6b-customvoice-Q4_K_M",
        voice="VIVIAN",
        language="en",
        seed=7,
        max_new_tokens=100,
        instruct="Warmly",
    )

    assert audio.sample_rate == 24000
    assert audio.channels == 1
    assert audio.format == "pcm"
    assert audio.data.dtype == np.float32
    assert audio.data.flags.writeable
    assert np.isfinite(audio.data).all()
    np.testing.assert_array_equal(audio.data, np.array([-1.0, 0.25, 0.0, 0.0, 1.0], dtype=np.float32))
    request = binding.requests[-1]
    assert vars(request) == {
        "text": "Hello",
        "language": "english",
        "speaker": "Vivian",
        "instruct": "Warmly",
        "reference_audio": None,
        "reference_text": None,
        "seed": 7,
        "max_new_tokens": 100,
        "do_sample": None,
        "temperature": None,
        "top_k": None,
        "top_p": None,
        "repetition_penalty": None,
    }


_LANGUAGES = (
    ("zh", "chinese"),
    ("en", "english"),
    ("de", "german"),
    ("it", "italian"),
    ("pt", "portuguese"),
    ("es", "spanish"),
    ("ja", "japanese"),
    ("ko", "korean"),
    ("fr", "french"),
    ("ru", "russian"),
)


def _language_test_engine(
    engine_setup: tuple[Any, FakeBinding, Any], monkeypatch: pytest.MonkeyPatch, mode: str
) -> tuple[Any, FakeBinding, dict[str, object]]:
    engine, binding, source = engine_setup
    if mode == "CustomVoice":
        return engine.Qwen3CustomVoiceTTSEngine(source, default_voice="Ryan"), binding, {}
    manifest = importlib.import_module("speechflow.qwen3._manifest")
    pair = manifest.ModelPair(
        model_id="qwen3-tts-0.6b-base-Q4_K_M",
        mode="Base",
        size="0.6B",
        quantization="Q4_K_M",
        talker=manifest.WeightFile("talker.gguf", 1, "0" * 64),
        codec=manifest.WeightFile("codec.gguf", 1, "0" * 64),
    )
    monkeypatch.setattr(engine, "validate_model_source", lambda *_args, **_kwargs: pair)
    reference = AudioData(np.array([0.1], dtype=np.float32), 24000)
    return engine.Qwen3CloneTTSEngine(source), binding, {"reference_audio": reference}


async def _synthesize_for_language(instance: Any, method: str, kwargs: dict[str, object]) -> None:
    if method == "get":
        await instance.get("Speech", **kwargs)
    else:
        _ = [chunk async for chunk in instance.stream("Speech", **kwargs)]


@pytest.mark.req("SP-SFQWEN3-030")
@pytest.mark.req("SP-SFQWEN3-031")
@pytest.mark.req("SP-SFQWEN3-032")
@pytest.mark.req("SP-SF-064")
@pytest.mark.req("SP-SF-065")
@pytest.mark.req("SP-SF-066")
@pytest.mark.parametrize("mode", ["CustomVoice", "Base"])
@pytest.mark.parametrize("method", ["get", "stream"])
@pytest.mark.parametrize(("code", "native_name"), [*_LANGUAGES, (None, None)])
async def test_language_code_reaches_native_request_as_qwen3_language_name(
    engine_setup: tuple[Any, FakeBinding, Any],
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    method: str,
    code: str | None,
    native_name: str | None,
) -> None:
    """Supported common language codes select the matching Qwen3 language; None leaves it unspecified."""
    instance, binding, kwargs = _language_test_engine(engine_setup, monkeypatch, mode)
    await _prepare_for_synthesis(instance, binding)
    await _synthesize_for_language(instance, method, {**kwargs, "language": code})
    assert binding.requests[-1].language == native_name


@pytest.mark.req("SP-SFQWEN3-030")
@pytest.mark.req("SP-SF-064")
@pytest.mark.parametrize("mode", ["CustomVoice", "Base"])
@pytest.mark.parametrize("method", ["get", "stream"])
@pytest.mark.parametrize(("code", "native_name"), _LANGUAGES)
@pytest.mark.parametrize("capitalized", [False, True])
async def test_native_language_name_is_rejected_with_common_code_suggestion(
    engine_setup: tuple[Any, FakeBinding, Any],
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    method: str,
    native_name: str,
    code: str,
    capitalized: bool,
) -> None:
    """Former Qwen3 language names fail before native synthesis and identify their replacement codes."""
    instance, binding, kwargs = _language_test_engine(engine_setup, monkeypatch, mode)
    await _prepare_for_synthesis(instance, binding)
    language = native_name.capitalize() if capitalized else native_name
    with pytest.raises(ConfigurationError) as caught:
        await _synthesize_for_language(instance, method, {**kwargs, "language": language})
    assert str(caught.value) == f"Qwen3 language={language!r} is not supported; use {code!r}"
    assert binding.load_calls == 1
    assert binding.synth_calls == 0
    assert binding.synth_calls == 0


@pytest.mark.req("SP-SFQWEN3-029")
@pytest.mark.req("SP-SF-063")
@pytest.mark.parametrize("mode", ["CustomVoice", "Base"])
@pytest.mark.parametrize("method", ["get", "stream"])
@pytest.mark.parametrize("language", ["EN", "en-US", "en_us", "eng"])
async def test_rejected_english_variants_identify_the_common_language_code(
    engine_setup: tuple[Any, FakeBinding, Any],
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    method: str,
    language: str,
) -> None:
    """Unambiguous English variants fail before synthesis and suggest the common code."""
    instance, binding, kwargs = _language_test_engine(engine_setup, monkeypatch, mode)
    await _prepare_for_synthesis(instance, binding)
    with pytest.raises(ConfigurationError) as caught:
        await _synthesize_for_language(instance, method, {**kwargs, "language": language})
    assert str(caught.value) == f"Qwen3 language={language!r} is not supported; use 'en'"
    assert binding.load_calls == 1
    assert binding.synth_calls == 0
    assert binding.synth_calls == 0


@pytest.mark.req("SP-SFQWEN3-029")
@pytest.mark.req("SP-SF-063")
@pytest.mark.parametrize(
    ("language", "code"),
    [
        ("zho", "zh"),
        ("deu", "de"),
        ("ita", "it"),
        ("por", "pt"),
        ("spa", "es"),
        ("jpn", "ja"),
        ("kor", "ko"),
        ("fra", "fr"),
        ("rus", "ru"),
    ],
)
async def test_rejected_three_letter_language_codes_identify_the_two_letter_code(
    engine_setup: tuple[Any, FakeBinding, Any],
    monkeypatch: pytest.MonkeyPatch,
    language: str,
    code: str,
) -> None:
    """A supported language written with its three letter code identifies its accepted code."""
    instance, binding, kwargs = _language_test_engine(engine_setup, monkeypatch, "CustomVoice")
    await _prepare_for_synthesis(instance, binding)
    with pytest.raises(ConfigurationError) as caught:
        await _synthesize_for_language(instance, "get", {**kwargs, "language": language})
    assert str(caught.value) == f"Qwen3 language={language!r} is not supported; use {code!r}"
    assert binding.load_calls == 1
    assert binding.synth_calls == 0
    assert binding.synth_calls == 0


@pytest.mark.req("SP-SFQWEN3-030")
@pytest.mark.req("SP-SF-064")
@pytest.mark.parametrize("language", ["yue", "haw", "Hawaiian", "xx"])
async def test_unsupported_language_lists_accepted_codes(
    engine_setup: tuple[Any, FakeBinding, Any], monkeypatch: pytest.MonkeyPatch, language: str
) -> None:
    """A language this model does not handle gets the accepted codes instead of a replacement code."""
    instance, binding, kwargs = _language_test_engine(engine_setup, monkeypatch, "CustomVoice")
    await _prepare_for_synthesis(instance, binding)
    with pytest.raises(ConfigurationError) as caught:
        await _synthesize_for_language(instance, "get", {**kwargs, "language": language})
    assert str(caught.value) == (
        f"Qwen3 language={language!r} is not supported; accepted values=zh, en, de, it, pt, es, ja, ko, fr, ru, or None"
    )
    assert binding.load_calls == 1
    assert binding.synth_calls == 0
    assert binding.synth_calls == 0


@pytest.mark.req("SP-SFQWEN3-030")
@pytest.mark.req("SP-SF-064")
@pytest.mark.parametrize("mode", ["CustomVoice", "Base"])
@pytest.mark.parametrize("method", ["get", "stream"])
@pytest.mark.parametrize("language", ["EN", "en-US", "auto", "beijing_dialect", "sichuan_dialect", "", 3])
async def test_non_common_language_value_fails_before_native_synthesis(
    engine_setup: tuple[Any, FakeBinding, Any],
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    method: str,
    language: object,
) -> None:
    """Codes outside the supported common values are rejected before Qwen3 synthesis starts."""
    instance, binding, kwargs = _language_test_engine(engine_setup, monkeypatch, mode)
    await _prepare_for_synthesis(instance, binding)
    with pytest.raises(ConfigurationError, match="language"):
        await _synthesize_for_language(instance, method, {**kwargs, "language": language})
    assert binding.load_calls == 1
    assert binding.synth_calls == 0
    assert binding.synth_calls == 0


@pytest.mark.req("SP-SFQWEN3-012")
@pytest.mark.req("SP-SFQWEN3-014")
async def test_custom_voice_listing_and_unknown_voice_behavior(engine_setup: tuple[Any, FakeBinding, Any]) -> None:
    """CustomVoice lists the model's speakers and rejects a missing or unknown voice before synthesis."""
    engine, binding, source = engine_setup
    instance = engine.Qwen3CustomVoiceTTSEngine(source)
    await _prepare_for_synthesis(instance, binding)

    assert instance.list_voices() == [
        {"name": "Ryan", "kind": "speaker"},
        {"name": "Vivian", "kind": "speaker"},
    ]
    with pytest.raises(ConfigurationError, match="voice"):
        await instance.get("missing effective voice")
    with pytest.raises(ConfigurationError, match=r"unknown.*voice"):
        await instance.get("unknown", voice="Nobody")
    assert binding.synth_calls == 0


@pytest.mark.req("SP-SFQWEN3-014")
async def test_model_mismatch_is_a_configuration_error(
    engine_setup: tuple[Any, FakeBinding, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Qwen3 rejects a requested model that differs from its prepared model."""
    engine, binding, source = engine_setup
    manifest = importlib.import_module("speechflow.qwen3._manifest")
    base_pair = manifest.ModelPair(
        model_id="qwen3-tts-0.6b-base-Q4_K_M",
        mode="Base",
        size="0.6B",
        quantization="Q4_K_M",
        talker=manifest.WeightFile("talker.gguf", 1, "0" * 64),
        codec=manifest.WeightFile("codec.gguf", 1, "0" * 64),
    )
    monkeypatch.setattr(engine, "validate_model_source", lambda *_args, **_kwargs: base_pair)
    clone = engine.Qwen3CloneTTSEngine(source)
    await _prepare_for_synthesis(clone, binding)
    reference = AudioData(np.array([0.0], dtype=np.float32), 24000)
    with pytest.raises(ConfigurationError, match="model"):
        await clone.get("text", model="different", reference_audio=reference)
    assert binding.synth_calls == 0


@pytest.mark.req("SP-SFQWEN3-016")
@pytest.mark.parametrize("mutation", ["basename", "size", "digest", "class-mode"])
async def test_engine_real_manifest_validation_rejects_identity_before_native_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    """Qwen3 rejects unapproved weight identity before loading its native model."""
    engine = importlib.import_module("speechflow.qwen3._engine")
    manifest = importlib.import_module("speechflow.qwen3._manifest")
    talker_payload = b"controlled talker"
    codec_payload = b"controlled codec"
    talker_name = "approved-talker.gguf"
    source_talker_name = "unapproved-talker.gguf" if mutation == "basename" else talker_name
    codec_name = "approved-codec.gguf"
    talker = tmp_path / source_talker_name
    codec = tmp_path / codec_name
    talker.write_bytes(talker_payload)
    codec.write_bytes(codec_payload)
    pair = _model_pair(
        manifest,
        talker_name=talker_name,
        talker_payload=talker_payload,
        codec_name=codec_name,
        codec_payload=codec_payload,
        mode="Base" if mutation == "class-mode" else "CustomVoice",
        talker_size=len(talker_payload) + 1 if mutation == "size" else None,
        talker_digest="0" * 64 if mutation == "digest" else None,
    )
    binding = FakeBinding()
    monkeypatch.setattr(manifest, "load_manifest", lambda: (pair,))
    monkeypatch.setattr(engine, "binding_factory", lambda: binding)
    instance = engine.Qwen3CustomVoiceTTSEngine(
        engine.Qwen3ModelSource(str(talker), str(codec)),
        default_voice="Ryan",
    )

    with pytest.raises(ConfigurationError):
        await instance.prepare()

    assert binding.load_calls == 0
    await instance.aclose()


@pytest.mark.req("SP-SFQWEN3-016")
async def test_engine_real_manifest_validation_passes_exact_pair_to_native_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Qwen3 loads the exact talker and codec pair accepted by its manifest."""
    engine = importlib.import_module("speechflow.qwen3._engine")
    manifest = importlib.import_module("speechflow.qwen3._manifest")
    talker_payload = b"controlled talker"
    codec_payload = b"controlled codec"
    talker = tmp_path / "approved-talker.gguf"
    codec = tmp_path / "approved-codec.gguf"
    talker.write_bytes(talker_payload)
    codec.write_bytes(codec_payload)
    pair = _model_pair(
        manifest,
        talker_name=talker.name,
        talker_payload=talker_payload,
        codec_name=codec.name,
        codec_payload=codec_payload,
    )
    binding = FakeBinding()
    monkeypatch.setattr(manifest, "load_manifest", lambda: (pair,))
    monkeypatch.setattr(engine, "binding_factory", lambda: binding)
    instance = engine.Qwen3CustomVoiceTTSEngine(
        engine.Qwen3ModelSource(str(talker), str(codec)),
        default_voice="Ryan",
    )

    await instance.prepare()

    assert binding.loaded_paths == [(str(talker), str(codec))]
    await instance.aclose()


@pytest.mark.req("SP-SFQWEN3-014")
@pytest.mark.parametrize(
    ("reference", "reference_text", "voice", "match"),
    [
        (None, None, None, "reference"),
        (AudioData(np.array([], dtype=np.float32), 24000), None, None, "reference"),
        (AudioData(np.array([0.0], dtype=np.float64), 24000), None, None, "float32"),
        (AudioData(np.array([0.0], dtype=np.float32), 16000), None, None, "24,000"),
        (AudioData(np.array([0.0], dtype=np.float32), 24000, channels=2), None, None, "mono"),
        (AudioData(np.array([np.nan], dtype=np.float32), 24000), None, None, "finite"),
        (AudioData(np.array([1.1], dtype=np.float32), 24000), None, None, r"\[-1, 1\]"),
        (AudioData(np.array([0.0], dtype=np.float32), 24000), 1, None, "reference_text"),
        (AudioData(np.array([0.0], dtype=np.float32), 24000), None, "named", "voice"),
    ],
)
async def test_clone_validation_precedes_native_synthesis(
    engine_setup: tuple[Any, FakeBinding, Any],
    monkeypatch: pytest.MonkeyPatch,
    reference: AudioData | None,
    reference_text: object,
    voice: str | None,
    match: str,
) -> None:
    """Clone synthesis rejects invalid reference audio and voice options before native synthesis."""
    engine, binding, source = engine_setup
    manifest = importlib.import_module("speechflow.qwen3._manifest")
    pair = manifest.ModelPair(
        model_id="qwen3-tts-0.6b-base-Q4_K_M",
        mode="Base",
        size="0.6B",
        quantization="Q4_K_M",
        talker=manifest.WeightFile("talker.gguf", 1, "0" * 64),
        codec=manifest.WeightFile("codec.gguf", 1, "0" * 64),
    )
    monkeypatch.setattr(engine, "validate_model_source", lambda *_args, **_kwargs: pair)
    instance = engine.Qwen3CloneTTSEngine(source)
    await _prepare_for_synthesis(instance, binding)

    with pytest.raises(ConfigurationError, match=match):
        await instance.get("Clone", voice=voice, reference_audio=reference, reference_text=reference_text)

    assert binding.load_calls == 1
    assert binding.synth_calls == 0


@pytest.mark.req("SP-SFQWEN3-012")
async def test_clone_uses_one_shot_reference_without_named_state(
    engine_setup: tuple[Any, FakeBinding, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clone synthesis passes the reference for one call without a named voice registry."""
    engine, binding, source = engine_setup
    manifest = importlib.import_module("speechflow.qwen3._manifest")
    pair = manifest.ModelPair(
        model_id="qwen3-tts-0.6b-base-Q4_K_M",
        mode="Base",
        size="0.6B",
        quantization="Q4_K_M",
        talker=manifest.WeightFile("talker.gguf", 1, "0" * 64),
        codec=manifest.WeightFile("codec.gguf", 1, "0" * 64),
    )
    monkeypatch.setattr(engine, "validate_model_source", lambda *_args, **_kwargs: pair)
    binding.samples = np.array([0.1], dtype=np.float32)
    instance = engine.Qwen3CloneTTSEngine(source)
    await _prepare_for_synthesis(instance, binding)
    reference = AudioData(np.array([0.2, -0.2], dtype=np.float32), 24000)

    await instance.get("Clone", reference_audio=reference, reference_text="Reference")

    request = binding.requests[-1]
    assert request.reference_text == "Reference"
    assert request.reference_audio is not reference.data
    np.testing.assert_array_equal(request.reference_audio, reference.data)
    assert not hasattr(instance, "list_voices")
    assert not hasattr(instance, "register_voice")


@pytest.mark.req("SP-SFQWEN3-036")
@pytest.mark.req("SP-SF-061")
@pytest.mark.parametrize(
    ("audio", "error_type"),
    [
        (None, EmptyOutputError),
        (np.array([], dtype=np.float32), EmptyOutputError),
    ],
)
async def test_buffered_empty_output_contract(
    engine_setup: tuple[Any, FakeBinding, Any], audio: np.ndarray | None, error_type: type[Exception]
) -> None:
    """Buffered synthesis raises EmptyOutputError when it produces no samples."""
    engine, binding, source = engine_setup
    instance = engine.Qwen3CustomVoiceTTSEngine(source, default_voice="Ryan")
    await _prepare_for_synthesis(instance, binding)
    if audio is None:
        binding.synthesize = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    else:
        binding.samples = audio

    with pytest.raises(error_type):
        await instance.get("Empty")


@pytest.mark.req("SP-SFQWEN3-036")
@pytest.mark.req("SP-SF-060")
@pytest.mark.parametrize(
    ("sample_rate", "channels", "shape"),
    [(16000, 1, (1,)), (24000, 2, (1,)), (24000, 1, (1, 1))],
)
async def test_buffered_invalid_native_audio_is_inference_error(
    engine_setup: tuple[Any, FakeBinding, Any], sample_rate: int, channels: int, shape: tuple[int, ...]
) -> None:
    """Invalid native audio is reported as an inference error."""
    engine, binding, source = engine_setup
    binding.samples = np.zeros(shape, dtype=np.float32)
    original = binding.synthesize

    def synthesize(*args: Any, **kwargs: Any) -> Any:
        result: Any = original(*args, **kwargs)
        result.sample_rate = sample_rate
        result.channels = channels
        return result

    binding.synthesize = synthesize  # type: ignore[method-assign]
    instance = engine.Qwen3CustomVoiceTTSEngine(source, default_voice="Ryan")
    await _prepare_for_synthesis(instance, binding)

    with pytest.raises(InferenceError):
        await instance.get("Bad audio")


@pytest.mark.req("SP-SFQWEN3-036")
@pytest.mark.req("SP-SF-061")
async def test_stream_preserves_chunk_order_and_rejects_empty_output(
    engine_setup: tuple[Any, FakeBinding, Any],
) -> None:
    """Streaming returns ordered 24 kHz mono chunks and raises EmptyOutputError for an empty stream."""
    engine, binding, source = engine_setup
    instance = engine.Qwen3CustomVoiceTTSEngine(source, default_voice="Ryan")
    await _prepare_for_synthesis(instance, binding)

    chunks = [chunk async for chunk in instance.stream("Stream")]

    np.testing.assert_array_equal(
        np.concatenate([chunk.data for chunk in chunks]), np.array([0.1, 0.2, 0.3], dtype=np.float32)
    )
    assert all(chunk.data.flags.writeable and chunk.sample_rate == 24000 and chunk.channels == 1 for chunk in chunks)

    binding.stream_samples = []
    with pytest.raises(EmptyOutputError):
        _ = [chunk async for chunk in instance.stream("Empty")]


@pytest.mark.req("SP-SFQWEN3-036")
@pytest.mark.req("SP-SF-060")
@pytest.mark.parametrize("valid_chunks", [0, 1])
async def test_stream_native_failure_is_inference_error_before_or_after_audio(
    engine_setup: tuple[Any, FakeBinding, Any], valid_chunks: int
) -> None:
    """A native stream failure becomes InferenceError before or after the first chunk."""
    engine, binding, source = engine_setup
    native = importlib.import_module("speechflow.qwen3._binding")
    instance = engine.Qwen3CustomVoiceTTSEngine(source, default_voice="Ryan")
    await _prepare_for_synthesis(instance, binding)
    binding.stream_samples = [np.array([0.1], dtype=np.float32)] * valid_chunks
    original = binding.synthesize

    def failing(*args: Any, **kwargs: Any) -> Any:
        original(*args, **kwargs)
        raise native.NativeCallError("qt_synthesize", -3, "failed")

    binding.synthesize = failing  # type: ignore[method-assign]
    with pytest.raises(InferenceError) as caught:
        _ = [chunk async for chunk in instance.stream("Failure")]

    assert "qt_synthesize" in str(caught.value)


@pytest.mark.req("SP-SFQWEN3-036")
@pytest.mark.req("SP-SF-060")
@pytest.mark.parametrize("valid_chunks", [0, 1])
async def test_stream_invalid_audio_is_inference_error_before_or_after_audio(
    engine_setup: tuple[Any, FakeBinding, Any], valid_chunks: int
) -> None:
    """A stream audio conversion failure becomes InferenceError before or after the first chunk."""
    engine, binding, source = engine_setup
    instance = engine.Qwen3CustomVoiceTTSEngine(source, default_voice="Ryan")
    await _prepare_for_synthesis(instance, binding)
    binding.stream_samples = [np.array([0.1], dtype=np.float32)] * valid_chunks + [np.array([[0.2]], dtype=np.float32)]
    with pytest.raises(InferenceError) as caught:
        _ = [chunk async for chunk in instance.stream("Failure")]

    assert "invalid audio" in str(caught.value)


@pytest.mark.req("SP-SFQWEN3-036")
@pytest.mark.req("SP-SF-055")
async def test_prepare_converges_on_one_attempt_and_failure_allows_a_new_attempt(
    engine_setup: tuple[Any, FakeBinding, Any],
) -> None:
    """Concurrent prepare calls load once, and a failed load can be tried again."""
    engine, binding, source = engine_setup
    native = importlib.import_module("speechflow.qwen3._binding")
    instance = engine.Qwen3CustomVoiceTTSEngine(source, default_voice="Ryan")

    await asyncio.gather(instance.prepare(), instance.prepare(), instance.prepare())
    assert binding.load_calls == 1

    other_binding = FakeBinding(load_errors=[native.NativeCallError("qt_init", -3, "load failed")])
    engine.binding_factory = lambda: other_binding
    failed = engine.Qwen3CustomVoiceTTSEngine(source, default_voice="Ryan")
    with pytest.raises(EnvironmentUnavailableError, match="qt_init") as caught:
        await failed.prepare()
    assert isinstance(caught.value.__cause__, native.NativeCallError)
    await failed.prepare()
    assert other_binding.load_calls == 2


@pytest.mark.req("SP-SFQWEN3-036")
@pytest.mark.req("SP-SF-055")
async def test_queued_prepare_retries_after_a_failed_load(engine_setup: tuple[Any, FakeBinding, Any]) -> None:
    """A queued prepare retries after a failed load and leaves the engine ready."""
    engine, binding, source = engine_setup
    native = importlib.import_module("speechflow.qwen3._binding")
    binding.load_errors.append(native.NativeCallError("qt_init", -3, "load failed"))
    binding.load_entered = threading.Event()
    binding.load_release = threading.Event()
    instance = engine.Qwen3CustomVoiceTTSEngine(source, default_voice="Ryan")
    first = asyncio.create_task(instance.prepare())
    assert await asyncio.to_thread(binding.load_entered.wait, 1)
    second = asyncio.create_task(instance.prepare())
    binding.load_release.set()
    results = await asyncio.gather(first, second, return_exceptions=True)
    assert isinstance(results[0], EnvironmentUnavailableError)
    assert results[1] is None
    assert binding.load_calls == 2
    assert instance.health().state == "ready"
    await instance.aclose()


@pytest.mark.req("SP-SFQWEN3-036")
@pytest.mark.req("SP-SF-055")
async def test_prepare_timeout_waits_for_native_load_and_keeps_success(
    engine_setup: tuple[Any, FakeBinding, Any],
) -> None:
    """A caller's timeout waits for a native load that cannot stop and retains the prepared model."""
    engine, binding, source = engine_setup
    binding.load_entered = threading.Event()
    binding.load_release = threading.Event()
    instance = engine.Qwen3CustomVoiceTTSEngine(source, default_voice="Ryan")

    async def timed_prepare() -> None:
        async with asyncio.timeout(0.01):
            await instance.prepare()

    caller = asyncio.create_task(timed_prepare())
    assert await asyncio.to_thread(binding.load_entered.wait, 1)
    await asyncio.sleep(0.03)
    assert not caller.done()
    binding.load_release.set()
    with pytest.raises(TimeoutError):
        await caller
    assert instance.health().state == "ready"
    await instance.prepare()
    assert binding.load_calls == 1
    await instance.aclose()


@pytest.mark.req("SP-SFQWEN3-036")
@pytest.mark.req("SP-SF-056")
async def test_close_during_prepare_waits_and_releases_late_context(engine_setup: tuple[Any, FakeBinding, Any]) -> None:
    """Closing during loading waits for native loading and releases its context once."""
    engine, binding, source = engine_setup
    binding.load_entered = threading.Event()
    binding.load_release = threading.Event()
    instance = engine.Qwen3CustomVoiceTTSEngine(source, default_voice="Ryan")
    loading = asyncio.create_task(instance.prepare())
    assert await asyncio.to_thread(binding.load_entered.wait, 1)
    close = asyncio.create_task(instance.aclose())
    await asyncio.sleep(0.01)
    assert not close.done()
    binding.load_release.set()
    with pytest.raises(EngineClosedError):
        await loading
    await close
    assert binding.free_calls == 1
    assert instance.health().state == "closed"


@pytest.mark.req("SP-SFQWEN3-036")
@pytest.mark.req("SP-SF-060")
async def test_prepare_translates_native_load_failure(engine_setup: tuple[Any, FakeBinding, Any]) -> None:
    """A native model load failure becomes TTSError with the native error as its cause."""
    engine, binding, source = engine_setup
    native = importlib.import_module("speechflow.qwen3._binding")
    binding.load_errors.append(native.NativeCallError("qt_init", -3, "GPU unavailable"))
    instance = engine.Qwen3CustomVoiceTTSEngine(source, default_voice="Ryan")

    with pytest.raises(EnvironmentUnavailableError, match="qt_init") as caught:
        await instance.prepare()

    assert isinstance(caught.value.__cause__, native.NativeCallError)


@pytest.mark.req("SP-SFQWEN3-036")
@pytest.mark.req("SP-SF-058")
async def test_caller_cancellation_reaches_native_and_waits_for_completion(
    engine_setup: tuple[Any, FakeBinding, Any],
) -> None:
    """Cancelling get() cancels the running inference and ends the call only after the inference finishes."""
    engine, binding, source = engine_setup
    instance = engine.Qwen3CustomVoiceTTSEngine(source, default_voice="Ryan")
    await _prepare_for_synthesis(instance, binding)
    binding.synth_entered = threading.Event()
    binding.synth_release = threading.Event()
    task = asyncio.create_task(instance.get("Cancel"))
    assert await asyncio.to_thread(binding.synth_entered.wait, 1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert binding.active_calls == 0


@pytest.mark.req("SP-SFQWEN3-036")
@pytest.mark.req("SP-SF-058")
async def test_stream_consumer_cancellation_reaches_native_and_waits_for_completion(
    engine_setup: tuple[Any, FakeBinding, Any],
) -> None:
    """Cancelling the wait for the next stream chunk cancels the running inference and waits for it to finish."""
    engine, binding, source = engine_setup
    instance = engine.Qwen3CustomVoiceTTSEngine(source, default_voice="Ryan")
    await _prepare_for_synthesis(instance, binding)
    binding.synth_entered = threading.Event()
    binding.synth_release = threading.Event()
    stream = instance.stream("Cancel stream")
    next_chunk = asyncio.create_task(anext(stream))
    assert await asyncio.to_thread(binding.synth_entered.wait, 1)

    next_chunk.cancel()
    with pytest.raises(asyncio.CancelledError):
        await next_chunk

    assert binding.active_calls == 0
    await stream.aclose()


@pytest.mark.req("SP-SFQWEN3-036")
@pytest.mark.req("SP-SF-058")
async def test_stream_early_close_after_first_chunk_reaches_native_cancellation(
    engine_setup: tuple[Any, FakeBinding, Any],
) -> None:
    """Closing a stream before reading it to the end cancels the running inference."""
    engine, binding, source = engine_setup
    native = importlib.import_module("speechflow.qwen3._binding")
    cancellation_observed = threading.Event()
    poll = threading.Event()

    def synthesize(
        _context: object,
        _request: object,
        *,
        on_chunk: Callable[[object], bool] | None,
        cancelled: Callable[[], bool],
    ) -> None:
        assert on_chunk is not None
        assert on_chunk(native.NativeAudio(np.array([0.25], dtype=np.float32), 24000, 1))
        while not cancelled():
            poll.wait(0.001)
        cancellation_observed.set()

    instance = engine.Qwen3CustomVoiceTTSEngine(source, default_voice="Ryan")
    await _prepare_for_synthesis(instance, binding)
    binding.synthesize = synthesize  # type: ignore[assignment]
    stream = instance.stream("Close after one chunk")

    first = await anext(stream)
    await asyncio.wait_for(stream.aclose(), 1)

    np.testing.assert_array_equal(first.data, np.array([0.25], dtype=np.float32))
    assert cancellation_observed.is_set()
    await instance.aclose()


@pytest.mark.req("SP-SFQWEN3-036")
@pytest.mark.req("SP-SF-057")
@pytest.mark.req("SP-SF-051")
async def test_native_operations_are_serialized_and_event_loop_keeps_progressing(
    engine_setup: tuple[Any, FakeBinding, Any],
) -> None:
    """One engine runs one native inference at a time while the caller's event loop keeps running."""
    engine, binding, source = engine_setup
    instance = engine.Qwen3CustomVoiceTTSEngine(source, default_voice="Ryan")
    await _prepare_for_synthesis(instance, binding)
    binding.synth_entered = threading.Event()
    binding.synth_release = threading.Event()
    first = asyncio.create_task(instance.get("one"))
    assert await asyncio.to_thread(binding.synth_entered.wait, 1)
    second = asyncio.create_task(instance.get("two"))
    progressed = False
    await asyncio.sleep(0.01)
    progressed = True
    binding.synth_release.set()
    await asyncio.gather(first, second)

    assert progressed
    assert binding.max_active_calls == 1
    assert [request.text for request in binding.requests] == ["one", "two"]
    assert set(binding.thread_ids) != {threading.get_ident()}


@pytest.mark.req("SP-SFQWEN3-036")
@pytest.mark.req("SP-SF-056")
async def test_close_is_idempotent_waits_for_active_call_and_rejects_future_work(
    engine_setup: tuple[Any, FakeBinding, Any],
) -> None:
    """Close can be repeated, waits for the active call, and rejects later work."""
    engine, binding, source = engine_setup
    instance = engine.Qwen3CustomVoiceTTSEngine(source, default_voice="Ryan")
    await _prepare_for_synthesis(instance, binding)
    binding.synth_entered = threading.Event()
    binding.synth_release = threading.Event()
    synthesis = asyncio.create_task(instance.get("active"))
    assert await asyncio.to_thread(binding.synth_entered.wait, 1)
    close = asyncio.create_task(instance.aclose())
    await asyncio.sleep(0.01)
    with pytest.raises(EngineClosedError, match="closed"):
        await instance.get("rejected")
    binding.synth_release.set()
    with pytest.raises(EngineClosedError):
        await synthesis
    await asyncio.gather(close, instance.aclose(), instance.aclose())

    assert binding.free_calls == 1


@pytest.mark.req("SP-SFQWEN3-036")
@pytest.mark.req("SP-SF-056")
@pytest.mark.parametrize("kind", ["custom", "clone"])
async def test_work_is_rejected_after_close_and_reads_show_closed(
    engine_setup: tuple[Any, FakeBinding, Any], monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """After close, model work fails with EngineClosedError, the speaker list that needs the loaded model fails with
    EngineStateError, and capabilities and health still return their values."""
    instance, _, reference = _tuning_engine(engine_setup, monkeypatch, kind)
    await instance.prepare()
    await instance.aclose()
    for operation in (
        instance.prepare(),
        instance.get("closed get", **reference),
        anext(instance.stream("closed stream", **reference)),
    ):
        with pytest.raises(EngineClosedError):
            await operation
    if kind == "custom":
        with pytest.raises(EngineStateError):
            instance.list_voices()
    assert instance.health().state == "closed"
    assert instance.capabilities.voice_listing is (kind == "custom")
    await instance.aclose()


@pytest.mark.req("SP-SFQWEN3-018")
async def test_missing_editable_payload_is_reported_at_model_load_time(
    engine_setup: tuple[Any, FakeBinding, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing editable native payload lets construction succeed and fails at model load with build instructions."""
    engine, _, source = engine_setup
    payload = importlib.import_module("speechflow.qwen3._payload")
    binding = importlib.import_module("speechflow.qwen3._binding")
    monkeypatch.setattr(payload, "_native_directory", lambda: tmp_path / "missing-native")
    monkeypatch.setattr(engine, "binding_factory", binding.CtypesNativeBinding)
    instance = engine.Qwen3CustomVoiceTTSEngine(source, default_voice="Ryan")

    with pytest.raises(EnvironmentUnavailableError, match=r"scripts/build_native\.sh"):
        await instance.prepare()


@pytest.mark.req("SP-SFQWEN3-036")
@pytest.mark.req("SP-SF-046")
def test_loaded_unclosed_engine_does_not_block_interpreter_exit() -> None:
    """A loaded engine that is never closed does not block interpreter exit."""
    script = """
import asyncio
from types import SimpleNamespace
import numpy as np
import speechflow.qwen3._engine as module
from speechflow.qwen3._binding import NativeAudio

class FakeBinding:
    def load_model(self, talker_path, codec_path, *, codec_chunk_sec=None):
        return object()
    def free_context(self, context):
        return None
    def list_speakers(self, context):
        return ("Ryan",)
    def synthesize(self, context, request, *, on_chunk, cancelled):
        return NativeAudio(np.array([0.0], dtype=np.float32), 24000, 1)

module.binding_factory = FakeBinding
module.validate_model_source = lambda *args, **kwargs: SimpleNamespace(
    model_id="qwen3-tts-0.6b-customvoice-Q4_K_M", mode="CustomVoice"
)
engine = module.Qwen3CustomVoiceTTSEngine(module.Qwen3ModelSource("/talker.gguf", "/codec.gguf"), default_voice="Ryan")
asyncio.run(engine.prepare())
print("loaded")
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert completed.stdout.strip() == "loaded"


@pytest.mark.req("SP-SFQWEN3-036")
@pytest.mark.req("SP-SF-055")
@pytest.mark.req("SP-SF-053")
@pytest.mark.parametrize("kind", ["custom", "clone"])
async def test_unprepared_work_reports_state_without_loading(
    engine_setup: tuple[Any, FakeBinding, Any], monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Before prepare, valid or invalid synthesis options report engine state without loading a model."""
    instance, binding, reference = _tuning_engine(engine_setup, monkeypatch, kind)
    for options in (reference, {**reference, "temperature": 0}):
        with pytest.raises(EngineStateError):
            await instance.get("Before preparation", **options)
        with pytest.raises(EngineStateError):
            await anext(instance.stream("Before preparation", **options))
    if kind == "custom":
        with pytest.raises(EngineStateError):
            instance.list_voices()
    assert binding.load_calls == binding.synth_calls == 0
    await instance.aclose()


@pytest.mark.req("SP-SFQWEN3-036")
@pytest.mark.req("SP-SF-055")
@pytest.mark.parametrize("kind", ["custom", "clone"])
async def test_prepare_loads_and_trials_once_with_internal_input(
    engine_setup: tuple[Any, FakeBinding, Any], monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Prepare loads once, trials native inference with a speaker or generated reference, and returns None."""
    instance, binding, _ = _tuning_engine(engine_setup, monkeypatch, kind)
    binding.samples = np.array([0.0], dtype=np.float32)
    assert await instance.prepare() is None
    await instance.prepare()
    assert binding.load_calls == binding.synth_calls == 1
    request = binding.requests[0]
    assert request.text.strip()
    assert request.max_new_tokens is not None and request.max_new_tokens > 0
    if kind == "custom":
        assert request.speaker == binding.speakers[0]
        assert request.reference_audio is None
        first = instance.list_voices()
        first.clear()
        assert len(instance.list_voices()) == len(binding.speakers)
        assert binding.load_calls == 1
    else:
        assert request.speaker is None
        assert request.reference_audio.dtype == np.float32
        assert request.reference_audio.ndim == 1 and request.reference_audio.size > 0
        assert np.isfinite(request.reference_audio).all()
        assert 0 < np.max(np.abs(request.reference_audio)) <= 1
    assert instance.health().state == "ready"
    await instance.aclose()


@pytest.mark.req("SP-SFQWEN3-036")
@pytest.mark.req("SP-SF-055")
@pytest.mark.req("SP-SF-061")
@pytest.mark.parametrize("kind", ["custom", "clone"])
async def test_empty_trial_releases_context_and_allows_prepare_retry(
    engine_setup: tuple[Any, FakeBinding, Any], monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """An empty trial fails preparation, releases its model, and allows a retry once samples are returned."""
    instance, binding, _ = _tuning_engine(engine_setup, monkeypatch, kind)
    binding.samples = np.array([], dtype=np.float32)
    try:
        with pytest.raises(EmptyOutputError):
            await instance.prepare()

        assert binding.load_calls == binding.synth_calls == binding.free_calls == 1
        assert instance.model_identifier is None
        assert instance.health().state == "unprepared"

        binding.samples = np.array([0.0], dtype=np.float32)
        assert await instance.prepare() is None
        assert binding.load_calls == binding.synth_calls == 2
        assert binding.free_calls == 1
        assert instance.model_identifier is not None
        assert instance.health().state == "ready"
    finally:
        await instance.aclose()
    assert binding.free_calls == 2


@pytest.mark.req("SP-SFQWEN3-036")
@pytest.mark.req("SP-SF-055")
@pytest.mark.req("SP-SF-060")
async def test_trial_failure_releases_partial_context_and_allows_retry(
    engine_setup: tuple[Any, FakeBinding, Any],
) -> None:
    """A failed native trial raises InferenceError, releases its model, and leaves preparation retryable."""
    engine, binding, source = engine_setup
    native = importlib.import_module("speechflow.qwen3._binding")
    binding.synth_error = native.NativeCallError("qt_synthesize", -3, "trial failed")
    instance = engine.Qwen3CustomVoiceTTSEngine(source)
    with pytest.raises(InferenceError, match="qt_synthesize"):
        await instance.prepare()
    assert binding.free_calls == 1
    assert instance.model_identifier is None
    assert instance.health().state == "unprepared"
    binding.synth_error = None
    await instance.prepare()
    assert binding.load_calls == 2
    await instance.aclose()
    assert binding.free_calls == 2


@pytest.mark.req("SP-SFQWEN3-023")
@pytest.mark.parametrize("kind", ["custom", "clone"])
async def test_get_uses_buffered_decode_and_stream_uses_callbacks(
    engine_setup: tuple[Any, FakeBinding, Any], monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Get preserves native buffered decoding while stream requests native callback decoding."""
    instance, binding, reference = _tuning_engine(engine_setup, monkeypatch, kind)
    await _prepare_for_synthesis(instance, binding)
    await instance.get("Buffered", **reference)
    _ = [chunk async for chunk in instance.stream("Streaming", **reference)]
    assert binding.streaming_calls == [False, True]
    await instance.aclose()


@pytest.mark.req("SP-SFQWEN3-023")
@pytest.mark.parametrize("method", ["get", "stream"])
async def test_caller_cannot_select_internal_decode_mode(
    engine_setup: tuple[Any, FakeBinding, Any], method: str
) -> None:
    """A caller cannot change the native decode mode through an internal option."""
    engine, binding, source = engine_setup
    instance = engine.Qwen3CustomVoiceTTSEngine(source, default_voice="Ryan")
    await _prepare_for_synthesis(instance, binding)
    with pytest.raises(ConfigurationError):
        if method == "get":
            await instance.get("Internal option", _qwen3_buffered=False)
        else:
            await anext(instance.stream("Internal option", _qwen3_buffered=True))
    assert binding.synth_calls == 0
    await instance.aclose()


@pytest.mark.req("SP-SFQWEN3-036")
@pytest.mark.req("SP-SF-058")
@pytest.mark.parametrize("operation", ["prepare", "get", "stream"])
async def test_cancelled_native_work_finishes_before_return_and_observes_failure(
    engine_setup: tuple[Any, FakeBinding, Any], operation: str
) -> None:
    """Cancellation reaches native work, waits for its exit, and consumes its cancellation diagnostic."""
    engine, binding, source = engine_setup
    native = importlib.import_module("speechflow.qwen3._binding")
    instance = engine.Qwen3CustomVoiceTTSEngine(source, default_voice="Ryan")
    if operation != "prepare":
        await _prepare_for_synthesis(instance, binding)
    entered = threading.Event()
    cancelled_observed = threading.Event()
    released = threading.Event()
    poll = threading.Event()

    def synthesize(_context: object, _request: object, *, on_chunk: object, cancelled: Callable[[], bool]) -> None:
        entered.set()
        while not cancelled():
            poll.wait(0.001)
        cancelled_observed.set()
        released.wait(2)
        raise native.NativeCallError("qt_synthesize", -5, "cancelled")

    binding.synthesize = synthesize  # type: ignore[assignment]
    loop = asyncio.get_running_loop()
    unhandled: list[dict[str, Any]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    stream = instance.stream("Cancellation")
    work = (
        instance.prepare()
        if operation == "prepare"
        else instance.get("Cancellation")
        if operation == "get"
        else anext(stream)
    )
    caller = asyncio.create_task(work)
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        caller.cancel()
        assert await asyncio.to_thread(cancelled_observed.wait, 1)
        assert not caller.done()
        released.set()
        with pytest.raises(asyncio.CancelledError):
            await caller
        await stream.aclose()
        await instance.aclose()
        del caller
        gc.collect()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not unhandled
        assert binding.free_calls == 1
    finally:
        released.set()
        loop.set_exception_handler(previous_handler)
