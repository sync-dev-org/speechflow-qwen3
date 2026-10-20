from __future__ import annotations

import asyncio
import contextlib
import ctypes
import math
import queue
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from speechflow.core import (
    AudioData,
    ConfigurationError,
    EmptyOutputError,
    EngineCapabilities,
    EngineState,
    EngineStateError,
    EnvironmentUnavailableError,
    InferenceError,
    TTSEngineBase,
    TTSError,
)

from ._binding import CtypesNativeBinding, NativeAudio, NativeBinding, NativeRequest
from ._manifest import ModelPair, validate_model_source

_NATIVE_C_INT_MAX = 2_147_483_647
_SAMPLE_RATE = 24_000
_LANGUAGE_TO_NATIVE = {
    "zh": "chinese",
    "en": "english",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "es": "spanish",
    "ja": "japanese",
    "ko": "korean",
    "fr": "french",
    "ru": "russian",
}
_NATIVE_TO_LANGUAGE = {native: code for code, native in _LANGUAGE_TO_NATIVE.items()}
_THREE_LETTER_TO_LANGUAGE = {
    "zho": "zh",
    "eng": "en",
    "deu": "de",
    "ita": "it",
    "por": "pt",
    "spa": "es",
    "jpn": "ja",
    "kor": "ko",
    "fra": "fr",
    "rus": "ru",
}
_LANGUAGE_SUGGESTIONS = {
    **{code: code for code in _LANGUAGE_TO_NATIVE},
    **_NATIVE_TO_LANGUAGE,
    **_THREE_LETTER_TO_LANGUAGE,
}


@dataclass(frozen=True, slots=True)
class Qwen3ModelSource:
    talker_path: str
    codec_path: str

    def __post_init__(self) -> None:
        _validate_source_path(self.talker_path, option="talker_path")
        _validate_source_path(self.codec_path, option="codec_path")


@dataclass(frozen=True, slots=True)
class _LoadedModel:
    context: object
    pair: ModelPair
    speakers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _StreamEnd:
    error: BaseException | None = None


type _StreamItem = NativeAudio | _StreamEnd


binding_factory: Callable[[], NativeBinding] = CtypesNativeBinding


_BUFFERED_DECODE = object()
_SAMPLING_OPTIONS = {
    "language",
    "seed",
    "max_new_tokens",
    "do_sample",
    "temperature",
    "top_k",
    "top_p",
    "repetition_penalty",
}


class _Qwen3TTSEngineBase(TTSEngineBase):
    _EXPECTED_MODE: str

    def __init__(self, model_source: Qwen3ModelSource, *, codec_chunk_sec: float | None = None) -> None:
        if not isinstance(model_source, Qwen3ModelSource):
            raise ConfigurationError("model_source must be a Qwen3ModelSource")
        if codec_chunk_sec is not None:
            _validate_positive_float(codec_chunk_sec, option="codec_chunk_sec")
        super().__init__()
        self.model_source = model_source
        self._codec_chunk_sec = codec_chunk_sec
        self._binding = binding_factory()
        self._executor: ThreadPoolExecutor | None = None
        self._loaded: _LoadedModel | None = None

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            languages=frozenset(_LANGUAGE_TO_NATIVE),
            auto_language=True,
            streaming=True,
            voice_listing=self._EXPECTED_MODE == "CustomVoice",
            voice_cloning=self._EXPECTED_MODE == "Base",
            sample_rate=_SAMPLE_RATE,
        )

    @property
    def codec_chunk_sec(self) -> float | None:
        return self._codec_chunk_sec

    @property
    def model_identifier(self) -> str | None:
        return self._loaded.pair.model_id if self._loaded is not None else None

    def _native_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="speechflow-qwen3")
        return self._executor

    async def _prepare(self) -> None:
        cancelled = threading.Event()
        future = self._native_executor().submit(self._load_and_trial_blocking, cancelled)
        wrapped = asyncio.wrap_future(future)
        try:
            try:
                loaded = await asyncio.shield(wrapped)
            except asyncio.CancelledError as cancellation:
                cancelled.set()
                try:
                    loaded = await _await_future_to_completion(future, wrapped=wrapped)
                except BaseException:
                    raise cancellation from None
            self._loaded = loaded
        except BaseException:
            assert self._executor is not None
            self._executor.shutdown(wait=False)
            self._executor = None
            raise

    def _load_and_trial_blocking(self, cancelled: threading.Event) -> _LoadedModel:
        pair = validate_model_source(
            self.model_source.talker_path,
            self.model_source.codec_path,
            expected_mode=self._EXPECTED_MODE,
        )
        try:
            context = self._binding.load_model(
                self.model_source.talker_path,
                self.model_source.codec_path,
                codec_chunk_sec=self.codec_chunk_sec,
            )
        except TTSError:
            raise
        except Exception as exc:
            raise EnvironmentUnavailableError(f"Qwen3 native model load failed during qt_init: {exc}") from exc
        try:
            speakers = self._binding.list_speakers(context) if self._EXPECTED_MODE == "CustomVoice" else ()
            if self._EXPECTED_MODE == "CustomVoice" and not speakers:
                raise EnvironmentUnavailableError("Qwen3 CustomVoice model contains no built-in speakers")
            loaded = _LoadedModel(context=context, pair=pair, speakers=speakers)
            reference = None
            if self._EXPECTED_MODE == "Base":
                time = np.arange(_SAMPLE_RATE, dtype=np.float32) / _SAMPLE_RATE
                reference = (0.01 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
            request = NativeRequest(
                text="Hello.",
                language="english",
                speaker=speakers[0] if speakers else None,
                instruct=None,
                reference_audio=reference,
                reference_text=None,
                seed=0,
                max_new_tokens=8,
                do_sample=False,
            )
            native = self._binding.synthesize(context, request, on_chunk=None, cancelled=cancelled.is_set)
            if native is None or native.samples.size == 0:
                raise EmptyOutputError("Qwen3 preparation trial returned no audio samples")
            return loaded
        except BaseException:
            self._binding.free_context(context)
            raise

    async def _release(self) -> None:
        loaded = self._loaded
        self._loaded = None
        try:
            if loaded is not None:
                await _await_future_to_completion(
                    self._native_executor().submit(self._binding.free_context, loaded.context)
                )
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=False)
                self._executor = None

    async def get(self, text: str, model: str | None = None, voice: str | None = None, **kwargs: object) -> AudioData:
        _reject_decode_option(kwargs)
        return await super().get(text, model=model, voice=voice, _qwen3_buffered=_BUFFERED_DECODE, **kwargs)

    async def stream(
        self, text: str, model: str | None = None, voice: str | None = None, **kwargs: object
    ) -> AsyncGenerator[AudioData, None]:
        _reject_decode_option(kwargs)
        source = super().stream(text, model=model, voice=voice, **kwargs)
        try:
            async for chunk in source:
                yield chunk
        finally:
            await source.aclose()

    async def _synthesize(
        self, text: str, model: str | None = None, voice: str | None = None, **kwargs: object
    ) -> AsyncIterator[AudioData]:
        buffered = kwargs.pop("_qwen3_buffered", None) is _BUFFERED_DECODE
        loaded = self._loaded
        assert loaded is not None
        request = self._request(loaded, text, model, voice, kwargs)
        source = self._buffered(loaded, request) if buffered else self._streaming(loaded, request)
        try:
            async for chunk in source:
                yield chunk
        finally:
            await source.aclose()

    def _request(
        self, loaded: _LoadedModel, text: str, model: str | None, voice: str | None, options: dict[str, object]
    ) -> NativeRequest:
        extra = {"instruct"} if self._EXPECTED_MODE == "CustomVoice" else {"reference_audio", "reference_text"}
        unknown = options.keys() - (_SAMPLING_OPTIONS | extra)
        if unknown:
            raise ConfigurationError(f"Qwen3 unsupported synthesis options: {', '.join(sorted(unknown))}")
        language = _validate_request_options(
            text=text,
            model=model,
            voice=voice,
            language=options.get("language"),
            seed=options.get("seed"),
            max_new_tokens=options.get("max_new_tokens"),
            do_sample=options.get("do_sample"),
            temperature=options.get("temperature"),
            top_k=options.get("top_k"),
            top_p=options.get("top_p"),
            repetition_penalty=options.get("repetition_penalty"),
        )
        if model is not None and model != loaded.pair.model_id:
            raise ConfigurationError(
                f"Qwen3 model mismatch: loaded model={loaded.pair.model_id!r}, requested model={model!r}"
            )
        speaker = None
        instruct = options.get("instruct")
        reference = None
        reference_text = options.get("reference_text")
        if self._EXPECTED_MODE == "CustomVoice":
            if instruct is not None:
                _validate_non_empty_string(instruct, option="instruct")
            default_voice = cast(Qwen3CustomVoiceTTSEngine, self).default_voice
            speaker = _resolve_speaker(loaded.speakers, voice if voice is not None else default_voice)
        else:
            if voice is not None:
                raise ConfigurationError(
                    "Clone voice option must be None; provide reference_audio for one-shot cloning"
                )
            reference = _validate_reference_audio(options.get("reference_audio"))
            if reference_text is not None and not isinstance(reference_text, str):
                raise ConfigurationError("Qwen3 reference_text must be a string or None")
        max_new_tokens = cast(int | None, options.get("max_new_tokens"))
        if max_new_tokens is None:
            # Native max_new_tokens counts 12.5 Hz talker frames.
            frames_per_character = 5 if language in (None, "chinese", "japanese", "korean") else 2
            max_new_tokens = 20 + frames_per_character * len(text)
        return NativeRequest(
            text=text,
            language=language,
            speaker=speaker,
            instruct=cast(str | None, instruct),
            reference_audio=reference,
            reference_text=cast(str | None, reference_text),
            seed=cast(int | None, options.get("seed")),
            max_new_tokens=max_new_tokens,
            do_sample=cast(bool | None, options.get("do_sample")),
            temperature=cast(float | None, options.get("temperature")),
            top_k=cast(int | None, options.get("top_k")),
            top_p=cast(float | None, options.get("top_p")),
            repetition_penalty=cast(float | None, options.get("repetition_penalty")),
        )

    async def _buffered(self, loaded: _LoadedModel, request: NativeRequest) -> AsyncGenerator[AudioData, None]:
        cancelled = threading.Event()
        future = self._native_executor().submit(
            self._binding.synthesize,
            loaded.context,
            request,
            on_chunk=None,
            cancelled=cancelled.is_set,
        )
        wrapped = asyncio.wrap_future(future)
        try:
            native = await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            cancelled.set()
            await _discard_future_result(future, wrapped=wrapped)
            raise
        if native is not None:
            yield _normalize_audio(native, operation="buffered synthesis")

    async def _streaming(self, loaded: _LoadedModel, request: NativeRequest) -> AsyncGenerator[AudioData, None]:
        cancelled = threading.Event()
        items: queue.Queue[_StreamItem] = queue.Queue()

        def on_chunk(chunk: NativeAudio) -> bool:
            if cancelled.is_set():
                return False
            items.put(chunk)
            return not cancelled.is_set()

        def run() -> None:
            try:
                self._binding.synthesize(loaded.context, request, on_chunk=on_chunk, cancelled=cancelled.is_set)
            except BaseException as exc:
                items.put(_StreamEnd(exc))
            else:
                items.put(_StreamEnd())

        future = self._native_executor().submit(run)
        try:
            while True:
                item = await asyncio.to_thread(items.get)
                if isinstance(item, _StreamEnd):
                    await _discard_future_result(future)
                    if item.error is not None:
                        raise item.error
                    return
                chunk = _normalize_audio(item, operation="streaming synthesis")
                if chunk.data.size:
                    yield chunk
        finally:
            cancelled.set()
            await _discard_future_result(future)


class Qwen3CustomVoiceTTSEngine(_Qwen3TTSEngineBase):
    """In-process Qwen3 CustomVoice engine with built-in speakers."""

    _EXPECTED_MODE = "CustomVoice"

    def __init__(
        self, model_source: Qwen3ModelSource, *, default_voice: str | None = None, codec_chunk_sec: float | None = None
    ) -> None:
        if default_voice is not None:
            _validate_non_empty_string(default_voice, option="default_voice")
        super().__init__(model_source, codec_chunk_sec=codec_chunk_sec)
        self.default_voice = default_voice

    def list_voices(self) -> list[dict[str, str]]:
        loaded = self._loaded
        if self.health().state != EngineState.READY or loaded is None:
            raise EngineStateError("Voice listing requires a prepared engine that has not been closed")
        return [{"name": name, "kind": "speaker"} for name in loaded.speakers]


class Qwen3CloneTTSEngine(_Qwen3TTSEngineBase):
    """In-process Qwen3 Base engine with call-local voice references."""

    _EXPECTED_MODE = "Base"


def _reject_decode_option(options: dict[str, object]) -> None:
    if "_qwen3_buffered" in options:
        raise ConfigurationError("Qwen3 native decode mode is internal and cannot be selected by a caller")


def _validate_source_path(value: object, *, option: str) -> None:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"Qwen3 {option} must be a non-empty string")
    if not Path(value).is_absolute():
        raise ConfigurationError(f"Qwen3 {option} must be an absolute path")


def _validate_request_options(
    *,
    text: str,
    model: object,
    voice: object,
    language: object,
    seed: object,
    max_new_tokens: object,
    do_sample: object,
    temperature: object,
    top_k: object,
    top_p: object,
    repetition_penalty: object,
) -> str | None:
    if "<|" in text:
        raise ConfigurationError("Qwen3 text must not contain the reserved '<|' token prefix")
    if model is not None:
        _validate_non_empty_string(model, option="model")
    if voice is not None:
        _validate_non_empty_string(voice, option="voice")
    native_language = _map_language(language)
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ConfigurationError("Qwen3 seed must be a bool-free integer")
    if max_new_tokens is not None:
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise ConfigurationError("Qwen3 max_new_tokens must be an integer")
        if not 1 <= max_new_tokens <= _NATIVE_C_INT_MAX:
            raise ConfigurationError(f"Qwen3 max_new_tokens must be between 1 and {_NATIVE_C_INT_MAX}")
    if do_sample is not None and not isinstance(do_sample, bool):
        raise ConfigurationError("Qwen3 do_sample must be a bool")
    if temperature is not None:
        _validate_positive_float(temperature, option="temperature")
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ConfigurationError("Qwen3 top_k must be an integer")
        if not 0 <= top_k <= _NATIVE_C_INT_MAX:
            raise ConfigurationError(f"Qwen3 top_k must be between 0 and {_NATIVE_C_INT_MAX}")
    if top_p is not None:
        _validate_positive_float(top_p, option="top_p", maximum=1.0)
    if repetition_penalty is not None:
        _validate_positive_float(repetition_penalty, option="repetition_penalty")
    return native_language


def _validate_positive_float(value: object, *, option: str, maximum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"Qwen3 {option} must be a finite positive number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ConfigurationError(f"Qwen3 {option} must be a finite positive number") from exc
    native_number = ctypes.c_float(number).value
    if (
        not math.isfinite(number)
        or number <= 0
        or (maximum is not None and number > maximum)
        or not math.isfinite(native_number)
        or native_number <= 0
        or (maximum is not None and native_number > maximum)
    ):
        limit = f" and at most {maximum}" if maximum is not None else ""
        raise ConfigurationError(f"Qwen3 {option} must be a finite positive number{limit} representable as float32")


def _map_language(language: object) -> str | None:
    if language is None:
        return None
    if isinstance(language, str):
        native = _LANGUAGE_TO_NATIVE.get(language)
        if native is not None:
            return native
        base = language.casefold().split("-", 1)[0].split("_", 1)[0]
        code = _LANGUAGE_SUGGESTIONS.get(base)
        if code is not None:
            raise ConfigurationError(f"Qwen3 language={language!r} is not supported; use {code!r}")
    accepted = ", ".join(_LANGUAGE_TO_NATIVE)
    raise ConfigurationError(f"Qwen3 language={language!r} is not supported; accepted values={accepted}, or None")


def _validate_non_empty_string(value: object, *, option: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"Qwen3 {option} must be a non-empty string")


def _validate_reference_audio(audio: object) -> np.ndarray:
    if not isinstance(audio, AudioData):
        raise ConfigurationError("Qwen3 reference_audio must be an AudioData value")
    if audio.sample_rate != _SAMPLE_RATE:
        raise ConfigurationError("Qwen3 reference_audio must use 24,000 Hz")
    if audio.channels != 1 or audio.data.ndim != 1:
        raise ConfigurationError("Qwen3 reference_audio must be mono with one-dimensional samples")
    if audio.data.dtype != np.float32:
        raise ConfigurationError("Qwen3 reference_audio must contain float32 samples")
    if audio.data.size == 0:
        raise ConfigurationError("Qwen3 reference_audio must be non-empty")
    if not np.isfinite(audio.data).all():
        raise ConfigurationError("Qwen3 reference_audio samples must be finite")
    if np.any(audio.data < -1.0) or np.any(audio.data > 1.0):
        raise ConfigurationError("Qwen3 reference_audio samples must be within [-1, 1]")
    return np.array(audio.data, dtype=np.float32, copy=True, order="C")


def _resolve_speaker(speakers: tuple[str, ...], requested: str | None) -> str:
    if requested is None:
        raise ConfigurationError("Qwen3 CustomVoice synthesis requires a voice or default_voice")
    folded = requested.casefold()
    for speaker in speakers:
        if speaker.casefold() == folded:
            return speaker
    raise ConfigurationError(f"Qwen3 unknown voice={requested!r}; available speakers={list(speakers)!r}")


def _normalize_audio(native: object, *, operation: str) -> AudioData:
    try:
        if not isinstance(native, NativeAudio):
            raise TypeError(f"expected NativeAudio, got {type(native).__name__}")
        if native.sample_rate != _SAMPLE_RATE:
            raise ValueError(f"sample rate must be {_SAMPLE_RATE}, got {native.sample_rate}")
        if native.channels != 1:
            raise ValueError(f"channel count must be 1, got {native.channels}")
        samples = np.asarray(native.samples)
        if samples.ndim != 1:
            raise ValueError(f"sample buffer must be one-dimensional, got shape={samples.shape}")
        converted = np.array(samples, dtype=np.float32, copy=True, order="C")
        np.nan_to_num(converted, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(converted, -1.0, 1.0, out=converted)
    except (TypeError, ValueError) as exc:
        raise InferenceError(f"Qwen3 {operation} returned invalid audio: {exc}") from exc
    return AudioData(data=converted, sample_rate=_SAMPLE_RATE, channels=1, format="pcm")


async def _await_future_to_completion[T](future: Future[T], *, wrapped: asyncio.Future[T] | None = None) -> T:
    wrapped = asyncio.wrap_future(future) if wrapped is None else wrapped
    while not wrapped.done():
        try:
            await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            continue
        except BaseException:
            break
    return wrapped.result()


async def _discard_future_result[T](future: Future[T], *, wrapped: asyncio.Future[T] | None = None) -> None:
    with contextlib.suppress(BaseException):
        await _await_future_to_completion(future, wrapped=wrapped)
