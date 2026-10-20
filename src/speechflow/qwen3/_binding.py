from __future__ import annotations

import ctypes
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Protocol, cast

import numpy as np
from speechflow.core import EnvironmentUnavailableError

QT_ABI_VERSION = 4


class QtStatus(IntEnum):
    OK = 0
    INVALID_PARAMS = -1
    MODE_INVALID = -2
    GENERATE_FAILED = -3
    OOM = -4
    CANCELLED = -5


class QtLogLevel(IntEnum):
    DEBUG = 0
    INFO = 1
    WARN = 2
    ERROR = 3


QT_STATUS_OK = QtStatus.OK
QT_STATUS_INVALID_PARAMS = QtStatus.INVALID_PARAMS
QT_STATUS_MODE_INVALID = QtStatus.MODE_INVALID
QT_STATUS_GENERATE_FAILED = QtStatus.GENERATE_FAILED
QT_STATUS_OOM = QtStatus.OOM
QT_STATUS_CANCELLED = QtStatus.CANCELLED
QT_LOG_DEBUG = QtLogLevel.DEBUG
QT_LOG_INFO = QtLogLevel.INFO
QT_LOG_WARN = QtLogLevel.WARN
QT_LOG_ERROR = QtLogLevel.ERROR

QtCancelCallback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p)
QtAudioChunkCallback = ctypes.CFUNCTYPE(
    ctypes.c_bool,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_void_p,
)
QtLogCallback = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)


class QtAudio(ctypes.Structure):
    _fields_ = [
        ("samples", ctypes.POINTER(ctypes.c_float)),
        ("n_samples", ctypes.c_int),
        ("sample_rate", ctypes.c_int),
        ("channels", ctypes.c_int),
    ]


class QtInitParams(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_int),
        ("talker_path", ctypes.c_char_p),
        ("codec_path", ctypes.c_char_p),
        ("use_fa", ctypes.c_bool),
        ("clamp_fp16", ctypes.c_bool),
        ("max_batch", ctypes.c_int),
        ("codec_chunk_sec", ctypes.c_float),
    ]


class QtVoiceRef(ctypes.Structure):
    _fields_ = [
        ("ref_spk_emb", ctypes.POINTER(ctypes.c_float)),
        ("ref_spk_dim", ctypes.c_int),
        ("ref_codes", ctypes.POINTER(ctypes.c_int32)),
        ("ref_T", ctypes.c_int),
        ("num_codebooks", ctypes.c_int),
    ]


class QtTtsParams(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_int),
        ("text", ctypes.c_char_p),
        ("lang", ctypes.c_char_p),
        ("instruct", ctypes.c_char_p),
        ("speaker", ctypes.c_char_p),
        ("ref_audio_24k", ctypes.POINTER(ctypes.c_float)),
        ("ref_n_samples", ctypes.c_int),
        ("ref_text", ctypes.c_char_p),
        ("seed", ctypes.c_int64),
        ("max_new_tokens", ctypes.c_int),
        ("do_sample", ctypes.c_bool),
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_int),
        ("top_p", ctypes.c_float),
        ("repetition_penalty", ctypes.c_float),
        ("subtalker_do_sample", ctypes.c_bool),
        ("subtalker_temperature", ctypes.c_float),
        ("subtalker_top_k", ctypes.c_int),
        ("subtalker_top_p", ctypes.c_float),
        ("dump_dir", ctypes.c_char_p),
        ("cancel", QtCancelCallback),
        ("cancel_user_data", ctypes.c_void_p),
        ("on_chunk", QtAudioChunkCallback),
        ("on_chunk_user_data", ctypes.c_void_p),
        ("ref_spk_emb", ctypes.POINTER(ctypes.c_float)),
        ("ref_spk_dim", ctypes.c_int),
        ("ref_codes", ctypes.POINTER(ctypes.c_int32)),
        ("ref_T", ctypes.c_int),
    ]


@dataclass
class NativeAudio:
    samples: np.ndarray
    sample_rate: int
    channels: int


@dataclass
class NativeRequest:
    text: str
    language: str | None
    speaker: str | None
    instruct: str | None
    reference_audio: np.ndarray | None
    reference_text: str | None
    seed: int | None
    max_new_tokens: int | None
    do_sample: bool | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None


@dataclass
class NativeVoiceReference:
    speaker_embedding: np.ndarray
    codes: np.ndarray


class NativeCallError(RuntimeError):
    def __init__(self, operation: str, status: int | None, diagnostic: str) -> None:
        self.operation = operation
        self.status = status
        self.diagnostic = diagnostic
        status_text = "null" if status is None else str(status)
        super().__init__(f"native operation={operation}; status={status_text}; diagnostic={diagnostic or '<empty>'}")


class NativeBinding(Protocol):
    def load_model(self, talker_path: str, codec_path: str, *, codec_chunk_sec: float | None = None) -> object: ...

    def free_context(self, context: object) -> None: ...

    def list_speakers(self, context: object) -> tuple[str, ...]: ...

    def synthesize(
        self,
        context: object,
        request: NativeRequest,
        *,
        on_chunk: Callable[[NativeAudio], bool] | None,
        cancelled: Callable[[], bool],
    ) -> NativeAudio | None: ...


def configure_library(library: Any) -> Any:
    """Attach the qwen.h prototypes to one loaded ctypes handle."""
    _prototype(library, "qt_version", [], ctypes.c_char_p)
    _prototype(library, "qt_last_error", [], ctypes.c_char_p)
    _prototype(library, "qt_audio_free", [ctypes.POINTER(QtAudio)], None)
    _prototype(library, "qt_init_default_params", [ctypes.POINTER(QtInitParams)], None)
    _prototype(library, "qt_init", [ctypes.POINTER(QtInitParams)], ctypes.c_void_p)
    _prototype(library, "qt_free", [ctypes.c_void_p], None)
    _prototype(
        library,
        "qt_extract_voice_ref",
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(QtVoiceRef)],
        ctypes.c_int,
    )
    _prototype(library, "qt_voice_ref_free", [ctypes.POINTER(QtVoiceRef)], None)
    _prototype(library, "qt_log_set", [QtLogCallback, ctypes.c_void_p], None)
    _prototype(library, "qt_tts_default_params", [ctypes.POINTER(QtTtsParams)], None)
    _prototype(library, "qt_num_codebooks", [ctypes.c_void_p], ctypes.c_int)
    _prototype(
        library,
        "qt_synthesize",
        [ctypes.c_void_p, ctypes.POINTER(QtTtsParams), ctypes.POINTER(QtAudio)],
        ctypes.c_int,
    )
    _prototype(library, "qt_duration_sec_to_tokens", [ctypes.c_void_p, ctypes.c_float], ctypes.c_int)
    _prototype(library, "qt_n_speakers", [ctypes.c_void_p], ctypes.c_int)
    _prototype(library, "qt_speaker_name", [ctypes.c_void_p, ctypes.c_int], ctypes.c_char_p)
    return library


def _prototype(library: Any, name: str, argtypes: list[object], restype: object) -> None:
    try:
        function = getattr(library, name)
        function.argtypes = argtypes
        function.restype = restype
    except (AttributeError, TypeError) as exc:
        error = EnvironmentUnavailableError(
            "why=the native payload is missing a required qwen.h ABI symbol; "
            f"what=symbol={name}; how=rebuild the native payload with scripts/build_native.sh; see README.md"
        )
        raise error from exc


class CtypesNativeBinding:
    """Private ctypes adapter for the vendored qwen.h ABI."""

    def __init__(self) -> None:
        self._library: Any | None = None

    def load_model(self, talker_path: str, codec_path: str, *, codec_chunk_sec: float | None = None) -> object:
        library = self._require_library()
        params = QtInitParams()
        library.qt_init_default_params(ctypes.byref(params))
        params.abi_version = QT_ABI_VERSION
        talker_bytes = talker_path.encode("utf-8")
        codec_bytes = codec_path.encode("utf-8")
        params.talker_path = talker_bytes
        params.codec_path = codec_bytes
        if codec_chunk_sec is not None:
            params.codec_chunk_sec = codec_chunk_sec
        context = library.qt_init(ctypes.byref(params))
        if not context:
            raise self._error("qt_init", None)
        return context

    def free_context(self, context: object) -> None:
        self._require_library().qt_free(context)

    def list_speakers(self, context: object) -> tuple[str, ...]:
        library = self._require_library()
        count = int(library.qt_n_speakers(context))
        if count < 0:
            raise self._error("qt_n_speakers", count)
        names: list[str] = []
        for index in range(count):
            raw = cast(bytes | None, library.qt_speaker_name(context, index))
            if not raw:
                raise NativeCallError("qt_speaker_name", None, f"speaker index {index} returned NULL")
            try:
                names.append(raw.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise NativeCallError("qt_speaker_name", None, f"speaker index {index} is not UTF-8") from exc
        return tuple(names)

    def synthesize(
        self,
        context: object,
        request: NativeRequest,
        *,
        on_chunk: Callable[[NativeAudio], bool] | None,
        cancelled: Callable[[], bool],
    ) -> NativeAudio | None:
        library = self._require_library()
        params = QtTtsParams()
        library.qt_tts_default_params(ctypes.byref(params))
        params.abi_version = QT_ABI_VERSION
        keepalive = self._apply_request(params, request)
        cancel_callback = QtCancelCallback(lambda _user_data: bool(cancelled()))
        params.cancel = cancel_callback
        callback_error: list[BaseException] = []
        chunk_callback: Any | None = None
        if on_chunk is not None:

            def emit(samples: Any, count: int, _user_data: object) -> bool:
                if count < 0 or (count and not samples):
                    callback_error.append(ValueError(f"invalid native streaming buffer length={count}"))
                    return False
                if count == 0:
                    return not cancelled()
                try:
                    copied = np.ctypeslib.as_array(samples, shape=(count,)).astype(np.float32, copy=True)
                    return bool(on_chunk(NativeAudio(copied, 24000, 1))) and not cancelled()
                except BaseException as exc:
                    callback_error.append(exc)
                    return False

            chunk_callback = QtAudioChunkCallback(emit)
            params.on_chunk = chunk_callback

        audio = QtAudio()
        try:
            status = int(library.qt_synthesize(context, ctypes.byref(params), ctypes.byref(audio)))
            if callback_error:
                raise NativeCallError("qt_synthesize callback", status, str(callback_error[0])) from callback_error[0]
            if status != QT_STATUS_OK:
                raise self._error("qt_synthesize", status)
            if on_chunk is not None:
                return None
            return self._copy_audio(audio)
        finally:
            library.qt_audio_free(ctypes.byref(audio))
            _ = keepalive, cancel_callback, chunk_callback

    def extract_voice_ref(self, context: object, samples: np.ndarray) -> NativeVoiceReference:
        library = self._require_library()
        contiguous = np.ascontiguousarray(samples, dtype=np.float32)
        reference = QtVoiceRef()
        try:
            status = int(
                library.qt_extract_voice_ref(
                    context,
                    contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    contiguous.size,
                    ctypes.byref(reference),
                )
            )
            if status != QT_STATUS_OK:
                raise self._error("qt_extract_voice_ref", status)
            if (
                reference.ref_spk_dim <= 0
                or reference.ref_T <= 0
                or reference.num_codebooks <= 0
                or not reference.ref_spk_emb
                or not reference.ref_codes
            ):
                raise NativeCallError("qt_extract_voice_ref", status, "native voice reference has invalid dimensions")
            embedding = np.ctypeslib.as_array(reference.ref_spk_emb, shape=(reference.ref_spk_dim,)).copy()
            code_count = reference.ref_T * reference.num_codebooks
            codes = np.ctypeslib.as_array(reference.ref_codes, shape=(code_count,)).copy()
            return NativeVoiceReference(
                speaker_embedding=embedding.astype(np.float32, copy=False),
                codes=codes.astype(np.int32, copy=False).reshape(reference.num_codebooks, reference.ref_T),
            )
        finally:
            library.qt_voice_ref_free(ctypes.byref(reference))

    def _require_library(self) -> Any:
        if self._library is None:
            from ._payload import load_native

            self._library = configure_library(load_native())
        return self._library

    @staticmethod
    def _apply_request(params: QtTtsParams, request: NativeRequest) -> tuple[object, ...]:
        values: list[object] = []

        def assign(name: str, value: str | None) -> None:
            if value is not None:
                raw = value.encode("utf-8")
                values.append(raw)
                setattr(params, name, raw)

        assign("text", request.text)
        assign("lang", request.language)
        assign("instruct", request.instruct)
        assign("speaker", request.speaker)
        assign("ref_text", request.reference_text)
        if request.seed is not None:
            params.seed = request.seed
        if request.max_new_tokens is not None:
            params.max_new_tokens = request.max_new_tokens
        if request.do_sample is not None:
            params.do_sample = request.do_sample
        if request.temperature is not None:
            params.temperature = request.temperature
        if request.top_k is not None:
            params.top_k = request.top_k
        if request.top_p is not None:
            params.top_p = request.top_p
        if request.repetition_penalty is not None:
            params.repetition_penalty = request.repetition_penalty
        if request.reference_audio is not None:
            reference = np.ascontiguousarray(request.reference_audio, dtype=np.float32)
            params.ref_audio_24k = reference.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            params.ref_n_samples = reference.size
            values.append(reference)
        return tuple(values)

    @staticmethod
    def _copy_audio(audio: QtAudio) -> NativeAudio:
        if audio.n_samples < 0 or (audio.n_samples and not audio.samples):
            raise NativeCallError("qt_synthesize output", None, f"invalid sample buffer length={audio.n_samples}")
        if audio.n_samples == 0:
            samples = np.empty(0, dtype=np.float32)
        else:
            samples = np.ctypeslib.as_array(audio.samples, shape=(audio.n_samples,)).astype(np.float32, copy=True)
        return NativeAudio(samples=samples, sample_rate=int(audio.sample_rate), channels=int(audio.channels))

    def _error(self, operation: str, status: int | None) -> NativeCallError:
        library = self._require_library()
        try:
            raw = cast(bytes | None, library.qt_last_error())
            diagnostic = raw.decode("utf-8") if raw else ""
        except (AttributeError, OSError, TypeError, UnicodeDecodeError) as exc:
            raise NativeCallError(operation, status, "native diagnostic is unavailable or invalid UTF-8") from exc
        return NativeCallError(operation, status, diagnostic)
