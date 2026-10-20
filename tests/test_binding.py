from __future__ import annotations

import ctypes
import importlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


class _Symbol:
    def __init__(self, result: object = None) -> None:
        self.result = result
        self.argtypes: object = None
        self.restype: object = None
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *args: object) -> object:
        self.calls.append(args)
        return self.result


def _binding_module() -> Any:
    return importlib.import_module("speechflow.qwen3._binding")


def test_public_c_abi_structs_match_qwen_header_field_order_and_types() -> None:
    binding = _binding_module()

    assert binding.QT_ABI_VERSION == 4
    assert [(name, field_type) for name, field_type in binding.QtAudio._fields_] == [
        ("samples", ctypes.POINTER(ctypes.c_float)),
        ("n_samples", ctypes.c_int),
        ("sample_rate", ctypes.c_int),
        ("channels", ctypes.c_int),
    ]
    assert [(name, field_type) for name, field_type in binding.QtInitParams._fields_] == [
        ("abi_version", ctypes.c_int),
        ("talker_path", ctypes.c_char_p),
        ("codec_path", ctypes.c_char_p),
        ("use_fa", ctypes.c_bool),
        ("clamp_fp16", ctypes.c_bool),
        ("max_batch", ctypes.c_int),
        ("codec_chunk_sec", ctypes.c_float),
    ]
    assert [(name, field_type) for name, field_type in binding.QtVoiceRef._fields_] == [
        ("ref_spk_emb", ctypes.POINTER(ctypes.c_float)),
        ("ref_spk_dim", ctypes.c_int),
        ("ref_codes", ctypes.POINTER(ctypes.c_int32)),
        ("ref_T", ctypes.c_int),
        ("num_codebooks", ctypes.c_int),
    ]
    assert [(name, field_type) for name, field_type in binding.QtTtsParams._fields_] == [
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
        ("cancel", binding.QtCancelCallback),
        ("cancel_user_data", ctypes.c_void_p),
        ("on_chunk", binding.QtAudioChunkCallback),
        ("on_chunk_user_data", ctypes.c_void_p),
        ("ref_spk_emb", ctypes.POINTER(ctypes.c_float)),
        ("ref_spk_dim", ctypes.c_int),
        ("ref_codes", ctypes.POINTER(ctypes.c_int32)),
        ("ref_T", ctypes.c_int),
    ]


def test_every_qt_symbol_receives_a_ctypes_prototype() -> None:
    binding = _binding_module()
    names = (
        "qt_version",
        "qt_last_error",
        "qt_audio_free",
        "qt_init_default_params",
        "qt_init",
        "qt_free",
        "qt_extract_voice_ref",
        "qt_voice_ref_free",
        "qt_log_set",
        "qt_tts_default_params",
        "qt_num_codebooks",
        "qt_synthesize",
        "qt_duration_sec_to_tokens",
        "qt_n_speakers",
        "qt_speaker_name",
    )
    library = SimpleNamespace(**{name: _Symbol() for name in names})

    binding.configure_library(library)

    expected = {
        "qt_version": ([], ctypes.c_char_p),
        "qt_last_error": ([], ctypes.c_char_p),
        "qt_audio_free": ([ctypes.POINTER(binding.QtAudio)], None),
        "qt_init_default_params": ([ctypes.POINTER(binding.QtInitParams)], None),
        "qt_init": ([ctypes.POINTER(binding.QtInitParams)], ctypes.c_void_p),
        "qt_free": ([ctypes.c_void_p], None),
        "qt_extract_voice_ref": (
            [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.POINTER(binding.QtVoiceRef),
            ],
            ctypes.c_int,
        ),
        "qt_voice_ref_free": ([ctypes.POINTER(binding.QtVoiceRef)], None),
        "qt_log_set": ([binding.QtLogCallback, ctypes.c_void_p], None),
        "qt_tts_default_params": ([ctypes.POINTER(binding.QtTtsParams)], None),
        "qt_num_codebooks": ([ctypes.c_void_p], ctypes.c_int),
        "qt_synthesize": (
            [ctypes.c_void_p, ctypes.POINTER(binding.QtTtsParams), ctypes.POINTER(binding.QtAudio)],
            ctypes.c_int,
        ),
        "qt_duration_sec_to_tokens": ([ctypes.c_void_p, ctypes.c_float], ctypes.c_int),
        "qt_n_speakers": ([ctypes.c_void_p], ctypes.c_int),
        "qt_speaker_name": ([ctypes.c_void_p, ctypes.c_int], ctypes.c_char_p),
    }
    assert set(expected) == set(names)
    for name, (argtypes, restype) in expected.items():
        symbol = getattr(library, name)
        assert symbol.argtypes == argtypes
        assert symbol.restype is restype


def test_ctypes_binding_uses_utf8_defaults_and_always_frees_buffered_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    binding = _binding_module()
    samples = (ctypes.c_float * 3)(-2.0, 0.25, 2.0)
    freed: list[int] = []
    captured: dict[str, object] = {}

    def default_params(raw: Any) -> None:
        params = ctypes.cast(raw, ctypes.POINTER(binding.QtTtsParams)).contents
        params.abi_version = binding.QT_ABI_VERSION
        params.seed = -1
        params.max_new_tokens = 2048

    def synthesize(_context: object, raw_params: Any, raw_audio: Any) -> int:
        params = ctypes.cast(raw_params, ctypes.POINTER(binding.QtTtsParams)).contents
        captured.update(
            text=params.text,
            lang=params.lang,
            speaker=params.speaker,
            seed=params.seed,
            max_new_tokens=params.max_new_tokens,
        )
        audio = ctypes.cast(raw_audio, ctypes.POINTER(binding.QtAudio)).contents
        audio.samples = ctypes.cast(samples, ctypes.POINTER(ctypes.c_float))
        audio.n_samples = 3
        audio.sample_rate = 24000
        audio.channels = 1
        return 0

    library = SimpleNamespace(
        qt_tts_default_params=_Symbol(),
        qt_synthesize=_Symbol(),
        qt_audio_free=_Symbol(),
        qt_last_error=_Symbol(b""),
    )
    library.qt_tts_default_params.__call__ = default_params
    library.qt_synthesize.__call__ = synthesize
    monkeypatch.setattr(library, "qt_tts_default_params", default_params)
    monkeypatch.setattr(library, "qt_synthesize", synthesize)
    monkeypatch.setattr(library, "qt_audio_free", lambda raw: freed.append(id(raw)))

    runtime = binding.CtypesNativeBinding()
    runtime._library = library
    request = binding.NativeRequest(
        text="こんにちは",
        language="japanese",
        speaker="Ryan",
        instruct=None,
        reference_audio=None,
        reference_text=None,
        seed=7,
        max_new_tokens=99,
    )

    result = runtime.synthesize(object(), request, on_chunk=None, cancelled=lambda: False)

    assert captured == {
        "text": "こんにちは".encode(),
        "lang": b"japanese",
        "speaker": b"Ryan",
        "seed": 7,
        "max_new_tokens": 99,
    }
    assert result is not None
    np.testing.assert_array_equal(result.samples, np.array([-2.0, 0.25, 2.0], dtype=np.float32))
    assert len(freed) == 1


@pytest.mark.req("SP-SFQWEN3-023")
def test_codec_chunk_sec_overrides_only_the_native_init_default() -> None:
    """The binding passes a requested codec width while preserving the native default when omitted."""
    binding = _binding_module()
    captured: list[float] = []

    def init_defaults(raw: Any) -> None:
        params = ctypes.cast(raw, ctypes.POINTER(binding.QtInitParams)).contents
        params.codec_chunk_sec = 19.5

    def initialize(raw: Any) -> object:
        params = ctypes.cast(raw, ctypes.POINTER(binding.QtInitParams)).contents
        captured.append(params.codec_chunk_sec)
        return object()

    runtime = binding.CtypesNativeBinding()
    runtime._library = SimpleNamespace(qt_init_default_params=init_defaults, qt_init=initialize)

    runtime.load_model("/talker.gguf", "/codec.gguf")
    runtime.load_model("/talker.gguf", "/codec.gguf", codec_chunk_sec=3.5)

    assert captured == [19.5, 3.5]


@pytest.mark.req("SP-SFQWEN3-023")
def test_sampling_options_override_only_the_native_tts_defaults() -> None:
    """The binding sends requested sampling fields and otherwise retains values from qt_tts_default_params."""
    binding = _binding_module()
    captured: list[dict[str, bool | int | float]] = []

    def tts_defaults(raw: Any) -> None:
        params = ctypes.cast(raw, ctypes.POINTER(binding.QtTtsParams)).contents
        params.do_sample = True
        params.temperature = 0.875
        params.top_k = 37
        params.top_p = 0.75
        params.repetition_penalty = 1.125

    def synthesize(_context: object, raw_params: Any, _raw_audio: Any) -> int:
        params = ctypes.cast(raw_params, ctypes.POINTER(binding.QtTtsParams)).contents
        captured.append(
            {
                "do_sample": params.do_sample,
                "temperature": params.temperature,
                "top_k": params.top_k,
                "top_p": params.top_p,
                "repetition_penalty": params.repetition_penalty,
            }
        )
        return 0

    runtime = binding.CtypesNativeBinding()
    runtime._library = SimpleNamespace(
        qt_tts_default_params=tts_defaults,
        qt_synthesize=synthesize,
        qt_audio_free=lambda _raw: None,
    )
    request = binding.NativeRequest("text", None, None, None, None, None, None, None)
    specified = binding.NativeRequest(
        "text",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        do_sample=False,
        temperature=1.25,
        top_k=0,
        top_p=0.5,
        repetition_penalty=1.5,
    )

    runtime.synthesize(object(), request, on_chunk=None, cancelled=lambda: False)
    runtime.synthesize(object(), specified, on_chunk=None, cancelled=lambda: False)

    assert captured == [
        {"do_sample": True, "temperature": 0.875, "top_k": 37, "top_p": 0.75, "repetition_penalty": 1.125},
        {"do_sample": False, "temperature": 1.25, "top_k": 0, "top_p": 0.5, "repetition_penalty": 1.5},
    ]


def test_voice_reference_helper_copies_results_and_always_frees_native_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    binding = _binding_module()
    embedding = (ctypes.c_float * 2)(0.25, 0.5)
    codes = (ctypes.c_int32 * 4)(1, 2, 3, 4)
    freed = 0

    def extract(_context: object, _samples: object, _count: int, raw_ref: Any) -> int:
        ref = ctypes.cast(raw_ref, ctypes.POINTER(binding.QtVoiceRef)).contents
        ref.ref_spk_emb = ctypes.cast(embedding, ctypes.POINTER(ctypes.c_float))
        ref.ref_spk_dim = 2
        ref.ref_codes = ctypes.cast(codes, ctypes.POINTER(ctypes.c_int32))
        ref.ref_T = 2
        ref.num_codebooks = 2
        return 0

    def free(_raw_ref: object) -> None:
        nonlocal freed
        freed += 1

    runtime = binding.CtypesNativeBinding()
    runtime._library = SimpleNamespace(
        qt_extract_voice_ref=extract,
        qt_voice_ref_free=free,
        qt_last_error=lambda: b"",
    )

    result = runtime.extract_voice_ref(object(), np.array([0.0, 0.5], dtype=np.float32))

    np.testing.assert_array_equal(result.speaker_embedding, np.array([0.25, 0.5], dtype=np.float32))
    np.testing.assert_array_equal(result.codes, np.array([[1, 2], [3, 4]], dtype=np.int32))
    assert freed == 1
