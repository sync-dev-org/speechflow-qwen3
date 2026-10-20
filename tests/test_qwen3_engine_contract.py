"""Concrete async engine contracts owned by the Qwen3 module."""

from __future__ import annotations

import inspect
from collections.abc import Callable

import pytest

from speechflow import EngineCapabilities, TTSEngineBase
from speechflow.qwen3 import Qwen3CloneTTSEngine, Qwen3CustomVoiceTTSEngine, Qwen3ModelSource

ENGINE_TYPES = (Qwen3CustomVoiceTTSEngine, Qwen3CloneTTSEngine)


def _assert_unified_signature(method: Callable[..., object]) -> None:
    parameters = list(inspect.signature(method).parameters.values())
    assert [parameter.name for parameter in parameters[:4]] == ["self", "text", "model", "voice"]
    assert parameters[2].default is None
    assert parameters[3].default is None


@pytest.mark.req("SP-SF-003")
@pytest.mark.req("SP-SF-053")
@pytest.mark.parametrize("engine_type", ENGINE_TYPES)
def test_qwen3_get_is_a_coroutine_with_the_unified_signature(engine_type: type[TTSEngineBase]) -> None:
    """Both Qwen3 get operations are async and accept text, model, and voice."""
    assert inspect.iscoroutinefunction(engine_type.get)
    _assert_unified_signature(engine_type.get)


@pytest.mark.req("SP-SF-003")
@pytest.mark.req("SP-SF-053")
@pytest.mark.parametrize("engine_type", ENGINE_TYPES)
def test_qwen3_stream_is_an_async_generator_with_the_unified_signature(engine_type: type[TTSEngineBase]) -> None:
    """Both Qwen3 stream operations return async generators with text, model, and voice."""
    assert inspect.isasyncgenfunction(engine_type.stream)
    _assert_unified_signature(engine_type.stream)


@pytest.mark.req("SP-SF-003")
@pytest.mark.parametrize("engine_type", ENGINE_TYPES)
def test_qwen3_engines_inherit_the_core_base(engine_type: type[TTSEngineBase]) -> None:
    """Both public Qwen3 engines implement the core engine base."""
    assert issubclass(engine_type, TTSEngineBase)


@pytest.mark.req("SP-SF-059")
@pytest.mark.parametrize("engine_type", ENGINE_TYPES)
def test_capabilities_and_health_are_available_without_loading(
    engine_type: type[Qwen3CustomVoiceTTSEngine] | type[Qwen3CloneTTSEngine],
) -> None:
    """Each engine declares its languages, voice support, streaming, and sample rate before preparation."""
    engine = engine_type(Qwen3ModelSource("/talker.gguf", "/codec.gguf"))

    assert engine.capabilities == EngineCapabilities(
        languages=frozenset({"zh", "en", "de", "it", "pt", "es", "ja", "ko", "fr", "ru"}),
        auto_language=True,
        streaming=True,
        voice_listing=engine_type is Qwen3CustomVoiceTTSEngine,
        voice_cloning=engine_type is Qwen3CloneTTSEngine,
        sample_rate=24000,
    )
    assert engine.health().state == "unprepared"
    assert engine.health().busy is False
    assert engine.health().last_progress_monotonic_s is None


@pytest.mark.req("SP-SF-054")
@pytest.mark.parametrize("engine_type", ENGINE_TYPES)
def test_engine_exposes_async_work_and_immediate_reads(engine_type: type[TTSEngineBase]) -> None:
    """Engine work uses the core's async operations and voice listing is an immediate read."""
    assert inspect.iscoroutinefunction(engine_type.prepare)
    assert inspect.iscoroutinefunction(engine_type.aclose)
    assert not any(name.endswith("_sync") for name in dir(engine_type))
    if engine_type is Qwen3CustomVoiceTTSEngine:
        assert not inspect.iscoroutinefunction(engine_type.list_voices)
    else:
        assert not hasattr(engine_type, "list_voices")
