from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from speechflow.core import AudioData

RUN_GPU = os.environ.get("SPEECHFLOW_QWEN3_RUN_GPU") == "1"
WEIGHTS_ROOT = Path(os.environ.get("SPEECHFLOW_QWEN3_WEIGHTS_DIR", "/weights-not-configured"))


@pytest.mark.req("SP-SFQWEN3-027")
@pytest.mark.gpu
@pytest.mark.skipif(not RUN_GPU, reason="set SPEECHFLOW_QWEN3_RUN_GPU=1 and SPEECHFLOW_QWEN3_WEIGHTS_DIR")
@pytest.mark.parametrize("method", ["get", "stream"])
@pytest.mark.parametrize(("text", "language", "seed", "max_frames"), [("はい。", "ja", 0, 35), ("Hey!", "en", 14, 28)])
async def test_customvoice_short_text_stops_within_default_length_cap(
    method: str, text: str, language: str, seed: int, max_frames: int
) -> None:
    """The 1.7B Q8_0 CustomVoice model caps short utterances without an explicit token limit."""
    from speechflow.qwen3 import Qwen3CustomVoiceTTSEngine, Qwen3ModelSource

    engine = Qwen3CustomVoiceTTSEngine(
        Qwen3ModelSource(
            str(WEIGHTS_ROOT / "qwen-talker-1.7b-customvoice-Q8_0.gguf"),
            str(WEIGHTS_ROOT / "qwen-tokenizer-12hz-Q8_0.gguf"),
        ),
        default_voice="Vivian",
    )
    try:
        await engine.prepare()
        if method == "get":
            audio = await engine.get(text, language=language, seed=seed)
            sample_count = audio.data.size
        else:
            sample_count = sum([chunk.data.size async for chunk in engine.stream(text, language=language, seed=seed)])

        assert 0 < sample_count <= max_frames * 1920  # 24,000 Hz / 12.5 frames per second.
    finally:
        await engine.aclose()


@pytest.mark.req("SP-SFQWEN3-011")
@pytest.mark.req("SP-SFQWEN3-012")
@pytest.mark.gpu
@pytest.mark.skipif(not RUN_GPU, reason="set SPEECHFLOW_QWEN3_RUN_GPU=1 and SPEECHFLOW_QWEN3_WEIGHTS_DIR")
async def test_customvoice_and_base_q4_gpu_smoke() -> None:
    """CustomVoice and Base synthesize nonempty normalized mono audio at 24 kHz on a configured GPU."""
    from speechflow.qwen3 import Qwen3CloneTTSEngine, Qwen3CustomVoiceTTSEngine, Qwen3ModelSource

    codec = WEIGHTS_ROOT / "qwen-tokenizer-12hz-Q4_K_M.gguf"
    custom = Qwen3CustomVoiceTTSEngine(
        Qwen3ModelSource(
            str(WEIGHTS_ROOT / "qwen-talker-0.6b-customvoice-Q4_K_M.gguf"),
            str(codec),
        ),
        default_voice="ryan",
    )
    base = Qwen3CloneTTSEngine(
        Qwen3ModelSource(
            str(WEIGHTS_ROOT / "qwen-talker-0.6b-base-Q4_K_M.gguf"),
            str(codec),
        )
    )
    result: dict[str, object] = {}
    try:
        started = time.monotonic()
        await custom.prepare()
        result["custom_load_s"] = time.monotonic() - started
        reference = await custom.get("Reference audio for voice cloning.", voice="ryan", max_new_tokens=64)
        result["custom_get_samples"] = reference.data.size
        started = time.monotonic()
        custom_chunks = [chunk async for chunk in custom.stream("Short streaming smoke.", max_new_tokens=64)]
        result["custom_stream_samples"] = sum(chunk.data.size for chunk in custom_chunks)
        result["custom_stream_s"] = time.monotonic() - started
        await custom.aclose()

        started = time.monotonic()
        await base.prepare()
        result["base_load_s"] = time.monotonic() - started
        cloned = await base.get(
            "This voice was cloned successfully.",
            reference_audio=reference,
            reference_text="Reference audio for voice cloning.",
            max_new_tokens=64,
        )
        result["base_get_samples"] = cloned.data.size
        started = time.monotonic()
        base_chunks = [
            chunk
            async for chunk in base.stream(
                "Short cloned streaming smoke.",
                reference_audio=reference,
                reference_text="Reference audio for voice cloning.",
                max_new_tokens=64,
            )
        ]
        result["base_stream_samples"] = sum(chunk.data.size for chunk in base_chunks)
        result["base_stream_s"] = time.monotonic() - started
        for audio in (reference, cloned, *custom_chunks, *base_chunks):
            assert isinstance(audio, AudioData)
            assert audio.sample_rate == 24000 and audio.channels == 1
            assert audio.data.dtype == np.float32 and np.isfinite(audio.data).all()
            assert audio.data.size and np.max(np.abs(audio.data)) <= 1.0
        print(json.dumps(result, sort_keys=True))
    finally:
        await custom.aclose()
        await base.aclose()
