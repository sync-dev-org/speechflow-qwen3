# speechflow-qwen3

Qwen3-TTS の qwentts.cpp runtime を Python process 内で実行する SpeechFlow engine module です。配布対象は Linux x86_64、glibc 2.28 以上、CUDA 13.2 以上に対応する NVIDIA GPU です。

## インストール

Python 3.12 / 3.13 で、[PyPI](https://pypi.org/project/speechflow-qwen3/) の wheel をインストールします。

```bash
pip install speechflow-qwen3
```

## 対応環境

CPU ISA の下限は x86-64-v3 (AVX2 / FMA / F16C) です。producer 固有の CPU 命令へ最適化せず、AVX-512 は有効にしません。

NVIDIA driver は host が提供し、CUDA 13.2 以上に対応する必要があります。CUDA 13.2 以上の user-space library (`libcudart.so.13`、`libcublas.so.13`、`libcublasLt.so.13`) は Python dependency が提供します。CUDA 13.2 GA に対応する package の下限は `nvidia-cuda-runtime` が 13.2、`nvidia-cublas` が 13.3 で、どちらも 14 未満です。runtime host に system CUDA toolkit、compiler、CMake、Ninja、外部 build directory は必要ありません。

wheel は、compute capability 8.6、compute capability 8.9、compute capability 12.0 の real cubin を同じ wheel に含めます。

| Architecture | Compute capability | Representative GPU | Native target | Runtime evidence | Record |
|---|---:|---|---|---|---|
| `sm_86` | 8.6 | NVIDIA RTX A6000 | included | qualified | [qwen3-tts-0.6b-customvoice-Q4_K_M--sm_86.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-0.6b-customvoice-Q4_K_M--sm_86.json) |
| `sm_89` | 8.9 | NVIDIA RTX 6000 Ada | included | qualified | [qwen3-tts-0.6b-customvoice-Q4_K_M--sm_89.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-0.6b-customvoice-Q4_K_M--sm_89.json) |
| `sm_120a` | 12.0 | NVIDIA RTX PRO 4000 Blackwell | included | qualified | [qwen3-tts-0.6b-base-Q4_K_M--sm_120a.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-0.6b-base-Q4_K_M--sm_120a.json), [qwen3-tts-0.6b-base-Q8_0--sm_120a.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-0.6b-base-Q8_0--sm_120a.json), [qwen3-tts-0.6b-customvoice-Q4_K_M--sm_120a.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-0.6b-customvoice-Q4_K_M--sm_120a.json), [qwen3-tts-0.6b-customvoice-Q8_0--sm_120a.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-0.6b-customvoice-Q8_0--sm_120a.json), [qwen3-tts-1.7b-base-Q4_K_M--sm_120a.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-1.7b-base-Q4_K_M--sm_120a.json), [qwen3-tts-1.7b-base-Q8_0--sm_120a.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-1.7b-base-Q8_0--sm_120a.json), [qwen3-tts-1.7b-customvoice-Q4_K_M--sm_120a.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-1.7b-customvoice-Q4_K_M--sm_120a.json), [qwen3-tts-1.7b-customvoice-Q8_0--sm_120a.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-1.7b-customvoice-Q8_0--sm_120a.json) |

wheel は PTX を同梱しません。したがって未列挙の GPU 世代は JIT target を含めて対象外です。`included` は native target が archive contract に合格したことだけを表し、実機での成功は [qualified evidence](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-evidence.md) に pass record が採用されるまで `pending` です。

表の `qualified` は、link した記録に記載された version と wheel で動作を確かめたことを表します。記録の wheel の tag と SHA-256 は、その測定時の値を保持しています。

## Engine の使い方

weights は wheel に含まれません。`Qwen3ModelSource` には talker GGUF と同じ quantization の codec GGUF の absolute path を渡します。constructor は path の構造だけを検査します。`await engine.prepare()` は wheel 同梱 `weights-manifest-v1.json` の basename、byte size、SHA-256 と weights を照合し、model の読み込みと短い試しの推論を行います。manifest の差し替えや照合 bypass はできません。

CustomVoice checkpoint は built-in speaker を使います。

```python
import asyncio

from speechflow.qwen3 import Qwen3CustomVoiceTTSEngine, Qwen3ModelSource


async def main() -> None:
    engine = Qwen3CustomVoiceTTSEngine(
        Qwen3ModelSource(
            talker_path="/models/qwen-talker-0.6b-customvoice-Q4_K_M.gguf",
            codec_path="/models/qwen-tokenizer-12hz-Q4_K_M.gguf",
        ),
        default_voice="ryan",
    )
    try:
        await engine.prepare()
        print(engine.list_voices())
        audio = await engine.get("こんにちは。", language="ja", max_new_tokens=256)
        print(audio.sample_rate, audio.data.shape)
        async for chunk in engine.stream("Streaming synthesis.", voice="Ryan"):
            print(chunk.data.shape)
    finally:
        await engine.aclose()


asyncio.run(main())
```

Base checkpoint の voice clone は、24 kHz / mono / finite float32 / `[-1, 1]` の `AudioData` を call ごとの `reference_audio` に渡します。参照条件は engine に登録・保持されず、毎 call で抽出されます。

```python
import asyncio

from speechflow.core import AudioData
from speechflow.qwen3 import Qwen3CloneTTSEngine, Qwen3ModelSource


async def synthesize_reference(reference_audio: AudioData) -> None:
    async with Qwen3CloneTTSEngine(
        Qwen3ModelSource(
            talker_path="/models/qwen-talker-0.6b-base-Q4_K_M.gguf",
            codec_path="/models/qwen-tokenizer-12hz-Q4_K_M.gguf",
        )
    ) as clone:
        cloned = await clone.get(
            "This voice was cloned.",
            reference_audio=reference_audio,
            reference_text="Reference transcript.",
            language="en",
        )
        print(cloned.data.shape)
        async for chunk in clone.stream("Streaming clone.", reference_audio=reference_audio):
            print(chunk.data.shape)


# reference_audio は application が用意した 24 kHz mono float32 の AudioData。
asyncio.run(synthesize_reference(reference_audio))
```

`language` は共通の言語コードで指定します。Qwen3-TTS が扱う言語では、受理する code は `zh`、`en`、`de`、`it`、`pt`、`es`、`ja`、`ko`、`fr`、`ru` です。`language=None` は言語を指定せず、model が text から自動判定します。受理しない値は合成前に `ConfigurationError` になります。言語名、大文字、地域付き、3 文字の表記から対応する受理可能な code が一つに決まる場合、error message にその code を示します。

CustomVoice と Clone の両方で、`get()` と `stream()` に `do_sample`、`temperature`、`top_k`、`top_p`、`repetition_penalty` を keyword argument として渡せます。これらは呼び出しごとの talker sampling に効きます。`do_sample` は bool、`temperature` と `repetition_penalty` は正の有限数、`top_k` は 0 から 2,147,483,647 までの整数、`top_p` は 0 より大きく 1 以下の有限数を受け付けます。`do_sample=False` は greedy、`top_k=0` と `top_p=1` はそれぞれの制限を使わない指定です。用意した engine に渡した不正な値は、native の合成を始める前に `ConfigurationError` になります。

両 engine の構築時には `codec_chunk_sec` を keyword argument として渡せます。正の有限数だけを受け付け、構築後に変更できません。この値は `get()` の buffered decode の chunk 幅に効き、streaming には効きません。sampling の引数と `codec_chunk_sec` を省いた場合、binding は `qt_tts_default_params` と `qt_init_default_params` が設定した native 既定値をそのまま渡します。

CustomVoice と Clone の `get()` と `stream()` は、`max_new_tokens` を省略した場合 (`None` を含む)、合成の長さに `20 + 係数 × len(text)` frame の上限を設けます (1 frame = 80 ms、係数は `language` が `zh`、`ja`、`ko`、`None` または未指定なら 5、それ以外の受理する言語なら 2)。`max_new_tokens` を明示した場合は、文の長さによる上限に代えて指定値を使います (1 から 2,147,483,647 までの整数)。

`get()` / `stream()` は float32、24,000 Hz、mono の `AudioData` を返します。合成の前には `prepare()` が必要です。`async with engine` も model を用意し、終了時に閉じます。用意の前の合成は `EngineStateError`、閉じた後の合成は `EngineClosedError` になります。`prepare()` を繰り返しても読み込みは繰り返しません。用意の時間の上限は利用側が `asyncio.timeout()` などで付けます。取り消した操作は native の仕事が実際に終わるまで待つため、上限を超えて戻る場合があります。

`engine.capabilities` と `engine.health()` は同期で手元の値を返します。CustomVoice の `engine.list_voices()` も同期の読み取りで、用意の後に model の speaker を返します。用意の前と閉じた後の声の一覧は `EngineStateError` になり、一覧の取得で model を読み込みません。Clone は声の一覧を持たず、`capabilities.voice_listing` は false です。両 engine は 10 言語の自動判定と逐次の合成を申告し、Clone だけが参照音声による声の作成を申告します。

engine の仕事をする操作は非同期です。同期の code から一連の非同期処理を呼ぶ場合は、core の `speechflow.core.run_sync()` を使えます。engine の失敗は共通の error 体系に従い、環境や native の部品の不足は `EnvironmentUnavailableError`、合成と資源の解放の失敗は `InferenceError`、中身のある text から音声が得られなかった時は `EmptyOutputError` になります。

同梱 catalog は CustomVoice / Base、0.6B / 1.7B、Q4_K_M / Q8_0 の 8 pair を識別します。`pending` は record 未採用、`qualified` は pass record 採用、`failed` は fail record 採用です。catalog で識別できても pass record がない pair を `qualified` または `supported` とは表示しません。

| Model identifier | Mode | Talker GGUF | Codec GGUF | Qualified state | Evidence |
|---|---|---|---|---|---|
| `qwen3-tts-0.6b-customvoice-Q4_K_M` | CustomVoice | `qwen-talker-0.6b-customvoice-Q4_K_M.gguf` | `qwen-tokenizer-12hz-Q4_K_M.gguf` | qualified | [qwen3-tts-0.6b-customvoice-Q4_K_M--sm_120a.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-0.6b-customvoice-Q4_K_M--sm_120a.json) |
| `qwen3-tts-0.6b-customvoice-Q8_0` | CustomVoice | `qwen-talker-0.6b-customvoice-Q8_0.gguf` | `qwen-tokenizer-12hz-Q8_0.gguf` | qualified | [qwen3-tts-0.6b-customvoice-Q8_0--sm_120a.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-0.6b-customvoice-Q8_0--sm_120a.json) |
| `qwen3-tts-0.6b-base-Q4_K_M` | Base | `qwen-talker-0.6b-base-Q4_K_M.gguf` | `qwen-tokenizer-12hz-Q4_K_M.gguf` | qualified | [qwen3-tts-0.6b-base-Q4_K_M--sm_120a.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-0.6b-base-Q4_K_M--sm_120a.json) |
| `qwen3-tts-0.6b-base-Q8_0` | Base | `qwen-talker-0.6b-base-Q8_0.gguf` | `qwen-tokenizer-12hz-Q8_0.gguf` | qualified | [qwen3-tts-0.6b-base-Q8_0--sm_120a.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-0.6b-base-Q8_0--sm_120a.json) |
| `qwen3-tts-1.7b-customvoice-Q4_K_M` | CustomVoice | `qwen-talker-1.7b-customvoice-Q4_K_M.gguf` | `qwen-tokenizer-12hz-Q4_K_M.gguf` | qualified | [qwen3-tts-1.7b-customvoice-Q4_K_M--sm_120a.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-1.7b-customvoice-Q4_K_M--sm_120a.json) |
| `qwen3-tts-1.7b-customvoice-Q8_0` | CustomVoice | `qwen-talker-1.7b-customvoice-Q8_0.gguf` | `qwen-tokenizer-12hz-Q8_0.gguf` | qualified | [qwen3-tts-1.7b-customvoice-Q8_0--sm_120a.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-1.7b-customvoice-Q8_0--sm_120a.json) |
| `qwen3-tts-1.7b-base-Q4_K_M` | Base | `qwen-talker-1.7b-base-Q4_K_M.gguf` | `qwen-tokenizer-12hz-Q4_K_M.gguf` | qualified | [qwen3-tts-1.7b-base-Q4_K_M--sm_120a.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-1.7b-base-Q4_K_M--sm_120a.json) |
| `qwen3-tts-1.7b-base-Q8_0` | Base | `qwen-talker-1.7b-base-Q8_0.gguf` | `qwen-tokenizer-12hz-Q8_0.gguf` | qualified | [qwen3-tts-1.7b-base-Q8_0--sm_120a.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-1.7b-base-Q8_0--sm_120a.json) |

## 配布とライセンス

distribution `speechflow-qwen3` は、一つの version に `py3-none-manylinux_2_28_x86_64` の wheel を一つ配布します。

公開物のライセンスは [MIT](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/LICENSE) です。同梱する第三者の部品の license の全文は、wheel 内と [licenses/](https://github.com/sync-dev-org/speechflow-qwen3/tree/main/licenses) にあります。
