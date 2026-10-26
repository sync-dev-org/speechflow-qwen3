# speechflow-qwen3

Qwen3-TTS の qwentts.cpp runtime を Python process 内で実行する SpeechFlow engine module です。この checkout の native build は Linux x86_64、glibc 2.38 以上、CUDA 13.2 を対象とします。

## 対応環境

CPU ISA の下限は x86-64-v3 (AVX2 / FMA / F16C) です。producer 固有の CPU 命令へ最適化せず、AVX-512 は有効にしません。

NVIDIA driver は host が提供し、CUDA 13.2 以上に対応する必要があります。CUDA 13.2 以上の user-space library (`libcudart.so.13`、`libcublas.so.13`、`libcublasLt.so.13`) は Python dependency が提供します。CUDA 13.2 GA に対応する package の下限は `nvidia-cuda-runtime` が 13.2、`nvidia-cublas` が 13.3 で、どちらも 14 未満です。runtime host に system CUDA toolkit、compiler、CMake、Ninja、外部 build directory は必要ありません。

標準 build は `CMAKE_CUDA_ARCHITECTURES=86-real;89-real;120a-real` を使い、compute capability 8.6、compute capability 8.9、compute capability 12.0 の real cubin を同じ wheel に含めます。

| Architecture | Compute capability | Representative GPU | Native target | Runtime evidence | Record |
|---|---:|---|---|---|---|
| `sm_86` | 8.6 | NVIDIA RTX A6000 | included | qualified | [qwen3-tts-0.6b-customvoice-Q4_K_M--sm_86.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-0.6b-customvoice-Q4_K_M--sm_86.json) |
| `sm_89` | 8.9 | NVIDIA RTX 6000 Ada | included | qualified | [qwen3-tts-0.6b-customvoice-Q4_K_M--sm_89.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-0.6b-customvoice-Q4_K_M--sm_89.json) |
| `sm_120a` | 12.0 | NVIDIA RTX PRO 4000 Blackwell | included | qualified | [qwen3-tts-0.6b-base-Q4_K_M--sm_120a.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-0.6b-base-Q4_K_M--sm_120a.json), [qwen3-tts-0.6b-base-Q8_0--sm_120a.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-0.6b-base-Q8_0--sm_120a.json), [qwen3-tts-0.6b-customvoice-Q4_K_M--sm_120a.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-0.6b-customvoice-Q4_K_M--sm_120a.json), [qwen3-tts-0.6b-customvoice-Q8_0--sm_120a.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-0.6b-customvoice-Q8_0--sm_120a.json), [qwen3-tts-1.7b-base-Q4_K_M--sm_120a.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-1.7b-base-Q4_K_M--sm_120a.json), [qwen3-tts-1.7b-base-Q8_0--sm_120a.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-1.7b-base-Q8_0--sm_120a.json), [qwen3-tts-1.7b-customvoice-Q4_K_M--sm_120a.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-1.7b-customvoice-Q4_K_M--sm_120a.json), [qwen3-tts-1.7b-customvoice-Q8_0--sm_120a.json](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-records/qwen3-tts-1.7b-customvoice-Q8_0--sm_120a.json) |

wheel は PTX を同梱しません。したがって未列挙の GPU 世代は JIT target を含めて対象外です。`included` は native target が archive contract に合格したことだけを表し、実機での成功は [qualified evidence](https://github.com/sync-dev-org/speechflow-qwen3/blob/main/docs/qualified-evidence.md) に pass record が採用されるまで `pending` です。

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

## 開発 checkout

この repository は standalone な `speechflow-qwen3` project です。sibling checkout や外側の workspace を必要とせず、repository root で開発します。

```bash
git clone <repository-url> speechflow-qwen3
cd speechflow-qwen3
uv sync
uv lock --check
uv run pytest
uv run ruff check
uv run ruff format --check
uv run mypy
python3 scripts/fetch_vendor.py
uv build --wheel
```

通常 dependency は `speechflow>=0.19.2` です。開発時も PyPI に公開した core を使い、tracked な `uv.lock` が core の version と file の SHA-256 を固定します。現在の lock は PyPI の core 0.19.2 を使います。editable sync と import は native compilation を行いません。

公開用の repository は上流の source の複製を含みません。既定の test は network を使わず、source を要する検査だけを取得 command の案内とともに skip します。native build と source を使う test の前に、Python 3.12 以上、Git、上流への network 接続を用意し、`python3 scripts/fetch_vendor.py` を実行します。取得済みの `vendor/qwentts.cpp/` がある場合、この取得 command は不要です。

PyPI に公開された新しい core を取り込む場合は、repository root で core package だけを更新します。

```bash
uv lock --upgrade-package speechflow
```

更新後は `uv.lock` の core の source が PyPI (`https://pypi.org/simple`) で、version と file の SHA-256 が意図した物になり、他 dependency が意図せず変わっていないことを確認してから、上の検証コマンドを再実行します。

未公開の core の変更は、commit しない local の override で試します。override と、それによって変わった `uv.lock` は commit しません。tracked な設定には、core に対する Git の URL、`[tool.uv.sources]`、workspace source、sibling path を残しません。

## 開発用 native build

CUDA 13.2 toolkit、CMake 3.31 以上、Ninja、GCC/G++ を備えた Linux x86_64 host で実行します。開発者は、toolkit の場所を `CUDA_HOME` で指定できます。未設定の場合、script は `/usr/local/cuda` を使います。CMake と Ninja の入れ方は問いません。`vendor/qwentts.cpp/` が無い場合は、先に `python3 scripts/fetch_vendor.py` で source を取得してください。

```bash
export CUDA_HOME=/absolute/path/to/cuda-13.2
scripts/build_native.sh
```

この script は `vendor/qwentts.cpp` を Release、CUDA、shared qwen target、x86-64-v3 baseline で build し、vendored ggml の CPU と CUDA backend を `libqwen.so` に静的に link します。`libqwen.so` の公開 symbol は `qt_*` に限定します。upstream 既定を `CMAKE_CUDA_ARCHITECTURES=86-real;89-real;120a-real` で override し、compile command が三つの real code generation だけを重複なく持つことを検査します。canonical architecture value は `BUILD-METADATA.json` に記録します。`test-abi-c` を実行した後、`libqwen.so` 一つを `src/speechflow/qwen3/_native/` に配置します。editable checkout で payload が無い場合、import と構築は成功し、`prepare()` が build の手順を示す `EnvironmentUnavailableError` を返します。

build 済み payload について、foreign な ggml library と backend file を置いても CPU backend で初期化することは、GPU と model weights を使わずに次の opt-in test で確認します。

```bash
CUDA_VISIBLE_DEVICES= uv run pytest -m native_payload tests/test_native_backend_isolation.py
```

## Wheel build と検査

標準 wheel build は repository root から実行し、custom Hatch build hook が同じ native build を実行します。payload が欠けている場合は pure wheel を出力せず失敗します。

以下の `<X.Y.Z>` は build する version に置き換えます。

```bash
export CUDA_HOME=/absolute/path/to/cuda-13.2
uv build --wheel
uv run python scripts/check_wheel.py "dist/speechflow_qwen3-<X.Y.Z>-py3-none-manylinux_2_38_x86_64.whl"
```

生成 tag は `py3-none-manylinux_2_38_x86_64` です。検査 script は wheel filename / `.dist-info` / metadata の identity、namespace ownership、payload、license notice、RECORD/CRC、native dependency allowlist、同梱 ELF の `$ORIGIN` 以外の RPATH / RUNPATH がないことを検証します。`libqwen.so` の公開 symbol が `qt_*` だけで、`DT_NEEDED` に `libggml*.so*` がないことも検証します。wheel から `libqwen.so` を取り出し、`cuobjdump` で cubin 集合が `sm_86` / `sm_89` / `sm_120a` と完全一致し、PTX がないことを検証します。`readelf` で要求する glibc symbol の最大が配布対象の下限 2.38 以下であることも検証します。CUDA user-space libraries は Linux x86_64 marker 付き dependency `nvidia-cuda-runtime>=13.2,<14` と `nvidia-cublas>=13.3,<14` が供給します。NVIDIA driver の `libcuda.so.1` は host の前提です。

## Qualified test

qualified runner は `full-pair` と `architecture-smoke` の二 profile を持ちます。どちらも Release module wheel と compatible core wheel を tracked tree 外の isolated venv へ install し、module wheel の SHA-256 を Release tooling が作った `SHA256SUMS` と照合します。editable import、checkout source からの import、wheel digest 不一致は拒否します。GPU 実行は `SPEECHFLOW_QWEN3_RUN_GPU=1` を明示した時だけ有効です。

この checkout は `prepare()` など新しい公開の操作を持つ core 0.19.2 以上を必要とします。開発に使う core の version と file の hash は `uv.lock` が固定します。新しい qualified test は、PyPI に公開された core 0.19.2 の wheel と module wheel で行います。core wheel は [PyPI の speechflow 0.19.2](https://pypi.org/project/speechflow/0.19.2/) の `speechflow-0.19.2-py3-none-any.whl` を取得し、install 前と qualified test の実行前に SHA-256 が `2224905a292ed399d45466f504e681f22772f11aadeda060147b38eebbe20ca0` と一致することを照合します。

以下の `<X.Y.Z>` は検査する Release の version に置き換え、module wheel の filename と `--release-tag` に同じ version を指定します。

```bash
QUALIFIED_ROOT=/absolute/path/to/qualified-runtime
uv venv "$QUALIFIED_ROOT/venv" --python 3.12
uv pip install --python "$QUALIFIED_ROOT/venv/bin/python" \
  "/absolute/path/to/speechflow-0.19.2-py3-none-any.whl" \
  "/absolute/path/to/speechflow_qwen3-<X.Y.Z>-py3-none-manylinux_2_38_x86_64.whl"

CUDA_VISIBLE_DEVICES=0 SPEECHFLOW_QWEN3_RUN_GPU=1 "$QUALIFIED_ROOT/venv/bin/python" \
  scripts/qualified_gpu.py \
  --profile architecture-smoke \
  --cuda-architecture sm_86 \
  --physical-gpu-index 0 \
  --model-id qwen3-tts-0.6b-customvoice-Q4_K_M \
  --weights-dir /absolute/path/to/qwen3-weights \
  --output "$QUALIFIED_ROOT/qwen3-tts-0.6b-customvoice-Q4_K_M--sm_86.json" \
  --core-wheel "/absolute/path/to/speechflow-0.19.2-py3-none-any.whl" \
  --module-wheel "/absolute/path/to/speechflow_qwen3-<X.Y.Z>-py3-none-manylinux_2_38_x86_64.whl" \
  --sha256sums /absolute/path/to/SHA256SUMS \
  --release-tag "v<X.Y.Z>" \
  --speaker ryan \
  --text 'SpeechFlow multi-architecture qualified synthesis.' \
  --language en \
  --seed 7 \
  --max-new-tokens 64 \
  --trial-count 3
```

`architecture-smoke` は上記の model、speaker、text、language、seed、token 上限、trial count を固定し、physical GPU index を `CUDA_VISIBLE_DEVICES` と `nvidia-smi -i` の両方へ適用します。CUDA process 内の選択 device は logical index 0 です。三 trial は別々の stream iterator で実行し、schema-v3 record 一件を出力します。

`full-pair` は CustomVoice / Base × 0.6B / 1.7B × Q4_K_M / Q8_0 の一 pair を一回ずつ実行します。isolated import、host/build boundary、manifest、native diagnostic、prepare と idempotence、get、stream の first chunk と正常終端、24 kHz mono float32 音声、voice contract、cancellation、反復 close、configuration diagnostic、VRAM sampling を検査します。Base は `--reference-audio-npy` と `--reference-source` を必要とします。

runner は pass record で exit 0、書き出せた gate failure で exit 1、record 作成前 failure で exit 2 を返し、自動 retry は行いません。Release tag の tracked tree は record を持たず全状態を `pending` とします。Release の read-back verify 後に record を採用し、README と `docs/qualified-evidence.md` を同じ `qualified` または `failed` 状態へ更新します。既定 pytest は schema、failure、文書整合を fixture で検査し、実 GPU synthesis を開始しません。

## Vendoring provenance と upstream 更新

取得元、qwentts.cpp / ggml の full revision と Git tree identity、除外 path、license、project patch series は `vendor/PROVENANCE.md` に記録します。同文書の TOML block が revision の唯一の定義元です。`vendor/MANIFEST.sha256` は vendored regular file 全件の SHA-256 を保持します。

`python3 scripts/fetch_vendor.py` は固定した revision を一時 directory に取得し、除外 path を取り除き、`vendor/patches/*.patch` を名前順に当て、manifest と全 file の SHA-256 を照合します。一致した場合だけ `vendor/qwentts.cpp/` に配置します。取得や照合に失敗した場合は理由と次の操作を表示し、未照合の source は配置しません。既存の source を上書きしないので、取得をやり直す時は既存の directory を別の場所へ移してください。

upstream 更新は次を一つの変更単位として扱います。

1. qwentts.cpp と recursive ggml dependency を新しい revision に固定する。
2. `.git` / `.gitmodules` を残さず regular file として `vendor/qwentts.cpp/` へ平坦化し、provenance に列挙した不要 path を除外する。
3. `vendor/MANIFEST.sha256`、license notice、`vendor/PROVENANCE.md` を同期する。
4. project patch series がある場合は再適用し、snapshot 差分を照合する。
5. pytest、`scripts/build_native.sh`、標準 wheel build、`scripts/check_wheel.py` を通す。

vendored source は原則 upstream と一致させます。module 固有の修正が必要な場合は snapshot を直接編集せず、理由と upstream base を記録した tracked patch series として管理します。

## Release

distribution `speechflow-qwen3` の platform wheel は PyPI に公開します。一つの version に `py3-none-manylinux_2_38_x86_64` の wheel を一つ置き、sdist は置きません。利用者は `pip install speechflow-qwen3` で wheel を取得できます。
