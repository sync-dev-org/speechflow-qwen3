# Qualified GPU evidence

この文書は `docs/qualified-records/*.json` の schema-v3 record と一致する公開状態です。
native target の同梱と実機 runtime evidence を区別し、性能値に合否閾値を設けません。

## Architecture evidence

| Architecture | Compute capability | Representative GPU | Native target | Runtime evidence | Record |
|---|---:|---|---|---|---|
| `sm_86` | 8.6 | NVIDIA RTX A6000 | included | pending | — |
| `sm_89` | 8.9 | NVIDIA RTX 6000 Ada | included | pending | — |
| `sm_120a` | 12.0 | NVIDIA RTX PRO 4000 Blackwell | included | qualified | [qwen3-tts-0.6b-base-Q4_K_M--sm_120a.json](qualified-records/qwen3-tts-0.6b-base-Q4_K_M--sm_120a.json), [qwen3-tts-0.6b-base-Q8_0--sm_120a.json](qualified-records/qwen3-tts-0.6b-base-Q8_0--sm_120a.json), [qwen3-tts-0.6b-customvoice-Q4_K_M--sm_120a.json](qualified-records/qwen3-tts-0.6b-customvoice-Q4_K_M--sm_120a.json), [qwen3-tts-0.6b-customvoice-Q8_0--sm_120a.json](qualified-records/qwen3-tts-0.6b-customvoice-Q8_0--sm_120a.json), [qwen3-tts-1.7b-base-Q4_K_M--sm_120a.json](qualified-records/qwen3-tts-1.7b-base-Q4_K_M--sm_120a.json), [qwen3-tts-1.7b-base-Q8_0--sm_120a.json](qualified-records/qwen3-tts-1.7b-base-Q8_0--sm_120a.json), [qwen3-tts-1.7b-customvoice-Q4_K_M--sm_120a.json](qualified-records/qwen3-tts-1.7b-customvoice-Q4_K_M--sm_120a.json), [qwen3-tts-1.7b-customvoice-Q8_0--sm_120a.json](qualified-records/qwen3-tts-1.7b-customvoice-Q8_0--sm_120a.json) |

## Model-pair results

| Model identifier | Gate | Executed at | Architecture | Record | Warm get (s) | Streaming TTFA (s) | Peak RSS (bytes) | Peak VRAM delta (MiB) |
|---|---|---|---|---|---:|---:|---:|---:|
| `qwen3-tts-0.6b-customvoice-Q4_K_M` | qualified | 2026-10-16T22:54:24+09:00 | sm_120a | [qwen3-tts-0.6b-customvoice-Q4_K_M--sm_120a.json](qualified-records/qwen3-tts-0.6b-customvoice-Q4_K_M--sm_120a.json) | 0.404 | 0.048 | 1765343232 | 2469.000 |
| `qwen3-tts-0.6b-customvoice-Q8_0` | qualified | 2026-10-16T22:54:29+09:00 | sm_120a | [qwen3-tts-0.6b-customvoice-Q8_0--sm_120a.json](qualified-records/qwen3-tts-0.6b-customvoice-Q8_0--sm_120a.json) | — | — | — | — |
| `qwen3-tts-0.6b-base-Q4_K_M` | qualified | 2026-10-16T22:54:45+09:00 | sm_120a | [qwen3-tts-0.6b-base-Q4_K_M--sm_120a.json](qualified-records/qwen3-tts-0.6b-base-Q4_K_M--sm_120a.json) | 0.680 | 0.052 | 1937727488 | 3033.000 |
| `qwen3-tts-0.6b-base-Q8_0` | qualified | 2026-10-16T22:54:51+09:00 | sm_120a | [qwen3-tts-0.6b-base-Q8_0--sm_120a.json](qualified-records/qwen3-tts-0.6b-base-Q8_0--sm_120a.json) | — | — | — | — |
| `qwen3-tts-1.7b-customvoice-Q4_K_M` | qualified | 2026-10-16T22:54:34+09:00 | sm_120a | [qwen3-tts-1.7b-customvoice-Q4_K_M--sm_120a.json](qualified-records/qwen3-tts-1.7b-customvoice-Q4_K_M--sm_120a.json) | — | — | — | — |
| `qwen3-tts-1.7b-customvoice-Q8_0` | qualified | 2026-10-16T22:54:39+09:00 | sm_120a | [qwen3-tts-1.7b-customvoice-Q8_0--sm_120a.json](qualified-records/qwen3-tts-1.7b-customvoice-Q8_0--sm_120a.json) | — | — | — | — |
| `qwen3-tts-1.7b-base-Q4_K_M` | qualified | 2026-10-16T22:54:57+09:00 | sm_120a | [qwen3-tts-1.7b-base-Q4_K_M--sm_120a.json](qualified-records/qwen3-tts-1.7b-base-Q4_K_M--sm_120a.json) | — | — | — | — |
| `qwen3-tts-1.7b-base-Q8_0` | qualified | 2026-10-16T22:55:04+09:00 | sm_120a | [qwen3-tts-1.7b-base-Q8_0--sm_120a.json](qualified-records/qwen3-tts-1.7b-base-Q8_0--sm_120a.json) | — | — | — | — |

`pending` は record 未採用、`qualified` は pass record 採用、`failed` は fail record 採用を示します。
fail record を `qualified` または `supported` と表示しません。
