# Vendored source provenance

## 固定した取得元と revision

この TOML block が source の URL、full revision、Git tree の値を保持する唯一の場所です。取得、native build の metadata、wheel の検査、qualified runner は `scripts/vendor_source.py` を通してこの block を読みます。

```toml
[qwentts]
url = "https://github.com/ServeurpersoCom/qwentts.cpp"
revision = "a8a7716b530e49fed537c57711247c12fbbb903c"
tree = "cc4503bb883ad2a2e820491934c7fae1390779e3"

[ggml]
url = "https://github.com/ServeurpersoCom/ggml.git"
revision = "c044c6f03892f9d5e98213b05f8afea1f8b0d3c9"
tree = "1ff2b1e30378c4a0503d7bb4d5922040078d39c8"
```

## qwentts.cpp

- 取込日: 2026-09-06
- 配置先: `qwentts.cpp/`
- source tree integrity: `MANIFEST.sha256`
- project patch series:
  - `patches/0001-ggml-disable-runtime-backend-loading.patch`: 上の block が指定する ggml の backend file 読み込み三関数を無効化する。`GGML_BACKEND_PATH` を読まず、同梱 backend だけを使うため、repository root で `git apply --unidiff-zero vendor/patches/0001-ggml-disable-runtime-backend-loading.patch` を実行して再適用する

## Recursive dependency

- component: ggml
- flattened path: `qwentts.cpp/ggml/`

## Source の取得と照合

Python 3.12 以上、Git、上流への network 接続を用意し、repository root で `python3 scripts/fetch_vendor.py` を実行します。script は、上の revision の qwentts.cpp と ggml を一時 directory に取得し、除外 path の削除、名前順の `patches/*.patch` の適用、`MANIFEST.sha256` と全 file の照合を行います。照合が一致した場合だけ `vendor/qwentts.cpp/` に配置します。失敗した取得結果は配置しません。

snapshot 内の通常の file を指す symlink は、その参照先の内容を持つ通常の file に変換します。snapshot 外の参照先と directory を指す symlink は受け付けません。

既存の `vendor/qwentts.cpp/` は上書きしません。取得をやり直す場合は、既存の source を別の場所へ移してから実行します。別の checkout を対象にする場合は `python3 scripts/fetch_vendor.py --repo-root /path/to/checkout` を使います。

## 複製から除外した path

- `.git` と recursive dependency 内を含む nested `.git`
- `.gitmodules` と recursive dependency 内の `.gitmodules`
- `.github/`
- `Dockerfile`
- `docker/`
- repository root の `build*.sh`
- repository root の `*.cmd`
- qwentts.cpp の `examples/base.cmd`、`examples/clone.cmd`、`examples/customvoice.cmd`、`examples/tts.cmd`

`.github/`、`Dockerfile`、`docker/`、`build*.sh`、`*.cmd` は qwentts.cpp と ggml それぞれの root から除外します。

## License 一覧

- qwentts.cpp: MIT、`qwentts.cpp/LICENSE`
- ggml: MIT、`qwentts.cpp/ggml/LICENSE`
- cpp-httplib: MIT、`qwentts.cpp/vendor/cpp-httplib/LICENSE`
- yyjson: MIT、`qwentts.cpp/vendor/yyjson/yyjson.h` と `yyjson.c` 冒頭の notice

wheel には上記 notice を `speechflow/qwen3/_licenses/` 以下へ複製して同梱します。module 自身の license は MIT で、repository root の `LICENSE` に記録します。
