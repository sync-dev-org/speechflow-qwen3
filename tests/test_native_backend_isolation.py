from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
PAYLOAD = ROOT / "src/speechflow/qwen3/_native/libqwen.so"


@pytest.mark.req("SP-SF-069")
def test_vendored_ggml_backend_loading_functions_do_nothing(native_source: Path) -> None:
    """取り込んだ ggml の中の `ggml_backend_load_all`、`ggml_backend_load_all_from_path`、`ggml_backend_load`
    の三つの関数を、何もしない形にする
    """
    source = (ROOT / "vendor/qwentts.cpp/ggml/src/ggml-backend-reg.cpp").read_text()
    source = re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.DOTALL)

    for name, return_type, expected_body in (
        ("ggml_backend_load_all", "void", ""),
        ("ggml_backend_load_all_from_path", "void", "(void)dir_path;"),
        ("ggml_backend_load", "ggml_backend_reg_t", "(void)path;returnnullptr;"),
    ):
        definitions = re.findall(rf"\b{return_type}\s+{name}\s*\([^)]*\)\s*\{{([^{{}}]*)\}}", source)
        assert len(definitions) == 1, f"{name} must have one definition without nested operations"
        assert "".join(definitions[0].split()) == expected_body, f"{name} must not explore or load backend files"


@pytest.mark.req("SP-SF-068")
@pytest.mark.native_payload
def test_qwen_initializes_its_cpu_backend_without_loading_foreign_ggml_files(tmp_path: Path) -> None:
    """環境変数 `GGML_BACKEND_PATH` を読まない。

    他の build の ggml の library と backend file がある条件で、自分の backend で動く
    """
    assert PAYLOAD.is_file(), "build the native payload before running the native_payload test"
    compiler = shutil.which("cc")
    assert compiler is not None, "a C compiler is required for the foreign backend probes"

    native_dir = tmp_path / "native"
    native_dir.mkdir()
    shutil.copy2(PAYLOAD, native_dir / "libqwen.so")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    marker = tmp_path / "loaded-backends.txt"

    probe_source = tmp_path / "foreign-backend.c"
    probe_source.write_text(
        """
#include <stdio.h>
#include <stdlib.h>

__attribute__((constructor)) static void record_load(void) {
    const char * path = getenv("QWEN_FOREIGN_BACKEND_MARKER");
    if (path) {
        FILE * file = fopen(path, "a");
        if (file) {
            fputs(PROBE_NAME "\\n", file);
            fclose(file);
        }
    }
}

int ggml_backend_score(void) { return 1; }
"""
    )
    for name, destination in (
        ("foreign-library", native_dir / "libggml.so"),
        ("native-directory", native_dir / "libggml-cpu-foreign.so"),
        ("working-directory", cwd / "libggml-cpu-foreign.so"),
        ("backend-path", tmp_path / "libforeign-backend.so"),
    ):
        subprocess.run(
            [compiler, "-shared", "-fPIC", f'-DPROBE_NAME="{name}"', str(probe_source), "-o", str(destination)],
            check=True,
        )

    script = """
import sys
from pathlib import Path
from speechflow.qwen3 import _payload
from speechflow.qwen3._binding import CtypesNativeBinding, NativeCallError

_payload._native_directory = lambda: Path(sys.argv[1])
try:
    CtypesNativeBinding().load_model(str(Path(sys.argv[2]) / "missing-talker.gguf"),
                                    str(Path(sys.argv[2]) / "missing-codec.gguf"))
except NativeCallError as error:
    if "pipeline_tts_load failed" not in str(error):
        raise
else:
    raise AssertionError("missing model files unexpectedly loaded")
"""
    env = os.environ.copy()
    env.update(
        {
            "GGML_BACKEND": "CPU",
            "GGML_BACKEND_PATH": str(tmp_path / "libforeign-backend.so"),
            "QWEN_FOREIGN_BACKEND_MARKER": str(marker),
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(native_dir), str(tmp_path)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "[Load] Talker backend: CPU" in result.stderr
    assert not marker.exists(), marker.read_text() if marker.exists() else ""
