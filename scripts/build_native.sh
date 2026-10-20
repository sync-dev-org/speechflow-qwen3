#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source_root="$repo_root/vendor/qwentts.cpp"
build_dir="$repo_root/build/native"
payload_dir="$repo_root/src/speechflow/qwen3/_native"
identity_tools="$repo_root/scripts/native-tools"

if [[ ! -f "$source_root/CMakeLists.txt" || ! -f "$source_root/ggml/CMakeLists.txt" ]]; then
  printf 'vendor/qwentts.cpp source is missing or incomplete. Run python3 scripts/fetch_vendor.py from the repository root; move an incomplete snapshot aside first. See vendor/PROVENANCE.md\n' >&2
  exit 1
fi
if [[ ! -x /usr/local/cuda/bin/nvcc ]]; then
  printf 'CUDA 13.3 compiler is missing: /usr/local/cuda/bin/nvcc\n' >&2
  exit 1
fi

# GGML does not use math errno. Let GCC (including nvcc's host compiler) emit
# hardware square roots instead of linking sqrtf to the host's GLIBC_2.43 libm.
PATH="$identity_tools:$PATH" cmake -S "$source_root" -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  '-DCMAKE_INSTALL_RPATH=$ORIGIN' \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_SHARED_LIBS=OFF \
  '-DCMAKE_C_FLAGS=-fvisibility=hidden -fno-math-errno' \
  '-DCMAKE_CXX_FLAGS=-fvisibility=hidden -fno-math-errno' \
  -DCMAKE_CUDA_FLAGS=-Xcompiler=-fno-math-errno \
  "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,--version-script=$repo_root/scripts/qwen.exports.map -Wl,--exclude-libs,ALL" \
  -DQWEN_SHARED=ON \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_SSE42=ON \
  -DGGML_AVX=ON \
  -DGGML_AVX2=ON \
  -DGGML_BMI2=ON \
  -DGGML_FMA=ON \
  -DGGML_F16C=ON \
  -DGGML_AVX_VNNI=OFF \
  -DGGML_AVX512=OFF \
  -DGGML_AVX512_VBMI=OFF \
  -DGGML_AVX512_VNNI=OFF \
  -DGGML_AVX512_BF16=OFF \
  '-DCMAKE_CUDA_ARCHITECTURES=86-real;89-real;120a-real' \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

PATH="$identity_tools:$PATH" cmake --build "$build_dir" \
  --target qwen test-abi-c \
  --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"

LD_LIBRARY_PATH="$build_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$build_dir/test-abi-c"

mkdir -p "$payload_dir"
find "$payload_dir" -maxdepth 1 -type f -name '*.so*' -delete
install -m 0755 "$build_dir/libqwen.so" "$payload_dir/libqwen.so"

expected_architectures='86-real;89-real;120a-real'
python3 - "$build_dir/compile_commands.json" <<'PY'
import json
import re
import sys
from pathlib import Path

expected = {
    "--generate-code=arch=compute_86,code=[sm_86]",
    "--generate-code=arch=compute_89,code=[sm_89]",
    "--generate-code=arch=compute_120a,code=[sm_120a]",
}
commands = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
cuda_commands = [entry["command"] for entry in commands if "--generate-code=" in entry["command"]]
if not cuda_commands:
    raise SystemExit("CUDA compile command inspection found no code generation flags")
for command in cuda_commands:
    flags = re.findall(r"--generate-code=arch=compute_[^ \\\"']+", command)
    if len(flags) != len(expected) or set(flags) != expected:
        raise SystemExit(
            "CUDA architecture mismatch: "
            f"expected={sorted(expected)}, actual={flags}"
        )
PY
architectures=$expected_architectures
cuda_toolkit_version=$(/usr/local/cuda/bin/nvcc --version | sed -n 's/.* V\([0-9][0-9.]*\)$/\1/p' | tail -n 1)
if [[ "$cuda_toolkit_version" != 13.3.* ]]; then
  printf 'unexpected CUDA toolkit version: %s\n' "$cuda_toolkit_version" >&2
  exit 1
fi
python3 "$repo_root/scripts/vendor_source.py" metadata \
  "$cuda_toolkit_version" "$architectures" > "$payload_dir/BUILD-METADATA.json"

printf 'native payload written to %s\n' "$payload_dir"
