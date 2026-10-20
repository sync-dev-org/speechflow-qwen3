from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
from scripts.vendor_source import load_manifest, load_sources

ROOT = Path(__file__).parents[1]
VENDOR_ROOT = ROOT / "vendor"
SOURCE_ROOT = VENDOR_ROOT / "qwentts.cpp"


def _snapshot_files() -> list[Path]:
    return sorted(path for path in SOURCE_ROOT.rglob("*") if path.is_file())


@pytest.mark.req("SP-SFQWEN3-026")
def test_vendored_snapshot_is_flat_regular_source(native_source: Path) -> None:
    """The available native snapshot contains regular build sources without nested Git repositories."""
    assert (SOURCE_ROOT / "CMakeLists.txt").is_file()
    assert (SOURCE_ROOT / "ggml/CMakeLists.txt").is_file()
    assert all(path.is_file() and not path.is_symlink() for path in _snapshot_files())
    assert not list(SOURCE_ROOT.rglob(".git"))
    assert not list(SOURCE_ROOT.rglob(".gitmodules"))

    for excluded in (
        ".github",
        "Dockerfile",
        "docker",
        "buildall.sh",
        "buildcpu.sh",
        "buildcuda.sh",
        "buildsycl.sh",
        "buildtermux.sh",
        "buildvulkan.sh",
        "buildall.cmd",
        "buildcuda.cmd",
        "buildvulkan.cmd",
    ):
        assert not (SOURCE_ROOT / excluded).exists()


@pytest.mark.req("SP-SFQWEN3-026")
def test_manifest_matches_every_vendored_file(native_source: Path) -> None:
    """Every available native source file matches the published SHA-256 manifest."""
    manifest_path = VENDOR_ROOT / "MANIFEST.sha256"
    entries: dict[str, str] = {}
    for line in manifest_path.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        entries[relative] = digest

    actual = {
        path.relative_to(VENDOR_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in _snapshot_files()
    }
    assert entries == actual


@pytest.mark.req("SP-SFQWEN3-026")
def test_source_identities_manifest_and_patches_are_consistent_without_native_source() -> None:
    """Fetch inputs name valid pinned sources and patch only files covered by the manifest."""
    assert set(load_sources()) == {"qwentts", "ggml"}
    entries = load_manifest()
    assert {"qwentts.cpp/CMakeLists.txt", "qwentts.cpp/ggml/CMakeLists.txt"} <= entries.keys()
    patches = list((VENDOR_ROOT / "patches").glob("*.patch"))
    assert patches
    for patch in patches:
        targets = re.findall(r"^\+\+\+ b/vendor/(.+)$", patch.read_text(encoding="utf-8"), re.MULTILINE)
        assert targets, f"patch has no source targets: {patch.name}"
        assert set(targets) <= entries.keys(), f"patch targets are missing from the manifest: {patch.name}"


def test_wheel_license_notices_are_present() -> None:
    license_root = ROOT / "src/speechflow/qwen3/_licenses"

    assert {path.name for path in license_root.iterdir() if path.is_file()} == {
        "qwentts.cpp-LICENSE.txt",
        "ggml-LICENSE.txt",
        "cpp-httplib-LICENSE.txt",
        "yyjson-LICENSE.txt",
    }
