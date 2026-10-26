"""Offline tests for reconstructing the pinned native source snapshot."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def _upstream(repository: Path, files: dict[str, str]) -> tuple[str, str]:
    repository.mkdir()
    _git(repository, "init", "--quiet")
    for name, content in files.items():
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    if repository.name == "ggml":
        (repository / "link.cpp").symlink_to("src/ggml.cpp")
    _git(repository, "add", "--all")
    _git(repository, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "source")
    return _git(repository, "rev-parse", "HEAD"), _git(repository, "rev-parse", "HEAD^{tree}")


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    vendor = root / "vendor"
    (vendor / "patches").mkdir(parents=True)
    excluded = {
        ".gitmodules": "submodule configuration\n",
        ".github/workflows/check.yml": "workflow\n",
        "Dockerfile": "container\n",
        "docker/build": "container helper\n",
        "buildcpu.sh": "build helper\n",
        "build.cmd": "build helper\n",
    }
    sources = {
        "qwentts": {"CMakeLists.txt": "project(qwen)\n", "src/qwen.cpp": "original\n", **excluded},
        "ggml": {"CMakeLists.txt": "project(ggml)\n", "src/ggml.cpp": "ggml\n", **excluded},
    }
    sources["qwentts"].update(
        {
            "examples/base.cmd": "example helper\n",
            "examples/clone.cmd": "example helper\n",
            "examples/customvoice.cmd": "example helper\n",
            "examples/tts.cmd": "example helper\n",
            "examples/keep.cmd": "retained file\n",
        }
    )
    blocks = []
    for component, files in sources.items():
        repository = tmp_path / component
        revision, tree = _upstream(repository, files)
        blocks.append(f'[{component}]\nurl = "{repository.as_uri()}"\nrevision = "{revision}"\ntree = "{tree}"\n')
    (vendor / "PROVENANCE.md").write_text("# Sources\n\n```toml\n" + "\n".join(blocks) + "```\n", encoding="utf-8")
    (vendor / "patches/0001-source.patch").write_text(
        "diff --git a/vendor/qwentts.cpp/src/qwen.cpp b/vendor/qwentts.cpp/src/qwen.cpp\n"
        "--- a/vendor/qwentts.cpp/src/qwen.cpp\n"
        "+++ b/vendor/qwentts.cpp/src/qwen.cpp\n"
        "@@ -1 +1 @@\n-original\n+patched\n",
        encoding="utf-8",
    )
    expected = {
        "CMakeLists.txt": "project(qwen)\n",
        "src/qwen.cpp": "patched\n",
        "examples/keep.cmd": "retained file\n",
        "ggml/CMakeLists.txt": "project(ggml)\n",
        "ggml/src/ggml.cpp": "ggml\n",
        "ggml/link.cpp": "ggml\n",
    }
    (vendor / "MANIFEST.sha256").write_text(
        "".join(
            f"{hashlib.sha256(content.encode()).hexdigest()}  qwentts.cpp/{name}\n"
            for name, content in sorted(expected.items())
        ),
        encoding="utf-8",
    )
    return root


def _fetch(checkout: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/fetch_vendor.py"), "--repo-root", str(checkout)],
        capture_output=True,
        text=True,
    )


@pytest.mark.req("SP-SFQWEN3-026")
def test_fetch_reconstructs_patched_flat_source_from_local_repositories(checkout: Path) -> None:
    """Pinned local upstreams are pruned, patched, verified, and installed without a network service."""
    result = _fetch(checkout)
    assert result.returncode == 0, result.stderr
    vendor = checkout / "vendor"
    actual = {
        path.relative_to(vendor).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (vendor / "qwentts.cpp").rglob("*")
        if path.is_file()
    }
    expected = {
        name: digest
        for digest, name in (line.split("  ", 1) for line in (vendor / "MANIFEST.sha256").read_text().splitlines())
    }
    assert actual == expected
    assert (vendor / "qwentts.cpp/src/qwen.cpp").read_text() == "patched\n"
    assert not list((vendor / "qwentts.cpp").rglob(".git"))
    assert not any(path.is_symlink() for path in (vendor / "qwentts.cpp").rglob("*"))


@pytest.mark.req("SP-SFQWEN3-026")
@pytest.mark.parametrize("failure", ["digest", "missing", "unexpected", "revision", "unavailable", "patch"])
def test_failed_fetch_explains_recovery_and_never_installs_unverified_source(checkout: Path, failure: str) -> None:
    """Unavailable upstreams, revisions, broken patches, and all manifest differences leave no source installed."""
    vendor = checkout / "vendor"
    manifest = vendor / "MANIFEST.sha256"
    provenance = vendor / "PROVENANCE.md"
    if failure == "digest":
        manifest.write_text("0" * 64 + manifest.read_text()[64:])
        expected = "qwentts.cpp/CMakeLists.txt"
    elif failure == "missing":
        manifest.write_text(manifest.read_text() + "0" * 64 + "  qwentts.cpp/missing.cpp\n")
        expected = "qwentts.cpp/missing.cpp"
    elif failure == "unexpected":
        manifest.write_text("\n".join(manifest.read_text().splitlines()[1:]) + "\n")
        expected = "qwentts.cpp/CMakeLists.txt"
    elif failure == "revision":
        revision = _git(checkout.parent / "ggml", "rev-parse", "HEAD")
        provenance.write_text(provenance.read_text().replace(revision, "f" * 40))
        expected = "revision"
    elif failure == "unavailable":
        url = (checkout.parent / "qwentts").as_uri()
        provenance.write_text(provenance.read_text().replace(url, (checkout.parent / "unavailable").as_uri()))
        expected = "fetch"
    else:
        patch = vendor / "patches/0001-source.patch"
        patch.write_text(patch.read_text().replace("-original", "-missing"))
        expected = "patch"
    result = _fetch(checkout)
    assert result.returncode != 0
    assert expected in result.stderr, result.stderr
    assert "Next:" in result.stderr, result.stderr
    assert not (vendor / "qwentts.cpp").exists()


@pytest.mark.req("SP-SFQWEN3-026")
def test_fetch_preserves_an_existing_snapshot(checkout: Path) -> None:
    """Fetching never replaces an existing source directory."""
    destination = checkout / "vendor/qwentts.cpp"
    destination.mkdir()
    (destination / "keep").write_text("local source")
    result = _fetch(checkout)
    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert "Next:" in result.stderr
    assert (destination / "keep").read_text() == "local source"


@pytest.mark.req("SP-SFQWEN3-017")
@pytest.mark.req("SP-SFQWEN3-026")
def test_provenance_revision_changes_reach_metadata_wheel_checks_and_qualification(checkout: Path) -> None:
    """One revision edit updates build metadata, wheel expectations, and the qualification revision."""
    shutil.copytree(ROOT / "scripts", checkout / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    result = subprocess.run(
        [
            sys.executable,
            str(checkout / "scripts/vendor_source.py"),
            "metadata",
            "13.2.51",
            "86-real;89-real;120a-real",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    assert metadata["qwentts_revision"] == _git(checkout.parent / "qwentts", "rev-parse", "HEAD")
    assert metadata["ggml_revision"] == _git(checkout.parent / "ggml", "rev-parse", "HEAD")
    probe = """
import io
import json
import sys
import zipfile
from scripts.check_wheel import WheelCheckError, _check_build_metadata
from scripts.qualified_gpu import QUALIFIED_QWENTTS_REVISION

metadata = json.loads(sys.argv[1])
assert QUALIFIED_QWENTTS_REVISION == metadata['qwentts_revision']
for valid in (True, False):
    if not valid:
        metadata['ggml_revision'] = '0' * 40
    with zipfile.ZipFile(io.BytesIO(), 'w') as archive:
        archive.writestr('speechflow/qwen3/_native/BUILD-METADATA.json', json.dumps(metadata))
        try:
            _check_build_metadata(archive)
        except WheelCheckError:
            assert not valid
        else:
            assert valid
"""
    subprocess.run([sys.executable, "-c", probe, json.dumps(metadata)], cwd=checkout, check=True)
    for query, revision in (
        (["rev-parse", "--short", "HEAD"], metadata["qwentts_revision"][:7]),
        (["describe", "--match=NeVeRmAtCh", "--always", "--abbrev=8"], metadata["ggml_revision"][:8]),
    ):
        identity = subprocess.run(
            [str(checkout / "scripts/native-tools/git"), *query], check=True, capture_output=True, text=True
        )
        assert identity.stdout.strip() == revision
