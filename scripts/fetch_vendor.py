#!/usr/bin/env python3
"""Fetch pinned upstreams, apply project patches, and install only a manifest-verified snapshot."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.vendor_source import ROOT, Source, load_manifest, load_sources


class FetchError(ValueError):
    """A snapshot cannot be reconstructed and verified."""


def _git(root: Path, *arguments: str) -> str:
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    except FileNotFoundError as error:
        raise FetchError("Git is missing. Next: install Git and retry python3 scripts/fetch_vendor.py.") from error
    except subprocess.CalledProcessError as error:
        raise FetchError(f"git {' '.join(arguments)} failed: {error.stderr.strip()}") from error
    return result.stdout.strip()


def _checkout(source: Source, destination: Path) -> None:
    destination.mkdir()
    _git(destination, "init", "--quiet")
    try:
        _git(destination, "fetch", "--quiet", "--depth=1", source.url, source.revision)
    except FetchError as error:
        raise FetchError(
            f"Cannot fetch revision {source.revision} from {source.url}: {error}\n"
            "Next: check network access and the upstream URL/revision in vendor/PROVENANCE.md, then retry."
        ) from error
    _git(destination, "-c", "core.autocrlf=false", "checkout", "--quiet", "--detach", "FETCH_HEAD")
    actual_revision = _git(destination, "rev-parse", "HEAD")
    actual_tree = _git(destination, "rev-parse", "HEAD^{tree}")
    if actual_revision != source.revision or actual_tree != source.tree:
        raise FetchError(
            f"Upstream identity mismatch for {source.url}: revision={actual_revision}, tree={actual_tree}.\n"
            "Next: verify the pinned revision and tree in vendor/PROVENANCE.md."
        )


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _prune(source: Path) -> None:
    for name in (".git", ".gitmodules", ".github", "Dockerfile", "docker"):
        _remove(source / name)
    for pattern in ("build*.sh", "*.cmd", "**/.git", "**/.gitmodules"):
        for path in source.glob(pattern):
            _remove(path)


def _flatten_symlinks(source: Path) -> None:
    for path in source.rglob("*"):
        if not path.is_symlink():
            continue
        target = path.resolve(strict=True)
        if not target.is_relative_to(source) or not target.is_file():
            raise FetchError(f"Cannot flatten source symlink: {path.relative_to(source)}")
        content = target.read_bytes()
        mode = target.stat().st_mode
        path.unlink()
        path.write_bytes(content)
        path.chmod(mode)


def verify_snapshot(source: Path, manifest: dict[str, str]) -> None:
    actual = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise FetchError(f"Source contains a symlink: {path.relative_to(source)}")
        if path.is_file():
            name = f"qwentts.cpp/{path.relative_to(source).as_posix()}"
            actual[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    differences = [f"missing: {name}" for name in sorted(manifest.keys() - actual.keys())]
    differences.extend(f"unexpected: {name}" for name in sorted(actual.keys() - manifest.keys()))
    differences.extend(
        f"SHA-256 mismatch: {name}"
        for name in sorted(manifest.keys() & actual.keys())
        if manifest[name] != actual[name]
    )
    if differences:
        raise FetchError("Manifest verification failed:\n" + "\n".join(differences))


def fetch_vendor(root: Path) -> Path:
    vendor = root / "vendor"
    destination = vendor / "qwentts.cpp"
    if destination.exists() or destination.is_symlink():
        raise FetchError(
            "vendor/qwentts.cpp already exists.\n"
            "Next: keep the existing snapshot, or move it aside before retrying python3 scripts/fetch_vendor.py."
        )
    sources = load_sources(root)
    manifest = load_manifest(root)
    with tempfile.TemporaryDirectory(prefix=".fetch-", dir=vendor) as temporary:
        staging = Path(temporary)
        source = staging / "vendor/qwentts.cpp"
        source.parent.mkdir()
        _checkout(sources["qwentts"], source)
        _remove(source / "ggml")
        _checkout(sources["ggml"], source / "ggml")
        _flatten_symlinks(source)
        _prune(source / "ggml")
        _prune(source)
        for name in ("base.cmd", "clone.cmd", "customvoice.cmd", "tts.cmd"):
            _remove(source / "examples" / name)
        # An isolated temporary repository makes patch paths independent of the enclosing checkout.
        _git(staging, "init", "--quiet")
        for patch in sorted((vendor / "patches").glob("*.patch")):
            try:
                _git(staging, "apply", "--unidiff-zero", str(patch.resolve()))
            except FetchError as error:
                raise FetchError(f"Cannot apply patch {patch.name}: {error}") from error
        verify_snapshot(source, manifest)
        source.rename(destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=ROOT, help="checkout containing vendor/PROVENANCE.md and manifest"
    )
    arguments = parser.parse_args()
    try:
        destination = fetch_vendor(arguments.repo_root.resolve())
    except (OSError, ValueError) as error:
        print(f"Cannot prepare native source: {error}", file=sys.stderr)
        if "Next:" not in str(error):
            print(
                "Next: restore matching vendor/PROVENANCE.md, MANIFEST.sha256, and patches, then retry "
                "python3 scripts/fetch_vendor.py.",
                file=sys.stderr,
            )
        return 1
    print(f"Verified native source written to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
