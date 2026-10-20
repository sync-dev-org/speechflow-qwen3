"""Read native source identities from the TOML block in vendor/PROVENANCE.md."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Source:
    url: str
    revision: str
    tree: str


def load_sources(root: Path = ROOT) -> dict[str, Source]:
    path = root / "vendor/PROVENANCE.md"
    blocks = re.findall(r"^```toml\n(.*?)^```\s*$", path.read_text(encoding="utf-8"), re.MULTILINE | re.DOTALL)
    if len(blocks) != 1:
        raise ValueError(f"{path}: expected one TOML source identity block")
    document = tomllib.loads(blocks[0])
    sources = {}
    for component in ("qwentts", "ggml"):
        entry = document.get(component, {})
        for key in ("url", "revision", "tree"):
            value = entry.get(key)
            if not isinstance(value, str) or not value or (key != "url" and not re.fullmatch(r"[0-9a-f]{40}", value)):
                raise ValueError(f"{path}: {component}.{key} is missing or invalid")
        sources[component] = Source(entry["url"], entry["revision"], entry["tree"])
    return sources


def load_manifest(root: Path = ROOT) -> dict[str, str]:
    path = root / "vendor/MANIFEST.sha256"
    entries = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (qwentts\.cpp/[^\\]+)", line)
        if match is None:
            raise ValueError(f"{path}:{number}: expected SHA-256 and a qwentts.cpp/ file path")
        digest, name = match.groups()
        relative = PurePosixPath(name)
        if ".." in relative.parts or relative.as_posix() != name or name in entries:
            raise ValueError(f"{path}:{number}: duplicate or invalid file path {name!r}")
        entries[name] = digest
    if not entries:
        raise ValueError(f"{path}: manifest has no source files")
    return entries


def build_metadata(cuda_toolkit_version: str, architectures: str) -> dict[str, str | int]:
    sources = load_sources()
    return {
        "schema_version": 1,
        "build_type": "Release",
        "cpu_baseline": "x86-64-v3",
        "qwentts_revision": sources["qwentts"].revision,
        "ggml_revision": sources["ggml"].revision,
        "cuda_toolkit_version": cuda_toolkit_version,
        "cmake_cuda_architectures": architectures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    revision = commands.add_parser("revision", help="print the pinned full revision")
    revision.add_argument("component", choices=("qwentts", "ggml"))
    metadata = commands.add_parser("metadata", help="write native build metadata to stdout")
    metadata.add_argument("cuda_toolkit_version")
    metadata.add_argument("architectures")
    arguments = parser.parse_args()
    try:
        if arguments.command == "revision":
            print(load_sources()[arguments.component].revision)
        else:
            print(json.dumps(build_metadata(arguments.cuda_toolkit_version, arguments.architectures), indent=2))
    except (OSError, ValueError) as error:
        print(f"Cannot read native source identities: {error}\nNext: restore vendor/PROVENANCE.md.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
