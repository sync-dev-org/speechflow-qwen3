"""The public repository retains build inputs without development publication files."""

from __future__ import annotations

import ast
import re
import subprocess
import tomllib
import unicodedata
from pathlib import Path
from typing import cast
from urllib.parse import unquote, urlsplit

import pytest
import yaml  # type: ignore[import-untyped]
from pathspec import GitIgnoreSpec

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def publication() -> dict[str, str]:
    configuration = ROOT / "noren.yaml"
    if not configuration.is_file():
        pytest.skip("the public repository does not contain development publication settings")
    result = yaml.safe_load(configuration.read_text(encoding="utf-8"))
    assert isinstance(result, dict)
    for key in ("destination", "branch", "include", "exclude", "word-list"):
        assert isinstance(result.get(key), str), f"noren.yaml requires the string setting {key}"
    return cast(dict[str, str], result)


@pytest.fixture
def tracked_files() -> list[str]:
    return (
        subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True, text=True)
        .stdout.rstrip("\0")
        .split("\0")
    )


@pytest.fixture
def public_files(publication: dict[str, str], tracked_files: list[str]) -> list[str]:
    include = GitIgnoreSpec.from_lines(publication["include"].splitlines())
    exclude = GitIgnoreSpec.from_lines(publication["exclude"].splitlines())
    unclassified = [name for name in tracked_files if not include.match_file(name) and not exclude.match_file(name)]
    assert not unclassified, f"files missing from include and exclude: {unclassified}"
    return [name for name in tracked_files if include.match_file(name) and not exclude.match_file(name)]


@pytest.mark.req("SP-SFQWEN3-026")
def test_public_repository_retains_fetch_inputs_and_excludes_upstream_sources(
    publication: dict[str, str], public_files: list[str]
) -> None:
    """The public repository provides reproducible fetch inputs and excludes the upstream source snapshot."""
    assert publication["destination"] == "sync-dev-org/speechflow-qwen3"
    assert publication["branch"] == "main"
    assert publication["word-list"] == "noren-words.txt"
    assert not any(name.startswith("vendor/qwentts.cpp/") for name in public_files)
    assert {
        "vendor/PROVENANCE.md",
        "vendor/MANIFEST.sha256",
        "scripts/fetch_vendor.py",
        "scripts/vendor_source.py",
        "vendor/patches/0001-ggml-disable-runtime-backend-loading.patch",
    } <= set(public_files)
    assert {".gitignore", "LICENSE", "README.md", "hatch_build.py", "pyproject.toml", "uv.lock"} <= set(public_files)
    for directory in ("docs/", "scripts/", "src/", "tests/", "vendor/"):
        assert any(name.startswith(directory) for name in public_files), f"missing public sources: {directory}"


@pytest.mark.req("SP-SFQWEN3-026")
def test_source_archive_build_inputs_are_published_or_reconstructed_from_the_manifest(
    public_files: list[str], tracked_files: list[str]
) -> None:
    """Source archive inputs are public files or native sources covered by the published manifest."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sdist = project["tool"]["hatch"]["build"]["targets"]["sdist"]
    reconstructed = {
        f"vendor/{line.split('  ', 1)[1]}"
        for line in (ROOT / "vendor/MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    }
    for name in sdist["only-include"]:
        assert (ROOT / name).exists(), f"missing source archive input: {name}"
        inputs = {file for file in tracked_files if file == name or file.startswith(f"{name}/")}
        assert inputs, f"source archive input has no tracked files: {name}"
        assert inputs <= set(public_files) | reconstructed, (
            f"source archive inputs are not available: {sorted(inputs - set(public_files) - reconstructed)}"
        )


def _imported_modules(tree: ast.AST, path: Path) -> list[tuple[int, str]]:
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            parent = list(path.parent.parts)
            if node.level:
                parent = parent[: len(parent) - node.level + 1]
                module = ".".join([*parent, *([node.module] if node.module else [])])
            else:
                module = node.module or ""
            imports.append((node.lineno, module))
            imports.extend((node.lineno, f"{module}.{alias.name}") for alias in node.names)
        elif isinstance(node, ast.Call) and node.args:
            function = node.func
            if (isinstance(function, ast.Name) and function.id == "__import__") or (
                isinstance(function, ast.Attribute) and function.attr == "import_module"
            ):
                argument = node.args[0]
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    imports.append((node.lineno, argument.value))
    return imports


@pytest.mark.req("SP-SFQWEN3-026")
def test_public_documents_and_python_imports_do_not_require_excluded_files(
    publication: dict[str, str], public_files: list[str]
) -> None:
    """Public Markdown links and Python imports remain usable without excluded development files."""
    exclude = GitIgnoreSpec.from_lines(publication["exclude"].splitlines())
    violations = []
    for name in public_files:
        path = Path(name)
        content = (ROOT / path).read_text(encoding="utf-8", errors="replace")
        if path.suffix == ".md":
            links = re.finditer(r"\[[^\]]*\]\(\s*<?([^\s)>]+)|^\s*\[[^\]]+\]:\s*<?([^\s>]+)", content, re.MULTILINE)
            for link in links:
                target = urlsplit(link.group(1) or link.group(2))
                if target.scheme or target.netloc or not target.path:
                    continue
                resolved = (ROOT / path.parent / unquote(target.path)).resolve()
                if resolved.is_relative_to(ROOT) and exclude.match_file(resolved.relative_to(ROOT).as_posix()):
                    line = content.count("\n", 0, link.start()) + 1
                    violations.append(f"{name}:{line}: excluded link {target.path}")
        elif path.suffix == ".py":
            tree = ast.parse(content, filename=name)
            for line, module in _imported_modules(tree, path):
                module_path = module.replace(".", "/")
                candidates = (f"{module_path}.py", f"{module_path}/__init__.py", f"{path.parent}/{module_path}.py")
                if any(exclude.match_file(candidate) for candidate in candidates):
                    violations.append(f"{name}:{line}: excluded import {module}")
    assert not violations, "\n".join(violations)


@pytest.mark.req("SP-SFQWEN3-026")
def test_public_file_contents_do_not_contain_forbidden_words(
    publication: dict[str, str], public_files: list[str]
) -> None:
    """No public file contains a forbidden word, including case variants and full-width characters."""
    words = (ROOT / publication["word-list"]).read_text(encoding="utf-8").splitlines()
    words = [unicodedata.normalize("NFKC", word.strip()).casefold() for word in words if word.strip()]
    assert words, "the publication word list must not be empty"
    violations = []
    for name in public_files:
        content = unicodedata.normalize("NFKC", (ROOT / name).read_bytes().decode("utf-8", errors="replace")).casefold()
        for word in words:
            position = content.find(word)
            if position >= 0:
                line = content.count("\n", 0, position) + 1
                violations.append(f"{name}:{line}: forbidden word {word}")
    assert not violations, "\n".join(violations)
