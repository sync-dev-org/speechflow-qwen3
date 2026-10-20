"""Standalone repository and wheel contracts for speechflow-qwen3."""

from __future__ import annotations

import ast
import email
import hashlib
import importlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from speechflow.core import TTSEngineBase

ROOT = Path(__file__).parents[1]
PACKAGE_ROOT = ROOT / "src/speechflow/qwen3"
CORE_PYPI_REGISTRY = "https://pypi.org/simple"
RUNTIME_REQUIREMENTS = [
    "speechflow>=0.19.2",
    "numpy>=1.26.4",
    "nvidia-cuda-runtime==13.3.29; sys_platform == 'linux' and platform_machine == 'x86_64'",
    "nvidia-cublas==13.6.0.2; sys_platform == 'linux' and platform_machine == 'x86_64'",
]
NATIVE_PAYLOAD_FILENAMES = (
    "libqwen.so",
    "BUILD-METADATA.json",
)


def _config(root: Path = ROOT) -> dict[str, Any]:
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"):
        environment.pop(name, None)
    return subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=check,
        capture_output=True,
        text=True,
    )


def _lock_package(lock: dict[str, Any], name: str) -> dict[str, Any]:
    packages: list[dict[str, Any]] = lock["package"]
    matches = [package for package in packages if package["name"] == name]
    assert len(matches) == 1
    return matches[0]


def _requirement_signature(requirement: str) -> tuple[str, frozenset[str], str | None, str | None]:
    parsed = Requirement(requirement)
    return (
        canonicalize_name(parsed.name),
        frozenset(str(parsed.specifier).split(",")),
        parsed.url,
        str(parsed.marker) if parsed.marker is not None else None,
    )


def _wheel_metadata(wheel: Path) -> tuple[email.message.Message, set[str]]:
    with zipfile.ZipFile(wheel) as archive:
        files = set(archive.namelist())
        metadata_path = next(path for path in files if path.endswith(".dist-info/METADATA"))
        metadata = email.message_from_bytes(archive.read(metadata_path))
    return metadata, files


def _commit(repository: Path, message: str) -> str:
    _run(("git", "add", "."), cwd=repository)
    _run(
        (
            "git",
            "-c",
            "user.name=speechflow-qwen3-test",
            "-c",
            "user.email=speechflow-qwen3-test@example.invalid",
            "commit",
            "-m",
            message,
        ),
        cwd=repository,
    )
    return _run(("git", "rev-parse", "HEAD"), cwd=repository).stdout.strip()


def _write_stub_native_builder(repository: Path) -> None:
    script = repository / "scripts/build_native.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    payload_names = " ".join(NATIVE_PAYLOAD_FILENAMES)
    script.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        "native_root=src/speechflow/qwen3/_native\n"
        'mkdir -p "$native_root"\n'
        f"for name in {payload_names}; do printf 'fixture payload\\n' > \"$native_root/$name\"; done\n",
        encoding="utf-8",
    )
    script.chmod(0o755)


def _copy_tracked_module_fixture(repository: Path) -> set[str]:
    tracked = set(
        _run(
            (
                "git",
                "ls-files",
                "--",
                "hatch_build.py",
                "src",
                "tests/conftest.py",
                "tests/test_qwen3_engine_contract.py",
            ),
            cwd=ROOT,
        ).stdout.splitlines()
    )
    for relative in tracked:
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    if (ROOT / "LICENSE").is_file():
        shutil.copy2(ROOT / "LICENSE", repository / "LICENSE")
    _write_stub_native_builder(repository)
    return tracked


def _build_fixture_wheel(repository: Path, wheelhouse: Path) -> Path:
    _run(("uv", "build", "--offline", "--wheel", "--out-dir", str(wheelhouse)), cwd=repository)
    wheels = list(wheelhouse.glob("speechflow_qwen3-*.whl"))
    assert len(wheels) == 1
    return wheels[0]


@pytest.fixture(scope="session")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    repository = tmp_path_factory.mktemp("qwen3-wheel-source")
    shutil.copy2(ROOT / "pyproject.toml", repository / "pyproject.toml")
    copied = _copy_tracked_module_fixture(repository)
    assert "hatch_build.py" in copied
    assert any(path.startswith("src/speechflow/qwen3/") for path in copied)
    assert not (repository / "src/speechflow/qwen3/_native").exists()
    _run(("git", "init", "--initial-branch=main"), cwd=repository)
    _commit(repository, "fixture source")
    _run(("git", "tag", "v9.8.7"), cwd=repository)
    return _build_fixture_wheel(repository, tmp_path_factory.mktemp("qwen3-wheel"))


@pytest.mark.req("SP-SFQWEN3-010")
def test_project_metadata_and_build_contract() -> None:
    """Project metadata declares the compatible core minimum and the module build contract."""
    config = _config()
    project = config["project"]

    assert project["name"] == "speechflow-qwen3"
    assert project["dynamic"] == ["version"]
    assert project["requires-python"] == ">=3.12,<3.14"
    assert project["dependencies"] == RUNTIME_REQUIREMENTS
    assert "optional-dependencies" not in project
    assert config["dependency-groups"]["dev"] == [
        "hatchling",
        "mypy>=2.2.0",
        "pytest>=8.3.5",
        "pytest-asyncio>=0.24.0",
        "pyyaml>=6.0.3",
        "ruff",
    ]
    uv_config = config.get("tool", {}).get("uv", {})
    assert "sources" not in uv_config
    assert "workspace" not in uv_config
    assert "prerelease" not in uv_config

    version = config["tool"]["hatch"]["version"]
    assert version["source"] == "vcs"
    assert version["fallback-version"] == "0.0.0.dev0"
    assert version["raw-options"]["version_scheme"] == "only-version"
    assert version["raw-options"]["local_scheme"] == "no-local-version"
    assert version["raw-options"]["tag_regex"] == r"^v(?P<version>[^/]+)$"
    assert version["raw-options"]["git_describe_command"] == [
        "git",
        "describe",
        "--dirty",
        "--tags",
        "--long",
        "--exact-match",
        "--match",
        "v*",
    ]
    assert config["tool"]["hatch"]["build"]["hooks"]["vcs"]["version-file"] == ("src/speechflow/qwen3/_version.py")
    assert config["tool"]["hatch"]["build"]["hooks"]["custom"]["path"] == "hatch_build.py"
    assert config["tool"]["hatch"]["build"]["targets"]["wheel"]["only-include"] == ["src/speechflow/qwen3"]
    assert config["tool"]["hatch"]["build"]["targets"]["wheel"]["sources"] == ["src"]


@pytest.mark.req("SP-SFQWEN3-011")
def test_default_pytest_excludes_resource_gates() -> None:
    """The default test command excludes GPU, native build, and native payload gates."""
    pytest_config = _config()["tool"]["pytest"]["ini_options"]

    assert pytest_config["addopts"] == "--strict-markers -m 'not gpu and not native_build and not native_payload'"
    assert {"gpu", "native_build", "native_payload"} <= {
        marker.partition(":")[0].strip() for marker in pytest_config["markers"]
    }


@pytest.mark.req("SP-SFQWEN3-010")
@pytest.mark.req("SP-SFQWEN3-019")
def test_module_local_lock_records_the_accepted_core_and_cuda_identities() -> None:
    """The module lock pins the public core version and file hashes alongside the CUDA versions."""
    lock_path = ROOT / "uv.lock"
    assert lock_path.is_file()
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    core = _lock_package(lock, "speechflow")

    assert core["version"] == "0.19.2"
    assert core["source"] == {"registry": CORE_PYPI_REGISTRY}
    artifacts = [core["sdist"], *core["wheels"]]
    assert core["wheels"]
    for artifact in artifacts:
        assert artifact["url"].startswith("https://files.pythonhosted.org/")
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", artifact["hash"])
    module = _lock_package(lock, "speechflow-qwen3")
    assert all(requirement["name"] != "speechflow" for requirement in module["metadata"]["requires-dev"]["dev"])
    assert _lock_package(lock, "nvidia-cuda-runtime")["version"] == "13.3.29"
    assert _lock_package(lock, "nvidia-cublas")["version"] == "13.6.0.2"
    assert "uv.lock" not in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()


@pytest.mark.req("SP-SFQWEN3-011")
@pytest.mark.req("SP-SFQWEN3-024")
@pytest.mark.req("SP-SFQWEN3-010")
def test_built_wheel_has_public_dependencies_and_owns_only_its_subtree(built_wheel: Path) -> None:
    """The Qwen3 wheel owns its Python subtree and declares its runtime dependencies."""
    metadata, files = _wheel_metadata(built_wheel)

    assert metadata["Name"] == "speechflow-qwen3"
    assert frozenset(metadata["Requires-Python"].split(",")) == {">=3.12", "<3.14"}
    assert {_requirement_signature(requirement) for requirement in metadata.get_all("Requires-Dist", [])} == {
        _requirement_signature(requirement) for requirement in RUNTIME_REQUIREMENTS
    }
    assert metadata.get_all("Provides-Extra", []) == []
    assert metadata["Version"] == "9.8.7"
    assert built_wheel.name.endswith("-py3-none-manylinux_2_38_x86_64.whl")

    package_files = {path for path in files if ".dist-info/" not in path}
    assert "speechflow/qwen3/py.typed" in package_files
    assert "speechflow/__init__.py" not in package_files
    assert "speechflow/py.typed" not in package_files
    assert all(path.startswith("speechflow/qwen3/") for path in package_files)


@pytest.mark.req("SP-SFQWEN3-025")
def test_built_wheel_carries_mit_metadata_and_complete_license_texts(built_wheel: Path) -> None:
    """The wheel declares MIT and preserves the module and third-party license texts."""
    metadata, files = _wheel_metadata(built_wheel)
    assert metadata["License-Expression"] == "MIT"
    assert metadata.get_all("License-File") == ["LICENSE"]
    license_name = next(path for path in files if path.endswith(".dist-info/licenses/LICENSE"))
    with zipfile.ZipFile(built_wheel) as archive:
        assert archive.read(license_name) == (ROOT / "LICENSE").read_bytes()
        assert b"Copyright (c) 2026 minamik" in archive.read(license_name)
        for path in (PACKAGE_ROOT / "_licenses").iterdir():
            assert archive.read(f"speechflow/qwen3/_licenses/{path.name}") == path.read_bytes()


@pytest.mark.req("SP-SFQWEN3-024")
@pytest.mark.req("SP-SFQWEN3-025")
def test_built_sdist_contains_only_build_sources_and_can_build_a_wheel(tmp_path: Path, native_source: Path) -> None:
    """The source archive excludes development files and retains every wheel build input."""
    _run(("uv", "build", "--offline", "--sdist", "--out-dir", str(tmp_path / "dist")), cwd=ROOT)
    sdist = next((tmp_path / "dist").glob("*.tar.gz"))
    allowed_files = {
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "hatch_build.py",
        "scripts/build_native.sh",
        "scripts/vendor_source.py",
        "scripts/qwen.exports.map",
        "scripts/native-tools/git",
        "PKG-INFO",
    }
    allowed_directories = ("src/speechflow/qwen3/", "vendor/")
    with tarfile.open(sdist) as archive:
        names = {member.name.split("/", 1)[1] for member in archive.getmembers() if member.isfile()}
        unexpected = {name for name in names if name not in allowed_files and not name.startswith(allowed_directories)}
        assert not unexpected
        assert allowed_files <= names
        assert {
            "vendor/qwentts.cpp/CMakeLists.txt",
            "vendor/qwentts.cpp/ggml/CMakeLists.txt",
            "src/speechflow/qwen3/weights-manifest-v1.json",
            "src/speechflow/qwen3/_version.py",
        } <= names
        assert not any(name.startswith("src/speechflow/qwen3/_native/") or name.endswith(".gguf") for name in names)
        archive.extractall(tmp_path / "unpacked", filter="data")
    repository = next((tmp_path / "unpacked").iterdir())
    _write_stub_native_builder(repository)
    wheel = _build_fixture_wheel(repository, tmp_path / "wheelhouse")
    metadata, _ = _wheel_metadata(wheel)
    assert metadata["License-Expression"] == "MIT"
    assert wheel.name.endswith("-py3-none-manylinux_2_38_x86_64.whl")


@pytest.mark.req("SP-SFQWEN3-011")
def test_package_subtree_owns_only_its_namespace_slice() -> None:
    """The Qwen3 module owns only its namespace subtree."""
    assert (PACKAGE_ROOT / "__init__.py").is_file()
    assert (PACKAGE_ROOT / "py.typed").is_file()
    assert not (ROOT / "src/speechflow/__init__.py").exists()
    assert not (ROOT / "src/speechflow/py.typed").exists()


@pytest.mark.req("SP-SFQWEN3-011")
def test_package_import_is_lazy_and_exports_public_engine_types() -> None:
    """Importing the Qwen3 module exposes its public engine classes without loading native payload."""
    sys.modules.pop("speechflow.qwen3._payload", None)
    sys.modules.pop("speechflow.qwen3", None)
    module = importlib.import_module("speechflow.qwen3")

    assert module.__all__ == [
        "Qwen3CloneTTSEngine",
        "Qwen3CustomVoiceTTSEngine",
        "Qwen3ModelSource",
        "__version__",
    ]
    assert isinstance(module.__version__, str)
    assert "speechflow.qwen3._payload" not in sys.modules
    assert module.Qwen3CloneTTSEngine.__name__ == "Qwen3CloneTTSEngine"
    assert module.Qwen3CustomVoiceTTSEngine.__name__ == "Qwen3CustomVoiceTTSEngine"
    assert module.Qwen3ModelSource.__name__ == "Qwen3ModelSource"


@pytest.mark.req("SP-SFQWEN3-011")
def test_module_source_uses_only_public_core_surface() -> None:
    """Qwen3 module source uses only the core surface published for module authors."""
    private_base_attributes = {
        name for name in vars(TTSEngineBase) if name.startswith("_") and not name.startswith("__")
    }
    violations: list[str] = []

    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in private_base_attributes:
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: core attribute {node.attr}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    if parts[:2] == ["speechflow", "core"] and any(part.startswith("_") for part in parts[2:]):
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                parts = node.module.split(".")
                if parts[:2] == ["speechflow", "core"] and (
                    any(part.startswith("_") for part in parts[2:])
                    or any(alias.name.startswith("_") for alias in node.names)
                ):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: import from {node.module}")

    assert violations == []


def _installed_core_package() -> Path:
    result = _run(
        (
            "uv",
            "run",
            "--offline",
            "python",
            "-c",
            "from pathlib import Path; import speechflow; print(Path(speechflow.__file__).resolve().parent)",
        ),
        cwd=ROOT,
    )
    package = Path(result.stdout.strip())
    assert package.is_relative_to((ROOT / ".venv").resolve())
    assert (package / "__init__.py").is_file()
    assert not (package / "qwen3").exists()
    return package


def _write_core_project(repository: Path, version: str, *, package_source: Path | None = None) -> None:
    package = repository / "src/speechflow"
    if package_source is not None:
        shutil.copytree(package_source, package, ignore=shutil.ignore_patterns("__pycache__"))
    assert package.is_dir()

    version_file = package / "_version.py"
    version_source = version_file.read_text(encoding="utf-8")
    version_source, version_count = re.subn(
        r"^__version__ = version = ['\"][^'\"]+['\"]$",
        f"__version__ = version = {version!r}",
        version_source,
        count=1,
        flags=re.MULTILINE,
    )
    version_tuple = tuple(int(part) for part in version.split("."))
    version_source, tuple_count = re.subn(
        r"^__version_tuple__ = version_tuple = .+$",
        f"__version_tuple__ = version_tuple = {version_tuple!r}",
        version_source,
        count=1,
        flags=re.MULTILINE,
    )
    assert version_count == tuple_count == 1
    version_file.write_text(version_source, encoding="utf-8")
    (repository / "pyproject.toml").write_text(
        f"""
[project]
name = "speechflow"
version = "{version}"
requires-python = ">=3.12,<3.14"
dependencies = ["numpy>=1.26.4"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/speechflow"]
""".lstrip(),
        encoding="utf-8",
    )


def _other_locked_packages(lock: dict[str, Any]) -> list[dict[str, Any]]:
    return [package for package in lock["package"] if package["name"] != "speechflow"]


def _publish_core_wheel(repository: Path, index: Path) -> None:
    package_index = index / "speechflow"
    _run(("uv", "build", "--offline", "--wheel", "--out-dir", str(package_index)), cwd=repository)
    links = [
        f'<a href="{wheel.name}#sha256={hashlib.sha256(wheel.read_bytes()).hexdigest()}">{wheel.name}</a>'
        for wheel in sorted(package_index.glob("*.whl"))
    ]
    (package_index / "index.html").write_text("\n".join(links), encoding="utf-8")


@pytest.mark.req("SP-SFQWEN3-010")
@pytest.mark.req("SP-SFQWEN3-026")
def test_core_lock_update_advances_only_the_registry_core(tmp_path: Path) -> None:
    """A core registry update installs the new wheel and leaves all other locked packages unchanged."""
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    assert _lock_package(lock, "speechflow")["source"] == {"registry": CORE_PYPI_REGISTRY}

    index = tmp_path / "simple"
    core = tmp_path / "core-work"
    module = tmp_path / "module"
    core.mkdir()
    module.mkdir()
    _write_core_project(core, "0.19.2", package_source=_installed_core_package())
    _publish_core_wheel(core, index)

    shutil.copy2(ROOT / "pyproject.toml", module / "pyproject.toml")
    (module / "uv.lock").write_bytes((ROOT / "uv.lock").read_bytes())
    copied = _copy_tracked_module_fixture(module)
    assert {"tests/conftest.py", "tests/test_qwen3_engine_contract.py"} <= copied
    assert any(path.startswith("src/speechflow/qwen3/") for path in copied)
    assert not (module / "src/speechflow/qwen3/_native").exists()
    identity_test = module / "tests/test_upgraded_core_identity.py"
    identity_test.write_text(
        """\
from pathlib import Path
import sys

import speechflow


def test_upgraded_core_is_imported_from_the_module_environment() -> None:
    assert Path(speechflow.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    assert speechflow.__version__ == "0.19.3"
""",
        encoding="utf-8",
    )

    # A filesystem simple index exercises registry resolution without external network access.
    index_arguments = ("--index", index.as_uri())
    update = ("uv", "lock", "--upgrade-package", "speechflow", "--offline", *index_arguments)
    _run(update, cwd=module)
    before = tomllib.loads((module / "uv.lock").read_text(encoding="utf-8"))
    first_core = _lock_package(before, "speechflow")
    assert first_core["version"] == "0.19.2"
    assert first_core["source"] == {"registry": str(index)}

    _write_core_project(core, "0.19.3")
    _publish_core_wheel(core, index)
    _run(update, cwd=module)
    after = tomllib.loads((module / "uv.lock").read_text(encoding="utf-8"))
    second_core = _lock_package(after, "speechflow")

    assert second_core["version"] == "0.19.3"
    assert second_core["source"] == {"registry": str(index)}
    second_wheel = index / "speechflow/speechflow-0.19.3-py3-none-any.whl"
    assert second_core["wheels"][0]["hash"] == f"sha256:{hashlib.sha256(second_wheel.read_bytes()).hexdigest()}"
    assert _other_locked_packages(after) == _other_locked_packages(before)

    _run(("uv", "lock", "--check", "--offline", *index_arguments), cwd=module)
    _run(("uv", "sync", "--offline", *index_arguments), cwd=module)
    core_probe = _run(
        (
            "uv",
            "run",
            "--offline",
            *index_arguments,
            "python",
            "-c",
            "from pathlib import Path; import speechflow; "
            "print(Path(speechflow.__file__).resolve()); print(speechflow.__version__)",
        ),
        cwd=module,
    ).stdout.splitlines()
    assert Path(core_probe[0]).is_relative_to((module / ".venv").resolve())
    assert core_probe[1] == "0.19.3"
    _run(
        (
            "uv",
            "run",
            "--offline",
            *index_arguments,
            "pytest",
            "-p",
            "no:cacheprovider",
            "tests/test_qwen3_engine_contract.py",
            "tests/test_upgraded_core_identity.py",
        ),
        cwd=module,
    )
    wheel = _build_fixture_wheel(module, tmp_path / "wheelhouse")
    metadata, _ = _wheel_metadata(wheel)
    assert {_requirement_signature(requirement) for requirement in metadata.get_all("Requires-Dist", [])} == {
        _requirement_signature(requirement) for requirement in RUNTIME_REQUIREMENTS
    }
    assert wheel.name.endswith("-py3-none-manylinux_2_38_x86_64.whl")
