from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

NATIVE_PAYLOAD_FILENAMES = (
    "libqwen.so",
    "BUILD-METADATA.json",
)


def configure_wheel_build(
    *,
    version: str,
    target_name: str,
    root: Path,
    build_data: dict[str, Any],
    run_native_build: Callable[[], None],
) -> None:
    """Compile and attach the platform payload for a standard wheel build."""
    if target_name != "wheel" or version == "editable":
        return
    if version != "standard":
        return

    run_native_build()
    native_root = root / "src/speechflow/qwen3/_native"
    payload = [native_root / filename for filename in NATIVE_PAYLOAD_FILENAMES]
    missing = [path for path in payload if not path.is_file() or path.is_symlink()]
    if missing:
        paths = ", ".join(str(path) for path in missing)
        raise RuntimeError(f"native payload is incomplete after build: {paths}")

    build_data["tag"] = "py3-none-manylinux_2_38_x86_64"
    build_data["pure_python"] = False
    force_include = build_data.setdefault("force_include", {})
    for path in payload:
        destination = f"speechflow/qwen3/_native/{path.name}"
        force_include[str(path)] = destination


class Qwen3BuildHook(BuildHookInterface[Any]):
    """Build qwentts.cpp for standard wheels while keeping editable sync GPU-free."""

    PLUGIN_NAME = "qwen3-platform"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        root = Path(self.root)
        if self.target_name == "sdist":
            # Hatch force-includes VCS ignore files outside the sdist allowlist.
            force_include = build_data.get("force_include", {})
            for source, destination in tuple(force_include.items()):
                if destination in {".gitignore", ".hgignore"}:
                    del force_include[source]
            return

        def run_native_build() -> None:
            subprocess.run(
                [root / "scripts/build_native.sh"],
                cwd=root,
                check=True,
            )

        configure_wheel_build(
            version=version,
            target_name=self.target_name,
            root=root,
            build_data=build_data,
            run_native_build=run_native_build,
        )
