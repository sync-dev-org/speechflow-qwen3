from __future__ import annotations

from pathlib import Path

import pytest
from hatch_build import NATIVE_PAYLOAD_FILENAMES, configure_wheel_build


def test_editable_wheel_does_not_compile_or_change_tag(tmp_path: Path) -> None:
    build_data: dict[str, object] = {}

    def unexpected_build() -> None:
        raise AssertionError("editable build attempted native compilation")

    configure_wheel_build(
        version="editable",
        target_name="wheel",
        root=tmp_path,
        build_data=build_data,
        run_native_build=unexpected_build,
    )

    assert build_data == {}


@pytest.mark.req("SP-SFQWEN3-024")
@pytest.mark.req("SP-SFQWEN3-015")
def test_standard_wheel_builds_and_force_includes_complete_payload(tmp_path: Path) -> None:
    """A standard build includes the Qwen3 native payload in a platform wheel."""
    native_root = tmp_path / "src/speechflow/qwen3/_native"
    build_calls = 0

    def fake_build() -> None:
        nonlocal build_calls
        build_calls += 1
        native_root.mkdir(parents=True)
        for filename in NATIVE_PAYLOAD_FILENAMES:
            (native_root / filename).write_bytes(b"native")

    build_data: dict[str, object] = {}
    configure_wheel_build(
        version="standard",
        target_name="wheel",
        root=tmp_path,
        build_data=build_data,
        run_native_build=fake_build,
    )

    assert build_calls == 1
    assert build_data["tag"] == "py3-none-manylinux_2_38_x86_64"
    assert build_data["pure_python"] is False
    assert set(build_data["force_include"].values()) == {
        f"speechflow/qwen3/_native/{filename}" for filename in NATIVE_PAYLOAD_FILENAMES
    }


@pytest.mark.req("SP-SFQWEN3-024")
@pytest.mark.req("SP-SFQWEN3-015")
def test_standard_wheel_rejects_incomplete_payload(tmp_path: Path) -> None:
    """A standard build cannot produce a wheel with an incomplete native payload."""
    native_root = tmp_path / "src/speechflow/qwen3/_native"

    def incomplete_build() -> None:
        native_root.mkdir(parents=True)
        (native_root / NATIVE_PAYLOAD_FILENAMES[0]).write_bytes(b"native")

    with pytest.raises(RuntimeError, match="native payload is incomplete"):
        configure_wheel_build(
            version="standard",
            target_name="wheel",
            root=tmp_path,
            build_data={},
            run_native_build=incomplete_build,
        )
