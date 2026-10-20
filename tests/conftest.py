from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def native_source() -> Path:
    source = ROOT / "vendor/qwentts.cpp"
    if not source.is_dir():
        pytest.skip(
            "native source is absent; run python3 scripts/fetch_vendor.py to fetch and verify vendor/qwentts.cpp"
        )
    return source
