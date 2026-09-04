"""pytest 公共 fixture：把 lab/src 加进 sys.path；MINIMIND_ROOT 未设置时依赖 tokenizer/模型的测试自动 skip。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
SRC = LAB / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def lab_root() -> Path:
    return LAB


@pytest.fixture(scope="session")
def minimind_root():
    from mm_probe.minimind_env import MiniMindNotFound, find_minimind_root

    try:
        return find_minimind_root()
    except MiniMindNotFound as e:
        pytest.skip(f"需要 MINIMIND_ROOT：{e}")


@pytest.fixture(scope="session")
def tokenizer(minimind_root):
    pytest.importorskip("transformers")
    from mm_probe.minimind_env import load_tokenizer

    return load_tokenizer(minimind_root)
