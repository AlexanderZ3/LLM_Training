"""pytest 公共设施：路径、MiniMind 定位、CPU gloo 多进程启动器。

这些测试全部在 **CPU + gloo** 上跑，不需要 GPU，也不需要 torchrun。
多进程用 ``torch.multiprocessing.spawn``；Windows 上 spawn 会重新 import 测试模块，
所以每个 worker 函数都必须是模块顶层的普通函数（不能是 lambda 或闭包），
并且 spawn 只能在 ``if __name__ == "__main__"`` 保护下的路径里发生——
pytest 是通过 fixture 调用的，import 阶段不会触发 spawn，这个条件自动满足。
"""

from __future__ import annotations

import json
import os
import socket
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.dirname(TESTS_DIR)
SRC_DIR = os.path.join(LAB_DIR, "src")
CONFIG_DIR = os.path.join(LAB_DIR, "configs")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
# 让各测试模块（以及 spawn 出来的子进程）都能 `import conftest`
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

# CPU 上单测要快，限制线程数；必须在 import torch 之前设
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def find_minimind_root() -> Optional[str]:
    """按 MINIMIND_ROOT → 常见相对位置的顺序找 MiniMind 仓库。"""
    candidates: List[str] = []
    env = os.environ.get("MINIMIND_ROOT")
    if env:
        candidates.append(env)
    here = LAB_DIR
    for _ in range(6):
        here = os.path.dirname(here)
        candidates.append(os.path.join(here, "minimind"))
    for cand in candidates:
        if cand and os.path.isfile(os.path.join(cand, "model",
                                                "model_minimind.py")):
            return os.path.abspath(cand)
    return None


MINIMIND_ROOT = find_minimind_root()
if MINIMIND_ROOT:
    os.environ["MINIMIND_ROOT"] = MINIMIND_ROOT

SKIP_NO_MINIMIND = (
    "找不到 MiniMind 仓库。请先 git clone https://github.com/jingyaogong/minimind.git，"
    "checkout 7a6fddd63a30c06b2fdd5fac4089922b29bc841b，"
    "再设置环境变量 MINIMIND_ROOT 指向该目录。")


@pytest.fixture(scope="session")
def lab_dir() -> str:
    return LAB_DIR


@pytest.fixture(scope="session")
def config_dir() -> str:
    return CONFIG_DIR


@pytest.fixture(scope="session")
def tiny_config(config_dir: str) -> str:
    return os.path.join(config_dir, "tiny_cpu.json")


@pytest.fixture(scope="session")
def moe_config(config_dir: str) -> str:
    return os.path.join(config_dir, "moe_tiny_cpu.json")


@pytest.fixture(scope="session")
def minimind_root() -> str:
    if not MINIMIND_ROOT:
        pytest.skip(SKIP_NO_MINIMIND)
    return MINIMIND_ROOT


def free_port() -> int:
    """让内核分配一个空闲端口，避免多个测试用同一个 MASTER_PORT 撞车。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _spawn(fn: Callable, world_size: int, out_dir: str,
           extra: Sequence[Any] = (), timeout: int = 600) -> List[Dict[str, Any]]:
    """启动 ``world_size`` 个 gloo 进程跑 ``fn(rank, world_size, port, out_dir, *extra)``。

    每个 worker 把结果写成 ``out_dir/rank<r>.json``；父进程读回并按 rank 排序返回。
    worker 里抛异常时 spawn 会把子进程的 traceback 抬到父进程，测试直接失败。
    """
    import torch.multiprocessing as mp

    os.makedirs(out_dir, exist_ok=True)
    port = free_port()
    args = (world_size, port, out_dir) + tuple(extra)
    mp.spawn(fn, args=args, nprocs=world_size, join=True)
    results: List[Dict[str, Any]] = []
    for rank in range(world_size):
        path = os.path.join(out_dir, "rank" + str(rank) + ".json")
        if not os.path.isfile(path):
            raise AssertionError(
                "rank " + str(rank) + " 没有写出结果文件 " + path
                + "；它可能在写文件之前就退出了。")
        with open(path, "r", encoding="utf-8") as fh:
            results.append(json.load(fh))
    return results


@pytest.fixture(scope="session")
def spawn_dist() -> Callable:
    """返回 :func:`_spawn`；用法见各 test 文件顶部的 worker 函数。"""
    return _spawn


def worker_setup(rank: int, world_size: int, port: int,
                 backend: str = "gloo") -> None:
    """worker 进程里设置分布式环境变量（必须在 init_process_group 之前）。"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.setdefault("GLOO_SOCKET_IFNAME", os.environ.get(
        "GLOO_SOCKET_IFNAME", ""))
    if not os.environ["GLOO_SOCKET_IFNAME"]:
        del os.environ["GLOO_SOCKET_IFNAME"]
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    if MINIMIND_ROOT:
        os.environ.setdefault("MINIMIND_ROOT", MINIMIND_ROOT)


def write_result(out_dir: str, rank: int, payload: Dict[str, Any]) -> None:
    with open(os.path.join(out_dir, "rank" + str(rank) + ".json"), "w",
              encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
