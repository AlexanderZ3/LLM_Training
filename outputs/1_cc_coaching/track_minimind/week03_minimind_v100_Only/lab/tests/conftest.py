"""pytest 公共设施：sys.path、临时 run 根目录、CPU gloo 多进程启动器。

来源
----
从 week02 lab/tests/conftest.py 复制并裁剪：去掉了 MiniMind 定位与
SKIP_NO_MINIMIND（本周的模型自带，不需要外部仓库），新增 mm_runs_root fixture
把 MM_RUNS_ROOT 指到 tmp_path，保证测试永远不往仓库里写产物。

多进程说明
----------
本机没有 torchrun，所以多进程一律用 torch.multiprocessing.spawn。
Windows 上 spawn 会重新 import 测试模块，因此每个 worker 函数都必须是
**模块顶层的普通函数**（不能是 lambda、闭包或 fixture 内部定义的函数）。
"""

from __future__ import annotations

import json
import os
import socket
import sys
from typing import Any, Callable, Dict, List, Sequence

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.dirname(TESTS_DIR)
SRC_DIR = os.path.join(LAB_DIR, "src")
CONFIG_DIR = os.path.join(LAB_DIR, "configs")
SCRIPTS_DIR = os.path.join(LAB_DIR, "scripts")

for path in (SRC_DIR, TESTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

# CPU 上单测要快；必须在 import torch 之前设。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


@pytest.fixture(scope="session")
def lab_dir() -> str:
    return LAB_DIR


@pytest.fixture(scope="session")
def scripts_dir() -> str:
    return SCRIPTS_DIR


@pytest.fixture(scope="session")
def tiny_config_path() -> str:
    return os.path.join(CONFIG_DIR, "tiny_cpu.json")


@pytest.fixture
def runs_root(tmp_path, monkeypatch) -> str:
    """把 MM_RUNS_ROOT 指到 tmp_path。

    没有这一条，任何一个忘了传 --out-dir 的测试都会往周包的 runs/ 里写文件，
    而那个目录在 V100 上代表公司产物落地位置——测试污染它是不可接受的。
    """
    target = tmp_path / "runs"
    target.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MM_RUNS_ROOT", str(target))
    return str(target)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def spawn(fn: Callable, world_size: int, out_dir: str,
          extra: Sequence[Any] = ()) -> List[Dict[str, Any]]:
    """起 world_size 个 gloo 进程跑 fn(rank, world_size, port, out_dir, *extra)。

    每个 worker 把结果写成 out_dir/rank<r>.json；父进程读回并按 rank 排序返回。
    worker 抛异常时 spawn 会把子进程 traceback 抬到父进程，测试直接失败。
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
                "rank " + str(rank) + " 没写出结果文件 " + path
                + "；它可能在写文件之前就退出了。")
        with open(path, "r", encoding="utf-8") as fh:
            results.append(json.load(fh))
    return results


@pytest.fixture(scope="session")
def spawn_dist() -> Callable:
    return spawn


def worker_setup(rank: int, world_size: int, port: int) -> None:
    """worker 进程里的环境准备。必须在 init_process_group 之前调用。"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["OMP_NUM_THREADS"] = "1"
    for path in (SRC_DIR, TESTS_DIR):
        if path not in sys.path:
            sys.path.insert(0, path)


def write_result(out_dir: str, rank: int, payload: Dict[str, Any]) -> None:
    with open(os.path.join(out_dir, "rank" + str(rank) + ".json"), "w",
              encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
