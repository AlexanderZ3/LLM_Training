"""故障注入：kill 条件、真的会退出、NCCL 环境变量助手不覆盖公司设置。

``kill_rank`` 用的是 ``os._exit(1)``，测试它唯一诚实的方式是**真的开一个子进程
让它死一次**，然后看退出码。在测试进程里直接调用会把 pytest 一起带走。
"""

from __future__ import annotations

import multiprocessing as mp
import os

import pytest

from mm_dist import faults


# ---------------------------------------------------------------------------
# 触发条件
# ---------------------------------------------------------------------------


def test_kill_rank_does_nothing_for_other_ranks():
    assert faults.kill_rank(rank=1, after_step=3, current_rank=0,
                            current_step=99) is False


def test_kill_rank_does_nothing_before_the_step():
    assert faults.kill_rank(rank=0, after_step=5, current_rank=0,
                            current_step=4) is False


def test_kill_rank_disabled_with_negative_rank():
    assert faults.kill_rank(rank=-1, after_step=0, current_rank=0,
                            current_step=100) is False


def test_injector_not_armed_by_default():
    injector = faults.KillRankInjector(rank=-1, after_step=3, current_rank=0)
    assert injector.armed is False
    for step in range(1, 10):
        injector.maybe_kill(step)  # 不该有任何副作用
    assert injector.describe()["armed"] is False


def test_injector_describe_fields():
    injector = faults.KillRankInjector(rank=2, after_step=7, current_rank=2)
    assert injector.describe() == {"kill_rank": 2, "after_step": 7, "armed": True}


# ---------------------------------------------------------------------------
# 真的会死
# ---------------------------------------------------------------------------


def _suicide_target(after_step: int, log_path: str) -> None:
    """子进程入口：走到 after_step 时应当 os._exit(1)。"""
    import sys

    sys.path.insert(0, os.environ["MM_DIST_SRC"])
    from mm_dist import faults as F

    injector = F.KillRankInjector(rank=0, after_step=after_step, current_rank=0,
                                  log_path=log_path)
    for step in range(1, after_step + 3):
        injector.maybe_kill(step)
    # 走到这里说明该死没死
    os._exit(0)


def test_kill_rank_really_exits_with_code_one(tmp_path):
    """Windows 上 multiprocessing 用 spawn，target 必须是模块级函数。"""
    import conftest as H

    os.environ["MM_DIST_SRC"] = H.SRC_DIR
    log_path = str(tmp_path / "killed.jsonl")
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_suicide_target, args=(2, log_path))
    proc.start()
    proc.join(timeout=120)
    assert proc.exitcode == 1, (
        "kill_rank 应该让进程以退出码 1 结束，实际 exitcode=" + str(proc.exitcode)
        + "（0 表示循环跑完了都没死）")
    assert os.path.isfile(log_path), "被杀之前应当把 killed 事件追加进日志"
    with open(log_path, "r", encoding="utf-8") as fh:
        content = fh.read()
    assert '"event": "killed"' in content
    assert '"step": 2' in content


# ---------------------------------------------------------------------------
# NCCL 环境
# ---------------------------------------------------------------------------


def test_set_nccl_timeout_env_writes_both_names(monkeypatch):
    for name in faults.WATCHED_VARS:
        monkeypatch.delenv(name, raising=False)
    written = faults.set_nccl_timeout_env(async_error_handling=True)
    assert written["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "1"
    assert written["NCCL_ASYNC_ERROR_HANDLING"] == "1"
    assert written["TORCH_NCCL_BLOCKING_WAIT"] == "0"
    assert os.environ["NCCL_ASYNC_ERROR_HANDLING"] == "1"


def test_set_nccl_timeout_env_does_not_overwrite_company_settings(monkeypatch):
    """公司启动脚本已经设过的值，默认不动——这是安全边界的一部分。"""
    monkeypatch.setenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0")
    written = faults.set_nccl_timeout_env(async_error_handling=True)
    assert "TORCH_NCCL_ASYNC_ERROR_HANDLING" not in written
    assert os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "0"
    written = faults.set_nccl_timeout_env(async_error_handling=True, overwrite=True)
    assert written["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "1"
    assert os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "1"


def test_nccl_debug_only_set_when_requested(monkeypatch):
    monkeypatch.delenv("NCCL_DEBUG", raising=False)
    written = faults.set_nccl_timeout_env()
    assert "NCCL_DEBUG" not in written
    written = faults.set_nccl_timeout_env(debug="WARN")
    assert written["NCCL_DEBUG"] == "WARN"


def test_nccl_env_report_covers_watched_vars():
    report = faults.nccl_env_report()
    assert set(report) == set(faults.WATCHED_VARS)


def test_recommended_env_lines_mention_timeout_is_a_flag():
    lines = faults.recommended_env_lines(600)
    joined = "\n".join(lines)
    assert "TORCH_NCCL_ASYNC_ERROR_HANDLING=1" in joined
    assert "--nccl-timeout-s 600" in joined
    assert "NCCL_DEBUG" in joined


def test_env_subcommand_runs():
    args = faults.build_parser().parse_args(["env", "--nccl-timeout-s", "300"])
    assert faults.cmd_env(args) == 0


def test_demo_requires_config():
    with pytest.raises(SystemExit):
        faults.main(["demo"])
