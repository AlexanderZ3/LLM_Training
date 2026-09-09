"""Day 1-5 的训练入口：转发给 mm_v100.bounded_train，并提供一个本机可用的启动器。

两种启动方式
------------
--launcher passthrough（默认）
    直接在当前进程里跑。多卡时由外层 torchrun 提供 RANK/WORLD_SIZE::

        torchrun --nproc_per_node 8 lab/scripts/train_bounded.py \\
            --config v100_768 --placement fsdp --dtype float16 --max-steps 20

--launcher spawn --nproc N
    本脚本自己用 torch.multiprocessing.spawn 起 N 个进程（gloo/CPU）。
    存在的理由：cc 的开发机上 torchrun 不可用，而「2 进程等价」这条门必须
    在本机真跑过，不能等到 V100 上才第一次运行::

        python lab/scripts/train_bounded.py --launcher spawn --nproc 2 \\
            --config tiny_cpu --placement ddp --force-cpu --backend gloo \\
            --dtype float32 --equiv-check --max-steps 3

为什么逻辑不写在 shell 里
------------------------
重试、端口分配、子进程回收、日志编码这些都在 Python 里做（CLAUDE.md 3.2）。
scripts/run_day*.sh 只负责用对解释器调一次这个文件。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mm_v100 import bounded_train  # noqa: E402
from mm_v100 import cli  # noqa: E402
from mm_v100 import common as C  # noqa: E402


def spawn_worker(rank: int, world_size: int, port: int,
                 argv: List[str]) -> None:
    """spawn 出来的每个进程执行的函数。必须是模块顶层函数才能被 pickle。"""
    C.set_worker_env(rank, world_size, port)
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    from mm_v100 import bounded_train as BT

    BT.main(list(argv))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="有界训练入口（passthrough / 本机 spawn 两种启动方式）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False)
    p.add_argument("--launcher", type=str, default="passthrough",
                   choices=["passthrough", "spawn"],
                   help="passthrough=当前进程（多卡交给 torchrun）；"
                        "spawn=本脚本自己起 --nproc 个 CPU 进程")
    p.add_argument("--nproc", type=int, default=1,
                   help="--launcher spawn 时的进程数")
    p.add_argument("--summary-json", type=str, default=None,
                   help="把 rank0 的 summary 另存一份（passthrough 模式有效）")
    p.add_argument("-h", "--help", action="store_true", dest="show_help",
                   help="打印本脚本与底层训练脚本的全部参数")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    cli.setup()
    raw = list(sys.argv[1:] if argv is None else argv)
    args, rest = build_parser().parse_known_args(raw)
    if args.show_help:
        build_parser().print_help()
        print("")
        print("=== 底层训练参数（mm_v100.bounded_train）===")
        bounded_train.build_parser().print_help()
        return 0
    if args.launcher == "spawn":
        if args.nproc < 1:
            raise SystemExit("--nproc 必须 >= 1")
        import torch.multiprocessing as mp

        port = C.free_port()
        print("[launcher] spawn nproc=%d port=%d" % (args.nproc, port),
              flush=True)
        mp.spawn(spawn_worker, args=(args.nproc, port, rest),
                 nprocs=args.nproc, join=True)
        print("[launcher] 全部 %d 个进程正常结束" % args.nproc, flush=True)
        return 0
    parsed = bounded_train.build_parser().parse_args(rest)
    summary = bounded_train.train(parsed)
    if C.is_main_process():
        text = json.dumps({k: v for k, v in summary.items() if k != "losses"},
                          ensure_ascii=False, indent=2, sort_keys=True)
        print(text, flush=True)
        if args.summary_json:
            print("[train] summary -> "
                  + cli.write_json(summary, args.summary_json), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
