"""一天的全部步骤，按门的顺序跑一遍。scripts/run_dayN_*.sh 只负责调这一个文件。

为什么编排写在 Python 里
------------------------
CLAUDE.md 3.2 那条教训：把多步骤流程写进 shell，会在最不该出问题的地方出问题
（PowerShell 5.1 把原生命令的 stderr 包成 ErrorRecord，一条无害提示终止了 27 GB
下载）。所以重试、顺序、失败停止、日志编码全部在这里，shell 只做一件事——
用对解释器调一次这个文件。

用法::

    # 第一次跑，先用 2 步试通
    MAX_STEPS=2 bash lab/scripts/run_day2_equiv_faults.sh

    # 直接调 Python 也一样
    python lab/scripts/run_day.py --day 2 --config tiny_cpu --force-cpu \\
        --nproc 2 --max-steps 4

    # 只看会跑哪几步
    python lab/scripts/run_day.py --day 3 --dry-run

失败即停
--------
默认在第一个失败的步骤停下（--keep-going 可以改）。理由见 CLAUDE.md 第 7 节：
前一门失败不得用扩大算力掩盖，也不该让后面的步骤在错误的前提上继续产出证据。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mm_v100 import cli  # noqa: E402

RUN_TAGS = {0: "day0_probe", 1: "day1_pretrain", 2: "day2_sft",
            3: "day3_dist", 4: "day4_profile", 5: "day5_rl"}


def load_script(name: str):
    """按文件名加载同目录下的另一个入口脚本，直接调它的 main()。

    用 importlib 而不是 subprocess：省掉每步 2-3 秒的 torch 导入，
    也让异常能原样冒到这里，而不是变成一个退出码。

    模块名必须是**真实可导入的名字**，而且要注册进 sys.modules
    ------------------------------------------------------------
    第一版用的是合成名 "mm_v100_script_" + 基名，且不进 sys.modules。
    这在单进程下没问题，一碰 `torch.multiprocessing.spawn` 就炸：

        _pickle.PicklingError: Can't pickle <function worker at 0x...>:
            import of module 'mm_v100_script_equiv_check' failed

    原因是 pickle 存函数时记的是「模块名 + 限定名」，子进程按那个名字
    `import` 一次来还原函数。合成名在任何地方都 import 不到，于是失败。
    受影响的是所有走 spawn 的入口：equiv_check / run_faults / reshard_check
    （wiring_smoke 用 subprocess，不受影响）。

    **这不是 Windows 特有的**：这些脚本显式指定 start_method="spawn"，
    在 V100 的 Linux 上同样命中。所以这里改成用脚本的真实基名，
    把脚本目录放进 sys.path，并在 exec 之前注册进 sys.modules ——
    multiprocessing 的 spawn 会把父进程的 sys.path 传给子进程，
    子进程于是能按同一个名字 import 到同一个文件。
    """
    path = os.path.join(_HERE, name)
    if not os.path.isfile(path):
        raise FileNotFoundError("找不到入口脚本 " + path)
    mod_name = name[:-3] if name.endswith(".py") else name
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    cached = sys.modules.get(mod_name)
    if cached is not None and getattr(cached, "__file__", None) == path:
        return cached
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    # 先注册再 exec：spawn 的子进程要按这个名字找回同一个模块。
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(mod_name, None)
        raise
    return module


def day0_steps(args: argparse.Namespace) -> List[Tuple[str, str, List[str]]]:
    out_dir = args.out_dir
    steps = [("环境探针", "probe_env.py", ["--out-dir", out_dir])]
    if os.environ.get("MINIMIND_ROOT", "").strip():
        steps.append(("MiniMind 兼容审计", "audit_compat.py",
                      ["--out-dir", out_dir, "--quiet"]))
    smoke = ["--out-dir", out_dir, "--config", args.config,
             "--sizes", args.sizes, "--steps", str(min(args.max_steps, 2)),
             "--global-batch", str(args.global_batch), "--dtype", args.dtype]
    # Day 0 这一步的全部意义就是回答「torchrun + NCCL 在这台机器上能不能起来」。
    # wiring_smoke 的默认 launcher 是 spawn——那条路径绕开了 rendezvous，
    # 在本机（没有 torchrun）是唯一能跑的选择，但在 V100 上用它等于没验。
    # 所以只要不是强制 CPU，就显式走 torchrun。
    if args.force_cpu:
        smoke.append("--force-cpu")
        smoke.extend(["--launcher", "spawn"])
    else:
        smoke.extend(["--launcher", args.launcher])
    steps.append(("wiring smoke", "wiring_smoke.py", smoke))
    return steps


def day1_steps(args: argparse.Namespace) -> List[Tuple[str, str, List[str]]]:
    out_dir = args.out_dir
    contract = ["--out-dir", out_dir, "--run-tag", RUN_TAGS[1]]
    if args.data_file:
        contract += ["--file", args.data_file]
    else:
        # 这一天的 run tag 是 day1_pretrain，数据契约也该验 pretrain 那条链
        # （字段 text、无 prompt 前缀、n_label = n_nonpad - 1）。
        # 之前默认 sft 会让「没给 --data-file 时」验的是另一条链路。
        contract += ["--toy", args.stage if args.stage else "pretrain"]
    numeric = ["--out-dir", out_dir, "--config", args.config,
               "--device", args.device, "--steps", str(args.max_steps),
               "--tol", "1e-3"]
    return [("数据契约", "check_data_contract.py", contract),
            ("误差账 fp32 vs fp16", "run_numeric_ref.py", numeric)]


def day2_steps(args: argparse.Namespace) -> List[Tuple[str, str, List[str]]]:
    out_dir = args.out_dir
    equiv = ["--out-dir", out_dir, "--config", args.config,
             "--global-batch", str(args.global_batch),
             "--a-nproc", "1", "--a-accum", str(max(args.nproc, 1)),
             "--b-nproc", str(args.nproc), "--b-accum", "1",
             "--max-steps", str(args.max_steps), "--backend", args.backend]
    faults = ["--out-dir", out_dir, "--config", args.config,
              "--nproc", str(args.nproc), "--max-steps", str(args.max_steps),
              "--global-batch", str(args.global_batch), "--dtype", args.dtype,
              "--backend", args.backend]
    if args.force_cpu:
        equiv.append("--force-cpu")
        faults.append("--force-cpu")
    return [("固定全局批等价", "equiv_check.py", equiv),
            ("三档故障判别", "run_faults.py", faults)]


def day3_steps(args: argparse.Namespace) -> List[Tuple[str, str, List[str]]]:
    out_dir = args.out_dir
    hand = ["--out-dir", out_dir, "--config", args.config, "--hand-only",
            "--world-size", str(args.nproc), "--mode", "full_shard",
            "--show-collectives"]
    measured = ["--out-dir", out_dir, "--config", args.config,
                "--device", args.device]
    reshard = ["--out-dir", out_dir, "--config", args.config,
               "--save-nproc", str(args.nproc), "--load-nproc",
               str(max(args.nproc // 2, 1)), "--max-steps", str(args.max_steps),
               "--global-batch", str(args.global_batch)]
    resume = ["--out-dir", out_dir, "--config", args.config,
              "--total-steps", str(max(args.max_steps, 2)),
              "--break-at", str(max(args.max_steps // 2, 1)),
              "--global-batch", str(args.global_batch)]
    if args.force_cpu:
        reshard.append("--force-cpu")
        resume.append("--force-cpu")
    return [("字节账（手算）", "byte_ledger.py", hand),
            ("字节账（实测）", "byte_ledger.py", measured),
            ("分片 checkpoint 重切", "reshard_check.py", reshard),
            ("resume 等价", "resume_check.py", resume)]


def day4_steps(args: argparse.Namespace) -> List[Tuple[str, str, List[str]]]:
    out_dir = args.out_dir
    before = ["--out-dir", out_dir, "--config", args.config, "--tag", "before",
              "--device", args.device]
    train = ["--config", args.config, "--out-dir", out_dir,
             "--stage", "sft", "--dtype", args.dtype,
             "--max-steps", str(args.max_steps),
             "--global-batch", str(args.global_batch),
             "--save-final", "1", "--log-name", "metrics_day4"]
    if args.force_cpu:
        train.append("--force-cpu")
    after = ["--out-dir", out_dir, "--config", args.config, "--tag", "after",
             "--device", args.device,
             "--checkpoint", os.path.join(out_dir, "ckpt", "final.pt")]
    return [("eval（训练前）", "eval_compare.py", before),
            ("有界 SFT", "train_bounded.py", train),
            ("eval（训练后）", "eval_compare.py", after)]


def day5_steps(args: argparse.Namespace) -> List[Tuple[str, str, List[str]]]:
    out_dir = args.out_dir
    rl = ["--out-dir", out_dir, "--config", args.config,
          "--device", args.device]
    if args.data_file:
        rl += ["--dpo-file", args.data_file]
    return [("RL 数值链", "rl_numeric_check.py", rl)]


PLANS: Dict[int, Callable[[argparse.Namespace],
                          List[Tuple[str, str, List[str]]]]] = {
    0: day0_steps, 1: day1_steps, 2: day2_steps,
    3: day3_steps, 4: day4_steps, 5: day5_steps,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="按门的顺序跑完某一天的全部步骤",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--day", type=int, default=0, choices=sorted(PLANS),
                   help="第几天")
    p.add_argument("--config", type=str, default="tiny_cpu",
                   help="configs/ 下的名字；V100 上用 smoke_v100 或 v100_768")
    p.add_argument("--device", type=str, default="cpu",
                   help="单卡步骤用哪个设备")
    p.add_argument("--dtype", type=str, default="float16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--nproc", type=int, default=2,
                   help="多进程步骤的进程数；V100 上用 8")
    p.add_argument("--sizes", type=str, default="1,2",
                   help="Day 0 wiring smoke 的 world_size 序列")
    p.add_argument("--max-steps", type=int,
                   default=int(os.environ.get("MAX_STEPS", "4")),
                   help="每个训练类步骤的步数上限；默认读环境变量 MAX_STEPS")
    p.add_argument("--global-batch", type=int, default=8)
    p.add_argument("--backend", type=str, default="gloo",
                   choices=["auto", "gloo", "nccl"])
    p.add_argument("--launcher", type=str, default="torchrun",
                   choices=["torchrun", "spawn"],
                   help="Day 0 wiring smoke 用哪个启动器。默认 torchrun，"
                        "因为那正是 V100 上要验的东西；--force-cpu 时自动降级为 spawn")
    p.add_argument("--stage", type=str, default=None,
                   choices=["pretrain", "sft"],
                   help="Day 1 数据契约验哪条链；默认按当天的 run tag 取 pretrain")
    p.add_argument("--force-cpu", action="store_true",
                   help="本机排练时加上；V100 上不要加")
    p.add_argument("--data-file", type=str, default=None,
                   help="Day 1/5 用的真数据文件名；不给就用内置 fixture")
    p.add_argument("--out-dir", type=str, default=None,
                   help="产物目录；默认 $MM_RUNS_ROOT/<当天的 run tag>")
    p.add_argument("--keep-going", action="store_true",
                   help="某一步失败也继续跑后面的步骤（默认失败即停）")
    p.add_argument("--dry-run", action="store_true",
                   help="只列出将要执行的步骤与参数，不真跑")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    cli.setup()
    args = build_parser().parse_args(argv)
    args.out_dir = args.out_dir or cli.resolve_out_dir(None, RUN_TAGS[args.day])
    steps = PLANS[args.day](args)

    print("== Day %d：%d 个步骤，产物写到 %s ==" % (args.day, len(steps),
                                                  args.out_dir))
    for i, (label, script, script_argv) in enumerate(steps, start=1):
        print("  %d) %-22s %s %s" % (i, label, script, " ".join(script_argv)))
    print("")
    if args.dry_run:
        print("[dry-run] 未执行任何步骤")
        return 0

    results: List[Dict[str, Any]] = []
    exit_code = 0
    for i, (label, script, script_argv) in enumerate(steps, start=1):
        print("")
        print("=" * 72)
        print("[%d/%d] %s  (%s)" % (i, len(steps), label, script), flush=True)
        print("=" * 72, flush=True)
        module = load_script(script)
        t0 = time.perf_counter()
        code = int(module.main(script_argv) or 0)
        dt = time.perf_counter() - t0
        results.append({"label": label, "script": script, "rc": code,
                        "wall_s": dt})
        print("[%d/%d] %s -> rc=%d (%.1fs)" % (i, len(steps), label, code, dt),
              flush=True)
        if code != 0:
            exit_code = code
            if not args.keep_going:
                print("")
                print("在第 %d 步失败，停止。后面的步骤会在错误的前提上产出证据，"
                      "所以不继续。要强行跑完加 --keep-going。" % i)
                break

    print("")
    print("== Day %d 小结 ==" % args.day)
    for r in results:
        print("  %-22s rc=%d  %.1fs" % (r["label"], r["rc"], r["wall_s"]))
    summary = {"schema": "mm_v100.run_day/1", "day": args.day,
               "config": args.config, "max_steps": args.max_steps,
               "steps": results, "exit_code": exit_code}
    print("[run_day] -> " + cli.write_json(
        summary, os.path.join(args.out_dir, "run_day.json")))
    print("下一步：python lab/scripts/make_evidence.py --day %d --skill <技能标签> "
          "--run-dir %s" % (args.day, args.out_dir))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
