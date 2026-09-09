"""Day 3 分片 checkpoint：N 卡写 -> M 卡读，step 换算，loss 连续性判定。

用法::

    # 本机：2 进程写分片，1 进程读回来验 loss 连续性（真跑，不是 smoke）
    python lab/scripts/reshard_check.py --config tiny_cpu --force-cpu \\
        --save-nproc 2 --load-nproc 1 --max-steps 3

    # V100：8 卡写，4 卡读
    python lab/scripts/reshard_check.py --config v100_768 --save-nproc 8 \\
        --load-nproc 4 --max-steps 20 --format flat

    # V100 上换成 FSDP + torch.distributed.checkpoint 的生产路径
    #（本机从未执行过，先把 --format flat 跑通再换）
    torchrun --nproc_per_node 8 lab/scripts/reshard_check.py --format fsdp ...

退出码：0 = loss 连续性通过；1 = 不通过。

判据
----
「恢复后 loss 没有跳变」必须固定三件事才可比：用哪一批数据、什么精度、怎么归约。
这里用的是第 saved_step+1 步那一批全局 batch 的 forward-only loss，
数据由 micro_batch_indices 从 step 算出来，与 world_size 无关。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch  # noqa: E402

from mm_v100 import ckpt_reshard as CK  # noqa: E402
from mm_v100 import cli  # noqa: E402
from mm_v100 import common as C  # noqa: E402
from mm_v100 import model as M  # noqa: E402


def make_model_and_data(cfg: Dict[str, Any], device: str, seed: int,
                        max_steps: int, global_batch: int):
    model, model_cfg = M.build_model(cfg["model"], device=device, seed=seed)
    train_cfg = cfg["train"]
    dataset = C.TinyDataset(
        num_samples=max(int(train_cfg.get("num_samples", 512)),
                        (max_steps + 2) * global_batch),
        seq_len=int(train_cfg["seq_len"]),
        vocab_size=int(model_cfg.vocab_size),
        seed=int(train_cfg.get("data_seed", 1234)))
    return model, dataset


def save_worker(rank: int, world_size: int, port: int, payload: str) -> None:
    """写端：每个 rank 训练相同的步数（数据切分保证结果一致），写自己那一片。"""
    C.set_worker_env(rank, world_size, port)
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    from mm_v100 import ckpt_reshard as K
    from mm_v100 import common as CC

    opts = json.loads(payload)
    info = CC.init_dist(backend="gloo", force_cpu=True)
    CC.set_seed(opts["seed"], per_rank=False)
    cfg = CC.load_config(opts["config"])
    model, dataset = make_model_and_data(cfg, "cpu", opts["seed"],
                                         opts["max_steps"],
                                         opts["global_batch"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts["lr"])
    accum = int(opts["accum"])
    for step in range(1, opts["max_steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        for acc_idx in range(accum):
            idx = CC.micro_batch_indices(step, opts["global_batch"],
                                         info.world_size, accum, info.rank,
                                         acc_idx)
            input_ids, labels = CC.collate_indices(dataset, idx, "cpu")
            out = model(input_ids, labels=labels)
            (out.loss / accum).backward()
        # 梯度必须先跨 rank 平均，否则各 rank 的参数会分叉，
        # 写出来的分片拼回去就不是同一个模型了。
        if info.world_size > 1:
            for p in model.parameters():
                if p.grad is not None:
                    torch.distributed.all_reduce(p.grad,
                                                 op=torch.distributed.ReduceOp.SUM)
                    p.grad /= info.world_size
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    next_loss = K.global_forward_loss(model, dataset, opts["max_steps"] + 1,
                                      opts["global_batch"], accum,
                                      info.world_size, info.rank, "cpu")
    K.save_flat_shard(opts["ckpt_dir"], model, info.rank, info.world_size,
                      meta_extra={"step": opts["max_steps"],
                                  "next_step": opts["max_steps"] + 1,
                                  "next_step_loss": next_loss,
                                  "global_batch": opts["global_batch"],
                                  "accum": accum,
                                  "config": opts["config"],
                                  "seed": opts["seed"]})
    if info.is_main:
        print("[save] world_size=%d step=%d next_step_loss=%.6f -> %s"
              % (info.world_size, opts["max_steps"], next_loss,
                 opts["ckpt_dir"]), flush=True)
    CC.cleanup_dist(info)


def load_and_verify(opts: Dict[str, Any], load_nproc: int) -> Dict[str, Any]:
    """读端（单进程即可：flat 格式每个进程都读整份再自己切）。"""
    cfg = C.load_config(opts["config"])
    meta = CK.read_meta(opts["ckpt_dir"])
    C.set_seed(opts["seed"], per_rank=False)
    model, dataset = make_model_and_data(cfg, "cpu", opts["seed"] + 999,
                                         opts["max_steps"],
                                         opts["global_batch"])
    CK.load_flat_into(model, opts["ckpt_dir"])
    actual = CK.global_forward_loss(model, dataset, int(meta["next_step"]),
                                    int(meta["global_batch"]),
                                    int(meta["accum"]), 1, 0, "cpu")
    verdict = CK.verify_continuity(float(meta["next_step_loss"]), actual,
                                   tol=opts["tol"])
    step_info = CK.convert_step(int(meta["step"]), int(meta["world_size"]),
                                load_nproc)
    slice_shapes = [CK.reshard_slice(opts["ckpt_dir"], load_nproc, r)[0].numel()
                    for r in range(load_nproc)]
    verdict.update({
        "step_conversion": step_info,
        "saved_world_size": int(meta["world_size"]),
        "load_world_size": load_nproc,
        "reshard_slice_numels": slice_shapes,
        "total_numel": int(meta["total_numel"]),
        "slices_sum_equals_total": sum(slice_shapes) == int(meta["total_numel"]),
    })
    return verdict


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="分片 checkpoint 的写/读/换 world_size 恢复与 loss 连续性",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", type=str, default="tiny_cpu")
    p.add_argument("--format", type=str, default="flat",
                   choices=["flat", "fsdp"],
                   help="flat=本包自带的分片格式（CPU 可跑）；"
                        "fsdp=FSDP+torch.distributed.checkpoint（只能在 CUDA 上）")
    p.add_argument("--save-nproc", type=int, default=2, help="写端进程数")
    p.add_argument("--load-nproc", type=int, default=1, help="读端进程数")
    p.add_argument("--max-steps", type=int, default=3)
    p.add_argument("--global-batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--ckpt-dir", type=str, default=None,
                   help="默认 <out-dir>/ckpt/reshard")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--run-tag", type=str, default="day3_dist")
    p.add_argument("--dry-run", action="store_true",
                   help="只打印分片边界，不训练也不写文件")
    return p


def main(argv=None) -> int:
    cli.setup()
    args = build_parser().parse_args(argv)
    out_dir = cli.resolve_out_dir(args.out_dir, args.run_tag)
    ckpt_dir = args.ckpt_dir or os.path.join(out_dir, "ckpt", "reshard")

    if args.format == "fsdp":
        print(CK.FSDP_ONLY_NOTE)
        print("")
        print("本脚本的 --format fsdp 只做前置检查，不代跑 torchrun。")
        print("在 V100 上请用：")
        print("  torchrun --nproc_per_node %d lab/scripts/train_bounded.py "
              "--config %s --placement fsdp --dtype float16 --max-steps %d "
              "--save-final 1" % (args.save_nproc, args.config, args.max_steps))
        print("再用 mm_v100.ckpt_reshard.fsdp_sharded_save/load 两个函数完成"
              "写读（见 02_LAB_GUIDE 第 3 节）。")
        return 0

    total_hint = None
    if args.dry_run:
        cfg = C.load_config(args.config)
        model, _ = M.build_model(cfg["model"], device="cpu", seed=args.seed)
        flat, _index = CK.flatten_state(model)
        total_hint = int(flat.numel())
        print("[dry-run] total_numel=%d" % total_hint)
        for ws in (args.save_nproc, args.load_nproc):
            bounds = [CK.shard_bounds(total_hint, ws, r) for r in range(ws)]
            print("  world_size=%d 分片边界 %s" % (ws, bounds))
        print("[dry-run] ckpt_dir=" + ckpt_dir)
        return 0

    opts = {
        "config": args.config, "seed": args.seed, "lr": args.lr,
        "max_steps": args.max_steps, "global_batch": args.global_batch,
        "accum": args.accum, "ckpt_dir": ckpt_dir, "tol": args.tol,
    }
    payload = json.dumps(opts, ensure_ascii=False)
    if args.save_nproc > 1:
        import torch.multiprocessing as mp

        port = C.free_port()
        mp.spawn(save_worker, args=(args.save_nproc, port, payload),
                 nprocs=args.save_nproc, join=True)
    else:
        C.set_worker_env(0, 1, C.free_port())
        save_worker(0, 1, int(os.environ["MASTER_PORT"]), payload)

    verdict = load_and_verify(opts, args.load_nproc)
    print("")
    print("== Day 3 reshard ==")
    print("saved_ws=%d -> load_ws=%d  total_numel=%d"
          % (verdict["saved_world_size"], verdict["load_world_size"],
             verdict["total_numel"]))
    print("分片长度（新 world_size）= %s  合计等于总长=%s"
          % (verdict["reshard_slice_numels"],
             verdict["slices_sum_equals_total"]))
    sc = verdict["step_conversion"]
    print("step 换算：%d (ws=%d) -> %d (ws=%d)"
          % (sc["saved_step"], sc["saved_world_size"], sc["converted_step"],
             sc["current_world_size"]))
    print("  前提：" + sc["assumption"])
    print("  数据侧真相：" + sc["data_side_truth"])
    print("loss 连续性：expected=%.6f actual=%.6f |delta|=%.3e tol=%.1e -> %s"
          % (verdict["expected_loss"], verdict["actual_loss"],
             verdict["abs_delta"], verdict["tol"],
             "PASS" if verdict["pass"] else "FAIL"))
    if not verdict["pass"]:
        print("  " + verdict["hint"])
    path = cli.write_json(verdict, os.path.join(out_dir, "reshard_check.json"))
    print("[reshard] -> " + path)
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
