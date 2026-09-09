"""Day 3 字节账：先手算，再实测，打出「手算列 + 实测列 + 偏差列」。

用法::

    # 只手算（闭卷之后用它对答案；不需要 GPU，也不建模型）
    python lab/scripts/byte_ledger.py --config v100_768 --hand-only \\
        --world-size 8 --mode full_shard --batch-per-rank 8 --seq-len 512

    # 手算 + 实测（本机用 tiny_cpu；V100 上用 v100_768 --device cuda:0）
    python lab/scripts/byte_ledger.py --config tiny_cpu --device cpu

    # 通信账的手算表
    python lab/scripts/byte_ledger.py --config v100_768 --hand-only \\
        --world-size 8 --mode full_shard --show-collectives

退出码：0 = param/grad/optim 三项手算与实测完全相等；1 = 不等（activation 是
估算项，不影响退出码）。

先手算再看实测
--------------
这个脚本故意把 --hand-only 放在前面：Day 3 的做法是先闭卷写出四项，
再用 --hand-only 对答案，最后才跑实测。反过来做就变成了抄。
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch  # noqa: E402

from mm_v100 import cli  # noqa: E402
from mm_v100 import collectives as CO  # noqa: E402
from mm_v100 import common as C  # noqa: E402
from mm_v100 import ledger as LG  # noqa: E402
from mm_v100 import model as M  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="每卡字节账：param / grad / optim / activation 手算 vs 实测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", type=str, default="tiny_cpu")
    p.add_argument("--mode", type=str, default="ddp",
                   choices=["ddp", "no_shard", "shard_grad_op", "full_shard"])
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--batch-per-rank", type=int, default=None,
                   help="默认取 global_batch / (world_size*accum)")
    p.add_argument("--seq-len", type=int, default=None,
                   help="默认取配置里的 train.seq_len")
    p.add_argument("--n-units", type=int, default=None,
                   help="FSDP 单元数；默认 layers+1")
    p.add_argument("--param-dtype", type=str, default="float32",
                   choices=["float32", "float16"],
                   help="参数常驻精度。AMP 下参数留 fp32，所以默认 float32")
    p.add_argument("--compute-dtype", type=str, default="float16",
                   choices=["float32", "float16"],
                   help="autocast 里中间张量的精度")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--hand-only", action="store_true",
                   help="只打手算表，不建模型不实测")
    p.add_argument("--show-collectives", action="store_true",
                   help="附带打印通信账的手算表")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--run-tag", type=str, default="day3_dist")
    p.add_argument("--dry-run", action="store_true",
                   help="只打印将要使用的形状参数，不算也不测")
    return p


def main(argv=None) -> int:
    cli.setup()
    args = build_parser().parse_args(argv)
    cfg = C.load_config(args.config)
    train_cfg = cfg["train"]
    accum = int(train_cfg.get("accum", 1))
    seq_len = int(args.seq_len or train_cfg["seq_len"])
    batch = args.batch_per_rank
    if batch is None:
        denom = max(int(args.world_size), 1) * accum
        batch = max(int(train_cfg["global_batch"]) // denom, 1)

    if args.dry_run:
        print("[dry-run] config=%s mode=%s world_size=%d batch/rank=%d seq=%d"
              % (cfg["name"], args.mode, args.world_size, batch, seq_len))
        print("[dry-run] model=" + str(cfg["model"]))
        return 0

    hand = LG.hand_ledger(cfg["model"], world_size=args.world_size,
                          mode=args.mode, batch_per_rank=batch,
                          seq_len=seq_len, n_units=args.n_units,
                          param_dtype=args.param_dtype,
                          compute_dtype=args.compute_dtype)
    print("== 手算（每卡）==")
    print("P = %s 参数" % format(hand["params_total"], ","))
    for key in ("param_bytes", "grad_bytes", "optim_bytes", "activation_bytes"):
        print("  %-18s %18s B  (%.3f GiB)"
              % (key, format(hand[key], ","), hand[key] / (1024 ** 3)))
    print("  %-18s %18s B" % ("all_gather_buffer",
                              format(hand["all_gather_buffer_bytes"], ",")))
    print("  %-18s %18s B  (%.3f GiB)"
          % ("total", format(hand["total_bytes"], ","), hand["total_gib"]))
    print("")
    for key, formula in hand["formulas"].items():
        print("  %-12s = %s" % (key, formula))

    if args.show_collectives:
        print("")
        n_units = hand["n_units"]
        if args.mode in ("ddp", "no_shard"):
            exp = CO.ddp_expected(hand["params_total"],
                                  LG.dtype_bytes(args.param_dtype))
        else:
            exp = CO.fsdp_expected(hand["params_total"], n_units,
                                   args.world_size,
                                   LG.dtype_bytes(args.compute_dtype),
                                   LG.dtype_bytes(args.compute_dtype),
                                   sharding=args.mode)
        print("== 通信账（手算，每卡每 step）==")
        print(CO.to_json(exp))

    out_dir = cli.resolve_out_dir(args.out_dir, args.run_tag)
    if args.hand_only:
        path = cli.write_json(hand, os.path.join(out_dir, "byte_ledger_hand.json"))
        print("")
        print("[ledger] 手算 -> " + path)
        return 0

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] 请求 " + device + " 但本机没有 CUDA，改用 cpu")
        device = "cpu"
    model, model_cfg = M.build_model(cfg["model"], device=device, seed=42)
    dataset = C.TinyDataset(num_samples=max(batch * 4, 16), seq_len=seq_len,
                            vocab_size=int(model_cfg.vocab_size),
                            seed=int(train_cfg.get("data_seed", 1234)))
    input_ids, labels = C.collate_indices(dataset, list(range(batch)), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    measured = LG.measure(model, input_ids, labels, optimizer,
                          dtype_name=("float32" if args.param_dtype == "float32"
                                      and args.compute_dtype == "float32"
                                      else args.compute_dtype),
                          device=device)
    # 实测是单进程的整份账；手算表若指定了 world_size>1，两者本来就不该相等。
    hand_single = LG.hand_ledger(cfg["model"], world_size=1, mode="ddp",
                                 batch_per_rank=batch, seq_len=seq_len,
                                 n_units=args.n_units,
                                 param_dtype=args.param_dtype,
                                 compute_dtype=args.compute_dtype)
    markdown = LG.render_markdown(hand_single, measured)
    print("")
    print(markdown)
    md_path = cli.write_text(markdown, os.path.join(out_dir, "byte_ledger.md"))
    js_path = cli.write_json({"hand_requested": hand,
                              "hand_single_rank": hand_single,
                              "measured": measured,
                              "compare": LG.compare(hand_single, measured)},
                             os.path.join(out_dir, "byte_ledger.json"))
    print("")
    print("[ledger] -> " + md_path)
    print("[ledger] -> " + js_path)
    cmp = LG.compare(hand_single, measured)
    if not cmp["exact_items_match"]:
        print("[ledger] param/grad/optim 手算与实测不一致，先查参数是否 tied、"
              "以及优化器是不是在 step 之后才测的。")
    return 0 if cmp["exact_items_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
