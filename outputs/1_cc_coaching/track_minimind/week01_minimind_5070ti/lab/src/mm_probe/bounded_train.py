"""有界训练循环：用 MiniMind 的模型类与 dataset 类，自写最小训练循环并挂满仪表。

与 MiniMind 原脚本的关系：
- 模型：`model.model_minimind.MiniMindForCausalLM`（原类，未改）；
- 数据：`dataset.lm_dataset.PretrainDataset / SFTDataset / DPODataset`（原类，未改）；
- loss：pretrain/sft 用模型内置 `res.loss + res.aux_loss`；dpo 用 `dpo_check.dpo_loss_minimind`（镜像 train_dpo.py）；
- lr：MiniMind `get_lr` 的同一公式 lr*(0.1+0.45*(1+cos(pi*step/total)))，total=max_steps；
- 原脚本没有步数上限参数，所以"有界训练"只能在这里做；原脚本只做整轮。

新增的仪表/能力：
- `--max-steps`、`--overfit-n`（固定前 N 条样本反复训练）、`--dtype float32|float16|bfloat16`（float16 带 GradScaler）；
- `--log-jsonl` 每步 step/loss/lr/grad_norm/scaler_scale/tokens_per_s/peak_mem_mb；
- `--save-every` + `--resume`：保存 model/optimizer/scheduler/scaler/RNG/step/数据游标，恢复后下一步 loss 与连续训练相等；
- `--mask-fault`（Day 3 故障注入）、`--export-pth`（导出 MiniMind 格式 half state_dict 供原脚本 `--from_weight` 使用）。

CPU 上 `--config tiny_cpu --stage sft --max-steps 3` 必须能跑（见 tests/test_bounded_train.py）。
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # MiniMind 的 Windows pyarrow/torch DLL 冲突规避：先 import datasets 再 import torch
    import datasets  # noqa: F401
except ImportError:
    datasets = None
import numpy as np
import torch

from . import fixtures
from .hooks import ThroughputMeter, grad_global_norm, peak_memory_mb
from .mask_fault import FAULT_MODES, apply_fault
from .minimind_env import (add_minimind_to_path, build_model, find_minimind_root, load_config,
                           load_state_dict_any, load_tokenizer, resolve_data_path, utf8_stdout)

STAGES = ("pretrain", "sft", "dpo")
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def minimind_lr(current_step: int, total_steps: int, lr: float) -> float:
    """逐字复现 trainer_utils.get_lr。"""
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / max(total_steps, 1))))


def make_order(n: int, epoch: int, seed: int, overfit_n: int) -> List[int]:
    if overfit_n > 0:
        return list(range(min(overfit_n, n)))
    g = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(n, generator=g).tolist()


class DataCursor:
    """确定性的数据游标：(epoch, pos) 完全决定下一 batch 的样本下标，可保存/恢复。"""

    def __init__(self, n: int, batch_size: int, seed: int, overfit_n: int, epoch: int = 0, pos: int = 0):
        self.n, self.batch_size, self.seed, self.overfit_n = n, batch_size, seed, overfit_n
        self.epoch, self.pos = epoch, pos
        self.order = make_order(n, epoch, seed, overfit_n)

    def next_indices(self) -> List[int]:
        L = len(self.order)
        idxs = [self.order[(self.pos + i) % L] for i in range(self.batch_size)]
        self.pos += self.batch_size
        if self.pos >= L:
            self.pos = self.pos % L
            self.epoch += 1
            self.order = make_order(self.n, self.epoch, self.seed, self.overfit_n)
        return idxs

    def state(self) -> Dict[str, int]:
        return {"epoch": self.epoch, "pos": self.pos, "n": self.n, "batch_size": self.batch_size,
                "seed": self.seed, "overfit_n": self.overfit_n}


def rng_state() -> Dict[str, Any]:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: Dict[str, Any]) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_scaler(device_type: str, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device_type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)  # torch 2.1 兼容路径


def build_dataset(stage: str, data_path: str, tokenizer, max_seq_len: int, save_dir: Path):
    from dataset.lm_dataset import DPODataset, PretrainDataset, SFTDataset  # type: ignore

    if data_path == "fixture":
        data_path = str(fixtures.write_fixture_jsonl(stage, save_dir / f"fixture_{stage}.jsonl"))
    if stage == "pretrain":
        return PretrainDataset(data_path, tokenizer, max_length=max_seq_len), data_path
    if stage == "sft":
        return SFTDataset(data_path, tokenizer, max_length=max_seq_len), data_path
    return DPODataset(data_path, tokenizer, max_length=max_seq_len), data_path


def collate(stage: str, items: List[Any]) -> Dict[str, torch.Tensor]:
    if stage in ("pretrain", "sft"):
        return {"input_ids": torch.stack([x[0] for x in items]), "labels": torch.stack([x[1] for x in items])}
    keys = ["x_chosen", "y_chosen", "mask_chosen", "x_rejected", "y_rejected", "mask_rejected"]
    return {k: torch.stack([it[k] for it in items]) for k in keys}


def effective_train_cfg(cfg: Dict[str, Any], stage: str, args: argparse.Namespace) -> Dict[str, Any]:
    t = dict(cfg.get("train", {}))
    t.update(cfg.get("stages", {}).get(stage, {}))
    for cli_key, cfg_key in [("max_seq_len", "max_seq_len"), ("batch_size", "batch_size"), ("lr", "learning_rate"),
                             ("dtype", "dtype"), ("device", "device"), ("accumulation_steps", "accumulation_steps"),
                             ("grad_clip", "grad_clip")]:
        v = getattr(args, cli_key, None)
        if v is not None:
            t[cfg_key] = v
    t.setdefault("max_seq_len", 64)
    t.setdefault("batch_size", 2)
    t.setdefault("learning_rate", 5e-4)
    t.setdefault("dtype", "float32")
    t.setdefault("device", "cpu")
    t.setdefault("accumulation_steps", 1)
    t.setdefault("grad_clip", 1.0)
    t.setdefault("beta", 0.15)
    return t


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MiniMind 有界训练（自写循环 + 仪表）")
    p.add_argument("--config", default="tiny_cpu", help="lab/configs/*.json 名或路径")
    p.add_argument("--stage", choices=STAGES, default="sft")
    p.add_argument("--minimind-root", default=None)
    p.add_argument("--data-path", default=None, help="覆盖配置中的数据路径；`fixture` 用内置样本")
    p.add_argument("--max-steps", type=int, default=3, help="总 micro-step 数（含 resume 前已完成的）")
    p.add_argument("--overfit-n", type=int, default=0, help=">0 时固定前 N 条样本反复训练")
    p.add_argument("--dtype", choices=list(DTYPES), default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--accumulation-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-jsonl", default=None, help="每步一行 JSON；缺省 <save-dir>/log_<stage>.jsonl")
    p.add_argument("--log-interval", type=int, default=1, help="stdout 打印间隔（JSONL 仍每步写）")
    p.add_argument("--save-dir", default=None, help="缺省 lab/runs/<config>_<stage>")
    p.add_argument("--save-every", type=int, default=0, help=">0 时每 N 步保存 ckpt_stepN.pt 与 latest.pt")
    p.add_argument("--resume", default=None, help="从 bounded_train 的 .pt 恢复")
    p.add_argument("--from-weight", default=None, help="初始权重：MiniMind .pth 或 bounded_train .pt")
    p.add_argument("--mask-fault", choices=FAULT_MODES, default="none", help="仅 sft：label mask 故障注入")
    p.add_argument("--export-pth", default=None, help="结束时导出 MiniMind 格式 half state_dict（供原脚本 --from_weight）")
    p.add_argument("--quiet", action="store_true")
    return p


def run(args: argparse.Namespace) -> Dict[str, Any]:
    root = find_minimind_root(args.minimind_root)
    add_minimind_to_path(root)
    cfg = load_config(args.config)
    stage = args.stage
    tcfg = effective_train_cfg(cfg, stage, args)
    device = str(tcfg["device"])
    if "cuda" in device and not torch.cuda.is_available():
        print(f"[bounded] 配置 device={device} 但本机无 CUDA，改用 cpu", file=sys.stderr)
        device = "cpu"
    device_type = "cuda" if "cuda" in device else "cpu"
    dtype_name = str(tcfg["dtype"])
    amp_dtype = DTYPES[dtype_name]
    use_amp = dtype_name != "float32"
    save_dir = Path(args.save_dir) if args.save_dir else Path(__file__).resolve().parents[2] / "runs" / f"{cfg['name']}_{stage}"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_jsonl) if args.log_jsonl else save_dir / f"log_{stage}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    seed_all(args.seed)
    tokenizer = load_tokenizer(root)
    model, lm_config = build_model(cfg["model"], device)
    if args.from_weight:
        model.load_state_dict(load_state_dict_any(args.from_weight), strict=False)
    ref_model = None
    if stage == "dpo":
        ref_model = copy.deepcopy(model).eval().requires_grad_(False)

    data_spec = args.data_path or cfg.get("data", {}).get(stage, "fixture")
    data_path = resolve_data_path(data_spec, root)
    ds, data_path = build_dataset(stage, data_path, tokenizer, int(tcfg["max_seq_len"]), save_dir)
    batch_size = int(tcfg["batch_size"])
    accum = int(tcfg["accumulation_steps"])
    base_lr = float(tcfg["learning_rate"])
    grad_clip = float(tcfg["grad_clip"])
    beta = float(tcfg["beta"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    scaler = make_scaler(device_type, enabled=(dtype_name == "float16"))
    scheduler_state = {"kind": "minimind_cosine", "total_steps": int(args.max_steps), "base_lr": base_lr, "last_step": 0}
    cursor = DataCursor(len(ds), batch_size, args.seed, args.overfit_n)
    start_step = 0

    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ck.get("stage") != stage:
            raise ValueError(f"checkpoint stage={ck.get('stage')} 与 --stage {stage} 不一致")
        model.load_state_dict(ck["model"])
        if ref_model is not None and ck.get("ref_model") is not None:
            ref_model.load_state_dict(ck["ref_model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        scheduler_state = dict(ck["scheduler"])
        scheduler_state["total_steps"] = int(args.max_steps)
        cs = ck["cursor"]
        cursor = DataCursor(len(ds), batch_size, cs["seed"], cs["overfit_n"], epoch=cs["epoch"], pos=cs["pos"])
        start_step = int(ck["step"])
        set_rng_state(ck["rng"])
        if not args.quiet:
            print(f"[bounded] resumed from {args.resume}: step={start_step} cursor={cursor.state()}")

    pad_id = tokenizer.pad_token_id
    meter = ThroughputMeter(device)
    model.train()
    rows: List[Dict[str, Any]] = []
    t_run0 = time.perf_counter()
    if torch.cuda.is_available() and device_type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    autocast_ctx = torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp)

    def save_ckpt(step: int) -> Path:
        payload = {
            "model": model.state_dict(),
            "ref_model": ref_model.state_dict() if ref_model is not None else None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler_state,
            "scaler": scaler.state_dict(),
            "rng": rng_state(),
            "step": step,
            "cursor": cursor.state(),
            "stage": stage,
            "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
            "train_cfg": tcfg,
            "data_path": data_path,
            "mask_fault": args.mask_fault,
        }
        path = save_dir / f"ckpt_step{step}.pt"
        torch.save(payload, path)
        torch.save(payload, save_dir / "latest.pt")
        return path

    log_f = open(log_path, "a" if args.resume else "w", encoding="utf-8")
    try:
        for step in range(start_step + 1, int(args.max_steps) + 1):
            lr = minimind_lr(step, int(args.max_steps), base_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr
            idxs = cursor.next_indices()
            batch = collate(stage, [ds[i] for i in idxs])
            batch = {k: v.to(device) for k, v in batch.items()}
            meter.start()
            extra: Dict[str, Any] = {}
            with autocast_ctx:
                if stage in ("pretrain", "sft"):
                    labels = batch["labels"]
                    if stage == "sft" and args.mask_fault != "none":
                        labels = apply_fault(labels, args.mask_fault, input_ids=batch["input_ids"], pad_token_id=pad_id)
                    res = model(batch["input_ids"], labels=labels)
                    loss = res.loss + res.aux_loss
                    n_tokens = int((batch["input_ids"] != pad_id).sum().item())
                    extra["n_label_tokens"] = int((labels != -100).sum().item())
                else:
                    from .dpo_check import dpo_loss, per_token_logps

                    x = torch.cat([batch["x_chosen"], batch["x_rejected"]])
                    y = torch.cat([batch["y_chosen"], batch["y_rejected"]])
                    mask = torch.cat([batch["mask_chosen"], batch["mask_rejected"]]).float()
                    with torch.no_grad():
                        ref_lp = (per_token_logps(ref_model(x).logits, y) * mask).sum(1)
                    res = model(x)
                    pol_lp = (per_token_logps(res.logits, y) * mask).sum(1)
                    b = x.shape[0] // 2
                    d = dpo_loss(pol_lp[:b], pol_lp[b:], ref_lp[:b], ref_lp[b:], beta)
                    loss = d["loss"] + res.aux_loss
                    n_tokens = int((x != pad_id).sum().item())
                    extra["reward_margin"] = float(d["reward_margin"].mean().item())
                    extra["dpo_accuracy"] = float(d["accuracy"].item())
                    extra["n_label_tokens"] = int(mask.sum().item())
                loss_scaled = loss / accum
            scaler.scale(loss_scaled).backward()
            grad_norm = None
            did_opt_step = False
            if step % accum == 0 or step == int(args.max_steps):
                scaler.unscale_(optimizer)
                grad_norm = grad_global_norm(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                did_opt_step = True
            scheduler_state["last_step"] = step
            tps = meter.stop(n_tokens)
            row = {
                "step": step,
                "stage": stage,
                "epoch": cursor.epoch,
                "cursor_pos": cursor.pos,
                "sample_indices": idxs,
                "loss": float(loss.item()),
                "lr": lr,
                "grad_norm": grad_norm,
                "optimizer_step": did_opt_step,
                "scaler_scale": float(scaler.get_scale()) if scaler.is_enabled() else None,
                "tokens": n_tokens,
                "tokens_per_s": tps,
                "peak_mem_mb": peak_memory_mb(device),
                "dtype": dtype_name,
                "elapsed_s": time.perf_counter() - t_run0,
            }
            row.update(extra)
            rows.append(row)
            log_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log_f.flush()
            if not args.quiet and (step % args.log_interval == 0 or step == int(args.max_steps)):
                gn = "-" if grad_norm is None else f"{grad_norm:.4f}"
                mem = "-" if row["peak_mem_mb"] is None else f"{row['peak_mem_mb']:.0f}MB"
                sc = "-" if row["scaler_scale"] is None else f"{row['scaler_scale']:.0f}"
                print(f"[bounded:{stage}] step {step}/{args.max_steps} loss {row['loss']:.4f} lr {lr:.2e} grad_norm {gn} "
                      f"scale {sc} tok/s {tps:.0f} peak_mem {mem}" + (f" margin {extra['reward_margin']:.4f}" if "reward_margin" in extra else ""))
            if args.save_every and step % args.save_every == 0:
                p = save_ckpt(step)
                if not args.quiet:
                    print(f"[bounded] saved {p}")
    finally:
        log_f.close()

    if args.export_pth:
        sd = {k: v.detach().half().cpu() for k, v in model.state_dict().items()}
        Path(args.export_pth).parent.mkdir(parents=True, exist_ok=True)
        torch.save(sd, args.export_pth)
        if not args.quiet:
            print(f"[bounded] exported MiniMind-format half state_dict → {args.export_pth}")
    summary = {"rows": rows, "log_path": str(log_path), "save_dir": str(save_dir), "data_path": data_path,
               "n_params": sum(p.numel() for p in model.parameters()), "device": device, "dtype": dtype_name,
               "final_step": rows[-1]["step"] if rows else start_step}
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    utf8_stdout()
    args = build_parser().parse_args(argv)
    summary = run(args)
    if not args.quiet:
        print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False))
    last = summary["rows"][-1]["loss"] if summary["rows"] else float("nan")
    return 0 if math.isfinite(last) else 2


if __name__ == "__main__":
    sys.exit(main())
