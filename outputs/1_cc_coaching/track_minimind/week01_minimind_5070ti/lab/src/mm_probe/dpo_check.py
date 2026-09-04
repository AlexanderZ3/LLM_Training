"""DPO 数值检查：序列 logp、policy/ref 差、DPO loss（beta=0.15）、reward margin、ref 是否冻结。

数学部分是纯 torch 函数，镜像 MiniMind `train_dpo.py`：
    logits_to_log_probs(logits, labels) = gather(log_softmax(logits), labels)          # [B, T]
    seq_logp = (per_token_logp * mask).sum(1)                                           # [B]
    logits_dpo = (pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)
    loss = -logsigmoid(beta * logits_dpo).mean()
    reward_margin = beta * logits_dpo（DPO 隐式 reward 之差；训练目标就是把它推正）
注意 MiniMind 的 mask 来自 DPODataset.generate_loss_mask（与 SFT 的 label 规则相同，只在 assistant 段 + <|im_end|>\\n 为 1），
且 x = ids[:-1]、y = ids[1:]、mask = loss_mask[1:]（数据集里已经做了 shift，模型 forward 不传 labels）。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .inspect_dataset import generate_labels, loss_mask_from_labels, special_sequences, EMPTY_THINK


# ---------------------------------------------------------------------------
# 纯数学
# ---------------------------------------------------------------------------
def per_token_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """[B, T, V] × [B, T] → [B, T]；与 MiniMind logits_to_log_probs 相同。"""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)


def sequence_logp(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """masked 求和后的序列 log-prob，[B]。"""
    return (per_token_logps(logits, labels) * mask.to(logits.dtype).float()).sum(dim=1)


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
             ref_chosen: torch.Tensor, ref_rejected: torch.Tensor, beta: float = 0.15) -> Dict[str, torch.Tensor]:
    """输入都是 [B] 序列 logp。返回 loss(标量)、logits_dpo[B]、reward_margin[B]、chosen/rejected 隐式 reward[B]。"""
    pi_logratios = policy_chosen - policy_rejected
    ref_logratios = ref_chosen - ref_rejected
    logits_dpo = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits_dpo).mean()
    return {
        "loss": loss,
        "logits_dpo": logits_dpo,
        "reward_margin": beta * logits_dpo,
        "chosen_reward": beta * (policy_chosen - ref_chosen),
        "rejected_reward": beta * (policy_rejected - ref_rejected),
        "accuracy": (logits_dpo > 0).float().mean(),
    }


def dpo_loss_minimind(ref_log_probs: torch.Tensor, policy_log_probs: torch.Tensor, mask: torch.Tensor, beta: float) -> torch.Tensor:
    """签名与 MiniMind `dpo_loss(ref_log_probs, policy_log_probs, mask, beta)` 一致：
    输入 [2B, T] 的 per-token logp，前一半 chosen、后一半 rejected。"""
    ref_seq = (ref_log_probs * mask).sum(dim=1)
    pol_seq = (policy_log_probs * mask).sum(dim=1)
    b = ref_seq.shape[0] // 2
    return dpo_loss(pol_seq[:b], pol_seq[b:], ref_seq[:b], ref_seq[b:], beta)["loss"]


def param_hash(model: torch.nn.Module) -> str:
    """参数字节的 sha256；两次调用相等 ⇔ 参数完全没变。"""
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(p.detach().cpu().contiguous().float().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 编码（镜像 DPODataset.__getitem__）
# ---------------------------------------------------------------------------
def _encode_side(tokenizer, messages: List[Dict[str, str]], max_length: int, keep_empty_think: bool) -> Dict[str, torch.Tensor]:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if not keep_empty_think and EMPTY_THINK in prompt:
        prompt = prompt.replace(EMPTY_THINK, "")
    ids = tokenizer(prompt, truncation=True, max_length=max_length, padding="max_length").input_ids
    bos_seq, eos_seq = special_sequences(tokenizer)
    loss_mask = loss_mask_from_labels(generate_labels(ids, bos_seq, eos_seq, max_length))
    return {"x": torch.tensor(ids[:-1]), "y": torch.tensor(ids[1:]), "mask": torch.tensor(loss_mask[1:]), "prompt": prompt}


def dpo_encode(tokenizer, chosen: List[Dict[str, str]], rejected: List[Dict[str, str]], max_length: int,
               keep_empty_think: bool = False) -> Dict[str, Any]:
    c = _encode_side(tokenizer, chosen, max_length, keep_empty_think)
    r = _encode_side(tokenizer, rejected, max_length, keep_empty_think)
    return {"x_chosen": c["x"], "y_chosen": c["y"], "mask_chosen": c["mask"],
            "x_rejected": r["x"], "y_rejected": r["y"], "mask_rejected": r["mask"],
            "chosen_prompt": c["prompt"], "rejected_prompt": r["prompt"]}


def dpo_batch_from_samples(tokenizer, samples: List[Dict[str, Any]], max_length: int) -> Dict[str, torch.Tensor]:
    encs = [dpo_encode(tokenizer, s["chosen"], s["rejected"], max_length) for s in samples]
    keys = ["x_chosen", "y_chosen", "mask_chosen", "x_rejected", "y_rejected", "mask_rejected"]
    return {k: torch.stack([e[k] for e in encs]) for k in keys}


@torch.no_grad()
def _forward_logits(model, x: torch.Tensor) -> torch.Tensor:
    return model(x).logits


def evaluate_pair(policy, ref, batch: Dict[str, torch.Tensor], beta: float, device: str) -> Dict[str, Any]:
    """一批 chosen/rejected 对的完整 DPO 数值（不反传）。"""
    x = torch.cat([batch["x_chosen"], batch["x_rejected"]]).to(device)
    y = torch.cat([batch["y_chosen"], batch["y_rejected"]]).to(device)
    mask = torch.cat([batch["mask_chosen"], batch["mask_rejected"]]).to(device)
    b = x.shape[0] // 2
    pol = sequence_logp(_forward_logits(policy, x), y, mask)
    rf = sequence_logp(_forward_logits(ref, x), y, mask)
    out = dpo_loss(pol[:b], pol[b:], rf[:b], rf[b:], beta)
    return {
        "beta": beta,
        "n_pairs": b,
        "mask_tokens_chosen": batch["mask_chosen"].sum(dim=1).tolist(),
        "mask_tokens_rejected": batch["mask_rejected"].sum(dim=1).tolist(),
        "policy_logp_chosen": pol[:b].tolist(),
        "policy_logp_rejected": pol[b:].tolist(),
        "ref_logp_chosen": rf[:b].tolist(),
        "ref_logp_rejected": rf[b:].tolist(),
        "logp_diff_chosen(policy-ref)": (pol[:b] - rf[:b]).tolist(),
        "logp_diff_rejected(policy-ref)": (pol[b:] - rf[b:]).tolist(),
        "logits_dpo": out["logits_dpo"].tolist(),
        "reward_margin": out["reward_margin"].tolist(),
        "dpo_loss": float(out["loss"].item()),
        "accuracy": float(out["accuracy"].item()),
        "ln2_reference": 0.6931471805599453,
    }


def one_policy_step(policy, ref, batch: Dict[str, torch.Tensor], beta: float, device: str, lr: float) -> float:
    """对 policy 做一步 AdamW（用于 --check-ref-frozen），返回该步 loss。"""
    x = torch.cat([batch["x_chosen"], batch["x_rejected"]]).to(device)
    y = torch.cat([batch["y_chosen"], batch["y_rejected"]]).to(device)
    mask = torch.cat([batch["mask_chosen"], batch["mask_rejected"]]).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    with torch.no_grad():
        ref_lp = per_token_logps(ref(x).logits, y)
    policy.train()
    pol_lp = per_token_logps(policy(x).logits, y)
    loss = dpo_loss_minimind(ref_lp, pol_lp, mask, beta)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    policy.eval()
    return float(loss.item())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DPO 数值检查（序列 logp / loss / margin / ref 冻结）")
    p.add_argument("--config", default="tiny_cpu", help="lab/configs/*.json 名或路径")
    p.add_argument("--minimind-root", default=None)
    p.add_argument("--policy", default="none", help="policy 权重（MiniMind .pth 或 bounded_train .pt）；none=随机初始化")
    p.add_argument("--ref", default="same", help="ref 权重；same=与 policy 相同（初始 loss 应为 ln2≈0.6931）")
    p.add_argument("--data-path", default=None, help="dpo.jsonl；缺省用内置 fixture")
    p.add_argument("--index", type=int, default=0, help="取第几条样本")
    p.add_argument("--n", type=int, default=1, help="连续取 n 条组成 batch")
    p.add_argument("--beta", type=float, default=0.15)
    p.add_argument("--max-seq-len", type=int, default=None, help="缺省用 config.train.max_seq_len")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--check-ref-frozen", action="store_true", help="对 policy 做一步更新，比较 ref 参数 hash 前后不变")
    p.add_argument("--step-lr", type=float, default=1e-4)
    p.add_argument("--out", default=None, help="结果 JSON")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    from . import fixtures
    from .minimind_env import (add_minimind_to_path, build_model, find_minimind_root, load_config,
                               load_state_dict_any, load_tokenizer, utf8_stdout)

    utf8_stdout()
    args = build_parser().parse_args(argv)
    root = find_minimind_root(args.minimind_root)
    add_minimind_to_path(root)
    cfg = load_config(args.config)
    device = args.device or cfg.get("train", {}).get("device", "cpu")
    if "cuda" in device and not torch.cuda.is_available():
        device = "cpu"
    max_len = args.max_seq_len or int(cfg.get("train", {}).get("max_seq_len", 64))
    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer(root)
    policy, _ = build_model(cfg["model"], device)
    if args.policy != "none":
        policy.load_state_dict(load_state_dict_any(args.policy), strict=False)
    if args.ref == "same":
        ref = copy.deepcopy(policy)
    else:
        ref, _ = build_model(cfg["model"], device)
        ref.load_state_dict(load_state_dict_any(args.ref), strict=False)
    policy.eval()
    ref.eval()
    ref.requires_grad_(False)

    if args.data_path:
        rows = []
        with open(args.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        rows = fixtures.DPO_SAMPLES
    samples = rows[args.index: args.index + args.n]
    batch = dpo_batch_from_samples(tokenizer, samples, max_len)
    result = evaluate_pair(policy, ref, batch, args.beta, device)
    result["config"] = cfg["name"]
    result["max_seq_len"] = max_len
    result["device"] = device
    if args.check_ref_frozen:
        ref_h0, pol_h0 = param_hash(ref), param_hash(policy)
        loss_before = one_policy_step(policy, ref, batch, args.beta, device, args.step_lr)
        ref_h1, pol_h1 = param_hash(ref), param_hash(policy)
        after = evaluate_pair(policy, ref, batch, args.beta, device)
        result["ref_frozen_check"] = {
            "ref_hash_before": ref_h0, "ref_hash_after": ref_h1, "ref_unchanged": ref_h0 == ref_h1,
            "policy_hash_before": pol_h0, "policy_hash_after": pol_h1, "policy_changed": pol_h0 != pol_h1,
            "loss_before_step": loss_before, "loss_after_step": after["dpo_loss"],
            "reward_margin_after": after["reward_margin"],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
