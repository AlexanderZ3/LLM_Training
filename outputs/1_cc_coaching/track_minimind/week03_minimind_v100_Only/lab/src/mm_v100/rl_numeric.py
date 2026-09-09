"""Day 5 RL 数值链：DPO 的两个解析断言 + GRPO 的两个统计量。

来源
----
从 week01 lab/src/mm_probe/dpo_check.py（序列 logp、DPO loss、ref 冻结检查）、
grpo_stats.py（组内 advantage、zero_std 统计）、rule_reward.py（规则 reward）
复制并裁剪：去掉 MiniMind tokenizer/dataset 依赖与 stdout 日志解析，
只留下能在 CPU 上闭式验证的部分，并把 ratio@step0 补上。

为什么这一天只做「解析断言」
----------------------------
64M 模型 + 有界步数下，DPO/GRPO 的效果好坏是没有信息量的。有信息量的是那些
不管模型多小、跑多少步都必须成立的等式。它们一旦不成立，就说明管线接错了，
而不是「效果不好」。四条：

D1  policy 与 ref 权重相同时，DPO 的初始 loss 必须等于 ln2 = 0.693147...
    因为 logits_dpo = (pi_c - pi_r) - (ref_c - ref_r) 恒为 0，
    loss = -logsigmoid(0) = ln2。偏离说明 ref 没冻住，或者 chosen/rejected
    的 mask 规则不一致。
D2  反向之后 ref 模型的所有参数 .grad 必须是 None。不是 0 —— 是 None。
    出现 0 说明 ref 参与了 autograd 图，只是梯度恰好为零，那在真实数据上迟早
    会变成非零，ref 就漂了。
G1  zero_std_group_ratio：组内 reward 全相等的比例。规则 reward 的值域只有
    [-3, 3]，分辨率很差，这个比例可能高到让 GRPO 没有学习信号。
    预注册的出口：>0.8 时本次实验判 INCONCLUSIVE，不判 FAIL。
G2  ratio@step0：重要性采样比 exp(logp_new - logp_old) 在第 0 步必须恒等于 1，
    因为 new 与 old 是同一个策略。偏离 1 说明 old_logp 被算错了
    （常见原因：忘了 detach，或者两次前向用了不同的 mask）。
"""

from __future__ import annotations

import copy
import math
import re
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .data_contract import IGNORE_INDEX

LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------


def per_token_logps(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """[B,T,V] x [B,T] -> [B,T]。在 fp32 里做 log_softmax：

    fp16 的 log_softmax 在 vocab=6400 上就能积累出可见误差，而 DPO 比的是两个
    序列 logp 的**差**，误差不会互相抵消。
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return torch.gather(log_probs, 2, targets.unsqueeze(2)).squeeze(-1)


def sequence_logp(logits: torch.Tensor, input_ids: torch.Tensor,
                  labels: torch.Tensor) -> torch.Tensor:
    """按本 lab 的 labels 约定算 masked 序列 logp，[B]。

    与模型内部一致地做一次移位：位置 i 的 logits 预测位置 i+1 的 token。
    mask 直接由 labels != -100 给出，所以 chosen/rejected 用的是同一条规则。
    """
    shift_logits = logits[:, :-1, :]
    shift_targets = input_ids[:, 1:]
    mask = (labels[:, 1:] != IGNORE_INDEX).float()
    token_logps = per_token_logps(shift_logits, shift_targets)
    return (token_logps * mask).sum(dim=1)


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
             ref_chosen: torch.Tensor, ref_rejected: torch.Tensor,
             beta: float = 0.15) -> Dict[str, torch.Tensor]:
    """输入都是 [B] 序列 logp。返回 loss、logits_dpo、reward margin、accuracy。"""
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


def dpo_init_check(model, batch: Dict[str, torch.Tensor], beta: float = 0.15,
                   device: str = "cpu", tol: float = 1e-5) -> Dict[str, Any]:
    """D1 + D2：初始 loss 是否为 ln2，反向后 ref 的梯度是否为 None。

    ref 用 deepcopy 而不是同 seed 重建：D1 要求两者**逐位相同**，
    重建在 CUDA 上不保证这一点，那样 ln2 会差在小数点后第三位，
    看起来像「数值问题」，其实是方法问题。
    """
    policy = model.to(device)
    ref = copy.deepcopy(policy).eval().requires_grad_(False)
    x_c = batch["input_ids_chosen"].to(device)
    y_c = batch["labels_chosen"].to(device)
    x_r = batch["input_ids_rejected"].to(device)
    y_r = batch["labels_rejected"].to(device)

    policy.train()
    pol_c = sequence_logp(policy(x_c).logits, x_c, y_c)
    pol_r = sequence_logp(policy(x_r).logits, x_r, y_r)
    with torch.no_grad():
        ref_c = sequence_logp(ref(x_c).logits, x_c, y_c)
        ref_r = sequence_logp(ref(x_r).logits, x_r, y_r)
    out = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
    loss = out["loss"]
    policy.zero_grad(set_to_none=True)
    loss.backward()

    ref_grads_none = all(p.grad is None for p in ref.parameters())
    policy_has_grad = any(p.grad is not None for p in policy.parameters())
    init_loss = float(loss.detach().item())
    return {
        "beta": float(beta),
        "init_loss": init_loss,
        "ln2": LN2,
        "abs_delta_to_ln2": abs(init_loss - LN2),
        "D1_init_loss_is_ln2": bool(abs(init_loss - LN2) < tol),
        "D2_ref_grads_all_none": bool(ref_grads_none),
        "policy_has_grad": bool(policy_has_grad),
        "max_abs_logits_dpo": float(out["logits_dpo"].abs().max().item()),
        "mask_tokens_chosen": int((y_c != IGNORE_INDEX).sum().item()),
        "mask_tokens_rejected": int((y_r != IGNORE_INDEX).sum().item()),
        "tol": float(tol),
        "hint": ("D1 不成立的第一个检查：chosen 与 rejected 的 mask 是不是由同一个 "
                 "encode_sft 生成；第二个检查：ref 是不是 deepcopy 而非重建。"),
    }


# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------


def _mean(xs: Sequence[float]) -> float:
    return sum(float(x) for x in xs) / len(xs) if xs else float("nan")


def _pstd(xs: Sequence[float]) -> float:
    """总体标准差（unbiased=False），与 MiniMind train_grpo.py 的 group.std() 一致。"""
    if not xs:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((float(x) - m) ** 2 for x in xs) / len(xs))


def group_advantage(groups: Sequence[Sequence[float]],
                    eps: float = 1e-4) -> List[List[float]]:
    """组内 (r - mean) / (pop_std + eps)。reward 全等的组 advantage 全 0。"""
    out: List[List[float]] = []
    for g in groups:
        m, s = _mean(g), _pstd(g)
        out.append([(float(x) - m) / (s + eps) for x in g])
    return out


def grpo_group_stats(groups: Sequence[Sequence[float]],
                     eps: float = 1e-4) -> Dict[str, Any]:
    """G1：zero_std_group_ratio 与相关统计。"""
    if not groups:
        raise ValueError("groups 不能为空")
    stds = [_pstd(g) for g in groups]
    n_zero = sum(1 for s in stds if s == 0.0)
    adv = group_advantage(groups, eps)
    flat_adv = [a for g in adv for a in g]
    flat_r = [float(x) for g in groups for x in g]
    ratio = n_zero / len(groups)
    return {
        "n_groups": len(groups),
        "group_size": len(groups[0]),
        "zero_std_groups": n_zero,
        "zero_std_group_ratio": ratio,
        "mean_group_std": _mean(stds),
        "mean_reward": _mean(flat_r),
        "adv_mean": _mean(flat_adv),
        "adv_std": _pstd(flat_adv),
        "inconclusive": bool(ratio > 0.8),
        "inconclusive_note": ("zero_std_group_ratio > 0.8：绝大多数组没有学习信号，"
                              "本次 GRPO 判 INCONCLUSIVE，不判 FAIL。"
                              "这是 Day 5 预注册的出口。"),
    }


def ratio_at_step0(logits: torch.Tensor, input_ids: torch.Tensor,
                   labels: torch.Tensor) -> Dict[str, Any]:
    """G2：第 0 步的重要性采样比必须恒等于 1。

    old_logp 就是同一次前向的结果 detach 出来的，所以 exp(new - old) 必须是 1。
    这个检查抓的是「old_logp 用了另一次前向 / 另一套 mask」这类接线错误。
    """
    new_logp = sequence_logp(logits, input_ids, labels)
    old_logp = new_logp.detach()
    ratio = torch.exp(new_logp - old_logp)
    max_dev = float((ratio - 1.0).abs().max().item())
    return {
        "ratio_mean": float(ratio.mean().item()),
        "ratio_max_abs_dev_from_1": max_dev,
        "G2_ratio_is_one": bool(max_dev < 1e-6),
        "hint": ("不为 1 的第一个检查：old_logp 是不是来自另一次前向；"
                 "第二个检查：两次算 logp 用的 mask 是不是同一个。"),
    }


# ---------------------------------------------------------------------------
# 规则 reward
# ---------------------------------------------------------------------------


def _text_ngrams(s: str, n: int) -> List[str]:
    s = re.sub(r"\s+", "", s)
    return [s[i:i + n] for i in range(max(len(s) - n + 1, 0))]


class RuleReward:
    """确定性规则打分，值域 [-3, 3]。

    它不是质量信号，只是让 GRPO 管线能跑起来并观察 advantage 统计。
    分辨率差正是 G1 要量化的那个问题——用模型打分（internlm2-1_8b-reward）
    是升级路径，但那需要额外 3.2 GB 权重和 trust_remote_code。
    """

    def __init__(self, min_len: int = 20, max_len: int = 800,
                 repeat_threshold: float = 0.2) -> None:
        self.min_len = int(min_len)
        self.max_len = int(max_len)
        self.repeat_threshold = float(repeat_threshold)

    def score(self, question: str, response: str) -> float:
        resp = response or ""
        if not resp.strip():
            return -1.0
        value = 0.0
        n = len(resp.strip())
        value += 1.0 if self.min_len <= n <= self.max_len else 0.0
        q2 = set(_text_ngrams(question or "", 2))
        r2 = set(_text_ngrams(resp, 2))
        if q2 and r2 and (q2 & r2):
            value += 1.0
        g3 = _text_ngrams(resp, 3)
        if g3:
            repeat = 1.0 - len(set(g3)) / len(g3)
            if repeat > self.repeat_threshold:
                value -= 1.0
        return max(min(value, 3.0), -3.0)

    def score_groups(self, question: str,
                     generations: Sequence[Sequence[str]]) -> List[List[float]]:
        return [[self.score(question, g) for g in group] for group in generations]


def summarize(dpo: Optional[Dict[str, Any]], grpo: Optional[Dict[str, Any]],
              ratio: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """把四条断言折成一份 rl_numeric.json。"""
    checks: List[Dict[str, Any]] = []
    if dpo is not None:
        checks.append({"id": "D1_init_loss_ln2", "ok": dpo["D1_init_loss_is_ln2"],
                       "detail": "init_loss=%.8f ln2=%.8f |delta|=%.2e"
                                 % (dpo["init_loss"], dpo["ln2"],
                                    dpo["abs_delta_to_ln2"])})
        checks.append({"id": "D2_ref_grads_none", "ok": dpo["D2_ref_grads_all_none"],
                       "detail": "ref 全部参数 .grad is None"})
    if ratio is not None:
        checks.append({"id": "G2_ratio_at_step0", "ok": ratio["G2_ratio_is_one"],
                       "detail": "max|ratio-1|=%.2e"
                                 % ratio["ratio_max_abs_dev_from_1"]})
    status = "PASS"
    if not all(c["ok"] for c in checks):
        status = "FAIL-SYSTEM"
    elif grpo is not None and grpo.get("inconclusive"):
        status = "INCONCLUSIVE"
    return {
        "schema": "mm_v100.rl_numeric/1",
        "dpo": dpo,
        "grpo": grpo,
        "ratio": ratio,
        "checks": checks,
        "status": status,
    }


def render_text(report: Dict[str, Any]) -> str:
    lines = ["== Day 5 RL 数值链 =="]
    for check in report["checks"]:
        lines.append("[%s] %-22s %s" % ("PASS" if check["ok"] else "FAIL",
                                        check["id"], check["detail"]))
    grpo = report.get("grpo")
    if grpo:
        lines.append("")
        lines.append("G1 zero_std_group_ratio=%.3f (n_groups=%d, group_size=%d)"
                     % (grpo["zero_std_group_ratio"], grpo["n_groups"],
                        grpo["group_size"]))
        lines.append("   mean_reward=%.3f adv_std=%.3f"
                     % (grpo["mean_reward"], grpo["adv_std"]))
        if grpo["inconclusive"]:
            lines.append("   " + grpo["inconclusive_note"])
    lines.append("")
    lines.append("status=" + report["status"])
    return "\n".join(lines)
