"""Day 1 数据契约：一条 JSONL 到 (input_ids, labels) 的每一步都要能被检查。

来源
----
从 week01 lab/src/mm_probe/inspect_dataset.py 与 mask_fault.py 复制并裁剪：
保留「渲染 -> 编码 -> 造 label -> 数不变量」这条链和两个不变量的定义，
去掉了对 MiniMind tokenizer 的硬依赖（换成可插拔 tokenizer + 内置回退），
去掉了终端彩色表格。

两个不变量（Day 1 的证据字段，也是 Day 2 故障 A 的判据）
------------------------------------------------------
I1  label 对齐：凡是 labels[i] != -100 的位置，必须有 labels[i] == input_ids[i]。
    labels 本来就是 input_ids 的一个「打了洞的副本」，所以任何非 -100 的位置
    都应该逐位相等。一旦 mask 错位一格，这个计数立刻非零，而 loss 曲线
    可能要几十步后才看出不对——这就是为什么它值得单独记一列。
I2  n_label_tokens <= n_nonpad 且 n_label_tokens > 0。
    等号成立说明 prompt 也进了 loss（pretrain 正常，SFT 就是错的）；
    为 0 说明没有任何 target，cross_entropy 会对 0 个元素取均值得到 nan。

标签约定
--------
本 lab 的模型在 forward 内部做移位（见 model.py 的说明），所以这里产出的
labels 与 input_ids **等长且对齐**。MiniMind 的 lm_dataset.py 是在数据侧就
切成 X=ids[:-1] / Y=ids[1:]，两种写法数学等价但不能混用。
to_minimind_pair() 提供两者之间的显式转换，并在单测里验证等价。
"""

from __future__ import annotations

import json
import os
import zlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

IGNORE_INDEX = -100

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
PAD_TOKEN = "<|endoftext|>"

# 每个阶段必须存在的顶层字段。字段名来自 datasets/DATA_MANIFEST.json 的 fields 列。
STAGE_FIELDS: Dict[str, Tuple[str, ...]] = {
    "pretrain": ("text",),
    "sft": ("conversations",),
    "dpo": ("chosen", "rejected"),
}


class DataContractError(ValueError):
    """数据不符合契约。消息里必须写清楚下一步做什么。"""


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------


class SimpleTokenizer:
    """确定性的字符级 tokenizer，纯标准库。

    存在的理由：MiniMind 的 tokenizer 是外部文件，cc 这台机器上没有；而数据契约
    的所有不变量（对齐、计数、mask 边界）与具体词表无关。用一个确定性映射，
    本机就能把这条链验完，到 V100 上再换成真 tokenizer 结果不变。

    id 分配：0=pad/eos，1=<|im_start|>，2=<|im_end|>，其余字符
    3 + crc32(char) % (vocab_size-3)。crc32 而不是 Python 的 hash()：
    后者带进程级随机盐，换个进程结果就变了。
    """

    def __init__(self, vocab_size: int = 6400) -> None:
        if vocab_size < 16:
            raise ValueError("vocab_size 至少 16")
        self.vocab_size = int(vocab_size)
        self.pad_token_id = 0
        self.eos_token_id = 0
        self.im_start_id = 1
        self.im_end_id = 2
        self._inverse: Dict[int, str] = {0: PAD_TOKEN, 1: IM_START, 2: IM_END}

    def _char_id(self, ch: str) -> int:
        idx = 3 + (zlib.crc32(ch.encode("utf-8")) % (self.vocab_size - 3))
        self._inverse.setdefault(idx, ch)
        return idx

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        i = 0
        while i < len(text):
            if text.startswith(IM_START, i):
                ids.append(self.im_start_id)
                i += len(IM_START)
                continue
            if text.startswith(IM_END, i):
                ids.append(self.im_end_id)
                i += len(IM_END)
                continue
            ids.append(self._char_id(text[i]))
            i += 1
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = False) -> str:
        out: List[str] = []
        for i in ids:
            i = int(i)
            if skip_special and i in (0, 1, 2):
                continue
            out.append(self._inverse.get(i, "�"))
        return "".join(out)


def load_tokenizer(path: Optional[str] = None, vocab_size: int = 6400):
    """有真 tokenizer 就用真的，没有就用 SimpleTokenizer。返回 (tok, 来源标签)。

    真 tokenizer 走 transformers.AutoTokenizer；公司机器上把 MiniMind 的
    model/ 目录或 $MM_WEIGHTS_ROOT/minimind_tokenizer 传进来即可。
    """
    if path and os.path.isdir(path):
        try:
            from transformers import AutoTokenizer  # noqa: WPS433
            tok = AutoTokenizer.from_pretrained(path)
            return tok, "transformers:" + os.path.basename(path.rstrip("/\\"))
        except (ImportError, OSError, ValueError):
            # 内网里 transformers 装不上或版本对不上是常态，不能让数据契约因此跑不了。
            pass
    return SimpleTokenizer(vocab_size=vocab_size), "SimpleTokenizer"


def tokenizer_pad_id(tok) -> int:
    value = getattr(tok, "pad_token_id", None)
    if value is None:
        value = getattr(tok, "eos_token_id", None)
    if value is None:
        raise DataContractError(
            "tokenizer 既没有 pad_token_id 也没有 eos_token_id。\n"
            "下一步：给 tokenizer 设一个 pad_token，否则 padding 位置无法从 "
            "n_nonpad 里剔除，第二个不变量就没有意义。"
        )
    return int(value)


def tokenizer_encode(tok, text: str) -> List[int]:
    """统一 SimpleTokenizer 与 transformers tokenizer 的 encode 调用。"""
    if isinstance(tok, SimpleTokenizer):
        return tok.encode(text)
    return list(tok(text, add_special_tokens=False)["input_ids"])


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------


def render_chat(messages: Sequence[Dict[str, str]],
                add_generation_prompt: bool = False) -> str:
    """ChatML 渲染，与 MiniMind 的 chat_template 同格式。

    自己写而不是调 tokenizer.apply_chat_template：模板本身是本周要检查的对象，
    交给 tokenizer 就看不见 assistant 段的边界到底落在哪里了。
    """
    parts: List[str] = []
    for msg in messages:
        role = str(msg["role"])
        content = str(msg["content"])
        parts.append(IM_START + role + "\n" + content + IM_END + "\n")
    if add_generation_prompt:
        parts.append(IM_START + "assistant\n")
    return "".join(parts)


def assistant_spans(messages: Sequence[Dict[str, str]], tok) -> List[Tuple[int, int]]:
    """返回每个 assistant 回复内容在完整编码里的 [start, end) 区间。

    做法是按消息逐段编码再累加长度，而不是在整条 token 序列里去找分隔符。
    找分隔符的写法在 tokenizer 把 <|im_end|>\\n 合并成一个 token 时会漏，
    那正是 SFT mask 最常见的一格错位来源。
    """
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for msg in messages:
        head = IM_START + str(msg["role"]) + "\n"
        body = str(msg["content"])
        tail = IM_END + "\n"
        n_head = len(tokenizer_encode(tok, head))
        n_body = len(tokenizer_encode(tok, body))
        n_tail = len(tokenizer_encode(tok, tail))
        start = cursor + n_head
        end = start + n_body + n_tail  # 回复内容 + 结束符都要学
        if str(msg["role"]) == "assistant":
            spans.append((start, end))
        cursor = end
    return spans


# ---------------------------------------------------------------------------
# 编码
# ---------------------------------------------------------------------------


def _pad_or_truncate(ids: List[int], max_len: int, pad_id: int) -> List[int]:
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [pad_id] * (max_len - len(ids))


def encode_pretrain(tok, text: str, max_len: int) -> Dict[str, torch.Tensor]:
    """pretrain：所有非 pad 位置都是 target。"""
    pad_id = tokenizer_pad_id(tok)
    ids = tokenizer_encode(tok, str(text))
    ids = _pad_or_truncate(ids, max_len, pad_id)
    input_ids = torch.tensor(ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[input_ids == pad_id] = IGNORE_INDEX
    return {"input_ids": input_ids, "labels": labels}


def encode_sft(tok, messages: Sequence[Dict[str, str]],
               max_len: int) -> Dict[str, torch.Tensor]:
    """SFT：只有 assistant 段（含结束符）是 target，其余全部 -100。"""
    pad_id = tokenizer_pad_id(tok)
    rendered = render_chat(messages)
    ids = tokenizer_encode(tok, rendered)
    spans = assistant_spans(messages, tok)
    labels_full = [IGNORE_INDEX] * len(ids)
    for start, end in spans:
        for i in range(start, min(end, len(ids))):
            labels_full[i] = ids[i]
    ids = _pad_or_truncate(ids, max_len, pad_id)
    labels_full = _pad_or_truncate(labels_full, max_len, IGNORE_INDEX)
    return {"input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels_full, dtype=torch.long)}


def encode_dpo(tok, chosen: Sequence[Dict[str, str]],
               rejected: Sequence[Dict[str, str]],
               max_len: int) -> Dict[str, torch.Tensor]:
    """DPO：chosen / rejected 各走一遍 SFT 编码。mask 规则必须完全一致，
    否则两边的序列 logp 不可比，reward margin 就是噪声。"""
    c = encode_sft(tok, chosen, max_len)
    r = encode_sft(tok, rejected, max_len)
    return {
        "input_ids_chosen": c["input_ids"], "labels_chosen": c["labels"],
        "input_ids_rejected": r["input_ids"], "labels_rejected": r["labels"],
    }


def to_minimind_pair(input_ids: torch.Tensor,
                     labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    """转成 MiniMind lm_dataset 的 (X, Y, loss_mask) 三元组。

    X = ids[:-1]，Y = ids[1:]，loss_mask = (labels != -100)[1:]。
    有了这个函数，本 lab 的 labels 语义与 MiniMind 的语义之间就只剩一次显式转换，
    不会出现「以为对齐了其实差一格」的隐性假设。
    """
    return {
        "x": input_ids[:-1].clone(),
        "y": input_ids[1:].clone(),
        "loss_mask": (labels != IGNORE_INDEX)[1:].to(torch.long),
    }


# ---------------------------------------------------------------------------
# 不变量
# ---------------------------------------------------------------------------


def label_alignment_violations(input_ids: torch.Tensor,
                               labels: torch.Tensor) -> int:
    """不变量 I1：非 -100 的位置上 labels 必须逐位等于 input_ids。返回违例个数。"""
    if input_ids.shape != labels.shape:
        raise DataContractError(
            "input_ids 形状 " + str(tuple(input_ids.shape)) + " 与 labels 形状 "
            + str(tuple(labels.shape)) + " 不一致。\n"
            "下一步：本 lab 的 labels 与 input_ids 等长对齐，移位在模型内部做；"
            "如果你拿的是 MiniMind 的 X/Y 对，先用 to_minimind_pair 的反向转换。"
        )
    active = labels != IGNORE_INDEX
    return int((active & (labels != input_ids)).sum().item())


def token_stats(input_ids: torch.Tensor, labels: torch.Tensor,
                pad_id: int) -> Dict[str, int]:
    """两个不变量需要的四个计数，外加违例数。"""
    n_total = int(input_ids.numel())
    n_nonpad = int((input_ids != pad_id).sum().item())
    n_label = int((labels != IGNORE_INDEX).sum().item())
    return {
        "n_total": n_total,
        "n_nonpad": n_nonpad,
        "n_label_tokens": n_label,
        "n_pad": n_total - n_nonpad,
        "label_align_violations": label_alignment_violations(input_ids, labels),
    }


def check_invariants(stats: Dict[str, int], stage: str) -> Dict[str, Any]:
    """把四个计数判成 PASS / FAIL，并说明失败时第一个该看的东西。"""
    problems: List[str] = []
    if stats["label_align_violations"] != 0:
        problems.append(
            "I1 违例 " + str(stats["label_align_violations"]) + " 个：labels 与 "
            "input_ids 在非 -100 位置对不上。首个检查：mask 是不是整体错位一格"
            "（看 to_minimind_pair 的用法），或者 assistant 段边界算错了。")
    if stats["n_label_tokens"] <= 0:
        problems.append(
            "I2 违例：n_label_tokens=0，没有任何 target。cross_entropy 会对 0 个"
            "元素取均值得到 nan。首个检查：assistant_spans 是不是返回了空列表。")
    if stats["n_label_tokens"] > stats["n_nonpad"]:
        problems.append(
            "I2 违例：n_label_tokens(" + str(stats["n_label_tokens"])
            + ") > n_nonpad(" + str(stats["n_nonpad"]) + ")，说明 pad 位置也进了 "
            "loss。首个检查：pad_id 传对了没有。")
    if stage == "sft" and stats["n_label_tokens"] == stats["n_nonpad"]:
        problems.append(
            "I2 警示：SFT 阶段 n_label_tokens == n_nonpad，prompt 也在学。"
            "这正是 Day 2 故障 A 的 user_in_loss 形态。首个检查：encode_sft 有没有"
            "被 encode_pretrain 顶替。")
    return {"pass": not problems, "problems": problems, "stats": stats}


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------


def detect_stage(record: Dict[str, Any]) -> Optional[str]:
    for stage, fields in STAGE_FIELDS.items():
        if all(f in record for f in fields):
            return stage
    return None


def read_first_records(path: str, limit: int = 8) -> List[Dict[str, Any]]:
    """只读前 limit 行。23 GB 的文件不能整份读进来，也没必要。"""
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DataContractError(
                    "第 " + str(len(out) + 1) + " 行不是合法 JSON：" + str(exc) + "\n"
                    "下一步：多半是传输时被改了换行（文本模式 FTP / 同步工具）。"
                    "用二进制方式重传，再跑 lab/scripts/check_data_layout.py。"
                ) from exc
            if len(out) >= limit:
                break
    if not out:
        raise DataContractError(
            "文件 " + path + " 里一行数据都没有。\n"
            "下一步：确认拷过来的是同一个文件，并跑 check_data_layout.py 比对字节数。"
        )
    return out


def inspect_jsonl(path: str, tok, max_len: int = 128, limit: int = 8,
                  stage: Optional[str] = None) -> Dict[str, Any]:
    """读前 limit 条，核对字段、编码、算不变量。返回可直接写进证据的字典。"""
    records = read_first_records(path, limit)
    detected = stage or detect_stage(records[0])
    if detected is None:
        raise DataContractError(
            "第一行的字段是 " + json.dumps(sorted(records[0].keys()),
                                          ensure_ascii=False)
            + "，不符合任何已知阶段。\n期望其中之一：\n  "
            + "\n  ".join(s + " -> " + ", ".join(f)
                          for s, f in STAGE_FIELDS.items())
            + "\n下一步：确认这个文件对应的阶段，或者用 --stage 显式指定。"
        )
    missing = [f for f in STAGE_FIELDS[detected] if f not in records[0]]
    if missing:
        raise DataContractError(
            "阶段 " + detected + " 需要字段 " + ", ".join(STAGE_FIELDS[detected])
            + "，第一行缺少 " + ", ".join(missing) + "。"
        )
    pad_id = tokenizer_pad_id(tok)
    per_sample: List[Dict[str, int]] = []
    for rec in records:
        if detected == "pretrain":
            enc = encode_pretrain(tok, rec["text"], max_len)
        elif detected == "sft":
            enc = encode_sft(tok, rec["conversations"], max_len)
        else:
            pair = encode_dpo(tok, rec["chosen"], rec["rejected"], max_len)
            enc = {"input_ids": pair["input_ids_chosen"],
                   "labels": pair["labels_chosen"]}
        per_sample.append(token_stats(enc["input_ids"], enc["labels"], pad_id))
    agg = {
        "n_samples": len(per_sample),
        "n_label_tokens_mean": sum(s["n_label_tokens"] for s in per_sample)
                               / len(per_sample),
        "n_nonpad_mean": sum(s["n_nonpad"] for s in per_sample) / len(per_sample),
        "label_align_violations_total": sum(s["label_align_violations"]
                                            for s in per_sample),
        "label_ratio_mean": sum(s["n_label_tokens"] / max(s["n_nonpad"], 1)
                                for s in per_sample) / len(per_sample),
    }
    # 逐条判，再合并。不能把各样本的 min/max 拼成一个虚构样本再判——
    # 那样 n_label_tokens 与 n_nonpad 可能来自不同的两条数据，比较没有意义。
    problems: List[str] = []
    for i, stats in enumerate(per_sample):
        one = check_invariants(stats, detected)
        for msg in one["problems"]:
            problems.append("sample[" + str(i) + "] " + msg)
    verdict = {"pass": not problems, "problems": problems,
               "n_samples_checked": len(per_sample)}
    return {
        "schema": "mm_v100.data_contract/1",
        "stage": detected,
        "max_len": int(max_len),
        "first_line_keys": sorted(records[0].keys()),
        "per_sample": per_sample,
        "aggregate": agg,
        "invariants": verdict,
    }


# ---------------------------------------------------------------------------
# 离线 fixture
# ---------------------------------------------------------------------------

TOY_PRETRAIN = [
    {"text": "梯度累积把一个大 batch 拆成几个小 batch 依次前反向。"},
    {"text": "GradScaler 在 fp16 下把 loss 放大，反向之后再还原。"},
    {"text": "RMSNorm 只有缩放没有平移，也不减去均值。"},
    {"text": "V100 的 Tensor Core 只接受 FP16 输入。"},
]

TOY_SFT = [
    {"conversations": [
        {"role": "user", "content": "什么是学习率？"},
        {"role": "assistant", "content": "每一步沿梯度方向走多远的比例系数。"},
    ]},
    {"conversations": [
        {"role": "user", "content": "为什么 pad 要排除出 loss？"},
        {"role": "assistant", "content": "因为它不是数据，只是对齐用的填充。"},
    ]},
    {"conversations": [
        {"role": "user", "content": "梯度裁剪做什么？"},
        {"role": "assistant", "content": "把全局梯度范数压到阈值以内。"},
        {"role": "user", "content": "阈值一般取多少？"},
        {"role": "assistant", "content": "常见取 1.0，需要按任务调。"},
    ]},
    {"conversations": [
        {"role": "user", "content": "FSDP 和 DDP 的区别？"},
        {"role": "assistant", "content": "DDP 每卡整份参数，FSDP 把参数切片。"},
    ]},
]

TOY_DPO = [
    {
        "chosen": [
            {"role": "user", "content": "解释一下 KL 惩罚。"},
            {"role": "assistant", "content": "它限制新策略偏离参考策略的程度。"},
        ],
        "rejected": [
            {"role": "user", "content": "解释一下 KL 惩罚。"},
            {"role": "assistant", "content": "不知道。"},
        ],
    },
    {
        "chosen": [
            {"role": "user", "content": "什么是 advantage？"},
            {"role": "assistant", "content": "回报相对基线的超额部分。"},
        ],
        "rejected": [
            {"role": "user", "content": "什么是 advantage？"},
            {"role": "assistant", "content": "advantage 就是 advantage。"},
        ],
    },
]

TOY_SETS = {"pretrain": TOY_PRETRAIN, "sft": TOY_SFT, "dpo": TOY_DPO}


def write_toy_fixture(path: str, stage: str, repeat: int = 1) -> str:
    """写一份离线 fixture。没有数据、没有网络时，Day 1 的链路照样能验完。"""
    if stage not in TOY_SETS:
        raise ValueError("stage 只能是 " + " / ".join(sorted(TOY_SETS)))
    rows = TOY_SETS[stage] * max(int(repeat), 1)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return os.path.abspath(path)
