"""mm_quant — Week M03 扩展模块 A：8-bit 量化 / QAT（实验 Q1–Q6）。

这个包对应 06_QUANT_LOWBIT.md 的 Q1–Q6，全部零新依赖（只 torch / numpy / 标准库）。
bitsandbytes 只作为**可选对照**出现在 opt8bit.try_import_bnb() 里，装不上是预期结果。

模块 -> 实验
------------
quantizers.py    量化器本体（对称/非对称 x per-tensor/per-channel/per-group）
error_budget.py  Q1 逐层误差与存储账；Q2 截断阈值扫描与 MSE 分解
opt8bit.py       Q3 block-wise 8-bit Adam 状态
weight_only.py   Q4 W8A16 weight-only PTQ
qat.py           Q5 per-channel fake-quant + STE（自写 vs torch.ao 两条路径）
smooth.py        Q6 激活 outlier 统计与幅度迁移

V100（sm70）边界，一句话版本
----------------------------
没有 INT8 Tensor Core（IMMA 从 sm75 起），所以**任何 8-bit 提速在这台机器上都是假的**；
但 fake-quant 有原生 CUDA 前反向 kernel，所以 **QAT 是真跑的**。
本包因此只回答两个问题：误差有多大、字节省了多少。
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "quantizers",
    "error_budget",
    "opt8bit",
    "weight_only",
    "qat",
    "smooth",
]
