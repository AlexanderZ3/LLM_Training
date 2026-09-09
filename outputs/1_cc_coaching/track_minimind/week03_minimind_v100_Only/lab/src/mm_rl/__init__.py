"""mm_rl — Week M03 扩展模块 B：低精度 + RL 后训练（实验 R1–R6）。

对应 07_RL_LOWPRECISION.md 的 R1–R6，零新依赖（只 torch / numpy / 标准库）。
**不装 vLLM / verl / OpenRLHF**：已核验（2026-09-08）vLLM 每个版本硬钉一个
torch 版本（0.2.7 要 torch==2.1.2，不是 2.1.0），官方最低算力 7.5（V100 是 7.0）；
verl 要 CUDA >= 12.8；OpenRLHF 绑 vllm 0.27.1。全部与目标机不相容。
trl 的 GRPOConfig.use_vllm 默认 False，真要用 trl 也不需要 vLLM——
但本模块连 trl 都不依赖，六个实验全部是自包含的数值实验。

模块 -> 实验
------------
logprob.py       R1 fp16 vs fp32 的 logprob 与 ratio，含四类归因
kl.py            R2 三个 KL 估计量 k1/k2/k3 的精度对比
scaler_rules.py  R3 GradScaler 三条规则的纪律检查 + inf/nan 首次出现定位
advantage.py     R4 组内 advantage 归一化的灾难性抵消
ref_model.py     R5 冻结 ref 模型 weight-only INT8（与 mm_quant 的交点）
lora_policy.py   R6 LoRA policy 的四项字节账与 step-0 不变量
toy.py           单测与 smoke 用的最小因果 LM（不是 MiniMind，只验路径与不变量）

V100（sm70）边界，一句话版本
----------------------------
只有 fp16，没有 BF16。fp16 的 eps = 2^-10 ≈ 9.8e-4、最小正规数 6.1e-5、最大 65504。
RL 的每一个关键量（log_ratio、advantage、KL）都是「两个几乎相等的数相减」，
所以本模块的六个实验全部围绕**灾难性抵消**这一件事展开。
额外一条来自 torch AMP 文档的警告：**bf16 预训练的模型在 fp16 下会梯度上溢
而不是下溢**，GradScaler 的 scale 会被一路压到 1 以下甚至失效——
选基座时要先做 fp16 前向的溢出统计，不能想当然。
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "logprob",
    "kl",
    "scaler_rules",
    "advantage",
    "ref_model",
    "lora_policy",
    "toy",
]
