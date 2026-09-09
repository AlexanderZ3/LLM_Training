"""一个最小的因果 LM，给 mm_rl 的单测和 smoke 用。

为什么要有它
------------
R5（量化 ref 模型）和 R6（LoRA policy）都需要「两个结构相同的模型」才能问出问题。
在本机（无 GPU、无权重）上不可能加载 MiniMind，所以需要一个几十 KB、
**结构与 MiniMind 同形**（q/k/v/o + gate/up/down 的 SwiGLU MLP + RMSNorm + 权重绑定）
的替身：模块命名一致，LoRA 的 target 选择、量化的 skip 列表就能原样搬到真模型上。

它不是 MiniMind，也不假装是。它只保证：
- 前向确定（同 seed、同输入 => 同 logits，CPU 上逐位可复现）；
- 命名与 MiniMind 的注意力/MLP 子模块一致；
- 小到能在 CPU 上一秒跑完几十步。

不能用它得出的结论
------------------
任何关于「真实模型的量化误差 / KL 量级 / outlier 结构」的数字。
这些必须在 V100 上用真权重跑。这里只验代码路径与不变量。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ToyConfig", "RMSNorm", "TinyCausalLM", "make_batch"]


class ToyConfig:
    """结构超参。默认值刻意选得很小，CPU 上单步 < 10 ms。"""

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        num_kv_heads: int = 2,
        intermediate_size: int = 128,
        max_seq_len: int = 64,
        tie_embeddings: bool = True,
        rms_eps: float = 1e-5,
    ) -> None:
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size 必须被 num_heads 整除")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads 必须被 num_kv_heads 整除（GQA 约定）")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.tie_embeddings = tie_embeddings
        self.rms_eps = rms_eps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "intermediate_size": self.intermediate_size,
            "max_seq_len": self.max_seq_len,
            "tie_embeddings": self.tie_embeddings,
        }


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm 的均方在 fp32 域算：fp16 下 x^2 的求和很容易溢出到 inf。
        dt = x.dtype
        x32 = x.to(torch.float32)
        norm = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm.to(dt) * self.weight.to(dt))


class _Attention(nn.Module):
    def __init__(self, cfg: ToyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_rep = cfg.num_heads // cfg.num_kv_heads
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        cfg = self.cfg
        q = self.q_proj(x).view(b, t, cfg.num_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        # 显式写 softmax 注意力而不是 SDPA：SDPA 在不同后端（math / mem-efficient /
        # flash）上的数值不同，而 R1 要区分「精度差异」和「kernel 路径差异」，
        # 参考实现必须是唯一确定的那一条。
        att = (q @ k.transpose(-1, -2)) / math.sqrt(cfg.head_dim)
        causal = torch.full((t, t), float("-inf"), device=x.device, dtype=att.dtype)
        causal = torch.triu(causal, diagonal=1)
        att = att + causal
        att = torch.softmax(att.to(torch.float32), dim=-1).to(att.dtype)
        out = (att @ v).transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(out)


class _MLP(nn.Module):
    def __init__(self, cfg: ToyConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Block(nn.Module):
    def __init__(self, cfg: ToyConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.self_attn = _Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.mlp = _MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class TinyCausalLM(nn.Module):
    """[B, T] token ids -> [B, T, V] logits。没有 KV cache，没有生成接口。"""

    def __init__(self, cfg: Optional[ToyConfig] = None, seed: Optional[int] = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else ToyConfig()
        if seed is not None:
            torch.manual_seed(seed)
        self.embed_tokens = nn.Embedding(self.cfg.vocab_size, self.cfg.hidden_size)
        self.layers = nn.ModuleList([_Block(self.cfg) for _ in range(self.cfg.num_layers)])
        self.norm = RMSNorm(self.cfg.hidden_size, self.cfg.rms_eps)
        self.lm_head = nn.Linear(self.cfg.hidden_size, self.cfg.vocab_size, bias=False)
        if self.cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

    def num_parameters(self, trainable_only: bool = False) -> int:
        ps = [p for p in self.parameters() if (p.requires_grad or not trainable_only)]
        # 绑权重时 embed_tokens.weight 与 lm_head.weight 是同一个张量，去重。
        seen = set()
        total = 0
        for p in ps:
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total


def make_batch(
    batch_size: int = 4,
    seq_len: int = 16,
    vocab_size: int = 256,
    seed: int = 0,
    device: str = "cpu",
    prompt_len: int = 4,
) -> Dict[str, torch.Tensor]:
    """造一批 (input_ids, labels, mask)。mask 里 prompt 部分为 0，只在补全上算 loss。

    这个 mask 就是 R1 里「mask 路径不一致」故障的注入点：
    两条路径拿到不同的 mask 时，logprob 会差得远超 fp16 精度能解释的范围。
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(0, vocab_size, (batch_size, seq_len + 1), generator=g)
    input_ids = ids[:, :-1].contiguous()
    labels = ids[:, 1:].contiguous()
    mask = torch.ones_like(labels, dtype=torch.bool)
    if prompt_len > 0:
        mask[:, : min(prompt_len, mask.shape[1])] = False
    return {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "mask": mask.to(device),
    }
