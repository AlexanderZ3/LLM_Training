"""自带的 MiniMind 形状 dense 因果语言模型（RMSNorm + RoPE + GQA + SwiGLU + tied embedding）。

为什么本周自带模型，而不是 import MiniMind
------------------------------------------
week02 的 lab 用 MINIMIND_ROOT 里的 MiniMindForCausalLM。那样做的代价是：
cc 这台机器上没有 MiniMind 仓库，于是 DDP 等价、字节账、故障注入这些**最需要
本机先验证一遍**的测试全部 skip，开环风险一路带到公司机器上。

本文件把同一套形状自己实现一遍，好处有三个：

1. **本机能真跑。** 所有 CPU 单测不依赖任何外部仓库。
2. **字节账可闭卷手算。** 每个 Linear 的 in/out 都写在这里，Day 3 的手算列
   是有唯一正确答案的，不用去猜第三方实现里有没有多一个 bias。
3. **量化/RL 扩展模块能改内部实现。** 外部仓库的模型类改不动。

代价是它不是 MiniMind 本身：**权重不通用，tokenizer 也不在这里**——MiniMind 的
`.pth` 不能加载进这个类，反过来也一样。要跑 MiniMind 原脚本就直接在
`MINIMIND_ROOT` 下按它自己的 README 跑，本周的 lab 不提供桥接。
两者的**参数形状完全一致**（同样的 ceil(hidden*pi/64)*64 与 QK-norm），
所以字节账、通信账、数值结论可以直接迁移。

形状约定（与 MiniMind model/model_minimind.py 对齐）
--------------------------------------------------
* head_dim = hidden_size // num_attention_heads
* q_proj: [hidden, num_heads*head_dim]，k/v_proj: [hidden, num_kv_heads*head_dim]
* 全部 Linear 无 bias
* intermediate_size 缺省时 = ceil(hidden * pi / 64) * 64（MiniMind 的实际规则，不是常见的 8/3）
* FFN = down(silu(gate(x)) * up(x))
* 每层两个 RMSNorm（attn 前、mlp 前）+ 模型末尾一个 RMSNorm
* lm_head 与 embed_tokens 权重共享（tie_embeddings=True）

标签约定（**与 MiniMind 的 dataset 不同，看清楚**）
--------------------------------------------------
本模型走 HuggingFace 风格：labels 与 input_ids **等长且对齐**，内部做
logits[..., :-1, :] 对 labels[..., 1:] 的移位。MiniMind 的 lm_dataset.py 是在
数据侧就切好 X=ids[:-1]、Y=ids[1:]，模型 forward 不接 labels。
两种写法数学等价，但混用会产生一格错位——这正是 Day 2 故障 A 要练的东西，
所以 data_contract.py 里有一个显式的不变量把它钉死。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -100


@dataclass
class ModelConfig:
    """模型形状。字段名与 MiniMind MiniMindConfig 保持一致，方便对照。"""

    vocab_size: int = 6400
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    intermediate_size: Optional[int] = None
    max_seq_len: int = 512
    rope_theta: float = 1e6
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size=" + str(self.hidden_size) + " 不能被 num_attention_heads="
                + str(self.num_attention_heads) + " 整除"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads=" + str(self.num_attention_heads)
                + " 不能被 num_key_value_heads=" + str(self.num_key_value_heads)
                + " 整除；GQA 要求每组 KV 头服务整数个 Q 头"
            )
        if self.intermediate_size is None:
            # MiniMind 用的是 ceil(hidden * pi / 64) * 64，不是 Llama 系常见的 8/3 倍。
            # 这两个规则在 hidden=768 上给出不同的值（2432 vs 2048），进而让整个模型
            # 差 7,079,424 个参数（63.9M vs 56.8M）——字节账、激活账、Gate 的手算题
            # 全都建立在这个数上，所以必须与 01_FOUNDATIONS.md 第 1 节一致。
            self.intermediate_size = math.ceil(self.hidden_size * math.pi / 64) * 64

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def n_rep(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def param_breakdown(cfg: ModelConfig) -> Dict[str, int]:
    """闭卷手算用的参数量分解。每一项都对应下面某个 nn.Module 的一个权重。

    Day 3 的 A0 题就是不看代码把这张表写出来，所以这里的分项必须和实现一一对应。
    """
    h = cfg.hidden_size
    hd = cfg.head_dim
    nq = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    inter = int(cfg.intermediate_size)
    per_layer = {
        "q_proj": h * (nq * hd),
        "k_proj": h * (nkv * hd),
        "v_proj": h * (nkv * hd),
        "o_proj": (nq * hd) * h,
        "gate_proj": h * inter,
        "up_proj": h * inter,
        "down_proj": inter * h,
        "q_norm": hd,
        "k_norm": hd,
        "input_layernorm": h,
        "post_attention_layernorm": h,
    }
    layer_total = sum(per_layer.values())
    out = {("layer." + k): v for k, v in per_layer.items()}
    out["per_layer_total"] = layer_total
    out["all_layers"] = layer_total * cfg.num_hidden_layers
    out["embed_tokens"] = cfg.vocab_size * h
    out["final_norm"] = h
    out["lm_head"] = 0 if cfg.tie_embeddings else cfg.vocab_size * h
    out["total"] = (out["all_layers"] + out["embed_tokens"] + out["final_norm"]
                    + out["lm_head"])
    # 参数**张量**个数（不是元素个数）。AdamW 会给每个张量额外存一个 step 标量，
    # 字节账里那一项就靠这个数算出来——这是「手算差了几十字节」的常见来源。
    out["n_tensors"] = (1 + cfg.num_hidden_layers * len(per_layer) + 1
                        + (0 if cfg.tie_embeddings else 1))
    return out


class RMSNorm(nn.Module):
    """RMSNorm：只有 scale，没有 bias，也不减均值。

    在 fp16 下这个 .float() 不是可选项：hidden 的平方和在 fp16 里很容易溢到 inf
    （65504 上限），一旦溢出整层输出变 nan，而 loss 曲线在下一步才会体现出来。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        norm = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * norm).to(dtype)


def build_rope_cache(head_dim: int, max_seq_len: int, theta: float,
                     device=None) -> tuple:
    """预计算 RoPE 的 cos/sin。以 float32 保存：角度精度直接决定长序列的位置分辨率。"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2,
                                             dtype=torch.float32,
                                             device=device) / head_dim))
    pos = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> tuple:
    """q/k 形状 [B, n_head, T, head_dim]；cos/sin 形状 [T, head_dim]。"""
    cos = cos.unsqueeze(0).unsqueeze(0).to(q.dtype)
    sin = sin.unsqueeze(0).unsqueeze(0).to(q.dtype)
    return (q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, n_kv, T, hd] -> [B, n_kv*n_rep, T, hd]。

    torch 2.1 的 scaled_dot_product_attention 没有 enable_gqa 参数（那是 2.5 才有的），
    所以 GQA 必须自己 repeat。这一步是显存的真实开销：KV cache 省了，
    但注意力计算时的 K/V 张量还是被展开成满头数。
    """
    if n_rep == 1:
        return x
    b, n_kv, t, hd = x.shape
    return (x[:, :, None, :, :]
            .expand(b, n_kv, n_rep, t, hd)
            .reshape(b, n_kv * n_rep, t, hd))


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hd = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size,
                                cfg.num_attention_heads * hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size,
                                cfg.num_key_value_heads * hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size,
                                cfg.num_key_value_heads * hd, bias=False)
        self.o_proj = nn.Linear(cfg.num_attention_heads * hd,
                                cfg.hidden_size, bias=False)
        # QK-norm：MiniMind 在 RoPE **之前**对 q/k 逐头做 RMSNorm(head_dim)。
        # 只有 2*head_dim 个参数（768/8=96 → 每层 192 个），占比可以忽略，
        # 但它是 MiniMind 真实结构的一部分：漏掉它，闭卷手算的总参数量会差
        # 8*192 = 1,536，和 01_FOUNDATIONS.md 第 1 节对不上。
        self.q_norm = RMSNorm(hd, cfg.norm_eps)
        self.k_norm = RMSNorm(hd, cfg.norm_eps)
        self.dropout_p = float(cfg.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        cfg = self.cfg
        hd = cfg.head_dim
        q = self.q_proj(x).view(b, t, cfg.num_attention_heads, hd).transpose(1, 2)
        k = self.k_proj(x).view(b, t, cfg.num_key_value_heads, hd).transpose(1, 2)
        v = self.v_proj(x).view(b, t, cfg.num_key_value_heads, hd).transpose(1, 2)
        # 顺序要紧：先 QK-norm 再 RoPE。反过来会把旋转后的相位一起归一化掉。
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rope(q, k, cos[:t], sin[:t])
        k = repeat_kv(k, cfg.n_rep)
        v = repeat_kv(v, cfg.n_rep)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(out)


class FeedForward(nn.Module):
    """SwiGLU：down(silu(gate(x)) * up(x))。三个矩阵，不是两个——字节账里最常算错的一项。"""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        inter = int(cfg.intermediate_size)
        self.gate_proj = nn.Linear(cfg.hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """一个 transformer block。FSDP 的 auto_wrap 就按这个类切单元。"""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.mlp = FeedForward(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


@dataclass
class LMOutput:
    """与 MiniMind forward 返回值同名的三个字段，便于两条路径共用训练循环。"""

    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None


class TinyCausalLM(nn.Module):
    """MiniMind 形状的 dense 因果 LM。forward(input_ids, labels=None) -> LMOutput。"""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.config = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            # 共享一份权重对象。字节账里 lm_head 因此是 0，
            # 但反向时它会收到两路梯度——DDP 的 bucket 划分要注意这一点。
            self.lm_head.weight = self.embed_tokens.weight
        cos, sin = build_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        # persistent=False：RoPE 表是纯函数，可由 (head_dim, max_seq_len, theta)
        # 完全重建，所以不进 state_dict，reshard 时不用为它做分片决策。
        #
        # 注意别把这句读成「checkpoint 大小 == 参数字节数」：tie_embeddings 打开时，
        # state_dict 里 `lm_head.weight` 与 `embed_tokens.weight` 是两个 key
        # 共享同一块 storage，扁平化后会**数两遍**。Gate 配置下参数是 63,912,192，
        # 而 flatten_state 的长度是 63,912,192 + 4,915,200 = 68,827,392。
        # 往返一致、判定不受影响，但这两个数不相等。
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> LMOutput:
        b, t = input_ids.shape
        if t > self.config.max_seq_len:
            raise ValueError(
                "序列长度 " + str(t) + " 超过 max_seq_len=" + str(self.config.max_seq_len)
                + "。下一步：改配置里的 model.max_seq_len，或者把 --seq-len 调小。"
            )
        x = self.embed_tokens(input_ids)
        cos = self.rope_cos[:t]
        sin = self.rope_sin[:t]
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            # HF 风格的内部移位：位置 i 的 logits 预测位置 i+1 的 label。
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.float().view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
        return LMOutput(logits=logits, loss=loss,
                        aux_loss=logits.new_zeros(()) if labels is not None else None)

    @torch.no_grad()
    def greedy_generate(self, input_ids: torch.Tensor, max_new_tokens: int,
                        eos_token_id: Optional[int] = None) -> torch.Tensor:
        """最小贪心解码。没有 KV cache——Day 4 只用它做定性对照，不测吞吐。"""
        self.eval()
        out = input_ids
        for _ in range(int(max_new_tokens)):
            window = out[:, -self.config.max_seq_len:]
            logits = self.forward(window).logits[:, -1, :]
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
            if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                break
        return out


def config_from_dict(d: Dict[str, Any]) -> ModelConfig:
    known = {f for f in ModelConfig.__dataclass_fields__}
    unknown = sorted(set(d) - known)
    if unknown:
        raise ValueError(
            "配置的 model 段有不认识的字段：" + ", ".join(unknown)
            + "\n可用字段：" + ", ".join(sorted(known))
        )
    return ModelConfig(**d)


def build_model(model_cfg: Dict[str, Any], device: str = "cpu",
                seed: Optional[int] = None):
    """按配置字典构造模型，返回 (model, cfg)。

    seed 不是 None 时先固定 RNG 再建模型——两次构造得到逐位相同的初始权重，
    fp32/fp16 双路对照与 DDP 等价实验都依赖这一点。
    """
    cfg = config_from_dict(dict(model_cfg))
    if seed is not None:
        torch.manual_seed(int(seed))
    model = TinyCausalLM(cfg).to(device)
    return model, cfg


def count_params(model: nn.Module) -> Dict[str, int]:
    """去重后的参数量。tied embedding 只能数一次，否则字节账会多算一份 vocab*hidden。"""
    seen = set()
    total = 0
    trainable = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return {"total": total, "trainable": trainable, "tensors": len(seen)}
