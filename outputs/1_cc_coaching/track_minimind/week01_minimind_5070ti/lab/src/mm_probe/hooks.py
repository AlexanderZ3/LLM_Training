"""训练仪表：吞吐计、显存峰值、全局梯度范数。纯 torch，不依赖 MiniMind。"""
from __future__ import annotations

import time
from typing import Optional

import torch


class ThroughputMeter:
    """按 step 统计 tokens/s。

    用法：
        meter = ThroughputMeter(device)
        meter.start()            # step 开始
        ...forward/backward...
        rate = meter.stop(n_tokens)   # step 结束，返回本步 tokens/s；同时累计总量
        meter.avg_rate()         # 自 reset 以来的平均 tokens/s
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or "cpu"
        self.reset()

    def reset(self) -> None:
        self.total_tokens = 0
        self.total_seconds = 0.0
        self.last_rate: Optional[float] = None
        self._t0: Optional[float] = None

    def _sync(self) -> None:
        if "cuda" in str(self.device) and torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self) -> None:
        self._sync()
        self._t0 = time.perf_counter()

    def stop(self, n_tokens: int) -> float:
        if self._t0 is None:
            raise RuntimeError("ThroughputMeter.stop() 前必须先 start()")
        self._sync()
        dt = max(time.perf_counter() - self._t0, 1e-9)
        self._t0 = None
        self.total_tokens += int(n_tokens)
        self.total_seconds += dt
        self.last_rate = float(n_tokens) / dt
        return self.last_rate

    def avg_rate(self) -> Optional[float]:
        if self.total_seconds <= 0:
            return None
        return self.total_tokens / self.total_seconds


def peak_memory_mb(device: Optional[str] = None, reset: bool = False) -> Optional[float]:
    """返回 CUDA 已分配显存峰值（MB）；无 CUDA 返回 None。reset=True 时读完后清零峰值统计。"""
    if not torch.cuda.is_available():
        return None
    if device is not None and "cuda" not in str(device):
        return None
    dev = torch.device(device) if device is not None else torch.device("cuda")
    peak = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
    if reset:
        torch.cuda.reset_peak_memory_stats(dev)
    return float(peak)


def grad_global_norm(model: torch.nn.Module, norm_type: float = 2.0) -> float:
    """所有含梯度参数的全局范数（与 clip_grad_norm_ 返回的 total_norm 同定义）。无梯度时返回 0.0。"""
    grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    norms = torch.stack([torch.linalg.vector_norm(g.float(), norm_type) for g in grads])
    return float(torch.linalg.vector_norm(norms, norm_type).item())
