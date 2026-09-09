"""R3：GradScaler 在 RL 里的正确用法，以及三条规则的可失败检查。

这个实验在 V100 上测的是什么
----------------------------
V100 上没有 BF16，低精度训练只能走 autocast(float16) + GradScaler。
RL 后训练比预训练难在两点：**一个 iteration 里有多个 loss、多个 optimizer**
（policy / value，或 policy / reward-model），而且几乎一定开梯度累积。
torch 2.1 的 AMP 文档对这种情形给了三条硬规则：

    规则 1  多个 loss 各自 scaler.scale(loss).backward()
    规则 2  多个 optimizer 各自 scaler.step(opt)
    规则 3  scaler.update() **每个 iteration 只调一次，且在所有 step 之后**
            （推论：梯度累积期间 scale 必须保持不变，否则不同 micro-batch 的
             梯度是按不同倍数缩放后加在一起的，累加出来的和没有意义）

违反规则 3 的后果特别隐蔽：loss 曲线照常下降（因为大部分 step 仍然是对的），
只有在 inf 出现的那几步上梯度被静默地按错误比例混合。这正是本周主线
「loss 看起来正常但学习信号已经断了」的同一类故障，所以本模块给的不是
一段示例代码，而是一个**会抛异常的纪律检查器**。

不能测什么
----------
- **本机测不了真实的溢出行为**：CPU 上 fp16 是软件模拟，GradScaler 在
  torch 2.1 里只有 CUDA 实现（torch.cuda.amp.GradScaler），
  enabled=False 时 scale 恒为 1，永远不会跳步。
  所以本机验的是**调用顺序的纪律**，V100 上才验 scale 的真实动态
  （inf 检测、backoff、growth_interval）。
- 不能用本机的 skipped-step 计数下任何结论——本机恒为 0。

inf/nan 定位
------------
first_nonfinite_hooks() 在每个子模块的前向输出与反向梯度上挂钩子，
记录**第一个**出现非有限值的位置。GradScaler 只告诉你「这一步跳过了」，
不告诉你是哪一层先坏的；这个钩子补上那一半信息。
注意它有开销，只在排查时打开，不要留在训练主循环里。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

__all__ = [
    "RULES",
    "ScalerRuleViolation",
    "make_grad_scaler",
    "ScalerDiscipline",
    "DisciplinedScaler",
    "AmpTask",
    "rl_amp_iteration",
    "NonFiniteTracker",
    "first_nonfinite_hooks",
]

RULES: Dict[str, str] = {
    "R3-1": "scaler.update() 每个 iteration 必须且只能调用一次",
    "R3-2": "scaler.update() 必须排在所有 scaler.step() 之后；update 之后不得再 scale/step",
    "R3-3": "梯度累积期间 scale 必须保持不变",
    "R3-4": "每个注册的 optimizer 都要有自己的 scaler.step()",
    "R3-5": "同一个 optimizer 在一个 iteration 内 scaler.unscale_() 只能调一次",
}


class ScalerRuleViolation(RuntimeError):
    """违反了 RULES 里的某一条。异常消息带规则号和下一步怎么改。"""

    def __init__(self, rule: str, detail: str) -> None:
        self.rule = rule
        super().__init__("[{}] {}\n规则原文：{}".format(rule, detail, RULES.get(rule, "")))


def make_grad_scaler(
    device_type: str = "cuda",
    enabled: bool = True,
    init_scale: float = 2.0 ** 16,
    growth_factor: float = 2.0,
    backoff_factor: float = 0.5,
    growth_interval: int = 2000,
) -> Tuple[Any, str]:
    """跨 torch 版本地拿一个 GradScaler。返回 (scaler, 用了哪个 API)。

    torch 2.1（目标机）只有 torch.cuda.amp.GradScaler；
    torch >= 2.4（本机 2.14）推荐 torch.amp.GradScaler(device_type, ...)，
    并对旧路径发 FutureWarning。两条都试，先试新的。

    在没有 CUDA 的机器上用 enabled=True 构造 CUDA scaler，torch 会警告并
    把它降级成 disabled——这正是本机的情况，所以本机的所有纪律测试用 enabled=False，
    不去假装有一个会跳步的 scaler。
    """
    new_api = getattr(torch, "amp", None)
    fallback_reason = "torch.amp.GradScaler 不存在（torch {}）".format(torch.__version__)
    if new_api is not None and hasattr(new_api, "GradScaler"):
        try:
            scaler = new_api.GradScaler(
                device_type,
                enabled=enabled,
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
            )
            return scaler, "torch.amp.GradScaler(device_type={!r})".format(device_type)
        except (TypeError, ValueError, RuntimeError) as exc:
            # torch 2.1 没有这个签名，落到下面的 cuda 专用 API；把原因带出去便于排错。
            fallback_reason = "{}: {}".format(type(exc).__name__, str(exc)[:120])
    scaler = torch.cuda.amp.GradScaler(
        enabled=enabled,
        init_scale=init_scale,
        growth_factor=growth_factor,
        backoff_factor=backoff_factor,
        growth_interval=growth_interval,
    )
    return scaler, "torch.cuda.amp.GradScaler（回退原因：{}）".format(fallback_reason)


@dataclass
class _Event:
    kind: str
    name: str
    scale: float
    micro: int


class ScalerDiscipline:
    """记录一个 iteration 内的 scaler 事件序列，并在违规时抛异常。

    它不碰梯度，也不改数值，只看调用顺序和 scale 的取值。
    因此它在**没有 GPU 的机器上也能完整跑**，这是本模块 CPU 可测的原因。

    strict=False 时把违规记进 violations 列表而不抛异常，
    用于「先跑完再一次看全部问题」的排查模式。
    """

    def __init__(self, optimizer_names: Sequence[str], strict: bool = True) -> None:
        if len(set(optimizer_names)) != len(optimizer_names):
            raise ValueError("optimizer_names 有重名：{}".format(list(optimizer_names)))
        if not optimizer_names:
            raise ValueError("至少要注册一个 optimizer 名字")
        self.optimizer_names: List[str] = list(optimizer_names)
        self.strict = strict
        self.events: List[_Event] = []
        self.violations: List[Tuple[str, str]] = []
        self.iteration = -1
        self._active = False
        self._first_scale: Optional[float] = None
        self._micro = 0
        self._stepped: List[str] = []
        self._unscaled: List[str] = []
        self._updates = 0

    # --- 生命周期 -----------------------------------------------------------
    def begin_iteration(self, scale: float) -> None:
        if self._active:
            self._fail("R3-1", "上一个 iteration 还没 end_iteration() 就又 begin 了")
        self._active = True
        self.iteration += 1
        self._first_scale = float(scale)
        self._micro = 0
        self._stepped = []
        self._unscaled = []
        self._updates = 0
        self.events.append(_Event("begin", "", float(scale), 0))

    def note_backward(self, loss_name: str, scale: float) -> None:
        """一个 micro-batch 上的一个 loss 做完 scale(loss).backward() 后调。"""
        self._require_active("note_backward")
        if self._updates > 0:
            self._fail(
                "R3-2",
                "update() 之后还在做第 {} 个 backward（loss={}）；"
                "update 必须是 iteration 的最后一个动作".format(len(self.events), loss_name),
            )
        if self._first_scale is not None and float(scale) != self._first_scale:
            self._fail(
                "R3-3",
                "梯度累积中途 scale 从 {} 变成了 {}（loss={}）；"
                "把 update() 移出 micro-batch 循环".format(self._first_scale, scale, loss_name),
            )
        self.events.append(_Event("backward", loss_name, float(scale), self._micro))

    def note_micro_boundary(self) -> None:
        self._require_active("note_micro_boundary")
        self._micro += 1

    def note_unscale(self, optimizer_name: str, scale: float) -> None:
        self._require_active("note_unscale")
        if optimizer_name in self._unscaled:
            self._fail(
                "R3-5",
                "optimizer {!r} 在同一个 iteration 内被 unscale_() 了两次".format(optimizer_name),
            )
        self._unscaled.append(optimizer_name)
        self.events.append(_Event("unscale", optimizer_name, float(scale), self._micro))

    def note_step(self, optimizer_name: str, scale: float) -> None:
        self._require_active("note_step")
        if optimizer_name not in self.optimizer_names:
            self._fail(
                "R3-4",
                "step() 了一个没注册的 optimizer {!r}；已注册：{}".format(
                    optimizer_name, self.optimizer_names
                ),
            )
        if self._updates > 0:
            self._fail(
                "R3-2",
                "update() 之后还在 step({!r})".format(optimizer_name),
            )
        self._stepped.append(optimizer_name)
        self.events.append(_Event("step", optimizer_name, float(scale), self._micro))

    def note_update(self, scale_before: float) -> None:
        self._require_active("note_update")
        self._updates += 1
        if self._updates > 1:
            self._fail(
                "R3-1",
                "本 iteration 第 {} 次调用 update()；每个 iteration 只能一次".format(self._updates),
            )
        missing = [n for n in self.optimizer_names if n not in self._stepped]
        if missing:
            self._fail(
                "R3-4",
                "update() 时还有 optimizer 没 step：{}".format(missing),
            )
        self.events.append(_Event("update", "", float(scale_before), self._micro))

    def end_iteration(self) -> None:
        self._require_active("end_iteration")
        if self._updates == 0:
            self._fail("R3-1", "整个 iteration 没有调用 update()")
        missing = [n for n in self.optimizer_names if n not in self._stepped]
        if missing:
            self._fail("R3-4", "iteration 结束时仍有 optimizer 没 step：{}".format(missing))
        self._active = False
        self.events.append(_Event("end", "", float("nan"), self._micro))

    # --- 内部 ---------------------------------------------------------------
    def _require_active(self, what: str) -> None:
        if not self._active:
            self._fail("R3-2", "在 begin_iteration() 之前调用了 {}".format(what))

    def _fail(self, rule: str, detail: str) -> None:
        self.violations.append((rule, detail))
        if self.strict:
            raise ScalerRuleViolation(rule, detail)

    def summary(self) -> Dict[str, Any]:
        return {
            "iterations": self.iteration + 1,
            "n_events": len(self.events),
            "violations": [{"rule": r, "detail": d} for r, d in self.violations],
            "event_kinds": [e.kind for e in self.events],
        }


class DisciplinedScaler:
    """把一个真 GradScaler 包起来，调用时自动记账并检查三条规则。

    用法与 GradScaler 完全一致，只多两个动作：begin_iteration() / end_iteration()。
    step() 和 unscale_() 需要传 optimizer 的名字，因为「哪个 optimizer 漏了 step」
    这件事必须能被点名。
    """

    def __init__(
        self,
        scaler: Any,
        optimizer_names: Sequence[str],
        strict: bool = True,
    ) -> None:
        self.scaler = scaler
        self.discipline = ScalerDiscipline(optimizer_names, strict=strict)

    def get_scale(self) -> float:
        return float(self.scaler.get_scale())

    def is_enabled(self) -> bool:
        return bool(getattr(self.scaler, "is_enabled", lambda: False)())

    def begin_iteration(self) -> None:
        self.discipline.begin_iteration(self.get_scale())

    def scale(self, loss: torch.Tensor, loss_name: str = "loss") -> torch.Tensor:
        out = self.scaler.scale(loss)
        self.discipline.note_backward(loss_name, self.get_scale())
        return out

    def micro_boundary(self) -> None:
        self.discipline.note_micro_boundary()

    def unscale_(self, optimizer: Optimizer, optimizer_name: str) -> None:
        self.scaler.unscale_(optimizer)
        self.discipline.note_unscale(optimizer_name, self.get_scale())

    def step(self, optimizer: Optimizer, optimizer_name: str) -> Any:
        out = self.scaler.step(optimizer)
        self.discipline.note_step(optimizer_name, self.get_scale())
        return out

    def update(self) -> None:
        before = self.get_scale()
        self.scaler.update()
        self.discipline.note_update(before)

    def end_iteration(self) -> None:
        self.discipline.end_iteration()


@dataclass
class AmpTask:
    """一个 (loss, optimizer) 对。RL 里 policy 和 value 各是一个 task。"""

    name: str
    optimizer: Optimizer
    loss_fn: Callable[[Any], torch.Tensor]
    params: List[torch.nn.Parameter] = field(default_factory=list)
    max_grad_norm: Optional[float] = None


def rl_amp_iteration(
    dscaler: DisciplinedScaler,
    tasks: Sequence[AmpTask],
    micro_batches: Sequence[Any],
    device_type: str = "cpu",
    amp_dtype: torch.dtype = torch.float16,
    amp_enabled: bool = False,
) -> Dict[str, Any]:
    """**正确**的一个 RL AMP iteration。这是 R3 的参考实现，照抄这个顺序。

    顺序（对应三条规则）：
      for each micro-batch:
          for each task: with autocast: loss = f(batch)
                         dscaler.scale(loss).backward()      <- 规则 1
          micro_boundary()                                    <- scale 在这中间不变（规则 3）
      for each task: unscale_ -> clip -> step                 <- 规则 2
      update()  一次                                          <- 规则 3

    注意 zero_grad 在 iteration **开头**做一次，不是每个 micro-batch 做一次——
    梯度累积的全部意义就在这里。
    """
    if not tasks:
        raise ValueError("rl_amp_iteration 至少需要一个 task")
    if not micro_batches:
        raise ValueError("rl_amp_iteration 至少需要一个 micro-batch")

    for t in tasks:
        t.optimizer.zero_grad(set_to_none=True)

    dscaler.begin_iteration()
    losses: Dict[str, List[float]] = {t.name: [] for t in tasks}
    n_micro = len(micro_batches)

    for batch in micro_batches:
        for t in tasks:
            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                loss = t.loss_fn(batch)
            # 梯度累积要除以 micro-batch 数，否则等效学习率随累积步数漂移。
            loss = loss / float(n_micro)
            dscaler.scale(loss, loss_name=t.name).backward()
            losses[t.name].append(float(loss.detach().to(torch.float32)) * n_micro)
        dscaler.micro_boundary()

    grad_norms: Dict[str, float] = {}
    for t in tasks:
        dscaler.unscale_(t.optimizer, t.name)
        params = t.params if t.params else [
            p for g in t.optimizer.param_groups for p in g["params"]
        ]
        if t.max_grad_norm is not None:
            gn = torch.nn.utils.clip_grad_norm_(params, t.max_grad_norm)
        else:
            gn = torch.nn.utils.clip_grad_norm_(params, float("inf"))
        grad_norms[t.name] = float(gn)
        dscaler.step(t.optimizer, t.name)

    scale_before = dscaler.get_scale()
    dscaler.update()
    scale_after = dscaler.get_scale()
    dscaler.end_iteration()

    return {
        "n_micro_batches": n_micro,
        "mean_loss": {k: (sum(v) / len(v) if v else float("nan")) for k, v in losses.items()},
        "grad_norm": grad_norms,
        "scale_before_update": scale_before,
        "scale_after_update": scale_after,
        "scale_changed": scale_before != scale_after,
        "n_violations": len(dscaler.discipline.violations),
    }


class NonFiniteTracker:
    """记录第一个出现 inf/nan 的位置。由 first_nonfinite_hooks() 创建。"""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        self._handles: List[Any] = []

    @property
    def first(self) -> Optional[Dict[str, Any]]:
        return self.records[0] if self.records else None

    def _note(self, where: str, name: str, tensor: torch.Tensor) -> None:
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            return
        bad = ~torch.isfinite(tensor)
        n_bad = int(bad.sum())
        if n_bad == 0:
            return
        self.records.append(
            {
                "order": len(self.records),
                "where": where,
                "module": name,
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "shape": list(tensor.shape),
                "n_nonfinite": n_bad,
                "frac_nonfinite": n_bad / float(tensor.numel()),
                "max_abs_finite": (
                    float(tensor[~bad].abs().max()) if int((~bad).sum()) > 0 else float("nan")
                ),
            }
        )

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def report(self) -> Dict[str, Any]:
        return {
            "n_records": len(self.records),
            "first": self.first,
            "records": self.records,
        }


def first_nonfinite_hooks(
    model: nn.Module, forward: bool = True, backward: bool = True
) -> NonFiniteTracker:
    """给 model 的每个叶子子模块挂前向/反向钩子，定位第一个非有限值。

    「第一个」按**钩子触发顺序**算，不是按模块定义顺序：前向是从前往后，
    反向是从后往前，所以 records 里 where='backward' 的第一条对应的是
    **最靠近 loss 的那一层**，往前找才是源头。排查时两条都要看。

    用完记得 tracker.remove()，否则钩子会一直留在模型上拖慢训练。
    """
    tracker = NonFiniteTracker()

    def make_fwd(name: str):
        def hook(_mod, _inp, out):
            if isinstance(out, torch.Tensor):
                tracker._note("forward", name, out)
            elif isinstance(out, (tuple, list)):
                for o in out:
                    if isinstance(o, torch.Tensor):
                        tracker._note("forward", name, o)
        return hook

    def make_bwd(name: str):
        def hook(_mod, grad_input, _grad_output):
            for g in grad_input:
                if isinstance(g, torch.Tensor):
                    tracker._note("backward", name, g)
        return hook

    for name, mod in model.named_modules():
        if list(mod.children()):
            continue  # 只挂叶子模块，避免同一个张量被父子模块重复记录
        label = name if name else type(mod).__name__
        if forward:
            tracker._handles.append(mod.register_forward_hook(make_fwd(label)))
        if backward:
            tracker._handles.append(mod.register_full_backward_hook(make_bwd(label)))
    return tracker
