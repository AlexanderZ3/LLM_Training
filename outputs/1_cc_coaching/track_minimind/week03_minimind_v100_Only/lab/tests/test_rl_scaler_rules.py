"""R3 的单测：GradScaler 三条规则的纪律检查 + inf/nan 定位。

本机可测的（不需要 GPU）：**调用顺序**。ScalerDiscipline 不碰梯度、不改数值，
只看事件序列，所以五种违规在 CPU 上都能被真实地抓到、并且规则号必须对。

本机**测不到**的（有 skipif）：
- GradScaler 的真实动态：init_scale=65536、遇 inf 时 backoff 0.5、
  growth_interval 之后 growth 2.0、以及 skipped step 的计数。
  torch 2.1 的 GradScaler 只有 CUDA 实现；本机构造出来的 scaler enabled=False，
  scale 恒为 1，永不跳步。这些必须在公司 8xV100 上验。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mm_rl import scaler_rules as SR  # noqa: E402

CUDA = torch.cuda.is_available()
NO_CUDA_REASON = (
    "torch 2.1 的 GradScaler 只有 CUDA 实现；没有 GPU 时 enabled=False、scale 恒为 1，"
    "测不到溢出/backoff/跳步的真实行为。必须在公司 8xV100 上验证"
)


def test_rules_table_is_complete():
    for rid in ("R3-1", "R3-2", "R3-3", "R3-4", "R3-5"):
        assert rid in SR.RULES and SR.RULES[rid]


def test_make_grad_scaler_returns_a_scaler_and_names_the_api():
    scaler, api = SR.make_grad_scaler("cpu", enabled=False)
    assert hasattr(scaler, "scale") and hasattr(scaler, "step") and hasattr(scaler, "update")
    assert "GradScaler" in api
    assert scaler.get_scale() == 1.0, "enabled=False 时 scale 恒为 1"


def test_discipline_rejects_duplicate_or_empty_optimizer_names():
    with pytest.raises(ValueError):
        SR.ScalerDiscipline(["policy", "policy"])
    with pytest.raises(ValueError):
        SR.ScalerDiscipline([])


def test_a_correct_iteration_raises_nothing():
    d = SR.ScalerDiscipline(["policy", "value"])
    d.begin_iteration(65536.0)
    for _ in range(4):
        d.note_backward("policy", 65536.0)
        d.note_backward("value", 65536.0)
        d.note_micro_boundary()
    d.note_unscale("policy", 65536.0)
    d.note_step("policy", 65536.0)
    d.note_unscale("value", 65536.0)
    d.note_step("value", 65536.0)
    d.note_update(65536.0)
    d.end_iteration()
    assert d.violations == []
    assert d.summary()["iterations"] == 1


def test_rule_1_update_called_twice():
    d = SR.ScalerDiscipline(["policy"])
    d.begin_iteration(65536.0)
    d.note_backward("policy", 65536.0)
    d.note_step("policy", 65536.0)
    d.note_update(65536.0)
    with pytest.raises(SR.ScalerRuleViolation) as exc:
        d.note_update(32768.0)
    assert exc.value.rule == "R3-1"


def test_rule_1_iteration_ends_without_update():
    d = SR.ScalerDiscipline(["policy"])
    d.begin_iteration(1.0)
    d.note_backward("policy", 1.0)
    d.note_step("policy", 1.0)
    with pytest.raises(SR.ScalerRuleViolation) as exc:
        d.end_iteration()
    assert exc.value.rule == "R3-1"


def test_rule_2_step_after_update():
    d = SR.ScalerDiscipline(["policy"])
    d.begin_iteration(1.0)
    d.note_backward("policy", 1.0)
    d.note_step("policy", 1.0)
    d.note_update(1.0)
    with pytest.raises(SR.ScalerRuleViolation) as exc:
        d.note_step("policy", 1.0)
    assert exc.value.rule == "R3-2"


def test_rule_2_backward_before_begin_iteration():
    d = SR.ScalerDiscipline(["policy"])
    with pytest.raises(SR.ScalerRuleViolation) as exc:
        d.note_backward("policy", 1.0)
    assert exc.value.rule == "R3-2"


def test_rule_3_scale_must_not_change_during_gradient_accumulation():
    """**最隐蔽的一条**：不同 micro-batch 的梯度按不同倍数缩放后相加，和没有意义。

    症状是 loss 曲线照常下降，只在 inf 出现的那几步梯度被静默污染。
    """
    d = SR.ScalerDiscipline(["policy"])
    d.begin_iteration(65536.0)
    d.note_backward("policy", 65536.0)
    d.note_micro_boundary()
    with pytest.raises(SR.ScalerRuleViolation) as exc:
        d.note_backward("policy", 32768.0)
    assert exc.value.rule == "R3-3"


def test_rule_4_an_optimizer_was_never_stepped():
    d = SR.ScalerDiscipline(["policy", "value"])
    d.begin_iteration(1.0)
    d.note_backward("policy", 1.0)
    d.note_step("policy", 1.0)
    with pytest.raises(SR.ScalerRuleViolation) as exc:
        d.note_update(1.0)
    assert exc.value.rule == "R3-4"


def test_rule_4_stepping_an_unregistered_optimizer():
    d = SR.ScalerDiscipline(["policy"])
    d.begin_iteration(1.0)
    d.note_backward("policy", 1.0)
    with pytest.raises(SR.ScalerRuleViolation) as exc:
        d.note_step("critic", 1.0)
    assert exc.value.rule == "R3-4"


def test_rule_5_double_unscale_on_the_same_optimizer():
    d = SR.ScalerDiscipline(["policy"])
    d.begin_iteration(1.0)
    d.note_backward("policy", 1.0)
    d.note_unscale("policy", 1.0)
    with pytest.raises(SR.ScalerRuleViolation) as exc:
        d.note_unscale("policy", 1.0)
    assert exc.value.rule == "R3-5"


def test_non_strict_mode_collects_violations_instead_of_raising():
    d = SR.ScalerDiscipline(["policy", "value"], strict=False)
    d.begin_iteration(1.0)
    d.note_backward("policy", 1.0)
    d.note_step("policy", 1.0)
    d.note_update(1.0)
    d.note_update(1.0)
    d.end_iteration()
    rules = [r for r, _ in d.violations]
    assert "R3-4" in rules and "R3-1" in rules
    assert len(d.summary()["violations"]) == len(d.violations)


def _two_task_setup(device: str = "cpu"):
    torch.manual_seed(0)
    p = nn.Linear(8, 8).to(device)
    v = nn.Linear(8, 1).to(device)
    op = torch.optim.SGD(p.parameters(), lr=0.1)
    ov = torch.optim.SGD(v.parameters(), lr=0.1)
    tasks = [
        SR.AmpTask("policy", op, lambda b: (p(b) ** 2).mean(), list(p.parameters()), 1.0),
        SR.AmpTask("value", ov, lambda b: (v(b) ** 2).mean(), list(v.parameters()), 1.0),
    ]
    micro = [torch.randn(4, 8, device=device) for _ in range(3)]
    return tasks, micro


def test_rl_amp_iteration_reference_implementation_runs_clean():
    scaler, _api = SR.make_grad_scaler("cpu", enabled=False)
    ds = SR.DisciplinedScaler(scaler, ["policy", "value"])
    tasks, micro = _two_task_setup()
    out = SR.rl_amp_iteration(ds, tasks, micro, device_type="cpu")
    assert out["n_violations"] == 0
    assert out["n_micro_batches"] == 3
    assert set(out["grad_norm"]) == {"policy", "value"}
    assert all(g > 0.0 for g in out["grad_norm"].values())
    assert ds.discipline.summary()["violations"] == []


def test_rl_amp_iteration_actually_updates_both_sets_of_parameters():
    scaler, _api = SR.make_grad_scaler("cpu", enabled=False)
    ds = SR.DisciplinedScaler(scaler, ["policy", "value"])
    tasks, micro = _two_task_setup()
    before = [p.detach().clone() for t in tasks for p in t.params]
    SR.rl_amp_iteration(ds, tasks, micro, device_type="cpu")
    after = [p.detach().clone() for t in tasks for p in t.params]
    assert any(not torch.equal(a, b) for a, b in zip(before, after))


def test_rl_amp_iteration_rejects_empty_inputs():
    scaler, _api = SR.make_grad_scaler("cpu", enabled=False)
    ds = SR.DisciplinedScaler(scaler, ["policy"])
    with pytest.raises(ValueError):
        SR.rl_amp_iteration(ds, [], [torch.randn(2, 2)])
    tasks, _micro = _two_task_setup()
    ds2 = SR.DisciplinedScaler(scaler, ["policy", "value"])
    with pytest.raises(ValueError):
        SR.rl_amp_iteration(ds2, tasks, [])


def test_first_nonfinite_hooks_locate_the_offending_module():
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
    tracker = SR.first_nonfinite_hooks(model)
    with torch.no_grad():
        model[2].weight.fill_(float("inf"))
    x = torch.randn(4, 8, requires_grad=True)
    model(x).sum().backward()
    tracker.remove()
    rep = tracker.report()
    assert rep["n_records"] > 0
    assert rep["first"]["where"] in ("forward", "backward")
    assert rep["first"]["n_nonfinite"] > 0
    assert rep["first"]["module"] == "2", "第一个非有限值应当出现在被污染的那一层"


def test_first_nonfinite_hooks_record_nothing_for_a_healthy_model():
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
    tracker = SR.first_nonfinite_hooks(model)
    x = torch.randn(4, 8, requires_grad=True)
    model(x).sum().backward()
    tracker.remove()
    assert tracker.report()["n_records"] == 0
    assert tracker.first is None


@pytest.mark.skipif(not CUDA, reason=NO_CUDA_REASON)
def test_enabled_scaler_backs_off_on_inf_and_skips_the_step():
    """真实溢出行为：遇 inf 时 update() 把 scale 乘 backoff_factor，且这一步被跳过。"""
    scaler, _api = SR.make_grad_scaler("cuda", enabled=True, init_scale=2.0 ** 16)
    lin = nn.Linear(8, 8).cuda()
    opt = torch.optim.SGD(lin.parameters(), lr=0.1)
    before_params = [p.detach().clone() for p in lin.parameters()]
    opt.zero_grad(set_to_none=True)
    loss = (lin(torch.randn(4, 8, device="cuda")) ** 2).mean()
    scaler.scale(loss).backward()
    for p in lin.parameters():
        p.grad.fill_(float("inf"))
    scale_before = scaler.get_scale()
    scaler.step(opt)
    scaler.update()
    assert scaler.get_scale() < scale_before, "遇 inf 之后 scale 必须回退"
    after = [p.detach().clone() for p in lin.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(before_params, after)), "这一步应当被跳过"


@pytest.mark.skipif(not CUDA, reason=NO_CUDA_REASON)
def test_rl_amp_iteration_under_real_autocast_float16():
    """V100 上唯一可用的低精度路径：autocast(float16)。bfloat16 在 sm70 上直接报错。"""
    scaler, _api = SR.make_grad_scaler("cuda", enabled=True)
    ds = SR.DisciplinedScaler(scaler, ["policy", "value"])
    tasks, micro = _two_task_setup(device="cuda")
    out = SR.rl_amp_iteration(
        ds, tasks, micro, device_type="cuda", amp_dtype=torch.float16, amp_enabled=True
    )
    assert out["n_violations"] == 0
    assert out["scale_before_update"] > 1.0
