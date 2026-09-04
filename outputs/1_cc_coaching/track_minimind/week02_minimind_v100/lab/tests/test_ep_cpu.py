"""教学版 EP 的三条底线：EP=1 等价 dense、2 进程往返恢复原顺序、空专家不崩。

三条测试对应 01_FOUNDATIONS.md 的三个不变量：
* I3（往返等价）：EP=1 时六步流水的输出必须等于原始 ``MOEFeedForward``；
* I2（split 一致）：2 进程 ep_size=2 时，每个 rank 拿回来的必须是**自己那些 token**
  的正确结果，顺序与 dense 完全一致；
* 失败模式 #2/#5：某个专家一个 token 都没收到时，前向不能崩、反向不能少参与者。
"""

from __future__ import annotations

import pytest
import torch

import conftest as H

TOL = 1e-5


def _build_dense(config_path: str, num_experts: int, seed: int = 1234):
    from mm_dist import common as C

    mm = C.import_minimind(None)
    cfg = C.load_config(config_path)
    kwargs = dict(cfg["model"])
    hidden = kwargs.pop("hidden_size")
    layers = kwargs.pop("num_hidden_layers")
    kwargs.pop("use_moe", None)
    kwargs["num_experts"] = num_experts
    lm_config = mm.MiniMindConfig(hidden_size=hidden, num_hidden_layers=layers,
                                  use_moe=True, **kwargs)
    C.set_seed(seed, per_rank=False)
    return mm, lm_config, mm.MOEFeedForward(lm_config)


# ---------------------------------------------------------------------------
# 单进程：EP=1 与 dense 数值一致
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_experts,top_k", [(4, 1), (4, 2), (6, 2)])
def test_ep_size_one_matches_dense(moe_config, minimind_root, num_experts, top_k):
    from mm_dist import ep_moe

    mm, lm_config, dense = _build_dense(moe_config, num_experts)
    dense.config.num_experts_per_tok = top_k
    torch.manual_seed(7)
    x = torch.randn(2, 17, lm_config.hidden_size)  # 17 个 token，故意不整除
    result = ep_moe.equivalent_to_dense(dense, x, tol=TOL)
    assert result["pass"], (
        "ep_size=1 的六步流水与 MOEFeedForward 不一致，max|diff|="
        + str(result["max_abs_diff"]))


def test_capacity_factor_drops_tokens(moe_config, minimind_root):
    """capacity_factor>0 时确实丢 token；=0（dropless）时一个都不丢。"""
    from mm_dist import ep_moe

    mm, lm_config, dense = _build_dense(moe_config, 4)
    torch.manual_seed(11)
    x = torch.randn(1, 64, lm_config.hidden_size)

    dropless = ep_moe.EPMoEFeedForward.from_dense(dense, ep_size=1,
                                                  capacity_factor=0.0)
    dropless.eval()
    with torch.no_grad():
        dropless(x)
    assert dropless.last_stats["dropped_pairs"] == 0

    tight = ep_moe.EPMoEFeedForward.from_dense(dense, ep_size=1,
                                               capacity_factor=0.25)
    tight.eval()
    with torch.no_grad():
        y = tight(x)
    assert tight.last_stats["dropped_pairs"] > 0, (
        "capacity_factor=0.25 时每专家上限是 ceil(0.25*64*1/4)=4，"
        "64 个 token 不可能都装下")
    assert torch.isfinite(y).all()


def test_validate_ep_constraints():
    from mm_dist import ep_moe

    ep_moe.validate_ep(4, 4, 8)
    ep_moe.validate_ep(2, 4, 8)
    with pytest.raises(ValueError):
        ep_moe.validate_ep(8, 4, 8)   # ep_size > num_experts
    with pytest.raises(ValueError):
        ep_moe.validate_ep(3, 4, 8)   # 不整除
    with pytest.raises(ValueError):
        ep_moe.validate_ep(4, 4, 6)   # world_size 不能被 ep_size 整除


# ---------------------------------------------------------------------------
# 2 进程：往返等价 + 空专家
# ---------------------------------------------------------------------------


def ep_roundtrip_worker(rank: int, world_size: int, port: int, out_dir: str,
                        config: str, num_experts: int) -> None:
    """每个 rank 用自己的 x 跑 EP；结果必须与本地 dense 前向逐元素一致。"""
    H.worker_setup(rank, world_size, port)
    from mm_dist import common as C
    from mm_dist import ep_moe

    info = C.init_dist(backend="gloo", timeout_s=120, force_cpu=True)
    mm, lm_config, dense = _build_dense(config, num_experts)  # 两 rank 权重相同
    group, ep_rank, all_ranks = ep_moe.make_ep_groups(world_size, world_size,
                                                      info.rank)
    ep = ep_moe.EPMoEFeedForward.from_dense(dense, ep_size=world_size,
                                            ep_rank=ep_rank, ep_group=group)
    dense.eval()
    ep.eval()
    gen = torch.Generator()
    gen.manual_seed(100 + rank)
    x = torch.randn(2, 15, lm_config.hidden_size, generator=gen)
    with torch.no_grad():
        y_dense = dense(x)
        y_ep = ep(x)
    max_abs = float((y_dense - y_ep).abs().max())
    H.write_result(out_dir, rank, {
        "rank": rank,
        "ep_rank": ep_rank,
        "ep_groups": all_ranks,
        "max_abs_diff": max_abs,
        "local_expert_ids": ep.local_expert_ids,
        "input_split_sizes": ep.last_stats["input_split_sizes"],
        "output_split_sizes": ep.last_stats["output_split_sizes"],
        "recv_tokens": ep.last_stats["recv_tokens"],
        "global_expert_counts": ep.last_stats["global_expert_counts"],
    })
    C.cleanup_dist(info)


def ep_zero_expert_worker(rank: int, world_size: int, port: int, out_dir: str,
                          config: str, num_experts: int) -> None:
    """把 router 全部偏向 expert 0：rank>0 的本地专家会收到 0 个 token。"""
    H.worker_setup(rank, world_size, port)
    from mm_dist import common as C
    from mm_dist import ep_moe
    from mm_dist.router_stats import inject_router_bias

    info = C.init_dist(backend="gloo", timeout_s=120, force_cpu=True)
    mm, lm_config, dense = _build_dense(config, num_experts)
    group, ep_rank, _ = ep_moe.make_ep_groups(world_size, world_size, info.rank)
    ep = ep_moe.EPMoEFeedForward.from_dense(dense, ep_size=world_size,
                                            ep_rank=ep_rank, ep_group=group)
    ep.train()
    handle = inject_router_bias(ep, expert_id=0, bias=50.0,
                                moe_cls=ep_moe.EPMoEFeedForward)
    gen = torch.Generator()
    gen.manual_seed(200 + rank)
    x = torch.randn(2, 12, lm_config.hidden_size, generator=gen)
    y = ep(x)
    loss = y.float().pow(2).mean() + ep.aux_loss.float()
    loss.backward()
    grad_present = sum(1 for p in ep.parameters() if p.grad is not None)
    handle.remove()
    H.write_result(out_dir, rank, {
        "rank": rank,
        "ep_rank": ep_rank,
        "finite": bool(torch.isfinite(y).all()),
        "loss": float(loss.detach()),
        "recv_tokens": ep.last_stats["recv_tokens"],
        "per_local_expert_tokens": ep.last_stats["per_local_expert_tokens"],
        "global_expert_counts": ep.last_stats["global_expert_counts"],
        "params_with_grad": grad_present,
        "params_total": sum(1 for _ in ep.parameters()),
    })
    C.cleanup_dist(info)


def test_ep_two_ranks_roundtrip(tmp_path, moe_config, minimind_root, spawn_dist):
    out = str(tmp_path / "ep2")
    try:
        results = spawn_dist(ep_roundtrip_worker, 2, out, extra=(moe_config, 4))
    except RuntimeError as exc:
        if "all_to_all" in str(exc).lower():
            pytest.skip("本机 gloo 后端不支持 all_to_all_single：" + str(exc)[:200])
        raise
    assert len(results) == 2
    for res in results:
        assert res["max_abs_diff"] < TOL, (
            "rank " + str(res["rank"]) + " 的 EP 输出与本地 dense 不一致，"
            "max|diff|=" + str(res["max_abs_diff"])
            + "；permute/unpermute 或 split sizes 有问题")
    # 不变量 I2：rank r 发给 rank j 的数量 == rank j 从 rank r 收到的数量
    a, b = results[0], results[1]
    assert a["input_split_sizes"][1] == b["output_split_sizes"][0]
    assert b["input_split_sizes"][0] == a["output_split_sizes"][1]
    assert a["local_expert_ids"] == [0, 1]
    assert b["local_expert_ids"] == [2, 3]
    assert a["ep_groups"] == [[0, 1]]


def test_ep_zero_token_expert_does_not_crash(tmp_path, moe_config, minimind_root,
                                             spawn_dist):
    out = str(tmp_path / "ep_zero")
    try:
        results = spawn_dist(ep_zero_expert_worker, 2, out, extra=(moe_config, 4))
    except RuntimeError as exc:
        if "all_to_all" in str(exc).lower():
            pytest.skip("本机 gloo 后端不支持 all_to_all_single：" + str(exc)[:200])
        raise
    assert len(results) == 2
    for res in results:
        assert res["finite"], "空专家路径产生了 NaN/Inf"
        assert res["params_with_grad"] == res["params_total"], (
            "rank " + str(res["rank"]) + " 有参数没有拿到梯度："
            + str(res["params_with_grad"]) + "/" + str(res["params_total"])
            + "；空专家的 0*sum(p) 分支没起作用，多卡训练会在梯度归约处 hang")
    rank1 = [r for r in results if r["rank"] == 1][0]
    assert rank1["recv_tokens"] == 0, (
        "全部偏向 expert 0 时，持有 expert 2/3 的 rank1 不该收到任何 token，"
        "实际收到 " + str(rank1["recv_tokens"]))
    assert rank1["per_local_expert_tokens"] == [0, 0]
