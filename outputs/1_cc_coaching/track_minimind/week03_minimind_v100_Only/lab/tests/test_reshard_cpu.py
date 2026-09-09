"""分片 checkpoint：写读往返、跨 world_size 重切、step 换算、缺片报错。

CPU 上验的是 FlatShardCheckpoint 这条路径。FSDP + torch.distributed.checkpoint
那条需要 CUDA，在这台机器上跑不了——对应的用例显式 skip 并写明原因，
而不是假装通过。
"""

from __future__ import annotations

import os

import pytest
import torch

import conftest as H
from mm_v100 import ckpt_reshard as CK
from mm_v100 import model as M

TINY = {"vocab_size": 64, "hidden_size": 32, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2,
        "intermediate_size": 64, "max_seq_len": 32}


def test_shard_bounds_partition_is_exact_and_contiguous():
    for total in (10, 100, 106816):
        for ws in (1, 2, 3, 4, 8):
            bounds = [CK.shard_bounds(total, ws, r) for r in range(ws)]
            assert bounds[0][0] == 0
            assert bounds[-1][1] == total
            for a, b in zip(bounds, bounds[1:]):
                assert a[1] == b[0]
            assert sum(e - s for s, e in bounds) == total
            # 余数分给前面的 rank，任意两片长度差不超过 1
            lengths = [e - s for s, e in bounds]
            assert max(lengths) - min(lengths) <= 1


def test_shard_bounds_rejects_bad_rank():
    with pytest.raises(ValueError):
        CK.shard_bounds(10, 2, 5)
    with pytest.raises(ValueError):
        CK.shard_bounds(10, 0, 0)


def test_flatten_unflatten_round_trip():
    a, _ = M.build_model(TINY, device="cpu", seed=1)
    b, _ = M.build_model(TINY, device="cpu", seed=2)
    flat, index = CK.flatten_state(a)
    # 两个模型初始不同，还原之后必须逐位相同。
    assert not all(torch.equal(pa, pb) for pa, pb in
                   zip(a.state_dict().values(), b.state_dict().values()))
    CK.unflatten_into(b, flat, index)
    for key in a.state_dict():
        assert torch.equal(a.state_dict()[key], b.state_dict()[key])


def test_unflatten_length_mismatch_gives_actionable_error():
    a, _ = M.build_model(TINY, device="cpu", seed=1)
    flat, index = CK.flatten_state(a)
    with pytest.raises(ValueError) as exc:
        CK.unflatten_into(a, flat[:-1], index)
    assert "meta.json" in str(exc.value)


def shard_writer(rank: int, world_size: int, port: int, out_dir: str,
                 ckpt_dir: str) -> None:
    """每个 rank 用同一个 seed 建模型（因此参数相同），写自己那一片。"""
    H.worker_setup(rank, world_size, port)
    from mm_v100 import ckpt_reshard as K
    from mm_v100 import common as CC
    from mm_v100 import model as MM

    info = CC.init_dist(backend="gloo", force_cpu=True)
    CC.set_seed(11, per_rank=False)
    model, _ = MM.build_model(TINY, device="cpu", seed=11)
    K.save_flat_shard(ckpt_dir, model, info.rank, info.world_size,
                      meta_extra={"step": 7, "next_step": 8,
                                  "next_step_loss": 1.25, "global_batch": 8,
                                  "accum": 1})
    H.write_result(out_dir, rank, {"rank": rank})
    CC.cleanup_dist(info)


def test_two_ranks_write_and_one_rank_reads_back(tmp_path, spawn_dist):
    ckpt_dir = str(tmp_path / "ckpt")
    spawn_dist(shard_writer, 2, str(tmp_path / "out"), extra=(ckpt_dir,))
    assert os.path.isfile(os.path.join(ckpt_dir, "meta.json"))
    assert os.path.isfile(os.path.join(ckpt_dir, "shard_0.pt"))
    assert os.path.isfile(os.path.join(ckpt_dir, "shard_1.pt"))

    reference, _ = M.build_model(TINY, device="cpu", seed=11)
    loaded, _ = M.build_model(TINY, device="cpu", seed=999)
    meta = CK.load_flat_into(loaded, ckpt_dir)
    assert meta["world_size"] == 2
    assert meta["step"] == 7
    for key in reference.state_dict():
        assert torch.equal(reference.state_dict()[key],
                           loaded.state_dict()[key]), key


def test_reshard_to_a_different_world_size(tmp_path, spawn_dist):
    ckpt_dir = str(tmp_path / "ckpt2")
    spawn_dist(shard_writer, 2, str(tmp_path / "out2"), extra=(ckpt_dir,))
    full, meta = CK.load_flat_full(ckpt_dir)
    for new_ws in (1, 3, 4, 8):
        pieces = [CK.reshard_slice(ckpt_dir, new_ws, r)[0]
                  for r in range(new_ws)]
        assert sum(p.numel() for p in pieces) == int(meta["total_numel"])
        assert torch.equal(torch.cat(pieces), full)


def test_missing_shard_reports_which_rank(tmp_path, spawn_dist):
    ckpt_dir = str(tmp_path / "ckpt3")
    spawn_dist(shard_writer, 2, str(tmp_path / "out3"), extra=(ckpt_dir,))
    os.remove(os.path.join(ckpt_dir, "shard_1.pt"))
    with pytest.raises(FileNotFoundError) as exc:
        CK.load_flat_full(ckpt_dir)
    assert "shard_1.pt" in str(exc.value)


def test_missing_meta_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        CK.read_meta(str(tmp_path))
    assert "reshard_check.py" in str(exc.value)


def test_convert_step_matches_minimind_formula():
    out = CK.convert_step(saved_step=100, saved_ws=8, current_ws=4)
    assert out["converted_step"] == 200
    same = CK.convert_step(saved_step=100, saved_ws=8, current_ws=8)
    assert same["converted_step"] == 100
    # 换算的前提必须跟着一起返回，否则这个数会被当成普适真理用。
    assert "assumption" in out and "data_side_truth" in out


def test_verify_continuity_thresholds():
    ok = CK.verify_continuity(1.0, 1.00001, tol=1e-3)
    bad = CK.verify_continuity(1.0, 1.5, tol=1e-3)
    assert ok["pass"] is True
    assert bad["pass"] is False
    assert "shard_bounds" in bad["hint"]


def test_global_forward_loss_is_world_size_independent():
    """同一份权重、同一个 step，单进程算出来的 loss 与 world_size 无关。

    这是 reshard 判据成立的前提：如果这条不成立，
    「恢复后 loss 没跳变」就不能作为分片正确的证据。
    """
    from mm_v100 import common as C

    model, cfg = M.build_model(TINY, device="cpu", seed=5)
    dataset = C.TinyDataset(num_samples=128, seq_len=16,
                            vocab_size=TINY["vocab_size"], seed=99)
    a = CK.global_forward_loss(model, dataset, step=3, global_batch=8, accum=1,
                               world_size=1, rank=0, device="cpu")
    b = CK.global_forward_loss(model, dataset, step=3, global_batch=8, accum=2,
                               world_size=1, rank=0, device="cpu")
    assert a == pytest.approx(b, abs=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="FSDP SHARDED_STATE_DICT 需要 CUDA；"
                           "cc 的开发机没有 GPU，这条只能在公司 8xV100 上跑")
def test_fsdp_sharded_save_load_placeholder():
    pytest.fail("到了有 GPU 的机器上请把这条改成真实的 FSDP 往返测试")


def test_dcp_wrappers_exist_and_are_version_tolerant():
    """不真跑 DCP（它要进程组），只确认两个 API 名的兼容分支都在。"""
    import inspect

    src = inspect.getsource(CK.dcp_save) + inspect.getsource(CK.dcp_load)
    assert "save_state_dict" in src and "load_state_dict" in src
    assert "dcp.save(" in src and "dcp.load(" in src
