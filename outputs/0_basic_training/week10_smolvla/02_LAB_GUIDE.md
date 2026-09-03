# Week 10 实践篇：从空目录审计并条件执行 SmolVLA 微调

> 核验日期：2026-09-03。本文不声称任何命令已在目标硬件运行；所有输出示例均为预期。公司 8×V100/Linux、PyTorch 2.1、32 GB/卡和拓扑为用户自述，先核验。公司代码、Task-P、日志、trace、图片/视频、checkpoint、性能数字和拓扑信息不得导出。

## 0. 成功状态与三条资源路径

完整 `PASS` 需要：精确 code/model/data revision 与许可；严格 processor/schema；单 batch→smoke→200+ step→resume→独立 base/FT eval；FP16 有限；checkpoint 状态缺口如实披露；有条件的四卡 DDP。三道先决门：

- `RUNTIME`：LeRobot v0.4.3 要求 `torch>=2.2.1,<2.8.0`，公司 PyTorch 2.1 不兼容。公司主环境直接 `ENV-BLOCKED`，绝不升级。
- `LICENSE`：核验时 `lerobot/smolvla_base` 模型卡未声明 license。需组织/法务书面批准；未获批则 `LICENSE-BLOCKED`，不下载权重。
- `DATA`：公开 `lerobot/svla_so101_pickplace` 为 Apache-2.0，可作为个人/批准环境的从零演练数据；Task-P 另需内部 revision/license/schema，只留公司。

资源路径：固定 v0.4.3 在公司 V100 上停在审计（PyTorch2.1 版本冲突且嵌套 VLM hard-code BF16）；只有另立 patch SHA 并经组织实测的适配分支才可另开实验，本文不提供通过声明。个人5070Ti可在现有兼容环境使用公开数据（不新建conda/venv、不接收公司产物）；H100可单列，不能替代V100证据。

止损：pip 计划替换 torch/CUDA、权重许可未批、revision/hash 不符、camera/action contract 不符、严格 load 有 missing/unexpected 核心 key、非有限 loss/grad、峰值 reserved 达物理显存 85%、resume step/optimizer/RNG 不连续、外部 upload/logger 未关闭。

## 1. 精确来源、许可、下载和离线替代

| 对象 | 固定 ID/revision | 许可状态 | 下载命令所在日 | 离线替代 |
|---|---|---|---|---|
| LeRobot | tag v0.4.3；commit `0b067df57d21d3a02d6c511f1609172fa39ac29b`；Git tree `b5891d6333c4eb611bd3e81a4d13ed3c71210a6e` | Apache-2.0 | Day 1 git clone/fetch | 管理员导入含 `.git` 的批准归档，同 commit/tree；另核本周 patch SHA |
| SmolVLA base | `lerobot/smolvla_base@04d96c1f9167360280aaa54de31418f824d1ef48` | 模型卡 license 字段未声明，必须额外批准 | Day 2，在批准文件存在后 | 批准快照 + 文件 SHA256SUMS；不得从个人搬到公司 |
| 嵌套 SmolVLM2 | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct@7b375e1b73b11138ff12fe22c8f2822d8fe03467` | Apache-2.0 | Day 2，与 base 一起固定 | 批准快照；必须含 model、processor、tokenizer 资产，排除非运行必需 ONNX |
| public dataset | `lerobot/svla_so101_pickplace@f641879e22172be7e8161d5e6c1503c2d2feb657` | Apache-2.0 | Day 2 | 批准快照 + SHA256SUMS |
| Task-P | owner 提供 ID/revision/split/license/schema | 当前未知 | 本手册不猜路径 | 公司内部批准导入；缺失即 DATA-BLOCKED |

公开数据已核字段：v3.0、50 episodes、11,939 frames、30 FPS，state/action 各 6 维，视频 key `observation.images.up/side`。base 固定配置期望 `observation.image/image2/image3`、chunk 50、内部 state/action pad 到 32、flow steps 10。

官方一手链接：[LeRobot v0.4.3](https://github.com/huggingface/lerobot/tree/v0.4.3)、[精确 SmolVLA 源码](https://github.com/huggingface/lerobot/tree/0b067df57d21d3a02d6c511f1609172fa39ac29b/src/lerobot/policies/smolvla)、[v0.4.3 SmolVLA 文档](https://huggingface.co/docs/lerobot/v0.4.3/en/smolvla)、[base 固定 tree](https://huggingface.co/lerobot/smolvla_base/tree/04d96c1f9167360280aaa54de31418f824d1ef48)、[SmolVLM2 固定 tree](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/tree/7b375e1b73b11138ff12fe22c8f2822d8fe03467)、[dataset 固定 tree](https://huggingface.co/datasets/lerobot/svla_so101_pickplace/tree/f641879e22172be7e8161d5e6c1503c2d2feb657)、[multi-GPU](https://huggingface.co/docs/lerobot/v0.4.3/en/multi_gpu_training)。2026-09-03 核验。

## 2. 环境前检：复用现有环境，不建 conda/venv

```bash
which python
python -VV
python -m pip --version
python -m pip freeze > /tmp/week10-before-freeze.txt
python - <<'PY'
import os,sys,torch
print({'python':sys.version,'torch':torch.__version__,'torch_file':torch.__file__,
       'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
       'gpu_count':torch.cuda.device_count(),'CVD':os.getenv('CUDA_VISIBLE_DEVICES')})
for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i); print(i,p.name,p.total_memory,(p.major,p.minor))
PY
nvidia-smi -L
nvidia-smi topo -m
ffmpeg -version | head -n 1
git --version
```

公司环境看到 torch 2.1 即记录后停止真实安装；不是故障排除目标。更关键的是固定源码的嵌套 VLM `from_pretrained(... torch_dtype="bfloat16")`，所以 V100 即使另有 torch>=2.2.1 环境仍为 `ENV-BLOCKED`，除非组织评审一个另立 code/patch SHA 的 dtype 适配；本文不声称该适配已验证。下文训练/eval 命令只适用于个人 5070Ti或其他“已有、兼容且批准、支持 BF16 加载”的 GPU 环境。V100 不装 FA2、不用 BF16/TF32/FP8、不升级 CUDA13。

## 3. 目录树

```text
week10-smolvla/
├── vendor/lerobot/
├── assets/smolvla_base/
├── assets/smolvlm2/
├── assets/dataset/
├── assets/dataset_train/
├── assets/runtime_bundle/
├── assets/LOCK.json
├── patches/
├── splits/episodes.json
├── artifacts/train_stats.json
├── artifacts/train_stats.lineage.json
├── artifacts/processors/
├── src/preflight.py
├── src/contracts.py
├── src/inspect_assets.py
├── src/lineage.py
├── src/apply_vendor_patch.py
├── src/make_split_stats.py
├── src/prepare_bundle.py
├── src/seal_processor_contract.py
├── src/audit_policy.py
├── src/compare_resume.py
├── src/test_action_mask.py
├── src/offline_eval.py
├── evidence/
├── logs/
├── runs/
├── traces/
└── reports/model_card.md
```

## Day 1：创建审计工具、固定代码并执行 runtime 门

### 为什么做

在任何依赖安装/权重下载前，先证明代码身份与环境兼容；公司 2.1 应在这里安全停止。

### 输入与前置检查

只需 git、Python 和已有 torch；联网或公司批准 git 镜像。

### 本日要创建/修改的文件

创建所有目录、十一个完整本地脚本和固定 `vendor/lerobot`。

### 实现

```bash
mkdir -p week10-smolvla/{vendor,assets/smolvla_base,assets/smolvlm2,assets/dataset,src,evidence,logs,runs,traces,reports,patches,splits,artifacts}
cd week10-smolvla
cat > src/preflight.py <<'PY'
#!/usr/bin/env python3
import argparse,json,re,subprocess,sys
from pathlib import Path
import torch

EXPECTED='0b067df57d21d3a02d6c511f1609172fa39ac29b'
EXPECTED_TREE='b5891d6333c4eb611bd3e81a4d13ed3c71210a6e'
def version_pair(text):
    m=re.match(r'(\d+)\.(\d+)',text)
    if not m: raise ValueError(text)
    return tuple(map(int,m.groups()))
def main():
    p=argparse.ArgumentParser(); p.add_argument('--repo',default='vendor/lerobot'); p.add_argument('--pip-report',default='')
    a=p.parse_args(); repo=Path(a.repo)
    sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    tree_sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD^{tree}'],text=True).strip()
    planned=[]
    if a.pip_report:
        report=json.loads(Path(a.pip_report).read_text(encoding='utf-8'))
        planned=[x.get('metadata',{}).get('name','').lower() for x in report.get('install',[])]
    gpu=[]
    for i in range(torch.cuda.device_count()):
        q=torch.cuda.get_device_properties(i); gpu.append({'index':i,'name':q.name,'bytes':q.total_memory,'cc':[q.major,q.minor]})
    runtime_ok=sys.version_info>=(3,10) and (2,2)<=version_pair(torch.__version__)<(2,8)
    forbidden=sorted(x for x in planned if x=='torch' or x.startswith('nvidia-'))
    is_v100=any(x['cc']==[7,0] or 'V100' in x['name'] for x in gpu)
    source=(repo/'src/lerobot/policies/smolvla/smolvlm_with_expert.py').read_text(encoding='utf-8')
    hardcoded_bf16='torch_dtype="bfloat16"' in source
    row={'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':gpu,'repo_sha':sha,
         'repo_tree_sha':tree_sha,'repo_ok':sha==EXPECTED and tree_sha==EXPECTED_TREE,
         'nested_vlm_hardcoded_bf16':hardcoded_bf16,'v100_detected':is_v100,
         'runtime_ok':runtime_ok,'planned_packages':planned,'forbidden_replacements':forbidden}
    print(json.dumps(row,indent=2))
    if sha!=EXPECTED or tree_sha!=EXPECTED_TREE: raise SystemExit('CODE_REVISION_MISMATCH')
    if not runtime_ok: raise SystemExit('ENV-BLOCKED: v0.4.3 requires Python>=3.10 and torch>=2.2.1,<2.8; do not upgrade company torch 2.1')
    if forbidden: raise SystemExit('INSTALL_PLAN_BLOCKED: torch/CUDA replacement requested')
    if is_v100 and hardcoded_bf16: raise SystemExit('ENV-BLOCKED: pinned SmolVLM2 loader hard-codes BF16; V100 has no native BF16 path')
if __name__=='__main__': main()
PY
cat > src/apply_vendor_patch.py <<'PY'
#!/usr/bin/env python3
import hashlib,subprocess
from pathlib import Path

ROOT=Path('vendor/lerobot')
EXPECTED='0b067df57d21d3a02d6c511f1609172fa39ac29b'
policy=ROOT/'src/lerobot/policies/smolvla/modeling_smolvla.py'
factory=ROOT/'src/lerobot/policies/factory.py'
trainer=ROOT/'src/lerobot/scripts/lerobot_train.py'
if subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()!=EXPECTED:
    raise SystemExit('CODE_REVISION_MISMATCH')
if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():
    raise SystemExit('REFUSE_DIRTY_VENDOR_BEFORE_PATCH')

text=policy.read_text(encoding='utf-8')
anchor='''class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None
'''
helper=anchor+'''

def _reduce_action_losses(losses: Tensor, action_is_pad: Tensor | None, reduction: str):
    """Token-weighted temporal mask; compensate for DDP's gradient averaging."""
    if action_is_pad is None:
        valid = torch.ones(losses.shape[:2], dtype=torch.bool, device=losses.device)
    else:
        if tuple(action_is_pad.shape) != tuple(losses.shape[:2]):
            raise ValueError(f"action_is_pad {action_is_pad.shape} != {losses.shape[:2]}")
        valid = ~action_is_pad.to(device=losses.device, dtype=torch.bool)
    valid_f = valid.to(losses.dtype)
    masked = losses * valid_f.unsqueeze(-1)
    if reduction == "none":
        denom = valid_f.sum(dim=1) * losses.shape[-1]
        torch._assert_async(torch.all(denom > 0), "ALL_ZERO_ACTION_MASK")
        per_sample = masked.sum(dim=(1, 2)) / denom.clamp_min(1)
        return per_sample, per_sample.mean()
    if reduction != "mean":
        raise ValueError(f"unsupported reduction: {reduction}")
    local_num = masked.sum()
    local_den = valid_f.sum() * losses.shape[-1]
    world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    global_den = local_den.detach().clone()
    global_num = local_num.detach().clone()
    if world > 1:
        torch.distributed.all_reduce(global_den, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(global_num, op=torch.distributed.ReduceOp.SUM)
    torch._assert_async(global_den > 0, "ALL_ZERO_ACTION_MASK")
    backward_loss = local_num * world / global_den.clamp_min(1)
    return backward_loss, global_num / global_den.clamp_min(1)
'''
if text.count(anchor)!=1: raise SystemExit('MASK_PATCH_ANCHOR_MISMATCH')
text=text.replace(anchor,helper)
old='''        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")
        loss_dict = {}
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        loss_dict["losses_after_forward"] = losses.clone()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone()

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict
'''
new='''        actions = self.prepare_action(batch)
        action_is_pad = batch.get("action_is_pad")
        loss_dict = {}
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        loss_dict["losses_after_forward"] = losses.clone()

        # Keep the model's internal action width, but exclude episode-boundary time padding
        # from both numerator and denominator. _reduce_action_losses also performs the
        # global valid-token reduction required by DDP for reduction="mean".
        losses = losses[:, :, : self.config.max_action_dim]
        valid = torch.ones(losses.shape[:2], dtype=torch.bool, device=losses.device) if action_is_pad is None else ~action_is_pad.to(device=losses.device, dtype=torch.bool)
        loss_dict["losses_after_in_ep_bound"] = losses * valid.unsqueeze(-1)
        loss_dict["losses_after_rm_padding"] = losses.clone()
        loss, report_loss = _reduce_action_losses(losses, action_is_pad, reduction)
        loss_dict["loss"] = report_loss.item()
        return loss, loss_dict
'''
if text.count(old)!=1: raise SystemExit('MASK_BLOCK_MISMATCH')
policy.write_text(text.replace(old,new),encoding='utf-8')

ft=factory.read_text(encoding='utf-8')
needle='policy = policy_cls.from_pretrained(**kwargs)'
if ft.count(needle)!=2: raise SystemExit('STRICT_LOAD_ANCHOR_MISMATCH')
factory.write_text(ft.replace(needle,'policy = policy_cls.from_pretrained(**kwargs, strict=True)'),encoding='utf-8')

# Resume 时只允许 device override；不得用当前 dataset stats/rename 覆盖 checkpoint processor state。
tr=trainer.read_text(encoding='utf-8')
old_resume='''    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }
'''
new_resume='''    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {"device_processor": {"device": device.type}}
        if not cfg.resume:
            processor_kwargs["preprocessor_overrides"]["normalizer_processor"] = {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            }
            processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
                "rename_map": cfg.rename_map
            }
            postprocessor_kwargs["postprocessor_overrides"] = {
                "unnormalizer_processor": {
                    "stats": dataset.meta.stats,
                    "features": policy.config.output_features,
                    "norm_map": policy.config.normalization_mapping,
                },
            }
'''
if tr.count(old_resume)!=1: raise SystemExit('STRICT_PROCESSOR_RESUME_ANCHOR_MISMATCH')
tr=tr.replace(old_resume,new_resume)

# Checkpoint 先写同文件系统 staging 目录，校验必需文件并 fsync，最后用目录 rename 提交。
# LEROBOT_STOP_AFTER_COMPLETE_CHECKPOINT_STEP 只在提交、last 更新和全 rank barrier 完成后生效。
import_anchor='''import dataclasses
import logging
import time
'''
import_replacement='''import dataclasses
import json
import logging
import os
import time
from pathlib import Path
'''
if tr.count(import_anchor)!=1: raise SystemExit('ATOMIC_IMPORT_ANCHOR_MISMATCH')
tr=tr.replace(import_anchor,import_replacement)

update_anchor='''

def update_policy(
'''
atomic_helpers='''

CHECKPOINT_COMPLETE_MARKER = "CHECKPOINT_COMPLETE.json"
REQUIRED_CHECKPOINT_PATHS = (
    Path("pretrained_model/config.json"),
    Path("pretrained_model/model.safetensors"),
    Path("pretrained_model/train_config.json"),
    Path("training_state/optimizer_param_groups.json"),
    Path("training_state/optimizer_state.safetensors"),
    Path("training_state/rng_state.safetensors"),
    Path("training_state/scheduler_state.json"),
    Path("training_state/training_step.json"),
)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _save_checkpoint_atomically(
    checkpoint_dir, step, cfg, policy, optimizer, scheduler, preprocessor, postprocessor
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    staging_dir = checkpoint_dir.with_name(f".{checkpoint_dir.name}.staging-{os.getpid()}")
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir.exists():
        raise FileExistsError(f"REFUSE_OVERWRITE_COMMITTED_CHECKPOINT {checkpoint_dir}")
    if staging_dir.exists():
        raise FileExistsError(f"STALE_CHECKPOINT_STAGING_DIR {staging_dir}")
    save_checkpoint(
        checkpoint_dir=staging_dir,
        step=step,
        cfg=cfg,
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )
    missing = [str(rel) for rel in REQUIRED_CHECKPOINT_PATHS if not (staging_dir / rel).is_file()]
    if missing:
        raise RuntimeError(f"INCOMPLETE_CHECKPOINT_STAGING missing={missing}")
    recorded_step = json.loads(
        (staging_dir / "training_state/training_step.json").read_text(encoding="utf-8")
    )["step"]
    if recorded_step != step:
        raise RuntimeError(f"CHECKPOINT_STEP_MISMATCH expected={step} actual={recorded_step}")
    marker_payload = {"schema": 1, "step": step, "total_steps": cfg.steps, "state": "COMPLETE"}
    marker = staging_dir / CHECKPOINT_COMPLETE_MARKER
    with marker.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(marker_payload, sort_keys=True) + "\\n")
        stream.flush()
        os.fsync(stream.fileno())
    for saved_file in sorted(staging_dir.rglob("*")):
        if saved_file.is_file() and saved_file != marker:
            _fsync_file(saved_file)
    _fsync_dir(staging_dir / "pretrained_model")
    _fsync_dir(staging_dir / "training_state")
    _fsync_dir(staging_dir)
    os.replace(staging_dir, checkpoint_dir)
    _fsync_dir(checkpoint_dir.parent)


def update_policy(
'''
if tr.count(update_anchor)!=1: raise SystemExit('ATOMIC_HELPER_ANCHOR_MISMATCH')
tr=tr.replace(update_anchor,atomic_helpers)

resume_state_anchor='''    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)
'''
resume_state_replacement=resume_state_anchor+'''
    stop_after_raw = os.environ.get("LEROBOT_STOP_AFTER_COMPLETE_CHECKPOINT_STEP", "").strip()
    try:
        stop_after_complete_checkpoint_step = int(stop_after_raw) if stop_after_raw else None
    except ValueError as exc:
        raise ValueError("LEROBOT_STOP_AFTER_COMPLETE_CHECKPOINT_STEP must be an integer") from exc
    if stop_after_complete_checkpoint_step is not None:
        if not cfg.save_checkpoint:
            raise ValueError("STOP_AFTER_CHECKPOINT_REQUIRES_SAVE_CHECKPOINT")
        if not (step < stop_after_complete_checkpoint_step <= cfg.steps):
            raise ValueError(
                f"STOP_AFTER_CHECKPOINT_OUT_OF_RANGE start={step} "
                f"stop={stop_after_complete_checkpoint_step} total={cfg.steps}"
            )
        if stop_after_complete_checkpoint_step % cfg.save_freq != 0:
            raise ValueError("STOP_AFTER_CHECKPOINT_MUST_ALIGN_WITH_SAVE_FREQ")
'''
if tr.count(resume_state_anchor)!=1: raise SystemExit('STOP_HOOK_CONFIG_ANCHOR_MISMATCH')
tr=tr.replace(resume_state_anchor,resume_state_replacement)

save_anchor='''                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
'''
save_replacement='''                _save_checkpoint_atomically(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
'''
if tr.count(save_anchor)!=1: raise SystemExit('ATOMIC_SAVE_ANCHOR_MISMATCH')
tr=tr.replace(save_anchor,save_replacement)

barrier_anchor='''            accelerator.wait_for_everyone()
        if cfg.env and is_eval_step:
'''
barrier_replacement='''            accelerator.wait_for_everyone()
        if stop_after_complete_checkpoint_step == step:
            if not (cfg.save_checkpoint and is_saving_step):
                raise RuntimeError("STOP_HOOK_REACHED_WITHOUT_CHECKPOINT")
            if is_main_process:
                committed_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                marker = committed_dir / CHECKPOINT_COMPLETE_MARKER
                marker_payload = json.loads(marker.read_text(encoding="utf-8"))
                if marker_payload != {"schema": 1, "step": step, "total_steps": cfg.steps, "state": "COMPLETE"}:
                    raise RuntimeError(f"INVALID_COMMITTED_CHECKPOINT_MARKER {marker}")
                logging.info(f"Stopping after complete checkpoint and barrier at step {step}")
            accelerator.wait_for_everyone()
            break
        if cfg.env and is_eval_step:
'''
if tr.count(barrier_anchor)!=1: raise SystemExit('STOP_HOOK_BARRIER_ANCHOR_MISMATCH')
tr=tr.replace(barrier_anchor,barrier_replacement)

trainer.write_text(tr,encoding='utf-8')
diff=subprocess.check_output(['git','-C',str(ROOT),'diff','--binary'])
Path('evidence/vendor.patch').write_bytes(diff)
digest=hashlib.sha256(diff).hexdigest()
Path('evidence/vendor-patch.sha256').write_text(digest+'  evidence/vendor.patch\n',encoding='utf-8')
print({'status':'PATCHED','patch_sha256':digest})
PY
cat > src/contracts.py <<'PY'
#!/usr/bin/env python3
import argparse,json,math,torch
from pathlib import Path

def formula_test():
    action=torch.tensor([[[1.,2.]]]); noise=torch.tensor([[[5.,-2.]]]); t=torch.tensor([.25])[:,None,None]
    x=t*noise+(1-t)*action; target=noise-action
    assert torch.allclose(x,torch.tensor([[[2.,1.]]]))
    assert torch.allclose(x-0.25*target,action)
    return {'formula':'OK','x':x.tolist(),'target':target.tolist()}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--formula-only',action='store_true')
    p.add_argument('--model-config',default='assets/runtime_bundle/config.json'); p.add_argument('--dataset-info',default='assets/dataset/meta/info.json')
    p.add_argument('--split',default='splits/episodes.json'); p.add_argument('--stats',default='artifacts/train_stats.json')
    p.add_argument('--processor-root',default='assets/runtime_bundle')
    a=p.parse_args(); out=formula_test()
    if not a.formula_only:
        mc=json.loads(Path(a.model_config).read_text()); di=json.loads(Path(a.dataset_info).read_text())
        split=json.loads(Path(a.split).read_text()); stats=json.loads(Path(a.stats).read_text())
        assert mc['type']=='smolvla' and mc['chunk_size']==50 and mc['max_action_dim']==32 and mc['empty_cameras']==0
        assert mc['output_features']['action']['shape']==[6]
        f=di['features']; assert f['action']['shape']==[6] and f['observation.state']['shape']==[6]
        assert di['codebase_version']=='v3.0' and di['total_episodes']==50 and di['total_frames']==11939
        cameras=sorted(k for k,v in f.items() if v['dtype']=='video')
        assert cameras==['observation.images.side','observation.images.up']
        groups=[set(split[k]) for k in ('train','dev','test')]
        assert groups[0].isdisjoint(groups[1]) and groups[0].isdisjoint(groups[2]) and groups[1].isdisjoint(groups[2])
        assert sorted(set.union(*groups))==list(range(50)) and [len(x) for x in groups]==[40,5,5]
        for key in ('observation.state','action'):
            assert key in stats and all(k in stats[key] for k in ('min','max','mean','std','count'))
            assert all(len(stats[key][k])==6 for k in ('min','max','mean','std'))
            assert all(math.isfinite(float(x)) for k in ('min','max','mean','std') for x in stats[key][k])
            assert all(float(x)>0 for x in stats[key]['std']) and int(stats[key]['count'][0])>0
        root=Path(a.processor_root)
        for name in ('policy_preprocessor.json','policy_postprocessor.json'):
            assert (root/name).is_file(),f'MISSING_PROCESSOR {name}'
        out|={'model_contract':'OK','dataset_contract':'OK','cameras':cameras,'rename_map':{
          'observation.images.up':'observation.image','observation.images.side':'observation.image2'},
          'episode_groups':{k:len(split[k]) for k in ('train','dev','test')},'stats_scope':'train-only',
          'processors':'present'}
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
PY
cat > src/inspect_assets.py <<'PY'
#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path

def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def tree(root):
    rows=[]
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root)
        if not p.is_file() or p.is_symlink() or '.cache' in rel.parts: continue
        if p.stat().st_size<200:
            head=p.read_bytes()[:200]
            if b'git-lfs.github.com/spec/v1' in head: raise SystemExit(f'LFS_POINTER_NOT_ASSET {p}')
        rows.append({'path':rel.as_posix(),'bytes':p.stat().st_size,'sha256':file_hash(p)})
    return rows
def digest(rows):
    raw=json.dumps(rows,sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(raw).hexdigest()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--lock',default='assets/LOCK.json')
    p.add_argument('--output',default='evidence/assets-manifest.json'); a=p.parse_args()
    lock=json.loads(Path(a.lock).read_text(encoding='utf-8')); out={'lock':lock,'trees':{}}
    for name,spec in lock['assets'].items():
        root=Path(spec['root'])
        missing=[x for x in spec['required_files'] if not (root/x).is_file()]
        if missing: raise SystemExit(f'MISSING_ASSET {name} {missing}')
        rows=tree(root); out['trees'][name]={'content_tree_sha256':digest(rows),'files':rows,
                                           'bytes':sum(x['bytes'] for x in rows)}
    dest=Path(a.output); dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({'status':'ASSETS_OK','tree_sha256':{k:v['content_tree_sha256'] for k,v in out['trees'].items()}}))
if __name__=='__main__': main()
PY
cat > src/lineage.py <<'PY'
#!/usr/bin/env python3
import hashlib,json,subprocess
from pathlib import Path
import torch

CODE_COMMIT='0b067df57d21d3a02d6c511f1609172fa39ac29b'
CODE_TREE='b5891d6333c4eb611bd3e81a4d13ed3c71210a6e'
DATASET_REVISION='f641879e22172be7e8161d5e6c1503c2d2feb657'
VLM_REVISION='7b375e1b73b11138ff12fe22c8f2822d8fe03467'
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def tensor_sha(t):
    x=t.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()
def tree_rows(root):
    root=Path(root); rows=[]
    if not root.is_dir(): raise RuntimeError(f'MISSING_TREE {root}')
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root)
        if '.cache' in rel.parts: continue
        if p.is_symlink(): raise RuntimeError(f'SYMLINK_NOT_FROZEN {p}')
        if not p.is_file(): continue
        if p.stat().st_size==0: raise RuntimeError(f'EMPTY_FILE {p}')
        if p.stat().st_size<200 and b'git-lfs.github.com/spec/v1' in p.read_bytes()[:200]:
            raise RuntimeError(f'LFS_POINTER_NOT_ASSET {p}')
        rows.append({'path':rel.as_posix(),'bytes':p.stat().st_size,'sha256':file_sha(p)})
    if not rows: raise RuntimeError(f'EMPTY_TREE {root}')
    return rows
def tree_digest(root): return canonical_sha(tree_rows(root))
def clone_tree(x):
    if torch.is_tensor(x): return x.detach().clone()
    if isinstance(x,dict): return {k:clone_tree(v) for k,v in x.items()}
    if isinstance(x,list): return [clone_tree(v) for v in x]
    if isinstance(x,tuple): return tuple(clone_tree(v) for v in x)
    return x
def value_record(x):
    if torch.is_tensor(x): return {'kind':'tensor','shape':list(x.shape),'dtype':str(x.dtype),'sha256':tensor_sha(x)}
    if isinstance(x,dict): return {'kind':'dict','items':{str(k):value_record(v) for k,v in sorted(x.items(),key=lambda z:str(z[0]))}}
    if isinstance(x,(list,tuple)): return {'kind':type(x).__name__,'items':[value_record(v) for v in x]}
    if x is None or isinstance(x,(str,int,float,bool)): return {'kind':type(x).__name__,'value':x}
    return {'kind':type(x).__name__,'value':str(x)}
def value_sha(x): return canonical_sha(value_record(x))
def processor_signature(pipe):
    rows=[]
    for i,step in enumerate(pipe.steps):
        state=step.state_dict() if hasattr(step,'state_dict') else {}
        rows.append({'index':i,'class':f'{type(step).__module__}.{type(step).__qualname__}',
          'config':value_record(step.get_config() if hasattr(step,'get_config') else {}),
          'state':value_record(state)})
    payload={'pipeline_class':f'{type(pipe).__module__}.{type(pipe).__qualname__}','steps':rows}
    return {'sha256':canonical_sha(payload),**payload}
def fixture_signature(pre,post,reloaded_pre,reloaded_post,raw_batch,fixture_id):
    left=pre(clone_tree(raw_batch)); right=reloaded_pre(clone_tree(raw_batch))
    left_sha,right_sha=value_sha(left),value_sha(right)
    if left_sha!=right_sha: raise RuntimeError('REAL_FIXTURE_PRE_RELOAD_MISMATCH')
    if not isinstance(left,dict) or 'action' not in left: raise RuntimeError('REAL_FIXTURE_ACTION_MISSING')
    post_input=clone_tree(left['action'])
    left_post=post(clone_tree(post_input)); right_post=reloaded_post(clone_tree(post_input))
    left_post_sha,right_post_sha=value_sha(left_post),value_sha(right_post)
    if left_post_sha!=right_post_sha: raise RuntimeError('REAL_FIXTURE_POST_RELOAD_MISMATCH')
    raw_subset={k:raw_batch[k] for k in ('observation.state','action') if k in raw_batch}
    return {'fixture_id':fixture_id,'raw_state_action_sha256':value_sha(raw_subset),
      'pre_output_sha256':left_sha,'reloaded_pre_output_sha256':right_sha,
      'post_input_sha256':value_sha(post_input),'post_output_sha256':left_post_sha,
      'reloaded_post_output_sha256':right_post_sha,'reload_exact':True}
def processor_files(root):
    root=Path(root); out=[]
    for p in root.rglob('*'):
        if not p.is_file() or p.is_symlink(): continue
        rel=p.relative_to(root)
        if any(x.startswith(('policy_preprocessor','policy_postprocessor')) for x in rel.parts): out.append(p)
    return sorted(out)
def verify_assets(lock_path,manifest_path):
    lock=json.loads(Path(lock_path).read_text(encoding='utf-8'))
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if manifest.get('lock')!=lock: raise RuntimeError('ASSET_LOCK_MANIFEST_MISMATCH')
    actual={}
    for name,spec in lock['assets'].items():
        rows=tree_rows(spec['root']); recorded=manifest.get('trees',{}).get(name,{})
        digest=canonical_sha(rows)
        if rows!=recorded.get('files') or digest!=recorded.get('content_tree_sha256'):
            raise RuntimeError(f'ASSET_TREE_CHANGED {name}')
        actual[name]=digest
    return lock,actual
def verify_split_stats(split_path,stats_path,stats_lineage_path,asset_manifest_path,dataset_tree,dataset_revision):
    split=json.loads(Path(split_path).read_text(encoding='utf-8'))
    groups=[set(split[k]) for k in ('train','dev','test')]
    if ([len(x) for x in groups]!=[40,5,5] or sorted(set.union(*groups))!=list(range(50)) or
        any(groups[i]&groups[j] for i,j in ((0,1),(0,2),(1,2)))):
        raise RuntimeError('EPISODE_SPLIT_INVALID')
    source=split.get('source',{})
    if source.get('revision')!=dataset_revision or source.get('content_tree_sha256')!=dataset_tree:
        raise RuntimeError('SPLIT_SOURCE_LINEAGE_MISMATCH')
    lineage=json.loads(Path(stats_lineage_path).read_text(encoding='utf-8'))
    expected={'dataset_revision':dataset_revision,'dataset_content_tree_sha256':dataset_tree,
      'asset_manifest_sha256':file_sha(asset_manifest_path),'split_sha256':file_sha(split_path),
      'stats_sha256':file_sha(stats_path),'train_episodes':split['train']}
    if any(lineage.get(k)!=v for k,v in expected.items()): raise RuntimeError('TRAIN_STATS_LINEAGE_MISMATCH')
    return split,lineage
def verify_processor_contract(model_root,contract_path,contract_sha_path):
    root=Path(model_root); contract_path=Path(contract_path); sha_path=Path(contract_sha_path)
    expected=sha_path.read_text(encoding='utf-8').strip().split()[0]
    if len(expected)!=64 or file_sha(contract_path)!=expected: raise RuntimeError('PROCESSOR_CONTRACT_SHA_MISMATCH')
    contract=json.loads(contract_path.read_text(encoding='utf-8'))
    if contract.get('schema')!=2 or contract.get('fixture',{}).get('reload_exact') is not True:
        raise RuntimeError('PROCESSOR_CONTRACT_SCHEMA_MISMATCH')
    actual={p.relative_to(root).as_posix():file_sha(p) for p in processor_files(root)}
    if not actual or actual!=contract.get('processor_files'): raise RuntimeError('PROCESSOR_FILE_CONTRACT_MISMATCH')
    return contract
def verify_processor_objects(pre,post,contract):
    ps=processor_signature(pre); qs=processor_signature(post)
    if ps['sha256']!=contract.get('preprocessor_signature_sha256'): raise RuntimeError('PREPROCESSOR_STATE_MISMATCH')
    if qs['sha256']!=contract.get('postprocessor_signature_sha256'): raise RuntimeError('POSTPROCESSOR_STATE_MISMATCH')
    return ps,qs
def verify_vendor(repo,patch):
    repo=Path(repo); patch=Path(patch)
    head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    tree=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD^{tree}'],text=True).strip()
    diff=subprocess.check_output(['git','-C',str(repo),'diff','--binary'])
    if head!=CODE_COMMIT or tree!=CODE_TREE or hashlib.sha256(diff).hexdigest()!=file_sha(patch):
        raise RuntimeError('CODE_OR_PATCH_LINEAGE_MISMATCH')
    return {'commit':head,'tree':tree,'patch_sha256':file_sha(patch)}
def verify_frozen_lineage(model_root,model_manifest_path,lock_path,asset_manifest_path,split_path,stats_path,
                          stats_lineage_path,processor_contract_path,processor_contract_sha_path,
                          vendor,patch,approval_file,dataset_revision):
    lock,trees=verify_assets(lock_path,asset_manifest_path)
    if lock['assets']['dataset']['revision']!=dataset_revision or dataset_revision!=DATASET_REVISION:
        raise RuntimeError('DATASET_REVISION_MISMATCH')
    if lock['assets']['smolvlm2']['revision']!=VLM_REVISION: raise RuntimeError('VLM_REVISION_MISMATCH')
    split,_=verify_split_stats(split_path,stats_path,stats_lineage_path,asset_manifest_path,
                               trees['dataset'],dataset_revision)
    contract=verify_processor_contract(model_root,processor_contract_path,processor_contract_sha_path)
    expected={'asset_manifest_sha256':file_sha(asset_manifest_path),'split_sha256':file_sha(split_path),
      'stats_sha256':file_sha(stats_path),'stats_lineage_sha256':file_sha(stats_lineage_path),
      'base_tree_sha256':trees['smolvla_base'],'dataset_tree_sha256':trees['dataset'],
      'nested_vlm_tree_sha256':trees['smolvlm2'],'model_use_approval_sha256':file_sha(approval_file),
      'code_commit':CODE_COMMIT,'code_tree':CODE_TREE,'patch_sha256':file_sha(patch),
      'model_config_sha256':file_sha(Path(model_root)/'config.json')}
    if any(contract.get(k)!=v for k,v in expected.items()): raise RuntimeError('PROCESSOR_LINEAGE_MISMATCH')
    cfg=json.loads((Path(model_root)/'config.json').read_text(encoding='utf-8'))
    expected_vlm=Path(lock['assets']['smolvlm2']['root']).resolve()
    if Path(cfg.get('vlm_model_name','')).resolve()!=expected_vlm: raise RuntimeError('NESTED_VLM_PATH_MISMATCH')
    code=verify_vendor(vendor,patch)
    policy_manifest=json.loads(Path(model_manifest_path).read_text(encoding='utf-8'))
    model_tree=tree_digest(model_root)
    if policy_manifest.get('model_tree_sha256')!=model_tree: raise RuntimeError('MODEL_TREE_CHANGED')
    if policy_manifest.get('processor_contract_sha256')!=file_sha(processor_contract_path):
        raise RuntimeError('MODEL_PROCESSOR_CONTRACT_MISMATCH')
    return {'asset_manifest_sha256':file_sha(asset_manifest_path),'asset_trees':trees,
      'split_sha256':file_sha(split_path),'stats_sha256':file_sha(stats_path),
      'stats_lineage_sha256':file_sha(stats_lineage_path),'processor_contract_sha256':file_sha(processor_contract_path),
      'model_manifest_sha256':file_sha(model_manifest_path),'model_tree_sha256':model_tree,
      'nested_vlm_path':str(expected_vlm),**code},split,contract
PY
cat > src/make_split_stats.py <<'PY'
#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

def dump(path,obj):
    raw=(json.dumps(obj,sort_keys=True,indent=2)+'\n').encode(); Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_bytes(raw); return hashlib.sha256(raw).hexdigest()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',default='assets/dataset')
    p.add_argument('--split-out',default='splits/episodes.json'); p.add_argument('--stats-out',default='artifacts/train_stats.json')
    p.add_argument('--asset-manifest',default='evidence/assets-manifest.json')
    p.add_argument('--stats-lineage-out',default='artifacts/train_stats.lineage.json')
    p.add_argument('--dataset-revision',default='f641879e22172be7e8161d5e6c1503c2d2feb657')
    a=p.parse_args(); manifest=json.loads(Path(a.asset_manifest).read_text(encoding='utf-8'))
    dataset_tree=manifest['trees']['dataset']['content_tree_sha256']
    split={'schema':2,'group_key':'episode_index','source':{'repo':'lerobot/svla_so101_pickplace',
      'revision':a.dataset_revision,'content_tree_sha256':dataset_tree},
      'train':list(range(40)),'dev':list(range(40,45)),'test':list(range(45,50))}
    episode=[]; values={k:[] for k in ('observation.state','action')}
    for path in sorted(Path(a.dataset).glob('data/**/*.parquet')):
        table=pq.read_table(path,columns=['episode_index','observation.state','action'])
        episode.extend(table['episode_index'].to_pylist())
        for key in values: values[key].extend(table[key].to_pylist())
    keep=np.isin(np.asarray(episode,dtype=np.int64),np.asarray(split['train']))
    if not keep.any(): raise SystemExit('NO_TRAIN_ROWS')
    stats={}
    for key,rows in values.items():
        x=np.asarray(rows,dtype=np.float64)[keep]
        if x.ndim!=2 or x.shape[1]!=6 or not np.isfinite(x).all(): raise SystemExit(f'BAD_STATS_INPUT {key} {x.shape}')
        stats[key]={'min':x.min(0).tolist(),'max':x.max(0).tolist(),'mean':x.mean(0).tolist(),
                    'std':x.std(0).tolist(),'count':[int(x.shape[0])]}
        if min(stats[key]['std'])<=0: raise SystemExit(f'ZERO_STD {key}')
    split_sha=dump(a.split_out,split); stats_sha=dump(a.stats_out,stats)
    lineage={'schema':1,'dataset_revision':a.dataset_revision,'dataset_content_tree_sha256':dataset_tree,
      'asset_manifest_sha256':sha(a.asset_manifest),'split_sha256':split_sha,'stats_sha256':stats_sha,
      'train_episodes':split['train'],'train_rows':int(keep.sum()),'heldout_rows':int((~keep).sum())}
    lineage_sha=dump(a.stats_lineage_out,lineage)
    print(json.dumps({'status':'SPLIT_STATS_OK','train_rows':int(keep.sum()),'heldout_rows':int((~keep).sum()),
                      'split_sha256':split_sha,'stats_sha256':stats_sha,'stats_lineage_sha256':lineage_sha}))
if __name__=='__main__': main()
PY
cat > src/prepare_bundle.py <<'PY'
#!/usr/bin/env python3
import argparse,json,shutil
from pathlib import Path
from torch.utils.data import DataLoader
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset,LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lineage import (CODE_COMMIT,CODE_TREE,file_sha,fixture_signature,processor_files,processor_signature,
                     verify_assets,verify_split_stats,verify_vendor)

RENAME={'observation.images.up':'observation.image','observation.images.side':'observation.image2'}
def scalar(batch,key,default):
    value=batch.get(key)
    return default if value is None else int(value.reshape(-1)[0].item())
def real_train_fixture(cfg,dataset_root,dataset_revision,split_path):
    split=json.loads(Path(split_path).read_text(encoding='utf-8')); episode=split['train'][0]
    meta=LeRobotDatasetMetadata('lerobot/svla_so101_pickplace',root=dataset_root,revision=dataset_revision)
    delta=resolve_delta_timestamps(cfg,meta)
    ds=LeRobotDataset('lerobot/svla_so101_pickplace',root=dataset_root,revision=dataset_revision,
      episodes=[episode],delta_timestamps=delta,download_videos=True)
    batch=next(iter(DataLoader(ds,batch_size=1,shuffle=False,num_workers=0)))
    fixture_id={'episode_index':scalar(batch,'episode_index',episode),'frame_index':scalar(batch,'frame_index',-1),
                'sample_index':scalar(batch,'index',0),'dataset_subset_index':0}
    if fixture_id['episode_index']!=episode or fixture_id['frame_index']<0: raise RuntimeError('BAD_REAL_TRAIN_FIXTURE_ID')
    return batch,fixture_id
def main():
    p=argparse.ArgumentParser(); p.add_argument('--base',default='assets/smolvla_base'); p.add_argument('--vlm',default='assets/smolvlm2')
    p.add_argument('--dataset',default='assets/dataset'); p.add_argument('--dataset-revision',default='f641879e22172be7e8161d5e6c1503c2d2feb657')
    p.add_argument('--split',default='splits/episodes.json'); p.add_argument('--stats',default='artifacts/train_stats.json')
    p.add_argument('--stats-lineage',default='artifacts/train_stats.lineage.json'); p.add_argument('--lock',default='assets/LOCK.json')
    p.add_argument('--asset-manifest',default='evidence/assets-manifest.json'); p.add_argument('--vendor',default='vendor/lerobot')
    p.add_argument('--patch',default='evidence/vendor.patch'); p.add_argument('--approval-file',required=True)
    p.add_argument('--output',default='assets/runtime_bundle'); a=p.parse_args()
    base,vlm,out=Path(a.base),Path(a.vlm),Path(a.output)
    if out.exists(): raise SystemExit(f'REFUSE_OVERWRITE {out}')
    lock,trees=verify_assets(a.lock,a.asset_manifest)
    split,_=verify_split_stats(a.split,a.stats,a.stats_lineage,a.asset_manifest,trees['dataset'],a.dataset_revision)
    code=verify_vendor(a.vendor,a.patch)
    for name in ('config.json','model.safetensors','train_config.json'):
        if not (base/name).is_file(): raise SystemExit(f'MISSING_BASE {name}')
    for name in ('config.json','model.safetensors','processor_config.json','preprocessor_config.json','tokenizer.json','tokenizer_config.json'):
        if not (vlm/name).is_file(): raise SystemExit(f'MISSING_VLM {name}')
    out.mkdir(parents=True); [shutil.copy2(base/name,out/name) for name in ('config.json','model.safetensors','train_config.json')]
    cfg=SmolVLAConfig.from_pretrained(str(out),local_files_only=True); cfg.vlm_model_name=str(vlm.resolve()); cfg.device='cuda'
    cfg.save_pretrained(out)
    stats=json.loads(Path(a.stats).read_text(encoding='utf-8'))
    pre,post=make_pre_post_processors(cfg,dataset_stats=stats)
    found=False
    for step in pre.steps:
        if hasattr(step,'rename_map'): step.rename_map=dict(RENAME); found=True
    if not found: raise SystemExit('RENAME_PROCESSOR_NOT_FOUND')
    pre.save_pretrained(out,config_filename='policy_preprocessor.json')
    post.save_pretrained(out,config_filename='policy_postprocessor.json')
    # A second construction must use only serialized artifacts; no dataset-stat override.
    pre2,post2=make_pre_post_processors(cfg,pretrained_path=str(out))
    raw,fixture_id=real_train_fixture(cfg,a.dataset,a.dataset_revision,a.split)
    pre_sig,post_sig=processor_signature(pre),processor_signature(post)
    fixture=fixture_signature(pre,post,pre2,post2,raw,fixture_id)
    contract={'schema':2,'base_revision':lock['assets']['smolvla_base']['revision'],
      'vlm_revision':lock['assets']['smolvlm2']['revision'],'dataset_revision':a.dataset_revision,
      'vlm_local_path':str(vlm.resolve()),'rename_map':RENAME,'model_config_sha256':file_sha(out/'config.json'),
      'asset_manifest_sha256':file_sha(a.asset_manifest),'base_tree_sha256':trees['smolvla_base'],
      'nested_vlm_tree_sha256':trees['smolvlm2'],'dataset_tree_sha256':trees['dataset'],
      'split_sha256':file_sha(a.split),'stats_sha256':file_sha(a.stats),'stats_lineage_sha256':file_sha(a.stats_lineage),
      'model_use_approval_sha256':file_sha(a.approval_file),'code_commit':CODE_COMMIT,'code_tree':CODE_TREE,
      'patch_sha256':file_sha(a.patch),'preprocessor_signature_sha256':pre_sig['sha256'],
      'postprocessor_signature_sha256':post_sig['sha256'],'preprocessor_signature':pre_sig,
      'postprocessor_signature':post_sig,'fixture':fixture,
      'processor_files':{p.relative_to(out).as_posix():file_sha(p) for p in processor_files(out)}}
    contract_path=out/'processor_contract.json'; contract_path.write_text(json.dumps(contract,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    digest=file_sha(contract_path); (out/'processor_contract.sha256').write_text(digest+'  processor_contract.json\n',encoding='utf-8')
    print(json.dumps({'status':'BUNDLE_OK','processor_contract_sha256':digest,'fixture':fixture_id}))
if __name__=='__main__': main()
PY
cat > src/seal_processor_contract.py <<'PY'
#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from torch.utils.data import DataLoader
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset,LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lineage import file_sha,fixture_signature,processor_files,processor_signature,verify_processor_contract

def scalar(batch,key,default):
    value=batch.get(key); return default if value is None else int(value.reshape(-1)[0].item())
def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--source-root',default='assets/runtime_bundle')
    p.add_argument('--dataset',default='assets/dataset'); p.add_argument('--dataset-revision',default='f641879e22172be7e8161d5e6c1503c2d2feb657')
    p.add_argument('--split',default='splits/episodes.json'); a=p.parse_args(); root=Path(a.model); source=Path(a.source_root)
    source_contract=verify_processor_contract(source,source/'processor_contract.json',source/'processor_contract.sha256')
    if (root/'processor_contract.json').exists() or (root/'processor_contract.sha256').exists(): raise SystemExit('REFUSE_RESEAL')
    cfg=SmolVLAConfig.from_pretrained(str(root),local_files_only=True); cfg.device='cuda'
    pre,post=make_pre_post_processors(cfg,pretrained_path=str(root)); pre2,post2=make_pre_post_processors(cfg,pretrained_path=str(root))
    split=json.loads(Path(a.split).read_text(encoding='utf-8')); episode=split['train'][0]
    meta=LeRobotDatasetMetadata('lerobot/svla_so101_pickplace',root=a.dataset,revision=a.dataset_revision)
    ds=LeRobotDataset('lerobot/svla_so101_pickplace',root=a.dataset,revision=a.dataset_revision,episodes=[episode],
      delta_timestamps=resolve_delta_timestamps(cfg,meta),download_videos=True)
    raw=next(iter(DataLoader(ds,batch_size=1,shuffle=False,num_workers=0)))
    fixture_id={'episode_index':scalar(raw,'episode_index',episode),'frame_index':scalar(raw,'frame_index',-1),
                'sample_index':scalar(raw,'index',0),'dataset_subset_index':0}
    fixture=fixture_signature(pre,post,pre2,post2,raw,fixture_id)
    pre_sig,post_sig=processor_signature(pre),processor_signature(post)
    for key,actual in (('preprocessor_signature_sha256',pre_sig['sha256']),('postprocessor_signature_sha256',post_sig['sha256'])):
        if source_contract[key]!=actual: raise SystemExit(f'CHECKPOINT_{key.upper()}_MISMATCH')
    for key in ('fixture_id','raw_state_action_sha256','pre_output_sha256','post_input_sha256','post_output_sha256'):
        if source_contract['fixture'][key]!=fixture[key]: raise SystemExit(f'CHECKPOINT_FIXTURE_MISMATCH {key}')
    contract=dict(source_contract); contract.update({'model_config_sha256':file_sha(root/'config.json'),
      'preprocessor_signature':pre_sig,'postprocessor_signature':post_sig,'fixture':fixture,
      'processor_files':{x.relative_to(root).as_posix():file_sha(x) for x in processor_files(root)}})
    path=root/'processor_contract.json'; path.write_text(json.dumps(contract,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    digest=file_sha(path); (root/'processor_contract.sha256').write_text(digest+'  processor_contract.json\n',encoding='utf-8')
    print(json.dumps({'status':'PROCESSOR_CONTRACT_SEALED','sha256':digest,'fixture':fixture_id}))
if __name__=='__main__': main()
PY
cat > src/audit_policy.py <<'PY'
#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lineage import file_sha,processor_files,tree_digest,verify_processor_contract,verify_processor_objects

def tensor_sha(t): return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--output',required=True)
    p.add_argument('--reference',default=''); p.add_argument('--processor-contract',default='processor_contract.json')
    p.add_argument('--contract-sha',default='processor_contract.sha256'); a=p.parse_args(); root=Path(a.model)
    for name in ('config.json','model.safetensors','policy_preprocessor.json','policy_postprocessor.json'):
        if not (root/name).is_file(): raise SystemExit(f'MISSING_RUNTIME_FILE {root/name}')
    # Contract digest and every serialized processor file are checked before any config/pipeline/model construction.
    contract_path=root/a.processor_contract; contract_sha_path=root/a.contract_sha
    contract=verify_processor_contract(root,contract_path,contract_sha_path)
    cfg=SmolVLAConfig.from_pretrained(str(root),local_files_only=True); cfg.device='cuda'
    policy=SmolVLAPolicy.from_pretrained(str(root),config=cfg,strict=True,local_files_only=True)
    pre,post=make_pre_post_processors(cfg,pretrained_path=str(root))
    pre_sig,post_sig=verify_processor_objects(pre,post,contract)
    params=[]
    for name,value in policy.named_parameters():
        params.append({'name':name,'shape':list(value.shape),'numel':value.numel(),'dtype':str(value.dtype),
                       'requires_grad':value.requires_grad,'sha256':tensor_sha(value)})
    row={'schema':2,'model':str(root),'strict_load':True,'processor_reload':True,
         'processor_contract_sha256':file_sha(contract_path),'model_tree_sha256':tree_digest(root),
         'processor_files':{x.relative_to(root).as_posix():file_sha(x) for x in processor_files(root)},
         'preprocessor_signature_sha256':pre_sig['sha256'],'postprocessor_signature_sha256':post_sig['sha256'],
         'total_numel':sum(x['numel'] for x in params),'trainable_numel':sum(x['numel'] for x in params if x['requires_grad']),
         'trainable_names':[x['name'] for x in params if x['requires_grad']],'parameters':params}
    if a.reference:
        ref=json.loads(Path(a.reference).read_text(encoding='utf-8'))
        if row['trainable_names']!=ref['trainable_names']: raise SystemExit('TRAINABLE_SET_CHANGED')
        old={x['name']:x['sha256'] for x in ref['parameters'] if not x['requires_grad']}
        new={x['name']:x['sha256'] for x in row['parameters'] if not x['requires_grad']}
        if old!=new: raise SystemExit('FROZEN_PARAMETER_CHANGED')
        row['frozen_equal_to_reference']=True
    raw=(json.dumps(row,sort_keys=True,indent=2)+'\n').encode(); Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_bytes(raw)
    print(json.dumps({'status':'POLICY_AUDIT_OK','manifest_sha256':hashlib.sha256(raw).hexdigest(),
                      'total_numel':row['total_numel'],'trainable_numel':row['trainable_numel']}))
if __name__=='__main__': main()
PY
cat > src/compare_resume.py <<'PY'
#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from lineage import file_sha

STATE_FILES=('optimizer_param_groups.json','optimizer_state.safetensors','scheduler_state.json',
             'rng_state.safetensors','training_step.json')
def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def keyed_params(manifest): return {x['name']:x for x in manifest['parameters']}
def find_values(obj,key):
    out=[]
    if isinstance(obj,dict):
        for k,v in obj.items():
            if k==key: out.append(v)
            out.extend(find_values(v,key))
    elif isinstance(obj,list):
        for v in obj: out.extend(find_values(v,key))
    return out
def main():
    p=argparse.ArgumentParser(); p.add_argument('--continuous-checkpoint',required=True); p.add_argument('--resumed-checkpoint',required=True)
    p.add_argument('--continuous-manifest',required=True); p.add_argument('--resumed-manifest',required=True)
    p.add_argument('--continuous-eval',required=True); p.add_argument('--resumed-eval',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); ca,ra=Path(a.continuous_checkpoint),Path(a.resumed_checkpoint)
    cs,rs=ca/'training_state',ra/'training_state'
    for root in (ca,ra):
        for rel in [Path('pretrained_model/model.safetensors'),*[Path('training_state')/x for x in STATE_FILES]]:
            if not (root/rel).is_file(): raise SystemExit(f'MISSING_RESUME_ARTIFACT {root/rel}')
    cm,rm=read_json(a.continuous_manifest),read_json(a.resumed_manifest); cp,rp=keyed_params(cm),keyed_params(rm)
    if cp.keys()!=rp.keys(): raise SystemExit('PARAMETER_NAME_SET_MISMATCH')
    trainable=[n for n in cp if cp[n]['requires_grad']]
    if trainable!=[n for n in rp if rp[n]['requires_grad']]: raise SystemExit('TRAINABLE_SET_MISMATCH')
    trainable_diff=[n for n in trainable if cp[n]['sha256']!=rp[n]['sha256']]
    frozen_diff=[n for n in cp if not cp[n]['requires_grad'] and cp[n]['sha256']!=rp[n]['sha256']]
    file_state={name:{'continuous_sha256':file_sha(cs/name),'resumed_sha256':file_sha(rs/name),
                      'exact_equal':file_sha(cs/name)==file_sha(rs/name)} for name in STATE_FILES}
    cstep,rstep=read_json(cs/'training_step.json')['step'],read_json(rs/'training_step.json')['step']
    clr=find_values(read_json(cs/'optimizer_param_groups.json'),'lr'); rlr=find_values(read_json(rs/'optimizer_param_groups.json'),'lr')
    ce,re=read_json(a.continuous_eval),read_json(a.resumed_eval)
    eval_control_equal=(ce['ordered_ids_sha256']==re['ordered_ids_sha256'] and ce['fixed_noise_seed']==re['fixed_noise_seed']
      and ce['heldout_split']==re['heldout_split'] and ce['dataset_revision']==re['dataset_revision'])
    if not eval_control_equal: raise SystemExit('FIXED_HELDOUT_OR_NOISE_CONTROL_MISMATCH')
    scaler_files=list(cs.glob('*scaler*'))+list(ra.glob('*scaler*'))
    sampler_files=list(cs.glob('*sampler*'))+list(cs.glob('*dataloader*'))+list(ra.glob('*sampler*'))+list(ra.glob('*dataloader*'))
    limitations=[]
    if not sampler_files: limitations.append({'code':'DATA-ORDER_RESUME-LIMITATION','reason':'checkpoint has no sampler permutation/cursor or dataloader offset'})
    if not scaler_files: limitations.append({'code':'SCALER_RESUME-LIMITATION','reason':'checkpoint has no Accelerator GradScaler state'})
    comparisons={'trainable_parameter_exact_equal':not trainable_diff,'trainable_parameter_differences':trainable_diff,
      'frozen_parameter_exact_equal':not frozen_diff,'frozen_parameter_differences':frozen_diff,
      'optimizer_state_exact_equal':file_state['optimizer_state.safetensors']['exact_equal'],
      'optimizer_param_groups_exact_equal':file_state['optimizer_param_groups.json']['exact_equal'],
      'scheduler_state_exact_equal':file_state['scheduler_state.json']['exact_equal'],
      'rng_state_exact_equal':file_state['rng_state.safetensors']['exact_equal'],
      'step_equal_and_200':cstep==rstep==200,'continuous_lr':clr,'resumed_lr':rlr,'lr_exact_equal':clr==rlr,
      'fixed_eval_controls_equal':True,'fixed_prediction_exact_equal':ce['prediction_sha256']==re['prediction_sha256'],
      'continuous_prediction_sha256':ce['prediction_sha256'],'resumed_prediction_sha256':re['prediction_sha256']}
    exact=all(v for k,v in comparisons.items() if k.endswith('_equal') or k=='step_equal_and_200')
    out={'schema':1,'status':'EQUIVALENT' if exact and not limitations else 'RESUME-LIMITATION',
      'continuous_checkpoint':str(ca.resolve()),'resumed_checkpoint':str(ra.resolve()),
      'state_files':file_state,'comparisons':comparisons,'limitations':limitations,
      'equivalence_claim_allowed':bool(exact and not limitations)}
    dest=Path(a.output); dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,sort_keys=True))
if __name__=='__main__': main()
PY
cat > src/test_action_mask.py <<'PY'
#!/usr/bin/env python3
import inspect,math,os,torch,torch.distributed as dist
from lerobot.policies.factory import make_policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy,_reduce_action_losses

world=int(os.getenv('WORLD_SIZE','1')); rank=int(os.getenv('RANK','0'))
if world>1: dist.init_process_group('gloo')
try:
    values=torch.tensor([[[1.,3.],[9999.,9999.]]] if rank==0 else [[[5.,7.],[9999.,9999.]]],requires_grad=True)
    pad=torch.tensor([[False,True]])
    loss,report=_reduce_action_losses(values,pad,'mean')
    expected=2.0 if world==1 else 4.0
    assert math.isclose(float(report),expected), (report,expected)
    changed=values.detach().clone(); changed[:,1]=float(rank+1)*123456
    _,report2=_reduce_action_losses(changed,pad,'mean')
    assert torch.equal(report,report2),'padding values changed masked loss'
    try: _reduce_action_losses(values,torch.ones_like(pad),'mean')
    except RuntimeError as exc: assert 'ALL_ZERO_ACTION_MASK' in str(exc)
    else: raise AssertionError('all-zero action mask was accepted')
    source=inspect.getsource(SmolVLAPolicy.forward); factory=inspect.getsource(make_policy)
    assert 'batch.get("action_is_pad")' in source and 'actions_id_pad' not in source
    assert factory.count('strict=True')>=2
    if rank==0: print({'status':'ACTION_MASK_REGRESSION_OK','world':world,'global_report':float(report)})
finally:
    if dist.is_initialized(): dist.destroy_process_group()
PY
cat > src/offline_eval.py <<'PY'
#!/usr/bin/env python3
import argparse,json,math,statistics,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Subset
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset,LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lineage import canonical_sha,tensor_sha,verify_frozen_lineage,verify_processor_objects

def pct(x,q):
    x=sorted(x); return x[max(0,math.ceil(q*len(x))-1)]
def scalar(batch,key,default):
    value=batch.get(key); return default if value is None else int(value.reshape(-1)[0].item())
def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--dataset-root',default='assets/dataset')
    p.add_argument('--dataset-revision',default='f641879e22172be7e8161d5e6c1503c2d2feb657')
    p.add_argument('--split-file',default='splits/episodes.json'); p.add_argument('--split',choices=['dev','test'],default='test')
    p.add_argument('--samples',type=int,default=32); p.add_argument('--seed',type=int,default=20260903); p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--stats',default='artifacts/train_stats.json'); p.add_argument('--stats-lineage',default='artifacts/train_stats.lineage.json')
    p.add_argument('--lock',default='assets/LOCK.json'); p.add_argument('--asset-manifest',default='evidence/assets-manifest.json')
    p.add_argument('--processor-contract',default='processor_contract.json'); p.add_argument('--contract-sha',default='processor_contract.sha256')
    p.add_argument('--model-manifest',required=True); p.add_argument('--vendor',default='vendor/lerobot')
    p.add_argument('--patch',default='evidence/vendor.patch'); p.add_argument('--approval-file',required=True)
    p.add_argument('--output',required=True); a=p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('CUDA_REQUIRED')
    device=torch.device('cuda',0); model_path=Path(a.model)
    for name in ('config.json','model.safetensors','policy_preprocessor.json','policy_postprocessor.json'):
        if not (model_path/name).is_file(): raise SystemExit(f'MISSING_STRICT_ARTIFACT {model_path/name}')
    contract_path=model_path/a.processor_contract; contract_sha_path=model_path/a.contract_sha
    frozen,split,contract=verify_frozen_lineage(model_path,a.model_manifest,a.lock,a.asset_manifest,a.split_file,a.stats,
      a.stats_lineage,contract_path,contract_sha_path,a.vendor,a.patch,a.approval_file,a.dataset_revision)
    lock=json.loads(Path(a.lock).read_text(encoding='utf-8'))
    if Path(a.dataset_root).resolve()!=Path(lock['assets']['dataset']['root']).resolve():
        raise SystemExit('EVAL_DATASET_ROOT_NOT_FROZEN_ASSET')
    # Everything above is file/digest validation and happens before constructing config, processor, or policy.
    cfg=SmolVLAConfig.from_pretrained(str(model_path),local_files_only=True); cfg.device=str(device)
    policy=SmolVLAPolicy.from_pretrained(str(model_path),config=cfg,strict=True,local_files_only=True).eval()
    pre,post=make_pre_post_processors(cfg,pretrained_path=str(model_path))
    verify_processor_objects(pre,post,contract)
    meta=LeRobotDatasetMetadata('lerobot/svla_so101_pickplace',root=a.dataset_root,revision=a.dataset_revision)
    delta=resolve_delta_timestamps(policy.config,meta)
    dataset=LeRobotDataset('lerobot/svla_so101_pickplace',root=a.dataset_root,revision=a.dataset_revision,
                           episodes=split[a.split],delta_timestamps=delta,download_videos=True)
    n=min(a.samples,len(dataset))
    if n <= 0: raise SystemExit('EMPTY_HELDOUT_SPLIT')
    ids=[round(i*(len(dataset)-1)/max(n-1,1)) for i in range(n)]
    raw_batches=list(DataLoader(Subset(dataset,ids),batch_size=1,shuffle=False,num_workers=0))
    sse=0.0; scalars=0; endpoints=[]; jerks=[]; ordered_ids=[]; prediction_records=[]
    for i,(dataset_index,batch) in enumerate(zip(ids,raw_batches)):
        identity={'episode_index':scalar(batch,'episode_index',-1),'frame_index':scalar(batch,'frame_index',-1),
                  'sample_index':scalar(batch,'index',dataset_index),'dataset_subset_index':dataset_index}
        if identity['episode_index'] not in split[a.split] or identity['frame_index']<0: raise SystemExit('HELDOUT_ID_CONTRACT_FAILED')
        ordered_ids.append(identity)
        raw_action=batch['action'].float(); pad=batch.get('action_is_pad')
        if pad is None: raise SystemExit('ACTION_PAD_MASK_MISSING')
        valid=~pad.bool(); processed=pre(batch)
        g=torch.Generator(device='cpu').manual_seed(a.seed+i)
        noise=torch.randn((1,policy.config.chunk_size,policy.config.max_action_dim),generator=g).to(device)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            pred=policy.predict_action_chunk(processed,noise=noise)
        pred=post(pred).float().cpu(); prediction_records.append({'id':identity,'shape':list(pred.shape),
          'dtype':str(pred.dtype),'tensor_sha256':tensor_sha(pred)})
        target=raw_action[:,:pred.shape[1],:pred.shape[2]]; m=valid[:,:pred.shape[1]]
        sse+=float(((pred-target).square()*m.unsqueeze(-1)).sum()); scalars+=int(m.sum())*pred.shape[-1]
        length=int(m[0].sum()); endpoints.append(float(torch.linalg.vector_norm(pred[0,length-1]-target[0,length-1])))
        if length>=4:
            j=pred[0,3:length]-3*pred[0,2:length-1]+3*pred[0,1:length-2]-pred[0,:length-3]
            jerks.append(float(j.square().mean()))
    # Latency dataset decode is excluded; processor + 10-step sampler + postprocessor is included.
    def fixed_noise(i):
        return torch.randn((1,policy.config.chunk_size,policy.config.max_action_dim),generator=torch.Generator().manual_seed(a.seed+i)).to(device)
    warmup_noises=[fixed_noise(1_000_000+i) for i in range(a.warmup)]
    timed_noises=[fixed_noise(i) for i in range(len(raw_batches))]
    def infer(batch,noise,measure):
        torch.cuda.synchronize(); wall=time.perf_counter(); processed=pre(batch)
        start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True); start.record()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16): pred=policy.predict_action_chunk(processed,noise=noise)
        end.record(); pred=post(pred); torch.cuda.synchronize()
        return (start.elapsed_time(end),(time.perf_counter()-wall)*1000) if measure else None
    for i in range(a.warmup): infer(raw_batches[i%len(raw_batches)],warmup_noises[i],False)
    pairs=[infer(b,noise,True) for b,noise in zip(raw_batches,timed_noises)]; model_ms=[x[0] for x in pairs]; e2e_ms=[x[1] for x in pairs]
    row={'status':'EVAL_OK','model':str(model_path),'samples':n,'fixed_noise_seed':a.seed,
         'heldout_split':a.split,'episode_groups':split[a.split],'split_file':a.split_file,
         'ordered_episode_frame_sample_ids':ordered_ids,'ordered_ids_sha256':canonical_sha(ordered_ids),
         'prediction_sha256':canonical_sha(prediction_records),'prediction_records':prediction_records,
         'verified_lineage':frozen,'verified_lineage_sha256':canonical_sha(frozen),
         'processor_contract_sha256':frozen['processor_contract_sha256'],'action_mse':sse/scalars,
         'endpoint_l2_mean':statistics.fmean(endpoints),'jerk_mean':statistics.fmean(jerks),
         'model_cuda_ms_p50':pct(model_ms,.5),'model_cuda_ms_p95':pct(model_ms,.95),
         'processor_model_post_e2e_ms_p50':pct(e2e_ms,.5),'processor_model_post_e2e_ms_p95':pct(e2e_ms,.95),
         'latency_warmup_chunks':a.warmup,'latency_sync':'before/after each measured chunk',
         'dataset_decode_in_latency':False,'sampler_steps':policy.config.num_steps,
         'flow_model_calls_per_chunk':policy.config.num_steps,'actions_returned_per_chunk':policy.config.chunk_size,
         'e2e_p50_ms_per_action_if_full_chunk_consumed':pct(e2e_ms,.5)/policy.config.chunk_size,
         'dataset_revision':a.dataset_revision,'strict_load':True,'processor_reload_without_overrides':True}
    dest=Path(a.output); dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(json.dumps(row,indent=2),encoding='utf-8')
    print(json.dumps(row))
if __name__=='__main__': main()
PY
chmod +x src/*.py
git clone --filter=blob:none --no-checkout https://github.com/huggingface/lerobot.git vendor/lerobot
git -C vendor/lerobot fetch --depth 1 origin tag v0.4.3
git -C vendor/lerobot checkout --detach 0b067df57d21d3a02d6c511f1609172fa39ac29b
python src/apply_vendor_patch.py
git -C vendor/lerobot diff --check
```

### 执行命令

```bash
python -m py_compile src/*.py
python src/contracts.py --formula-only
git -C vendor/lerobot rev-parse HEAD HEAD^{tree} | tee evidence/lerobot-commit-tree.txt
sha256sum -c evidence/vendor-patch.sha256
sha256sum vendor/lerobot/LICENSE vendor/lerobot/pyproject.toml src/*.py | tee evidence/source.sha256
set +e
python src/preflight.py --repo vendor/lerobot > evidence/runtime-preflight.json 2>&1
runtime_code=$?
set -e
printf 'runtime_exit=%s\n' "$runtime_code" | tee evidence/runtime-status.txt
```

### 预期观测（估算/示例，不是实测）

公式单测输出 `formula:OK`。公司 torch2.1 的 preflight 预期非零并明确 `ENV-BLOCKED`；兼容现有环境应 runtime/repo 均 true。

### 验收条件

代码 commit/tree 与非空 patch SHA 精确；所有脚本静态通过；补丁同时修正 `actions_id_pad` 拼写、padding 分母、DDP global denominator 与 factory strict load。公司 PyTorch2.1 或 V100/BF16 任一门不通过都不发生安装；只有兼容路径可进入 Day 2。

### 若失败，先看什么，再改什么

SHA 不符重新按 tag/commit 获取，不 pull main；runtime 不符就停止，不改 torch；离线则使用管理员批准且同 SHA 的归档。

### 当日证据清单

`evidence/lerobot-commit.txt`、`source.sha256`、`runtime-*`、环境 freeze/硬件清单。公司内容留内部。

## Day 2：依赖 dry-run、许可门、固定下载与 schema 合同

### 为什么做

在安装/大文件下载前看完整计划，并让许可、revision、hash、schema 成为机器可查证据。

### 输入与前置检查

Day 1 runtime 通过；仅个人或组织批准兼容环境。模型权重使用批准必须由组织生成非空 `evidence/model-use-approval.txt`；本文不会替用户伪造。

### 本日要创建/修改的文件

创建 pip 计划/前后 freeze、资产目录/manifest；代码目录已在 Day 1 创建。

### 实现

先 dry-run 并拒绝任何 torch/NVIDIA wheel 替换；批准后才在当前已有环境 editable install。回滚基线为 before-freeze 与组织锁文件；共享/生产环境不得直接安装。

### 执行命令

```bash
python -m pip freeze > evidence/pip-before.txt
python -m pip install --dry-run --report evidence/install-plan.json -e "vendor/lerobot[smolvla]"
python src/preflight.py --repo vendor/lerobot --pip-report evidence/install-plan.json | tee evidence/install-audit.json
# 仅当 install-audit 无 forbidden replacement 且当前环境获批：
python -m pip install -e "vendor/lerobot[smolvla]"
python -m pip check | tee evidence/pip-check.txt
python -m pip freeze > evidence/pip-after.txt
lerobot-train --help > evidence/lerobot-train-help.txt
accelerate launch --help > evidence/accelerate-help.txt
```

下载前先查 Hub 元数据并执行许可门：

```bash
curl -fsSL 'https://huggingface.co/api/models/lerobot/smolvla_base/revision/04d96c1f9167360280aaa54de31418f824d1ef48' > evidence/smolvla-api.json
curl -fsSL 'https://huggingface.co/api/models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/revision/7b375e1b73b11138ff12fe22c8f2822d8fe03467' > evidence/smolvlm2-api.json
curl -fsSL 'https://huggingface.co/api/datasets/lerobot/svla_so101_pickplace/revision/f641879e22172be7e8161d5e6c1503c2d2feb657' > evidence/dataset-api.json
: "${SMOLVLA_APPROVAL_FILE:?set to an existing organization approval record; do not create a placeholder}"
test -s "$SMOLVLA_APPROVAL_FILE"
cat > assets/LOCK.json <<'JSON'
{
  "schema": 1,
  "code": {"repo":"https://github.com/huggingface/lerobot","commit":"0b067df57d21d3a02d6c511f1609172fa39ac29b","git_tree":"b5891d6333c4eb611bd3e81a4d13ed3c71210a6e","license":"Apache-2.0"},
  "assets": {
    "smolvla_base": {"repo":"lerobot/smolvla_base","revision":"04d96c1f9167360280aaa54de31418f824d1ef48","license":"UNDECLARED-REQUIRES-APPROVAL","root":"assets/smolvla_base","required_files":["README.md","config.json","model.safetensors","train_config.json"]},
    "smolvlm2": {"repo":"HuggingFaceTB/SmolVLM2-500M-Video-Instruct","revision":"7b375e1b73b11138ff12fe22c8f2822d8fe03467","license":"Apache-2.0","root":"assets/smolvlm2","required_files":["README.md","added_tokens.json","chat_template.json","config.json","generation_config.json","merges.txt","model.safetensors","preprocessor_config.json","processor_config.json","special_tokens_map.json","tokenizer.json","tokenizer_config.json","vocab.json"]},
    "dataset": {"repo":"lerobot/svla_so101_pickplace","revision":"f641879e22172be7e8161d5e6c1503c2d2feb657","license":"Apache-2.0","root":"assets/dataset","required_files":["README.md","data/chunk-000/file-000.parquet","meta/episodes/chunk-000/file-000.parquet","meta/info.json","meta/stats.json","meta/tasks.parquet","videos/observation.images.side/chunk-000/file-000.mp4","videos/observation.images.up/chunk-000/file-000.mp4"]}
  }
}
JSON
python -c "from huggingface_hub import snapshot_download as d; d('lerobot/smolvla_base',revision='04d96c1f9167360280aaa54de31418f824d1ef48',local_dir='assets/smolvla_base')"
python -c "from huggingface_hub import snapshot_download as d; d('HuggingFaceTB/SmolVLM2-500M-Video-Instruct',revision='7b375e1b73b11138ff12fe22c8f2822d8fe03467',local_dir='assets/smolvlm2',ignore_patterns=['onnx/*'])"
python -c "from huggingface_hub import snapshot_download as d; d('lerobot/svla_so101_pickplace',repo_type='dataset',revision='f641879e22172be7e8161d5e6c1503c2d2feb657',local_dir='assets/dataset')"
python src/inspect_assets.py --lock assets/LOCK.json --output evidence/assets-manifest.json
python src/make_split_stats.py --dataset assets/dataset \
  --dataset-revision f641879e22172be7e8161d5e6c1503c2d2feb657 \
  --asset-manifest evidence/assets-manifest.json --split-out splits/episodes.json \
  --stats-out artifacts/train_stats.json --stats-lineage-out artifacts/train_stats.lineage.json \
  | tee evidence/split-stats.json
test ! -e assets/dataset_train
cp -al assets/dataset assets/dataset_train
rm -f assets/dataset_train/meta/stats.json
install -m 0444 artifacts/train_stats.json assets/dataset_train/meta/stats.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python src/prepare_bundle.py \
  --base assets/smolvla_base --vlm assets/smolvlm2 --dataset assets/dataset \
  --dataset-revision f641879e22172be7e8161d5e6c1503c2d2feb657 \
  --split splits/episodes.json --stats artifacts/train_stats.json \
  --stats-lineage artifacts/train_stats.lineage.json --lock assets/LOCK.json \
  --asset-manifest evidence/assets-manifest.json --vendor vendor/lerobot \
  --patch evidence/vendor.patch --approval-file "$SMOLVLA_APPROVAL_FILE" \
  --output assets/runtime_bundle | tee evidence/bundle.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python src/contracts.py \
  --model-config assets/runtime_bundle/config.json --dataset-info assets/dataset/meta/info.json \
  --split splits/episodes.json --stats artifacts/train_stats.json --processor-root assets/runtime_bundle | tee evidence/contracts.json
torchrun --standalone --nproc-per-node=2 src/test_action_mask.py | tee evidence/action-mask-regression.txt
find assets/runtime_bundle -maxdepth 1 -type f -printf '%f %s\n' | sort | tee evidence/runtime-bundle-files.txt
sha256sum assets/LOCK.json splits/episodes.json artifacts/train_stats.json assets/runtime_bundle/policy_* \
  assets/runtime_bundle/processor_contract.json | tee evidence/lineage-inputs.sha256
```

离线替代由管理员导入三棵批准 snapshot 到 `assets/smolvla_base`、`assets/smolvlm2`、`assets/dataset`，附在线暂存机生成的 canonical tree digest/`SHA256SUMS`。离线端先 `sha256sum -c`，再跑 `inspect_assets.py` 并逐项比较 `content_tree_sha256`；revision lock、文件树和组织批准三者都必须匹配。不得从个人机向公司回传权重、cache 或任何 Task-P 内容。

### 预期观测（估算/示例，不是实测）

安装计划不含 torch/nvidia 替换；三棵资产有精确 revision 与完整本地 content-tree digest；split 是 episode-level 40/5/5 且互斥；stats 只来自 train episode；runtime bundle 包含本地 nested VLM 路径、pre/postprocessor 配置和 state，能无 override reload；两 rank padding 回归给出全局加权结果。模型 API 的 license 为空/缺失时只有外部真实审批记录才能放行。

### 验收条件

runtime/许可/依赖三门均通过；code/model/data revision 与 hash 齐全；camera/action/state/chunk 合同精确。

### 若失败，先看什么，再改什么

dry-run 要换 torch 立即停止；许可文件缺失不下载；dataset AV1/文件不全让管理员重导；合同不符不以 `strict=false` 绕过。

### 当日证据清单

pip before/plan/audit/after/check/help、Hub API、批准记录引用、contracts、assets manifest。

## Day 3：严格 load、单 batch、memory scan

### 为什么做

先证明 processor→dataset→flow loss→action chunk 的真实链路，再选安全 batch。

### 输入与前置检查

Day 2 全过；`HF_HUB_OFFLINE=1` 下顶层权重、嵌套 VLM/tokenizer 与已序列化 processor 都在 `assets/runtime_bundle`/`assets/smolvlm2`，无网络 fallback。训练只读 `assets/dataset_train`（其 stats 是 train-only）并显式选择 episode 0–39；公开 map 固定为 up→image、side→image2。

### 本日要创建/修改的文件

官方 trainer 创建各自 `runs/one_batch` 与 `runs/mem_b*`；不改 vendor 源码。

### 实现

使用 patched fixed repo 安装产生的 `lerobot-train`；factory 已强制 strict load。先生成 base 的逐参数 trainable/frozen hash manifest，再做 1 step 与 batch 1/2/4 的 3-step 独立 run。以下仅适用个人5070Ti/兼容批准 GPU；公司 PyTorch2.1/V100 不执行。

### 执行命令

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export TRAIN_EPISODES='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]'
CUDA_VISIBLE_DEVICES=0 python src/audit_policy.py --model assets/runtime_bundle \
  --processor-contract processor_contract.json --contract-sha processor_contract.sha256 \
  --output evidence/base-parameter-manifest.json
accelerate launch --num_processes 1 --mixed_precision fp16 "$(command -v lerobot-train)" \
  --policy.path=assets/runtime_bundle --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/svla_so101_pickplace --dataset.root=assets/dataset_train --dataset.episodes="$TRAIN_EPISODES" \
  --dataset.revision=f641879e22172be7e8161d5e6c1503c2d2feb657 \
  --rename_map='{"observation.images.up":"observation.image","observation.images.side":"observation.image2"}' \
  --batch_size=1 --num_workers=0 --steps=1 --log_freq=1 --save_freq=1 --eval_freq=0 \
  --output_dir=runs/one_batch --job_name=one_batch --wandb.enable=false 2>&1 | tee logs/one_batch.log

for b in 1 2 4; do
  accelerate launch --num_processes 1 --mixed_precision fp16 "$(command -v lerobot-train)" \
    --policy.path=assets/runtime_bundle --policy.device=cuda --policy.push_to_hub=false \
    --dataset.repo_id=lerobot/svla_so101_pickplace --dataset.root=assets/dataset_train --dataset.episodes="$TRAIN_EPISODES" \
    --dataset.revision=f641879e22172be7e8161d5e6c1503c2d2feb657 \
    --rename_map='{"observation.images.up":"observation.image","observation.images.side":"observation.image2"}' \
    --batch_size="$b" --num_workers=0 --steps=3 --log_freq=1 --save_freq=3 --eval_freq=0 \
    --output_dir="runs/mem_b${b}" --job_name="mem_b${b}" --wandb.enable=false 2>&1 | tee "logs/mem_b${b}.log"
done
```

另开只读监控终端（结果保存公司/本机）：

```bash
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 1 | tee logs/gpu-memory-scan.csv
```

### 预期观测（估算/示例，不是实测）

日志显示 total/trainable params、effective batch、有限 loss/grad norm；checkpoint 含 pretrained model/processor/training state。显存随 batch 增加，具体数值不预填。

### 验收条件

base manifest 精确列出每个参数，`sum(trainable numel)` 与实际 optimizer 集合需在日志/审计中一致；batch1 单步严格完成；至少一个 batch 的3-step 完成；选取峰值 reserved/used <物理显存85%的最大稳定 batch。公司 V100 本日维持 `ENV-BLOCKED`，不能用这些个人GPU结果替代。

### 若失败，先看什么，再改什么

首先看 model/processor missing key、camera rename、action chunk、AV1 decode；然后 dtype/backend。OOM 只降 batch，记录 run；import 试图 FA2 则检查错误 extra，不编译 SM70 fork。

### 当日证据清单

one/memory 日志、GPU scan、checkpoint 文件表、选定 batch 与理由、退出码。

## Day 4：20-step smoke、200-step 正式短训与 resume 审计

### 为什么做

完成真实 optimizer 链与 checkpoint 恢复，同时明确 v0.4.3 对 GradScaler 连续性的证据缺口。

### 输入与前置检查

Day 3 选出安全 batch；下例保守用 1，若改必须所有关联命令和 manifest 同步。`apply_vendor_patch.py` 必须已产生新 patch SHA：它将 checkpoint 先写入同文件系统 staging 目录，校验必需文件和 step，fsync 文件/目录，写 `CHECKPOINT_COMPLETE.json`，再用同目录下的 `os.replace` 一次提交。run 目录必须不存在，已提交 checkpoint 也不允许覆盖。

### 本日要创建/修改的文件

创建 `runs/smoke20`、`runs/continuous200`、`runs/resume`、`logs/*.log`；由固定上游 trainer 加本周可审计 patch 写 checkpoint。不得预先创建空 checkpoint 文件。

### 实现

先独立 20-step；再用同 seed/config 做 continuous `0→200` 和 split `0→100→200`。split 首段仍声明 `--steps=200`，只用环境钩子 `LEROBOT_STOP_AFTER_COMPLETE_CHECKPOINT_STEP=100` 在 step 100 的原子提交、`last` 更新和全 rank barrier 后正常退出；因此 `train_config.json` 中的总目标从未改成 100。恢复只允许从编号固定的 `runs/resume/checkpoints/000100/pretrained_model/train_config.json`，最终只验收 `.../000200`；`last` 仅供人工导航，不作为训练、审计或 eval 输入。

原子保存保护的是上游核心 checkpoint。训练进程退出后，`seal_processor_contract.py` 会在编号目录中添加可验的 processor 合同 sidecar；它不改 model/optimizer/scheduler/RNG/step。两条 final 都 strict load；逐参数 manifest 必须证明 trainable 集合不变、所有 frozen tensor 与 base bitwise 相同。由于固定源码不显式保存 Accelerator GradScaler，continuous-vs-resume 的 trainable state/fixed eval 若不一致应标已知 `RESUME-LIMITATION`，不能放宽成 PASS。

### 执行命令

```bash
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
export TRAIN_EPISODES='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]'
: "${SMOLVLA_APPROVAL_FILE:?set it to the same non-empty approval file used on Day 2}"
test -s "$SMOLVLA_APPROVAL_FILE"
test ! -e runs/smoke20
test ! -e runs/continuous200
test ! -e runs/resume
accelerate launch --num_processes 1 --mixed_precision fp16 "$(command -v lerobot-train)" \
  --policy.path=assets/runtime_bundle --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/svla_so101_pickplace --dataset.root=assets/dataset_train --dataset.episodes="$TRAIN_EPISODES" --dataset.revision=f641879e22172be7e8161d5e6c1503c2d2feb657 \
  --rename_map='{"observation.images.up":"observation.image","observation.images.side":"observation.image2"}' \
  --batch_size=1 --num_workers=0 --steps=20 --log_freq=1 --save_checkpoint=true --save_freq=10 --eval_freq=0 \
  --seed=20260903 --output_dir=runs/smoke20 --job_name=smoke20 --wandb.enable=false 2>&1 | tee logs/smoke20.log

accelerate launch --num_processes 1 --mixed_precision fp16 "$(command -v lerobot-train)" \
  --policy.path=assets/runtime_bundle --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/svla_so101_pickplace --dataset.root=assets/dataset_train --dataset.episodes="$TRAIN_EPISODES" --dataset.revision=f641879e22172be7e8161d5e6c1503c2d2feb657 \
  --rename_map='{"observation.images.up":"observation.image","observation.images.side":"observation.image2"}' \
  --batch_size=1 --num_workers=0 --steps=200 --log_freq=5 --save_checkpoint=true --save_freq=100 --eval_freq=0 --seed=20260903 \
  --output_dir=runs/continuous200 --job_name=continuous200 --wandb.enable=false 2>&1 | tee logs/continuous200.log

test -f runs/continuous200/checkpoints/000200/CHECKPOINT_COMPLETE.json
test "$(python -c 'import json; print(json.load(open("runs/continuous200/checkpoints/000200/CHECKPOINT_COMPLETE.json"))["step"])')" = 200
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python src/seal_processor_contract.py \
  --model runs/continuous200/checkpoints/000200/pretrained_model --source-root assets/runtime_bundle \
  --dataset assets/dataset --dataset-revision f641879e22172be7e8161d5e6c1503c2d2feb657 \
  --split splits/episodes.json | tee evidence/continuous-processor-contract.json
CUDA_VISIBLE_DEVICES=0 python src/audit_policy.py \
  --model runs/continuous200/checkpoints/000200/pretrained_model \
  --reference evidence/base-parameter-manifest.json --processor-contract processor_contract.json \
  --contract-sha processor_contract.sha256 --output evidence/continuous-parameter-manifest.json

export LEROBOT_STOP_AFTER_COMPLETE_CHECKPOINT_STEP=100
accelerate launch --num_processes 1 --mixed_precision fp16 "$(command -v lerobot-train)" \
  --policy.path=assets/runtime_bundle --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/svla_so101_pickplace --dataset.root=assets/dataset_train --dataset.episodes="$TRAIN_EPISODES" --dataset.revision=f641879e22172be7e8161d5e6c1503c2d2feb657 \
  --rename_map='{"observation.images.up":"observation.image","observation.images.side":"observation.image2"}' \
  --batch_size=1 --num_workers=0 --steps=200 --log_freq=5 --save_checkpoint=true --save_freq=100 --eval_freq=0 --seed=20260903 \
  --output_dir=runs/resume --job_name=resume --wandb.enable=false 2>&1 | tee logs/resume_0_100.log
unset LEROBOT_STOP_AFTER_COMPLETE_CHECKPOINT_STEP

test -f runs/resume/checkpoints/000100/CHECKPOINT_COMPLETE.json
test -f runs/resume/checkpoints/000100/pretrained_model/train_config.json
test -f runs/resume/checkpoints/000100/training_state/optimizer_state.safetensors
test -f runs/resume/checkpoints/000100/training_state/rng_state.safetensors
test "$(python -c 'import json; print(json.load(open("runs/resume/checkpoints/000100/CHECKPOINT_COMPLETE.json"))["step"])')" = 100
test "$(python -c 'import json; print(json.load(open("runs/resume/checkpoints/000100/pretrained_model/train_config.json"))["steps"])')" = 200
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python src/seal_processor_contract.py \
  --model runs/resume/checkpoints/000100/pretrained_model --source-root assets/runtime_bundle \
  --dataset assets/dataset --dataset-revision f641879e22172be7e8161d5e6c1503c2d2feb657 \
  --split splits/episodes.json | tee evidence/resume-parent-processor-contract.json
CUDA_VISIBLE_DEVICES=0 python src/audit_policy.py \
  --model runs/resume/checkpoints/000100/pretrained_model \
  --reference evidence/base-parameter-manifest.json --processor-contract processor_contract.json \
  --contract-sha processor_contract.sha256 --output evidence/resume-parent-parameter-manifest.json
printf '%s\n' "$(pwd)/runs/resume/checkpoints/000100" | tee evidence/resume-parent-checkpoint.txt

env -u LEROBOT_STOP_AFTER_COMPLETE_CHECKPOINT_STEP \
  accelerate launch --num_processes 1 --mixed_precision fp16 "$(command -v lerobot-train)" \
  --resume=true --config_path=runs/resume/checkpoints/000100/pretrained_model/train_config.json \
  --steps=200 --save_checkpoint=true --save_freq=100 2>&1 | tee logs/resume_100_200.log

test -f runs/resume/checkpoints/000200/CHECKPOINT_COMPLETE.json
test "$(python -c 'import json; print(json.load(open("runs/resume/checkpoints/000200/CHECKPOINT_COMPLETE.json"))["step"])')" = 200
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python src/seal_processor_contract.py \
  --model runs/resume/checkpoints/000200/pretrained_model --source-root assets/runtime_bundle \
  --dataset assets/dataset --dataset-revision f641879e22172be7e8161d5e6c1503c2d2feb657 \
  --split splits/episodes.json | tee evidence/resume-final-processor-contract.json

find runs/resume/checkpoints -type f -printf '%p %s\n' | sort | tee evidence/resume-files.txt
rg -n "step|loss|grdn|lr|End of training" logs/resume_0_100.log logs/resume_100_200.log | tee evidence/resume-log-index.txt
CUDA_VISIBLE_DEVICES=0 python src/audit_policy.py --model runs/resume/checkpoints/000200/pretrained_model \
  --reference evidence/base-parameter-manifest.json --processor-contract processor_contract.json \
  --contract-sha processor_contract.sha256 --output evidence/resume-parameter-manifest.json
python - <<'PY'
import hashlib,json
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tree_sha(root):
    rows=[(p.relative_to(root).as_posix(),p.stat().st_size,sha(p)) for p in sorted(Path(root).rglob('*')) if p.is_file()]
    return hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()
parent=Path('runs/resume/checkpoints/000100').resolve()
continuous=Path('runs/continuous200/checkpoints/000200').resolve()
final=Path('runs/resume/checkpoints/000200').resolve()
row={'schema':1,'code_commit':'0b067df57d21d3a02d6c511f1609172fa39ac29b',
 'code_tree':'b5891d6333c4eb611bd3e81a4d13ed3c71210a6e','patch_sha256':sha('evidence/vendor.patch'),
 'asset_manifest_sha256':sha('evidence/assets-manifest.json'),'split_sha256':sha('splits/episodes.json'),
 'train_stats_sha256':sha('artifacts/train_stats.json'),'processor_contract_sha256':sha('assets/runtime_bundle/processor_contract.json'),
 'base_parameter_manifest_sha256':sha('evidence/base-parameter-manifest.json'),
 'continuous_parameter_manifest_sha256':sha('evidence/continuous-parameter-manifest.json'),
 'resume_parameter_manifest_sha256':sha('evidence/resume-parameter-manifest.json'),
 'parent_checkpoint':str(parent),'parent_training_state_tree_sha256':tree_sha(parent/'training_state'),
 'continuous_checkpoint':str(continuous),'continuous_training_state_tree_sha256':tree_sha(continuous/'training_state'),
 'final_checkpoint':str(final),'final_training_state_tree_sha256':tree_sha(final/'training_state'),
 'resume_target_step':200,'seed':20260903,
 'known_limitation':'Accelerator GradScaler state is not explicit in pinned v0.4.3 checkpoint code'}
Path('evidence/resume-lineage.json').write_text(json.dumps(row,sort_keys=True,indent=2)+'\n')
print(row)
PY
```

### 预期观测（估算/示例，不是实测）

smoke 到 20；continuous 产生编号 `000200`；split 首段的进程配置仍显示 `steps=200`，但日志只在 `000100` 原子提交且 barrier 后出现 `Stopping after complete checkpoint...`；resume 从该编号目录继续到 `000200`。loss/grad 有限。v0.4.3 文件表含 optimizer/scheduler/RNG/step，但不会凭空出现显式 Accelerator GradScaler state。

### 验收条件

`000100`/`000200` marker 与 `training_step.json` 一致，首段的 `train_config.json.steps==200`，且日志证明钩子在 checkpoint barrier 之后退出；model/processor/optimizer/scheduler/RNG/step 可查，resume 未重新从 0 开始。整个过程不引用 `last`。由于 GradScaler 未在固定源码 checkpoint 函数显式保存，严格 scaler 连续性标 `RESUME-LIMITATION`，不能声称完全 PASS。

### 若失败，先看什么，再改什么

先查具体编号目录的 marker、`training_step.json` 和 `train_config.json.steps`，再查训练 state 文件；不用 `last` 猜 parent。看到 `.*.staging-*` 说明未提交，保留该目录诊断，不把它 rename 成有效 checkpoint。钩子早于 barrier 退出、总 steps 被写成 100 或 CLI override 失效均为合同失败；保留证据并按 `lerobot-train --help`/固定源码确认 parser 语义，不编辑 checkpoint JSON。非有限立即停。

### 当日证据清单

三日志、固定 `000100/000200` checkpoint 文件表/marker/训练 step、三份 processor seal 记录、三份参数 manifest、resume lineage、已知 scaler 缺口、退出码。

## Day 5：四卡 DDP smoke、短 profile 与卡数止损

### 为什么做

在真实 VLA 载荷验证数据分片/通信，但不为“用满八卡”改变算法或长跑低效配置。

### 输入与前置检查

Day 4 单卡通过；Week07 DDP invariant 已通过；公司只有在兼容批准环境执行。四卡先使用真实拓扑中的同一高速域，编号 `0,1,2,3` 必须由 `nvidia-smi topo -m` 核验。

### 本日要创建/修改的文件

创建 `runs/ddp4`、日志；可选 `traces/ddp4*` 由已安装 Nsight Systems 创建。

### 实现

Accelerate 启动4进程，local batch=1、global batch=4；patched action loss 以跨 rank 的 valid-action numerator/denominator 反传。这里只做 correctness/smoke，不冒充强/弱扩展结论；公司V100仍因 runtime/BF16阻塞，命令只适用于四张相同的兼容批准 GPU。

### 执行命令

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
export TRAIN_EPISODES='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]'
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 --main_process_port 29610 \
  "$(command -v lerobot-train)" --policy.path=assets/runtime_bundle --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/svla_so101_pickplace --dataset.root=assets/dataset_train --dataset.episodes="$TRAIN_EPISODES" --dataset.revision=f641879e22172be7e8161d5e6c1503c2d2feb657 \
  --rename_map='{"observation.images.up":"observation.image","observation.images.side":"observation.image2"}' \
  --batch_size=1 --num_workers=0 --steps=20 --log_freq=1 --save_freq=20 --eval_freq=0 \
  --output_dir=runs/ddp4 --job_name=ddp4 --wandb.enable=false 2>&1 | tee logs/ddp4.log
```

若 `command -v nsys` 成功且公司批准采集，仅做 8-step 新 run：

```bash
command -v nsys | tee evidence/nsys-path.txt
CUDA_VISIBLE_DEVICES=0,1,2,3 nsys profile --trace=cuda,nvtx,osrt,cublas,nccl --sample=none \
  --output=traces/ddp4_short --force-overwrite=false \
  accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 --main_process_port 29611 \
  "$(command -v lerobot-train)" --policy.path=assets/runtime_bundle --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/svla_so101_pickplace --dataset.root=assets/dataset_train --dataset.episodes="$TRAIN_EPISODES" --dataset.revision=f641879e22172be7e8161d5e6c1503c2d2feb657 \
  --rename_map='{"observation.images.up":"observation.image","observation.images.side":"observation.image2"}' \
  --batch_size=1 --num_workers=0 --steps=8 --log_freq=1 --save_freq=8 --eval_freq=0 \
  --output_dir=runs/ddp4_profile --job_name=ddp4_profile --wandb.enable=false
```

### 预期观测（估算/示例，不是实测）

20-step 有限并显示 effective batch=4；trace 候选可分 image decode、VLM/action expert、backward、all-reduce。450M/0.5B 在四卡后可能通信主导，不预设效率。

### 验收条件

四卡20-step无 hang/非有限；每 rank 处理不同 shard（需从 dataloader/Accelerate 证据核）；trace 若采集只留公司。没有公平单卡 global-batch4 reference 时不声称 scaling efficiency。

### 若失败，先看什么，再改什么

单卡过、四卡失败先查 launcher/可见卡/data shard/unused params/NCCL；8卡不是修复手段。四卡通信主导则停止扩卡，不强跑8卡。

### 当日证据清单

四卡日志、拓扑/NCCL/进程映射、checkpoint、可选 trace 文件表/哈希、卡数决策（公司内部）。

## Day 6：独立 base vs FT offline eval、失败注入与 Model Card

### 为什么做

从新进程严格加载 base/FT 与 processor，在相同样本/noise下比较；同时验证 schema 错误不会静默通过。

### 输入与前置检查

Day 4 的两个最终 checkpoint 完整且 processor 合同已 seal；continuous 固定为 `runs/continuous200/checkpoints/000200/pretrained_model`，resumed 固定为 `runs/resume/checkpoints/000200/pretrained_model`，禁止用 `last`。eval 只加载 test episode 45–49，normalization 只来自各 model 目录已保存的 train-only processor；绝不以 test/full-dataset stats override。公开数据只读。

### 本日要创建/修改的文件

创建 base、continuous-200、resumed-200 三份 JSON eval，`resume-comparison.json`，schema 负测证据和含已知事实/待验证字段的 Model Card。

### 实现

`offline_eval.py` 由 Day1 已完整创建；它在构建模型前验证 asset/split/stats/code/patch/approval/processor/model-manifest 全部 lineage，然后 strict load policy，processor 无 override reload，按 policy delta timestamps 建 50 步 action chunk，固定同一 test sample/noise。质量与计时分离；计时排除 dataset decode、包含 preprocessor→10-step flow sampler→postprocessor，5 个 warmup，每个 chunk 前后显式同步，输出 model 与端到端 p50/p95、calls/chunk、actions/chunk 及完整消费 chunk 时的摊销口径。`compare_resume.py` 再对比固定的两个 `000200` 根目录、参数 manifest、optimizer/scheduler/RNG/step 与同控 eval。以下仅适用兼容批准 GPU。

### 执行命令

```bash
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true
: "${SMOLVLA_APPROVAL_FILE:?set it to the same non-empty approval file used on Day 2}"
test -s "$SMOLVLA_APPROVAL_FILE"
CUDA_VISIBLE_DEVICES=0 python src/offline_eval.py --model assets/runtime_bundle --dataset-root assets/dataset \
  --dataset-revision f641879e22172be7e8161d5e6c1503c2d2feb657 --split-file splits/episodes.json --split test \
  --samples 32 --seed 20260903 --warmup 5 --stats artifacts/train_stats.json \
  --stats-lineage artifacts/train_stats.lineage.json --lock assets/LOCK.json \
  --asset-manifest evidence/assets-manifest.json --processor-contract processor_contract.json \
  --contract-sha processor_contract.sha256 --model-manifest evidence/base-parameter-manifest.json \
  --vendor vendor/lerobot --patch evidence/vendor.patch --approval-file "$SMOLVLA_APPROVAL_FILE" \
  --output logs/eval_base.json
CUDA_VISIBLE_DEVICES=0 python src/offline_eval.py \
  --model runs/continuous200/checkpoints/000200/pretrained_model --dataset-root assets/dataset \
  --dataset-revision f641879e22172be7e8161d5e6c1503c2d2feb657 --split-file splits/episodes.json --split test \
  --samples 32 --seed 20260903 --warmup 5 --stats artifacts/train_stats.json \
  --stats-lineage artifacts/train_stats.lineage.json --lock assets/LOCK.json \
  --asset-manifest evidence/assets-manifest.json --processor-contract processor_contract.json \
  --contract-sha processor_contract.sha256 --model-manifest evidence/continuous-parameter-manifest.json \
  --vendor vendor/lerobot --patch evidence/vendor.patch --approval-file "$SMOLVLA_APPROVAL_FILE" \
  --output logs/eval_continuous200.json
CUDA_VISIBLE_DEVICES=0 python src/offline_eval.py \
  --model runs/resume/checkpoints/000200/pretrained_model --dataset-root assets/dataset \
  --dataset-revision f641879e22172be7e8161d5e6c1503c2d2feb657 --split-file splits/episodes.json --split test \
  --samples 32 --seed 20260903 --warmup 5 --stats artifacts/train_stats.json \
  --stats-lineage artifacts/train_stats.lineage.json --lock assets/LOCK.json \
  --asset-manifest evidence/assets-manifest.json --processor-contract processor_contract.json \
  --contract-sha processor_contract.sha256 --model-manifest evidence/resume-parameter-manifest.json \
  --vendor vendor/lerobot --patch evidence/vendor.patch --approval-file "$SMOLVLA_APPROVAL_FILE" \
  --output logs/eval_resumed200.json
python src/compare_resume.py \
  --continuous-checkpoint runs/continuous200/checkpoints/000200 \
  --resumed-checkpoint runs/resume/checkpoints/000200 \
  --continuous-manifest evidence/continuous-parameter-manifest.json \
  --resumed-manifest evidence/resume-parameter-manifest.json \
  --continuous-eval logs/eval_continuous200.json --resumed-eval logs/eval_resumed200.json \
  --output evidence/resume-comparison.json | tee evidence/resume-comparison.stdout.json

cp assets/dataset/meta/info.json evidence/bad-info.json
python - <<'PY'
import json
p='evidence/bad-info.json'; x=json.load(open(p)); x['features']['action']['shape']=[7]
open(p,'w').write(json.dumps(x))
PY
set +e
python src/contracts.py --model-config assets/runtime_bundle/config.json --dataset-info evidence/bad-info.json \
  --split splits/episodes.json --stats artifacts/train_stats.json --processor-root assets/runtime_bundle > evidence/bad-schema.stdout 2>&1
code=$?
set -e
test "$code" -ne 0
printf 'EXPECTED_FAILURE exit=%s\n' "$code" | tee evidence/bad-schema.result
```

创建报告（把 `PENDING` 只替换为日志可证明的值）：

```bash
cat > reports/model_card.md <<'MD'
# SmolVLA Week10 受控微调卡

- code: LeRobot v0.4.3 / `0b067df57d21d3a02d6c511f1609172fa39ac29b` / Apache-2.0
- base: `lerobot/smolvla_base@04d96c1f9167360280aaa54de31418f824d1ef48`; 权重许可批准引用：PENDING
- nested VLM: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct@7b375e1b73b11138ff12fe22c8f2822d8fe03467` / Apache-2.0
- data: `lerobot/svla_so101_pickplace@f641879e22172be7e8161d5e6c1503c2d2feb657` / Apache-2.0
- split/stats: episode-group train0–39/dev40–44/test45–49；normalization=train-only；split/stats/processor SHA: PENDING
- schema: state/action 6；up→image、side→image2；第三相机不伪造；30 FPS；chunk 50；flow steps 10
- runtime/GPU/precision/backend: PENDING（固定 v0.4.3 的 V100 路径因 hard-coded BF16 + torch2.1 冲突为 ENV-BLOCKED）
- trainable modules/params + frozen-invariance: PENDING（引用逐参数 manifest SHA）
- train command/checkpoint/resume lineage: PENDING（引用 code tree/patch/assets/split/stats/processor/parent/step）
- known recovery limitation: v0.4.3 checkpoint 源码未显式保存 Accelerator GradScaler
- base offline metrics: PENDING
- continuous-200 offline metrics: PENDING
- resumed-200 offline metrics: PENDING
- continuous-vs-resume comparison: PENDING（引用 `evidence/resume-comparison.json`）
- rollout: 未执行则 INCONCLUSIVE；offline 指标不代表机器人成功率
- security: 公司数据、日志、trace、视频、checkpoint、拓扑和指标仅留公司内部
- final status: PASS / INCONCLUSIVE / ENV-BLOCKED / DATA-BLOCKED / LICENSE-BLOCKED 中选择并附证据
MD
sha256sum logs/eval_base.json logs/eval_continuous200.json logs/eval_resumed200.json \
  evidence/resume-comparison.json reports/model_card.md | tee evidence/final.sha256
```

### 预期观测（估算/示例，不是实测）

base/continuous/resumed 各输出 `EVAL_OK`，held-out episode/sample、noise seed、dataset revision、processor lineage、sampler steps与计时边界一致；`resume-comparison.json` 诚实输出 `EQUIVALENT` 或带具体原因的 `RESUME-LIMITATION`；负测非零。FT 不保证优于 base。

### 验收条件

独立严格 load 三模型；公共指标同边界；comparison 的 checkpoint、manifest、eval 全部指向固定 `000200`；bad action shape 被拒绝；报告把 runtime/license/resume 限制写全。没有 rollout 只报 offline/`INCONCLUSIVE`。

### 若失败，先看什么，再改什么

eval 先查 checkpoint/processor文件，再查 delta timestamps、rename/stats、AV1；不要 `strict=false`。负测成功说明合同缺失，先修审计工具。FT 变差按 perception/instruction/no-motion/trajectory/timing/gripper/system 分桶，不挑样本。

### 当日证据清单

三份 eval、resume comparison、schema 负测、Model Card/hash、所有门状态、命令/退出码；公司证据不外传。

## 4. 官方版本的已知恢复缺口与如何诚实验收

v0.4.3 官方 checkpoint 明确保存 policy/processor、optimizer、scheduler、RNG、step；源码调用没有显式保存 `Accelerator.scaler`。因此本手册会实际跑官方 resume，但严格“scaler 连续”仍标 `INCONCLUSIVE`。若组织要补齐，只能在单独分支做经过测试/评审的 Accelerate `save_state/load_state` 集成，并给新 code SHA；不能在本文假装已有。

## 5. 最终证据矩阵

每项必须带路径和退出码：runtime与V100/BF16门、code commit/tree/patch/license、base与nested VLM revision/license/content-tree、dataset revision/license/tree/schema、episode split/train-only stats、pre/postprocessor完整hash与无override reload、strict policy load、逐参数trainable/frozen manifest、单batch/memory、continuous与split resume lineage、held-out base/FT eval latency口径、DDP/profile和负测。未跑就是 `PENDING`，环境或许可不允许就是对应 `*-BLOCKED`，证据不足就是 `INCONCLUSIVE`；虚构 PASS 不是。
