# Week 08 实践篇：从空目录完成 FSDP1、Hybrid 与分片恢复

> 核验日期：2026-09-03。命令与输出都尚未在目标机器实跑；示例只能作为预期。公司 8×V100/Linux、32 GB/卡、拓扑和 PyTorch 2.1 为用户自述，先核验。公司代码、数据、日志、trace、截图、checkpoint、拓扑和性能数字不得导出。

## 0. 结果定义、资源路线与止损

主结果是在公司已有 PyTorch 2.1 环境用 FSDP1 完成：tiny 正确性、0.6B Full Shard、可用时的 2×4 Hybrid、同 world-size 分片 checkpoint/resume、独立 eval、incomplete 拒绝、activation checkpointing A/B。个人 5070Ti 只可做单卡契约/源码检查；可选 H100 是独立环境对照，不能代替 V100 证据。

V100 路线显式 `fp16`，不使用 BF16，不安装 FlashAttention-2。公司 PyTorch 2.1 不升级。若本机 2.1.x 不支持本文检查到的 `HYBRID_SHARD + tuple process_group` 或 DCP 签名，Hybrid/相关迁移标 `ENV-BLOCKED`；Full Shard 主线继续。

立即止损：任一 rank 非有限 loss/grad；collective 超过 5 分钟；FSDP unit 覆盖不符；峰值超过物理显存 85% 后仍继续加压；checkpoint 无完成标志却被加载；恢复 step/RNG 不连续；为跑通而静默换策略、精度或模型。

## 1. 数据/模型/依赖获取清单

| 对象 | 精确身份 | 许可 | 下载/离线替代 | 完整性 |
|---|---|---|---|---|
| PyTorch | 公司现有、批准的 2.1.x | BSD-style | 不下载、不升级；缺失则 `ENV-BLOCKED` | `torch.__version__`、路径、`pip freeze` |
| 模型 | 本文 `FSDPLabModel`，正式参数由 CLI 固定 | 项目内部教学代码 | 文内完整创建 | `sha256sum src/fsdp_lab.py` |
| 数据 | sample ID 的确定性 token 公式 | 现场生成 | 无网络依赖 | 记录 seed/step/world/batch |
| DCP | `torch.distributed.checkpoint` 随 PyTorch | 同 PyTorch | 无额外安装 | 保存 `.metadata`/manifest/文件表 |

官方一手资料：[FSDP 2.1](https://pytorch.org/docs/2.1/fsdp.html)、[DCP 2.1](https://pytorch.org/docs/2.1/distributed.checkpoint.html)、[activation checkpointing 2.1](https://pytorch.org/docs/2.1/checkpoint.html)、[FSDP advanced tutorial](https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html)。链接于 2026-09-03 核验；具体 patch 版本的本机签名优先。DCP state dict 不假定跨 PyTorch 版本兼容。

离线方案就是本实验默认方案。不得从个人机器搬 wheel 到公司，也不得把公司 checkpoint 搬出。

## 2. 环境前检（禁止新建 conda/venv）

```bash
which python
python -VV
python -m pip --version
python -m pip freeze > /tmp/week08-pip-freeze.txt
python - <<'PY'
import inspect, torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
import torch.distributed.checkpoint as dcp
print('torch', torch.__version__, torch.__file__)
print('cuda', torch.version.cuda, 'available', torch.cuda.is_available(), 'count', torch.cuda.device_count())
print('FSDP', inspect.signature(FSDP))
print('strategies', list(ShardingStrategy))
print('dcp.save_state_dict', inspect.signature(dcp.save_state_dict))
print('dcp.load_state_dict', inspect.signature(dcp.load_state_dict))
for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i)
    print(i, p.name, p.total_memory, (p.major,p.minor))
PY
nvidia-smi -L
nvidia-smi topo -m
df -h .
```

判定：公司版本不是获批 2.1.x，停下确认而非升级；无 `FULL_SHARD`/DCP，主线 `ENV-BLOCKED`；无 `HYBRID_SHARD` 或构造签名/运行时拒绝 tuple process group，只将 Hybrid 标阻塞。核心无需额外依赖，因此没有 pip 安装与回滚；管理员未给出精确包名、版本和批准前不执行额外安装。

纸面资源：约 0.612B 模型在 FP32 AdamW 状态下界约 9.8 GB/卡（DDP）、2.45 GB/卡（四路 shard）、1.23 GB/卡（八路 shard），未含 activation、all-gather buffer、AMP 副本和 allocator。先 tiny，再 5-step formal probe；峰值不得直接由下界推断。

## 3. 目录树

```text
week08-fsdp/
├── src/fsdp_lab.py
├── artifacts/checkpoints/
├── evidence/
├── logs/
└── traces/
```

Day 1 先创建所有目录与唯一核心文件；后续命令不引用未创建路径。

## Day 1：创建完整实现、静态检查和 CPU 单测

### 为什么做

先固定模型、wrap、group、DCP 和状态恢复协议，再进入昂贵多卡实验。

### 输入与前置检查

使用已通过前检的环境；无需数据/模型下载。

### 本日要创建/修改的文件

创建 `week08-fsdp/src/fsdp_lab.py` 和四个产物目录。

### 实现

```bash
mkdir -p week08-fsdp/{src,artifacts/checkpoints,evidence,logs,traces}
cd week08-fsdp
cat > src/fsdp_lab.py <<'PY'
#!/usr/bin/env python3
import argparse
import functools
import gc
import hashlib
import inspect
import json
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def args_parser():
    p=argparse.ArgumentParser()
    p.add_argument('--mode', choices=['unit','api','train','eval','correctness','compare'], default='train')
    p.add_argument('--strategy', choices=['ddp','full','hybrid'], default='full')
    p.add_argument('--seed', type=int, default=20260903)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--eval-batches', type=int, default=10)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--seq', type=int, default=64)
    p.add_argument('--vocab', type=int, default=2048)
    p.add_argument('--dim', type=int, default=256)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--precision', choices=['fp32','fp16'], default='fp16')
    p.add_argument('--activation-checkpoint', action='store_true')
    p.add_argument('--save-every', type=int, default=10)
    p.add_argument('--log-every', type=int, default=5)
    p.add_argument('--checkpoint-root', default='artifacts/checkpoints/run')
    p.add_argument('--resume-from', default='')
    p.add_argument('--log', default='logs/train.jsonl')
    p.add_argument('--profile-steps', type=int, default=0)
    p.add_argument('--benchmark-warmup', type=int, default=0)
    p.add_argument('--trace-dir', default='traces')
    p.add_argument('--compare-from', default='')
    p.add_argument('--compare-atol', type=float, default=1e-5)
    p.add_argument('--compare-rtol', type=float, default=1e-4)
    return p.parse_args()


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1=nn.LayerNorm(dim)
        self.attn=nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.norm2=nn.LayerNorm(dim)
        self.mlp=nn.Sequential(nn.Linear(dim,4*dim), nn.GELU(), nn.Linear(4*dim,dim))

    def forward(self, x):
        q=self.norm1(x)
        x=x+self.attn(q,q,q,need_weights=False)[0]
        return x+self.mlp(self.norm2(x))


class FSDPLabModel(nn.Module):
    def __init__(self, vocab, dim, heads, layers):
        super().__init__()
        self.embed=nn.Embedding(vocab,dim)
        self.blocks=nn.ModuleList([Block(dim,heads) for _ in range(layers)])
        self.norm=nn.LayerNorm(dim)
        self.head=nn.Linear(dim,vocab,bias=False)

    def forward(self, ids):
        x=self.embed(ids)
        for block in self.blocks:
            x=block(x)
        return self.head(self.norm(x))


def init_dist():
    world=int(os.getenv('WORLD_SIZE','1')); rank=int(os.getenv('RANK','0')); local=int(os.getenv('LOCAL_RANK','0'))
    if not torch.cuda.is_available():
        raise RuntimeError('distributed train/eval requires CUDA; use --mode unit for CPU')
    if local >= torch.cuda.device_count():
        raise RuntimeError(f'LOCAL_RANK={local}, visible GPUs={torch.cuda.device_count()}')
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=timedelta(minutes=5))
    return rank,local,world,torch.device('cuda',local)


def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_batch(step, rank, world, batch, seq, vocab, device, eval_mode=False):
    offset=10_000_000 if eval_mode else 0
    first=offset+step*world*batch+rank*batch
    sample=torch.arange(first,first+batch,dtype=torch.long)
    pos=torch.arange(seq,dtype=torch.long).reshape(1,seq)
    x=(sample.reshape(-1,1)*104729+pos*1543+17)%vocab
    y=(x*31+pos*7+3)%vocab
    return x.to(device),y.to(device),sample.tolist()


def config(a,world):
    result={k:getattr(a,k) for k in ('strategy','seed','batch','seq','vocab','dim','heads','layers','lr','precision','activation_checkpoint')}
    result.update({'world_size':world,'optimizer':'AdamW','optimizer_weight_decay':0.01,'scheduler':'constant'})
    return result


def write_jsonl(path,row):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf-8') as f: f.write(json.dumps(row,sort_keys=True)+'\n')


def atomic_text(path,text):
    path=Path(path); tmp=path.with_suffix(path.suffix+f'.tmp.{os.getpid()}')
    tmp.write_text(text,encoding='utf-8'); os.replace(tmp,path)


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(path):
    target=Path(path); marker_file=target/'COMPLETED'
    if not marker_file.is_file():
        raise RuntimeError(f'INCOMPLETE_CHECKPOINT: {target} lacks COMPLETED')
    marker=json.loads(marker_file.read_text(encoding='utf-8'))
    current_torch=str(torch.__version__)
    if marker.get('format_version')!=1:
        raise RuntimeError(f"unsupported checkpoint format_version={marker.get('format_version')}")
    if marker.get('torch_version')!=current_torch:
        raise RuntimeError(
            f"TORCH_VERSION_MISMATCH saved={marker.get('torch_version')} current={current_torch}"
        )
    manifest_file=target/'manifest.json'
    if not manifest_file.is_file() or sha256_file(manifest_file)!=marker.get('manifest_sha256'):
        raise RuntimeError(f'INCOMPLETE_CHECKPOINT: manifest missing or not bound by marker: {target}')
    manifest=json.loads(manifest_file.read_text(encoding='utf-8'))
    if manifest.get('format_version')!=1 or manifest.get('step')!=marker.get('step'):
        raise RuntimeError('checkpoint marker/manifest schema or step mismatch')
    if manifest.get('torch_version')!=current_torch:
        raise RuntimeError('checkpoint manifest torch version differs from current interpreter')
    files=manifest.get('files'); hashes=manifest.get('sha256')
    if not isinstance(files,list) or not isinstance(hashes,dict) or set(files)!=set(hashes):
        raise RuntimeError('checkpoint manifest files/hash set is incomplete')
    saved_world=manifest.get('world_size')
    if not isinstance(saved_world,int) or saved_world<1:
        raise RuntimeError(f'invalid checkpoint world_size: {saved_world}')
    expected_extra={f'extra_rank{rank}.pt' for rank in range(saved_world)}
    if not expected_extra.issubset(set(files)):
        raise RuntimeError(f'checkpoint is missing rank-local extras: expected={expected_extra}')
    strategy=manifest.get('strategy')
    if strategy=='ddp' and 'ddp.pt' not in files:
        raise RuntimeError('DDP checkpoint is missing ddp.pt')
    if strategy in {'full','hybrid'} and 'dcp/.metadata' not in files:
        raise RuntimeError('FSDP checkpoint is missing dcp/.metadata')
    if strategy not in {'ddp','full','hybrid'}:
        raise RuntimeError(f'unknown checkpoint strategy: {strategy}')
    for rel,expected in hashes.items():
        rel_path=Path(rel)
        if rel_path.is_absolute() or '..' in rel_path.parts:
            raise RuntimeError(f'unsafe checkpoint manifest path: {rel}')
        file_path=target/rel
        if not file_path.is_file() or sha256_file(file_path)!=expected:
            raise RuntimeError(f'checkpoint file missing/corrupt: {file_path}')
    return target,manifest


def distributed_validate_checkpoint(path,rank):
    payload=[None]
    if rank==0:
        try:
            target,manifest=validate_checkpoint(path)
            payload[0]={'ok':True,'target':str(target),'manifest':manifest}
        except Exception as exc:
            payload[0]={'ok':False,'error':f'{type(exc).__name__}: {exc}'}
    dist.broadcast_object_list(payload,src=0)
    if not payload[0]['ok']:
        raise RuntimeError(payload[0]['error'])
    return Path(payload[0]['target']),payload[0]['manifest']


def make_hybrid_groups(rank,world):
    if world != 8:
        raise RuntimeError('this audited 2x4 layout requires world_size=8')
    rows=[list(range(0,4)),list(range(4,8))]
    cols=[[i,i+4] for i in range(4)]
    row_groups=[dist.new_group(ranks=r) for r in rows]
    col_groups=[dist.new_group(ranks=c) for c in cols]
    shard=next(g for members,g in zip(rows,row_groups) if rank in members)
    replica=next(g for members,g in zip(cols,col_groups) if rank in members)
    membership={'shard':next(r for r in rows if rank in r),'replica':next(c for c in cols if rank in c)}
    return (shard,replica),membership


def parameter_coverage(model):
    unique={id(p):(name,p) for name,p in model.named_parameters()}
    all_ids=set(unique)
    claimed=set(); units=[]
    for index,block in enumerate(model.blocks):
        ids={id(p) for p in block.parameters()}
        overlap=claimed & ids
        if overlap:
            raise RuntimeError(f'parameter appears in multiple wrap units: {overlap}')
        claimed |= ids
        units.append({'name':f'blocks.{index}','parameter_tensors':len(ids),
            'numel':sum(unique[x][1].numel() for x in ids),
            'trainable_numel':sum(unique[x][1].numel() for x in ids if unique[x][1].requires_grad)})
    root_ids=all_ids-claimed
    if claimed | root_ids != all_ids or claimed & root_ids:
        raise RuntimeError('original parameter ownership does not form a unique cover')
    covered_numel=sum(p.numel() for _,p in unique.values())
    if covered_numel != sum(x['numel'] for x in units)+sum(unique[x][1].numel() for x in root_ids):
        raise RuntimeError('original parameter numel coverage mismatch')
    return {'original_unique_tensors':len(all_ids),'original_numel':covered_numel,
        'original_trainable_numel':sum(p.numel() for _,p in unique.values() if p.requires_grad),
        'units':units,'root':{'names':sorted(unique[x][0] for x in root_ids),
            'parameter_tensors':len(root_ids),'numel':sum(unique[x][1].numel() for x in root_ids),
            'trainable_numel':sum(unique[x][1].numel() for x in root_ids if unique[x][1].requires_grad)}}


def wrap_model(a,model,rank,local,world,device,ownership):
    if a.strategy=='ddp':
        return DDP(model.to(device),device_ids=[local]),{'strategy':'ddp','ownership':ownership}
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    auto=functools.partial(transformer_auto_wrap_policy,transformer_layer_cls={Block})
    mp=None
    if a.precision=='fp16':
        mp=MixedPrecision(param_dtype=torch.float16,reduce_dtype=torch.float16,buffer_dtype=torch.float16)
    kwargs=dict(auto_wrap_policy=auto,mixed_precision=mp,device_id=local,use_orig_params=False,limit_all_gathers=True)
    membership=None
    if a.strategy=='full':
        kwargs['sharding_strategy']=ShardingStrategy.FULL_SHARD
    else:
        if not hasattr(ShardingStrategy,'HYBRID_SHARD'):
            raise RuntimeError('ENV-BLOCKED: HYBRID_SHARD absent in this PyTorch')
        groups,membership=make_hybrid_groups(rank,world)
        kwargs['sharding_strategy']=ShardingStrategy.HYBRID_SHARD
        kwargs['process_group']=groups
    try:
        wrapped=FSDP(model.to(device),**kwargs)
    except (TypeError,ValueError,RuntimeError) as exc:
        if a.strategy=='hybrid':
            raise RuntimeError(f'ENV-BLOCKED: local FSDP rejects audited hybrid groups: {exc}') from exc
        raise
    if a.activation_checkpoint:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper,
        )
        wrapper=functools.partial(checkpoint_wrapper,checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        apply_activation_checkpointing(wrapped,checkpoint_wrapper_fn=wrapper,check_fn=lambda m:isinstance(m,Block))
    wrapper_names=[name or '<root>' for name,module in wrapped.named_modules() if isinstance(module,FSDP)]
    expected_wrappers=len(ownership['units'])+1
    if len(wrapper_names) != expected_wrappers:
        raise RuntimeError(f'FSDP wrapper coverage mismatch: got={wrapper_names} expected_count={expected_wrappers}')
    return wrapped,{'strategy':a.strategy,'membership':membership,
        'fsdp_wrapper_names':wrapper_names,'expected_wrapper_count':expected_wrappers,
        'ownership':ownership}


def unwrapped(model):
    return model.module if isinstance(model,DDP) else model


def is_fsdp(model):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    return isinstance(model,FSDP)


def save_checkpoint(a,model,opt,sched,scaler,step,attempted,successful,rank,world):
    root=Path(a.checkpoint_root); target=root/f'ckpt_step{step:06d}'
    temporary=root/f'.tmp_ckpt_step{step:06d}'
    exists=torch.tensor(1 if rank==0 and (target.exists() or temporary.exists()) else 0,
                        device=torch.device('cuda',torch.cuda.current_device()))
    dist.broadcast(exists,src=0)
    if exists.item():
        raise RuntimeError(f'refuse to overwrite completed/staging checkpoint for step={step}')
    if rank==0:
        root.mkdir(parents=True,exist_ok=True); temporary.mkdir(parents=False,exist_ok=False)
    dist.barrier()
    if is_fsdp(model):
        import torch.distributed.checkpoint as dcp
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP, ShardedOptimStateDictConfig,
            ShardedStateDictConfig, StateDictType,
        )
        with FSDP.state_dict_type(
            model,StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
            ShardedOptimStateDictConfig(offload_to_cpu=True),
        ):
            state={'model':model.state_dict(),'optimizer':FSDP.optim_state_dict(model,opt)}
            dcp.save_state_dict(state_dict=state,storage_writer=dcp.FileSystemWriter(str(temporary/'dcp')))
    else:
        if rank==0:
            torch.save({'model':unwrapped(model).state_dict(),'optimizer':opt.state_dict()},temporary/'ddp.pt')
    extra={
        'config':config(a,world),'step':step,'attempted_updates':attempted,
        'successful_updates':successful,'scheduler':sched.state_dict(),'scaler':scaler.state_dict(),
        'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state(device=torch.cuda.current_device()),'torch_version':str(torch.__version__),
    }
    torch.save(extra,temporary/f'extra_rank{rank}.pt')
    dist.barrier()
    if rank==0:
        files=sorted(str(p.relative_to(temporary)) for p in temporary.rglob('*') if p.is_file())
        manifest={'format_version':1,'step':step,'attempted_updates':attempted,
            'successful_updates':successful,'world_size':world,'torch_version':str(torch.__version__),
            'strategy':a.strategy,'files':files,
            'sha256':{name:sha256_file(temporary/name) for name in files}}
        atomic_text(temporary/'manifest.json',json.dumps(manifest,indent=2,sort_keys=True))
        manifest_sha=sha256_file(temporary/'manifest.json')
        os.replace(temporary,target)
        marker={'format_version':1,'step':step,'torch_version':str(torch.__version__),
                'manifest_sha256':manifest_sha}
        atomic_text(target/'COMPLETED',json.dumps(marker,sort_keys=True)+'\n')
    dist.barrier()
    return target


def load_checkpoint(a,model,opt,sched,scaler,rank,world,device):
    target,manifest=distributed_validate_checkpoint(a.resume_from,rank)
    extra_file=target/f'extra_rank{rank}.pt'
    if not extra_file.is_file():
        raise RuntimeError('same-world-size resume requires every rank extra state')
    extra=torch.load(extra_file,map_location='cpu')
    if extra.get('torch_version') != str(torch.__version__):
        raise RuntimeError(f"TORCH_VERSION_MISMATCH saved={extra.get('torch_version')} current={torch.__version__}")
    if manifest.get('step') != extra.get('step'):
        raise RuntimeError('checkpoint manifest/extra step mismatch')
    if manifest.get('attempted_updates') != extra.get('attempted_updates') or \
       manifest.get('successful_updates') != extra.get('successful_updates'):
        raise RuntimeError('checkpoint manifest/extra clocks mismatch')
    if extra.get('successful_updates') != extra.get('step') or \
       extra.get('attempted_updates',-1) < extra.get('successful_updates',0):
        raise RuntimeError('checkpoint attempted/successful clocks are invalid')
    if manifest.get('world_size') != world:
        raise RuntimeError(f"world-size mismatch saved={manifest.get('world_size')} current={world}")
    if extra['config'] != config(a,world):
        raise RuntimeError(f"resume config mismatch saved={extra['config']} current={config(a,world)}")
    if is_fsdp(model):
        import torch.distributed.checkpoint as dcp
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP, ShardedOptimStateDictConfig,
            ShardedStateDictConfig, StateDictType,
        )
        with FSDP.state_dict_type(
            model,StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
            ShardedOptimStateDictConfig(offload_to_cpu=True),
        ):
            state={'model':model.state_dict(),'optimizer':FSDP.optim_state_dict(model,opt)}
            dcp.load_state_dict(state_dict=state,storage_reader=dcp.FileSystemReader(str(target/'dcp')))
            model.load_state_dict(state['model'])
            optim_to_load=FSDP.optim_state_dict_to_load(model,opt,state['optimizer'])
            opt.load_state_dict(optim_to_load)
    else:
        shared=torch.load(target/'ddp.pt',map_location=device)
        unwrapped(model).load_state_dict(shared['model']); opt.load_state_dict(shared['optimizer'])
    sched.load_state_dict(extra['scheduler']); scaler.load_state_dict(extra['scaler'])
    random.setstate(extra['python_rng']); torch.set_rng_state(extra['torch_rng'])
    torch.cuda.set_rng_state(extra['cuda_rng'],device=device)
    dist.barrier()
    return int(extra['step']),int(extra['attempted_updates']),int(extra['successful_updates'])


def unit_test():
    torch.manual_seed(3)
    m=FSDPLabModel(101,32,4,2)
    ids=torch.arange(14).reshape(2,7)%101
    logits=m(ids); loss=nn.functional.cross_entropy(logits.reshape(-1,101),((ids*3+1)%101).reshape(-1))
    loss.backward()
    assert logits.shape==(2,7,101) and torch.isfinite(loss)
    count=sum(p.numel() for p in m.parameters())
    estimate=12*2*32*32+2*101*32
    print(json.dumps({'status':'UNIT_OK','shape':list(logits.shape),'params':count,'leading_formula':estimate}))


def api_audit():
    import torch.distributed.checkpoint as dcp
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,ShardingStrategy
    print(json.dumps({'torch':torch.__version__,'FSDP':str(inspect.signature(FSDP)),
        'strategies':[str(x) for x in ShardingStrategy],
        'full_optim_state_dict':str(inspect.signature(FSDP.full_optim_state_dict)),
        'dcp_save':str(inspect.signature(dcp.save_state_dict)),
        'dcp_load':str(inspect.signature(dcp.load_state_dict))},indent=2))


def compare_nested(left,right,path='root',atol=1e-5,rtol=1e-4):
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape!=right.shape or left.dtype!=right.dtype: raise AssertionError(f'{path}: tensor metadata differs')
        if left.is_floating_point():
            if not torch.allclose(left,right,atol=atol,rtol=rtol):
                raise AssertionError(f'{path}: max_abs={float((left.double()-right.double()).abs().max())}')
            return float((left.double()-right.double()).abs().max()) if left.numel() else 0.0
        if not torch.equal(left,right): raise AssertionError(f'{path}: integer/RNG tensor differs')
        return 0.0
    if isinstance(left,dict) and isinstance(right,dict):
        if left.keys()!=right.keys(): raise AssertionError(f'{path}: keys differ')
        return max((compare_nested(left[k],right[k],f'{path}.{k}',atol,rtol) for k in left),default=0.0)
    if isinstance(left,(list,tuple)) and isinstance(right,(list,tuple)):
        if len(left)!=len(right): raise AssertionError(f'{path}: length differs')
        return max((compare_nested(x,y,f'{path}[{i}]',atol,rtol)
                    for i,(x,y) in enumerate(zip(left,right))),default=0.0)
    if left!=right: raise AssertionError(f'{path}: {left!r} != {right!r}')
    return 0.0


def full_model_and_optim(model,opt,rank):
    if is_fsdp(model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,FullStateDictConfig,StateDictType
        with FSDP.state_dict_type(model,StateDictType.FULL_STATE_DICT,
                                  FullStateDictConfig(offload_to_cpu=True,rank0_only=True)):
            model_state=model.state_dict()
        optim_state=FSDP.full_optim_state_dict(model,opt,rank0_only=True)
        return (model_state,optim_state) if rank==0 else ({},{})
    if rank==0:
        return ({k:v.detach().cpu() for k,v in unwrapped(model).state_dict().items()},opt.state_dict())
    return {},{}


def one_fp32_step(model,opt,a,rank,world,device):
    model.train(); opt.zero_grad(set_to_none=True)
    x,y,_=make_batch(0,rank,world,a.batch,a.seq,a.vocab,device)
    logits=model(x); loss=nn.functional.cross_entropy(logits.reshape(-1,a.vocab),y.reshape(-1))
    finite=torch.tensor(int(torch.isfinite(loss).item()),device=device); dist.all_reduce(finite,op=dist.ReduceOp.MIN)
    if not finite.item(): raise FloatingPointError('correctness loss is non-finite')
    loss.backward()
    grad_ok=torch.tensor(int(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())),device=device)
    dist.all_reduce(grad_ok,op=dist.ReduceOp.MIN)
    if not grad_ok.item(): raise FloatingPointError('correctness gradient is non-finite')
    grad_norm=model.clip_grad_norm_(1.0) if is_fsdp(model) else nn.utils.clip_grad_norm_(model.parameters(),1.0)
    opt.step(); report=loss.detach().float(); dist.all_reduce(report); report/=world
    return float(report.item()),float(grad_norm)


def correctness_check(a,rank,local,world,device):
    if world<2 or a.precision!='fp32' or a.strategy!='full':
        raise ValueError('correctness mode requires >=2 ranks, --strategy full, --precision fp32')
    seed_all(a.seed); raw_ddp=FSDPLabModel(a.vocab,a.dim,a.heads,a.layers)
    ownership_ddp=parameter_coverage(raw_ddp)
    if ownership_ddp['original_numel']>20_000_000: raise ValueError('correctness mode is limited to <=20M params')
    ddp=DDP(raw_ddp.to(device),device_ids=[local]); opt_ddp=torch.optim.AdamW(ddp.parameters(),lr=a.lr)
    seed_all(a.seed); raw_fsdp=FSDPLabModel(a.vocab,a.dim,a.heads,a.layers)
    ownership_fsdp=parameter_coverage(raw_fsdp)
    fsdp,wrap=wrap_model(a,raw_fsdp,rank,local,world,device,ownership_fsdp)
    opt_fsdp=torch.optim.AdamW(fsdp.parameters(),lr=a.lr)
    ddp_loss,ddp_grad=one_fp32_step(ddp,opt_ddp,a,rank,world,device)
    fsdp_loss,fsdp_grad=one_fp32_step(fsdp,opt_fsdp,a,rank,world,device)
    ddp_state,_=full_model_and_optim(ddp,opt_ddp,rank)
    fsdp_state,_=full_model_and_optim(fsdp,opt_fsdp,rank)
    ok=torch.ones((),dtype=torch.int64,device=device); error=''; max_abs=0.0
    if rank==0:
        try:
            max_abs=compare_nested(ddp_state,fsdp_state,'post_step_model',a.compare_atol,a.compare_rtol)
            if not math.isclose(ddp_loss,fsdp_loss,abs_tol=a.compare_atol,rel_tol=a.compare_rtol):
                raise AssertionError(f'global losses differ: {ddp_loss} vs {fsdp_loss}')
            if not math.isclose(ddp_grad,fsdp_grad,abs_tol=a.compare_atol,rel_tol=a.compare_rtol):
                raise AssertionError(f'global grad norms differ: {ddp_grad} vs {fsdp_grad}')
        except AssertionError as exc:
            ok.zero_(); error=str(exc)
    dist.broadcast(ok,src=0)
    if not ok.item(): raise AssertionError(f'DDP_FSDP_MISMATCH: {error}')
    if rank==0:
        row={'status':'DDP_FSDP_EQUIVALENT','global_loss_ddp':ddp_loss,'global_loss_fsdp':fsdp_loss,
             'grad_norm_ddp':ddp_grad,'grad_norm_fsdp':fsdp_grad,'post_step_max_abs':max_abs,'wrap':wrap}
        print(json.dumps(row)); write_jsonl(a.log,row)


def capture_checkpoint(a,path,rank,local,world,device):
    local_args=argparse.Namespace(**vars(a)); local_args.resume_from=path
    seed_all(a.seed); raw=FSDPLabModel(a.vocab,a.dim,a.heads,a.layers); ownership=parameter_coverage(raw)
    if ownership['original_numel']>20_000_000: raise ValueError('compare mode is limited to <=20M params')
    model,_=wrap_model(local_args,raw,rank,local,world,device,ownership)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr); sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.0)
    scaler=torch.cuda.amp.GradScaler(enabled=a.precision=='fp16')
    step,attempted,successful=load_checkpoint(local_args,model,opt,sched,scaler,rank,world,device)
    model.eval(); x,y,_=make_batch(step,rank,world,a.batch,a.seq,a.vocab,device)
    with torch.no_grad(),torch.cuda.amp.autocast(enabled=a.precision=='fp16',dtype=torch.float16):
        logits=model(x); next_loss=nn.functional.cross_entropy(logits.reshape(-1,a.vocab),y.reshape(-1)).float()
    dist.all_reduce(next_loss); next_loss/=world
    model_state,optim_state=full_model_and_optim(model,opt,rank)
    extra=torch.load(Path(path)/f'extra_rank{rank}.pt',map_location='cpu')
    return model_state,optim_state,extra,float(next_loss.item()),(step,attempted,successful)


def resume_compare(a,rank,local,world,device):
    if not a.resume_from or not a.compare_from: raise ValueError('compare mode requires --resume-from and --compare-from')
    first=capture_checkpoint(a,a.resume_from,rank,local,world,device); gc.collect(); torch.cuda.empty_cache()
    second=capture_checkpoint(a,a.compare_from,rank,local,world,device)
    local_error=''
    try: compare_nested(first[2],second[2],'rank_extra',a.compare_atol,a.compare_rtol)
    except AssertionError as exc: local_error=str(exc)
    errors=[None for _ in range(world)]; dist.all_gather_object(errors,local_error)
    ok=torch.tensor(int(not any(errors)),device=device)
    max_model=max_optim=0.0
    if rank==0 and ok.item():
        try:
            max_model=compare_nested(first[0],second[0],'model',a.compare_atol,a.compare_rtol)
            max_optim=compare_nested(first[1],second[1],'optimizer',a.compare_atol,a.compare_rtol)
            if first[4]!=second[4]: raise AssertionError(f'clocks differ: {first[4]} vs {second[4]}')
            if not math.isclose(first[3],second[3],abs_tol=a.compare_atol,rel_tol=a.compare_rtol):
                raise AssertionError(f'next losses differ: {first[3]} vs {second[3]}')
        except AssertionError as exc:
            ok.zero_(); errors[0]=str(exc)
    dist.broadcast(ok,src=0)
    if not ok.item(): raise AssertionError(f'RESUME_MISMATCH: {errors}')
    if rank==0:
        row={'status':'RESUME_EQUIVALENT','step':first[4][0],'next_loss_a':first[3],'next_loss_b':second[3],
             'max_model_abs':max_model,'max_optimizer_abs':max_optim}
        print(json.dumps(row)); write_jsonl(a.log,row)


def run(a,rank,local,world,device,evaluate=False):
    if a.dim%a.heads: raise ValueError('dim must be divisible by heads')
    seed_all(a.seed)
    raw=FSDPLabModel(a.vocab,a.dim,a.heads,a.layers)
    ownership=parameter_coverage(raw)
    total_params=ownership['original_numel']
    model,wrap=wrap_model(a,raw,rank,local,world,device,ownership)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.0)
    amp=a.precision=='fp16'
    scaler=torch.cuda.amp.GradScaler(enabled=amp)
    if a.resume_from:
        start,attempted,successful=load_checkpoint(a,model,opt,sched,scaler,rank,world,device)
    else:
        start,attempted,successful=0,0,0
    if rank==0:
        print(json.dumps({'event':'START','params':total_params,'config':config(a,world),'wrap':wrap},default=str))
    if evaluate:
        if a.eval_batches<1: raise ValueError('--eval-batches must be >= 1')
        eval_id_min=10_000_000; train_id_max=start*world*a.batch-1
        if eval_id_min<=train_id_max:
            raise ValueError(f'eval IDs overlap training IDs: eval_start={eval_id_min}, train_end={train_id_max}')
        model.eval(); total=torch.zeros((),device=device)
        with torch.no_grad():
            for i in range(a.eval_batches):
                x,y,_=make_batch(i,rank,world,a.batch,a.seq,a.vocab,device,True)
                with torch.cuda.amp.autocast(enabled=amp,dtype=torch.float16):
                    z=model(x); loss=nn.functional.cross_entropy(z.reshape(-1,a.vocab),y.reshape(-1))
                total+=loss.float()
        dist.all_reduce(total); total/=world*a.eval_batches
        row={'status':'EVAL_OK','from_step':start,'eval_loss':float(total.item()),'world_size':world,
             'train_id_max':train_id_max,'eval_id_min':eval_id_min,
             'eval_id_max':eval_id_min+a.eval_batches*world*a.batch-1,'disjoint_from_train':True}
        if rank==0: print(json.dumps(row)); write_jsonl(a.log,row)
        return
    if a.steps<=start:
        raise ValueError(f'training target --steps={a.steps} must exceed resume step={start}')
    profiler=None
    if a.profile_steps:
        activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
        profiler=torch.profiler.profile(activities=activities,
            schedule=torch.profiler.schedule(wait=1,warmup=1,active=max(1,a.profile_steps-2),repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(Path(a.trace_dir)/f'rank{rank}')),
            record_shapes=True,profile_memory=True)
        profiler.start()
    torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device); begin=time.perf_counter(); last=begin
    benchmark_start=None; benchmark_elapsed=None; benchmark_start_step=max(start,a.benchmark_warmup)
    for step in range(start,a.steps):
        if benchmark_start is None and step==benchmark_start_step and not a.profile_steps:
            dist.barrier(); torch.cuda.synchronize(device); benchmark_start=time.perf_counter()
        attempted+=1
        model.train(); opt.zero_grad(set_to_none=True)
        x,y,ids=make_batch(step,rank,world,a.batch,a.seq,a.vocab,device)
        with torch.cuda.amp.autocast(enabled=amp,dtype=torch.float16):
            z=model(x); loss=nn.functional.cross_entropy(z.reshape(-1,a.vocab),y.reshape(-1))
        loss_ok=torch.tensor(int(torch.isfinite(loss).item()),device=device)
        dist.all_reduce(loss_ok,op=dist.ReduceOp.MIN)
        if not loss_ok.item(): raise FloatingPointError(f'non-finite loss on at least one rank; local_rank={rank} step={step}')
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        local_grad_ok=all(p.grad is None or bool(torch.isfinite(p.grad).all().item()) for p in model.parameters())
        grad_ok=torch.tensor(int(local_grad_ok),device=device)
        dist.all_reduce(grad_ok,op=dist.ReduceOp.MIN)
        if not grad_ok.item(): raise FloatingPointError(f'non-finite grad on at least one rank; local_rank={rank} step={step}')
        if is_fsdp(model): grad_norm=model.clip_grad_norm_(1.0)
        else: grad_norm=nn.utils.clip_grad_norm_(model.parameters(),1.0)
        norm_ok=torch.tensor(int(math.isfinite(float(grad_norm))),device=device)
        dist.all_reduce(norm_ok,op=dist.ReduceOp.MIN)
        if not norm_ok.item():
            raise FloatingPointError(f'non-finite global grad norm before optimizer step; rank={rank} step={step}')
        old_scale=float(scaler.get_scale())
        scaler.step(opt); scaler.update()
        skipped=float(scaler.get_scale()) < old_scale
        skip_flag=torch.tensor(int(skipped),device=device)
        skip_min=skip_flag.clone(); skip_max=skip_flag.clone()
        dist.all_reduce(skip_min,op=dist.ReduceOp.MIN); dist.all_reduce(skip_max,op=dist.ReduceOp.MAX)
        if skip_min.item()!=skip_max.item(): raise RuntimeError('AMP skip decision diverged across ranks')
        if skipped: raise FloatingPointError(f'AMP skipped optimizer step rank={rank} step={step}')
        successful+=1; sched.step()
        if benchmark_start is not None and successful==a.steps:
            torch.cuda.synchronize(device); dist.barrier(); benchmark_elapsed=time.perf_counter()-benchmark_start
        if profiler: profiler.step()
        done=step+1
        if done%a.log_every==0 or done==a.steps:
            report=loss.detach().float(); dist.all_reduce(report); report/=world
            torch.cuda.synchronize(device); now=time.perf_counter()
            interval=a.log_every if done%a.log_every==0 else done-start
            row={'step':done,'loss':float(report.item()),'world_size':world,'strategy':a.strategy,
                'tokens_per_s':interval*world*a.batch*a.seq/max(now-last,1e-9),
                'max_allocated':torch.cuda.max_memory_allocated(device),'max_reserved':torch.cuda.max_memory_reserved(device),
                'grad_norm':float(grad_norm),'scale':float(scaler.get_scale()),'attempted_updates':attempted,
                'successful_updates':successful,'sample_ids_rank0':ids if rank==0 else None}
            if rank==0: print(json.dumps(row),flush=True); write_jsonl(a.log,row)
            last=now
        if done%a.save_every==0 or done==a.steps:
            save_checkpoint(a,model,opt,sched,scaler,done,attempted,successful,rank,world)
    if profiler: profiler.stop()
    if rank==0:
        benchmark_steps=max(0,a.steps-benchmark_start_step)
        benchmark_tokens=benchmark_steps*world*a.batch*a.seq
        print(json.dumps({'status':'TRAIN_OK','steps':a.steps,'attempted_updates':attempted,
            'successful_updates':successful,'world_size':world,'strategy':a.strategy,
            'steady_start_step':benchmark_start_step,'steady_steps':benchmark_steps,
            'steady_seconds':benchmark_elapsed,
            'steady_tokens_per_s':benchmark_tokens/benchmark_elapsed if benchmark_elapsed else None,
            'run_seconds_including_aux':time.perf_counter()-begin}))


def main():
    a=args_parser()
    if a.mode=='unit': unit_test(); return
    if a.mode=='api': api_audit(); return
    rank,local,world,device=init_dist()
    try:
        if a.mode=='correctness': correctness_check(a,rank,local,world,device)
        elif a.mode=='compare': resume_compare(a,rank,local,world,device)
        else: run(a,rank,local,world,device,evaluate=a.mode=='eval')
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


if __name__=='__main__': main()
PY
chmod +x src/fsdp_lab.py
```

### 执行命令

```bash
python -m py_compile src/fsdp_lab.py
python src/fsdp_lab.py --mode unit
python src/fsdp_lab.py --mode api | tee evidence/api-audit.txt
sha256sum src/fsdp_lab.py | tee evidence/source.sha256
```

### 预期观测（估算/示例，不是实测）

CPU 输出 `UNIT_OK`、shape `[2,7,101]` 和真实参数数；API 审计列出 FSDP 签名、枚举、DCP save/load 签名。

### 验收条件

静态/CPU 均退出 0；API 中有 `FULL_SHARD`、DCP 两接口；源码哈希落盘。Hybrid 是否支持此时只做初筛，最终由 8 卡 smoke 确认。

### 若失败，先看什么，再改什么

先核对 Python 与 torch 路径；API 缺失则标 `ENV-BLOCKED`，不升级。CPU shape 错先修模型/label，不进入 GPU。

### 当日证据清单

环境清单、`evidence/api-audit.txt`、`evidence/source.sha256`、命令与退出码。

## Day 2：单卡一批与 2 卡 DDP/Full Shard tiny smoke

### 为什么做

按静态→CPU→单 batch→smoke 顺序验证 CUDA/AMP、初始化和最小分片通信。

### 输入与前置检查

Day 1 通过；至少两张获分配 GPU。V100 用 FP16；tiny FP32 对照用于数值定位。

### 本日要创建/修改的文件

创建独立 run 的日志和 checkpoint 目录，不改源码。

### 实现

单卡使用 DDP wrapper 建立 CUDA reference；两卡 smoke 后用 `correctness` 模式在同一进程组内，从相同初始化与全局 sample IDs 分别执行 DDP/Full Shard 一步，汇聚 full state 并比较 global loss、global grad norm 与更新后参数。

### 执行命令

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node=1 src/fsdp_lab.py --mode train --strategy ddp \
  --steps 1 --save-every 1 --log-every 1 --batch 2 --seq 32 --vocab 512 --dim 128 --heads 4 --layers 2 \
  --precision fp32 --checkpoint-root artifacts/checkpoints/onebatch --log logs/onebatch.jsonl

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 src/fsdp_lab.py --mode train --strategy ddp \
  --steps 20 --save-every 20 --log-every 5 --batch 1 --seq 32 --vocab 512 --dim 128 --heads 4 --layers 2 \
  --precision fp32 --checkpoint-root artifacts/checkpoints/tiny_ddp --log logs/tiny_ddp.jsonl

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 src/fsdp_lab.py --mode train --strategy full \
  --steps 20 --save-every 20 --log-every 5 --batch 1 --seq 32 --vocab 512 --dim 128 --heads 4 --layers 2 \
  --precision fp32 --checkpoint-root artifacts/checkpoints/tiny_full --log logs/tiny_full.jsonl

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 src/fsdp_lab.py --mode correctness --strategy full \
  --batch 1 --seq 32 --vocab 512 --dim 128 --heads 4 --layers 2 --precision fp32 \
  --log logs/ddp-fsdp-correctness.jsonl
```

### 预期观测（估算/示例，不是实测）

三条训练均 `TRAIN_OK`；START 的 ownership 给出 wrap 前每个 block/root 的原始唯一参数覆盖，且实际 wrapper 数通过断言。correctness 输出 `DDP_FSDP_EQUIVALENT`；FP32 使用预注册容差，不要求逐 bit 相等。

### 验收条件

单 batch、两个20-step smoke和 correctness 均退出0；参数 ownership union 等于原模型且无重复；DDP/FSDP global loss、grad norm、更新后 full parameter 在容差内；DCP checkpoint 有 `.metadata`、每 rank extra、全文件 hash manifest，以及绑定该 manifest 的 `COMPLETED`。

### 若失败，先看什么，再改什么

单卡失败先排 AMP（当前 FP32）和模型；DDP 过而 Full hang，查 FSDP wrap/所有 rank 控制流；DCP 报错对照 Day 1 签名，不改 torch 版本。

### 当日证据清单

训练 JSONL、`logs/ddp-fsdp-correctness.jsonl`、START ownership/wrapper 输出、checkpoint 文件表、退出码。

## Day 3：同 world-size checkpoint/resume 与独立 eval

### 为什么做

证明恢复的是训练状态而不只是权重，并从新进程组独立加载评估。

### 输入与前置检查

Day 2 的两卡 Full Shard 成功；目标目录必须是新目录，脚本拒绝覆盖已有 step。

### 本日要创建/修改的文件

创建独立 `control_full/ckpt_step000020` 与 `resume_full/ckpt_step000010/000020`，以及 train/compare/eval 日志。

### 实现

每次保存由所有 rank 在 step-versioned 临时目录调用 DCP；scheduler/scaler/RNG/config 每 rank 保存；rank0 校验并 hash 全文件，原子发布目录后再写绑定 manifest 的完成标志。恢复在任何 DCP load 前严格核对相同 torch patch、world/strategy/wrap/precision 和文件 hash；compare 对照 uninterrupted control 的 full model/optimizer、每-rank extra 和下一固定训练 batch loss。

### 执行命令

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 src/fsdp_lab.py --mode train --strategy full \
  --steps 20 --save-every 20 --log-every 20 --batch 1 --seq 64 --vocab 2048 --dim 256 --heads 8 --layers 4 \
  --precision fp16 --checkpoint-root artifacts/checkpoints/control_full --log logs/control_full.jsonl

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 src/fsdp_lab.py --mode train --strategy full \
  --steps 10 --save-every 10 --log-every 5 --batch 1 --seq 64 --vocab 2048 --dim 256 --heads 8 --layers 4 \
  --precision fp16 --checkpoint-root artifacts/checkpoints/resume_full --log logs/resume_full.jsonl

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 src/fsdp_lab.py --mode train --strategy full \
  --steps 20 --save-every 10 --log-every 5 --batch 1 --seq 64 --vocab 2048 --dim 256 --heads 8 --layers 4 \
  --precision fp16 --resume-from artifacts/checkpoints/resume_full/ckpt_step000010 \
  --checkpoint-root artifacts/checkpoints/resume_full --log logs/resume_full.jsonl

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 src/fsdp_lab.py --mode compare --strategy full \
  --batch 1 --seq 64 --vocab 2048 --dim 256 --heads 8 --layers 4 --precision fp16 \
  --resume-from artifacts/checkpoints/control_full/ckpt_step000020 \
  --compare-from artifacts/checkpoints/resume_full/ckpt_step000020 --log logs/resume-equivalence.jsonl

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 src/fsdp_lab.py --mode eval --strategy full \
  --eval-batches 10 --batch 1 --seq 64 --vocab 2048 --dim 256 --heads 8 --layers 4 --precision fp16 \
  --resume-from artifacts/checkpoints/resume_full/ckpt_step000020 \
  --checkpoint-root artifacts/checkpoints/resume_full --log logs/eval_full.jsonl

find artifacts/checkpoints/control_full artifacts/checkpoints/resume_full -type f -printf '%p %s\n' | sort | tee evidence/resume-files.txt
```

### 预期观测（估算/示例，不是实测）

resume 第二段从step10开始并结束于20；compare 输出 `RESUME_EQUIVALENT`，证明 full model/optimizer、scaler/scheduler/RNG/clocks 和 next loss 与连续20步一致；eval 新进程组输出 `EVAL_OK, from_step:20`、有限 loss、训练 ID 上界、独立评估区间与 `disjoint_from_train=true`。

### 验收条件

step/attempted/successful 连续；control/resume 的 step20 均完整；compare 与 eval 退出0；loader 实际执行精确 torch patch gate，而不是只把版本写进 extra。

### 若失败，先看什么，再改什么

先查完成标志、`.metadata`、每 rank extra；next loss 异常查 RNG/sample offset/scaler/optimizer；不可把 2 卡 checkpoint 用 1 卡命令当普通 resume。

### 当日证据清单

`logs/control_full.jsonl`、`logs/resume_full.jsonl`、`logs/resume-equivalence.jsonl`、`logs/eval_full.jsonl`、`evidence/resume-files.txt` 与全部退出码。

## Day 4：0.6B Full Shard 4/8 卡与 Hybrid 能力门

### 为什么做

测 8-way 最省状态的 Full Shard，并在真实拓扑与本机 API 都允许时验证 2×4 Hybrid。

### 输入与前置检查

Day 3 通过；先确认 46 层约 0.612B 参数的 4 卡 5-step probe。正式参数固定：`V=16384,D=1024,H=16,L=46,S=256,B_local=1`。

### 本日要创建/修改的文件

创建 `formal_full4`、`formal_full8`、`formal_hybrid8` 各自日志/checkpoint。

### 实现

Hybrid 固定候选 rows `0..3/4..7` 与 columns `0-4/1-5/2-6/3-7`；必须先由拓扑核验。所有 rank 以相同顺序创建全部 group。

### 执行命令

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 src/fsdp_lab.py --mode train --strategy full \
  --steps 5 --save-every 5 --log-every 1 --batch 1 --seq 256 --vocab 16384 --dim 1024 --heads 16 --layers 46 \
  --precision fp16 --checkpoint-root artifacts/checkpoints/probe_full4 --log logs/probe_full4.jsonl

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 src/fsdp_lab.py --mode train --strategy full \
  --steps 100 --benchmark-warmup 20 --save-every 100 --log-every 100 --batch 1 --seq 256 --vocab 16384 --dim 1024 --heads 16 --layers 46 \
  --precision fp16 --checkpoint-root artifacts/checkpoints/formal_full4 --log logs/formal_full4.jsonl

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc-per-node=8 src/fsdp_lab.py --mode train --strategy full \
  --steps 100 --benchmark-warmup 20 --save-every 100 --log-every 100 --batch 1 --seq 256 --vocab 16384 --dim 1024 --heads 16 --layers 46 \
  --precision fp16 --checkpoint-root artifacts/checkpoints/formal_full8 --log logs/formal_full8.jsonl
```

只有确认物理拓扑与候选 group 一致后才执行 Hybrid：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc-per-node=8 src/fsdp_lab.py --mode train --strategy hybrid \
  --steps 20 --benchmark-warmup 5 --save-every 20 --log-every 20 --batch 1 --seq 256 --vocab 16384 --dim 1024 --heads 16 --layers 46 \
  --precision fp16 --checkpoint-root artifacts/checkpoints/formal_hybrid8 --log logs/formal_hybrid8.jsonl
```

若输出 `ENV-BLOCKED`，保存原错并停止 Hybrid；不升级。若20-step通过，再以新路径把 `--steps/--save-every/--log-every` 同时改为100、`--benchmark-warmup` 改为20做正式 run。

### 预期观测（估算/示例，不是实测）

Full8 状态下界低于 Full4，但通信范围更大；Hybrid 理想状态约四路切分，峰值可能高于 Full8、吞吐可能因拓扑更好而改善，也可能没有改善。没有预填胜者。

### 验收条件

Full4/8 各完成100 step，loss有限、checkpoint可恢复，并报告同为80步的 profiler-off `steady_tokens_per_s`；Hybrid 要么完整通过，要么附 API/运行错误标 `ENV-BLOCKED`，不能伪造。

### 若失败，先看什么，再改什么

probe OOM 先确认参数量与 wrap 生效，再统一降层数/seq 并用新配置名；Hybrid hang 查 group 顺序和真实 rank 映射；loss 非有限先在 tiny FP32 复现。

### 当日证据清单

拓扑、START wrap/group、三份日志/checkpoint 或 Hybrid 阻塞证据、环境/退出码（公司内部）。

## Day 5：activation checkpointing A/B 与短 profiler

### 为什么做

区分状态、activation 与 all-gather buffer 主导的显存，并量化重算代价。

### 输入与前置检查

Day 4 至少一个 Full4 正式配置可跑；A/B 固定模型、global batch、seq、precision、world、step。

### 本日要创建/修改的文件

创建 profiler-off 的 `ac_perf_off/on` 性能日志/checkpoint，以及独立 `ac_trace_off/on` 的每-rank短 trace。

### 实现

`--activation-checkpoint` 用 PyTorch2.1 checkpoint wrapper 包 block。性能 A/B 预热10步后计40步，窗口内不写日志/checkpoint且关闭 profiler；另起8步诊断 run，每 rank 独立 trace，诊断吞吐不进入 A/B。

### 执行命令

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 src/fsdp_lab.py --mode train --strategy full \
  --steps 50 --benchmark-warmup 10 --save-every 50 --log-every 50 --batch 1 --seq 256 --vocab 16384 --dim 1024 --heads 16 --layers 46 \
  --precision fp16 --checkpoint-root artifacts/checkpoints/ac_perf_off --log logs/ac_perf_off.jsonl

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 src/fsdp_lab.py --mode train --strategy full \
  --steps 50 --benchmark-warmup 10 --save-every 50 --log-every 50 --batch 1 --seq 256 --vocab 16384 --dim 1024 --heads 16 --layers 46 \
  --precision fp16 --activation-checkpoint --checkpoint-root artifacts/checkpoints/ac_perf_on \
  --log logs/ac_perf_on.jsonl

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 src/fsdp_lab.py --mode train --strategy full \
  --steps 8 --profile-steps 6 --save-every 8 --log-every 8 --batch 1 --seq 256 --vocab 16384 --dim 1024 --heads 16 --layers 46 \
  --precision fp16 --checkpoint-root artifacts/checkpoints/ac_trace_off --log logs/ac_trace_off.jsonl --trace-dir traces/ac_trace_off

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 src/fsdp_lab.py --mode train --strategy full \
  --steps 8 --profile-steps 6 --save-every 8 --log-every 8 --batch 1 --seq 256 --vocab 16384 --dim 1024 --heads 16 --layers 46 \
  --precision fp16 --activation-checkpoint --checkpoint-root artifacts/checkpoints/ac_trace_on \
  --log logs/ac_trace_on.jsonl --trace-dir traces/ac_trace_on

find traces -type f -printf '%p %s\n' | sort | tee evidence/trace-files.txt
```

### 预期观测（估算/示例，不是实测）

checkpoint-on 通常降低 activation 峰值并增加计算时间；具体幅度不预设。`ac_perf_*` 输出可比的 `steady_tokens_per_s`；`ac_trace_*` 应能看到 all-gather/reduce-scatter 候选事件与重算额外算子，但其时间不作性能值。

### 验收条件

A/B 唯一配置差异是 activation checkpoint 开关和输出路径；两个性能窗口同为40步且 profiler-off、无窗口内日志/保存；记录 allocated/reserved/steady tokens/s。trace 只用于归因并另存文件表。

### 若失败，先看什么，再改什么

on 模式 shape/梯度错误先在 tiny 配置复现 checkpoint wrapper；无显存下降先检查 activation 是否本来不主导；trace 过大缩 active 窗口，不导出公司 trace。

### 当日证据清单

四份 JSONL、性能配置 diff、两组 steady summary、trace 文件表/哈希、内部瓶颈说明。

## Day 6：incomplete 负测、恢复复核与收口

### 为什么做

验证 loader 在最危险的半成品目录上 fail closed，并用新进程完成最终评估。

### 输入与前置检查

Day 3 完整 checkpoint 可用；故障注入只创建新空目录，不破坏成功产物。

### 本日要创建/修改的文件

创建 `artifacts/checkpoints/incomplete/`、`evidence/incomplete.*` 和最终清单。

### 实现

loader 在任何 DCP I/O 前检查 `COMPLETED`；缺标志即抛 `INCOMPLETE_CHECKPOINT`。

### 执行命令

```bash
mkdir -p artifacts/checkpoints/incomplete
printf 'partial\n' > artifacts/checkpoints/incomplete/some-shard.tmp
set +e
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 src/fsdp_lab.py --mode eval --strategy full \
  --eval-batches 1 --batch 1 --seq 64 --vocab 2048 --dim 256 --heads 8 --layers 4 --precision fp16 \
  --resume-from artifacts/checkpoints/incomplete --checkpoint-root artifacts/checkpoints/unused \
  --log logs/should_not_exist.jsonl > evidence/incomplete.stdout 2>&1
code=$?
set -e
test "$code" -ne 0
grep -q INCOMPLETE_CHECKPOINT evidence/incomplete.stdout
printf 'EXPECTED_FAILURE exit=%s\n' "$code" | tee evidence/incomplete.result

python - <<'PY'
import json
from pathlib import Path
for p in sorted(Path('logs').glob('*.jsonl')):
    rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    bad=[r for r in rows if 'loss' in r and not isinstance(r['loss'],(int,float))]
    assert not bad
    print(p, len(rows), rows[-1].get('status','TRAIN_LOG') if rows else 'EMPTY')
PY
```

### 预期观测（估算/示例，不是实测）

负测命令非零退出且命中 `INCOMPLETE_CHECKPOINT`；不会创建成功 eval 日志。汇总列出每个真实日志的记录数。

### 验收条件

负测 fail closed；Full Shard 有 CPU→单 batch→20-step→100-step→save/resume→独立 eval 完整链；Hybrid 有实测或明确 `ENV-BLOCKED`；A/B 和纸面预算均可追溯。

### 若失败，先看什么，再改什么

若负测被加载，立即停止并修 loader 前置检查；若 grep 不匹配先读完整 stderr，不把任意失败当正确负测；汇总失败定位具体日志，不手工删行。

### 当日证据清单

`evidence/incomplete.*`、所有 manifest/完成标志、环境/API/源码哈希、配置差异、最终 `VERIFIED/PENDING/ENV-BLOCKED/INCONCLUSIVE` 表。

## 4. 恢复与兼容边界

- 本实现只承诺相同 torch patch、相同 world size、相同策略/wrap/config 的恢复。
- 8→4 world-size migration 是 stretch；必须走本机 2.1 官方支持的重分片或 tiny full-state 验证，不能手拼 shard。
- Hybrid DCP 若本机文档要求只由一个 shard group 写，应按其明确 process group 调整并重新做负测；不能让 replica 组覆盖同一文件。
- `use_orig_params=False` 被写入实现语义；改变它要视为新实验。
- 保存期间磁盘满、进程终止或任一 rank 失败，临时目录或无 `COMPLETED` 的已发布目录都不可加载，只能回到上一个完整 checkpoint；残留目录需按公司留存/清理流程另行隔离，脚本不会覆盖。

## 5. 最终报告字段

每个 run 保存：命令、源码 SHA-256、torch/CUDA/NCCL、GPU/拓扑、策略/group、wrap unit、参数数、precision、batch/seq/global batch、activation checkpoint、step、loss/grad norm/scaler、峰值 allocated/reserved、正式 `steady_tokens_per_s`、含辅助操作的 run 时间、DCP 用时与大小、退出码、恢复来源和状态标签。所有数字只有在对应原始日志存在时才可标 `VERIFIED`。
