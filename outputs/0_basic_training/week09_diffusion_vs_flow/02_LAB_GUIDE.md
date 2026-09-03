# Week 09 实践篇：从零执行 Diffusion vs Flow Matching 受控实验

> 核验日期：2026-09-03。文中结果均为待运行的预期，不是实测。公司 8×V100/Linux、PyTorch 2.1 与拓扑为用户自述；V100 走 FP16、通常无原生 BF16。公司 Task-P、代码、日志、trace、图片、rollout、checkpoint 和性能数字不得导出。

## 0. 两条轨道与结论上限

Track A 是本文完整创建的合成 conditional-action 实验，能从空目录跑完 static→CPU unit→单 batch→smoke→resume→四主 run→独立 eval→DDP→故障注入。它验证目标函数和实验系统，只能对该 surrogate 下结论。

Track B 是固定 LeRobot v0.6.0 + 获批 Task-P 的真实 MultiTask DiT 集成。v0.6.0 固定 commit 为 `30da8e687a6dfc617fcd94afc367ac7071c376ce`，代码 Apache-2.0；其声明 Python `>=3.12`、torch `>=2.7`，与公司 PyTorch 2.1 冲突，故公司默认 `ENV-BLOCKED`，不升级。只有已有、独立、获批且兼容的环境才继续官方文档链；Track A 结果绝不冒充 Track B。

资源路线：公司 V100 是 Track A FP16/DDP 主证据并在兼容门允许时跑 Track B；个人 5070Ti 可公开复跑 Track A，不接收公司内容；H100 可单列结果，不能替代 V100。止损：D/F config 除 objective 外意外不同、有效 windows 差 >1%、非有限 loss/grad、AMP 连续 skip、mask 全零未报错、resume step 错、数据/license 未批准或任何路径需要升级公司 torch。

## 1. 来源、许可、下载与离线替代

| 对象 | ID/revision | 许可 | 在线获取 | 离线方案/完整性 |
|---|---|---|---|---|
| Track A model/data | 文内 `src/objective_lab.py`；synthetic formula | 项目内部教学代码/现场生成 | 无下载 | SHA-256 + seed/config manifest |
| LeRobot | v0.6.0 / `30da8e687a6dfc617fcd94afc367ac7071c376ce` | Apache-2.0 | Day 6 精确 clone/fetch | 公司批准镜像/归档；解包后 `git rev-parse` 必须同 SHA |
| Task-P | 必须由数据 owner 给 repo/path、revision、split、schema、license | 当前未知，待批准 | 不提供公共替代来冒充 | 缺 manifest 即 `DATA-BLOCKED`；不得搬到个人机器 |
| DDPM / Flow Matching | arXiv `2006.11239` / `2210.02747` | 论文版权 | 官方 arXiv | 可预先批准 PDF；实验不依赖网络 |

官方链接：[MultiTask DiT](https://huggingface.co/docs/lerobot/main/en/multi_task_dit)、[LeRobot v0.6.0](https://github.com/huggingface/lerobot/tree/v0.6.0)、[DDPM](https://arxiv.org/abs/2006.11239)、[Flow Matching](https://arxiv.org/abs/2210.02747)。2026-09-03 核验。

## 2. 环境预检（复用现有环境，不建 conda/venv）

```bash
which python
python -VV
python -m pip --version
python -m pip freeze > /tmp/week09-pip-freeze.txt
python - <<'PY'
import os, torch
print({'torch':torch.__version__,'file':torch.__file__,'cuda':torch.version.cuda,
       'available':torch.cuda.is_available(),'count':torch.cuda.device_count(),
       'CVD':os.getenv('CUDA_VISIBLE_DEVICES')})
for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i); print(i,p.name,p.total_memory,(p.major,p.minor))
PY
nvidia-smi -L
nvidia-smi topo -m
```

Track A 只依赖已有 PyTorch；没有安装动作，回滚为零。公司不是已批准 2.1.x 时先确认，不自行换版本。正式默认 `B=16,T=8,A=6,C=12,D=128,L=4`，显存很小；先 20-step smoke。Track B 安装 dry-run 放在 Day 6，检测冲突即停。

## 3. 最终目录

```text
week09-objectives/
├── src/
│   ├── objective_lab.py
│   └── run_controlled.sh          # Day 3 创建，Day 4/5 复用
├── reports/pre_registration.md
├── artifacts/checkpoints/
├── evidence/
├── logs/
└── traces/
```

## Day 1：创建预注册和完整核心实现

### 为什么做

先冻结唯一主变量、预算、target、mask、sampler 和 checkpoint 语义，避免结果出来后改规则。

### 输入与前置检查

环境预检通过；无需外部数据或模型。

### 本日要创建/修改的文件

创建目录、`reports/pre_registration.md` 与 `src/objective_lab.py`。

### 实现

```bash
mkdir -p week09-objectives/{src,reports,artifacts/checkpoints,evidence,logs,traces}
cd week09-objectives
cat > reports/pre_registration.md <<'MD'
# Diffusion vs Flow 预注册（运行前冻结）

- 唯一主变量：`objective=diffusion|flow` 及配套 `DDIM-like|Euler` sampler。
- 固定：synthetic formula、模型、初始化策略、batch、optimizer/LR、precision、successful updates、valid windows、checkpoint step、eval samples/initial noise/hardware。
- 主实验：D/F × seed 0/1，各 1000 successful updates；运行顺序预注册为 `D(seed0)→F(seed0)→F(seed1)→D(seed1)`（ABBA），第二 seed 未完成则结论 INCONCLUSIVE。
- eval 沿用同一 ABBA；每个卡数的 profile 做 `D(rep0)→F(rep0)→F(rep1)→D(rep1)`。不得见到结果后改序或只保留最快一次，报告每个配对差值及其中位数/范围。
- 系统控制：同一物理 GPU 集合、driver/power/application-clock policy、precision 和 exclusive allocation；每个 run 前要求无外来 GPU 进程、utilization≤5%、temperature≤50°C。记录运行前后 clocks/temp/utilization/load 与全程 `nvidia-smi dmon`；管理员已锁 clock 就保持原锁，未锁则全组保持同一动态策略，操作者不临时改系统设置。
- 评估：固定 checkpoint，sampling steps 5/10/20；action MSE、endpoint、jerk、latency。
- 判定：两 seed 多数共同指标支持才可 PASS-DIFFUSION/PASS-FLOW；质量/延迟分裂为 TRADEOFF；系统不公平为 FAIL-SYSTEM。
MD
cat > src/objective_lab.py <<'PY'
#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import random
import time
import uuid
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['unit','train','eval','compare'],default='train')
    p.add_argument('--objective',choices=['diffusion','flow'],default='diffusion')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--horizon',type=int,default=8)
    p.add_argument('--action-dim',type=int,default=6)
    p.add_argument('--cond-dim',type=int,default=12)
    p.add_argument('--width',type=int,default=128)
    p.add_argument('--heads',type=int,default=8)
    p.add_argument('--layers',type=int,default=4)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--precision',choices=['fp32','fp16'],default='fp16')
    p.add_argument('--log-every',type=int,default=20)
    p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--checkpoint-root',default='artifacts/checkpoints/run')
    p.add_argument('--resume-from',default='')
    p.add_argument('--checkpoint',default='')
    p.add_argument('--log',default='logs/run.jsonl')
    p.add_argument('--eval-samples',type=int,default=128)
    p.add_argument('--sampler-steps',default='5,10,20')
    p.add_argument('--eval-seed',type=int,default=91023)
    p.add_argument('--eval-warmup',type=int,default=10)
    p.add_argument('--benchmark-warmup',type=int,default=100)
    p.add_argument('--checkpoint-a',default='')
    p.add_argument('--checkpoint-b',default='')
    p.add_argument('--atol',type=float,default=0.0)
    p.add_argument('--rtol',type=float,default=0.0)
    p.add_argument('--profile-steps',type=int,default=0)
    p.add_argument('--trace-dir',default='traces')
    p.add_argument('--inject-bad-mask',action='store_true')
    return p.parse_args()


class ActionTransformer(nn.Module):
    def __init__(self,horizon,action_dim,cond_dim,width,heads,layers):
        super().__init__()
        self.horizon=horizon
        self.action_in=nn.Linear(action_dim,width)
        self.cond=nn.Linear(cond_dim,width)
        self.time=nn.Sequential(nn.Linear(4,width),nn.SiLU(),nn.Linear(width,width))
        self.pos=nn.Parameter(torch.zeros(1,horizon,width))
        block=nn.TransformerEncoderLayer(width,heads,4*width,dropout=0.0,batch_first=True,norm_first=True)
        self.blocks=nn.TransformerEncoder(block,layers)
        self.norm=nn.LayerNorm(width)
        self.out=nn.Linear(width,action_dim)
        nn.init.normal_(self.pos,std=0.02)

    def forward(self,noisy,time_value,condition):
        if noisy.ndim!=3 or time_value.shape!=(noisy.shape[0],1,1):
            raise ValueError('shape contract violated')
        t=time_value[:,0,0]
        tf=torch.stack([t,t*t,torch.sin(math.pi*t),torch.cos(math.pi*t)],dim=-1)
        x=self.action_in(noisy)+self.cond(condition)[:,None,:]+self.time(tf)[:,None,:]+self.pos[:,:noisy.shape[1]]
        return self.out(self.norm(self.blocks(x)))


def setup_train_dist():
    world=int(os.getenv('WORLD_SIZE','1')); rank=int(os.getenv('RANK','0')); local=int(os.getenv('LOCAL_RANK','0'))
    if not torch.cuda.is_available(): raise RuntimeError('GPU train/eval required; CPU is reserved for --mode unit')
    torch.cuda.set_device(local); device=torch.device('cuda',local)
    if world>1: dist.init_process_group('nccl',timeout=timedelta(minutes=5))
    return rank,local,world,device


def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def synthetic_batch(sample_ids,horizon,action_dim,cond_dim,device):
    sid=sample_ids.float().reshape(-1,1)
    ck=torch.arange(cond_dim,dtype=torch.float32).reshape(1,-1)
    cond=torch.sin(sid*0.017+ck*0.31)+0.5*torch.cos(sid*0.013-ck*0.19)
    base=(cond*torch.linspace(0.2,1.0,cond_dim)).sum(-1,keepdim=True)/cond_dim
    hp=torch.arange(horizon,dtype=torch.float32).reshape(1,horizon,1)
    ad=torch.arange(action_dim,dtype=torch.float32).reshape(1,1,action_dim)
    clean=torch.tanh(base.reshape(-1,1,1)+0.25*hp+0.17*ad+0.1*torch.sin(base.reshape(-1,1,1)*ad))
    lengths=horizon-(sample_ids%3)
    mask=(torch.arange(horizon).reshape(1,horizon)<lengths.reshape(-1,1)).unsqueeze(-1)
    return cond.to(device),clean.to(device),mask.to(device)


def stream_cpu(a,step,rank,world):
    """A counter-based stream: objective and process RNG consumption cannot change it."""
    ids=torch.arange(step*world*a.batch+rank*a.batch,step*world*a.batch+(rank+1)*a.batch,dtype=torch.long)
    g=torch.Generator(device='cpu').manual_seed(1_000_003*a.seed+97_003*step+7_919*rank+17)
    noise=torch.randn((a.batch,a.horizon,a.action_dim),generator=g)
    time_value=torch.rand((a.batch,1,1),generator=g)*0.998+0.001
    return {'sample_ids':ids,'noise':noise,'time':time_value}


def path_and_target(objective,clean,noise,t):
    if objective=='diffusion':
        alpha=torch.cos(t*math.pi/2); sigma=torch.sin(t*math.pi/2)
        return alpha*clean+sigma*noise,noise
    return (1-t)*noise+t*clean,clean-noise


def masked_terms(pred,target,mask):
    if pred.shape!=target.shape or mask.shape!=target.shape[:2]+(1,): raise ValueError('MASK_SHAPE_ERROR')
    local_num=((pred.float()-target.float()).square()*mask.float()).sum()
    local_den=mask.float().sum()*target.shape[-1]
    return local_num,local_den


def global_masked_loss(pred,target,mask,world):
    """DDP averages gradients, hence W*N_r/sum(D_r) is the per-rank backward scalar."""
    local_num,local_den=masked_terms(pred,target,mask)
    global_den=local_den.detach().clone()
    if world>1: dist.all_reduce(global_den,op=dist.ReduceOp.SUM)
    torch._assert_async(global_den>0,'ALL_ZERO_MASK')
    backward_loss=local_num*world/global_den.clamp_min(1)
    return backward_loss,local_num.detach(),global_den


def model_config(a):
    return {k:getattr(a,k) for k in ('horizon','action_dim','cond_dim','width','heads','layers')}


def train_config(a,world):
    return {'objective':a.objective,'seed':a.seed,'batch':a.batch,'lr':a.lr,'precision':a.precision,
            'world_size':world,'optimizer':'AdamW','optimizer_weight_decay':0.01,
            'scheduler':'constant-v1','ema':'none','checkpoint_schema':2,
            'data_formula_revision':'synthetic-v1','time_noise_revision':'step-rank-v2',**model_config(a)}


def atomic_save(obj,path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+f'.tmp.{os.getpid()}')
    torch.save(obj,tmp); os.replace(tmp,p)


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def write_jsonl(path,row):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf-8') as f: f.write(json.dumps(row,sort_keys=True)+'\n')


def save_ckpt(a,model,opt,scheduler,scaler,step,attempted,successful,valid_scalars,rank,world):
    target=Path(a.checkpoint_root)/f'ckpt_step{step:06d}'
    exists=[target.exists() if rank==0 else None]
    if world>1: dist.broadcast_object_list(exists,src=0)
    if exists[0]: raise RuntimeError(f'refuse overwrite {target}')
    stage=target.parent/f'.{target.name}.staging-{uuid.uuid4().hex}' if rank==0 else None
    stages=[str(stage) if rank==0 else None]
    if world>1: dist.broadcast_object_list(stages,src=0)
    stage=Path(stages[0])
    if rank==0:
        target.parent.mkdir(parents=True,exist_ok=True); stage.mkdir()
        atomic_save({'step':step,'attempted_updates':attempted,'successful_updates':successful,
                     'valid_scalars_seen':int(valid_scalars.item()),
                     'model':(model.module if isinstance(model,DDP) else model).state_dict(),
                     'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
                     'scaler':scaler.state_dict(),'ema':None,'train_config':train_config(a,world),
                     'torch_version':torch.__version__},stage/'shared.pt')
    if world>1: dist.barrier()
    atomic_save({'python':random.getstate(),'torch':torch.get_rng_state(),
                  'cuda':torch.cuda.get_rng_state(torch.cuda.current_device())},stage/f'rng_rank{rank}.pt')
    atomic_save(stream_cpu(a,step,rank,world),stage/f'next_stream_rank{rank}.pt')
    if world>1: dist.barrier()
    if rank==0:
        names=['shared.pt']+[f'rng_rank{r}.pt' for r in range(world)]+[f'next_stream_rank{r}.pt' for r in range(world)]
        manifest={'schema':2,'step':step,'world_size':world,'files':[
          {'path':n,'bytes':(stage/n).stat().st_size,'sha256':sha256_file(stage/n)} for n in names]}
        raw=json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()
        (stage/'manifest.json').write_bytes(raw)
        marker={'schema':2,'step':step,'manifest_sha256':hashlib.sha256(raw).hexdigest()}
        (stage/'COMPLETED.json').write_text(json.dumps(marker,sort_keys=True),encoding='utf-8')
        os.replace(stage,target)
    if world>1: dist.barrier()


def verify_ckpt(target):
    target=Path(target); marker_path=target/'COMPLETED.json'; manifest_path=target/'manifest.json'
    if not marker_path.is_file() or not manifest_path.is_file(): raise RuntimeError('INCOMPLETE_CHECKPOINT')
    marker=json.loads(marker_path.read_text(encoding='utf-8')); raw=manifest_path.read_bytes()
    if marker.get('schema')!=2 or marker.get('manifest_sha256')!=hashlib.sha256(raw).hexdigest():
        raise RuntimeError('CHECKPOINT_MARKER_MISMATCH')
    manifest=json.loads(raw)
    for entry in manifest['files']:
        p=target/entry['path']
        if not p.is_file() or p.stat().st_size!=entry['bytes'] or sha256_file(p)!=entry['sha256']:
            raise RuntimeError(f'CHECKPOINT_FILE_MISMATCH {p}')
    return manifest


def same_tree(a,b):
    if torch.is_tensor(a) and torch.is_tensor(b): return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict) and isinstance(b,dict): return a.keys()==b.keys() and all(same_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,type(a)): return len(a)==len(b) and all(same_tree(x,y) for x,y in zip(a,b))
    return a==b


def resume(a,model,opt,scheduler,scaler,rank,world,device):
    target=Path(a.resume_from)
    manifest=verify_ckpt(target)
    if manifest['world_size']!=world: raise RuntimeError('resume world mismatch')
    shared=torch.load(target/'shared.pt',map_location=device)
    if shared['train_config']!=train_config(a,world): raise RuntimeError('resume config/world mismatch')
    incompatible=(model.module if isinstance(model,DDP) else model).load_state_dict(shared['model'],strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys: raise RuntimeError(str(incompatible))
    opt.load_state_dict(shared['optimizer']); scheduler.load_state_dict(shared['scheduler']); scaler.load_state_dict(shared['scaler'])
    if shared.get('ema','MISSING') is not None: raise RuntimeError('EMA_CONTRACT_MISMATCH')
    rng=torch.load(target/f'rng_rank{rank}.pt',map_location='cpu')
    frozen_stream=torch.load(target/f'next_stream_rank{rank}.pt',map_location='cpu')
    if not same_tree(frozen_stream,stream_cpu(a,int(shared['step']),rank,world)):
        raise RuntimeError('NEXT_STREAM_MISMATCH')
    random.setstate(rng['python']); torch.set_rng_state(rng['torch']); torch.cuda.set_rng_state(rng['cuda'],device=device)
    if world>1: dist.barrier()
    return (int(shared['step']),int(shared['attempted_updates']),
            int(shared['successful_updates']),int(shared['valid_scalars_seen']))


def unit_test():
    clean=torch.tensor([[[1.,2.],[3.,4.]]]); noise=torch.tensor([[[.5,-.5],[1.,-1.]]]); t=torch.tensor([[[.25]]])
    mask=torch.tensor([[[True],[False]]])
    dx,dt=path_and_target('diffusion',clean,noise,t); fx,ft=path_and_target('flow',clean,noise,t)
    assert dx.shape==fx.shape==dt.shape==ft.shape==(1,2,2)
    dn,dd=masked_terms(dt,dt,mask); fn,fd=masked_terms(ft,ft,mask)
    assert dn.item()==fn.item()==0 and dd.item()==fd.item()==2
    _,zero_den=masked_terms(dt,dt,torch.zeros_like(mask))
    assert zero_den.item()==0
    rank_num=torch.tensor([2.,90.]); rank_den=torch.tensor([2.,18.])
    weighted=(rank_num.sum()/rank_den.sum()).item(); rank_mean=(rank_num/rank_den).mean().item()
    assert math.isclose(weighted,4.6) and math.isclose(rank_mean,3.0) and weighted!=rank_mean
    torch.manual_seed(9); m=ActionTransformer(2,2,3,16,4,1)
    cond=torch.ones(1,3); a=m(dx,t,cond); b=m(dx,t,cond)
    assert torch.equal(a,b) and a.shape==(1,2,2)
    print(json.dumps({'status':'UNIT_OK','diff_shape':list(dx.shape),'flow_shape':list(fx.shape),
                      'global_weighted_loss':weighted,'wrong_rank_mean':rank_mean,'deterministic':True}))


def finite_gate(tensor,label,world):
    flag=torch.isfinite(tensor).all().to(dtype=torch.int32)
    if world>1: dist.all_reduce(flag,op=dist.ReduceOp.MIN)
    torch._assert_async(flag.bool(),label)


def train(a,rank,local,world,device):
    if a.width%a.heads: raise ValueError('width must be divisible by heads')
    if not hasattr(torch,'_assert_async'): raise RuntimeError('torch._assert_async required; audit the installed PyTorch')
    seed_all(7000+a.seed)
    raw=ActionTransformer(**model_config(a)).to(device)
    model=DDP(raw,device_ids=[local]) if world>1 else raw
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.0)
    amp=a.precision=='fp16'; scaler=torch.cuda.amp.GradScaler(enabled=amp)
    if a.resume_from:
        start,attempted,successful,valid_seen=resume(a,model,opt,scheduler,scaler,rank,world,device)
    else:
        start,attempted,successful,valid_seen=0,0,0,0
    valid_scalars=torch.tensor(valid_seen,dtype=torch.int64,device=device)
    profiler=None
    if a.profile_steps:
        profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
          schedule=torch.profiler.schedule(wait=1,warmup=1,active=max(1,a.profile_steps-2),repeat=1),
          on_trace_ready=torch.profiler.tensorboard_trace_handler(str(Path(a.trace_dir)/f'rank{rank}')),
          record_shapes=True,profile_memory=True); profiler.start()
    torch.cuda.reset_peak_memory_stats(device)
    timed_from=max(start,a.benchmark_warmup); timer_start=None; timer_end=None
    if timed_from<a.steps and start>=timed_from:
        if world>1: dist.barrier()
        torch.cuda.synchronize(); timer_start=torch.cuda.Event(enable_timing=True); timer_start.record()
    last_grad=torch.zeros((),device=device); last_local_num=torch.zeros((),device=device); last_global_den=torch.ones((),device=device)
    for step in range(start,a.steps):
        if step==timed_from and timer_start is None:
            if world>1: dist.barrier()
            torch.cuda.synchronize(); timer_start=torch.cuda.Event(enable_timing=True); timer_start.record()
        attempted+=1
        stream=stream_cpu(a,step,rank,world)
        cond,clean,mask=synthetic_batch(stream['sample_ids'],a.horizon,a.action_dim,a.cond_dim,device)
        if a.inject_bad_mask: mask=torch.zeros_like(mask)
        noise=stream['noise'].to(device); t=stream['time'].to(device)
        noisy,target=path_and_target(a.objective,clean,noise,t)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp,dtype=torch.float16): pred=model(noisy,t,cond)
        loss,local_num,global_den=global_masked_loss(pred,target,mask,world)
        finite_gate(loss,'NON_FINITE_GLOBAL_LOSS',world)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        grad_flag=torch.ones((),dtype=torch.int32,device=device)
        for parameter in model.parameters():
            if parameter.grad is not None: grad_flag.mul_(torch.isfinite(parameter.grad).all().to(torch.int32))
        if world>1: dist.all_reduce(grad_flag,op=dist.ReduceOp.MIN)
        torch._assert_async(grad_flag.bool(),'NON_FINITE_GRAD_ON_AT_LEAST_ONE_RANK')
        last_grad=nn.utils.clip_grad_norm_(model.parameters(),1.0)
        finite_gate(last_grad,'NON_FINITE_GRAD_NORM',world)
        scaler.step(opt); scaler.update(); scheduler.step()
        successful+=1; valid_scalars.add_(global_den.to(torch.int64))
        last_local_num=local_num; last_global_den=global_den
        if profiler: profiler.step()
        done=step+1
        if done==a.steps and timer_start is not None:
            timer_end=torch.cuda.Event(enable_timing=True); timer_end.record()
        if done%a.log_every==0 or done==a.steps:
            report_num=last_local_num.clone()
            if world>1: dist.all_reduce(report_num,op=dist.ReduceOp.SUM)
            report=report_num/last_global_den
            torch.cuda.synchronize()
            row={'step':done,'objective':a.objective,'seed':a.seed,'loss':float(report.item()),
                  'attempted_updates':attempted,'successful_updates':successful,'skipped_updates':attempted-successful,
                  'global_valid_scalars_seen':int(valid_scalars.item()),'grad_norm':float(last_grad.item()),
                  'scale':float(scaler.get_scale()),'world_size':world,
                  'max_cuda_bytes':torch.cuda.max_memory_allocated(device)}
            if rank==0: print(json.dumps(row),flush=True); write_jsonl(a.log,row)
        if done%a.save_every==0 or done==a.steps:
            save_ckpt(a,model,opt,scheduler,scaler,done,attempted,successful,valid_scalars,rank,world)
    if profiler: profiler.stop()
    local_elapsed_ms=None; slowest_rank_elapsed_ms=None
    if timer_start is not None and timer_end is not None:
        torch.cuda.synchronize(); local_elapsed_ms=timer_start.elapsed_time(timer_end)
        elapsed=torch.tensor(local_elapsed_ms,dtype=torch.float64,device=device)
        if world>1: dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
        slowest_rank_elapsed_ms=float(elapsed.item())
    if rank==0:
        timed_updates=max(0,a.steps-timed_from)
        print(json.dumps({'status':'TRAIN_OK','objective':a.objective,'seed':a.seed,'step':a.steps,
          'attempted_updates':attempted,'successful_updates':successful,'training_model_calls_per_rank':successful,
          'steady_state_warmup_updates':timed_from-start,'steady_state_updates':timed_updates,
          'timing_start_barrier':world>1,'rank0_local_steady_state_ms':local_elapsed_ms,
          'slowest_rank_steady_state_ms':slowest_rank_elapsed_ms,
          'global_updates_per_s':None if not slowest_rank_elapsed_ms else timed_updates*1000/slowest_rank_elapsed_ms,
          'global_samples_per_s':None if not slowest_rank_elapsed_ms else timed_updates*a.batch*world*1000/slowest_rank_elapsed_ms}))


def build_from_checkpoint(path,device):
    p=Path(path)
    verify_ckpt(p)
    shared=torch.load(p/'shared.pt',map_location=device); cfg=shared['train_config']
    mc={k:cfg[k] for k in ('horizon','action_dim','cond_dim','width','heads','layers')}
    model=ActionTransformer(**mc).to(device)
    incompatible=model.load_state_dict(shared['model'],strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys: raise RuntimeError(str(incompatible))
    model.eval()
    return model,cfg,int(shared['step']),shared


@torch.no_grad()
def generate(model,cfg,cond,initial_noise,steps):
    x=initial_noise.clone(); b=x.shape[0]
    if cfg['objective']=='flow':
        for k in range(steps):
            t=torch.full((b,1,1),k/steps,device=x.device)
            x=x+model(x,t,cond)/steps
    else:
        for k in range(steps):
            # Python scalars avoid a device->host conversion inside the timed loop.
            t=torch.full((b,1,1),0.999*(1-k/steps),device=x.device)
            s=torch.full((b,1,1),0.999*(1-(k+1)/steps),device=x.device)
            eps=model(x,t,cond)
            at=torch.cos(t*math.pi/2).clamp_min(1e-4); st=torch.sin(t*math.pi/2)
            clean=(x-st*eps)/at
            x=torch.cos(s*math.pi/2)*clean+torch.sin(s*math.pi/2)*eps
    return x


def percentile(values,q):
    ordered=sorted(values)
    return ordered[max(0,math.ceil(q*len(ordered))-1)]


@torch.no_grad()
def evaluate(a,device):
    model,cfg,ckpt_step,_=build_from_checkpoint(a.checkpoint,device)
    for steps in [int(x) for x in a.sampler_steps.split(',')]:
        for i in range(a.eval_warmup):
            ids=torch.tensor([30_000_000+i]); cond,clean,_=synthetic_batch(ids,cfg['horizon'],cfg['action_dim'],cfg['cond_dim'],device)
            initial=torch.randn(clean.shape,generator=torch.Generator().manual_seed(a.eval_seed+1_000_000+i)).to(device)
            with torch.cuda.amp.autocast(enabled=cfg['precision']=='fp16',dtype=torch.float16):
                generate(model,cfg,cond,initial,steps)
        torch.cuda.synchronize()
        starts=[]; ends=[]; predictions=[]; cleans=[]; masks=[]; lengths=[]
        for i in range(a.eval_samples):
            sample_id=20_000_000+i
            ids=torch.tensor([sample_id]); cond,clean,mask=synthetic_batch(ids,cfg['horizon'],cfg['action_dim'],cfg['cond_dim'],device)
            initial=torch.randn(clean.shape,generator=torch.Generator().manual_seed(a.eval_seed+i)).to(device)
            start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.cuda.amp.autocast(enabled=cfg['precision']=='fp16',dtype=torch.float16):
                pred=generate(model,cfg,cond,initial,steps)
            end.record(); starts.append(start); ends.append(end)
            predictions.append(pred); cleans.append(clean); masks.append(mask)
            lengths.append(cfg['horizon']-(sample_id%3))
        # One synchronization per (checkpoint,K), never one host sync per timed sample.
        torch.cuda.synchronize()
        latency=[s.elapsed_time(e) for s,e in zip(starts,ends)]
        pred=torch.cat(predictions).float(); clean=torch.cat(cleans).float(); valid=torch.cat(masks).float()
        sse=((pred-clean).square()*valid).sum(); scalar_count=valid.sum()*cfg['action_dim']
        torch._assert_async(scalar_count>0,'EVAL_ALL_ZERO_MASK')
        row_index=torch.arange(a.eval_samples,device=device); last=torch.tensor(lengths,device=device)-1
        endpoint=torch.linalg.vector_norm(pred[row_index,last]-clean[row_index,last],dim=-1).mean()
        jerk=pred[:,3:]-3*pred[:,2:-1]+3*pred[:,1:-2]-pred[:,:-3]
        jerk_mask=valid[:,3:]
        jerk_mean=(jerk.square()*jerk_mask).sum()/(jerk_mask.sum()*cfg['action_dim']).clamp_min(1)
        row={'status':'EVAL_OK','objective':cfg['objective'],'seed':cfg['seed'],'checkpoint_step':ckpt_step,
              'sampler_steps':steps,'samples':a.eval_samples,'action_mse':float((sse/scalar_count).item()),
              'endpoint_l2':float(endpoint.item()),
              'jerk':float(jerk_mean.item()),'latency_mean_ms':sum(latency)/len(latency),
              'latency_p50_ms':percentile(latency,.50),'latency_p95_ms':percentile(latency,.95),
              'warmup_chunks':a.eval_warmup,'timing_boundary':'model_sampler_only_cuda_events',
              'sampler_model_calls_per_chunk':steps,'total_eval_model_calls':steps*a.eval_samples,
              'actions_per_chunk':cfg['horizon'],
              'p50_ms_per_action_if_full_chunk_consumed':percentile(latency,.50)/cfg['horizon'],
              'eval_sample_id_start':20_000_000,'eval_seed':a.eval_seed,'telemetry_schema':'df-common-v2'}
        print(json.dumps(row,sort_keys=True)); write_jsonl(a.log,row)


def compare_tree(a,b,path,atol,rtol,failures):
    if torch.is_tensor(a) and torch.is_tensor(b):
        if a.shape!=b.shape or a.dtype!=b.dtype or not torch.allclose(a.cpu(),b.cpu(),atol=atol,rtol=rtol):
            failures.append(path)
        return
    if isinstance(a,dict) and isinstance(b,dict):
        if a.keys()!=b.keys(): failures.append(path+'.keys'); return
        for key in a: compare_tree(a[key],b[key],f'{path}.{key}',atol,rtol,failures)
        return
    if isinstance(a,(list,tuple)) and isinstance(b,type(a)):
        if len(a)!=len(b): failures.append(path+'.len'); return
        for i,(x,y) in enumerate(zip(a,b)): compare_tree(x,y,f'{path}[{i}]',atol,rtol,failures)
        return
    if a!=b: failures.append(path)


@torch.no_grad()
def fixed_next_loss(path,device):
    model,cfg,step,_=build_from_checkpoint(path,device); a=argparse.Namespace(**cfg)
    total_num=torch.zeros((),device=device); total_den=torch.zeros((),device=device)
    for rank in range(cfg['world_size']):
        stream=stream_cpu(a,step,rank,cfg['world_size'])
        cond,clean,mask=synthetic_batch(stream['sample_ids'],cfg['horizon'],cfg['action_dim'],cfg['cond_dim'],device)
        noise=stream['noise'].to(device); t=stream['time'].to(device)
        noisy,target=path_and_target(cfg['objective'],clean,noise,t)
        with torch.cuda.amp.autocast(enabled=cfg['precision']=='fp16',dtype=torch.float16): pred=model(noisy,t,cond)
        n,d=masked_terms(pred,target,mask); total_num.add_(n); total_den.add_(d)
    torch.cuda.synchronize(); return float((total_num/total_den).item())


def compare_checkpoints(a,device):
    pa,pb=Path(a.checkpoint_a),Path(a.checkpoint_b); ma,mb=verify_ckpt(pa),verify_ckpt(pb)
    sa=torch.load(pa/'shared.pt',map_location='cpu'); sb=torch.load(pb/'shared.pt',map_location='cpu')
    failures=[]
    for key in ('step','attempted_updates','successful_updates','valid_scalars_seen','train_config','model',
                'optimizer','scheduler','scaler','ema'):
        compare_tree(sa[key],sb[key],key,a.atol,a.rtol,failures)
    if ma['world_size']!=mb['world_size']: failures.append('manifest.world_size')
    for rank in range(ma['world_size']):
        for stem in ('rng_rank','next_stream_rank'):
            xa=torch.load(pa/f'{stem}{rank}.pt',map_location='cpu'); xb=torch.load(pb/f'{stem}{rank}.pt',map_location='cpu')
            compare_tree(xa,xb,f'{stem}{rank}',a.atol,a.rtol,failures)
    next_a=fixed_next_loss(pa,device); next_b=fixed_next_loss(pb,device)
    if not math.isclose(next_a,next_b,abs_tol=a.atol,rel_tol=a.rtol): failures.append('fixed_next_loss')
    row={'status':'COMPARE_OK' if not failures else 'COMPARE_FAIL','checkpoint_a':str(pa),'checkpoint_b':str(pb),
         'atol':a.atol,'rtol':a.rtol,'fixed_next_loss_a':next_a,'fixed_next_loss_b':next_b,
         'compared_state':['model','optimizer','scheduler','scaler','ema=none','python/cpu/cuda_rng','clocks','next_stream'],
         'failures':failures}
    print(json.dumps(row,sort_keys=True)); write_jsonl(a.log,row)
    if failures: raise SystemExit(4)


def main():
    a=parse_args()
    if a.mode=='unit': unit_test(); return
    rank,local,world,device=setup_train_dist()
    try:
        if a.mode=='train': train(a,rank,local,world,device)
        elif a.mode=='eval':
            if world!=1: raise RuntimeError('independent eval uses one process/GPU')
            evaluate(a,device)
        else:
            if world!=1: raise RuntimeError('checkpoint comparison uses one process/GPU')
            if not a.checkpoint_a or not a.checkpoint_b: raise ValueError('--checkpoint-a/--checkpoint-b required')
            compare_checkpoints(a,device)
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


if __name__=='__main__': main()
PY
chmod +x src/objective_lab.py
```

### 执行命令

```bash
python -m py_compile src/objective_lab.py
python src/objective_lab.py --mode unit
sha256sum src/objective_lab.py reports/pre_registration.md | tee evidence/source.sha256
```

### 预期观测（估算/示例，不是实测）

输出 `UNIT_OK`，两个 path shape `[1,2,2]`、zero mask 被拒绝、模型重复前向逐元素相同。

### 验收条件

静态/CPU 单测退出 0；预注册在任何训练日志之前冻结；两个文件 hash 存档。

### 若失败，先看什么，再改什么

先看 shape/mask traceback，再核对 PyTorch 导入；不得跳过单测进入 GPU。

### 当日证据清单

预注册、源码 hash、CPU 输出、环境清单与退出码。

## Day 2：单 batch 与 D/F 20-step smoke

### 为什么做

确认 GPU、FP16、两 target、checkpoint 和共同日志字段都能运行。

### 输入与前置检查

Day 1 通过；使用同一张 GPU、同一 seed/model/batch，输出路径必须不同。

### 本日要创建/修改的文件

只创建 `one_*`、`smoke_*` checkpoint/log；不改代码。

### 实现

四条训练命令只有 objective、输出路径和 run 名不同；sample/noise/time 由 seed+step 确定。

### 执行命令

```bash
CUDA_VISIBLE_DEVICES=0 python src/objective_lab.py --mode train --objective diffusion --seed 0 --steps 1 \
  --save-every 1 --log-every 1 --precision fp16 --checkpoint-root artifacts/checkpoints/one_d --log logs/one_d.jsonl
CUDA_VISIBLE_DEVICES=0 python src/objective_lab.py --mode train --objective flow --seed 0 --steps 1 \
  --save-every 1 --log-every 1 --precision fp16 --checkpoint-root artifacts/checkpoints/one_f --log logs/one_f.jsonl
CUDA_VISIBLE_DEVICES=0 python src/objective_lab.py --mode train --objective diffusion --seed 0 --steps 20 \
  --save-every 20 --log-every 5 --precision fp16 --checkpoint-root artifacts/checkpoints/smoke_d --log logs/smoke_d.jsonl
CUDA_VISIBLE_DEVICES=0 python src/objective_lab.py --mode train --objective flow --seed 0 --steps 20 \
  --save-every 20 --log-every 5 --precision fp16 --checkpoint-root artifacts/checkpoints/smoke_f --log logs/smoke_f.jsonl
```

### 预期观测（估算/示例，不是实测）

每条输出有限 loss/grad norm、`successful_updates=attempted_updates`（若无 overflow）、有效 windows、峰值显存和 `TRAIN_OK`。D/F loss 数值无需相同。

### 验收条件

4 条退出 0；D/F 参数量和除 objective 外配置一致；step 20 checkpoint 都有 shared/RNG/manifest/完成标志。

### 若失败，先看什么，再改什么

单 batch 先查 dtype/shape；仅一个 objective NaN 就用相同 tiny FP32 重跑定位公式；不得只为该组换 batch/LR。

### 当日证据清单

4 份日志、4 个 manifest、命令/退出码、配置字段 diff。

## Day 3：恢复测试与四个等预算正式 run

### 为什么做

证明 resume 连续，并获得最低两 seed 配对证据。

### 输入与前置检查

Day 2 通过；预注册目标为每组 1000 successful update。时间不足可统一缩短，但四组同预算且结论降级。

### 本日要创建/修改的文件

创建恢复 run、`formal_{d,f}_s{0,1}` 独立目录/日志、`src/run_controlled.sh` 和配对分布证据。

### 实现

先用 flow 同时做 continuous `0→20` 与 split `0→10→20`，再用 compare 模式逐项比较 model/optimizer/constant scheduler/GradScaler/RNG/clock/`EMA=none`/下一随机流和 fixed next loss。随后四 run 每个从 step 0 训练 1000 次成功更新，严格按预注册 `D0→F0→F1→D1`（ABBA）执行。`run_controlled.sh` 固定 GPU UUID、driver、power/application-clock policy，拒绝外来 GPU 进程或高温/高利用率起跑，并保存 before/after 与全程 telemetry；它不擅自修改公司 clock。正式吞吐只取 warmup 100 后的 steady-state；`log/save=1000`，不让每步 host 同步污染 D/F 的计时。单卡使用本地 elapsed；DDP 使用起点 barrier 后各 rank elapsed 的 MAX。

### 执行命令

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CUDA_VISIBLE_DEVICES=0 python src/objective_lab.py --mode train --objective flow --seed 9 --steps 20 \
  --save-every 20 --log-every 20 --benchmark-warmup 5 --precision fp16 \
  --checkpoint-root artifacts/checkpoints/resume_control --log logs/resume_control.jsonl
CUDA_VISIBLE_DEVICES=0 python src/objective_lab.py --mode train --objective flow --seed 9 --steps 10 \
  --save-every 10 --log-every 10 --benchmark-warmup 5 --precision fp16 \
  --checkpoint-root artifacts/checkpoints/resume_split --log logs/resume_split_0_10.jsonl
CUDA_VISIBLE_DEVICES=0 python src/objective_lab.py --mode train --objective flow --seed 9 --steps 20 \
  --save-every 20 --log-every 20 --benchmark-warmup 5 --precision fp16 \
  --resume-from artifacts/checkpoints/resume_split/ckpt_step000010 \
  --checkpoint-root artifacts/checkpoints/resume_split --log logs/resume_split_10_20.jsonl
CUDA_VISIBLE_DEVICES=0 python src/objective_lab.py --mode compare \
  --checkpoint-a artifacts/checkpoints/resume_control/ckpt_step000020 \
  --checkpoint-b artifacts/checkpoints/resume_split/ckpt_step000020 \
  --atol 0 --rtol 0 --log logs/resume_compare.jsonl

cat > src/run_controlled.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

control_static_policy() {
  local key baseline current
  key=${CONTROL_GPU_IDS//,/_}
  baseline="evidence/control_static_gpu_${key}.csv"
  current="$(nvidia-smi -i "$CONTROL_GPU_IDS" \
    --query-gpu=index,uuid,driver_version,persistence_mode,power.limit,clocks.applications.graphics,clocks.applications.memory \
    --format=csv,noheader)"
  if [[ -f "$baseline" ]]; then
    [[ "$current" == "$(<"$baseline")" ]] || { printf '%s\n' 'STATIC_GPU_POLICY_CHANGED' >&2; return 1; }
  else
    printf '%s\n' "$current" > "$baseline"
  fi
}

wait_for_idle_start() {
  local attempt apps rows temp util ok seen
  for attempt in $(seq 1 60); do
    apps="$(nvidia-smi -i "$CONTROL_GPU_IDS" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null | sed -n '/^[[:space:]]*[0-9][0-9]*[[:space:]]*$/p' || true)"
    rows="$(nvidia-smi -i "$CONTROL_GPU_IDS" --query-gpu=temperature.gpu,utilization.gpu \
      --format=csv,noheader,nounits)"
    ok=1; seen=0
    [[ -z "$apps" ]] || ok=0
    while IFS=',' read -r temp util; do
      temp=${temp//[[:space:]]/}; util=${util//[[:space:]]/}; seen=$((seen+1))
      [[ "$temp" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] || ok=0
      (( temp <= 50 && util <= 5 )) || ok=0
    done <<< "$rows"
    (( seen > 0 && ok == 1 )) && return 0
    sleep 10
  done
  printf '%s\n' 'IDLE_START_GATE_TIMEOUT' >&2
  return 1
}

snapshot_system() {
  local dest=$1
  {
    date --iso-8601=ns
    printf 'host=%s CUDA_VISIBLE_DEVICES=%s SLURM_JOB_ID=%s\n' "$(hostname)" "${CUDA_VISIBLE_DEVICES:-}" "${SLURM_JOB_ID:-NONE}"
    uptime
    nvidia-smi -i "$CONTROL_GPU_IDS" \
      --query-gpu=timestamp,index,uuid,pstate,temperature.gpu,utilization.gpu,utilization.memory,clocks.current.graphics,clocks.current.memory,power.draw,power.limit,memory.used \
      --format=csv
    nvidia-smi -i "$CONTROL_GPU_IDS" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv || true
  } > "$dest"
}

controlled_run() {
  local label=$1 monitor_pid rc
  shift
  mkdir -p evidence logs
  control_static_policy
  wait_for_idle_start
  snapshot_system "evidence/${label}.before.txt"
  nvidia-smi dmon -i "$CONTROL_GPU_IDS" -s pucvmet -d 1 -o DT > "logs/${label}.dmon.csv" &
  monitor_pid=$!
  sleep 1
  kill -0 "$monitor_pid" 2>/dev/null || { printf '%s\n' 'DMON_START_FAILED' >&2; return 1; }
  set +e
  "$@" 2>&1 | tee "logs/${label}.stdout.log"
  rc=${PIPESTATUS[0]}
  set -e
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  snapshot_system "evidence/${label}.after.txt"
  printf 'exit=%s\n' "$rc" > "evidence/${label}.exit.txt"
  [[ "$rc" -eq 0 ]]
}
SH
chmod +x src/run_controlled.sh
source src/run_controlled.sh
export CONTROL_GPU_IDS=0
TRAIN_ORDER=('diffusion:0:d_s0' 'flow:0:f_s0' 'flow:1:f_s1' 'diffusion:1:d_s1')
for spec in "${TRAIN_ORDER[@]}"; do
  IFS=: read -r objective seed suffix <<< "$spec"
  controlled_run "formal_${suffix}" env CUDA_VISIBLE_DEVICES=0 python src/objective_lab.py \
    --mode train --objective "$objective" --seed "$seed" --steps 1000 \
    --save-every 1000 --log-every 1000 --benchmark-warmup 100 --precision fp16 \
    --checkpoint-root "artifacts/checkpoints/formal_${suffix}" --log "logs/formal_${suffix}.jsonl"
done

python - <<'PY'
import json,statistics
from pathlib import Path
def result(label):
    rows=[]
    for line in Path(f'logs/formal_{label}.stdout.log').read_text().splitlines():
        try: row=json.loads(line)
        except json.JSONDecodeError: continue
        if row.get('status')=='TRAIN_OK': rows.append(row)
    if len(rows)!=1: raise SystemExit(f'EXPECTED_ONE_TRAIN_OK {label} {len(rows)}')
    return rows[0]
pairs=[]
for seed in (0,1):
    d,f=result(f'd_s{seed}'),result(f'f_s{seed}')
    pairs.append({'seed':seed,'order':'D→F' if seed==0 else 'F→D',
      'diffusion':d,'flow':f,
      'delta_f_minus_d':{k:f[k]-d[k] for k in ('slowest_rank_steady_state_ms','global_samples_per_s')}})
summary={k:{'paired_deltas':[p['delta_f_minus_d'][k] for p in pairs],
            'median':statistics.median(p['delta_f_minus_d'][k] for p in pairs),
            'min':min(p['delta_f_minus_d'][k] for p in pairs),
            'max':max(p['delta_f_minus_d'][k] for p in pairs)}
         for k in ('slowest_rank_steady_state_ms','global_samples_per_s')}
out={'schema':1,'pre_registered_order':['D_s0','F_s0','F_s1','D_s1'],'pairs':pairs,'summary':summary}
Path('evidence/train_paired_distribution.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
print(json.dumps(out,sort_keys=True))
PY
```

### 预期观测（估算/示例，不是实测）

恢复日志从 10 接续到 20；四组 successful update、world size、batch 和有效 windows 接近/相同。loss 可各自下降，但不横比数值。ABBA 四 run 均有静态 policy、before/after、dmon、exit 和 stdout 证据；报告保留两个 seed 的配对差值分布，而非只给总体最快值。

### 验收条件

compare 必须 `COMPARE_OK` 且 fixed next loss 精确一致；四个 step-1000 完成 checkpoint；successful updates 和 global valid scalars 必须相同，否则不判算法胜负。ABBA 顺序、GPU 静态 policy、空闲/温度门任一不满足即 `FAIL-SYSTEM`。若目标 GPU/版本不能保证 bitwise deterministic，可预注册非零容差后重跑两条恢复路径，不能事后放宽。

### 若失败，先看什么，再改什么

恢复失败查完成标志与同 world config；某组 AMP skip 查首个异常 step/scale，四组保持共同策略；时间不足统一缩成例如 300 step并标低证据。

### 当日证据清单

恢复日志/manifest、四正式日志/checkpoint、ABBA 顺序、系统 telemetry、`train_paired_distribution.json`、预算核对表、所有退出码。

## Day 4：独立 5/10/20-step 质量—延迟评估

### 为什么做

在共同动作空间比较，而不是比较不同 target 的 train loss。

### 输入与前置检查

Day 3 四个 `ckpt_step001000` 完整；同一张空闲 GPU，batch=1，评估 seed/样本固定。

### 本日要创建/修改的文件

创建四份 eval JSONL、对应系统 telemetry 与配对分布；checkpoint 只读。

### 实现

每种 step 先 10 个 warmup，不计加载；128 个 held-out synthetic IDs 使用固定 initial noise。CUDA event 只包 sampler，所有 event 记录完才一次同步；不在 timed loop 做 `.item()`/device→host。四 checkpoint 严格按 `D0→F0→F1→D1`（ABBA）运行，复用 Day 3 的空闲门和系统 telemetry。D/F 共用同一字段，记录 action MSE、endpoint、jerk、latency mean/p50/p95、每 chunk/总模型调用，以及仅在完整消费 8-action chunk 时成立的摊销 latency/action；汇总保留每 seed、每 K 的配对差值分布。

### 执行命令

```bash
source src/run_controlled.sh
export CONTROL_GPU_IDS=0
EVAL_ORDER=('d:0' 'f:0' 'f:1' 'd:1')
for spec in "${EVAL_ORDER[@]}"; do
  IFS=: read -r obj seed <<< "$spec"
  controlled_run "eval_${obj}_s${seed}" env CUDA_VISIBLE_DEVICES=0 python src/objective_lab.py --mode eval \
    --checkpoint "artifacts/checkpoints/formal_${obj}_s${seed}/ckpt_step001000" \
    --sampler-steps 5,10,20 --eval-samples 128 --eval-seed 91023 \
    --log "logs/eval_${obj}_s${seed}.jsonl"
done

python - <<'PY'
import json,statistics
from pathlib import Path
def rows(label):
    p=Path(f'logs/eval_{label}.jsonl')
    rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    assert [r['sampler_steps'] for r in rows]==[5,10,20]
    assert all(r['samples']==128 and r['status']=='EVAL_OK' for r in rows)
    return {r['sampler_steps']:r for r in rows}
pairs=[]
for seed in (0,1):
    d,f=rows(f'd_s{seed}'),rows(f'f_s{seed}')
    for k in (5,10,20):
        metrics=('action_mse','endpoint_l2','jerk','latency_p50_ms','latency_p95_ms')
        pairs.append({'seed':seed,'sampler_steps':k,'order':'D→F' if seed==0 else 'F→D',
          'diffusion':d[k],'flow':f[k],'delta_f_minus_d':{m:f[k][m]-d[k][m] for m in metrics}})
summary={}
for k in (5,10,20):
    summary[str(k)]={}
    for metric in ('action_mse','endpoint_l2','jerk','latency_p50_ms','latency_p95_ms'):
        values=[p['delta_f_minus_d'][metric] for p in pairs if p['sampler_steps']==k]
        summary[str(k)][metric]={'paired_deltas':values,'median':statistics.median(values),'min':min(values),'max':max(values)}
out={'schema':1,'pre_registered_order':['D_s0','F_s0','F_s1','D_s1'],'pairs':pairs,'summary':summary}
Path('evidence/eval_paired_distribution.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
print(json.dumps(out,sort_keys=True))
PY
```

### 预期观测（估算/示例，不是实测）

每文件三行；每行明确 `timing_boundary=model_sampler_only_cuda_events`、10 个 warmup、128×K 次模型调用。latency 通常随 steps 增加，但质量不保证单调；没有预填 D/F 胜者。ABBA 的 before/after/dmon 与六组（2 seed×3 K）配对差值均保留。

### 验收条件

四 checkpoint × 三 steps 共 12 个点，样本/noise/hardware/边界相同；ABBA 顺序和系统门完整；指标有限（若 jerk 因 horizon<4 才可 NaN，本配置 horizon=8 不应）；不得以未配对的总体均值替代 `eval_paired_distribution.json`。

### 若失败，先看什么，再改什么

加载失败查完成标志/权重 key；diffusion 爆炸查 `t≈1` clamp 与训练 path；latency 异常先查 GPU 共享负载和 warmup，不删离群点。

### 当日证据清单

四 eval JSONL、`eval_paired_distribution.json`、ABBA 顺序、before/after/dmon、GPU/环境状态、评估命令与退出码。

## Day 5：2/4/8 卡短 profile 与 bad-mask 负测

### 为什么做

确认目标函数差异没有被分布式/数据错误污染，并证明全零 mask 会 fail closed。

### 输入与前置检查

Week07 DDP 不变量已通过；正式 profile 固定每卡 batch，明确这是弱扩展，不作为算法训练质量主 run。

### 本日要创建/修改的文件

创建 12 个 profile 日志/trace、系统 telemetry、`profile_paired_distribution.json` 以及 `evidence/bad-mask.*`。

### 实现

同卡数 D/F 的模型/batch/step/profile 窗口一致；每 rank trace 独立目录。每个卡数做 rep0 的 `D→F` 与 rep1 的 `F→D`，形成 ABBA；rep 内 seed 相同，输出目录不复用。计时 warmup=10，开始前由每个 rank 进入 barrier，吞吐分母取各 rank elapsed 的 MAX。每个 GPU 集合各自冻结静态 clock/power policy，复用空闲/温度/共享负载门。

### 执行命令

```bash
source src/run_controlled.sh
for n in 2 4 8; do
  ids=$(seq -s, 0 $((n-1)))
  export CONTROL_GPU_IDS="$ids"
  PROFILE_ORDER=('diffusion:0' 'flow:0' 'flow:1' 'diffusion:1')
  for spec in "${PROFILE_ORDER[@]}"; do
    IFS=: read -r obj rep <<< "$spec"
    controlled_run "profile_${obj}_${n}_r${rep}" env CUDA_VISIBLE_DEVICES="$ids" \
      torchrun --standalone --nproc-per-node="$n" src/objective_lab.py --mode train \
      --objective "$obj" --seed "$rep" --steps 50 --batch 16 --save-every 50 --log-every 50 \
      --benchmark-warmup 10 --precision fp16 --profile-steps 6 --trace-dir "traces/${obj}_${n}_r${rep}" \
      --checkpoint-root "artifacts/checkpoints/profile_${obj}_${n}_r${rep}" \
      --log "logs/profile_${obj}_${n}_r${rep}.jsonl"
  done
done

python - <<'PY'
import json,statistics
from pathlib import Path
def result(obj,n,rep):
    p=Path(f'logs/profile_{obj}_{n}_r{rep}.stdout.log'); rows=[]
    for line in p.read_text().splitlines():
        try: row=json.loads(line)
        except json.JSONDecodeError: continue
        if row.get('status')=='TRAIN_OK': rows.append(row)
    if len(rows)!=1: raise SystemExit(f'EXPECTED_ONE_TRAIN_OK {p} {len(rows)}')
    assert rows[0]['timing_start_barrier'] is True
    return rows[0]
pairs=[]
for n in (2,4,8):
    for rep in (0,1):
        d,f=result('diffusion',n,rep),result('flow',n,rep)
        pairs.append({'world_size':n,'rep':rep,'order':'D→F' if rep==0 else 'F→D',
          'diffusion':d,'flow':f,
          'delta_f_minus_d':{k:f[k]-d[k] for k in ('slowest_rank_steady_state_ms','global_samples_per_s')}})
summary={}
for n in (2,4,8):
    summary[str(n)]={}
    for metric in ('slowest_rank_steady_state_ms','global_samples_per_s'):
        values=[p['delta_f_minus_d'][metric] for p in pairs if p['world_size']==n]
        summary[str(n)][metric]={'paired_deltas':values,'median':statistics.median(values),'min':min(values),'max':max(values)}
out={'schema':1,'per_world_size_order':['D_r0','F_r0','F_r1','D_r1'],'pairs':pairs,'summary':summary,
     'throughput_denominator':'MAX rank elapsed after synchronized start'}
Path('evidence/profile_paired_distribution.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
print(json.dumps(out,sort_keys=True))
PY

set +e
CUDA_VISIBLE_DEVICES=0 python src/objective_lab.py --mode train --objective diffusion --steps 1 \
  --inject-bad-mask --save-every 1 --log-every 1 --checkpoint-root artifacts/checkpoints/badmask \
  --log logs/badmask.jsonl > evidence/bad-mask.stdout 2>&1
code=$?
set -e
test "$code" -ne 0
grep -q ALL_ZERO_MASK evidence/bad-mask.stdout
printf 'EXPECTED_FAILURE exit=%s\n' "$code" | tee evidence/bad-mask.result
```

### 预期观测（估算/示例，不是实测）

正常 run 有限并产出每 rank trace；D/F 训练都只一次模型调用/step。每个卡数的两个配对差值都保留，`global_samples_per_s` 使用最慢 rank elapsed。负测非零且命中 `ALL_ZERO_MASK`。

### 验收条件

每个卡数四组均成功，同 rep 的 D/F 配置仅 objective/output 不同；起点 barrier 和 slowest-rank MAX 字段可查；只报告配对分布，不以单次最快值下结论。静态 policy、起跑温度/utilization 或 exclusive-load 门失败即该配对 `FAIL-SYSTEM`。负测按预期失败；多卡失败不污染单卡算法结果，而标 `FAIL-SYSTEM` 调查。若仅有4卡，8卡为 `NOT-RUN`，不租 H100 补数字。

### 若失败，先看什么，再改什么

hang 回 Week07 collective/rank/data 审计；仅一个 objective 非有限先 tiny FP32；负测成功是严重 bug，先修分母检查。

### 当日证据清单

12 个 stdout/JSONL/dmon/before/after、`profile_paired_distribution.json`、trace 文件表、NCCL/拓扑内部证据、`evidence/bad-mask.*`。

## Day 6：LeRobot v0.6.0 兼容门与最终判定

### 为什么做

把官方 MultiTask DiT 的真实依赖/数据条件与 surrogate 结论明确隔离，避免在公司 2.1 环境盲装。

### 输入与前置检查

Track A 已闭环；联网/仓库镜像须获批。Task-P owner 必须给 revision、split、字段、lag、normalization、camera order 和许可证/用途批准。

### 本日要创建/修改的文件

条件创建 `vendor/lerobot/` 与 `evidence/lerobot-*`；缺网络时从批准离线归档恢复同 commit。不会创建官方训练输出，除非兼容门全部通过。

### 实现

固定 tag 后校验 commit、LICENSE 和 `pyproject.toml`；仅做 pip dry-run，不安装。公司 torch 2.1 与 v0.6.0 声明冲突时结果就是 `ENV-BLOCKED`。

### 执行命令

```bash
mkdir -p vendor evidence
git clone --filter=blob:none --no-checkout https://github.com/huggingface/lerobot.git vendor/lerobot
git -C vendor/lerobot fetch --depth 1 origin tag v0.6.0
git -C vendor/lerobot checkout --detach 30da8e687a6dfc617fcd94afc367ac7071c376ce
test "$(git -C vendor/lerobot rev-parse HEAD)" = "30da8e687a6dfc617fcd94afc367ac7071c376ce"
git -C vendor/lerobot rev-parse HEAD | tee evidence/lerobot-commit.txt
sha256sum vendor/lerobot/LICENSE vendor/lerobot/pyproject.toml | tee evidence/lerobot-source.sha256
grep -nE 'requires-python|torch[<=>]' vendor/lerobot/pyproject.toml | tee evidence/lerobot-runtime-requirements.txt
set +e
python -m pip install --dry-run -e vendor/lerobot > evidence/lerobot-pip-dry-run.txt 2>&1
dry_code=$?
set -e
printf 'dry_run_exit=%s\n' "$dry_code" | tee evidence/lerobot-dry-run-status.txt
python - <<'PY'
import sys,torch
ok=sys.version_info>=(3,12) and tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])>=(2,7)
print('TRACK_B_RUNTIME_OK' if ok else 'ENV-BLOCKED: v0.6.0 requires Python>=3.12 and torch>=2.7; do not upgrade company PyTorch 2.1')
raise SystemExit(0 if ok else 3)
PY
```

最后一条在公司 2.1 路径预期非零；这是能力门，不是要修掉的错误。只有已有批准兼容环境且 Task-P manifest 完整，才按照该固定 commit 的 `multi_task_dit` 文档生成真实配置；先运行该仓库实际提供的 `--help`/配置测试并把确切 key 固定后，再复制 Track A 的 D/F×2 seeds 验收矩阵。本文不猜测 v0.6.0 CLI key，也不以 `main` 文档驱动旧/不同 commit。

### 预期观测（估算/示例，不是实测）

commit/hash 可复核；公司 PyTorch 2.1 被明确判 `ENV-BLOCKED`。若组织确有兼容环境，dry-run 仍需证明不会替换已批准 torch/CUDA 栈。

### 验收条件

Track A 证据完整；Track B 有 commit/license/runtime/data 四门状态。任何一门缺失就不声称 MultiTask DiT 实验完成。

### 若失败，先看什么，再改什么

clone 失败先用批准镜像/归档且校验同 SHA；dry-run 想替换 torch 立即停；Task-P 缺许可/manifest 标 `DATA-BLOCKED`。不在个人 5070Ti 上接收公司 Task-P。

### 当日证据清单

Track A 配置/预算/12 点 eval/负测汇总；Track B commit、LICENSE hash、runtime requirements、dry-run 与数据批准状态；最终标签。

## 4. 最终验收与结论模板

必须逐项可追溯：两 target/sampler 单测；D/F×2 seed 的 ABBA 顺序与配对分布；successful updates/valid windows 差 <1%；5/10/20 的 12 点配对表；checkpoint/resume；bad-mask；DDP 起点 barrier、最慢 rank 吞吐分母和 AB/BA profile 分布；clocks/temp/utilization/共享负载证据；Track B 边界。

结论只允许：`PASS-DIFFUSION`、`PASS-FLOW`、`TRADEOFF`、`INCONCLUSIVE`、`FAIL-SYSTEM`。若只有 Track A，应写“synthetic surrogate 上的结果”；若 Track B 因公司 2.1 阻塞，应写 `ENV-BLOCKED`，不能把 surrogate 改名为 MultiTask DiT。所有公司证据只在内部查看与归档。
