# Week 04 实践篇：从空目录实现 TinyFlowPolicy 完整闭环

> **全篇待执行、待验证。** 主路径使用本机生成的非敏感 `point-mass-v1`，不依赖下载。公司V100上的代码、数据、日志、轨迹、图、trace、checkpoint和性能数字不得导出；个人5070 Ti从本文公开代码独立重跑。公司复用既有Conda，不创建Conda/venv，不升级PyTorch2.1。

## Day 0：环境、目录与依赖冻结

### 为什么做

把模型问题与环境问题分开，并确保后续每个命令引用的路径已创建。

### 输入与前置检查

估算磁盘小于2GB；默认约10M模型，micro-batch64估算显存远低于16GB，但必须从小batch实测；100-step smoke数分钟，正式5000-step可能数小时。

`[公司 Linux]`：

```bash
which python
python -V
python -m pip check
nvidia-smi
df -h .
python -c "import torch; print(torch.__version__,torch.version.cuda,torch.cuda.is_available()); print(torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0),torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else None"
```

`[个人 PowerShell]`：

```powershell
Get-Command python
python -V
python -m pip check
nvidia-smi
Get-PSDrive -PSProvider FileSystem
python -c "import torch; print(torch.__version__,torch.version.cuda,torch.cuda.is_available())"
```

### 本日要创建/修改的文件

`[公司 Linux]`：

```bash
mkdir -p week04-tinyflow/{configs,data,manifests,src,tests,runs,reports}
cd week04-tinyflow
touch src/__init__.py reports/decision.md
python -m pip freeze > manifests/pip-before.txt
```

`[个人 PowerShell]`：

```powershell
New-Item -ItemType Directory -Force week04-tinyflow,week04-tinyflow/configs,week04-tinyflow/data,week04-tinyflow/manifests,week04-tinyflow/src,week04-tinyflow/tests,week04-tinyflow/runs,week04-tinyflow/reports | Out-Null
Set-Location week04-tinyflow
New-Item -ItemType File -Force src/__init__.py,reports/decision.md | Out-Null
python -m pip freeze | Set-Content -Encoding utf8 manifests/pip-before.txt
```

```text
src/make_data.py       point-mass-v1 ID/OOD episodes
src/toy2d.py           2D flow数学oracle
src/tinyflow.py        windows、BC/Flow、EMA、train/eval/resume
tests/test_core.py     target/mask/window/solver
configs/*.json         完整冻结运行配置
data/*.npz             本机生成episode
runs/                  本机证据
```

### 实现

只需要现有torch、NumPy、pytest。公司不安装LeRobot current（其当前依赖要求与PyTorch2.1不相容）。

### 执行命令

`[两者/Python]`：

```bash
python -m pip install --dry-run "numpy<2" "pytest<9"
```

缺包且不会改变torch时才执行同规格install；公司需批准镜像。随后 `python -m pip check`。

### 预期观测（估算/示例，不是实测）

公司V100通常BF16 false；个人环境版本可不同。目录与环境快照存在。

### 验收条件

torch可import、磁盘足够、公司torch仍2.1.x、无环境创建/外部遥测。

### 若失败，按什么顺序查

解释器→pip check→CUDA分配→torch build→管理员；不自行升级CUDA。

### 当日证据清单

`pip-before.txt`、GPU探针、估算预算与止损线。

## Day 1：推导并运行 2D flow oracle

### 为什么做

先在无episode/mask的2D分布证明 `xt`、velocity target和Euler方向，避免把数学bug带入控制数据。

### 输入与前置检查

无需文件或网络；device可CPU，速度慢时用CUDA。

### 本日要创建/修改的文件

```python
# src/toy2d.py
import argparse, json, math, torch
from torch import nn
def time_emb(t,k=16):
    f=2**torch.arange(k,device=t.device,dtype=t.dtype)*math.pi
    return torch.cat([torch.sin(t*f),torch.cos(t*f)],-1)
class Net(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(34,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,2))
    def forward(self,x,t): return self.net(torch.cat([x,time_emb(t)],-1))
def data(n,g,device):
    mode=torch.randint(0,8,(n,),generator=g); angle=mode.float()*math.pi/4
    center=torch.stack([2*torch.cos(angle),2*torch.sin(angle)],-1)
    return (center+0.08*torch.randn(n,2,generator=g)).to(device)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--device",default="cpu"); p.add_argument("--steps",type=int,default=2000); p.add_argument("--samples",type=int,default=4096); a=p.parse_args()
    torch.manual_seed(0); dev=torch.device(a.device); g=torch.Generator().manual_seed(1); m=Net().to(dev); opt=torch.optim.AdamW(m.parameters(),lr=1e-3); first=None
    for step in range(a.steps):
        x1=data(256,g,dev); x0=torch.randn(256,2,generator=g).to(dev); t=torch.rand(256,1,generator=g).to(dev)
        xt=(1-t)*x0+t*x1; target=(x1-x0).detach(); loss=(m(xt,t)-target).square().mean()
        if first is None:first=float(loss)
        opt.zero_grad(); loss.backward(); opt.step()
    z=torch.randn(a.samples,2,generator=g).to(dev)
    with torch.no_grad():
        for i in range(50): z=z+0.02*m(z,torch.full((a.samples,1),i/50,device=dev))
    result={"initial_loss":first,"final_loss":float(loss),"sample_mean":z.mean(0).tolist(),"sample_cov":torch.cov(z.T).tolist()}
    print(json.dumps(result))
if __name__=="__main__":main()
```

### 实现

八个Gaussian mode位于半径2圆周。Euler固定50步；终端只打印mean/cov，图不是依赖。

### 执行命令

`[两者/Python]` CPU最小：

```bash
python -m py_compile src/toy2d.py
python src/toy2d.py --device cpu --steps 20 --samples 128
```

`[公司 Linux]` 或个人CUDA正式toy：

```bash
CUDA_VISIBLE_DEVICES=0 python src/toy2d.py --device cuda --steps 2000 --samples 4096
```

`[个人 PowerShell]` 将前缀替换为 `$env:CUDA_VISIBLE_DEVICES="0"` 后运行Python命令。

### 预期观测（估算/示例，不是实测）

loss应总体下降，sample统计向环形多模态数据移动；mean接近0只是对称性粗检，不能证明八mode全部学到。

### 验收条件

CPU20-step finite；正式toy loss显著低于初值；手工target单测（Day2）通过；50步sample无NaN。

### 若失败，按什么顺序查

手工x0/x1/t→t shape→target是否detach→Euler是否0→1→device/generator→LR。不可通过画图跳过数值单测。

### 当日证据清单

toy命令/config/seed、初末loss、sample mean/cov、状态标签。

## Day 2：生成episode、实现windows和BC oracle

### 为什么做

先固定data/action/success合同并用BC过拟合，证明window、condition与action head正确。

### 输入与前置检查

无外部下载。可选PushT只在公司审批后用于Week05 transfer；本日主命令不依赖它。

### 本日要创建/修改的文件

```python
# src/make_data.py
import argparse, hashlib, json
from pathlib import Path
import numpy as np
def build(n,max_len,seed,goal_scale):
    states=np.zeros((n,max_len,4),np.float32); actions=np.zeros((n,max_len,2),np.float32); goals=np.zeros((n,2),np.float32); lengths=np.zeros(n,np.int64)
    for e in range(n):
        r=np.random.default_rng(seed+e); L=int(r.integers(24,max_len+1)); lengths[e]=L
        pos=r.uniform(-1,1,2).astype(np.float32); vel=np.zeros(2,np.float32); goal=r.uniform(-goal_scale,goal_scale,2).astype(np.float32); goals[e]=goal
        for k in range(L):
            states[e,k]=np.r_[pos,vel]; a=np.clip(2*(goal-pos)-.3*vel,-1,1).astype(np.float32); actions[e,k]=a
            vel=.8*vel+.2*a; pos=pos+.1*vel
    return states,actions,goals,lengths
def main():
    p=argparse.ArgumentParser(); p.add_argument("--episodes",type=int,default=500); p.add_argument("--max-len",type=int,default=64); p.add_argument("--seed",type=int,default=100)
    p.add_argument("--goal-scale",type=float,default=1.0); p.add_argument("--output",required=True); p.add_argument("--manifest",required=True); a=p.parse_args()
    x=build(a.episodes,a.max_len,a.seed,a.goal_scale); Path(a.output).parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.output,states=x[0],actions=x[1],goals=x[2],lengths=x[3])
    h=hashlib.sha256(Path(a.output).read_bytes()).hexdigest(); m={"schema":"point-mass-v1","episodes":a.episodes,"max_len":a.max_len,"seed":a.seed,"goal_scale":a.goal_scale,"sha256":h}
    Path(a.manifest).write_text(json.dumps(m,indent=2)+"\n",encoding="utf-8"); print(json.dumps(m))
if __name__=="__main__":main()
```

创建完整训练核心：

```python
# src/tinyflow.py
import argparse, hashlib, json, math, os, random, statistics, time
from pathlib import Path
import numpy as np, torch
from torch import nn

def load_npz(path): return dict(np.load(path,allow_pickle=False))
def split_eps(n,split):
    return [e for e in range(n) if (e%10<8 and split=="train") or (e%10==8 and split=="dev") or (e%10==9 and split=="test")]
def stats(raw):
    es=split_eps(len(raw["lengths"]),"train"); cond=[]; act=[]
    for e in es:
        L=int(raw["lengths"][e]); cond.append(np.c_[raw["states"][e,:L],np.repeat(raw["goals"][e][None],L,0)]); act.append(raw["actions"][e,:L])
    c=np.concatenate(cond); a=np.concatenate(act); return {"cmean":c.mean(0),"cstd":c.std(0)+1e-6,"amean":a.mean(0),"astd":a.std(0)+1e-6}
class Windows:
    def __init__(self,raw,split,H,s):
        self.r=raw; self.H=H; self.s=s; self.keys=[]
        for e in split_eps(len(raw["lengths"]),split): self.keys += [(e,k) for k in range(int(raw["lengths"][e]))]
    def __len__(self):return len(self.keys)
    def __getitem__(self,i):
        e,k=self.keys[i]; L=int(self.r["lengths"][e]); n=min(self.H,L-k); x=np.zeros((self.H,2),np.float32); m=np.zeros(self.H,np.float32)
        x[:n]=(self.r["actions"][e,k:k+n]-self.s["amean"])/self.s["astd"]; m[:n]=1
        c=np.r_[self.r["states"][e,k],self.r["goals"][e]]; c=(c-self.s["cmean"])/self.s["cstd"]
        return torch.from_numpy(c.astype(np.float32)),torch.from_numpy(x),torch.from_numpy(m),f"{e}:{k}"
class Stream:
    def __init__(self,data,seed,limit=0): self.d=data; self.n=min(limit,len(data)) if limit else len(data); self.g=torch.Generator().manual_seed(seed); self.order=torch.randperm(self.n,generator=self.g); self.pos=0
    def next(self,n):
        if n>self.n: raise ValueError("batch larger than stream")
        if self.pos+n>self.n:self.order=torch.randperm(self.n,generator=self.g);self.pos=0
        idx=self.order[self.pos:self.pos+n].tolist();self.pos+=n; rows=[self.d[i] for i in idx]
        return torch.stack([x[0] for x in rows]),torch.stack([x[1] for x in rows]),torch.stack([x[2] for x in rows]),[x[3] for x in rows]
    def state(self):return {"g":self.g.get_state(),"order":self.order,"pos":self.pos}
    def load(self,s):self.g.set_state(s["g"]);self.order=s["order"];self.pos=s["pos"]
def temb(t,k=32):
    f=(2**torch.arange(k,device=t.device,dtype=t.dtype))*math.pi
    return torch.cat([torch.sin(t*f),torch.cos(t*f)],-1)
class Res(nn.Module):
    def __init__(self,d):super().__init__();self.n=nn.LayerNorm(d);self.a=nn.Linear(d,d);self.b=nn.Linear(d,d)
    def forward(self,x):return x+self.b(torch.nn.functional.silu(self.a(self.n(x))))
class Policy(nn.Module):
    def __init__(self,kind,H,hidden,depth):
        super().__init__();self.kind=kind;self.H=H; inp=6 if kind=="bc" else 6+2*H+H+64
        self.inp=nn.Linear(inp,hidden);self.blocks=nn.ModuleList([Res(hidden) for _ in range(depth)]);self.out=nn.Linear(hidden,2*H)
    def forward(self,c,x=None,t=None,mask=None):
        if self.kind=="bc":z=c
        else:
            if mask is None:mask=torch.ones(x.shape[:2],device=x.device,dtype=x.dtype)
            x=x*mask[:,:,None];z=torch.cat([c,x.flatten(1),mask.to(x.dtype),temb(t)],-1)
        z=self.inp(z)
        for b in self.blocks:z=b(z)
        return self.out(z).view(-1,self.H,2)
def masked_mse(pred,target,mask):return ((pred-target).square()*mask[:,:,None]).sum()/(mask.sum()*pred.size(-1)).clamp_min(1)
@torch.no_grad()
def euler(model,c,z,steps,mask=None):
    x=z if mask is None else z*mask[:,:,None]
    for i in range(steps):
        t=torch.full((x.size(0),1),i/steps,device=x.device,dtype=x.dtype)
        velocity=model(c,x,t) if mask is None else model(c,x,t,mask)
        x=x+velocity/steps
        if mask is not None:x=x*mask[:,:,None]
    return x
class EMA:
    def __init__(self,m,decay):self.decay=decay;self.updates=0;self.shadow={k:v.detach().float().clone() for k,v in m.state_dict().items()}
    def update(self,m):
        self.updates+=1
        for k,v in m.state_dict().items():self.shadow[k].lerp_(v.detach().float(),1-self.decay)
    def state(self):return {"decay":self.decay,"updates":self.updates,"shadow":self.shadow}
    def load(self,s,m):
        current=m.state_dict();self.decay=s["decay"];self.updates=s["updates"]
        self.shadow={k:v.to(device=current[k].device,dtype=torch.float32) for k,v in s["shadow"].items()}
def seed_all(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def sched(step,total,warm):
    if step<warm:return (step+1)/max(1,warm)
    r=min(1,(step-warm)/max(1,total-warm));return .1+.45*(1+math.cos(math.pi*r))
def rng():return {"py":random.getstate(),"np":np.random.get_state(),"cpu":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
def set_rng(s):
    random.setstate(s["py"]);np.random.set_state(s["np"]);torch.set_rng_state(s["cpu"])
    if torch.cuda.is_available() and s["cuda"]:torch.cuda.set_rng_state_all(s["cuda"])
def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,m,opt,scaler,ema,state,stream,noise,c,s,meta):
    o={"model":m.state_dict(),"optimizer":opt.state_dict(),"scaler":scaler.state_dict(),"ema":ema.state(),"state":state,"stream":stream.state(),"noise":noise.get_state(),"rng":rng(),"config":c,"stats":{k:v.tolist() for k,v in s.items()},"meta":meta}
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(".tmp");torch.save(o,tmp);os.replace(tmp,p)
def read_ck(path):
    try:return torch.load(path,map_location="cpu",weights_only=False)
    except TypeError:return torch.load(path,map_location="cpu")
def train(a):
    c=json.loads(Path(a.config).read_text());out=Path(a.output)
    if out.exists() and any(out.iterdir()) and not a.resume:raise RuntimeError("non-empty output")
    if a.fault_attempt is not None and (a.fault_attempt<0 or c["precision"]!="fp16"):raise ValueError("fault injection requires a non-negative attempt in an isolated FP16 run")
    meta={"data_sha256":file_sha(c["data"]),"config_sha256":file_sha(a.config),"code_sha256":file_sha(__file__),"torch":torch.__version__,"fault_attempt":a.fault_attempt}
    seed_all(c["seed"]);raw=load_npz(c["data"]);s=stats(raw);d=Windows(raw,"train",c["horizon"],s);stream=Stream(d,c["seed"]+1,c["overfit"])
    dev=torch.device(c["device"]);m=Policy(c["kind"],c["horizon"],c["hidden"],c["depth"]).to(dev);opt=torch.optim.AdamW(m.parameters(),lr=c["lr"])
    scaler=torch.cuda.amp.GradScaler(enabled=c["precision"]=="fp16");ema=EMA(m,c["ema_decay"]);noise=torch.Generator().manual_seed(c["seed"]+2)
    state={"step":0,"attempted":0,"attempted_action_steps":0,"effective_windows":0,"effective_action_steps":0,"attempted_time_s":0.0,"consecutive_skips":0};
    if a.resume:
        ck=read_ck(a.resume)
        if ck["config"]!=c or ck.get("meta")!=meta:raise RuntimeError("resume config/data/code/environment mismatch")
        m.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"]);scaler.load_state_dict(ck["scaler"]);ema.load(ck["ema"],m);state=ck["state"];stream.load(ck["stream"]);noise.set_state(ck["noise"]);set_rng(ck["rng"])
    out.mkdir(parents=True,exist_ok=True);(out/"run_manifest.json").write_text(json.dumps({"config":c,**meta},indent=2)+"\n")
    log=out/"metrics.jsonl";print(json.dumps({"parameters":sum(p.numel() for p in m.parameters())}))
    if dev.type=="cuda":torch.cuda.reset_peak_memory_stats(dev)
    target_steps=min(c["max_steps"],a.stop_after if a.stop_after is not None else c["max_steps"])
    while state["step"]<target_steps:
        lr=c["lr"]*sched(state["step"],c["schedule_steps"],c["warmup"])
        for g in opt.param_groups:g["lr"]=lr
        if dev.type=="cuda":torch.cuda.synchronize()
        t0=time.perf_counter()
        inject_fault=a.fault_attempt is not None and state["attempted"]==a.fault_attempt
        fault_snapshot=[p.detach().clone() for p in m.parameters()] if inject_fault else None
        fault_ema_snapshot={k:v.clone() for k,v in ema.shadow.items()} if inject_fault else None
        opt.zero_grad(set_to_none=True);ids=[];valid=0;noise_hash=hashlib.sha256();micros=[]
        for _ in range(c["accum"]):
            cond,x1,mask,bids=stream.next(c["batch"]);ids+=bids;valid+=int(mask.sum());micros.append((cond,x1,mask))
        total_num=None
        for cond,x1,mask in micros:
            mask_cpu=mask;cond,x1,mask=cond.to(dev),x1.to(dev),mask.to(dev)
            if c["kind"]=="flow":
                valid_mask=mask[:,:,None];x1=x1*valid_mask
                if c["fixed_noise"]:
                    x0_cpu=torch.zeros(tuple(x1.shape),dtype=torch.float32);t_cpu=torch.full((x1.size(0),1),.5)
                else:
                    x0_cpu=torch.randn(tuple(x1.shape),generator=noise)*mask_cpu[:,:,None];t_cpu=torch.rand(x1.size(0),1,generator=noise)
                noise_hash.update(x0_cpu.numpy().tobytes());noise_hash.update(t_cpu.numpy().tobytes());x0=x0_cpu.to(dev);t=t_cpu.to(dev);xt=((1-t[:,:,None])*x0+t[:,:,None]*x1)*valid_mask;y=(x1-x0).detach()
            else:xt=t=None;y=x1
            with torch.cuda.amp.autocast(enabled=c["precision"]=="fp16",dtype=torch.float16):pred=m(cond,xt,t,mask)
            num=((pred.float()-y.float()).square()*mask[:,:,None]).sum()
            if not torch.isfinite(num):raise FloatingPointError("nonfinite loss numerator")
            loss_part=num/max(1,valid*pred.size(-1))
            if inject_fault:loss_part=loss_part*torch.tensor(float("inf"),device=loss_part.device)
            scaler.scale(loss_part).backward();total_num=num.detach() if total_num is None else total_num+num.detach()
        scaler.unscale_(opt);grad=float(torch.nn.utils.clip_grad_norm_(m.parameters(),c["clip"]));
        if not scaler.is_enabled() and not math.isfinite(grad):raise FloatingPointError("FP32 nonfinite grad")
        before=float(scaler.get_scale());scaler.step(opt);scaler.update();success=math.isfinite(grad) and float(scaler.get_scale())>=before;state["attempted"]+=1
        fault_params_unchanged=None if not inject_fault else all(bool(torch.equal(p.detach(),q)) for p,q in zip(m.parameters(),fault_snapshot))
        fault_ema_unchanged=None if not inject_fault else all(bool(torch.equal(ema.shadow[k],v)) for k,v in fault_ema_snapshot.items())
        if inject_fault and (success or not fault_params_unchanged or not fault_ema_unchanged):raise RuntimeError("fault injection was not skipped atomically")
        if success:
            state["step"]+=1;state["effective_windows"]+=c["batch"]*c["accum"];state["effective_action_steps"]+=valid;state["consecutive_skips"]=0;ema.update(m)
        else:
            state["consecutive_skips"]+=1
            if state["consecutive_skips"]>8:raise RuntimeError("more than 8 consecutive AMP overflows")
        if dev.type=="cuda":torch.cuda.synchronize()
        step_ms=1000*(time.perf_counter()-t0);state["attempted_action_steps"]+=valid;state["attempted_time_s"]+=step_ms/1000
        peak_alloc=torch.cuda.max_memory_allocated() if dev.type=="cuda" else None;peak_reserved=torch.cuda.max_memory_reserved() if dev.type=="cuda" else None
        loss_value=float(total_num)/max(1,valid*2)
        rec={**state,"loss":loss_value,"lr":lr,"grad_norm":grad if math.isfinite(grad) else None,"loss_scale":float(scaler.get_scale()),"skipped":not success,"fault_injected":inject_fault,"fault_params_unchanged":fault_params_unchanged,"fault_ema_unchanged":fault_ema_unchanged,"valid_steps":valid,"window_ids":ids,"noise_hash":noise_hash.hexdigest(),"ema_updates":ema.updates,
             "step_ms":step_ms,"attempted_action_steps_per_s":valid/(step_ms/1000),"effective_action_steps_per_s":valid/(step_ms/1000) if success else 0.0,
             "cumulative_attempted_action_steps_per_s":state["attempted_action_steps"]/state["attempted_time_s"],"cumulative_effective_action_steps_per_s":state["effective_action_steps"]/state["attempted_time_s"],"peak_allocated":peak_alloc,"peak_reserved":peak_reserved}
        with log.open("a") as f:f.write(json.dumps(rec)+"\n");print(json.dumps(rec))
        if success and state["step"]%c["save_every"]==0:save(out/"last.pt",m,opt,scaler,ema,state,stream,noise,c,s,meta)
    save(out/"last.pt",m,opt,scaler,ema,state,stream,noise,c,s,meta)
def wilson(k,n):
    if n==0:return [None,None]
    z=1.96;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z/d*math.sqrt(p*(1-p)/n+z*z/(4*n*n));return [c-h,c+h]
@torch.no_grad()
def evaluate(a):
    ck=read_ck(a.checkpoint);c=ck["config"];s={k:np.asarray(v,np.float32) for k,v in ck["stats"].items()};eval_data=a.data or c["data"];raw=load_npz(eval_data);d=Windows(raw,a.split,c["horizon"],s)
    dev=torch.device(c["device"]);m=Policy(c["kind"],c["horizon"],c["hidden"],c["depth"]).to(dev)
    m.load_state_dict(ck["ema"]["shadow"] if a.weights=="ema" else ck["model"]);m.eval();g=torch.Generator().manual_seed(a.seed);errs=[];lat=[]
    n=min(a.samples,len(d))
    if n==0:raise RuntimeError("empty eval split")
    for i in range(n):
        cond,x1,mask,sid=d[i];cond=cond[None].to(dev);x1=x1[None].to(dev);mask=mask[None].to(dev)
        if a.condition=="zero":cond.zero_()
        elif a.condition=="shuffle":
            j=(i+max(1,n//2))%n
            for _ in range(n):
                if d[j][3].split(":")[0]!=sid.split(":")[0]:break
                j=(j+1)%n
            if d[j][3].split(":")[0]==sid.split(":")[0]:raise RuntimeError("shuffle requires at least two episodes")
            cond=d[j][0][None].to(dev)
        torch.cuda.synchronize() if dev.type=="cuda" else None;t0=time.perf_counter()
        pred=m(cond) if c["kind"]=="bc" else euler(m,cond,torch.randn(x1.shape,generator=g).to(dev),a.ode_steps,mask)
        torch.cuda.synchronize() if dev.type=="cuda" else None;lat.append(1000*(time.perf_counter()-t0))
        physical=(pred.cpu().numpy()*s["astd"]+s["amean"]);gold=(x1.cpu().numpy()*s["astd"]+s["amean"]);valid=mask.cpu().numpy()[:,:,None]
        errs.append(float((((physical-gold)**2)*valid).sum()/(valid.sum()*2)))
    result={"masked_action_mse":float(np.mean(errs)),"samples":n,"condition":a.condition,"weights":a.weights,"ode_steps":a.ode_steps,"seed":a.seed,
            "checkpoint_sha256":file_sha(a.checkpoint),"eval_data_sha256":file_sha(eval_data),"config_sha256":ck["meta"]["config_sha256"],"code_sha256":ck["meta"]["code_sha256"],
            "latency_ms_p50":statistics.median(lat),"latency_ms_p95":sorted(lat)[max(0,int(.95*len(lat))-1)]}
    if a.rollouts:
        wins=0; eps=split_eps(len(raw["lengths"]),a.split)[:a.rollouts]
        for e in eps:
            state=raw["states"][e,0].copy();goal=raw["goals"][e].copy()
            for _ in range(64):
                cn=(np.r_[state,goal]-s["cmean"])/s["cstd"];ct=torch.from_numpy(cn.astype(np.float32))[None].to(dev)
                if c["kind"]=="bc":chunk=m(ct)
                else:chunk=euler(m,ct,torch.randn((1,c["horizon"],2),generator=g).to(dev),a.ode_steps)
                action=np.clip(chunk[0,0].cpu().numpy()*s["astd"]+s["amean"],-1,1)
                vel=.8*state[2:]+.2*action;pos=state[:2]+.1*vel;state=np.r_[pos,vel].astype(np.float32)
                if np.linalg.norm(pos-goal)<.15:wins+=1;break
        result.update(rollout_success=wins/max(1,len(eps)),rollout_n=len(eps),rollout_wilson95=wilson(wins,len(eps)))
    op=Path(a.output);op.parent.mkdir(parents=True,exist_ok=True);op.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))
def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="cmd",required=True)
    t=sub.add_parser("train");t.add_argument("--config",required=True);t.add_argument("--output",required=True);t.add_argument("--resume");t.add_argument("--stop-after",type=int);t.add_argument("--fault-attempt",type=int)
    e=sub.add_parser("eval");e.add_argument("--checkpoint",required=True);e.add_argument("--data");e.add_argument("--split",choices=["dev","test"],default="dev");e.add_argument("--weights",choices=["raw","ema"],default="ema")
    e.add_argument("--condition",choices=["correct","shuffle","zero"],default="correct");e.add_argument("--ode-steps",type=int,default=20);e.add_argument("--samples",type=int,default=200);e.add_argument("--rollouts",type=int,default=0);e.add_argument("--seed",type=int,default=900);e.add_argument("--output",required=True)
    a=p.parse_args();train(a) if a.cmd=="train" else evaluate(a)
if __name__=="__main__":main()
```

创建单测：

```python
# tests/test_core.py
import numpy as np, torch
from src.tinyflow import Windows,Policy,masked_mse,euler
def raw():
    s=np.zeros((10,3,4),np.float32);a=np.zeros((10,3,2),np.float32);a[0,:,0]=[10,11,12]
    return {"states":s,"actions":a,"goals":np.zeros((10,2),np.float32),"lengths":np.array([3]*10)}
def stat():return {"cmean":np.zeros(6,np.float32),"cstd":np.ones(6,np.float32),"amean":np.zeros(2,np.float32),"astd":np.ones(2,np.float32)}
def test_last_window_padding():
    d=Windows(raw(),"train",4,stat());_,x,m,_=d[2];assert x[:,0].tolist()==[12,0,0,0] and m.tolist()==[1,0,0,0]
def test_target_and_mask():
    x0=torch.tensor([[[1.,2.],[3.,4.]]]);x1=torch.tensor([[[5.,6.],[99.,99.]]]);t=torch.tensor([[[.25]]]);xt=(1-t)*x0+t*x1
    torch.testing.assert_close(xt[0,0],torch.tensor([2.,3.]));target=x1-x0;pred=target.clone();pred[:,1]=1000
    assert masked_mse(pred,target,torch.tensor([[1.,0.]])).item()==0
def test_padding_latent_cannot_change_valid_prediction():
    torch.manual_seed(3);m=Policy("flow",4,32,2).eval();c=torch.randn(1,6);t=torch.tensor([[.4]]);mask=torch.tensor([[1.,1.,0.,0.]])
    a=torch.randn(1,4,2);b=a.clone();b[:,2:]=9999
    torch.testing.assert_close(m(c,a,t,mask)[:,:2],m(c,b,t,mask)[:,:2])
def test_euler_constant():
    class C:
        def __call__(self,c,x,t):return torch.ones_like(x)*2
    z=torch.zeros(2,4,2);torch.testing.assert_close(euler(C(),torch.zeros(2,6),z,10),torch.ones_like(z)*2)
def test_single_batch_forward_backward():
    m=Policy("flow",4,32,2);c=torch.randn(3,6);x=torch.randn(3,4,2);t=torch.rand(3,1);mask=torch.tensor([[1,1,1,1],[1,1,0,0],[1,0,0,0.]])
    loss=masked_mse(m(c,x,t,mask),torch.randn_like(x),mask);loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
```

### 实现

代码实现从 episode 到按全局有效 action 数归一的累积 loss、BC/Flow、EMA、原子 checkpoint 和固定 `x0+t` hash。Flow 在进入 dense MLP 前把 `x0/x1/xt` 的 padding 位归零，并把 mask 本身输入模型；因此改变 padding latent 不得影响有效位置。AMP 首次 gradient overflow 交给 GradScaler skip/backoff，只有连续超过 8 次才止损；forward/loss numerator 本身非有限仍立即失败。EMA、step 与 effective counters 仅随成功更新推进。每次 attempt 同步记录 step time、瞬时及累计 attempted/effective action-step/s和显存峰值。`schedule_steps` 与 resume 分段无关；checkpoint 绑定 data/config/code hash 与 torch 版本；非空输出无 resume 拒绝覆盖。

### 执行命令

生成ID/OOD：

`[两者/Python]`：

```bash
python -m py_compile src/make_data.py src/toy2d.py src/tinyflow.py
python -m pytest -q tests/test_core.py
python src/make_data.py --episodes 1000 --max-len 64 --seed 100 --goal-scale 1.0 --output data/id.npz --manifest manifests/id.json
python src/make_data.py --episodes 1000 --max-len 64 --seed 10000 --goal-scale 1.5 --output data/ood.npz --manifest manifests/ood.json
```

创建完整BC配置：

```json
{
  "data":"data/id.npz","kind":"bc","horizon":16,"hidden":1024,"depth":5,
  "seed":41,"device":"cuda","precision":"fp32","batch":16,"accum":1,
  "lr":0.0003,"clip":1.0,"ema_decay":0.999,"warmup":20,
  "max_steps":500,"schedule_steps":500,"save_every":50,"overfit":128,"fixed_noise":false
}
```

将上述纯JSON保存为 `configs/bc_overfit.json`。

### 预期观测（估算/示例，不是实测）

id/ood 各有1000个episode且manifest hash不同；因此各自的10% test split恰有100个episode。pytest 全部通过，包括 padding-latent invariance；BC 参数约 10M 但以打印值为准。

### 验收条件

微型尾窗正确、padding 不影响 loss 且 padding latent 不影响有效预测、Euler 常量场通过；数据 schema/hash 固定。

### 若失败，按什么顺序查

episode length→split ID→window key→padding/mask→train-only stats→action维度→solver方向。

### 当日证据清单

两个manifest、pytest输出、配置hash、随机抽5个window的ID/shape/mask。

## Day 3：BC与Flow两级 overfit gate

### 为什么做

BC先验证数据/head；Flow先固定x0/t，再恢复随机x0/t，逐层增加难度。

### 输入与前置检查

Day2单测全过，`configs/bc_overfit.json`已创建。

### 本日要创建/修改的文件

从BC配置生成两份完整Flow配置（命令会先创建路径）：

`[两者/Python]`：

```bash
python -c "import json; c=json.load(open('configs/bc_overfit.json')); c.update(kind='flow',fixed_noise=True); open('configs/flow_fixed.json','w').write(json.dumps(c,indent=2)+'\n'); c.update(fixed_noise=False); open('configs/flow_random.json','w').write(json.dumps(c,indent=2)+'\n'); s=dict(c); s.update(max_steps=50,schedule_steps=50,save_every=25); open('configs/flow_smoke.json','w').write(json.dumps(s,indent=2)+'\n')"
```

### 实现

fixed配置令`x0=0,t=.5`，只用于debugging；random配置才对应正式flow objective。三者都用相同128 windows、500 successful updates与模型容量量级。

### 执行命令

`[公司 Linux]`：

```bash
CUDA_VISIBLE_DEVICES=0 python src/tinyflow.py train --config configs/flow_smoke.json --output runs/flow_smoke
CUDA_VISIBLE_DEVICES=0 python src/tinyflow.py train --config configs/bc_overfit.json --output runs/bc_overfit
CUDA_VISIBLE_DEVICES=0 python src/tinyflow.py train --config configs/flow_fixed.json --output runs/flow_fixed
CUDA_VISIBLE_DEVICES=0 python src/tinyflow.py train --config configs/flow_random.json --output runs/flow_random
```

`[个人 PowerShell]`：

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
python src/tinyflow.py train --config configs/flow_smoke.json --output runs/flow_smoke
python src/tinyflow.py train --config configs/bc_overfit.json --output runs/bc_overfit
python src/tinyflow.py train --config configs/flow_fixed.json --output runs/flow_fixed
python src/tinyflow.py train --config configs/flow_random.json --output runs/flow_random
```

### 预期观测（估算/示例，不是实测）

三组 loss 总体下降；通常 BC/fixed 更容易，但不设虚构终值。EMA updates 应等于 successful step；日志必须同时有 attempted/successful、step time、有效 action-step/s、allocated/reserved memory。

### 验收条件

BC 显著 overfit；fixed flow 明显下降；random flow 继续下降；正常 run gradient finite；padding 有效数和性能字段均记录。单次 FP16 overflow 可被 skip；连续 8 次以上 backoff 则停止。任一失败不进入长训。

### 若失败，按什么顺序查

同一window IDs→BC→fixed target手算→time embedding→random noise hash→mask denominator→LR/grad。不要先扩大网络。

### 当日证据清单

三组metrics、初末loss、window/noise hash、EMA update count、参数量。

## Day 4：FP16、resume 与 raw/EMA eval

### 为什么做

验证V100数值稳定、成功update时钟和完整恢复。

### 输入与前置检查

random flow overfit通过；V100只用FP16，不用BF16/FA2。

### 本日要创建/修改的文件

创建FP16与resume配置：

`[两者/Python]`：

```bash
python -c "import json; c=json.load(open('configs/flow_random.json')); c.update(precision='fp16',max_steps=200,schedule_steps=200,save_every=50); open('configs/flow_fp16_200.json','w').write(json.dumps(c,indent=2)+'\n'); q=dict(c); q.update(max_steps=40,schedule_steps=40,save_every=20); open('configs/flow_fp16_fault40.json','w').write(json.dumps(q,indent=2)+'\n'); c.update(precision='fp32',max_steps=40,schedule_steps=40,save_every=20); open('configs/resume40.json','w').write(json.dumps(c,indent=2)+'\n')"
```

### 实现

resume checkpoint含window stream、noise generator、EMA、scaler与RNG；连续/分段必须使用同一 `configs/resume40.json`。独立 `flow_fp16_fault40.json` 只用于在attempt 25注入一次非有限梯度：注入前复制参数，随后必须观察到GradScaler skip、参数逐元素不变、EMA/成功step不前进，且下一attempt恢复；它不能作为正常训练checkpoint。

### 执行命令

`[公司 Linux]`：

```bash
CUDA_VISIBLE_DEVICES=0 python src/tinyflow.py train --config configs/flow_fp16_200.json --output runs/fp16_200
CUDA_VISIBLE_DEVICES=0 python src/tinyflow.py train --config configs/flow_fp16_fault40.json --fault-attempt 25 --output runs/fp16_fault40
python -c "import json; R=[json.loads(x) for x in open('runs/fp16_fault40/metrics.jsonl')]; H=[(i,r) for i,r in enumerate(R) if r['fault_injected']]; assert len(H)==1; i,h=H[0]; assert i>0 and h['skipped'] and h['fault_params_unchanged'] and h['fault_ema_unchanged'] and h['ema_updates']==h['step'] and h['loss_scale']<R[i-1]['loss_scale']; assert h['effective_windows']==R[i-1]['effective_windows'] and h['effective_action_steps']==R[i-1]['effective_action_steps']; assert i+1<len(R) and not R[i+1]['skipped'] and R[i+1]['step']==h['step']+1; print('AMP_SKIP_RECOVERY_PASS',h['attempted'],h['step'])"
CUDA_VISIBLE_DEVICES=0 python src/tinyflow.py train --config configs/resume40.json --output runs/continuous40
CUDA_VISIBLE_DEVICES=0 python src/tinyflow.py train --config configs/resume40.json --stop-after 20 --output runs/resumed40
CUDA_VISIBLE_DEVICES=0 python src/tinyflow.py train --config configs/resume40.json --output runs/resumed40 --resume runs/resumed40/last.pt
python -c "import json,math; A=[json.loads(x) for x in open('runs/continuous40/metrics.jsonl')]; B=[json.loads(x) for x in open('runs/resumed40/metrics.jsonl')]; assert len(A)==len(B)==40; assert all((x['step'],x['window_ids'],x['noise_hash'],x['lr'],x['loss_scale'])==(y['step'],y['window_ids'],y['noise_hash'],y['lr'],y['loss_scale']) and math.isclose(x['loss'],y['loss'],rel_tol=1e-6,abs_tol=1e-8) for x,y in zip(A,B)); print('RESUME_PASS',len(A))"
```

`--stop-after 20` 只控制本次正常退出点，不改变完整配置中的 `max_steps/schedule_steps=40`；它在step20原子保存。不得kill正在写盘的进程。

raw/EMA：

```bash
CUDA_VISIBLE_DEVICES=0 python src/tinyflow.py eval --checkpoint runs/fp16_200/last.pt --weights raw --condition correct --ode-steps 20 --samples 100 --output runs/fp16_200/raw.json
CUDA_VISIBLE_DEVICES=0 python src/tinyflow.py eval --checkpoint runs/fp16_200/last.pt --weights ema --condition correct --ode-steps 20 --samples 100 --output runs/fp16_200/ema.json
```

### 预期观测（估算/示例，不是实测）

FP16 finite，warmup后skip低；EMA update count=successful steps。raw/EMA均能独立评估，不保证EMA一定更好。

### 验收条件

FP16 200step无非有限、warmup后skip `<1%`；受控fault恰有一次skip，参数与EMA不变，下一attempt恢复；EMA只随成功step；resume的next window/noise hash、LR/scale/loss对齐。

### 若失败，按什么顺序查

FP32同batch→masked loss FP32→scaler/clip→EMA更新条件→checkpoint完整性→window/noise hash→总schedule。OOM先降batch、增accum。

### 当日证据清单

FP16 metrics、raw/EMA eval、safe-stop证据、continuous/resume首个续跑hash。

## Day 5：同预算正式对照、反事实与 ID/OOD

### 为什么做

在相同训练预算下比较BC/Flow，并用condition与OOD揭露“offline好看”的伪结论。

### 输入与前置检查

Day4所有系统门通过，包括 `--stop-after` 的受控resume。

### 本日要创建/修改的文件

创建两份正式配置：

`[两者/Python]`：

```bash
python -c "import json; b=json.load(open('configs/bc_overfit.json')); b.update(overfit=0,precision='fp16',max_steps=5000,schedule_steps=5000,save_every=250); open('configs/bc_formal.json','w').write(json.dumps(b,indent=2)+'\n'); f=dict(b); f.update(kind='flow',fixed_noise=False); open('configs/flow_formal.json','w').write(json.dumps(f,indent=2)+'\n')"
```

### 实现

正式BC/Flow同episode、window、successful updates、global batch、capacity量级。Flow主结果固定20 Euler steps，并额外5/10/50作延迟-误差曲线。`eval --rollouts N` 使用与 `make_data.py` 相同的动力学，每次执行预测chunk的第一个动作；offline与closed-loop分开报告。

### 执行命令

`[公司 Linux]`：

```bash
CUDA_VISIBLE_DEVICES=0 python src/tinyflow.py train --config configs/bc_formal.json --output runs/bc_formal
CUDA_VISIBLE_DEVICES=0 python src/tinyflow.py train --config configs/flow_formal.json --output runs/flow_formal
CUDA_VISIBLE_DEVICES=0 python src/tinyflow.py eval --checkpoint runs/bc_formal/last.pt --weights ema --condition correct --samples 200 --rollouts 100 --output runs/bc_formal/id.json
CUDA_VISIBLE_DEVICES=0 python src/tinyflow.py eval --checkpoint runs/flow_formal/last.pt --weights ema --condition correct --ode-steps 20 --samples 200 --rollouts 100 --output runs/flow_formal/id.json
CUDA_VISIBLE_DEVICES=0 python src/tinyflow.py eval --checkpoint runs/flow_formal/last.pt --weights ema --condition correct --ode-steps 5 --samples 200 --output runs/flow_formal/ode5.json
CUDA_VISIBLE_DEVICES=0 python src/tinyflow.py eval --checkpoint runs/flow_formal/last.pt --weights ema --condition correct --ode-steps 10 --samples 200 --output runs/flow_formal/ode10.json
CUDA_VISIBLE_DEVICES=0 python src/tinyflow.py eval --checkpoint runs/flow_formal/last.pt --weights ema --condition correct --ode-steps 50 --samples 200 --output runs/flow_formal/ode50.json
CUDA_VISIBLE_DEVICES=0 python src/tinyflow.py eval --checkpoint runs/flow_formal/last.pt --weights ema --condition shuffle --ode-steps 20 --samples 200 --output runs/flow_formal/shuffle.json
CUDA_VISIBLE_DEVICES=0 python src/tinyflow.py eval --checkpoint runs/flow_formal/last.pt --weights ema --condition zero --ode-steps 20 --samples 200 --output runs/flow_formal/zero.json
CUDA_VISIBLE_DEVICES=0 python src/tinyflow.py eval --checkpoint runs/flow_formal/last.pt --data data/ood.npz --split test --weights ema --condition correct --ode-steps 20 --samples 100 --rollouts 100 --output runs/flow_formal/ood.json
```

`[个人 PowerShell]` 将正式步数在新配置缩至1000，使用公开本文代码独立运行，不复制公司结果。

### 预期观测（估算/示例，不是实测）

每份eval有masked MSE、latency p50/p95、condition、weights、ODE steps。correct应优于shuffle/zero才初步证明条件依赖；OOD通常更难只是待验假设。

### 验收条件

BC/Flow预算可比；ID/condition/OOD使用固定IDs/noise；ODE 5/10/20/50固定初始noise；rollout报告success与Wilson区间；任何绝对“Flow更好”只由实测支持。

### 若失败，按什么顺序查

checkpoint config/stats→same IDs/noise→反归一化→solver0→1→condition置换是否真跨样本→OOD只改goal范围。

### 当日证据清单

```text
runs/{bc_formal,flow_formal}/
  last.pt
  metrics.jsonl
  id.json / ode5.json / ode10.json / ode50.json / shuffle.json / zero.json / ood.json
  run_manifest.json
```

跨运行结论写入 Day 0 已创建的 `reports/decision.md`，不虚构每个 run 目录中的文件。

## 最终发布门

数学toy、window/mask、train-only stats、BC gate、fixed/random flow gate、FP16、EMA、受控safe-stop resume、同预算offline/rollout eval和condition对照必须逐项标PASS/FAIL/MISSING。任一命令未执行都不得写成实测能力。

技术链接核验日期：**2026-09-03**：[Flow Matching](https://arxiv.org/abs/2210.02747)、[LeRobot PushT](https://huggingface.co/datasets/lerobot/pusht)、[Diffusion Policy](https://github.com/real-stanford/diffusion_policy)、[PyTorch 2.1 AMP](https://pytorch.org/docs/2.1/notes/amp_examples.html)。
