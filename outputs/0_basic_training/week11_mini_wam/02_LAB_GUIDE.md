# Week 11 实践篇：从空目录运行 Mini-WAM A/B/C

> 全部命令均为“待执行”；示例输出不是实测。公司环境中的数据、日志、模型、指标、trace 与拓扑信息不得导出。  
> 本手册的可执行核心使用 ToyPush-v1；它只证明代码与因果对照闭环。未在固定的公开/获批机器人数据上复测时，模型结论必须为 `INCONCLUSIVE`。

## 0. 下载前：机器、预算与只读预检

公司 8×V100 是主训练环境（用户自述待核验）；个人 5070 Ti 只重跑公开缩小版；H100 只是可选对照。估算最低空间：ToyPush 0.2GB、三组 checkpoint/log 2GB；可选 PushT snapshot 预留 10GB；DINOv2 预留 0.5GB。Toy 模型单卡显存估算 <2GB，20-step <5 分钟；真实载荷必须现场重估。

复用当前批准的 Conda 环境，不执行 `conda create`、`python -m venv`、torch/CUDA 升级：

```bash
which python
python --version
python - <<'PY'
import shutil, torch
print({"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
       "cuda_available": torch.cuda.is_available(), "gpu_count": torch.cuda.device_count(),
       "git": shutil.which("git"), "disk": shutil.disk_usage(".")})
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p=torch.cuda.get_device_properties(i)
        print(i, p.name, p.major, p.minor, p.total_memory)
PY
nvidia-smi
nvidia-smi topo -m
git --version
df -h .
```

公司预期 PyTorch 2.1；版本不符只记录并交环境负责人。保存依赖基线，先 dry-run，不直接安装：

```bash
mkdir -p week11-miniwam/manifests
python -m pip freeze > week11-miniwam/manifests/pip-freeze.before.txt
python -m pip check | tee week11-miniwam/manifests/pip-check.before.txt
python - <<'PY'
import importlib
for x in ("torch","numpy"):
    m=importlib.import_module(x); print(x, getattr(m,"__version__","unknown"))
PY
```

核心路径只需当前 `torch`、`numpy`。若缺失，停止；经批准后只从内网 wheelhouse 安装。回滚方式是停止作业并由环境负责人按 `pip-freeze.before.txt` 恢复，不能为本周自行重装 torch。

## 1. 数据/模型获取与完整性

| 对象 | 官方 ID | revision | 所需文件 | 许可检查 | 完整性检查 |
|---|---|---|---|---|---|
| ToyPush-v1 | 本文代码 | 以 `make_toy_data.py` SHA 固定 | `toy_push_v1.npz/meta.json` | 项目自生成、非敏感 | SHA、shape、split、有限值 |
| PushT（可选 transfer） | `lerobot/pusht` | `7628202a2180972f291ba1bc6723834921e72c19` | snapshot 中 `meta/ data/ videos/ README.md` | Hub 卡 MIT；保存卡片 | snapshot SHA 清单、episode/schema |
| DINOv2-small（可选） | `facebook/dinov2-small` | `ed25f3a31f01632728cabb09d1542f84ab7b0056` | `config.json`、`preprocessor_config.json`、`model.safetensors` | Apache-2.0 | revision、文件 SHA、离线 load |

核心命令不依赖 Hub。若公司允许联网且已安装/批准 `huggingface_hub`，才执行可选下载：

```bash
cd week11-miniwam
mkdir -p third_party/pusht third_party/dinov2-small manifests
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="lerobot/pusht", repo_type="dataset",
 revision="7628202a2180972f291ba1bc6723834921e72c19",
 local_dir="third_party/pusht")
snapshot_download(repo_id="facebook/dinov2-small", repo_type="model",
 revision="ed25f3a31f01632728cabb09d1542f84ab7b0056",
 allow_patterns=["*.json","*.safetensors","README.md","LICENSE*"],
 local_dir="third_party/dinov2-small")
PY
find third_party -type f -print0 | sort -z | xargs -0 sha256sum > manifests/third_party.sha256
test -f third_party/pusht/meta/info.json
test -f third_party/dinov2-small/model.safetensors
```

网络不可达时，只能使用管理员提供且 revision/hash 相同的获批只读 snapshot；否则跳过 transfer，并记录 `ENV-BLOCKED`。不要使用个人 token、代理、网盘或自动上传。

## 2. 工程创建

从批准工作区执行：

```bash
mkdir -p week11-miniwam/{configs,data,manifests,runs,tests,reports}
cd week11-miniwam
```

上面的命令只创建目录，不会生成任何源码或配置。请把下文每个代码块原样保存为其标题指定的相对路径；保存后再执行 Day 1 的 `py_compile`/`json.tool` 预检。

最终目录：

```text
week11-miniwam/
  make_toy_data.py          # 生成确定性 episode 与 split
  run.py                    # validate/train/eval/resume/DDP 唯一核心入口
  configs/*.json            # 三组唯一差异
  tests/test_core.py        # alignment/shuffle/shape/stop-grad
  data/                     # 生成或获批 snapshot
  manifests/                # 环境、SHA、来源；训练读取
  runs/                     # 仅本机输出
  reports/                  # 预注册和结论
```

### 2.1 `make_toy_data.py`：完整代码

将紧随其后的代码块原样保存为 `make_toy_data.py`：

```python
# file: make_toy_data.py
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def render(pos, target, size=32):
    im=np.zeros((3,size,size),np.uint8)
    def dot(p,c):
        x=int(np.clip(p[0],0,1)*(size-1)); y=int(np.clip(p[1],0,1)*(size-1))
        im[c,max(0,y-1):min(size,y+2),max(0,x-1):min(size,x+2)]=255
    dot(target,1); dot(pos,0)
    return im

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True)
    ap.add_argument("--episodes",type=int,default=60); ap.add_argument("--steps",type=int,default=24)
    ap.add_argument("--seed",type=int,default=1101); a=ap.parse_args()
    assert a.episodes>=20 and a.steps>=8
    images=[]; states=[]; actions=[]
    for e in range(a.episodes):
        r=np.random.default_rng(a.seed+e); pos=r.uniform(.15,.85,2); vel=np.zeros(2,np.float32)
        target=r.uniform(.15,.85,2); ims=[]; sts=[]; acts=[]
        for t in range(a.steps):
            act=np.clip(1.5*(target-pos)+r.normal(0,.08,2),-1,1).astype(np.float32)
            ims.append(render(pos,target)); sts.append(np.r_[pos,vel].astype(np.float32)); acts.append(act)
            vel=.65*vel+.12*act; pos=np.clip(pos+vel*.08,0,1)
        images.append(ims); states.append(sts); actions.append(acts)
    split=np.full(a.episodes,2,np.int64); split[:int(.7*a.episodes)]=0
    split[int(.7*a.episodes):int(.85*a.episodes)]=1
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,images=np.asarray(images),states=np.asarray(states),
                        actions=np.asarray(actions),split=split,seed=np.int64(a.seed))
    sha=hashlib.sha256(out.read_bytes()).hexdigest()
    meta={"name":"ToyPush-v1","sha256":sha,"episodes":a.episodes,"steps":a.steps,
          "shapes":{"images":[a.episodes,a.steps,3,32,32],"states":[a.episodes,a.steps,4],
                    "actions":[a.episodes,a.steps,2]},"split_ids":{"train":0,"dev":1,"test":2}}
    out.with_suffix(".meta.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(json.dumps({"status":"PASS","data":str(out),**meta},ensure_ascii=False))
if __name__=="__main__": main()
```

### 2.2 三份完整配置

`configs/A_action_only.json`：

将紧随其后的代码块原样保存为 `configs/A_action_only.json`：

```json
{"run":"A_action_only","seed":1107,"split":"train","future_offset":4,"action_horizon":4,
 "latent_dim":64,"global_batch":64,"lr":0.0003,"weight_decay":0.01,"lambda_world":0.0,
 "shuffle_action":false,"max_steps":300,"precision":"fp16","clip_norm":1.0,"save_every":100}
```

`configs/B_action_world.json`：

将紧随其后的代码块原样保存为 `configs/B_action_world.json`：

```json
{"run":"B_action_world","seed":1107,"split":"train","future_offset":4,"action_horizon":4,
 "latent_dim":64,"global_batch":64,"lr":0.0003,"weight_decay":0.01,"lambda_world":0.2,
 "shuffle_action":false,"max_steps":300,"precision":"fp16","clip_norm":1.0,"save_every":100}
```

`configs/C_action_world_shuffle.json`：

将紧随其后的代码块原样保存为 `configs/C_action_world_shuffle.json`：

```json
{"run":"C_action_world_shuffle","seed":1107,"split":"train","future_offset":4,"action_horizon":4,
 "latent_dim":64,"global_batch":64,"lr":0.0003,"weight_decay":0.01,"lambda_world":0.2,
 "shuffle_action":true,"max_steps":300,"precision":"fp16","clip_norm":1.0,"save_every":100}
```

### 2.3 `run.py`：完整核心训练/评测代码

将紧随其后的代码块原样保存为 `run.py`：

```python
# file: run.py
import argparse, contextlib, hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def object_sha(x):
    raw=json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
def checkpoint_manifest_path(path): return Path(str(path)+".manifest.json")
def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(str(path)+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding="utf-8")
    os.replace(tmp,path)
def validate_checkpoint_file(path,expected_contract=None):
    p=Path(path); sidecar=checkpoint_manifest_path(p)
    if not p.is_file() or p.stat().st_size==0: raise RuntimeError(f"checkpoint missing/empty: {p}")
    if not sidecar.is_file() or sidecar.stat().st_size==0: raise RuntimeError(f"checkpoint sidecar missing/empty: {sidecar}")
    try: m=json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as e: raise RuntimeError(f"invalid checkpoint sidecar: {e}") from e
    if m.get("format_version")!=1 or m.get("checkpoint_file")!=p.name:
        raise RuntimeError("checkpoint sidecar schema/path mismatch")
    if m.get("bytes")!=p.stat().st_size or m.get("sha256")!=sha(p):
        raise RuntimeError("checkpoint bytes/SHA mismatch; refuse load")
    if m.get("torch_version")!=str(torch.__version__):
        raise RuntimeError("checkpoint torch version mismatch; refuse load")
    current_code_sha=sha(__file__)
    if m.get("code_sha")!=current_code_sha:
        raise RuntimeError("checkpoint code SHA mismatch; refuse load")
    if expected_contract is not None and m.get("contract_sha")!=object_sha(expected_contract):
        raise RuntimeError("resume contract mismatch")
    return m
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True

class Windows:
    def __init__(self,path,split,k,h):
        x=np.load(path,allow_pickle=False); sid={"train":0,"dev":1,"test":2}[split]
        self.images=x["images"]; self.states=x["states"]; self.actions=x["actions"]
        self.ids=[(e,t) for e in np.where(x["split"]==sid)[0].tolist()
                  for t in range(self.images.shape[1]-max(k,h))]
        self.k,self.h=k,h
        assert self.ids
    def __len__(self): return len(self.ids)
    def batch(self,ii):
        e=np.array([self.ids[i][0] for i in ii]); t=np.array([self.ids[i][1] for i in ii])
        im0=np.stack([self.images[x,y] for x,y in zip(e,t)]).astype("float32")/255
        imf=np.stack([self.images[x,y+self.k] for x,y in zip(e,t)]).astype("float32")/255
        st=np.stack([self.states[x,y] for x,y in zip(e,t)])
        ac=np.stack([self.actions[x,y:y+self.h] for x,y in zip(e,t)])
        return {"image":torch.from_numpy(im0),"future":torch.from_numpy(imf),
                "state":torch.from_numpy(st),"action":torch.from_numpy(ac),
                "ids":[f"e{x}:t{y}" for x,y in zip(e,t)]}

class Encoder(nn.Module):
    def __init__(self,d):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(3,16,3,2,1),nn.ReLU(),
            nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(32,d))
    def forward(self,x): return self.net(x)

class MiniWAM(nn.Module):
    def __init__(self,d,h):
        super().__init__(); self.ctx=Encoder(d); self.target=Encoder(d)
        self.target.load_state_dict(self.ctx.state_dict()); self.target.requires_grad_(False)
        self.action=nn.Sequential(nn.Linear(d+4,128),nn.ReLU(),nn.Linear(128,h*2)); self.h=h
        self.world=nn.Sequential(nn.Linear(d+4+h*2,128),nn.ReLU(),nn.Linear(128,d))
    def forward(self,b,shuffle=False):
        z=self.ctx(b["image"]); action_hat=self.action(torch.cat([z,b["state"]],1)).view(-1,self.h,2)
        cond=b["action"]
        if shuffle:
            if len(cond)<2: raise ValueError("shuffle needs batch>=2")
            cond=cond[torch.roll(torch.arange(len(cond),device=cond.device),1)]
        zhat=self.world(torch.cat([z,b["state"],cond.flatten(1)],1))
        with torch.no_grad(): zfuture=self.target(b["future"])
        return action_hat,zhat,zfuture,z

def setup():
    world=int(os.getenv("WORLD_SIZE","1")); rank=int(os.getenv("RANK","0")); local=int(os.getenv("LOCAL_RANK","0"))
    if world>1: torch.cuda.set_device(local); dist.init_process_group("nccl")
    dev=torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    return world,rank,dev

def indices(n,global_batch,seed,step,rank,world):
    assert global_batch%world==0
    if n<global_batch: raise ValueError(f"dataset windows {n} < global_batch {global_batch}")
    ids=np.random.default_rng(seed+step).choice(n,size=global_batch,replace=False).tolist(); m=global_batch//world
    return ids[rank*m:(rank+1)*m]

def local_rng_state(dev):
    return {"torch":torch.get_rng_state(),"numpy":np.random.get_state(),"python":random.getstate(),
            "cuda":torch.cuda.get_rng_state(dev) if dev.type=="cuda" else None}

def save_ckpt(path,model,opt,scaler,step,attempted,contract,rank,world,dev):
    mine=local_rng_state(dev); rng_by_rank=[None for _ in range(world)]
    if world>1: dist.all_gather_object(rng_by_rank,mine)
    else: rng_by_rank[0]=mine
    error=[None]
    if rank==0:
        try:
            raw=model.module if isinstance(model,DDP) else model; tmp=Path(str(path)+".tmp")
            state={"model":raw.state_dict(),"optimizer":opt.state_dict(),"scaler":scaler.state_dict(),
                   "successful_step":step,"attempted_step":attempted,"rng_by_rank":rng_by_rank,
                   "contract":contract,"format_version":1,"torch_version":str(torch.__version__),
                   "code_sha":sha(__file__)}
            torch.save(state,tmp)
            checkpoint_sha=sha(tmp); checkpoint_bytes=tmp.stat().st_size
            os.replace(tmp,path)
            atomic_json(checkpoint_manifest_path(path),{
                "format_version":1,"checkpoint_file":Path(path).name,
                "bytes":checkpoint_bytes,"sha256":checkpoint_sha,
                "successful_step":step,"attempted_step":attempted,
                "torch_version":str(torch.__version__),"code_sha":sha(__file__),
                "contract_sha":object_sha(contract)})
        except Exception as e: error[0]=repr(e)
    if world>1: dist.broadcast_object_list(error,src=0)
    if error[0] is not None: raise RuntimeError("checkpoint publish failed on rank0: "+error[0])

def load_ckpt(path,model,opt,scaler,contract,rank,dev):
    sidecar=validate_checkpoint_file(path,contract)
    x=torch.load(path,map_location="cpu"); raw=model.module if isinstance(model,DDP) else model
    if x.get("contract")!=contract: raise RuntimeError("resume contract mismatch")
    if (x.get("format_version")!=1 or x.get("torch_version")!=sidecar["torch_version"] or
        x.get("code_sha")!=sidecar["code_sha"] or
        x.get("successful_step")!=sidecar["successful_step"] or
        x.get("attempted_step")!=sidecar["attempted_step"]):
        raise RuntimeError("checkpoint payload/sidecar mismatch")
    raw.load_state_dict(x["model"]); opt.load_state_dict(x["optimizer"]); scaler.load_state_dict(x["scaler"])
    if len(x["rng_by_rank"])!=contract["world_size"]: raise RuntimeError("per-rank RNG state count mismatch")
    r=x["rng_by_rank"][rank]; torch.set_rng_state(r["torch"]); np.random.set_state(r["numpy"]); random.setstate(r["python"])
    if dev.type=="cuda" and r["cuda"] is not None: torch.cuda.set_rng_state(r["cuda"],dev)
    return int(x["successful_step"]),int(x["attempted_step"])

def validate(data,cfg):
    ds=Windows(data,"train",cfg["future_offset"],cfg["action_horizon"]); b=ds.batch([0,1])
    assert b["image"].shape==(2,3,32,32) and b["action"].shape==(2,cfg["action_horizon"],2)
    assert torch.isfinite(b["state"]).all() and all("e" in x for x in b["ids"])
    print(json.dumps({"status":"PASS","windows":len(ds),"first_ids":b["ids"],
                      "image_shape":list(b["image"].shape),"action_shape":list(b["action"].shape)}))

def train(a,cfg):
    world,rank,dev=setup(); seed_all(cfg["seed"]); ds=Windows(a.data,cfg["split"],cfg["future_offset"],cfg["action_horizon"])
    model=MiniWAM(cfg["latent_dim"],cfg["action_horizon"]).to(dev)
    if world>1: model=DDP(model,device_ids=[dev.index])
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
    fp16=(a.precision or cfg["precision"])=="fp16"; scaler=torch.cuda.amp.GradScaler(enabled=fp16)
    out=Path(a.out); conflict=out.exists() and any(out.iterdir())
    if world>1:
        flag=torch.tensor(int(conflict),device=dev); dist.all_reduce(flag,op=dist.ReduceOp.MAX); conflict=bool(flag.item())
    if conflict and not a.resume: raise FileExistsError("non-resume run refuses non-empty output")
    out.mkdir(parents=True,exist_ok=True); data_sha=sha(a.data); code_sha=sha(__file__)
    resolved_precision="fp16" if fp16 else "fp32"
    contract={"config":cfg,"config_sha":sha(a.config),"data_sha":data_sha,"code_sha":code_sha,
              "world_size":world,"global_batch":cfg["global_batch"],"precision":resolved_precision}
    step=0; attempted=0
    if a.resume: step,attempted=load_ckpt(a.resume,model,opt,scaler,contract,rank,dev)
    limit=a.max_steps or cfg["max_steps"]
    if limit<step: raise ValueError(f"max_steps {limit} is behind resumed successful_step {step}")
    manifest={"status":"STARTED_NOT_VALIDATED","config":cfg,"config_sha":sha(a.config),"data_sha":data_sha,
              "code_sha":code_sha,"world_size":world,"global_batch":cfg["global_batch"],
              "precision":resolved_precision,"max_successful_steps":limit,"resume":a.resume,
              "resume_sha":sha(a.resume) if a.resume else None,"resume_contract":contract}
    if rank==0:(out/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    log=open(out/f"metrics.rank{rank}.jsonl","a",encoding="utf-8")
    while step<limit:
        ii=indices(len(ds),cfg["global_batch"],cfg["seed"],step,rank,world); b=ds.batch(ii)
        b={k:(v.to(dev,non_blocking=True) if torch.is_tensor(v) else v) for k,v in b.items()}
        opt.zero_grad(set_to_none=True)
        if dev.type=="cuda": torch.cuda.synchronize(dev)
        t0=time.perf_counter(); attempted+=1
        with torch.cuda.amp.autocast(enabled=fp16,dtype=torch.float16):
            ah,zh,zf,_=model(b,cfg["shuffle_action"]); la=(ah.float()-b["action"].float()).pow(2).mean()
            lf=(zh.float()-zf.float()).pow(2).mean(); loss=la+cfg["lambda_world"]*lf
        finite=torch.isfinite(loss).to(dev,dtype=torch.int32)
        if world>1: dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        if not finite.item(): raise FloatingPointError("nonfinite loss on at least one rank")
        old=scaler.get_scale(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        local_grad_ok=all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
        grad_ok=torch.tensor(int(local_grad_ok),device=dev,dtype=torch.int32)
        if world>1: dist.all_reduce(grad_ok,op=dist.ReduceOp.MIN)
        if not grad_ok.item(): raise FloatingPointError("nonfinite gradient on at least one rank; no optimizer step committed")
        gn=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["clip_norm"])
        norm_ok=torch.tensor(int(bool(torch.isfinite(torch.as_tensor(gn)).item())),device=dev,dtype=torch.int32)
        if world>1: dist.all_reduce(norm_ok,op=dist.ReduceOp.MIN)
        if not norm_ok.item(): raise FloatingPointError("nonfinite global grad norm on at least one rank; no optimizer step committed")
        scaler.step(opt); scaler.update()
        local_skipped=int(scaler.get_scale()<old); skip_min=torch.tensor(local_skipped,device=dev); skip_max=skip_min.clone()
        if world>1:
            dist.all_reduce(skip_min,op=dist.ReduceOp.MIN); dist.all_reduce(skip_max,op=dist.ReduceOp.MAX)
        if skip_min.item()!=skip_max.item(): raise RuntimeError("AMP skip decision diverged across ranks")
        skipped=bool(skip_max.item())
        if skipped: raise FloatingPointError("AMP skipped an update; aborting so successful_step remains unambiguous")
        step+=1
        if dev.type=="cuda": torch.cuda.synchronize(dev)
        rec={"attempted_step":attempted,"successful_step":step,"loss":loss.item(),"loss_action":la.item(),"loss_future":lf.item(),
             "grad_norm":float(gn),"loss_scale":scaler.get_scale(),"skipped":bool(skipped),
             "step_ms":1000*(time.perf_counter()-t0),"sample_ids":b["ids"][:4],
             "peak_allocated_gb":torch.cuda.max_memory_allocated(dev)/1024**3 if dev.type=="cuda" else 0.0,
             "peak_reserved_gb":torch.cuda.max_memory_reserved(dev)/1024**3 if dev.type=="cuda" else 0.0}
        log.write(json.dumps(rec)+"\n"); log.flush()
        if step%cfg["save_every"]==0 or step==limit:
            save_ckpt(out/f"step_{step:06d}.pt",model,opt,scaler,step,attempted,contract,rank,world,dev)
            if rank==0: print(json.dumps(rec))
        if world>1: dist.barrier()
    log.close()
    if rank==0:
        manifest["status"]="COMPLETED_PENDING_EVAL";manifest["successful_steps"]=step;manifest["attempted_steps"]=attempted
        (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    if world>1: dist.destroy_process_group()

@torch.no_grad()
def evaluate(a,cfg):
    _,_,dev=setup(); seed_all(cfg["seed"]); ds=Windows(a.data,a.split,cfg["future_offset"],cfg["action_horizon"])
    sidecar=validate_checkpoint_file(a.checkpoint)
    m=MiniWAM(cfg["latent_dim"],cfg["action_horizon"]).to(dev); x=torch.load(a.checkpoint,map_location="cpu")
    if (x.get("format_version")!=1 or x.get("torch_version")!=sidecar["torch_version"] or
        x.get("code_sha")!=sidecar["code_sha"] or
        x.get("successful_step")!=sidecar["successful_step"]):
        raise RuntimeError("checkpoint payload/sidecar mismatch")
    contract=x.get("contract",{})
    expected={"config":cfg,"config_sha":sha(a.config),"data_sha":sha(a.data),"code_sha":sha(__file__)}
    if any(contract.get(k)!=v for k,v in expected.items()): raise RuntimeError("eval lineage mismatch")
    m.load_state_dict(x["model"]); m.eval(); sums={"n":0,"action":0.,"future":0.,"last":0.}
    for start in range(0,len(ds),a.batch_size):
        b=ds.batch(list(range(start,min(start+a.batch_size,len(ds)))))
        b={k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in b.items()}
        ah,zh,zf,_=m(b,False); zlast=m.target(b["image"]); n=len(b["image"]); sums["n"]+=n
        sums["action"]+=(ah.float()-b["action"].float()).pow(2).mean().item()*n
        sums["future"]+=(zh.float()-zf.float()).pow(2).mean().item()*n
        sums["last"]+=(zlast.float()-zf.float()).pow(2).mean().item()*n
    out={"status":"PASS","split":a.split,"checkpoint":a.checkpoint,"checkpoint_sha":sha(a.checkpoint),
         "config_sha":expected["config_sha"],"data_sha":expected["data_sha"],"code_sha":expected["code_sha"],
         **{k:v/sums["n"] for k,v in sums.items() if k!="n"}}
    out["relative_improvement"]=(out["last"]-out["future"])/(out["last"]+1e-12)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out))

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
    for name in ("validate","train","eval"):
        q=sub.add_parser(name); q.add_argument("--config",required=True); q.add_argument("--data",required=True)
        if name=="train":
            q.add_argument("--out",required=True); q.add_argument("--max-steps",type=int); q.add_argument("--precision",choices=["fp32","fp16"]); q.add_argument("--resume")
        if name=="eval":
            q.add_argument("--checkpoint",required=True); q.add_argument("--split",default="dev"); q.add_argument("--batch-size",type=int,default=64); q.add_argument("--out",required=True)
    a=p.parse_args(); cfg=json.loads(Path(a.config).read_text())
    if a.cmd=="validate": validate(a.data,cfg)
    elif a.cmd=="train": train(a,cfg)
    else: evaluate(a,cfg)
if __name__=="__main__": main()
```

注意：为了使教程代码短且 resume 可解释，每个 successful step 都以 `seed+successful_step` 做无放回抽样，再按 rank 切成互不重叠的 shard；数据窗口数小于 global batch 会立即拒绝。它不是生产级 DataLoader。正式视频载荷需替换 `Windows.batch`，但保持相同返回 schema 和 episode split。不应偷偷将 optional PushT 接入现有命令；必须先写、测试并 code-review adapter。

### 2.4 `tests/test_core.py`：完整单测

将紧随其后的代码块原样保存为 `tests/test_core.py`：

```python
# file: tests/test_core.py
import json, tempfile, unittest
from pathlib import Path
import torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run import MiniWAM, indices

class CoreTest(unittest.TestCase):
    def test_shapes_and_stop_grad(self):
        m=MiniWAM(64,4); b={"image":torch.rand(3,3,32,32),"future":torch.rand(3,3,32,32),
          "state":torch.rand(3,4),"action":torch.rand(3,4,2)}
        ah,zh,zf,_=m(b,False); self.assertEqual(tuple(ah.shape),(3,4,2)); self.assertEqual(tuple(zh.shape),(3,64))
        self.assertFalse(zf.requires_grad); self.assertTrue(all(not p.requires_grad for p in m.target.parameters()))
    def test_derangement(self):
        x=torch.arange(7); p=torch.roll(x,1); self.assertTrue(torch.all(x!=p))
        with self.assertRaises(ValueError):
            m=MiniWAM(8,2); m({"image":torch.rand(1,3,32,32),"future":torch.rand(1,3,32,32),
              "state":torch.rand(1,4),"action":torch.rand(1,2,2)},True)
    def test_global_partition(self):
        parts=[indices(100,16,9,3,r,4) for r in range(4)]
        flat=sum(parts,[])
        self.assertEqual(sum(map(len,parts)),16); self.assertEqual(len(parts[0]),4)
        self.assertEqual(len(set(flat)),16)
        with self.assertRaises(ValueError): indices(15,16,9,3,0,4)
if __name__=="__main__": unittest.main()
```

## Day 1：构造数据并证明时序正确

### 为什么做

先消除跨 episode、future offset 和 split 泄漏；这些错会制造最可信的假结果。

### 输入与前置检查

从 `week11-miniwam` 执行；确认上述三个 Python 文件和三份 JSON 已复制完成，`python -m json.tool configs/B_action_world.json` 成功。

### 本日要创建/修改的文件

生成 `data/toy_push_v1.npz`、`.meta.json`、`manifests/core.sha256` 和 `reports/pre_registration.md`。预注册写唯一 primary：dev action MSE；世界指标为 relative improvement；不能看 test 后改。

### 实现

实现已完整给在 §2；Day 1 不新增隐藏脚本。将紧随其后的代码块保存为 `reports/pre_registration.md`，只能在查看 dev/test 指标前填写括号中的数值并冻结 SHA：

```markdown
# MiniWAM pre-registration
- primary: dev action MSE（越低越好）
- world metric: dev RI=(last-future)/(last+1e-12)
- world gate: RI >= 0.10
- action non-inferiority tolerance: [运行前填写]
- seed: 1107; max successful steps: 300; global batch: 64
- selection: 只用 dev；test 仅在选择冻结后运行一次
- status: PRE_REGISTERED_NOT_RUN
```

### 执行命令

```bash
cd week11-miniwam
python -m py_compile make_toy_data.py run.py tests/test_core.py
python -m unittest -v tests/test_core.py
python make_toy_data.py --out data/toy_push_v1.npz --episodes 60 --steps 24 --seed 1101
sha256sum make_toy_data.py run.py configs/*.json data/toy_push_v1.npz > manifests/core.sha256
python run.py validate --config configs/B_action_world.json --data data/toy_push_v1.npz
```

### 预期观测（估算/示例，不是实测）

单测显示 3 tests `OK`；generator/validator 输出 JSON，其中 `status=PASS`、image `[2,3,32,32]`、action `[2,4,2]`。耗时估算 <2 分钟，CPU 即可。

### 验收条件

文件 SHA 已记录；三个 split 都有 episode；没有非有限数；window 不跨 episode；配置可解析。

### 若失败，按什么顺序查

先 `python --version` → `python -c 'import numpy,torch;print(numpy.__version__,torch.__version__)'` → 单独运行 generator → `python -c 'import numpy as n;x=n.load("data/toy_push_v1.npz");print(x.files,{k:x[k].shape for k in x.files if hasattr(x[k],"shape")})'`。

### 当日证据清单

`pip-freeze.before.txt`、`core.sha256`、meta JSON、validator stdout、预注册。均标“待用户运行验证”。

## Day 2：单 batch 前反向、20-step smoke 与恢复

### 为什么做

在昂贵训练前证明 loss、FP16、梯度、checkpoint 和 resume 路径成立。

### 输入与前置检查

Day 1 PASS；GPU 探针通过。若只有 CPU，先跑 2-step FP32，FP16 跳过并标 `ENV-BLOCKED`。

### 本日要创建/修改的文件

只生成 `runs/day2_*`；不改代码来“让测试过”。

### 实现

`run.py` 已实现完整状态 checkpoint、原子发布和 lineage 校验。

### 执行命令

```bash
cd week11-miniwam
CUDA_VISIBLE_DEVICES=0 python run.py train --config configs/B_action_world.json \
  --data data/toy_push_v1.npz --out runs/day2_fp32 --precision fp32 --max-steps 1
CUDA_VISIBLE_DEVICES=0 python run.py train --config configs/B_action_world.json \
  --data data/toy_push_v1.npz --out runs/day2_smoke --precision fp16 --max-steps 20
CUDA_VISIBLE_DEVICES=0 python run.py train --config configs/B_action_world.json \
  --data data/toy_push_v1.npz --out runs/day2_continuous30 --precision fp16 --max-steps 30
CUDA_VISIBLE_DEVICES=0 python run.py train --config configs/B_action_world.json \
  --data data/toy_push_v1.npz --out runs/day2_resume --precision fp16 --max-steps 30 \
  --resume runs/day2_smoke/step_000020.pt
python - <<'PY'
import torch
from run import validate_checkpoint_file
pa="runs/day2_continuous30/step_000030.pt"; pb="runs/day2_resume/step_000030.pt"
validate_checkpoint_file(pa); validate_checkpoint_file(pb)
a=torch.load(pa,map_location="cpu"); b=torch.load(pb,map_location="cpu")
assert a["contract"]==b["contract"] and a["successful_step"]==b["successful_step"]==30
d=max((a["model"][k]-b["model"][k]).abs().max().item() for k in a["model"])
print({"continuous_vs_resume_max_abs":d}); assert d<=1e-6
PY
```

### 预期观测（估算/示例，不是实测）

JSONL 含 `loss/loss_action/loss_future/grad_norm/loss_scale/skipped/sample_ids`；checkpoint 到 step 20，连续路径与 resume 路径都到 30，并打印 model max-abs 差。Toy 模型显存估算 <2GB、总耗时 <8 分钟。

### 验收条件

loss/grad 全有限；target 无梯度；`skipped` 必须恒为 false（任一 rank overflow 都使全作业失败且不发布新 checkpoint）；resume 的 attempted/successful step 连续；同机同精度 continuous-vs-resume model max-abs≤1e-6；`.tmp` 不残留。

### 若失败，按什么顺序查

FP32 也坏：查数据/shape；仅 FP16 坏：把 reduction dtype、LR、输入范围依次查；`resume contract mismatch`：不得 `strict=False`，比较 `sha256sum run.py data/toy_push_v1.npz configs/B_action_world.json` 以及 checkpoint 的 world/precision；OOM：batch 64→16 并同步修改三配置，重新冻结 config hash。

### 当日证据清单

四条训练命令、各自 metrics、step 20/30 checkpoint、continuous-vs-resume 对比 stdout、错误日志或 PASS 标签。

## Day 3：A/B/C 等预算正式短训

### 为什么做

建立唯一变量为 world loss/action shuffle 的因果对照。

### 输入与前置检查

Day 2 PASS；运行 `diff -u configs/B_action_world.json configs/C_action_world_shuffle.json`，确认只有 run 与 shuffle 不同；A 只额外把 lambda 置 0。

### 本日要创建/修改的文件

冻结三份配置到各 run 目录；训练过程中不修改。

### 实现

单卡先执行；公司获批后可把同一命令前缀换成 `torchrun --standalone --nproc_per_node=4`。全局 batch=64，4 卡每 rank 16；DDP 不再手工除 world size。

### 执行命令

```bash
cd week11-miniwam
for c in A_action_only B_action_world C_action_world_shuffle; do
  CUDA_VISIBLE_DEVICES=0 python run.py train --config configs/${c}.json \
    --data data/toy_push_v1.npz --out runs/${c}_s1107 --precision fp16 --max-steps 300
done
# 公司四卡可选；与单卡是新的 scaling run，不混写结果：
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 run.py train \
  --config configs/B_action_world.json --data data/toy_push_v1.npz \
  --out runs/B_action_world_s1107_ws4 --precision fp16 --max-steps 100
```

### 预期观测（估算/示例，不是实测）

每组有 `step_000300.pt` 和逐步 JSONL。三组 successful steps 相等、sample ID 序列相同；B/C 比 A 多 world-loss 计算。单卡估算每组数分钟，真实视觉载荷不可套用。

### 验收条件

三组都完成 300 successful steps；样本 ID、seed、global batch 相同；非有限=0；每组 skip=0（出现一次即 FAIL-SYSTEM）；多卡各 rank 最终参数由 DDP 同步。

### 若失败，按什么顺序查

对比三配置 SHA/diff → 比 sample_ids → 看首个 diverge step → 单卡重跑该 step → 多卡挂起时设 `NCCL_DEBUG=INFO` 做 2-step 有界复现，不能直接禁用 P2P 掩盖问题。

### 当日证据清单

config SHA、code/data SHA、三组 JSONL/checkpoint、world size/rank mapping、终端状态。公司内容不外传。

## Day 4：独立 eval、transfer 与失败注入

### 为什么做

teacher-forced loss 不能回答 action/world 是否真正改善；eval 必须是独立进程。

### 输入与前置检查

Day 3 的三个完整 checkpoint 可加载；test split 此前未用于调参。

### 本日要创建/修改的文件

生成 `runs/eval/*.json`。若要接 PushT，需另建经 review 的 adapter 与 tests；本手册核心命令不假定它存在。

### 实现

评测实现已在 `run.py eval`：先核对 checkpoint/config/data/code lineage；A/B/C 评测一律输入同一条正确对齐 action（C 只在训练时 shuffle，不能在 eval 再制造输入劣势）；再以同一个冻结 target encoder 同时编码当前帧与未来帧作为 last-frame 基线，计算 action MSE、future MSE、last-frame MSE 与相对改善。不得用训练中的 context encoder 生成基线当前帧特征。

### 执行命令

```bash
cd week11-miniwam
mkdir -p runs/eval
for c in A_action_only B_action_world C_action_world_shuffle; do
  CUDA_VISIBLE_DEVICES=0 python run.py eval --config configs/${c}.json \
    --data data/toy_push_v1.npz --checkpoint runs/${c}_s1107/step_000300.pt \
    --split dev --batch-size 64 --out runs/eval/${c}_dev.json
done
# 数据损坏注入：副本被改后 resume 必须因 data SHA 不同而拒绝；同时断言失败确实发生且原因正确
cp data/toy_push_v1.npz data/corrupt_probe.npz
printf x >> data/corrupt_probe.npz
set +e
CUDA_VISIBLE_DEVICES=0 python run.py train --config configs/B_action_world.json \
  --data data/corrupt_probe.npz --out runs/corrupt_probe --max-steps 301 \
  --resume runs/B_action_world_s1107/step_000300.pt > runs/corrupt_probe.stderr 2>&1
code=$?
set -e
test "$code" -ne 0
grep -q "resume contract mismatch" runs/corrupt_probe.stderr
printf '{"status":"PASS","exit_code":%s,"expected_error":"resume contract mismatch"}\n' "$code" \
  | tee runs/corrupt_probe_assertion.json
```

### 预期观测（估算/示例，不是实测）

三个 eval JSON 可并排读且含 checkpoint/config/data/code SHA；损坏副本 resume 报 `resume contract mismatch`。不要预填 B 优于 C 的数字。

### 验收条件

B 相对 last-frame 的门槛按预注册；B 必须优于 C 才能说利用动作；B action dev 不比 A 恶化超过预注册容差。只有 Toy 数据时结论上限为 pipeline PASS / domain INCONCLUSIVE。

### 若失败，按什么顺序查

checkpoint/config 是否配错 → eval split/hash → B/C shuffle 路径 → target latent 方差 → action grad → future offset/运动分桶。系统错未清零前不得标 FAIL-MODEL。

### 当日证据清单

三个 eval JSON、命令、checkpoint SHA、坏数据拒绝日志、transfer 是否 `PASS/ENV-BLOCKED`。

## Day 5：成本、结论与可恢复交接

### 为什么做

把“跑过”压缩为可审计结论，并决定是否进入 Week 14。

### 输入与前置检查

Day 1–4 证据齐全；没有拿 test 反复调 λ。

### 本日要创建/修改的文件

`reports/decision.md`、`reports/evidence_manifest.json`。不得把公司实际内容复制回本教程仓库。

### 实现

报告必须逐行引用 run/config/data/checkpoint/eval；指标缺失填 `NOT-RUN`，不能填 0。

将紧随其后的第一个代码块保存为 `reports/decision.md`，第二个保存为 `reports/evidence_manifest.json`；运行后只能用真实路径/数值替换 `NOT-RUN`：

```markdown
# MiniWAM decision
- status: NOT-RUN
- selected label: NOT-RUN
- primary/dev action MSE: NOT-RUN
- world RI / shuffled-action contrast: NOT-RUN
- system cost and failure bucket: NOT-RUN
- scope: Toy-only 不得标 PASS-WORLD
```

```json
{"status":"NOT-RUN","config_sha":"NOT-RUN","data_sha":"NOT-RUN","code_sha":"NOT-RUN",
 "checkpoint_sha":"NOT-RUN","eval_files":[],"company_artifacts_exported":false}
```

### 执行命令

```bash
cd week11-miniwam
sha256sum configs/*.json data/toy_push_v1.npz run.py runs/*/step_*.pt* runs/eval/*.json \
  > reports/artifacts.sha256
python - <<'PY'
import glob,json
for p in sorted(glob.glob("runs/eval/*_dev.json")):
    x=json.load(open(p)); print(p,{k:x[k] for k in ("action","future","last","relative_improvement")})
PY
```

### 预期观测（估算/示例，不是实测）

终端只汇总已存在 JSON，不上传。`artifacts.sha256` 能将报告定位到唯一对象。

### 验收条件

选择且只选择一个标签；写明系统成本、限制、失败桶、下一动作。`PASS-WORLD` 还要求真实公开/获批数据与动作/OOD 指标；Toy-only 不可获得该标签。

### 若失败，按什么顺序查

先补 lineage → 补独立 eval → 查公平性 → 仍缺 seed/transfer 则标 `INCONCLUSIVE`，不延长无上限训练。

### 当日证据清单

decision、artifact SHA、每条 PASS/FAIL 的证据路径、未决风险、Week 14 是否采用本载荷。

## 3. 曲线读取与恢复速查

| 曲线/症状 | 含义候选 | 最小动作 |
|---|---|---|
| action/future 同时平滑下降 | 仅说明优化在进行 | 仍跑独立 eval 与 C |
| future 降、action 升 | world 梯度冲突或 λ 过大 | 查看 shared grad；按预注册停止，不看 test 调 λ |
| loss scale 连续回退 | overflow 或坏 batch | 固定首个 sample IDs，FP32 重放 |
| step time 周期尖峰 | save/同步/I/O | 对齐 save step，拆时间 |
| rank 日志 sample ID 重合 | global partition 错 | 停止多卡，检查 `global_batch/world` |

恢复原则：只从完整 `.pt`；`data_sha/code_sha` 不同拒绝；OOM 先降低 micro/global batch并把它登记为新实验；NaN 先重放首个坏 step；DDP hang 先有界 2-step + `NCCL_DEBUG=INFO`。任何修复后从 smoke 重新过门，不在失败 run 上续写“成功”。
