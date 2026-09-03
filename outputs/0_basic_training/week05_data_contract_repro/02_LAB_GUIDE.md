# Week 05 实践篇：从空目录构建 PushT 数据合同与可复现探针

> **全篇待执行、待验证。** 唯一正式Task-P是公开 `lerobot/pusht` 固定snapshot；Week04合成数据只作离线fallback/corruption oracle。公司机器上的数据、视频、代码、路径、hash、统计、日志、图、checkpoint和性能信息不得导出。个人 RTX 5070 Ti 16GB 必须从公开源独立下载并重跑，不能沿用公司产物。公司复用既有Conda，不创建Conda/venv，不升级PyTorch2.1，也不安装current LeRobot。

## Day 0：只读环境、磁盘与兼容审计

### 为什么做

PushT snapshot是Parquet+MP4；先确认磁盘、pyarrow/video工具与现有torch环境，不让current LeRobot的新依赖替换公司栈。

### 输入与前置检查

估算：metadata+parquet较小，完整视频需先由远端file list估算后再批准；建议预留10GB但不得当成实值。validator数分钟，hash取决于文件量，50-step probe数分钟。GPU不是Day1–4必需。

`[公司 Linux]`：

```bash
which python
python -V
python -m pip check
df -h .
git --version
ffprobe -version
python -c "import torch; print(torch.__version__,torch.version.cuda,torch.cuda.is_available())"
python -c "import pandas,pyarrow,huggingface_hub; print(pandas.__version__,pyarrow.__version__,huggingface_hub.__version__)"
```

`ffprobe`或Python包缺失只记录，不自行改系统包。`[个人 PowerShell]`：

```powershell
python -V
python -m pip check
Get-PSDrive -PSProvider FileSystem
ffprobe -version
python -c "import pandas,pyarrow,huggingface_hub,torch; print(pandas.__version__,pyarrow.__version__,huggingface_hub.__version__,torch.__version__)"
```

### 本日要创建/修改的文件

`[公司 Linux]`：

```bash
mkdir -p week05-data/{configs,data/raw/pusht,data/processed,manifests,src,tests,reports,runs}
cd week05-data
touch src/__init__.py reports/task_card.md reports/data_card.md reports/decision.md
python -m pip freeze > manifests/pip-before.txt
```

`[个人 PowerShell]`：

```powershell
New-Item -ItemType Directory -Force week05-data,week05-data/configs,week05-data/data/raw/pusht,week05-data/data/processed,week05-data/manifests,week05-data/src,week05-data/tests,week05-data/reports,week05-data/runs | Out-Null
Set-Location week05-data
New-Item -ItemType File -Force src/__init__.py,reports/task_card.md,reports/data_card.md,reports/decision.md | Out-Null
python -m pip freeze | Set-Content -Encoding utf8 manifests/pip-before.txt
```

```text
src/fetch_public.py       固定revision选择性下载
src/contract.py           parquet加载、validator、split、lag、windows、manifest
src/repro_probe.py        stateful sampler与50-step resume probe
tests/test_contract.py    corruption/window/lag/sampler单测
reports/task_card.md      语义先验与待核验项
reports/data_card.md      统计/限制/适用范围
manifests/                source/splits/stats/files
```

### 实现

公司候选依赖只审计 `pandas 2.0–2.2, pyarrow 12–17, huggingface_hub 0.26.x, numpy<2`；具体以当前已批准组合为准。current LeRobot在核验日要求较新Python/torch，禁止安装。

### 执行命令

`[两者/Python]` resolver dry-run：

```bash
python -m pip install --dry-run "numpy<2" "pandas>=2,<2.3" "pyarrow>=12,<18" "huggingface_hub==0.26.5" "pytest<9"
```

只有不改变torch/torchvision且公司镜像批准时才安装同规格；随后`python -m pip check`。回滚按`pip-before.txt`由管理员恢复，不用无依据的uninstall。

### 预期观测（估算/示例，不是实测）

Python能够import parquet/Hub客户端；torch版本未变化；完整视频磁盘预算仍待远端清单核验。

### 验收条件

目录存在、环境快照存在、resolver不改torch、至少parquet依赖可用。

### 若失败，按什么顺序查

解释器→pip check→pyarrow ABI→批准wheel→磁盘；不安装current LeRobot“顺便解决”。

### 当日证据清单

`pip-before.txt`、依赖矩阵实测列、磁盘估算和缺失工具。

## Day 1：固定revision下载、Task Card与只读抽查

### 为什么做

先保存dataset card、metadata、源hash和许可，再解释字段。只有train split意味着本周的dev/OOD是内部split。

### 输入与前置检查

固定 `lerobot/pusht@7628202a2180972f291ba1bc6723834921e72c19`；dataset card 标 MIT，固定 snapshot 的 `meta/info.json` 明确声明 `codebase_version: "v3.0"`。公司先审批网络、许可与磁盘。

| 实际对象 | allow pattern | 完整性 |
|---|---|---|
| card/metadata | `README.md`, `meta/*` | resolved sha、JSON可解析、metadata统计 |
| state/action | `data/*/*.parquet` | 文件数/bytes/SHA、schema、row总数 |
| 可选图像 | `videos/*/*/*.mp4` | ffprobe、固定30 episode人工对齐；未下则标未验证 |

### 本日要创建/修改的文件

```python
# src/fetch_public.py
import argparse, hashlib, json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument("--repo",default="lerobot/pusht");p.add_argument("--revision",required=True);p.add_argument("--out",required=True);p.add_argument("--with-videos",action="store_true");p.add_argument("--manifest",required=True)
    p.add_argument("--local-only",action="store_true");p.add_argument("--declared-resolved-revision");a=p.parse_args()
    root=Path(a.out)
    if a.local_only:
        if not a.declared_resolved_revision:raise ValueError("local-only requires administrator-provided --declared-resolved-revision")
        if a.declared_resolved_revision!=a.revision:raise ValueError("local snapshot revision does not match frozen course revision")
        if not root.is_dir():raise FileNotFoundError(root)
        resolved=a.declared_resolved_revision;source_mode="approved-local-snapshot"
    else:
        if a.declared_resolved_revision:raise ValueError("do not combine network mode with --declared-resolved-revision")
        resolved=HfApi().dataset_info(a.repo,revision=a.revision).sha;source_mode="huggingface-hub"
        if resolved!=a.revision:raise ValueError(f"resolved revision {resolved} != requested immutable revision {a.revision}")
    patterns=["README.md","meta/*","data/*/*.parquet"]+(["videos/*/*/*.mp4"] if a.with_videos else [])
    if not a.local_only:snapshot_download(repo_id=a.repo,repo_type="dataset",revision=a.revision,allow_patterns=patterns,local_dir=a.out,local_dir_use_symlinks=False)
    required=[root/"README.md",root/"meta"/"info.json"]
    if any(not x.is_file() for x in required):raise FileNotFoundError(f"missing required snapshot files: {required}")
    if not list(root.glob("data/*/*.parquet")):raise FileNotFoundError("no data/*/*.parquet in snapshot")
    videos=sorted(root.glob("videos/*/*/*.mp4"))
    if a.with_videos and not videos:raise FileNotFoundError("--with-videos requested but no videos/*/*/*.mp4 found")
    files=[]
    for f in sorted(x for x in root.rglob("*") if x.is_file()):files.append({"path":f.relative_to(root).as_posix(),"bytes":f.stat().st_size,"sha256":sha(f)})
    obj={"repo":a.repo,"requested_revision":a.revision,"resolved_revision":resolved,"source_mode":source_mode,"requested_with_videos":a.with_videos,"with_videos":bool(videos),"video_files":len(videos),"files":files}
    Path(a.manifest).write_text(json.dumps(obj,indent=2)+"\n",encoding="utf-8");print(json.dumps({"resolved":resolved,"files":len(files),"bytes":sum(x["bytes"] for x in files)}))
if __name__=="__main__":main()
```

`reports/task_card.md` 至少写：source revision、Push-T任务、image/state/action字段、已确认与待确认的单位/坐标、10FPS/0.1s、observation→action顺序、H=16、success、内部ID/OOD、排除规则。不能把`motor_0/1`自行解释成关节或力。

### 实现

下载函数先解析 resolved sha，再按 patterns 下载和逐文件 hash。`--local-only` 不联网，只审计管理员已放入目标目录的 snapshot，并要求管理员提供的 resolved revision 与课程冻结值完全一致；manifest 会明确记录来源模式。`--with-videos` 必须单独审批；不开启时图像对齐项保持未验证。

### 执行命令

先在任一网络分支之前静态检查刚创建的下载器：

```bash
python -m py_compile src/fetch_public.py
```

网络获批，先metadata/parquet：

`[公司 Linux]` 或 `[个人 PowerShell]`：

```bash
python -m src.fetch_public --revision 7628202a2180972f291ba1bc6723834921e72c19 --out data/raw/pusht --manifest manifests/source.json
```

审批视频后在**同一空snapshot目录**重跑带视频；为避免非空内容混淆，先由用户选择新目录 `data/raw/pusht_with_video` 并创建：

`[公司 Linux]`：

```bash
mkdir -p data/raw/pusht_with_video
python -m src.fetch_public --revision 7628202a2180972f291ba1bc6723834921e72c19 --out data/raw/pusht_with_video --with-videos --manifest manifests/source_with_video.json
```

`[个人 PowerShell]`：

```powershell
New-Item -ItemType Directory -Force data/raw/pusht_with_video | Out-Null
python -m src.fetch_public --revision 7628202a2180972f291ba1bc6723834921e72c19 --out data/raw/pusht_with_video --with-videos --manifest manifests/source_with_video.json
```

公司网络不可达：仅管理员提供同一 snapshot 到 `data/raw/pusht`，并把其 resolved revision 作为获批文字交付；随后运行下列本地审计，不得个人转运或绕代理：

```bash
python -m src.fetch_public --local-only --revision 7628202a2180972f291ba1bc6723834921e72c19 --declared-resolved-revision 7628202a2180972f291ba1bc6723834921e72c19 --out data/raw/pusht --manifest manifests/source.json
```

无论采用网络还是管理员预置分支，随后读取全部实际存在的data parquet schema（该固定snapshot目前只有一个）：

```bash
python -c "from pathlib import Path; import pyarrow.parquet as pq; P=sorted(Path('data/raw/pusht').glob('data/*/*.parquet')); assert P; [pq.read_schema(p) for p in P[:min(3,len(P))]]; print('PARQUET_READ_PASS',len(P),[str(p) for p in P[:min(3,len(P))]])"
```

没有公开 snapshot 时可读 Week04 合成 NPZ 改写 validator 测试，但周结论 `INCONCLUSIVE`。

### 预期观测（估算/示例，不是实测）

resolved sha应匹配请求；metadata源声明206 episodes/25,650 frames/10FPS。实际parquet row统计由Day2验证，不预填。

### 验收条件

card/许可/revision/file hash齐；文件非LFS指针；读取 `min(3, parquet文件总数)` 个文件（固定v3 snapshot当前只有一个data parquet，因此应读这一个）；30 episode视频人工检查完成或明确`IMAGE_ALIGNMENT_UNVERIFIED`。

### 若失败，按什么顺序查

revision→网络错误类型→LFS指针/size→磁盘→pyarrow→批准镜像。不能切main继续。

### 当日证据清单

`source.json`、README/license审批、Task Card、人工检查表（全部本机）。

## Day 2：完整 validator 与 corruption suite

### 为什么做

把schema、时间、边界、范围和重复假设写成会失败的程序。

### 输入与前置检查

`data/raw/pusht/meta/info.json`与parquet存在；视频不是validator核心依赖。

### 本日要创建/修改的文件

```python
# src/contract.py
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
REQ={"episode_index","frame_index","timestamp","observation.state","action","next.done","next.success","index","task_index"}
FEATURES={"observation.state":("float32",[2]),"action":("float32",[2]),"episode_index":("int64",[1]),"frame_index":("int64",[1]),"timestamp":("float32",[1]),"next.done":("bool",[1]),"next.success":("bool",[1]),"index":("int64",[1]),"task_index":("int64",[1])}
ARROW_LEAF={"observation.state":"float","action":"float","episode_index":"int64","frame_index":"int64","timestamp":"float","next.done":"bool","next.success":"bool","index":"int64","task_index":"int64"}
def scalar(x):
    a=np.asarray(x);return a.item() if a.size==1 else x
def vector(x,n):
    a=np.asarray(x)
    if a.shape!=(n,):raise ValueError(f"expected ({n},), got {a.shape}")
    if a.dtype!=np.dtype("float32"):raise ValueError(f"expected source float32, got {a.dtype}")
    return a.astype(np.float32,copy=False)
def binary_flag(x):
    a=np.asarray(x)
    if a.size!=1:raise ValueError("flag must be scalar")
    v=a.item()
    if not isinstance(v,(bool,np.bool_,int,np.integer)) or int(v) not in (0,1):raise ValueError("flag must be bool/0/1")
    return bool(v)
def load_info(root):
    p=Path(root)/"meta"/"info.json"
    if not p.is_file():raise FileNotFoundError(p)
    info=json.loads(p.read_text(encoding="utf-8"))
    if info.get("codebase_version")!="v3.0":raise ValueError(f"expected v3.0, got {info.get('codebase_version')}")
    if int(info.get("fps",-1))!=10:raise ValueError(f"expected fps=10, got {info.get('fps')}")
    for name,(dtype,shape) in FEATURES.items():
        got=info.get("features",{}).get(name)
        if not got or got.get("dtype")!=dtype or got.get("shape")!=shape:raise ValueError(f"info.json feature mismatch for {name}: {got}")
    return info
def arrow_leaf(t):
    while pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):t=t.value_type
    return str(t)
def check_parquet_schema(path):
    schema=pq.read_schema(path)
    missing=REQ-set(schema.names)
    if missing:raise ValueError(f"{path}: missing {sorted(missing)}")
    for name,expected in ARROW_LEAF.items():
        got=arrow_leaf(schema.field(name).type)
        if got!=expected:raise ValueError(f"{path}: {name} physical dtype {got}, expected {expected}")
def load_episodes(root):
    info=load_info(root);groups={};paths=sorted(Path(root).glob("data/*/*.parquet"));total_rows=0
    if int(info.get("total_frames",0))<=0 or int(info.get("total_episodes",0))<=0:raise ValueError("metadata declares zero frames/episodes")
    if not paths:raise FileNotFoundError("no data/*/*.parquet")
    for path in paths:
        check_parquet_schema(path)
        df=pd.read_parquet(path)
        if not REQ.issubset(df.columns):raise ValueError(f"{path}: missing {sorted(REQ-set(df.columns))}")
        if len(df)==0:raise ValueError(f"{path}: empty parquet")
        total_rows+=len(df)
        for row in df.to_dict("records"):
            e=int(scalar(row["episode_index"]));groups.setdefault(e,[]).append(row)
    if not groups or total_rows<=0:raise ValueError("snapshot contains no episodes/frames")
    if total_rows!=int(info["total_frames"]):raise ValueError(f"frame count {total_rows} != info.json {info['total_frames']}")
    if len(groups)!=int(info["total_episodes"]):raise ValueError(f"episode count {len(groups)} != info.json {info['total_episodes']}")
    if set(groups)!=set(range(int(info["total_episodes"]))):raise ValueError("episode IDs are not contiguous 0..total_episodes-1")
    return {e:sorted(rows,key=lambda r:int(scalar(r["frame_index"]))) for e,rows in groups.items()}
def validate_episodes(eps,fps=10,total_tasks=None):
    errors=[];global_ids=set();success_nonterminal=0;success_terminal=0
    if not eps:errors.append(["dataset","no_episodes"])
    for e,rows in sorted(eps.items()):
        if not rows:errors.append([e,"empty"]);continue
        frames=[];times=[];dones=[];successes=[]
        for r in rows:
            missing=REQ-set(r)
            if missing:errors.append([e,"missing_keys",sorted(missing)]);continue
            try:s=vector(r["observation.state"],2);a=vector(r["action"],2)
            except Exception as ex:errors.append([e,"vector_contract",str(ex)]);continue
            if not np.isfinite(s).all() or not np.isfinite(a).all():errors.append([e,"nonfinite"])
            frames.append(int(scalar(r["frame_index"])));tv=float(scalar(r["timestamp"]));times.append(tv)
            if not np.isfinite(tv):errors.append([e,"nonfinite_timestamp"])
            try:dv=binary_flag(r["next.done"]);sv=binary_flag(r["next.success"])
            except Exception as ex:errors.append([e,"flag",str(ex)]);dv=False;sv=False
            dones.append(dv);successes.append(sv)
            gid=int(scalar(r["index"]));errors += [[e,"duplicate_global",gid]] if gid in global_ids else [];global_ids.add(gid)
            if int(scalar(r["episode_index"]))!=e:errors.append([e,"episode_mismatch"])
            task=int(scalar(r["task_index"]))
            if task<0 or (total_tasks is not None and task>=total_tasks):errors.append([e,"task_index",task])
        if frames!=list(range(len(rows))):errors.append([e,"frame_sequence",frames[:5]])
        if not dones or not dones[-1] or any(dones[i] and not dones[i+1] for i in range(len(dones)-1)):errors.append([e,"done_not_terminal_suffix",dones[-5:]])
        success_nonterminal+=sum(int(s and not d) for s,d in zip(successes,dones));success_terminal+=sum(int(s and d) for s,d in zip(successes,dones))
        dt=np.diff(times)
        if len(dt):
            bad=(~np.isfinite(dt))|(dt<=0)|(np.abs(dt-1/fps)>0.02)
            if np.any(bad):
                detail=[[int(i),None if not np.isfinite(dt[i]) else float(dt[i])] for i in np.flatnonzero(bad)[:10]]
                errors.append([e,"timestamp_intervals",detail])
    return {"episodes":len(eps),"frames":sum(map(len,eps.values())),"success_done_relation":{"success_nonterminal":success_nonterminal,"success_terminal":success_terminal,"asserted_invariant":"reported_only; upstream metadata does not promise success=>done"},"errors":errors,"clean":not errors}
def split_ids(eps):
    ranked=sorted(eps,key=lambda e:(float(vector(eps[e][0]["observation.state"],2)[0]),e));cut=max(1,math.ceil(.2*len(ranked)));test=set(ranked[-cut:]);train=[];dev=[]
    for e in ranked[:-cut]:
        bucket=int(hashlib.sha256(f"week05:{e}".encode()).hexdigest()[:8],16)%10;(dev if bucket==0 else train).append(e)
    if not train or not dev:raise RuntimeError("split too small")
    return {"train":sorted(train),"dev":sorted(dev),"test":sorted(test)}
def write_ids(s,out):
    p=Path(out);p.mkdir(parents=True,exist_ok=True)
    for k,v in s.items():(p/f"{k}_ids.txt").write_text("".join(f"{x}\n" for x in v))
def read_ids(root,name):return [int(x) for x in (Path(root)/f"{name}_ids.txt").read_text().split()]
def train_stats(eps,ids):
    if not ids:raise ValueError("train IDs empty")
    c=[];a=[]
    for e in ids:
        for r in eps[e]:c.append(vector(r["observation.state"],2));a.append(vector(r["action"],2))
    c=np.stack(c);a=np.stack(a)
    return {"state_mean":c.mean(0),"state_std":c.std(0)+1e-6,"state_min":c.min(0),"state_max":c.max(0),"state_p01":np.quantile(c,.01,axis=0),"state_p99":np.quantile(c,.99,axis=0),
            "action_mean":a.mean(0),"action_std":a.std(0)+1e-6,"action_min":a.min(0),"action_max":a.max(0),"action_p01":np.quantile(a,.01,axis=0),"action_p99":np.quantile(a,.99,axis=0)}
def build_windows(eps,ids,H,offset,st):
    obs=[];acts=[];masks=[];eids=[];frames=[];source_ids=[]
    for e in ids:
        rows=eps[e]
        for k in range(len(rows)):
            start=k+offset
            if start<0 or start>=len(rows):continue
            n=min(H,len(rows)-start);x=np.zeros((H,2),np.float32);m=np.zeros(H,np.bool_)
            for j in range(n):x[j]=(vector(rows[start+j]["action"],2)-st["action_mean"])/st["action_std"]
            m[:n]=True;obs.append((vector(rows[k]["observation.state"],2)-st["state_mean"])/st["state_std"]);acts.append(x);masks.append(m);eids.append(e);frames.append(k);source_ids.append(int(scalar(rows[k]["index"])))
    out={"state":np.stack(obs).astype(np.float32),"action":np.stack(acts).astype(np.float32),"mask":np.stack(masks).astype(np.bool_),"episode":np.asarray(eids,dtype=np.int64),"frame":np.asarray(frames,dtype=np.int64),"source_index":np.asarray(source_ids,dtype=np.int64)}
    verdict=validate_windows(out,H)
    if not verdict["clean"]:raise ValueError(verdict["errors"])
    return out
def validate_windows(w,H):
    errors=[];required={"state","action","mask","episode","frame","source_index"}
    if set(w)!=required:errors.append(["keys",sorted(set(w)),sorted(required)])
    if not required.issubset(w):return {"clean":False,"errors":errors}
    n=len(w["state"])
    if n==0:errors.append(["empty"])
    if w["state"].shape!=(n,2) or w["state"].dtype!=np.float32:errors.append(["state_contract",str(w["state"].shape),str(w["state"].dtype)])
    if w["action"].shape!=(n,H,2) or w["action"].dtype!=np.float32:errors.append(["action_contract",str(w["action"].shape),str(w["action"].dtype)])
    if w["state"].shape==(n,2) and not np.isfinite(w["state"]).all():errors.append(["state_nonfinite"])
    if w["action"].shape==(n,H,2) and not np.isfinite(w["action"]).all():errors.append(["action_nonfinite"])
    if w["mask"].shape!=(n,H) or w["mask"].dtype!=np.bool_:errors.append(["mask_contract",str(w["mask"].shape),str(w["mask"].dtype)])
    if w["episode"].shape!=(n,) or not np.issubdtype(w["episode"].dtype,np.integer):errors.append(["episode_contract"])
    if w["frame"].shape!=(n,) or not np.issubdtype(w["frame"].dtype,np.integer):errors.append(["frame_contract"])
    if w["source_index"].shape!=(n,) or not np.issubdtype(w["source_index"].dtype,np.integer):errors.append(["source_index_contract"])
    if w["mask"].shape==(n,H):
        if n and not w["mask"][:,0].all():errors.append(["mask_first_step_false"])
        if np.any(np.diff(w["mask"].astype(np.int8),axis=1)>0):errors.append(["mask_false_then_true"])
        if w["action"].shape==(n,H,2) and np.any(w["action"][~w["mask"]]!=0):errors.append(["nonzero_padding"])
    pairs=list(zip(w["episode"].tolist(),w["frame"].tolist()))
    if len(pairs)!=len(set(pairs)):errors.append(["duplicate_episode_frame"])
    return {"clean":not errors,"errors":errors,"windows":n}
def lag_scores(eps,ids,lo=-3,hi=3):
    out={}
    for lag in range(lo,hi+1):
        err=[]
        for e in ids:
            rows=eps[e]
            for k,r in enumerate(rows):
                j=k+lag
                if 0<=j<len(rows):err.append(((vector(r["action"],2)-vector(rows[j]["observation.state"],2))**2).mean())
        out[str(lag)]={"mse":float(np.mean(err)),"pairs":len(err)}
    return out
class StatefulSampler:
    def __init__(self,n,seed):self.n=n;self.g=np.random.default_rng(seed);self.order=self.g.permutation(n);self.pos=0
    def take(self,k):
        if k>self.n:raise ValueError("batch>dataset")
        if self.pos+k>self.n:self.order=self.g.permutation(self.n);self.pos=0
        x=self.order[self.pos:self.pos+k].copy();self.pos+=k;return x
    def state(self):return {"bitgen":self.g.bit_generator.state,"order":self.order.tolist(),"pos":self.pos}
    def load(self,s):self.g.bit_generator.state=s["bitgen"];self.order=np.asarray(s["order"]);self.pos=s["pos"]
def file_manifest(root):
    base=Path(root)
    if not base.is_dir():raise FileNotFoundError(base)
    out=[]
    for p in sorted(x for x in base.rglob("*") if x.is_file()):
        h=hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
        out.append({"path":p.relative_to(base).as_posix(),"bytes":p.stat().st_size,"sha256":h.hexdigest()})
    if not out:raise ValueError(f"manifest root is empty: {base}")
    return out
def bound_manifest(paths):
    out=[]
    for name in paths:
        p=Path(name);h=hashlib.sha256(p.read_bytes()).hexdigest();out.append({"path":p.as_posix(),"bytes":p.stat().st_size,"sha256":h})
    return out
def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="cmd",required=True)
    for n in ["validate","split","build","lag"]:sub.add_parser(n).add_argument("--root",required=True)
    sub.choices["validate"].add_argument("--out",required=True)
    sub.choices["split"].add_argument("--out-dir",required=True)
    b=sub.choices["build"];b.add_argument("--ids",required=True);b.add_argument("--out-dir",required=True);b.add_argument("--stats-out",required=True);b.add_argument("--horizon",type=int,default=16);b.add_argument("--action-offset",type=int,default=0)
    l=sub.choices["lag"];l.add_argument("--ids",required=True);l.add_argument("--out",required=True)
    m=sub.add_parser("manifest");m.add_argument("--root",required=True);m.add_argument("--out",required=True);m.add_argument("--bind",nargs="*",default=[])
    a=p.parse_args()
    if a.cmd=="manifest":obj={"root":a.root,"files":file_manifest(a.root),"bound_files":bound_manifest(a.bind)}
    else:
        info=load_info(a.root);eps=load_episodes(a.root)
        source_verdict=validate_episodes(eps,fps=int(info["fps"]),total_tasks=int(info["total_tasks"]))
        if a.cmd!="validate" and not source_verdict["clean"]:raise ValueError(f"source validator failed: {source_verdict['errors'][:5]}")
    if a.cmd=="validate":
        obj=source_verdict;obj.update(codebase_version=info["codebase_version"],metadata_total_episodes=int(info["total_episodes"]),metadata_total_frames=int(info["total_frames"]))
        Path(a.out).write_text(json.dumps(obj,indent=2)+"\n")
        if not obj["clean"]:raise ValueError(f"source validator failed: {obj['errors'][:5]}")
    elif a.cmd=="split":obj=split_ids(eps);write_ids(obj,a.out_dir)
    elif a.cmd=="lag":obj=lag_scores(eps,read_ids(a.ids,"train")+read_ids(a.ids,"dev"));Path(a.out).write_text(json.dumps(obj,indent=2)+"\n")
    elif a.cmd=="build":
        ids={k:read_ids(a.ids,k) for k in ["train","dev","test"]};st=train_stats(eps,ids["train"]);Path(a.stats_out).write_text(json.dumps({k:v.tolist() for k,v in st.items()},indent=2)+"\n")
        Path(a.out_dir).mkdir(parents=True,exist_ok=True)
        for k in ids:np.savez_compressed(Path(a.out_dir)/f"{k}.npz",**build_windows(eps,ids[k],a.horizon,a.action_offset,st))
        obj={"horizon":a.horizon,"action_offset":a.action_offset,"counts":{k:len(ids[k]) for k in ids}}
    elif a.cmd=="manifest":Path(a.out).write_text(json.dumps(obj,indent=2)+"\n")
    print(json.dumps(obj))
if __name__=="__main__":main()
```

```python
# tests/test_contract.py
import copy,json
from pathlib import Path
import numpy as np,pytest
import pyarrow as pa,pyarrow.parquet as pq
from src.contract import FEATURES,check_parquet_schema,load_episodes,load_info,validate_episodes,validate_windows,vector,split_ids,train_stats,build_windows,lag_scores,StatefulSampler
def eps():
    out={}
    for e in range(20):
        out[e]=[]
        for k in range(4):out[e].append({"episode_index":e,"frame_index":k,"timestamp":.1*k,"observation.state":np.array([k,e],np.float32),"action":np.array([k+1,e],np.float32),"next.done":k==3,"next.success":False,"index":100*e+k,"task_index":0})
    return out
def codes(verdict):return {x[1] for x in verdict["errors"]}
def info(total_frames=1,total_episodes=1,version="v3.0"):
    return {"codebase_version":version,"fps":10,"total_frames":total_frames,"total_episodes":total_episodes,"total_tasks":1,
            "features":{k:{"dtype":dtype,"shape":shape} for k,(dtype,shape) in FEATURES.items()}}
def write_info(root,obj):
    p=Path(root)/"meta";p.mkdir(parents=True,exist_ok=True);(p/"info.json").write_text(json.dumps(obj),encoding="utf-8")
def empty_table(include_action=True,action_type=None):
    if action_type is None:
        action_type=pa.float32()
    types={"observation.state":pa.list_(pa.float32(),list_size=2),"action":pa.list_(action_type,list_size=2),"episode_index":pa.int64(),"frame_index":pa.int64(),"timestamp":pa.float32(),"next.done":pa.bool_(),"next.success":pa.bool_(),"index":pa.int64(),"task_index":pa.int64()}
    if not include_action:types.pop("action")
    return pa.table({k:pa.array([],type=v) for k,v in types.items()})
def test_loader_rejects_missing_empty_bad_metadata_and_physical_schema(tmp_path):
    root=tmp_path/"missing";write_info(root,info())
    with pytest.raises(FileNotFoundError,match="no data"):load_episodes(root)
    data=root/"data"/"chunk-000";data.mkdir(parents=True);empty= data/"file-000.parquet";pq.write_table(empty_table(),empty)
    with pytest.raises(ValueError,match="empty parquet"):load_episodes(root)
    bad_meta=tmp_path/"bad_meta";write_info(bad_meta,info(version="v2.0"))
    with pytest.raises(ValueError,match="expected v3.0"):load_info(bad_meta)
    zero=tmp_path/"zero";write_info(zero,info(total_frames=0,total_episodes=0))
    with pytest.raises(ValueError,match="zero frames/episodes"):load_episodes(zero)
    missing=tmp_path/"missing_col.parquet";pq.write_table(empty_table(include_action=False),missing)
    with pytest.raises(ValueError,match="missing"):check_parquet_schema(missing)
    bad_dtype=tmp_path/"bad_dtype.parquet";pq.write_table(empty_table(action_type=pa.float64()),bad_dtype)
    with pytest.raises(ValueError,match="physical dtype"):check_parquet_schema(bad_dtype)
def test_clean_and_corruptions():
    x=eps();assert validate_episodes(x)["clean"]
    assert "no_episodes" in codes(validate_episodes({}))
    y=copy.deepcopy(x);y[0][1]["timestamp"]=0;assert "timestamp_intervals" in codes(validate_episodes(y))
    y=copy.deepcopy(x);y[0][1]["action"]=np.array([np.nan,0],np.float32);assert "nonfinite" in codes(validate_episodes(y))
    y=copy.deepcopy(x);y[0][1]["frame_index"]=7;assert "frame_sequence" in codes(validate_episodes(y))
    y=copy.deepcopy(x);y[0][-1]["next.done"]=False;assert "done_not_terminal_suffix" in codes(validate_episodes(y))
    y=copy.deepcopy(x);y[1][0]["index"]=y[0][0]["index"];assert "duplicate_global" in codes(validate_episodes(y))
    y=copy.deepcopy(x);y[0][0]["action"]=np.zeros(3,np.float32);assert "vector_contract" in codes(validate_episodes(y))
    y=copy.deepcopy(x);y[0][0]["task_index"]=-1;assert "task_index" in codes(validate_episodes(y,total_tasks=1))
    y=copy.deepcopy(x);del y[0][0]["action"];assert "missing_keys" in codes(validate_episodes(y))
    with pytest.raises(ValueError):vector(np.zeros(2,np.float64),2)
def test_split_window_and_stats():
    x=eps();s=split_ids(x);assert not(set(s["train"])&set(s["test"]));st=train_stats(x,s["train"]);w=build_windows(x,[s["train"][0]],4,0,st)
    assert w["action"].shape[1:]==(4,2) and w["mask"][-1].tolist()==[True,False,False,False] and w["source_index"].shape==w["frame"].shape and validate_windows(w,4)["clean"]
    bad={k:v.copy() for k,v in w.items()};bad["mask"]=bad["mask"].astype(np.int8);assert not validate_windows(bad,4)["clean"]
    bad={k:v.copy() for k,v in w.items()};bad["action"][~bad["mask"]]=1;assert not validate_windows(bad,4)["clean"]
    bad={k:v.copy() for k,v in w.items()};bad["action"][0,0,0]=np.nan;assert "action_nonfinite" in {e[0] for e in validate_windows(bad,4)["errors"]}
    corrupt=copy.deepcopy(x);corrupt[s["train"][0]][0]["action"][0]=np.nan
    with pytest.raises(ValueError):build_windows(corrupt,[s["train"][0]],4,0,st)
    before=train_stats(x,s["train"]);y=copy.deepcopy(x)
    for e in s["test"]:
        for row in y[e]:row["observation.state"]+=10000;row["action"]+=10000
    after=train_stats(y,s["train"])
    for k in before:np.testing.assert_array_equal(before[k],after[k])
def test_known_lag():
    x=eps();score=lag_scores(x,[0],0,2);assert score["1"]["mse"]<score["0"]["mse"]
def test_sampler_resume():
    a=StatefulSampler(20,3);a.take(7);state=a.state();expected=a.take(5);b=StatefulSampler(20,999);b.load(state);assert np.array_equal(expected,b.take(5))
```

### 实现

validator 纯函数便于故障注入；主 CLI 只按固定 parquet 读，未依赖 LeRobot 包。加载器先把 `meta/info.json` 的 v3.0、fps、总 episode/frame 和每个 feature 的 dtype/shape 与 Parquet 物理 leaf dtype 对齐；零文件、零行、零 episode 都直接失败。固定 PushT snapshot 可出现连续多个 `next.done=true`，所以硬不变量是“done 为非空 terminal suffix，出现 true 后不得回到 false”，而不是“仅最后一帧为 true”。`next.success` 与 done 的联合计数写入报告，但由于上游 metadata 未承诺 `success => done`，不擅自把它升级为硬不变量。范围阈值没有臆造，先输出分位数到 Data Card 后再预注册。

### 执行命令

`[两者/Python]`：

```bash
python -m py_compile src/fetch_public.py src/contract.py
python -m pytest -q tests/test_contract.py
python -m src.contract validate --root data/raw/pusht --out reports/validation.json
```

### 预期观测（估算/示例，不是实测）

corruption tests 全过：缺文件、零行、坏metadata、缺列/错误Arrow dtype、空数据、时间倒退、NaN、错 frame、错 done、重复 global ID、错 shape/source dtype、非法 task、bad mask、非零 padding、processed nonfinite 和 stats 泄漏均能被具体错误码捕获。真实报告 episodes/frames/format 应与 metadata 精确相等。任何 error 都列类型/episode，不静默 drop。

### 验收条件

clean fixture 100% 通过、全部注入被抓；真实 schema、Arrow/source dtype、shape、finite、frame、timestamp、global index、task index、非空 totals 通过或有明确处理审批。

### 若失败，按什么顺序查

文件排序→列名→scalar/list编码→episode排序→frame→timestamp→finite→重复global ID。源异常先记录，不修改原始parquet。

### 当日证据清单

pytest、`validation.json`（含 `success_done_relation`）、异常清单与处理决定、30 条人工字段说明。

## Day 3：action语义、lag oracle与真实lag扫描

### 为什么做

确定observation k与action k的时间含义；统计proxy只能在坐标/单位已确认后解释。

### 输入与前置检查

Task Card已通过固定上游代码确认：state为agent position、action为absolute target position或明确不是。若未确认，先标`CONTRACT_UNRESOLVED`，lag数字不做物理解读。

### 本日要创建/修改的文件

无需新增核心代码；将正lag定义写入 `reports/task_card.md`：`action[k]` 对齐 `state[k+lag]`。`tests/test_known_lag`是已知lag=1 oracle。

### 实现

真实扫描使用episode内pair，`-3..3`每项报告MSE与pair数。当前proxy假定action/state同坐标；否则应换成动作与state delta的相关，而非强解释。

### 执行命令

先创建显式split，再lag：

`[两者/Python]`：

```bash
python -m src.contract split --root data/raw/pusht --out-dir manifests
python -m src.contract lag --root data/raw/pusht --ids manifests --out reports/lag.json
```

可选视频已下载时，`[公司 Linux]` 只读探针示例（先用实际存在路径替换，不可直接运行的模板）：

```text
ffprobe -v error -show_entries stream=width,height,r_frame_rate -of json <APPROVED_LOCAL_MP4>
```

该块明确是需替换路径的模板，不被后续命令依赖。

### 预期观测（估算/示例，不是实测）

known-lag测试选1；真实曲线可能在0/1附近最低，也可能平坦，均需实测。不得预填最佳lag。

### 验收条件

known-lag误差≤1 step；真实每lag pair数/score完整；最终action offset明确写0/其他或`INCONCLUSIVE`；30段视频/轨迹交叉检查或标未验证。

### 若失败，按什么顺序查

lag正负定义→episode边界→action/state单位→known oracle→真实曲线→扩大区间（需先预注册）。

### 当日证据清单

`lag.json`、oracle结果、Task Card语义来源、人工对齐表。

## Day 4：episode split、train-only stats、windows与完整manifest

### 为什么做

固化内部train/dev/OOD IDs，杜绝frame泄漏，并让每个window可追溯。

### 输入与前置检查

validator通过；split规则未看success outcome；action offset已决定。本示例用0，若lag审计要求其他值，先在配置记录再改命令。

### 本日要创建/修改的文件

创建完整配置：

```json
{
  "dataset":"lerobot/pusht",
  "revision":"7628202a2180972f291ba1bc6723834921e72c19",
  "format":"v3.0",
  "horizon":16,
  "action_offset":0,
  "split":"episode; top-20%-initial-x OOD; stable-hash train/dev",
  "normalization":"train-only mean/std + 1e-6",
  "camera_order":["observation.image"]
}
```

将纯JSON保存为 `configs/data_contract.json`。

### 实现

split 命令已写明确 IDs；build 仅 fit train stats，并对三个 split 应用相同 stats。构建后再次检查 processed NPZ 的 dtype/shape/finite、mask 单调性、padding 零值、`(episode, frame)` 唯一性，并保存每个window observation row的 `source_index`。test 是单轴内部 OOD，不是官方 test。

### 执行命令

`[两者/Python]`：

```bash
python -m src.contract split --root data/raw/pusht --out-dir manifests
python -m src.contract build --root data/raw/pusht --ids manifests --horizon 16 --action-offset 0 --out-dir data/processed --stats-out manifests/train_stats.json
python -m src.contract manifest --root data/raw/pusht --out manifests/raw_files.json --bind manifests/source.json configs/data_contract.json src/contract.py
python -m src.contract manifest --root data/processed --out manifests/processed_files.json --bind manifests/source.json manifests/raw_files.json manifests/train_ids.txt manifests/dev_ids.txt manifests/test_ids.txt manifests/train_stats.json configs/data_contract.json src/contract.py
```

检查overlap：

```bash
python -c "from pathlib import Path; s={k:set(Path(f'manifests/{k}_ids.txt').read_text().split()) for k in ['train','dev','test']}; print({a+'_'+b:len(s[a]&s[b]) for a,b in [('train','dev'),('train','test'),('dev','test')]}); assert not any(s[a]&s[b] for a,b in [('train','dev'),('train','test'),('dev','test')])"
```

### 预期观测（估算/示例，不是实测）

三个overlap为0；processed三个NPZ非空；尾window mask正确；stats JSON只列train provenance。具体均值/范围不预填。

### 验收条件

episode/global row 不跨 split；OOD 只有 initial-x 轴；normalization 只 fit train，corruption test 证明改变 test 数值不会改变 train stats；processed manifest 直接绑定 `source.json`、`raw_files.json`、splits、config、stats与code hash。

### 若失败，按什么顺序查

split IDs→initial-x排序→stable hash→build source IDs→stats provenance→window offset/尾部mask→processed hash。

### 当日证据清单

三份IDs、两个file manifest、stats、data contract、随机5个window的 `episode/frame/source_index` 追溯链。

## Day 5：sampler/RNG、50-step同seed与resume

### 为什么做

证明“可复现”具体到sample IDs、loss与恢复后的下一batch，不用“GPU本来随机”作为结论。

### 输入与前置检查

Day4 `data/processed/train.npz`存在；先CPU/`num_workers=0` oracle。

### 本日要创建/修改的文件

```python
# src/repro_probe.py
import argparse,hashlib,json,os,random
from pathlib import Path
import numpy as np,torch
from torch import nn
from src.contract import StatefulSampler
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s)
def loadck(p):
    try:return torch.load(p,map_location="cpu",weights_only=False)
    except TypeError:return torch.load(p,map_location="cpu")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument("--data",default="data/processed/train.npz");p.add_argument("--seed",type=int,default=77);p.add_argument("--batch",type=int,default=16)
    p.add_argument("--total-steps",type=int,default=50);p.add_argument("--stop-after",type=int);p.add_argument("--output",required=True);p.add_argument("--resume");a=p.parse_args()
    out=Path(a.output)
    if out.exists() and any(out.iterdir()) and not a.resume:raise RuntimeError("non-empty output")
    seed(a.seed);d=dict(np.load(a.data));H=d["action"].shape[1]
    contract={"data_sha256":sha(a.data),"code_sha256":sha(__file__),"contract_code_sha256":sha(Path(__file__).with_name("contract.py")),"seed":a.seed,"batch":a.batch,
              "total_steps":a.total_steps,"horizon":H,"rows":len(d["state"])}
    m=nn.Sequential(nn.Linear(2,64),nn.Tanh(),nn.Linear(64,H*2));opt=torch.optim.AdamW(m.parameters(),lr=1e-3);sam=StatefulSampler(len(d["state"]),a.seed+1);step=0
    if a.resume:
        if Path(a.resume).resolve().parent!=out.resolve():raise ValueError("resume checkpoint must be in the same output directory")
        c=loadck(a.resume)
        if c["contract"]!=contract:raise ValueError("resume data/code/seed/batch/total-step contract changed")
        m.load_state_dict(c["model"]);opt.load_state_dict(c["opt"]);sam.load(c["sampler"]);step=c["step"];random.setstate(c["py"]);np.random.set_state(c["np"]);torch.set_rng_state(c["torch"])
    if a.total_steps<=0 or (a.stop_after is not None and not 0<=a.stop_after<=a.total_steps):raise ValueError("invalid total/stop steps")
    out.mkdir(parents=True,exist_ok=True);(out/"run_manifest.json").write_text(json.dumps(contract,indent=2)+"\n",encoding="utf-8")
    target=min(a.total_steps,a.stop_after if a.stop_after is not None else a.total_steps)
    with (out/"metrics.jsonl").open("a") as f:
        while step<target:
            idx=sam.take(a.batch);x=torch.from_numpy(d["state"][idx]).float();y=torch.from_numpy(d["action"][idx]).float();mask=torch.from_numpy(d["mask"][idx]).float()
            pred=m(x).view(-1,H,2);loss=((pred-y).square()*mask[:,:,None]).sum()/(mask.sum()*2).clamp_min(1);opt.zero_grad();loss.backward();opt.step();step+=1
            rec={"step":step,"ids":idx.tolist(),"loss":float(loss)};f.write(json.dumps(rec)+"\n");print(json.dumps(rec))
    ck={"model":m.state_dict(),"opt":opt.state_dict(),"sampler":sam.state(),"step":step,"contract":contract,
        "py":random.getstate(),"np":np.random.get_state(),"torch":torch.get_rng_state()}
    tmp=out/"last.pt.tmp";torch.save(ck,tmp);os.replace(tmp,out/"last.pt")
if __name__=="__main__":main()
```

### 实现

这是 CPU deterministic oracle：sampler 保存 bit-generator、permutation 和 cursor；`run_manifest.json`与checkpoint共同绑定 processed data、probe/contract code SHA-256、seed、batch、horizon、rows 与 `total_steps`。`--stop-after`只决定本段停止点，不改变 `total_steps=50` 合同；resume只接受同一输出目录的原子checkpoint。

### 执行命令

`[两者/Python]`：

```bash
python -m py_compile src/repro_probe.py
python -m src.repro_probe --total-steps 50 --output runs/continuous50
python -m src.repro_probe --total-steps 50 --stop-after 25 --output runs/resumed50
python -m src.repro_probe --total-steps 50 --output runs/resumed50 --resume runs/resumed50/last.pt
python -c "import json; A=[json.loads(x) for x in open('runs/continuous50/metrics.jsonl')]; B=[json.loads(x) for x in open('runs/resumed50/metrics.jsonl')]; assert [x['ids'] for x in A]==[x['ids'] for x in B]; import math; assert all(math.isclose(x['loss'],y['loss'],rel_tol=0,abs_tol=1e-10) for x,y in zip(A,B)); print('REPRO_PASS',len(A))"
```

不同seed对照使用新目录：

```bash
python -m src.repro_probe --seed 78 --total-steps 50 --output runs/seed78
```

`[公司 Linux]` 后续GPU probe可复用Week04 Trainer，但必须记录CUDA非确定算子/容差；公司结果不外传。`[个人 PowerShell]` 同上命令直接运行CPU。

### 预期观测（估算/示例，不是实测）

continuous/resumed 50批IDs完全相同，CPU loss在严容差内相同；seed78 IDs应不同。若不同，先定位首次分叉step。

### 验收条件

same-seed IDs一致、前20 loss一致、resume step26 IDs/loss一致、different-seed确有差异；Data Card十项完整。

### 若失败，按什么顺序查

源NPZ hash→samplerstate保存时点→permutation/cursor→model/optimizer→Python/NumPy/torch RNG→首次分叉ID→首次分叉loss。多worker只在oracle过后引入。

### 当日证据清单

三个run JSONL及各自 `run_manifest.json`、resume checkpoint、本机比较结果、最终Task/Data Card、`decision.md`。

## 最终发布门

只有source/license/revision/hash、语义合同、validator/corruption、lag oracle、episode split、train-only stats、window/mask、manifest、same-seed与resume全部有证据才写PASS。未下载视频则明确图像对齐未验证；无公开snapshot则整体`INCONCLUSIVE`。本文未声称任何命令已执行。

技术链接核验日期：**2026-09-03**：[PushT dataset card](https://huggingface.co/datasets/lerobot/pusht)、[LeRobotDataset v3.0（LeRobot v0.4.3）](https://huggingface.co/docs/lerobot/v0.4.3/en/lerobot-dataset-v3)、[action representations](https://huggingface.co/docs/lerobot/main/en/action_representations)、[Diffusion Policy](https://github.com/real-stanford/diffusion_policy)、[PyTorch 2.1 reproducibility](https://pytorch.org/docs/2.1/notes/randomness.html)。
