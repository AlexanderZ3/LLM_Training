# Week 14 实践篇：从空输出目录交付 Mini-WAM Capstone

> 本手册选定推荐载荷：Week 11 Mini-WAM `A_action_only vs B_action_world`。周一冻结后不换题。  
> 全部命令均待执行；示例/阈值不是实测。公司数据、代码运行产物、checkpoint、日志、trace、指标和拓扑信息不得导出。

## 0. 下载前与跨周恢复

公司 V100/PyTorch2.1 为正式环境（用户自述待核验）；5070 Ti 只从公开/Toy 数据重跑；H100 只能形成独立 transfer。估算空间：Toy 数据<0.2GB、1/4/8卡 run 与 checkpoint 10GB、trace 5–20GB，最低预留40GB。总训练限定步数；先20-step推算GPU小时。

```bash
which python; python --version; git --version; df -h .
python - <<'PY'
import shutil,torch
print({"torch":torch.__version__,"cuda":torch.version.cuda,"available":torch.cuda.is_available(),
       "gpus":torch.cuda.device_count(),"disk":shutil.disk_usage(".")})
if torch.cuda.is_available():
 for i in range(torch.cuda.device_count()):
  p=torch.cuda.get_device_properties(i);print(i,p.name,p.major,p.minor,p.total_memory)
PY
nvidia-smi
nvidia-smi topo -m
command -v nsys || true
```

不创建 Conda/venv，不升级 torch/CUDA，不安装新框架。保存环境：

```bash
mkdir -p week14-capstone/{configs,workload/configs,data,manifests,runs,reports,tests}
cd week14-capstone
python -m pip freeze > manifests/pip-freeze.before.txt
python -m pip check | tee manifests/pip-check.before.txt
```

从同一批准工作区复制 Week11 已定义的核心；若目录缺失，严格按 Week11 Lab §2 重建这些文件并重新生成 ToyPush，不能从个人聊天/网盘补：

```bash
test -f ../week11-miniwam/run.py || { echo '先按 Week11 Lab 重建'; exit 2; }
test -f ../week11-miniwam/make_toy_data.py
test -f ../week11-miniwam/data/toy_push_v1.npz
test -f ../week11-miniwam/data/toy_push_v1.meta.json
cp ../week11-miniwam/{run.py,make_toy_data.py} workload/
cp ../week11-miniwam/configs/{A_action_only.json,B_action_world.json} workload/configs/
cp ../week11-miniwam/data/toy_push_v1.npz data/
cp ../week11-miniwam/data/toy_push_v1.meta.json data/
python workload/run.py validate --config workload/configs/B_action_world.json --data data/toy_push_v1.npz
```

Toy 数据来自本地确定性代码；未在 fixed `lerobot/pusht@7628202a2180972f291ba1bc6723834921e72c19` 或其他获批真实载荷复测时，模型结论必须限制为 pipeline 证据。所有公司产物继续留公司机器。

## 1. 最终目录与每个文件职责

不要用 `touch` 预造空文件。下面每个小节都要求把紧随其后的完整代码块保存到指定路径；所有块保存后，再按 Day 1 顺序执行静态检查、A/B 合同生成与冻结。

```text
week14-capstone/
  README_OFFLINE.md           # 新 shell 唯一操作说明
  run_experiment.py           # validate/train/resume/eval/profile 顶层入口
  freeze_inputs.py            # 计算 code/data/config SHA，周一冻结
  register_artifact.py        # checkpoint/file/dir 完整性 marker
  verify_ab.py                # 初始权重/预算/样本流/eval协议合同
  verify_publication.py       # 最终 run/eval/checkpoint 一致性门
  configs/reference.json      # A_action_only
  configs/treatment.json      # B_action_world
  configs/pre_registration.json
  workload/run.py             # Week11 frozen core
  workload/evidence_run.py    # 观测完整 train/eval ID 流的唯一入口
  workload/configs/*.json
  data/toy_push_v1.npz
  manifests/frozen_inputs.json
  manifests/ab_contract.json
  tests/test_runner.py
  runs/ reports/
```

### 1.1 完整 capstone configs

将紧随其后的代码块原样保存为 `configs/reference.json`：

```json
{"name":"reference_action_only","kind":"miniwam","seed":1107,"precision":"fp16","max_steps":300,
 "formal_world_size":1,"global_batch":64,"formal_successful_updates":300,
 "budget_unit":"samples","target_tokens_seen":0,
 "profile_steps":50,"data":"data/toy_push_v1.npz","workload_code":"workload/evidence_run.py",
 "workload_config":"workload/configs/A_action_only.json","eval_split":"dev","eval_batch":64}
```

将紧随其后的代码块原样保存为 `configs/treatment.json`：

```json
{"name":"treatment_action_world","kind":"miniwam","seed":1107,"precision":"fp16","max_steps":300,
 "formal_world_size":1,"global_batch":64,"formal_successful_updates":300,
 "budget_unit":"samples","target_tokens_seen":0,
 "profile_steps":50,"data":"data/toy_push_v1.npz","workload_code":"workload/evidence_run.py",
 "workload_config":"workload/configs/B_action_world.json","eval_split":"dev","eval_batch":64}
```

将紧随其后的代码块原样保存为 `configs/pre_registration.json`：

```json
{"question":"Does correct action-conditioned future supervision help action prediction?",
 "only_change":"lambda_world: 0.0 -> 0.2; all shared action/data/eval settings fixed",
 "primary_metric":"dev action MSE","mechanism_metric":"future relative_improvement vs last-frame",
 "numeric_tolerance":{"fp32_repeat_relative":0.001,"fp16_ddp_first20_relative":0.02,"resume_next_loss_relative":0.01},
 "stability":{"nonfinite":0,"skipped_update_fraction_lt":0.01,"reserved_gb_stop":30.5},
 "scaling":"weak scaling for performance; fixed global batch for correctness",
 "budgets":{"max_steps_per_formal_run":300,"max_trace_minutes":3,"max_disk_gb":40},
 "labels":["PASS","FAIL-MODEL","FAIL-SYSTEM","INCONCLUSIVE"],
 "frozen_before_results":true,"test_used_for_tuning":false}
```

### 1.2 `workload/evidence_run.py`：记录实际完整样本流

将紧随其后的代码块原样保存为 `workload/evidence_run.py`。它不重写 Week11 算法，只在 `indices` 与 `Windows.batch` 的实际调用点记录训练完整 batch IDs；eval 完成后把实际处理的 ordered IDs 数量与 SHA 写回 `eval.json`：

```python
# file: workload/evidence_run.py
import hashlib,json,os,sys
from pathlib import Path
import run as core

MODE=sys.argv[1] if len(sys.argv)>1 else ""
RANK=int(os.getenv("RANK","0"));WORLD=int(os.getenv("WORLD_SIZE","1"))
CURRENT={};EVAL_IDS=[]
def cli_value(flag):
 i=sys.argv.index(flag);return sys.argv[i+1]
def ids_sha(ids):return hashlib.sha256("\n".join(ids).encode()).hexdigest()
def atomic_json(path,doc):
 path=Path(path);tmp=path.with_name(path.name+".tmp")
 tmp.write_text(json.dumps(doc,indent=2),encoding="utf-8");os.replace(tmp,path)

original_indices=core.indices
def audited_indices(n,global_batch,seed,step,rank,world):
 ii=original_indices(n,global_batch,seed,step,rank,world);CURRENT["successful_step"]=step+1;return ii
core.indices=audited_indices
original_batch=core.Windows.batch
def audited_batch(self,ii):
 out=original_batch(self,ii)
 if MODE=="train" and "successful_step" in CURRENT:
  dst=Path(cli_value("--out"))/f"sample_ids.rank{RANK}.jsonl";dst.parent.mkdir(parents=True,exist_ok=True)
  rec={"successful_step":CURRENT.pop("successful_step"),"rank":RANK,"sample_ids":list(out["ids"])}
  with dst.open("a",encoding="utf-8") as f:f.write(json.dumps(rec,separators=(",",":"))+"\n")
 elif MODE=="eval":EVAL_IDS.extend(map(str,out["ids"]))
 return out
core.Windows.batch=audited_batch

core.main()
if MODE=="train" and RANK==0:
 out=Path(cli_value("--out"));rows=[]
 for rank in range(WORLD):
  p=out/f"sample_ids.rank{rank}.jsonl"
  if not p.is_file():raise RuntimeError("missing full sample-ID audit:"+str(p))
  rows.extend(json.loads(x) for x in p.read_text(encoding="utf-8").splitlines())
 rows.sort(key=lambda x:(int(x["successful_step"]),int(x["rank"])))
 steps=sorted({int(x["successful_step"]) for x in rows});ordered=[]
 for step in steps:
  group=[x for x in rows if int(x["successful_step"])==step]
  if [int(x["rank"]) for x in group]!=list(range(WORLD)):raise RuntimeError("sample audit rank coverage mismatch")
  for x in group:ordered.extend(map(str,x["sample_ids"]))
 atomic_json(out/"observed_sample_stream.json",{"status":"PASS","world_size":WORLD,"steps":steps,
  "successful_updates":len(steps),"successful_samples":len(ordered),"ordered_ids_sha256":ids_sha(ordered)})
elif MODE=="eval":
 dst=Path(cli_value("--out"));doc=json.loads(dst.read_text(encoding="utf-8"))
 if not EVAL_IDS:raise RuntimeError("eval observed no sample IDs")
 doc["eval_n"]=len(EVAL_IDS);doc["eval_ordered_ids_sha256"]=ids_sha(EVAL_IDS);atomic_json(dst,doc)
```

### 1.3 `verify_ab.py`：A/B 因果与预算合同

将紧随其后的代码块原样保存为 `verify_ab.py`：

```python
# file: verify_ab.py
import argparse,hashlib,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parent))
from workload.run import MiniWAM,Windows,indices,seed_all

def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state_sha(model):
  h=hashlib.sha256()
  for name,t in sorted(model.state_dict().items()):
    h.update(name.encode());h.update(b"\0");h.update(t.detach().cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
def stream(ds,cfg,steps):
  rows=[]
  for step in range(steps):
    ii=indices(len(ds),cfg["global_batch"],cfg["seed"],step,0,1)
    rows.extend(ds.batch(ii)["ids"])
  return hashlib.sha256("\n".join(rows).encode()).hexdigest(),len(rows)
def main():
  ap=argparse.ArgumentParser();ap.add_argument("--reference",required=True);ap.add_argument("--treatment",required=True)
  ap.add_argument("--out",required=True);a=ap.parse_args()
  rc=json.loads(Path(a.reference).read_text(encoding="utf-8"));tc=json.loads(Path(a.treatment).read_text(encoding="utf-8"))
  cap_allowed={"name","workload_config"}
  cap_diff={k for k in set(rc)|set(tc) if rc.get(k)!=tc.get(k)}
  if cap_diff!=cap_allowed:raise RuntimeError("capstone A/B diff must be exactly name+workload_config:"+repr(cap_diff))
  rw=json.loads(Path(rc["workload_config"]).read_text(encoding="utf-8"));tw=json.loads(Path(tc["workload_config"]).read_text(encoding="utf-8"))
  workload_allowed={"run","lambda_world"}
  workload_diff={k for k in set(rw)|set(tw) if rw.get(k)!=tw.get(k)}
  if workload_diff!=workload_allowed or rw["lambda_world"]!=0.0 or tw["lambda_world"]<=0:
    raise RuntimeError("workload A/B diff must be run + lambda_world 0->positive")
  for c,w in ((rc,rw),(tc,tw)):
    if c["seed"]!=w["seed"] or c["global_batch"]!=w["global_batch"]:raise RuntimeError("cap/workload budget mismatch")
    if c["formal_successful_updates"]!=c["max_steps"] or c["formal_world_size"]!=1:raise RuntimeError("formal budget contract mismatch")
    if c["budget_unit"]!="samples" or c["target_tokens_seen"]!=0:raise RuntimeError("Mini-WAM token budget must be explicit N/A=0")
  seed_all(rw["seed"]);ma=MiniWAM(rw["latent_dim"],rw["action_horizon"]);ha=state_sha(ma)
  seed_all(tw["seed"]);mb=MiniWAM(tw["latent_dim"],tw["action_horizon"]);hb=state_sha(mb)
  if ha!=hb:raise RuntimeError("A/B initial weights differ")
  da=Windows(rc["data"],rw["split"],rw["future_offset"],rw["action_horizon"])
  db=Windows(tc["data"],tw["split"],tw["future_offset"],tw["action_horizon"])
  sa,na=stream(da,rw,rc["formal_successful_updates"]);sb,nb=stream(db,tw,tc["formal_successful_updates"])
  if (sa,na)!=(sb,nb):raise RuntimeError("A/B successful-step sample streams differ")
  ea=Windows(rc["data"],rc["eval_split"],rw["future_offset"],rw["action_horizon"])
  eb=Windows(tc["data"],tc["eval_split"],tw["future_offset"],tw["action_horizon"])
  eval_a=[f"e{e}:t{t}" for e,t in ea.ids];eval_b=[f"e{e}:t{t}" for e,t in eb.ids]
  if eval_a!=eval_b:raise RuntimeError("A/B eval IDs/order differ")
  inputs={str(p):file_sha(p) for p in (a.reference,a.treatment,rc["workload_config"],tc["workload_config"],
    rc["workload_code"],rc["data"])}
  out={"status":"PASS","inputs_sha256":inputs,"only_change":{"lambda_world":[rw["lambda_world"],tw["lambda_world"]]},
    "initial_weight_sha256":ha,"formal_world_size":rc["formal_world_size"],"global_batch":rc["global_batch"],
    "successful_updates":rc["formal_successful_updates"],"successful_samples":na,
    "target_tokens_seen":0,"target_tokens_semantics":"not_applicable_for_Mini-WAM",
    "sample_stream_sha256":sa,"eval_ids_sha256":hashlib.sha256("\n".join(eval_a).encode()).hexdigest(),
    "eval_n":len(eval_a),"eval_split":rc["eval_split"],"eval_batch":rc["eval_batch"]}
  dst=Path(a.out);dst.parent.mkdir(parents=True,exist_ok=True)
  dst.write_text(json.dumps(out,indent=2,sort_keys=True),encoding="utf-8");print(json.dumps(out))
if __name__=="__main__":main()
```

### 1.4 `freeze_inputs.py`：完整冻结代码

将紧随其后的代码块原样保存为 `freeze_inputs.py`：

```python
# file: freeze_inputs.py
import argparse,hashlib,json,platform,subprocess
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--configs",nargs="+",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
 if {Path(x).name for x in a.configs}!={"reference.json","treatment.json"}:raise SystemExit("freeze exactly reference+treatment configs")
 paths={"runner":"run_experiment.py","freeze":"freeze_inputs.py","register":"register_artifact.py",
   "ab_verifier":"verify_ab.py","publication_verifier":"verify_publication.py","tests":"tests/test_runner.py",
   "readme":"README_OFFLINE.md","prereg":"configs/pre_registration.json","workload_core":"workload/run.py",
   "generator":"workload/make_toy_data.py",
  "data_meta":"data/toy_push_v1.meta.json","ab_contract":"manifests/ab_contract.json"}
 for cp in a.configs:
  c=json.loads(Path(cp).read_text(encoding="utf-8")); paths["cap_config:"+cp]=cp
  for k in ("data","workload_code","workload_config"):paths[f"{cp}:{k}"]=c[k]
 missing=[p for p in paths.values() if not Path(p).is_file()]
 if missing:raise SystemExit("missing inputs:"+repr(missing))
 out_path=Path(a.out).resolve()
 if out_path in {Path(p).resolve() for p in paths.values()}:raise SystemExit("freeze manifest cannot hash itself")
 ab=json.loads(Path("manifests/ab_contract.json").read_text(encoding="utf-8"))
 if ab.get("status")!="PASS":raise SystemExit("A/B contract is not PASS")
 for p,h in ab["inputs_sha256"].items():
  if not Path(p).is_file() or sha(p)!=h:raise SystemExit("A/B contract input drift:"+p)
 try:git=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
 except Exception:git="NOT_A_GIT_WORKTREE"
 out={"status":"FROZEN_NOT_RUN","files":{k:{"path":v,"sha256":sha(v),"bytes":Path(v).stat().st_size} for k,v in paths.items()},
      "allowed_runtime_configs":[str(Path(x)) for x in sorted(a.configs)],"git_commit":git,"python":platform.python_version()}
 Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(out,indent=2),encoding="utf-8")
 print(json.dumps({"status":out["status"],"files":len(paths)}))
if __name__=="__main__":main()
```

### 1.5 `register_artifact.py`：先验 hash、后反序列化、最后 completion marker

将紧随其后的代码块原样保存为 `register_artifact.py`：

```python
# file: register_artifact.py
import argparse,hashlib,json,os
from pathlib import Path
import torch
def file_sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
 return h.hexdigest()
def listing(p):
 p=Path(p)
 if p.is_file():return {p.name:{"sha256":file_sha(p),"bytes":p.stat().st_size}}
 return {str(x.relative_to(p)):{"sha256":file_sha(x),"bytes":x.stat().st_size}
         for x in sorted(p.rglob("*")) if x.is_file() and not x.name.startswith("COMPLETE.json")}
def marker(p):
 p=Path(p);return p.with_name(p.name+".complete.json") if p.is_file() else p/"COMPLETE.json"
def producer_receipt(p):
 p=Path(p);return p.with_name(p.name+".producer.json")
def verify_producer(p):
 p=Path(p).resolve();root=Path.cwd().resolve();runs=(root/"runs").resolve()
 try:p.relative_to(runs)
 except ValueError:raise SystemExit("artifact is outside approved runs tree")
 if not p.is_file() or p.suffix!=".pt":raise SystemExit("only runner-produced .pt checkpoints are registrable")
 rp=producer_receipt(p)
 if not rp.is_file():raise SystemExit("producer receipt missing before deserialization:"+str(rp))
 rec=json.loads(rp.read_text(encoding="utf-8"));frozen_path=root/"manifests/frozen_inputs.json"
 if not frozen_path.is_file():raise SystemExit("frozen manifest missing")
 frozen=json.loads(frozen_path.read_text(encoding="utf-8"));wrapper=p.parent.parent/"manifest.json";workload=p.parent/"run_manifest.json"
 required={"status":"PRODUCED_BY_FROZEN_RUNNER","artifact":str(p),"artifact_sha256":file_sha(p),
  "frozen_manifest_sha256":file_sha(frozen_path),"wrapper_manifest":str(wrapper),
  "wrapper_manifest_sha256":file_sha(wrapper),"workload_manifest_sha256":file_sha(workload),
  "runner_sha256":file_sha(root/"run_experiment.py")}
 if any(rec.get(k)!=v for k,v in required.items()):raise SystemExit("producer receipt/lineage mismatch")
 for key in ("runner","register"):
  entry=frozen.get("files",{}).get(key,{})
  if not entry or file_sha(root/entry["path"])!=entry.get("sha256"):raise SystemExit("producer trust code is not frozen:"+key)
 wrapper_doc=json.loads(wrapper.read_text(encoding="utf-8"))
 if wrapper_doc.get("status")!="COMPLETED_PENDING_REVIEW" or wrapper_doc.get("returncode")!=0:
  raise SystemExit("producer wrapper did not complete successfully")
 return rec
def verify_hash_manifest(p,m,allowed_status):
 p,m=Path(p),Path(m)
 if not m.is_file():raise SystemExit("hash marker missing:"+str(m))
 doc=json.loads(m.read_text(encoding="utf-8"));current=listing(p)
 if doc.get("status") not in allowed_status:raise SystemExit("marker status is not loadable")
 if not current or any(v["bytes"]<=0 for v in current.values()):raise SystemExit("empty/incomplete artifact")
 if doc.get("files")!=current:raise SystemExit("artifact hash mismatch; refusing before torch.load")
 return doc
def semantic_check(p):
 p=Path(p)
 if p.is_file() and p.suffix==".pt":
  x=torch.load(p,map_location="cpu")
  need={"model","optimizer","scaler","successful_step","attempted_step","rng_by_rank","contract"}
  missing=need-set(x) if isinstance(x,dict) else need
  if missing:raise SystemExit("checkpoint state missing:"+repr(sorted(missing)))
  contract=x["contract"];contract_need={"config","config_sha","data_sha","code_sha","world_size","global_batch","precision"}
  if not isinstance(contract,dict) or not contract_need.issubset(contract):raise SystemExit("checkpoint contract incomplete")
  if not isinstance(contract["config"],dict) or int(contract["world_size"])<1 or int(contract["global_batch"])<1:
   raise SystemExit("invalid checkpoint contract values")
  if contract["precision"] not in {"fp16","fp32"}:raise SystemExit("invalid checkpoint precision")
  if any(not isinstance(contract[k],str) or not contract[k] for k in ("config_sha","data_sha","code_sha")):
   raise SystemExit("checkpoint lineage hashes absent")
  if not isinstance(x["model"],dict) or not x["model"] or not all(torch.is_tensor(v) for v in x["model"].values()):
   raise SystemExit("empty/non-tensor model state")
  opt=x["optimizer"]
  if not isinstance(opt,dict) or not isinstance(opt.get("state"),dict) or not opt["state"] or not isinstance(opt.get("param_groups"),list) or not opt["param_groups"]:
   raise SystemExit("optimizer state/param_groups incomplete")
  scaler=x["scaler"]
  if not isinstance(scaler,dict):raise SystemExit("scaler state is not a dict")
  if contract["precision"]=="fp16" and not {"scale","growth_factor","backoff_factor","growth_interval","_growth_tracker"}.issubset(scaler):
   raise SystemExit("FP16 GradScaler state incomplete")
  if int(x["successful_step"])<1 or int(x["successful_step"])!=int(x["attempted_step"]):raise SystemExit("published update clocks are not equal/positive")
  if not isinstance(x["rng_by_rank"],list) or len(x["rng_by_rank"])!=int(contract["world_size"]):
   raise SystemExit("per-rank RNG state mismatch")
  for rank,r in enumerate(x["rng_by_rank"]):
   if not isinstance(r,dict) or not {"torch","numpy","python","cuda"}.issubset(r):raise SystemExit(f"rank {rank} RNG schema incomplete")
   if not torch.is_tensor(r["torch"]) or not isinstance(r["numpy"],tuple) or not isinstance(r["python"],tuple):
    raise SystemExit(f"rank {rank} RNG value types invalid")
   if r["cuda"] is not None and not torch.is_tensor(r["cuda"]):raise SystemExit(f"rank {rank} CUDA RNG invalid")
def verify_complete(p):
 p=Path(p);verify_producer(p);m=marker(p);doc=verify_hash_manifest(p,m,{"COMPLETE"});semantic_check(p);return doc
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--artifact",required=True);ap.add_argument("--verify",action="store_true");a=ap.parse_args()
 p=Path(a.artifact);m=marker(p)
 if not p.exists():raise SystemExit("artifact missing")
 if a.verify:
  doc=verify_complete(p);print(json.dumps({"status":"PASS","artifact":str(p),"files":len(doc["files"])}));return
 if m.exists():raise SystemExit("completion marker already exists; use --verify")
 verify_producer(p)
 current=listing(p)
 if not current or any(v["bytes"]<=0 for v in current.values()):raise SystemExit("empty/incomplete artifact")
 candidate=m.with_name(m.name+".candidate")
 if candidate.exists():raise SystemExit("stale candidate marker exists; inspect and remove only after confirming no producer is active")
 candidate.write_text(json.dumps({"status":"HASHED_PENDING_SEMANTIC","artifact":str(p),"files":current},indent=2),encoding="utf-8")
 try:
  verify_hash_manifest(p,candidate,{"HASHED_PENDING_SEMANTIC"})
  semantic_check(p)
  final={"status":"COMPLETE","artifact":str(p),"files":current}
  tmp=m.with_name(m.name+".tmp");tmp.write_text(json.dumps(final,indent=2),encoding="utf-8");os.replace(tmp,m)
 finally:
  candidate.unlink(missing_ok=True)
 doc=verify_complete(p)
 print(json.dumps({"status":"COMPLETE_VERIFIED","artifact":str(p),"files":len(doc["files"])}))
if __name__=="__main__":main()
```

首次登记只允许指向本周已批准 runner 刚生成的本地 artifact。runner 成功结束后写 producer receipt，绑定 artifact、wrapper/workload/frozen/runner SHA；登记器先验证 receipt、`runs/` 解析路径与已冻结 trust code，再建立候选 marker 并复算 bytes/SHA，之后才做 pickle 语义检查。语义检查要求非空 model/optimizer、FP16 scaler 完整字段、正且相等的双时钟、逐 rank 完整 RNG schema 与 contract；通过后最终 `COMPLETE` marker 才以原子替换发布。resume/eval 调用 `--verify` 时同样在 `torch.load` 前完成 producer+marker/hash 门。hash 不能把不可信外来 pickle 变可信。

### 1.6 `run_experiment.py`：完整唯一入口

将紧随其后的代码块原样保存为 `run_experiment.py`：

```python
# file: run_experiment.py
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
 return h.hexdigest()
def check_frozen(path="manifests/frozen_inputs.json"):
 m=json.loads(Path(path).read_text(encoding="utf-8"))
 for k,v in m["files"].items():
  if not Path(v["path"]).is_file() or sha(v["path"])!=v["sha256"]:raise SystemExit("frozen mismatch:"+k)
 return m
def registered_keys(frozen,path):
 target=Path(path).resolve();return [k for k,v in frozen["files"].items() if Path(v["path"]).resolve()==target]
def require_registered(frozen,path,role=None):
 keys=registered_keys(frozen,path)
 if not keys or (role is not None and not any(k.startswith(role) for k in keys)):
  raise SystemExit("runtime path is not registered for role "+repr(role)+":"+str(path))
def artifact_marker(p):
 p=Path(p);return p.with_name(p.name+".complete.json") if p.is_file() else p/"COMPLETE.json"
def require_complete(p):
 m=artifact_marker(p)
 if not m.is_file():raise SystemExit("checkpoint completion marker missing:"+str(m))
 subprocess.run([sys.executable,"register_artifact.py","--artifact",str(p),"--verify"],check=True)
 return {"checkpoint_sha256":sha(p) if Path(p).is_file() else None,"marker_sha256":sha(m)}
def atomic_json(path,doc):
 path=Path(path);tmp=path.with_name(path.name+".tmp")
 tmp.write_text(json.dumps(doc,indent=2),encoding="utf-8");os.replace(tmp,path)
def emit_producer_receipts(out,manifest):
 wrapper=(Path(out)/"manifest.json").resolve();workload=(Path(out)/"workload"/"run_manifest.json").resolve()
 if not workload.is_file():raise RuntimeError("successful producer lacks workload manifest")
 made=[]
 for p in sorted((Path(out)/"workload").glob("step_*.pt")):
  p=p.resolve();receipt=p.with_name(p.name+".producer.json")
  doc={"status":"PRODUCED_BY_FROZEN_RUNNER","artifact":str(p),"artifact_sha256":sha(p),
   "frozen_manifest_sha256":manifest["frozen_manifest_sha256"],"wrapper_manifest":str(wrapper),
   "wrapper_manifest_sha256":sha(wrapper),"workload_manifest_sha256":sha(workload),"runner_sha256":sha(__file__)}
  atomic_json(receipt,doc);made.append(str(receipt))
 return made
def launcher(world,code,args):
 if world==1:return [sys.executable,code]+args
 if not shutil.which("torchrun"):raise SystemExit("torchrun missing")
 return ["torchrun","--standalone",f"--nproc_per_node={world}",code]+args
def main():
 p=argparse.ArgumentParser();sp=p.add_subparsers(dest="cmd",required=True)
 for name in ("validate-data","train","resume","eval","profile"):
  q=sp.add_parser(name);q.add_argument("--config",required=True);q.add_argument("--run-id",required=True)
  q.add_argument("--world-size",type=int,default=1);q.add_argument("--max-steps",type=int);q.add_argument("--precision",choices=["fp32","fp16"])
  if name in ("resume","eval"):q.add_argument("--checkpoint",required=True)
 a=p.parse_args(); frozen=check_frozen();require_registered(frozen,a.config,"cap_config:")
 allowed={Path(x).resolve() for x in frozen.get("allowed_runtime_configs",[])}
 if Path(a.config).resolve() not in allowed:raise SystemExit("unregistered --config rejected")
 c=json.loads(Path(a.config).read_text(encoding="utf-8"))
 if c["kind"]!="miniwam":raise SystemExit("this released capstone runner only accepts miniwam")
 for k,role in (("data",None),("workload_code",None),("workload_config",None)):require_registered(frozen,c[k],role)
 ab_path="manifests/ab_contract.json";require_registered(frozen,ab_path,"ab_contract")
 ab=json.loads(Path(ab_path).read_text(encoding="utf-8"))
 if ab.get("status")!="PASS":raise SystemExit("A/B contract not PASS")
 if a.world_size not in (1,4,8):raise SystemExit("world-size must be 1,4,8")
 visible=os.getenv("CUDA_VISIBLE_DEVICES",""); vis=[x for x in visible.split(",") if x.strip()]
 if a.cmd!="validate-data" and len(vis)<a.world_size:raise SystemExit("visible GPU count smaller than world-size")
 out=Path("runs")/a.run_id
 if out.exists():raise SystemExit("run-id already exists; clean output required")
 cap=c["profile_steps"] if a.cmd=="profile" else c["max_steps"]
 steps=a.max_steps if a.max_steps is not None else cap
 if steps<1 or steps>cap:raise SystemExit(f"max-steps must be within 1..{cap}")
 precision=a.precision or c["precision"]
 common=["--config",c["workload_config"],"--data",c["data"]]
 checkpoint_evidence=require_complete(a.checkpoint) if a.cmd in ("resume","eval") else None
 out.mkdir(parents=True)
 if a.cmd=="validate-data":cmd=[sys.executable,c["workload_code"],"validate"]+common
 elif a.cmd in ("train","resume","profile"):
  args=["train"]+common+["--out",str(out/"workload"),"--max-steps",str(steps),"--precision",precision]
  if a.cmd=="resume":args += ["--resume",a.checkpoint]
  inner=launcher(a.world_size,c["workload_code"],args)
  if a.cmd=="profile":
   if not shutil.which("nsys"):raise SystemExit("nsys missing: profile is ENV-BLOCKED")
   cmd=["nsys","profile","--trace=cuda,nvtx,osrt,cudnn,cublas","--sample=none","--force-overwrite=true",
        "--output",str(out/"trace")]+inner
  else:cmd=inner
 else:
  cmd=[sys.executable,c["workload_code"],"eval"]+common+[
   "--checkpoint",a.checkpoint,"--split",c["eval_split"],"--batch-size",str(c["eval_batch"]),"--out",str(out/"eval.json")]
 env=os.environ.copy();env.update({"WANDB_DISABLED":"true","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"})
 manifest={"status":"STARTED_NOT_VALIDATED","run_id":a.run_id,"command":cmd,"config":c,
  "config_sha256":sha(a.config),"frozen_manifest_sha256":sha("manifests/frozen_inputs.json"),
  "ab_contract_sha256":sha(ab_path),"initial_weight_sha256":ab["initial_weight_sha256"],
  "expected_sample_stream_sha256":ab["sample_stream_sha256"],"expected_eval_ids_sha256":ab["eval_ids_sha256"],
  "world_size":a.world_size,"visible_mapping":visible,"precision":precision,"max_steps":steps,
  "parent_checkpoint":getattr(a,"checkpoint",None),"checkpoint_evidence":checkpoint_evidence,
  "budget_unit":c["budget_unit"],"target_tokens_seen":c["target_tokens_seen"]}
 (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
 t=time.time();r=subprocess.run(cmd,env=env);manifest["elapsed_seconds"]=time.time()-t;manifest["returncode"]=r.returncode
 evidence_ok=True
 workload_manifest=out/"workload"/"run_manifest.json"
 if workload_manifest.is_file():
  w=json.loads(workload_manifest.read_text(encoding="utf-8"));manifest["workload_manifest_sha256"]=sha(workload_manifest)
  manifest["attempted_updates"]=w.get("attempted_steps");manifest["successful_updates"]=w.get("successful_steps")
  manifest["actual_global_batch"]=w.get("global_batch")
  if isinstance(manifest["successful_updates"],int):
   manifest["successful_samples"]=manifest["successful_updates"]*int(manifest["actual_global_batch"])
 elif r.returncode==0 and a.cmd in ("train","resume","profile"):
  evidence_ok=False;manifest["evidence_error"]="workload manifest missing"
 if a.cmd in ("train","resume","profile") and r.returncode==0:
  observed=out/"workload"/"observed_sample_stream.json"
  audit_files=sorted((out/"workload").glob("sample_ids.rank*.jsonl"))
  if not observed.is_file() or not audit_files:evidence_ok=False;manifest["evidence_error"]="full sample-ID evidence missing"
  else:
   o=json.loads(observed.read_text(encoding="utf-8"));manifest["observed_sample_stream"]=o
   manifest["observed_sample_stream_artifact_sha256"]=sha(observed)
   manifest["sample_id_audit_sha256"]={str(p.relative_to(out)):sha(p) for p in audit_files}
  metrics=sorted((out/"workload").glob("metrics.rank*.jsonl"))
  if not metrics:evidence_ok=False;manifest["evidence_error"]="optimizer metrics evidence missing"
  else:manifest["metrics_sha256"]={str(p.relative_to(out)):sha(p) for p in metrics}
 if a.cmd=="eval" and (out/"eval.json").is_file():
  e=json.loads((out/"eval.json").read_text(encoding="utf-8"));manifest["eval_artifact_sha256"]=sha(out/"eval.json")
  manifest["eval_protocol"]={"split":e.get("split"),"batch":c["eval_batch"],"eval_ids_sha256":e.get("eval_ordered_ids_sha256"),
    "eval_n":e.get("eval_n"),
    "checkpoint_sha256":e.get("checkpoint_sha"),"data_sha256":e.get("data_sha"),"code_sha256":e.get("code_sha")}
 elif a.cmd=="eval" and r.returncode==0:evidence_ok=False;manifest["evidence_error"]="eval artifact missing"
 if r.returncode==0 and not evidence_ok:manifest["returncode"]=2
 manifest["status"]="COMPLETED_PENDING_REVIEW" if manifest["returncode"]==0 else "FAILED"
 (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
 receipts=emit_producer_receipts(out,manifest) if manifest["status"]=="COMPLETED_PENDING_REVIEW" and a.cmd in ("train","resume","profile") else []
 print(json.dumps({"status":manifest["status"],"run_id":a.run_id,"returncode":manifest["returncode"],"producer_receipts":len(receipts)}));raise SystemExit(manifest["returncode"])
if __name__=="__main__":main()
```

入口用参数数组而非 `shell=True`；不会联网/上传；拒绝覆盖已有 run，也拒绝未登记的 `--config` 及其 code/config/data。每次运行重验 frozen 白名单；checkpoint 先由 artifact guard 验 producer receipt 与 marker/hash 再交给 workload。`CUDA_VISIBLE_DEVICES` 的逻辑编号映射进入 manifest；workload 的 attempted/successful updates、完整实际 sample-ID audit/summary、逐 rank metrics 与各自 SHA 会回填，eval protocol 只接受 adapter 实际观察到的 ordered IDs/n。成功训练的 wrapper manifest 落盘后才为 checkpoint 写 producer receipt；状态仍是 `COMPLETED_PENDING_REVIEW`，不会自动写 PASS。

### 1.7 `verify_publication.py`：发布证据一致性门

将紧随其后的代码块原样保存为 `verify_publication.py`：

```python
# file: verify_publication.py
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
 return h.hexdigest()
def verify_frozen():
 frozen=load("manifests/frozen_inputs.json")
 if frozen.get("status")!="FROZEN_NOT_RUN":raise RuntimeError("frozen manifest status drift")
 for key,v in frozen.get("files",{}).items():
  p=Path(v["path"])
  if not p.is_file() or sha(p)!=v["sha256"] or p.stat().st_size!=v["bytes"]:raise RuntimeError("frozen input drift:"+key)
 ab=load("manifests/ab_contract.json")
 if ab.get("status")!="PASS":raise RuntimeError("A/B contract is not PASS")
 for p,h in ab.get("inputs_sha256",{}).items():
  if not Path(p).is_file() or sha(p)!=h:raise RuntimeError("A/B input drift:"+p)
 return sha("manifests/frozen_inputs.json"),ab
def verify_hash_map(root,mapping,label):
 if not isinstance(mapping,dict) or not mapping:raise RuntimeError(label+" hash map absent")
 for rel,h in mapping.items():
  p=Path(root)/rel
  if not p.is_file() or sha(p)!=h:raise RuntimeError(label+" evidence drift:"+str(p))
def observed(run,manifest,ab):
 root=Path(run);wm=root/"workload"/"run_manifest.json"
 if not wm.is_file() or sha(wm)!=manifest.get("workload_manifest_sha256"):raise RuntimeError("workload manifest drift")
 verify_hash_map(root,manifest.get("metrics_sha256"),"optimizer metrics")
 verify_hash_map(root,manifest.get("sample_id_audit_sha256"),"sample-ID audit")
 metric_steps=[]
 for rel in sorted(manifest["metrics_sha256"]):
  rank=Path(rel).stem.replace("metrics.rank","")
  for line in (root/rel).read_text(encoding="utf-8").splitlines():
   x=json.loads(line)
   if x.get("skipped") or x["attempted_step"]!=x["successful_step"]:raise RuntimeError("non-successful update in formal A/B")
   metric_steps.append((int(x["successful_step"]),rank))
 expected_steps=list(range(1,ab["successful_updates"]+1))
 if sorted({x[0] for x in metric_steps})!=expected_steps:raise RuntimeError("formal optimizer-step logs incomplete")
 records=[]
 for rel in sorted(manifest["sample_id_audit_sha256"]):
  records.extend(json.loads(x) for x in (root/rel).read_text(encoding="utf-8").splitlines())
 records.sort(key=lambda x:(int(x["successful_step"]),int(x["rank"])))
 ordered=[]
 for step in expected_steps:
  group=[x for x in records if int(x["successful_step"])==step]
  if [int(x["rank"]) for x in group]!=list(range(ab["formal_world_size"])):raise RuntimeError("full sample stream rank/step coverage mismatch")
  for x in group:ordered.extend(map(str,x["sample_ids"]))
 observed_sha=hashlib.sha256("\n".join(ordered).encode()).hexdigest();summary=root/"workload"/"observed_sample_stream.json"
 if not summary.is_file() or sha(summary)!=manifest.get("observed_sample_stream_artifact_sha256"):raise RuntimeError("sample-stream summary drift")
 doc=load(summary)
 if doc.get("ordered_ids_sha256")!=observed_sha or doc.get("successful_samples")!=len(ordered):raise RuntimeError("sample-stream summary does not match actual audit rows")
 if observed_sha!=ab["sample_stream_sha256"] or len(ordered)!=ab["successful_samples"]:raise RuntimeError("observed full sample stream differs from frozen A/B contract")
 return observed_sha
def verify_artifact(path):
 subprocess.run([sys.executable,"register_artifact.py","--artifact",path,"--verify"],check=True)
def main():
 ap=argparse.ArgumentParser()
 for name in ("reference_run","treatment_run","reference_eval","treatment_eval"):ap.add_argument("--"+name.replace("_","-"),required=True)
 ap.add_argument("--out",required=True);a=ap.parse_args()
 frozen_sha,ab=verify_frozen()
 rr,tr=load(Path(a.reference_run)/"manifest.json"),load(Path(a.treatment_run)/"manifest.json")
 re,te=load(Path(a.reference_eval)/"manifest.json"),load(Path(a.treatment_eval)/"manifest.json")
 for x in (rr,tr,re,te):
  if x.get("status")!="COMPLETED_PENDING_REVIEW" or x.get("frozen_manifest_sha256")!=frozen_sha:
   raise RuntimeError("run status/frozen lineage mismatch")
 if (rr["config"]["name"],tr["config"]["name"])!=("reference_action_only","treatment_action_world"):
  raise RuntimeError("formal A/B role mismatch")
 expected=(ab["successful_updates"],ab["successful_samples"],ab["global_batch"],ab["formal_world_size"])
 for x in (rr,tr):
  got=(x.get("successful_updates"),x.get("successful_samples"),x.get("actual_global_batch"),x.get("world_size"))
  if got!=expected or x.get("attempted_updates")!=x.get("successful_updates"):raise RuntimeError("formal A/B budget mismatch")
  if x.get("target_tokens_seen")!=0 or x.get("precision")!="fp16":raise RuntimeError("formal precision/token-N/A contract mismatch")
  if x.get("initial_weight_sha256")!=ab["initial_weight_sha256"]:raise RuntimeError("initial weight contract mismatch")
 hs_r=observed(a.reference_run,rr,ab);hs_t=observed(a.treatment_run,tr,ab)
 if hs_r!=hs_t:raise RuntimeError("observed A/B full successful-step sample streams differ")
 for ev,train,run_dir in ((re,rr,a.reference_run),(te,tr,a.treatment_run)):
  parent=ev.get("parent_checkpoint")
  if not parent:raise RuntimeError("eval lacks parent checkpoint")
  expected_parent=Path(run_dir)/"workload"/f"step_{ab['successful_updates']:06d}.pt"
  if Path(parent).resolve()!=expected_parent.resolve():raise RuntimeError("eval is not bound to its formal A/B checkpoint")
  verify_artifact(parent)
  if ev.get("checkpoint_evidence",{}).get("checkpoint_sha256")!=sha(parent):raise RuntimeError("eval parent hash drift")
  eval_path=Path(ev["command"][-1])
  if not eval_path.is_file() or sha(eval_path)!=ev.get("eval_artifact_sha256"):raise RuntimeError("eval artifact drift")
  protocol=ev.get("eval_protocol",{});actual=load(eval_path)
  if actual.get("status")!="PASS":raise RuntimeError("eval artifact status is not PASS")
  if protocol.get("checkpoint_sha256")!=sha(parent):raise RuntimeError("workload eval checkpoint hash mismatch")
  if protocol.get("eval_ids_sha256")!=actual.get("eval_ordered_ids_sha256") or protocol.get("eval_n")!=actual.get("eval_n"):
   raise RuntimeError("eval manifest is not derived from actual ordered IDs")
  if protocol.get("split")!=ab["eval_split"] or protocol.get("batch")!=ab["eval_batch"]:raise RuntimeError("eval split/batch drift")
  if protocol.get("eval_ids_sha256")!=ab["eval_ids_sha256"] or protocol.get("eval_n")!=ab["eval_n"]:raise RuntimeError("eval ID protocol drift")
 shared=("split","batch","eval_ids_sha256","eval_n","data_sha256","code_sha256")
 if any(re["eval_protocol"].get(k)!=te["eval_protocol"].get(k) for k in shared):raise RuntimeError("A/B eval protocols differ")
 er,et=load(Path(a.reference_eval)/"eval.json"),load(Path(a.treatment_eval)/"eval.json")
 for e in (er,et):
  if not {"action","future","last","relative_improvement"}.issubset(e):raise RuntimeError("eval lacks action/future/last evidence")
 out={"status":"EVIDENCE_CONSISTENT_PENDING_HUMAN_MODEL_DECISION","frozen_manifest_sha256":frozen_sha,
  "ab_contract_sha256":sha("manifests/ab_contract.json"),"initial_weight_sha256":ab["initial_weight_sha256"],
  "successful_updates":ab["successful_updates"],"successful_samples":ab["successful_samples"],
  "target_tokens_seen":0,"target_tokens_semantics":ab["target_tokens_semantics"],
  "analytic_sample_stream_sha256":ab["sample_stream_sha256"],"observed_full_sample_stream_sha256":hs_r,
  "eval_ids_sha256":re["eval_protocol"]["eval_ids_sha256"],"reference_checkpoint_sha256":sha(re["parent_checkpoint"]),
  "treatment_checkpoint_sha256":sha(te["parent_checkpoint"]),"model_metrics_interpreted":False}
 dst=Path(a.out);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(json.dumps(out,indent=2),encoding="utf-8")
 print(json.dumps(out))
if __name__=="__main__":main()
```

该脚本只给出“证据内部一致，等待人工解释”，不会根据 action MSE 自动写 `PASS`。它先逐项重算 frozen 与 A/B input SHA，再验证两个 checkpoint producer/marker/hash、workload manifest、逐 rank metrics 与完整 sample-ID audit；实际完整 stream 必须与分析合同及另一 arm 相等。两份 eval JSON 也须与 wrapper 记录 SHA一致，其实际 ordered-ID SHA/n、split/batch/data/code相同并命中冻结合同；同时强制保留正确 target-encoder 下的 `last` 基线。

### 1.8 `tests/test_runner.py` 与 README

将紧随其后的代码块原样保存为 `tests/test_runner.py`：

```python
# file: tests/test_runner.py
import json,random,sys,tempfile,unittest
from unittest import mock
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));import register_artifact,run_experiment
class T(unittest.TestCase):
 def valid_state(self):
  contract={"config":{},"config_sha":"c","data_sha":"d","code_sha":"x","world_size":1,"global_batch":2,"precision":"fp16"}
  optimizer={"state":{0:{"step":torch.tensor(1.)}},"param_groups":[{"params":[0]}]}
  scaler={"scale":65536.0,"growth_factor":2.0,"backoff_factor":0.5,"growth_interval":2000,"_growth_tracker":1}
  rng={"torch":torch.get_rng_state(),"numpy":np.random.get_state(),"python":random.getstate(),"cuda":None}
  return {"model":{"w":torch.ones(1)},"optimizer":optimizer,"scaler":scaler,"successful_step":1,
          "attempted_step":1,"rng_by_rank":[rng],"contract":contract}
 def test_file_marker_detects_corruption(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"x.pt";torch.save(self.valid_state(),p)
   register_artifact.marker(p).write_text(json.dumps({"status":"COMPLETE","artifact":str(p),
    "files":register_artifact.listing(p)}),encoding="utf-8")
   with mock.patch.object(register_artifact,"verify_producer",return_value={}):register_artifact.verify_complete(p)
   before=p.read_bytes();p.write_bytes(before[:-1]+bytes([before[-1]^0xFF]))
   with mock.patch.object(register_artifact,"verify_producer",return_value={}):
    with mock.patch.object(register_artifact.torch,"load",side_effect=AssertionError("torch.load must not run")):
     with self.assertRaises(SystemExit):register_artifact.verify_complete(p)
 def test_semantic_state_groups_fail_closed(self):
  for field,bad in (("optimizer",{}),("scaler",{}),("rng_by_rank",[{"torch":torch.get_rng_state()}])):
   with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
    state=self.valid_state();state[field]=bad;p=Path(d)/"bad.pt";torch.save(state,p)
    with self.assertRaises(SystemExit):register_artifact.semantic_check(p)
 def test_untrusted_first_registration_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"outside.pt";torch.save(self.valid_state(),p)
   with self.assertRaises(SystemExit):register_artifact.verify_producer(p)
 def test_unregistered_config_rejected(self):
  frozen={"files":{"cap_config:configs/reference.json":{"path":"configs/reference.json"}}}
  with self.assertRaises(SystemExit):run_experiment.require_registered(frozen,"configs/not_registered.json","cap_config:")
 def test_configs(self):
  for p in ("configs/reference.json","configs/treatment.json","configs/pre_registration.json"):json.load(open(p))
if __name__=="__main__":unittest.main()
```

将紧随其后的 Markdown 代码块原样保存为 `README_OFFLINE.md`：

```markdown
<!-- file: README_OFFLINE.md -->
# Offline run order
1. Activate the already-approved environment; do not create or upgrade it.
2. `python -m unittest -v tests/test_runner.py`.
3. Generate `manifests/ab_contract.json` with `verify_ab.py`, then freeze exactly reference+treatment with `freeze_inputs.py`, before results.
4. Run `validate-data`, then 1-GPU FP32/FP16 smoke, then 4/8-GPU bounded system runs.
5. Run both formal A/B arms for 300 successful updates on one GPU and evaluate both with the frozen dev protocol.
6. Register every checkpoint; resume/eval reject unregistered or hash-mismatched artifacts before loading.
7. Run `verify_publication.py`; it checks evidence consistency but never chooses the model conclusion.
8. Build `reports/artifacts.sha256` while explicitly excluding that file itself. All artifacts remain local.
```

## Day 1：冻结、静态检查与 clean-room dry run

### 为什么做

证明新 shell 只看 README 就能从空 runs 目录工作，并冻结所有因果变量。

### 输入与前置检查

§0复制/validator通过；`runs/` 为空；pre-registration已由人读完，primary未看结果。

### 本日要创建/修改的文件

完成 §1 全部文件；先生成 `manifests/ab_contract.json`，再生成唯一 `manifests/frozen_inputs.json`，之后不改 frozen 文件。

### 实现（固定代码路径）

使用本文完整 wrapper/guard/A-B verifier/publication verifier/freeze/test；workload固定 Week11 `run.py` 与本文 `evidence_run.py` 两个 SHA。A/B 合同先实例化两次模型比较初始 state hash，再枚举 300 个 successful step 的预期完整 sample stream 和 dev ID 顺序；运行时 evidence adapter 独立记录实际完整 IDs供发布门复核。

### 执行命令

```bash
cd week14-capstone
python -m py_compile run_experiment.py freeze_inputs.py register_artifact.py verify_ab.py verify_publication.py tests/test_runner.py workload/run.py workload/evidence_run.py
python -m unittest -v tests/test_runner.py
python -m json.tool configs/pre_registration.json
python verify_ab.py --reference configs/reference.json --treatment configs/treatment.json --out manifests/ab_contract.json
python freeze_inputs.py --configs configs/reference.json configs/treatment.json --out manifests/frozen_inputs.json
python - <<'PY'
import json
from pathlib import Path
x=json.loads(Path("manifests/frozen_inputs.json").read_text())
assert Path("manifests/frozen_inputs.json").resolve() not in {Path(v["path"]).resolve() for v in x["files"].values()}
assert set(map(Path,x["allowed_runtime_configs"]))=={Path("configs/reference.json"),Path("configs/treatment.json")}
print({"frozen_files":len(x["files"]),"self_included":False,"runtime_configs":x["allowed_runtime_configs"]})
PY
CUDA_VISIBLE_DEVICES=0 python run_experiment.py validate-data --config configs/reference.json --run-id day1_validate --world-size 1
```

### 预期观测（估算/示例，不是实测）

测试 `OK`；A/B verifier 输出相同 initial state、300-step sample stream 与 dev IDs；freeze显示完整白名单文件数且明确 self-included=false；validator显示 `status=PASS`。CPU/单卡耗时估算<5分钟。

### 验收条件

所有路径已存在；A/B only-change、相同初始权重/预算/stream/eval合同通过；所有实际 code/config/data 在 SHA 白名单；未登记 config 单测拒绝；freeze manifest 不自包含；runs不覆盖；无网络/云logger；新shell仅README可复现；validate manifest关联完整。

### 若失败，按什么顺序查

缺文件 → Week11重建 → unit → frozen mismatch → config JSON → data validator。冻结后需要改代码则新建 `frozen_inputs.v2.json` 和新实验，不能覆盖。

### 当日证据清单

依赖freeze、unit、pre-registration、frozen manifest、validate manifest/stdout。

## Day 2：单卡 FP32 reference、FP16 smoke、checkpoint 与 eval

### 为什么做

建立多卡前的数值 reference 和可独立加载的完整状态。

### 输入与前置检查

Day1 PASS；GPU0可用；所有 frozen SHA仍匹配。

### 本日要创建/修改的文件

只生成新的 run IDs；不修改代码/config。

### 实现（固定代码路径）

先20-step FP32与重复run，再100-step FP16；register marker最后写。

### 执行命令

```bash
cd week14-capstone
CUDA_VISIBLE_DEVICES=0 python run_experiment.py train --config configs/reference.json --run-id ref_one_step --world-size 1 --precision fp32 --max-steps 1
CUDA_VISIBLE_DEVICES=0 python run_experiment.py train --config configs/reference.json --run-id ref_fp32_a --world-size 1 --precision fp32 --max-steps 20
CUDA_VISIBLE_DEVICES=0 python run_experiment.py train --config configs/reference.json --run-id ref_fp32_b --world-size 1 --precision fp32 --max-steps 20
CUDA_VISIBLE_DEVICES=0 python run_experiment.py train --config configs/reference.json --run-id ref_fp16_100 --world-size 1 --precision fp16 --max-steps 100
python register_artifact.py --artifact runs/ref_fp16_100/workload/step_000100.pt
python register_artifact.py --artifact runs/ref_fp16_100/workload/step_000100.pt --verify
CUDA_VISIBLE_DEVICES=0 python run_experiment.py eval --config configs/reference.json --run-id ref_eval --world-size 1 \
 --checkpoint runs/ref_fp16_100/workload/step_000100.pt
```

### 预期观测（估算/示例，不是实测）

单 batch 前反向先完成；两个20-step FP32 run sample IDs相同；FP16记录loss scale/skips/peak；eval产生 action/future/last/relative improvement。

### 验收条件

1-step finite；FP32重复在预注册容差；FP16无非有限、skip<1%、reserved<30.5GB；checkpoint marker/SHA；eval独立进程且绑定checkpoint。

### 若失败，按什么顺序查

sample IDs/RNG → config/data/code SHA → FP32 first diverge → FP16 loss scale → checkpoint marker → eval split。系统门未过不进多卡。

### 当日证据清单

两次FP32 metrics、FP16 metrics/checkpoint/marker、eval JSON、显存/step p50/p95、待验证标签。

## Day 3：4/8卡 correctness、scaling 与 profile

### 为什么做

证明多卡没有改变任务，再测扩展；correctness与performance分开。

### 输入与前置检查

Day2 PASS；`nvidia-smi topo -m`现场确认卡组；共享机器调度批准。

### 本日要创建/修改的文件

生成 `ws4/ws8/profile` runs；trace只留公司。

### 实现（固定代码路径）

Toy config global batch64：1卡每rank64、4卡16、8卡8，由 workload按world自动分区。这里是固定global correctness；弱扩展需另建config增大global batch，不混表。

### 执行命令

```bash
cd week14-capstone
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_experiment.py train --config configs/reference.json --run-id ref_ws4_20 --world-size 4 --max-steps 20
CUDA_VISIBLE_DEVICES=4,5,6,7 python run_experiment.py train --config configs/reference.json --run-id ref_ws4_symmetry --world-size 4 --max-steps 20
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_experiment.py train --config configs/reference.json --run-id ref_ws8_20 --world-size 8 --max-steps 20
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_experiment.py profile --config configs/treatment.json --run-id trt_ws8_profile --world-size 8 --max-steps 50
```

### 预期观测（估算/示例，不是实测）

各rank metrics/sample IDs互补；前20 successful steps loss落在预注册2%容差；profile产生 `.nsys-rep`。Toy小模型E8可能很低，不自动判NCCL故障。

### 验收条件

4/8卡rank参数由DDP同步；global sample IDs覆盖且不重叠；loss/grad容差；step p50/p95用同warmup窗口；trace有界≤3分钟或标ENV-BLOCKED。

### 若失败，按什么顺序查

world/visible mapping → 每rank sample IDs → global batch →额外/world除法 →首个rank异常 →2-step `NCCL_DEBUG=INFO`。不以禁用P2P作为永久修复。

### 当日证据清单

1/4/8 manifests/rank logs、correctness表、throughput/E_N定义、NCCL/profile摘要和trace本机路径。

## Day 4：同预算正式 A/B、中断恢复与坏 checkpoint 拒绝

### 为什么做

先完成真正同预算的正式 A/B 与同协议 eval；再用独立 recovery run 证明失败后不会静默续出错误结果，避免把“是否经历 resume”混入模型 treatment。

### 输入与前置检查

Day3 correctness PASS；`manifests/ab_contract.json` 仍在 frozen manifest 中且 hash 匹配；reference/treatment 唯一差异已核对；正式 A/B 各 300 successful updates 与额外 recovery 的磁盘预算足够。

### 本日要创建/修改的文件

生成 `ref_formal_300/trt_formal_300` 及两份 eval；另生成 `trt_recovery_{continuous,partial,resumed}` 与 bad probe。

### 实现（固定代码路径）

正式 A/B 都从同一 seed 初始化、单卡 FP16、global batch64、300 successful updates/19,200 samples、token N/A=0，并在实际 batch 调用点记录同一完整 sample stream；两边都走 continuous，eval 记录相同实际 dev ordered-ID SHA/n、batch/evaluator。recovery 使用独立 treatment：常数 LR schedule、相同 config/seed/sample stream，100→200 resume 与 continuous 200 比较 step101 完整 IDs/LR/scale/loss/grad，并在终点比较 model/optimizer/scaler/逐 rank RNG。坏 checkpoint 必须在进入 `torch.load` 前因 producer/marker/SHA 被拒绝。

### 执行命令

```bash
cd week14-capstone
CUDA_VISIBLE_DEVICES=0 python run_experiment.py train --config configs/reference.json --run-id ref_formal_300 --world-size 1 --max-steps 300
python register_artifact.py --artifact runs/ref_formal_300/workload/step_000300.pt
python register_artifact.py --artifact runs/ref_formal_300/workload/step_000300.pt --verify
CUDA_VISIBLE_DEVICES=0 python run_experiment.py train --config configs/treatment.json --run-id trt_formal_300 --world-size 1 --max-steps 300
python register_artifact.py --artifact runs/trt_formal_300/workload/step_000300.pt
python register_artifact.py --artifact runs/trt_formal_300/workload/step_000300.pt --verify
CUDA_VISIBLE_DEVICES=0 python run_experiment.py eval --config configs/reference.json --run-id ref_formal_eval --world-size 1 \
 --checkpoint runs/ref_formal_300/workload/step_000300.pt
CUDA_VISIBLE_DEVICES=0 python run_experiment.py eval --config configs/treatment.json --run-id trt_formal_eval --world-size 1 \
 --checkpoint runs/trt_formal_300/workload/step_000300.pt

CUDA_VISIBLE_DEVICES=0 python run_experiment.py train --config configs/treatment.json --run-id trt_recovery_continuous_200 --world-size 1 --max-steps 200
python register_artifact.py --artifact runs/trt_recovery_continuous_200/workload/step_000200.pt
python register_artifact.py --artifact runs/trt_recovery_continuous_200/workload/step_000200.pt --verify
CUDA_VISIBLE_DEVICES=0 python run_experiment.py train --config configs/treatment.json --run-id trt_recovery_partial_100 --world-size 1 --max-steps 100
python register_artifact.py --artifact runs/trt_recovery_partial_100/workload/step_000100.pt
python register_artifact.py --artifact runs/trt_recovery_partial_100/workload/step_000100.pt --verify
CUDA_VISIBLE_DEVICES=0 python run_experiment.py resume --config configs/treatment.json --run-id trt_recovery_resumed_200 --world-size 1 --max-steps 200 \
 --checkpoint runs/trt_recovery_partial_100/workload/step_000100.pt
python register_artifact.py --artifact runs/trt_recovery_resumed_200/workload/step_000200.pt
python register_artifact.py --artifact runs/trt_recovery_resumed_200/workload/step_000200.pt --verify
python - <<'PY'
import json,subprocess,sys
import numpy as np
import torch
def rows(p):return {int(x["successful_step"]):x for x in map(json.loads,open(p)) if not x["skipped"]}
def ids(p):return {int(x["successful_step"]):x["sample_ids"] for x in map(json.loads,open(p))}
def equal(a,b):
 if torch.is_tensor(a) or torch.is_tensor(b):return torch.is_tensor(a) and torch.is_tensor(b) and torch.equal(a,b)
 if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):return isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and np.array_equal(a,b)
 if isinstance(a,dict) or isinstance(b,dict):return isinstance(a,dict) and isinstance(b,dict) and set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)) or isinstance(b,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
ca="runs/trt_recovery_continuous_200/workload/step_000200.pt"
cb="runs/trt_recovery_resumed_200/workload/step_000200.pt"
for p in (ca,cb):subprocess.run([sys.executable,"register_artifact.py","--artifact",p,"--verify"],check=True)
a=rows("runs/trt_recovery_continuous_200/workload/metrics.rank0.jsonl");b=rows("runs/trt_recovery_resumed_200/workload/metrics.rank0.jsonl")
ia=ids("runs/trt_recovery_continuous_200/workload/sample_ids.rank0.jsonl");ib=ids("runs/trt_recovery_resumed_200/workload/sample_ids.rank0.jsonl")
assert set(range(101,201)).issubset(a) and set(b)==set(range(101,201));assert all(ia[s]==ib[s] for s in range(101,201))
assert a[101]["loss_scale"]==b[101]["loss_scale"]
loss_rel=abs(a[101]["loss"]-b[101]["loss"])/max(abs(a[101]["loss"]),1e-12);assert loss_rel<0.01
grad_rel=abs(a[101]["grad_norm"]-b[101]["grad_norm"])/max(abs(a[101]["grad_norm"]),1e-12);assert grad_rel<0.01
x=torch.load(ca,map_location="cpu");y=torch.load(cb,map_location="cpu")
assert x["contract"]==y["contract"] and x["contract"]["config"]["lr"]==y["contract"]["config"]["lr"]
assert x["successful_step"]==x["attempted_step"]==y["successful_step"]==y["attempted_step"]==200
assert equal(x["optimizer"],y["optimizer"]);assert equal(x["scaler"],y["scaler"]);assert equal(x["rng_by_rank"],y["rng_by_rank"])
assert set(x["model"])==set(y["model"]);delta=max(float((x["model"][k]-y["model"][k]).abs().max()) for k in x["model"])
print({"fixed_next_step":101,"next_loss_relative":loss_rel,"next_grad_relative":grad_rel,
 "lr":x["contract"]["config"]["lr"],"terminal_optimizer_scaler_rng_equal":True,"final_state_max_abs":delta});assert delta<=1e-6
PY
bad=runs/trt_recovery_partial_100/workload/step_000100.pt
backup=runs/trt_recovery_partial_100/workload/step_000100.pt.clean-backup
test ! -e "$backup"; cp "$bad" "$backup"
restore_bad_probe(){ mv -f "$backup" "$bad"; }
trap restore_bad_probe EXIT
python - <<'PY'
from pathlib import Path
p=Path("runs/trt_recovery_partial_100/workload/step_000100.pt");before=p.read_bytes();assert before
p.write_bytes(before[:-1]+bytes([before[-1]^0xFF]));print({"corrupted":str(p),"bytes":len(before)})
PY
if CUDA_VISIBLE_DEVICES=0 python run_experiment.py resume --config configs/treatment.json --run-id must_reject_bad --world-size 1 --max-steps 101 \
 --checkpoint "$bad"; then
  echo 'ERROR: corrupted checkpoint was accepted'; exit 2
else
  echo 'EXPECTED: hash mismatch rejected before torch.load'
fi
restore_bad_probe; trap - EXIT
python register_artifact.py --artifact "$bad" --verify
```

### 预期观测（估算/示例，不是实测）

正式 A/B 都完成 300 successful/attempted updates、19,200 successful samples，实际完整 sample-stream SHA 与合同相同；两份独立 eval 的实际 ordered IDs/n相同且都含 action/future/last/relative improvement。recovery 从 step100 到200，step101 的完整 IDs/LR/scale/loss/grad及终态 model/optimizer/scaler/RNG均通过；坏 artifact 因 producer/hash不一致被 wrapper 拒绝，不进入 `torch.load`/训练。

### 验收条件

A/B 初始 weight hash、world/global batch/precision、successful updates/samples、token N/A、实际完整 sample stream与 eval ordered-ID SHA/n 全相等；两份 eval 各自绑定本 arm 的唯一 step300 checkpoint。recovery next fixed-batch loss/grad相对偏差<1%、终态 max-abs≤1e-6，optimizer/scaler/逐 rank RNG相等且 LR固定；producer receipt/marker完整；坏 checkpoint 确定在 load 前拒绝。

### 若失败，按什么顺序查

frozen/AB contract → 两臂 initial hash/budget → successful/attempted与 sample IDs → eval checkpoint/protocol → recovery marker/SHA → RNG/optimizer/scaler → fixed-next-batch loss/state。任何偏差未解释即 FAIL-SYSTEM。

### 当日证据清单

A/B formal manifests/markers/两份 eval，continuous/partial/resume manifests、fixed-next/state 对比、坏 checkpoint 拒绝 stderr。

## Day 5：报告、闭卷答辩与交接

### 为什么做

把执行证据变成可审计能力，而不是漂亮但无法追踪的结果图。

### 输入与前置检查

Day1–4证据完整；test未调参；预计与实测分栏；公司边界再次确认。

### 本日要创建/修改的文件

```text
reports/executive_summary.md
reports/correctness.md
reports/scaling.md
reports/profiling.md
reports/failure_recovery.md
reports/model_results.md
reports/capability_matrix.md
reports/next_8_weeks.md
reports/publication_check.json
reports/artifacts.sha256
```

每一节写结论、证据路径、未决风险、状态；模型结果表至少含 A/B 的 action/future/last、成本和结论。Toy-only 明确 domain INCONCLUSIVE。

### 实现（固定代码路径）

不新增数据处理。报告只读取 JSON/JSONL/manifest；缺失写 `NOT-RUN`。

### 执行命令

```bash
cd week14-capstone
python verify_publication.py --reference-run runs/ref_formal_300 --treatment-run runs/trt_formal_300 \
 --reference-eval runs/ref_formal_eval --treatment-eval runs/trt_formal_eval \
 --out reports/publication_check.json
find runs -type f \( -name manifest.json -o -name eval.json -o -name '*.jsonl' -o -name '*.complete.json' \) \
 | sort > reports/evidence_paths.txt
sha_tmp=$(mktemp)
find configs workload data manifests runs reports -type f ! -path 'reports/artifacts.sha256' -print0 \
 | sort -z | xargs -0 sha256sum > "$sha_tmp"
mv "$sha_tmp" reports/artifacts.sha256
! grep -qE '(^|[[:space:]])reports/artifacts\.sha256$' reports/artifacts.sha256
sha256sum -c reports/artifacts.sha256
python - <<'PY'
import glob,json
for p in sorted(glob.glob("runs/*/manifest.json")):
 x=json.load(open(p));print(x["run_id"],x["status"],x["returncode"])
PY
```

### 预期观测（估算/示例，不是实测）

publication checker 输出 `EVIDENCE_CONSISTENT_PENDING_HUMAN_MODEL_DECISION`；每个 run 均有状态/命令/hashes，报告可反查 arm→checkpoint marker/hash→eval，并保留 `last` 基线。`reports/artifacts.sha256` 不含自身且 `sha256sum -c` 通过。没有预填 PASS 或效果数字。

### 验收条件

publication consistency gate 先过，再由人给唯一结论标签；FAIL-MODEL/FAIL-SYSTEM分开；正式同预算 A/B、1/4/8、profile、continuous-vs-resume、bad-checkpoint证据齐；SHA清单不自引用；6–10页等价信息量；30–45分钟闭卷讲通端到端；所有公司资产仍在边界内。

### 若失败，按什么顺序查

lineage → 数值reference → 多卡等价 → checkpoint/recovery →独立eval → 公平性 → 报告。无法补证据则INCONCLUSIVE，不用预计值补洞。

### 当日证据清单

八份报告、artifact SHA、答辩录分（A0/A1）、错误清单、下一阶段回归日期。

## 2. FinExec 备选迁移说明

若周一选择 FinExec，不能直接使用本篇 Mini-WAM config/runner。应复制 Week12 的 `evaluator.py/prepare.py/train.py/evaluate.py`、固定 processed/model hashes，并先为 wrapper 增加且测试 `kind=finexec` 的 train/resume/eval 参数映射；在该适配代码完整进入 freeze manifest 前，不得执行。本发布版本故意只提供一个完整载荷，避免“二选一”变成两个半成品。

## 3. 止损与恢复总表

- frozen mismatch：拒绝；变更形成v2实验。
- OOM：先micro/图像/latent；作为新config，不覆盖原run。
- NaN/Inf：固定首个sample/step，FP32重放；全rank一致停止。
- DDP hang：2-step、每rank traceback、`NCCL_DEBUG=INFO`；按公司流程恢复。
- profile工具缺失：标`ENV-BLOCKED`，可用已有PyTorch profiler，但不得安装新栈。
- checkpoint缺marker/sha错：拒绝并回到上一完整点。
- 模型门未过而系统门全过：`FAIL-MODEL`；系统任一关键门坏：`FAIL-SYSTEM`；证据不足：`INCONCLUSIVE`。
