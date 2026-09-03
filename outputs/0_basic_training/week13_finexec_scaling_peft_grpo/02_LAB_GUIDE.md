# Week 13 实践篇：1.5B FSDP 主路径、4B LoRA 与 GRPO gate

> 所有命令均待执行；示例/阈值不是实测。公司数据、模型、checkpoint、日志、trace、指标与拓扑信息不得导出。本任务不构成投资建议，不连接交易或外部 API。

## 0. 下载前：只读探针、预算和跨周恢复

公司 V100/PyTorch 2.1 是主路径（用户自述待核验）；5070 Ti 只做 reward/LoRA 小测；H100 为独立可选环境。估算空间：1.5B snapshot 3–5GB，3 个 FSDP checkpoint 30–60GB，0.5B 对照 10GB，至少预留 80GB；4B stretch 另预留 15–30GB。先用 50 step 推算时间，禁止无上限长训。

```bash
which python; python --version; git --version; df -h .
python - <<'PY'
import shutil,torch
print({"torch":torch.__version__,"cuda":torch.version.cuda,"available":torch.cuda.is_available(),
       "gpus":torch.cuda.device_count(),"disk":shutil.disk_usage(".")})
if torch.cuda.is_available():
  for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i); print(i,p.name,p.major,p.minor,p.total_memory)
PY
nvidia-smi
nvidia-smi topo -m
```

复用当前环境，不创建 Conda/venv，不升级 torch/CUDA：

```bash
mkdir -p week13-scale/{configs,data,manifests,models,runs,tests,reports,third_party}
cd week13-scale
python -m pip freeze > manifests/pip-freeze.before.txt
python -m pip check | tee manifests/pip-check.before.txt
```

从同一批准工作区的 Week 12 恢复冻结输入；每个本地路径在引用前创建：

```bash
test -d ../week12-finexec || { echo '先执行 Week 12 Lab'; exit 2; }
test -f ../week12-finexec/data/processed/manifest.json
test -f ../week12-finexec/evaluator.py
test -f ../week12-finexec/train.py
mkdir -p inherited
cp ../week12-finexec/{evaluator.py,train.py,evaluate.py} inherited/
cp -a ../week12-finexec/data/processed data/
cp ../week12-finexec/data/manifests/processed.sha256 manifests/week12-processed.sha256
python ../week12-finexec/evaluator.py
(cd ../week12-finexec && python -m unittest -v tests/test_core.py)
(cd ../week12-finexec && sha256sum -c data/manifests/processed.sha256)
find inherited data/processed -type f -print0 | sort -z | xargs -0 sha256sum > manifests/inherited.sha256
```

若相对目录不存在，按 Week 12 的 fixed revisions 重建，不从聊天附件/个人网盘补文件。依赖主栈仍是 Transformers 4.46.3、Accelerate 1.1.1；版本不符停止。回滚交环境负责人按 `pip-freeze.before.txt`/基础镜像处理。

## 1. 模型与可选依赖获取

| 对象 | 官方 ID | revision/tag | 必需文件 | 许可 | 完整性 |
|---|---|---|---|---|---|
| 主模型 | `Qwen/Qwen2.5-1.5B` | `8faed761d45a263340a0528343f099c05c9a4323` | config/tokenizer/safetensors/LICENSE | Apache-2.0 | revision、SHA、offline config/model load |
| 0.5B 对照 | Week 12 snapshot | `060db6499f32faf8b98477b0a26969ef7d8b9987` | 同 Week 12 | Apache-2.0 | 继承 SHA 重验 |
| 4B stretch | `Qwen/Qwen3-4B-Base` | `906bfd4b4dc7f14ee4320094d8b41684abff8539` | 同上 | Apache-2.0 | 先 `AutoConfig` 兼容 gate；失败不下载大权重 |
| PEFT stretch | `peft v0.13.2` | `34d479632d63a7e29c4b75202c9eef94335ceb14` | 获批 wheel/source | Apache-2.0 | version/signature |
| GRPO stretch | `trl v0.15.2` | `77f9e82ff963b0a82581e78a21554fc90f96b843` | 获批 wheel/source/docs | Apache-2.0 | 独立 dry-run；不污染主栈 |

获批联网下载主模型：

```bash
cd week13-scale
mkdir -p models/Qwen2.5-1.5B
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen2.5-1.5B",revision="8faed761d45a263340a0528343f099c05c9a4323",
 local_dir="models/Qwen2.5-1.5B",allow_patterns=["*.json","*.safetensors","*.txt","*.model","LICENSE","README.md"])
from pathlib import Path
Path("models/Qwen2.5-1.5B/SOURCE_REVISION").write_text(
 "Qwen/Qwen2.5-1.5B@8faed761d45a263340a0528343f099c05c9a4323\n",encoding="utf-8")
PY
test -f models/Qwen2.5-1.5B/model.safetensors.index.json -o -f models/Qwen2.5-1.5B/model.safetensors
find models/Qwen2.5-1.5B -type f -print0 | sort -z | xargs -0 sha256sum > manifests/qwen15b.sha256
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained("models/Qwen2.5-1.5B",local_files_only=True)
print({"type":c.model_type,"layers":c.num_hidden_layers,"hidden":c.hidden_size,"vocab":c.vocab_size})
assert c.model_type=="qwen2"
PY
```

离线只接受管理员提供的相同 snapshot+SHA+许可。Qwen3-4B 先让负责人只提供 config/tokenizer 做兼容 probe；`AutoConfig` 若报 unknown model type，记录 `ENV-BLOCKED`，不升级主环境。

## 2. 工程树与文件创建

先保留 §0 已创建的目录；不要用 `touch` 制造看似存在却没有实现的空文件。下面每个标题都会明确要求把紧随其后的完整代码块保存到对应路径，全部保存后才执行 `py_compile`。

```text
week13-scale/
  inherited/                 # Week 12 冻结代码
  data/processed/            # Week 12 supported subset
  train_scale.py             # 1.5B FSDP1 + Trainer
  audit_checkpoint.py        # checkpoint 完整性 marker
  train_lora.py              # generic 4B FP16 LoRA stretch
  compare_resume.py          # next-update 与最终 tensor state 对比
  rewards.py                 # 分项 deterministic reward
  grpo_smoke.py              # TRL 0.15.2 gated smoke
  tests/test_scale.py
  configs/*.json
  manifests/ models/ runs/ reports/
```

### 2.1 FSDP 配置

将紧随其后的代码块原样保存为 `configs/fsdp_15b.json`：

```json
{"model_dir":"models/Qwen2.5-1.5B","model_id":"Qwen/Qwen2.5-1.5B",
 "model_revision":"8faed761d45a263340a0528343f099c05c9a4323",
 "train_file":"data/processed/train.jsonl","data_manifest":"data/processed/manifest.json","max_length":512,
 "per_device_batch":1,"grad_accum":2,"max_steps":200,"learning_rate":0.00001,"warmup_steps":20,
 "weight_decay":0.01,"lr_scheduler_type":"linear","max_grad_norm":1.0,
 "save_steps":50,"logging_steps":1,"seed":1307,"full_determinism":true,"fp16":true,
 "fsdp":"full_shard auto_wrap","fsdp_state_dict_type":"SHARDED_STATE_DICT",
 "fsdp_config":{"transformer_layer_cls_to_wrap":["Qwen2DecoderLayer"],
 "backward_prefetch":"BACKWARD_PRE","forward_prefetch":false,"use_orig_params":true,
 "sync_module_states":true,"cpu_ram_efficient_loading":false}}
```

四卡时 global batch=`1×2×4=8`，与 Week 12 单卡 `1×8×1=8` 对照。八卡时 CLI 覆盖 `grad_accum=1`。

### 2.2 `train_scale.py`：完整 FSDP 主入口

将紧随其后的代码块原样保存为 `train_scale.py`：

```python
# file: train_scale.py
import argparse,hashlib,importlib.metadata,json,os,subprocess,sys
from pathlib import Path
import torch
from torch.utils.data import SequentialSampler
from transformers import AutoModelForCausalLM,AutoTokenizer,Trainer,TrainerCallback,TrainingArguments,set_seed
sys.path.insert(0,str(Path(__file__).parent/"inherited"))
from train import Records,Collate,StopAfter,bind_data,tokenizer_probe,tree_sha

def sha(p):
  h=hashlib.sha256()
  with Path(p).open("rb") as f:
    for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
  return h.hexdigest()

class IdCollate(Collate):
  def __call__(self,rows):
    ids=[x["id"] for x in rows]; out=super().__call__(rows);out["_sample_ids"]=ids;return out

class LineageOnSave(TrainerCallback):
  """Make every fully saved checkpoint independently auditable, including after a later crash."""
  def __init__(self,out,effective,rank0):
    self.out=Path(out);self.payload=json.dumps(effective,indent=2,ensure_ascii=False);self.rank0=rank0
  def on_save(self,args,state,control,**kwargs):
    distributed=torch.distributed.is_available() and torch.distributed.is_initialized()
    if distributed:torch.distributed.barrier()
    error=[None]
    if self.rank0:
      try:
        ckpt=self.out/f"checkpoint-{state.global_step}"
        if not ckpt.is_dir():raise RuntimeError("on_save checkpoint directory missing:"+str(ckpt))
        tmp=ckpt/"RESUME_CONTRACT.json.tmp";dst=ckpt/"RESUME_CONTRACT.json"
        tmp.write_text(self.payload,encoding="utf-8");os.replace(tmp,dst)
      except Exception as e:error[0]=repr(e)
    if distributed:torch.distributed.broadcast_object_list(error,src=0)
    if error[0] is not None:raise RuntimeError("checkpoint lineage publish failed:"+error[0])
    if distributed:torch.distributed.barrier()
    return control

class FSDPAuditTrainer(Trainer):
  """Sequential stream plus one audit record for every attempted optimizer update."""
  def __init__(self,*args,**kwargs):
    self._pending_ids=[];self.audit_attempted=0;self.audit_successful=0;self._last_scale=None
    super().__init__(*args,**kwargs)
  def _get_train_sampler(self):return SequentialSampler(self.train_dataset)
  def compute_loss(self,model,inputs,return_outputs=False,**kwargs):
    self._pending_ids.extend(map(str,inputs.pop("_sample_ids")))
    outputs=model(**inputs);loss=outputs["loss"] if isinstance(outputs,dict) else outputs.loss
    return (loss,outputs) if return_outputs else loss
  def log(self,logs,*args,**kwargs):
    update_log="loss" in logs and self.state.global_step>0
    skipped=False
    if update_log:
      skipped=bool(getattr(getattr(self,"accelerator",None),"optimizer_step_was_skipped",False))
      distributed=torch.distributed.is_available() and torch.distributed.is_initialized()
      if distributed:
        lo=torch.tensor(int(skipped),device=self.args.device);hi=lo.clone()
        torch.distributed.all_reduce(lo,op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(hi,op=torch.distributed.ReduceOp.MAX)
        if lo.item()!=hi.item():raise RuntimeError("AMP skip decision diverged across ranks")
        skipped=bool(hi.item());parts=[None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(parts,list(self._pending_ids));sample_ids=sum(parts,[])
      else:sample_ids=list(self._pending_ids)
      expected=self.args.per_device_train_batch_size*self.args.gradient_accumulation_steps*self.args.world_size
      if len(sample_ids)!=expected or len(sample_ids)!=len(set(sample_ids)):
        raise RuntimeError(f"sample stream violation: got {len(sample_ids)}, expected {expected}, unique={len(set(sample_ids))}")
      scaler=getattr(getattr(self,"accelerator",None),"scaler",None)
      scale=float(scaler.get_scale()) if scaler is not None else None
      self.audit_attempted=int(self.state.global_step)
      self.audit_successful=self.audit_attempted-int(skipped)
      logs={**logs,"attempted_update":self.audit_attempted,"successful_update":self.audit_successful,
            "optimizer_step_was_skipped":skipped,"sample_ids":sample_ids,"loss_scale":scale}
      self._pending_ids.clear();self._last_scale=scale
    result=super().log(logs,*args,**kwargs)
    if update_log and skipped:
      raise FloatingPointError("AMP skipped an update; attempted advanced but successful did not; abort before publish")
    return result

def main():
  ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--output-dir",required=True)
  ap.add_argument("--max-steps",type=int); ap.add_argument("--grad-accum",type=int); ap.add_argument("--resume"); ap.add_argument("--stop-after",type=int)
  a=ap.parse_args(); c=json.loads(Path(a.config).read_text(encoding="utf-8")); set_seed(c["seed"])
  if importlib.metadata.version("transformers")!="4.46.3":raise RuntimeError("this path is frozen to transformers 4.46.3")
  if c["logging_steps"]!=1:raise ValueError("every update must be logged for AMP/sample-stream audit")
  os.environ["FSDP_STATE_DICT_TYPE"]=c["fsdp_state_dict_type"]
  marker=Path(c["model_dir"])/"SOURCE_REVISION";expected=f"{c['model_id']}@{c['model_revision']}"
  if not marker.is_file() or marker.read_text(encoding="utf-8").strip()!=expected:raise RuntimeError("base SOURCE_REVISION mismatch")
  tok=AutoTokenizer.from_pretrained(c["model_dir"],local_files_only=True,use_fast=True)
  if tok.eos_token_id is None:raise RuntimeError("EOS required")
  if tok.pad_token_id is None:tok.pad_token=tok.eos_token
  train_sha,data_manifest_sha=bind_data(c["train_file"],c["data_manifest"],tok)
  ds=Records(c["train_file"],tok,c["max_length"])
  g=torch.Generator().manual_seed(c["seed"]);order=torch.randperm(len(ds),generator=g).tolist()
  ds.rows=[ds.rows[i] for i in order]
  stream_sha=hashlib.sha256("\n".join(x["id"] for x in ds.rows).encode()).hexdigest()
  model=AutoModelForCausalLM.from_pretrained(c["model_dir"],local_files_only=True,torch_dtype=torch.float32,
                                             attn_implementation="eager")
  model.config.use_cache=False; total=sum(p.numel() for p in model.parameters())
  wrapped=sum(type(m).__name__=="Qwen2DecoderLayer" for m in model.modules())
  if wrapped!=model.config.num_hidden_layers: raise RuntimeError(f"wrap class mismatch {wrapped} vs {model.config.num_hidden_layers}")
  steps=a.max_steps if a.max_steps is not None else c["max_steps"]
  accum=a.grad_accum if a.grad_accum is not None else c["grad_accum"]
  world=int(os.getenv("WORLD_SIZE","1"));global_batch=c["per_device_batch"]*accum*world
  if steps<1 or accum<1:raise ValueError("steps and grad-accum must be positive")
  if len(ds)<steps*global_batch:raise RuntimeError("eligible train rows cannot supply the fixed no-replacement stream")
  if steps>1 and steps%c["save_steps"]!=0: raise ValueError("multi-step runs must end on complete save boundary")
  if a.stop_after and (not 0<a.stop_after<steps or a.stop_after%c["save_steps"]!=0): raise ValueError("stop-after must be an interior save boundary")
  args=TrainingArguments(output_dir=a.output_dir,max_steps=steps,per_device_train_batch_size=c["per_device_batch"],
    gradient_accumulation_steps=accum,learning_rate=c["learning_rate"],warmup_steps=c["warmup_steps"],
    weight_decay=c["weight_decay"],lr_scheduler_type=c["lr_scheduler_type"],max_grad_norm=c["max_grad_norm"],
    logging_steps=c["logging_steps"],save_steps=min(c["save_steps"],steps),save_total_limit=5,
    fp16=True,bf16=False,report_to=[],remove_unused_columns=False,dataloader_drop_last=True,
    dataloader_num_workers=0,seed=c["seed"],data_seed=c["seed"],save_safetensors=True,
    full_determinism=c["full_determinism"],fsdp=c["fsdp"],fsdp_config=c["fsdp_config"])
  out=Path(a.output_dir)
  conflict=out.exists() and any(out.iterdir());rank0=int(os.getenv("RANK","0"))==0
  if conflict and not a.resume: raise FileExistsError("non-resume run refuses non-empty output-dir")
  out.mkdir(parents=True,exist_ok=True)
  effective={"config_sha":sha(a.config),"code_sha":sha(__file__),"inherited_train_sha":sha("inherited/train.py"),
    "evaluator_sha":sha("inherited/evaluator.py"),"train_sha":train_sha,"data_manifest_sha":data_manifest_sha,
    "model_id":c["model_id"],"model_revision":c["model_revision"],"model_tree_sha":tree_sha(c["model_dir"]),
    "tokenizer_probe_sha":tokenizer_probe(tok),"max_length":c["max_length"],"per_device_batch":c["per_device_batch"],
    "gradient_accumulation_steps":accum,"global_batch":global_batch,"max_steps":steps,
    "optimizer":{"name":"AdamW","learning_rate":c["learning_rate"],"weight_decay":c["weight_decay"],"max_grad_norm":c["max_grad_norm"]},
    "scheduler":{"type":c["lr_scheduler_type"],"warmup_steps":c["warmup_steps"],"total_steps":steps},
    "seed":c["seed"],"world_size":world,"precision":"fp16","attention":"eager","fsdp":c["fsdp"],
    "fsdp_state_dict_type":c["fsdp_state_dict_type"],"fsdp_config":c["fsdp_config"],
    "sample_stream":{"kind":"seeded_permutation_then_sequential_no_replacement","sha256":stream_sha,"required_rows":steps*global_batch}}
  if a.resume:
    rp=Path(a.resume).resolve()
    if not rp.is_dir() or rp.parent!=out.resolve():raise RuntimeError("resume checkpoint must be checkpoint-* inside the same output-dir")
    subprocess.run([sys.executable,"audit_checkpoint.py","--checkpoint",str(rp),"--verify"],check=True)
    previous=json.loads((out/"run_manifest.json").read_text(encoding="utf-8"))
    if previous.get("effective_contract")!=effective:raise RuntimeError("effective resume contract mismatch")
  manifest={"status":"STARTED_NOT_VALIDATED","config":c,"effective_contract":effective,"total_params":total,
    "wrap_matches":wrapped,"stop_after":a.stop_after,"resume":a.resume,
    "resume_tree_sha":tree_sha(a.resume) if a.resume else None,
    "update_clock":"attempted advances on every Trainer optimizer attempt; successful advances only when no rank skips"}
  if rank0:(out/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
  callbacks=[LineageOnSave(out,effective,rank0)]
  if a.stop_after:callbacks.append(StopAfter(a.stop_after))
  trainer=FSDPAuditTrainer(model=model,args=args,train_dataset=ds,data_collator=IdCollate(tok.pad_token_id),callbacks=callbacks)
  try:result=trainer.train(resume_from_checkpoint=a.resume)
  except Exception as e:
    if trainer.is_world_process_zero():
      manifest.update(status="FAIL_SYSTEM",attempted_steps=trainer.audit_attempted,
        successful_steps=trainer.audit_successful,error=repr(e))
      (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    raise
  interrupted=bool(a.stop_after and trainer.state.global_step>=a.stop_after)
  if not interrupted and trainer.state.global_step!=steps:raise RuntimeError("training ended before max_steps")
  lineage=json.dumps(effective,indent=2,ensure_ascii=False)
  if trainer.is_world_process_zero():
    for ckpt in out.glob("checkpoint-*"):
      if ckpt.is_dir() and not (ckpt/"RESUME_CONTRACT.json").is_file():
        raise RuntimeError("checkpoint returned without on-save lineage:"+str(ckpt))
  if torch.distributed.is_available() and torch.distributed.is_initialized():torch.distributed.barrier()
  trainer.save_state()
  manifest.update(attempted_steps=trainer.state.global_step,successful_steps=trainer.state.global_step)
  if interrupted:
    if trainer.is_world_process_zero():
      ckpt=out/f"checkpoint-{trainer.state.global_step}"
      if not ckpt.is_dir():raise RuntimeError("StopAfter did not publish its checkpoint")
      manifest.update(status="INTERRUPTED_FOR_RESUME",stop_checkpoint=str(ckpt))
      (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return
  if trainer.is_fsdp_enabled:
    plugin=trainer.accelerator.state.fsdp_plugin
    plugin.state_dict_config=None; plugin.optim_state_dict_config=None
    plugin.set_state_dict_type("FULL_STATE_DICT")
  trainer.save_model(out/"final_model")
  if trainer.is_world_process_zero():
    tok.save_pretrained(out/"final_model")
    (out/"final_model"/"SOURCE_REVISION").write_text(expected+"\n",encoding="utf-8")
    (out/"final_model"/"MODEL_LINEAGE.json").write_text(lineage,encoding="utf-8")
    (out/"train_metrics.json").write_text(json.dumps(result.metrics,indent=2),encoding="utf-8")
    manifest["status"]="COMPLETED_PENDING_CHECKPOINT_AUDIT_AND_EVAL"
    (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"status":manifest["status"],"params":total,"wrap":wrapped,**result.metrics}))
if __name__=="__main__":main()
```

Transformers 4.46.3 的 `fsdp_config` 使用无 `fsdp_` 前缀的键；`FSDP_STATE_DICT_TYPE` 则须在 Trainer/Accelerator 构造前显式设置。中间 checkpoint 使用 `SHARDED_STATE_DICT`；每次 `on_save` 都在保存完成后的两次 barrier 之间原子写 lineage，所以后来崩溃不会让更早 savepoint 全部失去恢复资格。训练结束后清空旧的 sharded config，再切换 `FULL_STATE_DICT`。`SequentialSampler` 消费一次固定 permutation，且入口要求本次 run 所需 global rows 足够，因此 200-step continuous/resume 共享同一无放回 sample stream。逐步日志中的 `attempted_update/successful_update/sample_ids/loss_scale` 是恢复证据；任何 AMP skip 立即失败，不把 Trainer `global_step` 静默冒充成功更新。rank0 保存/最终导出可能有 CPU RAM/磁盘峰值；若 export OOM，sharded resume checkpoint 仍可保留，但独立 eval 标 `EXPORT-BLOCKED`。

### 2.3 `audit_checkpoint.py`：完整发布检查

将紧随其后的代码块原样保存为 `audit_checkpoint.py`：

```python
# file: audit_checkpoint.py
import argparse,hashlib,json,os
from pathlib import Path
def sha(p):
  h=hashlib.sha256()
  with Path(p).open("rb") as f:
    for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
  return h.hexdigest()
def snapshot(p):
  return {str(x.relative_to(p)):sha(x) for x in sorted(p.rglob("*"))
          if x.is_file() and x.name not in {"COMPLETE.json","COMPLETE.json.tmp"}}
def inspect(p):
  files=[x for x in p.rglob("*") if x.is_file() and x.name not in {"COMPLETE.json","COMPLETE.json.tmp"}]
  names=[x.relative_to(p).as_posix() for x in files]
  lp=p/"RESUME_CONTRACT.json";contract=json.loads(lp.read_text(encoding="utf-8")) if lp.is_file() else {}
  world=int(contract.get("world_size",0));precision=contract.get("precision")
  rng=[n for n in names if Path(n).name.startswith("rng_state") and n.endswith((".pth",".pt"))]
  distcp=[n for n in names if n.endswith(".distcp")]
  required_lineage={"config_sha","code_sha","inherited_train_sha","evaluator_sha","train_sha",
    "data_manifest_sha","model_revision","model_tree_sha","tokenizer_probe_sha","optimizer",
    "scheduler","world_size","precision","fsdp_state_dict_type","sample_stream"}
  groups={
    "model":any("pytorch_model_fsdp" in n or Path(n).name.startswith(("model","pytorch_model")) for n in names),
    "optimizer":any("optimizer" in n.lower() for n in names),
    "scheduler":any(Path(n).name=="scheduler.pt" for n in names),
    "scaler":precision=="fp32" or any(Path(n).name=="scaler.pt" for n in names),
    "rng":world>0 and len(rng)>=world,
    "shards":any("pytorch_model_fsdp" in n for n in distcp) and any("optimizer" in n.lower() for n in distcp),
    "trainer":any(Path(n).name=="trainer_state.json" for n in names),
    "lineage":required_lineage.issubset(contract) and contract.get("fsdp_state_dict_type")=="SHARDED_STATE_DICT"}
  nonempty=bool(files) and all(x.stat().st_size>0 for x in files)
  return groups,nonempty,snapshot(p),contract
def verify(p,marker):
  if not marker.is_file():raise SystemExit("completion marker missing")
  doc=json.loads(marker.read_text(encoding="utf-8"));groups,nonempty,files,contract=inspect(p)
  ok=(doc.get("status")=="COMPLETE" and all(groups.values()) and nonempty and
      doc.get("groups")==groups and doc.get("files")==files and doc.get("lineage_sha256")==sha(p/"RESUME_CONTRACT.json"))
  if not ok:raise SystemExit("checkpoint marker/hash/state-group verification failed")
  return {"status":"PASS","groups":groups,"files":len(files),"lineage_sha256":doc["lineage_sha256"]}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",required=True);ap.add_argument("--write-marker",action="store_true");ap.add_argument("--verify",action="store_true");a=ap.parse_args()
 if a.write_marker==a.verify:raise SystemExit("choose exactly one of --write-marker or --verify")
 p=Path(a.checkpoint);marker=p/"COMPLETE.json"
 if not p.is_dir():raise SystemExit("checkpoint directory missing")
 if a.verify:report=verify(p,marker)
 else:
  if marker.exists():raise SystemExit("marker already exists; use --verify")
  groups,nonempty,files,contract=inspect(p)
  if not nonempty or not all(groups.values()):raise SystemExit("checkpoint state groups incomplete:"+json.dumps(groups))
  doc={"status":"COMPLETE","groups":groups,"files":files,"lineage_sha256":sha(p/"RESUME_CONTRACT.json")}
  tmp=p/"COMPLETE.json.tmp";tmp.write_text(json.dumps(doc,indent=2),encoding="utf-8");os.replace(tmp,marker)
  report=verify(p,marker)
 print(json.dumps(report))
if __name__=="__main__":main()
```

### 2.4 `compare_resume.py`：固定 next update 与最终 state

将紧随其后的代码块原样保存为 `compare_resume.py`：

```python
# file: compare_resume.py
import argparse,json
from pathlib import Path
from safetensors import safe_open

def history(run):
  rows=json.loads((Path(run)/"trainer_state.json").read_text(encoding="utf-8"))["log_history"]
  return {int(x["successful_update"]):x for x in rows if "successful_update" in x}
def tensors(root):
  out={}
  for p in sorted(Path(root).rglob("*.safetensors")):
    with safe_open(str(p),framework="pt",device="cpu") as f:
      for k in f.keys():
        if k in out:raise RuntimeError("duplicate tensor key:"+k)
        out[k]=p
  if not out:raise RuntimeError("no safetensors in "+str(root))
  return out
def main():
  ap=argparse.ArgumentParser();ap.add_argument("--continuous",required=True);ap.add_argument("--resumed",required=True)
  ap.add_argument("--next-step",type=int,required=True);ap.add_argument("--loss-rtol",type=float,default=.01)
  ap.add_argument("--state-atol",type=float,default=1e-5);a=ap.parse_args()
  ha,hb=history(a.continuous),history(a.resumed);x,y=ha[a.next_step],hb[a.next_step]
  if x["sample_ids"]!=y["sample_ids"]:raise RuntimeError("next-update sample IDs differ")
  if float(x["learning_rate"])!=float(y["learning_rate"]):raise RuntimeError("next-update LR differs")
  if x.get("loss_scale")!=y.get("loss_scale"):raise RuntimeError("next-update loss scale differs")
  rel=abs(float(x["loss"])-float(y["loss"]))/max(abs(float(x["loss"])),1e-12)
  if rel>=a.loss_rtol:raise RuntimeError(f"next loss relative error {rel} >= {a.loss_rtol}")
  ma=tensors(Path(a.continuous)/"final_model");mb=tensors(Path(a.resumed)/"final_model")
  if set(ma)!=set(mb):raise RuntimeError("final tensor key sets differ")
  max_abs=0.0
  for k in sorted(ma):
    with safe_open(str(ma[k]),framework="pt",device="cpu") as fa, safe_open(str(mb[k]),framework="pt",device="cpu") as fb:
      ta,tb=fa.get_tensor(k),fb.get_tensor(k)
    if ta.shape!=tb.shape:raise RuntimeError("shape mismatch:"+k)
    max_abs=max(max_abs,float((ta.float()-tb.float()).abs().max()))
  if max_abs>a.state_atol:raise RuntimeError(f"final state max_abs {max_abs} > {a.state_atol}")
  print(json.dumps({"status":"PASS","next_step":a.next_step,"next_loss_relative":rel,
                    "final_state_max_abs":max_abs,"tensor_keys":len(ma)}))
if __name__=="__main__":main()
```

### 2.5 LoRA stretch 配置、lineage 与 offline reload/eval

将紧随其后的代码块原样保存为 `configs/lora_4b.json`：

```json
{"model_dir":"models/Qwen3-4B-Base","model_id":"Qwen/Qwen3-4B-Base",
 "model_revision":"906bfd4b4dc7f14ee4320094d8b41684abff8539",
 "train_file":"data/processed/train.jsonl","eval_file":"data/processed/dev.jsonl",
 "data_manifest":"data/processed/manifest.json","max_length":256,"max_new_tokens":128,"eval_limit":16,
 "per_device_batch":1,"grad_accum":8,"max_steps":50,"learning_rate":0.0001,"seed":1311,
 "rank":16,"alpha":32,"dropout":0.05,"targets":["q_proj","k_proj","v_proj","o_proj"]}
```

将紧随其后的代码块原样保存为 `train_lora.py`：

```python
# file: train_lora.py
import argparse,gc,hashlib,importlib.metadata,json,sys
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM,AutoTokenizer,Trainer,TrainingArguments,set_seed
from peft import LoraConfig,PeftModel,get_peft_model,get_peft_model_state_dict
sys.path.insert(0,str(Path(__file__).parent/"inherited"))
from evaluator import METRIC_KEYS,score
from train import Records,Collate,AuditTrainer,tree_sha,tokenizer_probe
def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);ap.add_argument("--output-dir",required=True);a=ap.parse_args()
 c=json.loads(Path(a.config).read_text(encoding="utf-8"));set_seed(c["seed"]);out=Path(a.output_dir)
 if importlib.metadata.version("peft")!="0.13.2":raise RuntimeError("LoRA stretch is frozen to peft 0.13.2")
 if out.exists() and any(out.iterdir()):raise FileExistsError("LoRA output must be empty")
 out.mkdir(parents=True,exist_ok=True)
 marker=Path(c["model_dir"])/"SOURCE_REVISION";expected=f"{c['model_id']}@{c['model_revision']}"
 if not marker.is_file() or marker.read_text(encoding="utf-8").strip()!=expected:raise RuntimeError("base revision mismatch")
 base_tree=tree_sha(c["model_dir"])
 tok=AutoTokenizer.from_pretrained(c["model_dir"],local_files_only=True,use_fast=True)
 if tok.eos_token_id is None:raise RuntimeError("EOS required")
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 data_manifest=json.loads(Path(c["data_manifest"]).read_text(encoding="utf-8"))
 for p in (c["train_file"],c["eval_file"]):
  if data_manifest.get("artifacts_sha256",{}).get(Path(p).name)!=file_sha(p):raise RuntimeError("data manifest mismatch:"+p)
 base=AutoModelForCausalLM.from_pretrained(c["model_dir"],local_files_only=True,torch_dtype=torch.float16,
                                           attn_implementation="eager");base.config.use_cache=False
 model=get_peft_model(base,LoraConfig(r=c["rank"],lora_alpha=c["alpha"],lora_dropout=c["dropout"],
   target_modules=c["targets"],bias="none",task_type="CAUSAL_LM"))
 trainable=sum(p.numel() for p in model.parameters() if p.requires_grad); total=sum(p.numel() for p in model.parameters())
 trainable_rows=[{"name":n,"shape":list(p.shape),"dtype":str(p.dtype)} for n,p in model.named_parameters() if p.requires_grad]
 if not trainable_rows or any("lora_" not in x["name"] for x in trainable_rows):raise RuntimeError("unexpected trainable params")
 print(json.dumps({"full":total,"trainable":trainable,"ratio":trainable/total,"first":trainable_rows[:20]}))
 ds=Records(c["train_file"],tok,c["max_length"])
 ta=TrainingArguments(output_dir=a.output_dir,max_steps=c["max_steps"],per_device_train_batch_size=c["per_device_batch"],
   gradient_accumulation_steps=c["grad_accum"],learning_rate=c["learning_rate"],fp16=True,bf16=False,
   logging_steps=1,save_strategy="no",report_to=[],remove_unused_columns=False,dataloader_drop_last=True,seed=c["seed"],data_seed=c["seed"])
 tr=AuditTrainer(model=model,args=ta,train_dataset=ds,data_collator=Collate(tok.pad_token_id));result=tr.train()
 if tr.state.global_step!=c["max_steps"]:raise RuntimeError("LoRA did not complete requested successful updates")
 expected_adapter={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
 adapter=out/"adapter";tr.save_model(adapter);tok.save_pretrained(adapter)
 weight_file=adapter/"adapter_model.safetensors"
 if not weight_file.is_file():raise RuntimeError("strict reload path requires adapter_model.safetensors")
 with safe_open(str(weight_file),framework="pt",device="cpu") as f:
  disk_adapter={k:f.get_tensor(k) for k in f.keys()}
 if set(disk_adapter)!=set(expected_adapter):
  raise RuntimeError("saved adapter missing/unexpected keys:"+repr((sorted(set(expected_adapter)-set(disk_adapter)),sorted(set(disk_adapter)-set(expected_adapter)))))
 for k in expected_adapter:
  if disk_adapter[k].shape!=expected_adapter[k].shape or not torch.equal(disk_adapter[k],expected_adapter[k]):
   raise RuntimeError("saved adapter tensor mismatch:"+k)
 tensor_manifest={k:{"shape":list(v.shape),"dtype":str(v.dtype)} for k,v in sorted(disk_adapter.items())}
 lineage={"base_model_id":c["model_id"],"base_revision":c["model_revision"],"base_tree_sha256":base_tree,
  "train_config_sha256":file_sha(a.config),"train_data_sha256":file_sha(c["train_file"]),
  "eval_data_sha256":file_sha(c["eval_file"]),"data_manifest_sha256":file_sha(c["data_manifest"]),
  "evaluator_sha256":file_sha("inherited/evaluator.py"),"tokenizer_probe_sha256":tokenizer_probe(tok),
  "adapter_weights_sha256":file_sha(weight_file),
  "peft_version":importlib.metadata.version("peft"),"transformers_version":importlib.metadata.version("transformers"),
  "attempted_updates":tr.state.global_step,"successful_updates":tr.state.global_step}
 (adapter/"BASE_LINEAGE.json").write_text(json.dumps(lineage,indent=2),encoding="utf-8")
 (adapter/"TRAINABLE_MANIFEST.json").write_text(json.dumps({"full_params":total,"trainable_params":trainable,
   "ratio":trainable/total,"parameters":trainable_rows,"adapter_tensors":tensor_manifest},indent=2),encoding="utf-8")
 required=[adapter/"adapter_config.json",adapter/"adapter_model.safetensors",adapter/"TRAINABLE_MANIFEST.json",adapter/"BASE_LINEAGE.json"]
 if not all(p.is_file() and p.stat().st_size for p in required):raise RuntimeError("adapter/config/lineage incomplete")
 del tr,model,base;gc.collect()
 if torch.cuda.is_available():torch.cuda.empty_cache()
 saved=json.loads((adapter/"BASE_LINEAGE.json").read_text(encoding="utf-8"))
 if saved!=lineage or tree_sha(c["model_dir"])!=saved["base_tree_sha256"]:raise RuntimeError("offline reload lineage mismatch")
 reload_tok=AutoTokenizer.from_pretrained(adapter,local_files_only=True)
 if tokenizer_probe(reload_tok)!=saved["tokenizer_probe_sha256"]:raise RuntimeError("offline reload tokenizer probe mismatch")
 base2=AutoModelForCausalLM.from_pretrained(c["model_dir"],local_files_only=True,torch_dtype=torch.float16,
                                            attn_implementation="eager")
 reloaded=PeftModel.from_pretrained(base2,adapter,is_trainable=False);reloaded.eval()
 if set(reloaded.peft_config)!={"default"}:raise RuntimeError("unexpected adapter set")
 loaded_adapter=get_peft_model_state_dict(reloaded)
 if set(loaded_adapter)!=set(disk_adapter):raise RuntimeError("reloaded adapter missing/unexpected keys")
 for k in disk_adapter:
  if loaded_adapter[k].shape!=disk_adapter[k].shape or not torch.equal(loaded_adapter[k].float().cpu(),disk_adapter[k].float()):
   raise RuntimeError("reloaded adapter tensor mismatch:"+k)
 dev=torch.device("cuda" if torch.cuda.is_available() else "cpu");reloaded.to(dev)
 rows=[json.loads(x) for x in open(c["eval_file"],encoding="utf-8")][:c["eval_limit"]]
 counts={k:0.0 for k in METRIC_KEYS};preds=[]
 for row in rows:
  z=reload_tok(row["prompt"],return_tensors="pt",truncation=False)
  if z.input_ids.shape[1]>c["max_length"]:raise RuntimeError("eval prompt exceeds LoRA gate:"+row["id"])
  z={k:v.to(dev) for k,v in z.items()}
  with torch.no_grad():y=reloaded.generate(**z,do_sample=False,num_beams=1,max_new_tokens=c["max_new_tokens"],
    eos_token_id=reload_tok.eos_token_id,pad_token_id=reload_tok.pad_token_id)
  text=reload_tok.decode(y[0,z["input_ids"].shape[1]:],skip_special_tokens=True).strip()
  flags,error=score(text,row["gold"],row["allowed_evidence"])
  if tuple(flags)!=tuple(METRIC_KEYS):raise RuntimeError("Week12 evaluator flag contract drift")
  for k,v in flags.items():counts[k]+=float(v)
  preds.append({"id":row["id"],"flags":flags,"error":error,"text":text})
 summary={"status":"OFFLINE_RELOAD_EVALUATED_NOT_INTERPRETED","n":len(rows),
  "rates":{k:v/max(1,len(rows)) for k,v in counts.items()},"metric_keys":list(METRIC_KEYS),"lineage":lineage,
  "adapter_tree_sha256":tree_sha(adapter),"generation":{"do_sample":False,"num_beams":1,"max_new_tokens":c["max_new_tokens"]}}
 (out/"reload_predictions.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in preds),encoding="utf-8")
 (out/"reload_eval.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
 (out/"run_manifest.json").write_text(json.dumps({"status":"COMPLETED_PENDING_REVIEW",**lineage,
   "adapter_tree_sha256":summary["adapter_tree_sha256"]},indent=2),encoding="utf-8")
 print(json.dumps({"status":summary["status"],"n":len(rows),"metric_keys":list(METRIC_KEYS)}))
if __name__=="__main__":main()
```

该代码只有在 Qwen3 config 与已批准 Transformers/PEFT 均通过时才执行；不量化、不 merge。它保存 adapter/config/tokenizer/base lineage/trainable manifest；保存时将 safetensors 的完整 key/shape/value 与训练态 adapter state 对齐，释放训练对象后比较 tokenizer probe，再从磁盘加载固定 base+adapter并复核加载态 tensor，最后用 Week12 原 evaluator 跑冻结 dev 前 16 条。任何 missing/unexpected/shape/value 漂移均 fail closed。

### 2.6 `rewards.py`、GRPO smoke 与 adversarial tests

将紧随其后的代码块原样保存为 `rewards.py`：

```python
# file: rewards.py
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/"inherited"));from evaluator import score
KEYS=("json","schema","evidence_ids_valid","evidence","evidence_precision","evidence_recall",
      "evidence_f1","scale","program","execution","answer","consistent")
_OBSERVED={"count":0,"unique":set()}
def normalize_completion(c):return c[0].get("content","") if isinstance(c,list) else c
def reset_reward_observation():_OBSERVED["count"]=0;_OBSERVED["unique"].clear()
def reward_observation():
 return {"completion_count":_OBSERVED["count"],"unique_completions":len(_OBSERVED["unique"]),
         "unique_response_rate":len(_OBSERVED["unique"])/max(1,_OBSERVED["count"])}
def components(completion,gold_json,allowed_json):
 g=json.loads(gold_json); allowed=json.loads(allowed_json); flags,_=score(completion,g,allowed)
 if tuple(flags)!=KEYS:raise RuntimeError("Week12 evaluator flag contract drift")
 return {k:float(flags[k]) for k in KEYS}
def make_reward(key):
 def reward(completions,gold_json,allowed_json,**kwargs):
  out=[]
  for c,g,a in zip(completions,gold_json,allowed_json):
   c=normalize_completion(c)
   if key=="json":_OBSERVED["count"]+=1;_OBSERVED["unique"].add(c)
   out.append(components(c,g,a)[key])
  return out
 reward.__name__="reward_"+key;return reward
REWARDS=[make_reward(k) for k in KEYS]
```

将紧随其后的代码块原样保存为 `grpo_smoke.py`：

```python
# file: grpo_smoke.py
import argparse,json,os
from pathlib import Path
import torch
from datasets import Dataset
from trl import GRPOConfig,GRPOTrainer
from transformers import AutoTokenizer
from rewards import REWARDS,reset_reward_observation,reward_observation
class SafeGRPOTrainer(GRPOTrainer):
 def __init__(self,*args,**kwargs):
  self.audit_attempted=0;self.audit_successful=0;super().__init__(*args,**kwargs)
 def log(self,logs,*args,**kwargs):
  update=self.state.global_step>self.audit_attempted
  skipped=False
  if update:
   if self.state.global_step!=self.audit_attempted+1:raise RuntimeError("GRPO update audit clock jumped")
   skipped=bool(getattr(getattr(self,"accelerator",None),"optimizer_step_was_skipped",False))
   distributed=torch.distributed.is_available() and torch.distributed.is_initialized()
   if distributed:
    lo=torch.tensor(int(skipped),device=self.args.device);hi=lo.clone()
    torch.distributed.all_reduce(lo,op=torch.distributed.ReduceOp.MIN);torch.distributed.all_reduce(hi,op=torch.distributed.ReduceOp.MAX)
    if lo.item()!=hi.item():raise RuntimeError("GRPO AMP skip decision diverged across ranks")
    skipped=bool(hi.item())
   scaler=getattr(getattr(self,"accelerator",None),"scaler",None)
   self.audit_attempted=int(self.state.global_step);self.audit_successful=self.audit_attempted-int(skipped)
   logs={**logs,"attempted_update":self.audit_attempted,"successful_update":self.audit_successful,
         "optimizer_step_was_skipped":skipped,"loss_scale":float(scaler.get_scale()) if scaler is not None else None}
  result=super().log(logs,*args,**kwargs)
  if update and skipped:raise FloatingPointError("GRPO FP16 skipped an update; abort before publishing smoke evidence")
  return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--model",required=True);ap.add_argument("--data",required=True);ap.add_argument("--out",required=True)
 ap.add_argument("--steps",type=int,default=2);a=ap.parse_args()
 out=Path(a.out)
 if out.exists() and any(out.iterdir()):raise FileExistsError("GRPO output must be empty")
 rows=[json.loads(x) for x in open(a.data,encoding="utf-8")][:16]
 ds=Dataset.from_list([{"prompt":x["prompt"],"gold_json":json.dumps(x["gold"]),
                         "allowed_json":json.dumps(x["allowed_evidence"])} for x in rows])
 micro,generations,world=2,2,int(os.getenv("WORLD_SIZE","1"));generation_batch=micro*world
 if generation_batch%generations:raise RuntimeError("global generation batch must divide num_generations")
 if len(ds)<generation_batch:raise RuntimeError("not enough rows for one complete generation group")
 print(json.dumps({"gate":"PASS","world":world,"per_device_train_batch_size":micro,
                   "global_generation_batch":generation_batch,"num_generations":generations}))
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,padding_side="left",use_fast=True)
 if tok.eos_token_id is None:raise RuntimeError("GRPO tokenizer requires EOS")
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 reset_reward_observation()
 cfg=GRPOConfig(output_dir=a.out,max_steps=a.steps,per_device_train_batch_size=micro,gradient_accumulation_steps=1,
   num_generations=generations,max_completion_length=128,learning_rate=1e-6,beta=.02,fp16=True,bf16=False,
   model_init_kwargs={"local_files_only":True,"attn_implementation":"eager","torch_dtype":"float32"},
   use_vllm=False,logging_steps=1,save_strategy="no",report_to=[],remove_unused_columns=False,dataloader_drop_last=True)
 tr=SafeGRPOTrainer(model=a.model,reward_funcs=REWARDS,args=cfg,train_dataset=ds,processing_class=tok);result=tr.train()
 if tr.audit_attempted!=a.steps or tr.audit_successful!=a.steps:raise RuntimeError("GRPO attempted/successful update contract failed")
 tr.save_state();names=sorted({k for row in tr.state.log_history for k in row})
 if not any("reward" in k.lower() for k in names) or not any("length" in k.lower() for k in names) or not any("kl" in k.lower() for k in names):
  raise RuntimeError("GRPO required reward/KL/length metrics absent:"+repr(names))
 evidence={"status":"SMOKE_COMPLETED_PENDING_REVIEW","attempted_updates":tr.audit_attempted,
  "successful_updates":tr.audit_successful,"trainer_metrics":result.metrics,"logged_metric_names":names,
  "response_observation":reward_observation(),"offline":True,"attention":"eager"}
 out.mkdir(parents=True,exist_ok=True);(out/"smoke_metrics.json").write_text(json.dumps(evidence,indent=2),encoding="utf-8")
 print(json.dumps(evidence))
if __name__=="__main__":main()
```

将紧随其后的代码块原样保存为 `tests/test_scale.py`：

```python
# file: tests/test_scale.py
import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));from rewards import components
G=json.dumps({"evidence":["text_0"],"program":"add(2,3)","answer":5,"scale":"none"});A=json.dumps(["text_0"])
def out(ev,prog,ans):return json.dumps({"evidence":ev,"program":prog,"answer":ans,"scale":"none"})
class T(unittest.TestCase):
 def test_good(self):self.assertTrue(all(components(out(["text_0"],"add(2,3)",5),G,A).values()))
 def test_contract(self):self.assertEqual(tuple(components(out(["text_0"],"add(2,3)",5),G,A)),
  ("json","schema","evidence_ids_valid","evidence","evidence_precision","evidence_recall","evidence_f1",
   "scale","program","execution","answer","consistent"))
 def test_fake_id(self):
  x=components(out(["fake"],"add(2,3)",5),G,A);self.assertEqual(x["evidence_ids_valid"],0);self.assertEqual(x["evidence"],0)
 def test_answer_only(self):self.assertEqual(components(out(["text_0"],"divide(1,0)",5),G,A)["execution"],0)
 def test_inconsistent(self):self.assertEqual(components(out(["text_0"],"add(2,3)",9),G,A)["consistent"],0)
 def test_wrong_scale(self):
  x=json.dumps({"evidence":["text_0"],"program":"add(2,3)","answer":5,"scale":"percent"})
  self.assertEqual(components(x,G,A)["scale"],0)
 def test_prompt_copy(self):self.assertEqual(components("QUESTION add 2 3",G,A)["json"],0)
if __name__=="__main__":unittest.main()
```

## Day 1：冻结输入、预算与 wrap gate

### 为什么做

在启动 1.5B 前证明数据/evaluator 未漂移、模型可离线加载、wrap class 真正命中。

### 输入与前置检查

§0/1 PASS；四卡 0–3 是否真为优先组由现场 topo 决定，用户自述不能替代探针。

### 本日要创建/修改的文件

创建 §2 的 FSDP config、`train_scale.py`、`audit_checkpoint.py`；写 `reports/budget.md`。

### 实现（固定代码路径）

使用本文完整代码与 `models/Qwen2.5-1.5B` fixed snapshot；不从 main 引入脚本。

### 执行命令

```bash
cd week13-scale
python -m py_compile train_scale.py audit_checkpoint.py compare_resume.py train_lora.py rewards.py grpo_smoke.py tests/test_scale.py
python -m json.tool configs/fsdp_15b.json
python - <<'PY'
for p in (.5e9,1.5e9,4e9,8e9):print(p,{n:round(16*p/n/1024**3,2) for n in (1,4,8)})
PY
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python - <<'PY'
from transformers import AutoModelForCausalLM
m=AutoModelForCausalLM.from_pretrained("models/Qwen2.5-1.5B",local_files_only=True,attn_implementation="eager")
n=sum(type(x).__name__=="Qwen2DecoderLayer" for x in m.modules());print({"params":sum(p.numel() for p in m.parameters()),"wrap_blocks":n})
assert n==m.config.num_hidden_layers
PY
```

### 预期观测（估算/示例，不是实测）

打印 16P 纸面表与实际 config/参数/wrap 数。CPU load 可能需数 GB RAM；不是 GPU 峰值。

### 验收条件

Week12 SHA 重验；model_type qwen2；wrap 数=层数；预算覆盖 model states、activation/logits、all-gather、checkpoint/RAM；止损 `reserved>30.5GB`、连续非有限或预计卡时超限。

### 若失败，按什么顺序查

snapshot SHA → Transformers version → config model_type → class 名 → CPU RAM。wrap 不匹配时停止，不改成“wrap whole model”。

### 当日证据清单

inherited/model SHA、dependency freeze、budget、wrap stdout、选择卡组依据。

## Day 2：2 卡一阶 smoke、50-step 四卡与 checkpoint 审计

### 为什么做

逐级暴露 FSDP wrap、collective、FP16 与保存问题。

### 输入与前置检查

Day 1 PASS；每次只启动有界步数；共享机器已获调度批准。

### 本日要创建/修改的文件

只生成 `runs/fsdp_*`；不边跑边改 config。

### 实现（固定代码路径）

先 2 卡 50 step（global batch=4），它是 smoke，不与 0.5B任务结果比较；四卡才是 global batch=8 主设置。

### 执行命令

```bash
cd week13-scale
CUDA_VISIBLE_DEVICES=0,1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 torchrun --standalone --nproc_per_node=2 \
 train_scale.py --config configs/fsdp_15b.json --output-dir runs/fsdp_2g_one_step --max-steps 1 --grad-accum 1
CUDA_VISIBLE_DEVICES=0,1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 torchrun --standalone --nproc_per_node=2 \
 train_scale.py --config configs/fsdp_15b.json --output-dir runs/fsdp_2g_smoke --max-steps 50 --grad-accum 2
python audit_checkpoint.py --checkpoint runs/fsdp_2g_smoke/checkpoint-50 --write-marker
python audit_checkpoint.py --checkpoint runs/fsdp_2g_smoke/checkpoint-50 --verify
CUDA_VISIBLE_DEVICES=0,1,2,3 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 NCCL_DEBUG=INFO \
 torchrun --standalone --nproc_per_node=4 train_scale.py --config configs/fsdp_15b.json \
 --output-dir runs/fsdp_4g_smoke --max-steps 50 --grad-accum 2
python audit_checkpoint.py --checkpoint runs/fsdp_4g_smoke/checkpoint-50 --write-marker
python audit_checkpoint.py --checkpoint runs/fsdp_4g_smoke/checkpoint-50 --verify
python - <<'PY'
from pathlib import Path
import shutil
src=Path("runs/fsdp_4g_smoke/checkpoint-50");dst=Path("runs/bad_checkpoint_probe")
assert not dst.exists(),"bad probe output already exists"
shutil.copytree(src,dst)
candidates=sorted(p for p in dst.rglob("*") if p.is_file() and "optim" in p.as_posix().lower())
assert candidates,"optimizer file not found"
target=candidates[0];before=target.read_bytes();assert before,"optimizer file was already empty"
target.write_bytes(before[:-1]+bytes([before[-1]^0xFF]))
print({"corrupted":str(target),"bytes":len(before)})
PY
if python audit_checkpoint.py --checkpoint runs/bad_checkpoint_probe --verify; then
  echo 'ERROR: corrupted checkpoint was accepted'; exit 2
else
  echo 'EXPECTED: stale marker/hash rejected before resume'
fi
```

### 预期观测（估算/示例，不是实测）

1-step 完成单 batch 前反向；随后每 rank 初始化、wrap count 一致；50-step 日志逐步含 `attempted_update/successful_update/sample_ids/loss/loss_scale`，训练汇总另含 runtime/samples_per_second；checkpoint audit 的 model/optimizer/scheduler/scaler/RNG/shards/trainer/lineage 全为 true。时间/显存必须现场记录。

### 验收条件

1-step 前反向和50 successful steps均完成；无 NaN/Inf；本路径任一 skip 即失败；peak reserved<30.5GB；checkpoint marker 最后创建且随即 `--verify`；破坏前先定位到的 optimizer 文件被篡改后，其旧 marker/hash 被拒；final full export 可离线列出或明确 `EXPORT-BLOCKED`。

### 若失败，按什么顺序查

单 rank exception → wrap/class → `nvidia-smi` 峰值 → all-gather/save 时点 → NCCL rank日志。OOM 先缩 max_length 512→256形成新 config；hang 用2-step复现，不永久禁用 P2P。

### 当日证据清单

2/4卡 manifest、rank logs、memory、loss scale、checkpoint文件/SHA/marker、NCCL摘要（仅公司内）。

## Day 3：200-step 主跑、同 world-size resume 与独立 eval

### 为什么做

完成核心交付并证明恢复/评测，而非停在 50-step 可启动状态。

### 输入与前置检查

Day 2 四卡 PASS；按实测 50-step runtime 计算 200-step 上界，磁盘足够。

### 本日要创建/修改的文件

`runs/fsdp_4g_continuous`、`runs/fsdp_4g_resume`、`reports/scale_table.md`。

### 实现（固定代码路径）

同 world-size、同 200-step scheduler 恢复：一条连续跑到 200；另一条从一开始就配置 200，但在 100 主动保存退出，再在同一 output 目录恢复。world-size 改变不在本周默认保证内。评测使用 full `final_model`；若 export 被阻，则先解决/记录，不把训练 loss 代替 eval。

### 执行命令

```bash
cd week13-scale
CUDA_VISIBLE_DEVICES=0,1,2,3 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 torchrun --standalone --nproc_per_node=4 \
 train_scale.py --config configs/fsdp_15b.json --output-dir runs/fsdp_4g_continuous --max-steps 200 --grad-accum 2
python audit_checkpoint.py --checkpoint runs/fsdp_4g_continuous/checkpoint-200 --write-marker
python audit_checkpoint.py --checkpoint runs/fsdp_4g_continuous/checkpoint-200 --verify
CUDA_VISIBLE_DEVICES=0,1,2,3 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 torchrun --standalone --nproc_per_node=4 \
 train_scale.py --config configs/fsdp_15b.json --output-dir runs/fsdp_4g_resume --max-steps 200 --grad-accum 2 --stop-after 100
python audit_checkpoint.py --checkpoint runs/fsdp_4g_resume/checkpoint-100 --write-marker
python audit_checkpoint.py --checkpoint runs/fsdp_4g_resume/checkpoint-100 --verify
CUDA_VISIBLE_DEVICES=0,1,2,3 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 torchrun --standalone --nproc_per_node=4 \
 train_scale.py --config configs/fsdp_15b.json --output-dir runs/fsdp_4g_resume --max-steps 200 --grad-accum 2 \
 --resume runs/fsdp_4g_resume/checkpoint-100
python audit_checkpoint.py --checkpoint runs/fsdp_4g_resume/checkpoint-200 --write-marker
python audit_checkpoint.py --checkpoint runs/fsdp_4g_resume/checkpoint-200 --verify
CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python inherited/evaluate.py \
 --model runs/fsdp_4g_resume/final_model --tokenizer models/Qwen2.5-1.5B \
 --data data/processed/dev.jsonl --manifest data/processed/manifest.json \
 --out runs/fsdp_15b_dev --max-input 512 --max-new 128
python compare_resume.py --continuous runs/fsdp_4g_continuous --resumed runs/fsdp_4g_resume \
 --next-step 101 --loss-rtol 0.01 --state-atol 1e-5
```

### 预期观测（估算/示例，不是实测）

continuous 与 resume 都使用同一条 200-step LR schedule 和同一 fixed permutation；resume 从 successful step 100 到 200。对比器先核对 step101 的全局 sample IDs、LR、loss scale、next loss，再流式对比两个 `final_model` 的 tensor key/shape/value；独立 eval 生成完整 Week12 waterfall。1.5B FP16 inference 理论可单卡，但仍由实机 memory gate 决定。

### 验收条件

200 successful updates 且 attempted=successful；global batch8；完整 checkpoint/marker；resume step101 sample IDs/LR/scale一致、next-loss相对差<1%、终态 max-abs≤1e-5；dev 与唯一 full export、eligibility manifest关联；0.5B/1.5B使用同一 eligible dev JSONL、ID顺序、evaluator及 generation。

### 若失败，按什么顺序查

marker/world-size → optimizer/scheduler/scaler/RNG → final export文件 → tokenizer base revision → eval input length。导出 OOM 时用 CPU offload full-state方案单独验证，不能从不完整 shard直接 AutoModel load。

### 当日证据清单

step100/200 manifests/checkpoints、resume日志、full export SHA、dev metrics/predictions、0.5B对照路径。

## Day 4：LoRA 或 GRPO stretch（主路径未过则跳过）

### 为什么做

比较“分片全参”与“冻结 base 只训 adapter”，并仅在 RL 就绪时验证 reward 管线。

### 输入与前置检查

Day 3 完整 PASS。LoRA：Qwen3 config、许可、PEFT signature通过。GRPO：Week12 gold evaluator=100% supported subset、SFT valid JSON达到预注册门、adversarial tests全过，且获批独立 wheel set。

### 本日要创建/修改的文件

创建 §2.5–2.6 文件/config；可选下载固定 Qwen3 snapshot和 detached TRL tag，均先审计。未通过 gate 不创建虚假 run。

### 实现（固定代码/官方路径）

LoRA 使用 `train_lora.py`，无量化；GRPO 使用本文 `grpo_smoke.py` + TRL v0.15.2，`use_vllm=False`、2 step。先运行 signature：

### 执行命令

```bash
cd week13-scale
python -m unittest -v tests/test_scale.py
```

若本日选择 LoRA，只运行下面这一块；`set -e` 保证 config gate 失败后不会继续下载权重：

```bash
set -euo pipefail
cd week13-scale
python - <<'PY'
import importlib.metadata
assert importlib.metadata.version("peft")=="0.13.2"
print({"peft":"0.13.2","gate":"PASS"})
PY
mkdir -p models/Qwen3-4B-Base
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen3-4B-Base",revision="906bfd4b4dc7f14ee4320094d8b41684abff8539",
 local_dir="models/Qwen3-4B-Base",allow_patterns=["*.json","*.txt","*.model","LICENSE","README.md"])
PY
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python - <<'PY'
from transformers import AutoConfig
c=AutoConfig.from_pretrained("models/Qwen3-4B-Base",local_files_only=True);print(c.model_type)
PY
# 上条通过且下载获批时，补齐固定 safetensors：
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen3-4B-Base",revision="906bfd4b4dc7f14ee4320094d8b41684abff8539",
 local_dir="models/Qwen3-4B-Base",allow_patterns=["*.json","*.safetensors","*.txt","*.model","LICENSE","README.md"])
from pathlib import Path
Path("models/Qwen3-4B-Base/SOURCE_REVISION").write_text(
 "Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539\n",encoding="utf-8")
PY
find models/Qwen3-4B-Base -type f -print0 | sort -z | xargs -0 sha256sum > manifests/qwen3-4b.sha256
CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python train_lora.py \
 --config configs/lora_4b.json --output-dir runs/lora_4b_smoke
```

若本日选择 GRPO，只运行下面这一块；本机必须已由管理员提供 TRL v0.15.2 的批准依赖集：

```bash
set -euo pipefail
cd week13-scale
python - <<'PY'
import inspect
import importlib.metadata
from trl import GRPOConfig,GRPOTrainer
assert importlib.metadata.version("trl")=="0.15.2"
print(inspect.signature(GRPOConfig));print(inspect.signature(GRPOTrainer))
for k in ("num_generations","max_completion_length","use_vllm","model_init_kwargs"):
 assert k in inspect.signature(GRPOConfig).parameters,k
assert "processing_class" in inspect.signature(GRPOTrainer).parameters
world,micro,generations=1,2,2
assert (world*micro)%generations==0
print({"grpo_batch_gate":"PASS","global_generation_batch":world*micro,"num_generations":generations})
PY
test -d ../week12-finexec/runs/S_base_to_sft/final_model
CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python grpo_smoke.py \
 --model ../week12-finexec/runs/S_base_to_sft/final_model \
 --data data/processed/train.jsonl --out runs/grpo_2step --steps 2
```

### 预期观测（估算/示例，不是实测）

LoRA 打印 full/trainable/ratio/module名单，落盘 adapter/config/tokenizer/lineage/trainable manifest，并在释放训练对象后离线 reload/eval；GRPO 启动前打印 `global_generation_batch=2,num_generations=2` 的整除 gate，并记录 Week12 全部 flags。2 step 只验证管线，不能证明提升。

### 验收条件

LoRA 只有 `lora_` 参数可训练，adapter/config/tokenizer与 base revision/tree SHA、训练 config/data/evaluator及完整 trainable manifest绑定，offline reload/eval 产生同一组 `METRIC_KEYS`；GRPO 无 API/vLLM、合法 `2/2` generation batch、reward全分项、KL/长度/unique可观察、adversarial拒绝。任一依赖 gate 失败即 `ENV-BLOCKED/RL-NOT-READY`。

### 若失败，按什么顺序查

版本/signature → model_type → target module命中 → trainable params → reward单测 → local generation内存。禁止为 stretch升级主 torch或跳过安全 evaluator。

### 当日证据清单

gate checklist、版本/source SHA、LoRA trainable manifest或GRPO reward分项、明确 SKIPPED 原因。

## Day 5：系统/任务同表与决策

### 为什么做

决定放大收益是否抵成本，以及下一步投模型、数据/evaluator还是继续SFT。

### 输入与前置检查

0.5B与1.5B至少各有同协议dev评测；缺 seed/方差则只能 INCONCLUSIVE。

### 本日要创建/修改的文件

`reports/scale_table.md`、`reports/decision.md`、`reports/artifacts.sha256`。

### 实现（固定路径）

表中每个数字引用 metrics/manifest/checkpoint；缺失写 `NOT-RUN`。不同 tokenizer不比 raw loss。

### 执行命令

```bash
cd week13-scale
find runs -name run_manifest.json -o -name train_metrics.json -o -name metrics.json | sort
find configs data/processed models/Qwen2.5-1.5B runs -type f -print0 | sort -z | xargs -0 sha256sum \
 > reports/artifacts.sha256
python - <<'PY'
import glob,json
for p in sorted(glob.glob("runs/**/metrics.json",recursive=True)):
 try: print(p,json.load(open(p)))
 except Exception as e: print(p,"INVALID",e)
PY
```

### 预期观测（估算/示例，不是实测）

同表至少含 params/trainable/cards/global batch/target tokens/peak GB/tokens-s/valid program/execution/evidence F1/checkpoint bytes。

### 验收条件

一个且仅一个决策标签；解释预算与实测差异、FSDP通信/导出成本、任务指标是否超过方差；GRPO/LoRA未做也如实列出。

### 若失败，按什么顺序查

lineage缺失 → generation不一致 → evaluator版本不同 → target tokens不公平 → 方差不足。无法补齐时标 INCONCLUSIVE，不制造结论。

### 当日证据清单

scale table、artifact SHA、checkpoint audit、错误桶、止损记录、决策及下一步。

## 3. 故障恢复速查

- OOM：区分 load、forward activation、all-gather、save/export；依次修 CPU load、sequence/micro、wrap、full-state CPU offload。
- NaN/Inf：保留首个 bad sample/step，全 rank 一致停止，FP32 重放；不可单 rank 跳步。
- DDP/FSDP hang：2-step 有界复现，保存每 rank traceback 与 `NCCL_DEBUG=INFO`；不以永久 `NCCL_P2P_DISABLE=1` 当修复。
- checkpoint 损坏：没有 `COMPLETE.json`，或 model/optimizer/scheduler/scaler/RNG/shards/lineage 任一 group/hash 失败就拒绝；从上一完整点恢复。
- LoRA load：必须同时提供 base revision/tree SHA、adapter config/weights、tokenizer和 trainable manifest，并实际 offline reload/eval；unexpected/missing key 不能用宽松加载隐藏。
- reward hacking：冻结恶意集复测，检查各分项、KL、entropy、长度、unique；不达门立即停 RL。
