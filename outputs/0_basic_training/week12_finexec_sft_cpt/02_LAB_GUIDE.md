# Week 12 实践篇：从官方 FinQA 到可执行 FinExec SFT/CPT

> 所有命令均为待执行；预计输出不是实测。本文不提供投资建议，不连接交易、RAG、外部 API 或云日志。公司侧所有数据、checkpoint、日志和指标留在公司环境。

## 0. 下载前：环境、空间和回滚

主路径：公司 Linux V100 上用 PyTorch 2.1 + FP16；个人 5070 Ti 只做公开小样本；H100 仅独立对照。磁盘估算：Qwen snapshot 1.2GB、FinQA <0.2GB、一个全量 Trainer checkpoint 约 6–10GB，至少预留 30GB；0.5B 全参单卡峰值可能 12–24GB，必须 smoke 实测。20-step smoke 估算 5–20 分钟，正式 300 step 依序列/卡数现场估算。

复用批准环境，不创建 Conda/venv，不升级 torch/CUDA：

```bash
which python; python --version; git --version; git lfs version || true
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
df -h .
```

创建项目根并保存变更前状态：

```bash
mkdir -p week12-finexec/{configs,data/{processed,manifests},manifests,models,runs,tests,reports,third_party}
cd week12-finexec
python -m pip freeze > manifests/pip-freeze.before.txt
python -m pip check | tee manifests/pip-check.before.txt
```

公司预期 torch 2.1；不符合时停止并记录。将紧随其后的代码块原样保存为 `requirements-lock.txt`，再 dry-run；`APPROVED_WHEELHOUSE` 必须由环境负责人指向现有只读 wheel 目录：

```text
# file: requirements-lock.txt
transformers==4.46.3
accelerate==1.1.1
huggingface_hub==0.26.2
safetensors==0.4.5
```

```bash
python - <<'PY'
import importlib.metadata as m
for p in ("transformers","accelerate","huggingface_hub","safetensors"):
  try: print(p,m.version(p))
  except m.PackageNotFoundError: print(p,"MISSING")
PY
test -n "$APPROVED_WHEELHOUSE" && test -d "$APPROVED_WHEELHOUSE"
python -m pip install --dry-run --no-index --find-links "$APPROVED_WHEELHOUSE" -r requirements-lock.txt
```

只有审批后才去掉 `--dry-run`。若不兼容，停止并由环境负责人按 `pip-freeze.before.txt`/基础镜像回滚；不要用 pip 重装 torch。禁止 W&B、Hub push 和个人 token。

## 1. 固定获取数据与模型

| 对象 | 官方 ID/仓库 | 固定 revision | 必需文件 | 许可 | 完整性 |
|---|---|---|---|---|---|
| FinQA | `https://github.com/czyssrs/FinQA.git` | `0f16e2867befa6840783e58be38c9efb9229d742` | `dataset/{train,dev,test}.json`、LICENSE | MIT | commit、文件 SHA、JSON 样本/schema |
| TAT-QA（transfer） | `https://github.com/NExTplusplus/TAT-QA.git` | `870accc41953dcde885aabeb963d94aabdc0fbc3` | dataset、README、LICENSE | 数据 CC BY 4.0 | commit/SHA；本周核心不读取 |
| Qwen base | `Qwen/Qwen2.5-0.5B` | `060db6499f32faf8b98477b0a26969ef7d8b9987` | JSON、tokenizer、`model.safetensors`、LICENSE | Apache-2.0 | resolved revision、SHA、离线 load |

联网且获批时：

```bash
cd week12-finexec
git clone --no-checkout https://github.com/czyssrs/FinQA.git third_party/FinQA
git -C third_party/FinQA checkout --detach 0f16e2867befa6840783e58be38c9efb9229d742
test "$(git -C third_party/FinQA rev-parse HEAD)" = "0f16e2867befa6840783e58be38c9efb9229d742"
test -f third_party/FinQA/LICENSE
test -f third_party/FinQA/dataset/train.json
printf '%s\n' 'czyssrs/FinQA@0f16e2867befa6840783e58be38c9efb9229d742' > third_party/FinQA/SOURCE_REVISION
python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
snapshot_download(repo_id="Qwen/Qwen2.5-0.5B",
 revision="060db6499f32faf8b98477b0a26969ef7d8b9987",local_dir="models/Qwen2.5-0.5B",
 allow_patterns=["*.json","*.safetensors","*.txt","*.model","LICENSE","README.md"])
Path("models/Qwen2.5-0.5B/SOURCE_REVISION").write_text(
 "Qwen/Qwen2.5-0.5B@060db6499f32faf8b98477b0a26969ef7d8b9987\n",encoding="utf-8")
PY
test -f models/Qwen2.5-0.5B/model.safetensors
test "$(cat models/Qwen2.5-0.5B/SOURCE_REVISION)" = "Qwen/Qwen2.5-0.5B@060db6499f32faf8b98477b0a26969ef7d8b9987"
find third_party/FinQA/dataset models/Qwen2.5-0.5B -type f -print0 | sort -z | xargs -0 sha256sum > data/manifests/sources.sha256
git -C third_party/FinQA rev-parse HEAD > data/manifests/finqa.revision
```

离线只接受管理员提供的相同 commit/model snapshot；对方同时提供 SHA 清单、许可文件，以及内容分别为固定仓库 ID/revision 的 `third_party/FinQA/SOURCE_REVISION` 与 `models/Qwen2.5-0.5B/SOURCE_REVISION`。没有合法模型时可用已在本机的公开小模型跑代码门，但必须另建配置和 manifest，结果标签为 `MODEL-BLOCKED/INCONCLUSIVE`。不得绕过网络策略。

## 2. 工程树与文件职责

```bash
cd week12-finexec
mkdir -p configs tests
```

上面的命令只创建目录，不会生成任何源码或配置。请把下文每个代码块原样保存为其标题指定的相对路径；保存后再执行 Day 1 的语法与单元测试。

```text
week12-finexec/
  evaluator.py             # 安全 DSL、schema、waterfall
  prepare.py               # 官方 JSON→固定 JSONL/CPT；异常清单
  train.py                 # 无 packing 的 SFT/CPT + Trainer checkpoint
  evaluate.py              # 独立 greedy generation + evaluator
  tests/test_core.py       # 30+ executor、mask、安全测试
  configs/*.json           # 完整、可冻结配置
  third_party/FinQA/       # 固定 commit，只读
  models/Qwen2.5-0.5B/    # 固定 snapshot，只读
  data/processed/          # JSONL 与 manifest
  runs/                    # 本机训练/eval
```

### 2.1 `evaluator.py`：完整安全 executor

将紧随其后的代码块原样保存为 `evaluator.py`：

```python
# file: evaluator.py
import json, math, re

OPS={
 "add":lambda a,b:a+b,"subtract":lambda a,b:a-b,"multiply":lambda a,b:a*b,
 "divide":lambda a,b:a/b,"exp":lambda a,b:a**b,"greater":lambda a,b:1.0 if a>b else 0.0,
}
def split_top(s):
    out=[]; start=0; depth=0
    for i,ch in enumerate(s):
        depth += (ch=="(")-(ch==")")
        if depth<0: raise ValueError("unbalanced")
        if ch=="," and depth==0: out.append(s[start:i].strip()); start=i+1
    if depth!=0: raise ValueError("unbalanced")
    out.append(s[start:].strip())
    if any(not x for x in out): raise ValueError("empty token")
    return out
def number(x):
    x=x.strip().lower().replace(",","").replace("$","")
    const={"const_m1":-1.,"const_1":1.,"const_100":100.,"const_1000":1000.}
    if x in const:return const[x]
    if x.endswith("%"): return float(x[:-1])/100.0
    return float(x)
def execute(program,max_steps=16):
    if len(program)>1000: raise ValueError("program too long")
    vals=[]; steps=split_top(program)
    if not steps or len(steps)>max_steps: raise ValueError("bad step count")
    for step in steps:
        m=re.fullmatch(r"([a-z_]+)\((.*)\)",step.strip())
        if not m or m.group(1) not in OPS: raise ValueError("unknown syntax/op")
        args=split_top(m.group(2))
        if len(args)!=2: raise ValueError("arity")
        def val(x):
            if re.fullmatch(r"#\d+",x):
                i=int(x[1:])
                if i>=len(vals): raise ValueError("forward reference")
                return vals[i]
            return number(x)
        a,b=map(val,args)
        if m.group(1)=="divide" and b==0: raise ValueError("divide zero")
        y=float(OPS[m.group(1)](a,b))
        if not math.isfinite(y) or abs(y)>1e30: raise ValueError("nonfinite/range")
        vals.append(y)
    return vals[-1]
def close(a,b): return abs(a-b)<=max(1e-4,1e-3*abs(b))
def parse_answer(x):
    if isinstance(x,(int,float)): return float(x)
    s=str(x).strip().lower()
    if s in {"yes","true"}: return 1.0
    if s in {"no","false"}: return 0.0
    return number(s)
METRIC_KEYS=("json","schema","evidence_ids_valid","evidence","evidence_precision","evidence_recall",
             "evidence_f1","scale","program","execution","answer","consistent")
def evidence_scores(pred,gold):
    p,g=set(pred),set(gold); hit=len(p&g)
    precision=hit/len(p) if p else (1.0 if not g else 0.0)
    recall=hit/len(g) if g else (1.0 if not p else 0.0)
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    return float(precision),float(recall),float(f1),p==g
def score(text,gold,allowed_evidence):
    flags={k:0.0 for k in METRIC_KEYS}
    try: obj=json.loads(text); flags["json"]=True
    except Exception as e: return flags,"json:"+str(e)
    if not isinstance(obj,dict) or set(obj)!={"evidence","program","answer","scale"}: return flags,"schema:keys"
    if (not isinstance(obj["evidence"],list) or not all(isinstance(x,str) for x in obj["evidence"])
        or len(obj["evidence"])!=len(set(obj["evidence"]))): return flags,"schema:evidence"
    if not isinstance(obj["program"],str): return flags,"schema:program"
    if obj["scale"] not in {"none","percent","thousand","million","billion"}: return flags,"schema:scale"
    if isinstance(obj["answer"],bool) or not isinstance(obj["answer"],(int,float)): return flags,"schema:answer"
    reported=float(obj["answer"])
    if not math.isfinite(reported): return flags,"schema:answer"
    if (not isinstance(gold,dict) or set(gold)!={"evidence","program","answer","scale"}
        or not isinstance(gold["evidence"],list) or not isinstance(gold["program"],str)
        or gold["scale"] not in {"none","percent","thousand","million","billion"}): return flags,"gold:schema"
    if isinstance(gold["answer"],bool) or not isinstance(gold["answer"],(int,float)): return flags,"gold:answer"
    gold_answer=float(gold["answer"])
    if not math.isfinite(gold_answer): return flags,"gold:answer"
    flags["schema"]=True
    if not set(obj["evidence"]).issubset(set(allowed_evidence)): return flags,"evidence:id"
    flags["evidence_ids_valid"]=True
    ep,er,ef,exact=evidence_scores(obj["evidence"],gold["evidence"])
    flags.update(evidence_precision=ep,evidence_recall=er,evidence_f1=ef,evidence=exact)
    flags["scale"]=obj["scale"]==gold["scale"]
    try: y=execute(obj["program"]); flags["program"]=flags["execution"]=True
    except Exception as e: return flags,"program:"+str(e)
    flags["answer"]=close(y,gold_answer)
    flags["consistent"]=close(y,reported)
    if not flags["evidence"]: return flags,"evidence:mismatch"
    if not flags["scale"]: return flags,"scale:mismatch"
    if not flags["answer"]: return flags,"answer:mismatch"
    if not flags["consistent"]: return flags,"reported:inconsistent"
    return flags,"ok"
if __name__=="__main__":
    tests=[("add(2,3)",5),("subtract(2,3)",-1),("multiply(2,3)",6),("divide(6,3)",2),
           ("exp(2,3)",8),("greater(3,2)",1),("add(2,3), multiply(#0,4)",20)]
    for p,y in tests: assert close(execute(p),y),(p,execute(p),y)
    assert parse_answer("yes")==1.0 and parse_answer("no")==0.0
    for bad in ("__import__(os)","divide(1,0)","add(1)","add(#0,1)","open(x,y)","add(1,,2)"):
        try: execute(bad); raise AssertionError(bad)
        except (ValueError,ZeroDivisionError): continue
    print(json.dumps({"status":"PASS","cases":len(tests)+8}))
```

`score(text, gold, allowed_evidence)` 的稳定返回接口是 `(flags, error)`。`flags` 固定含 `json/schema/evidence_ids_valid/evidence/evidence_precision/evidence_recall/evidence_f1/scale/program/execution/answer/consistent`；其中 `evidence` 表示 gold set exact match，而不是仅“ID 合法”。落盘 `gold` 固定 schema 为 `{"evidence": list[str], "program": str, "answer": finite number, "scale": enum}`；score 消费其中 evidence/answer/scale，执行答案评分以预测 program 的实际执行值对 `gold.answer`。

### 2.2 `prepare.py`：完整 adapter、split 与清洗清单

将紧随其后的代码块原样保存为 `prepare.py`：

```python
# file: prepare.py
import argparse, hashlib, json, re, subprocess
from pathlib import Path
from transformers import AutoTokenizer
from evaluator import execute, parse_answer, close

FINQA_COMMIT="0f16e2867befa6840783e58be38c9efb9229d742"
QWEN_REVISION="060db6499f32faf8b98477b0a26969ef7d8b9987"
def context(item):
    lines=[]; ids=[]
    texts=list(item.get("pre_text",[]))+list(item.get("post_text",[]))
    for i,s in enumerate(texts): ids.append(f"text_{i}"); lines.append(f"[text_{i}] {s.strip()}")
    for i,row in enumerate(item.get("table",[])):
        ids.append(f"table_{i}"); lines.append(f"[table_{i}] "+" | ".join(map(str,row)))
    return "\n".join(lines),ids
def display_number(answer):
    s=str(answer).strip().lower()
    if s in {"yes","true"}: return 1.0
    if s in {"no","false"}: return 0.0
    for token in ("percent","billion","million","thousand","%"):
        s=s.replace(token,"")
    return parse_answer(s.strip())
def display_matches(answer,raw_answer,scale):
    """Accept both official percent program conventions, with displayed rounding."""
    displayed=display_number(answer); raw=parse_answer(raw_answer)
    candidates=[raw,100.0*raw] if scale=="percent" else [raw]
    cleaned=str(answer).strip().lower().replace(",","").replace("$","")
    for token in ("percent","billion","million","thousand","%"):
        cleaned=cleaned.replace(token,"")
    m=re.fullmatch(r"[+-]?\d+(?:\.(\d+))?",cleaned.strip())
    rounding_tol=(0.5*10**(-len(m.group(1) or ""))+1e-12) if m else 1e-4
    return any(abs(displayed-x)<=max(rounding_tol,1e-6*max(1.0,abs(x))) for x in candidates)
def infer_scale(answer,raw_answer):
    s=str(answer).strip().lower()
    if "%" in s or "percent" in s: return "percent"
    for unit in ("billion","million","thousand"):
        if unit in s: return unit
    try:
        if not close(display_number(answer),parse_answer(raw_answer)) and display_matches(answer,raw_answer,"percent"):
            return "percent"
    except Exception: pass
    return "none"
def target_for(item):
    qa=item["qa"]
    if not isinstance(qa.get("gold_inds"),dict): raise ValueError("gold_inds must be a mapping")
    source_program=qa["program"].strip(); program=re.sub(r"\s+","",source_program)
    ans=qa.get("exe_ans",qa.get("answer"))
    return ({"evidence":sorted(map(str,qa["gold_inds"].keys())),"program":program,
             "answer":parse_answer(ans),"scale":infer_scale(qa.get("answer",""),ans)},source_program)
def prompt(ctx,q):
    return ("Use only listed evidence IDs. Return exactly one JSON object with keys evidence, program, answer, scale. "
            "Answer is the raw restricted-program result; scale is a separately checked presentation unit. "
            "Allowed ops: add, subtract, multiply, divide, exp, greater. This is dataset arithmetic, not investment advice.\n"
            f"DOCUMENT:\n{ctx}\nQUESTION: {q.strip()}\nOUTPUT_JSON:")
def digest(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def tokenizer_probe(tok):
    probes=["[text_0] Revenue 12.5%",'{"evidence":["table_0"],"program":"divide(1,2)"}',"财报 question"]
    payload={"class":tok.__class__.__name__,"size":len(tok),"eos":tok.eos_token_id,
             "encodings":[tok(x,add_special_tokens=True)["input_ids"] for x in probes]}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--source",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--tokenizer",required=True); ap.add_argument("--tokenizer-revision",default=QWEN_REVISION)
    ap.add_argument("--max-length",type=int,default=512); ap.add_argument("--max-input",type=int,default=512)
    ap.add_argument("--max-new",type=int,default=128); a=ap.parse_args()
    source=Path(a.source); src=source/"dataset"; out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    source_marker=source/"SOURCE_REVISION"; marker_revision=""
    if source_marker.is_file(): marker_revision=source_marker.read_text(encoding="utf-8").strip()
    try: revision=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()
    except (subprocess.CalledProcessError,FileNotFoundError):
        if marker_revision!=f"czyssrs/FinQA@{FINQA_COMMIT}": raise RuntimeError("FinQA git revision and approved marker both unavailable")
        revision=FINQA_COMMIT
    if revision!=FINQA_COMMIT: raise RuntimeError(f"FinQA revision mismatch: {revision}")
    if marker_revision and marker_revision!=f"czyssrs/FinQA@{FINQA_COMMIT}": raise RuntimeError("FinQA SOURCE_REVISION mismatch")
    if not (source/"LICENSE").is_file(): raise RuntimeError("FinQA LICENSE missing")
    marker=Path(a.tokenizer)/"SOURCE_REVISION"
    expected_marker=f"Qwen/Qwen2.5-0.5B@{a.tokenizer_revision}"
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip()!=expected_marker:
        raise RuntimeError("tokenizer SOURCE_REVISION missing or mismatched")
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True,use_fast=True)
    if tok.eos_token_id is None: raise RuntimeError("EOS required")
    summary={}; gold_exceptions=[]; length_exclusions=[]; eligible_ids={}; cpt_rows=[]
    for split in ("train","dev","test"):
        src_file=src/f"{split}.json"; raw=json.loads(src_file.read_text(encoding="utf-8")); kept=[]; supported=0; seen=set()
        for i,item in enumerate(raw):
            source_id=item.get("id"); sid=str(source_id) if source_id else f"{split}:MISSING_ID:{i:06d}"
            try:
                if not source_id: raise ValueError("missing official item id")
                if sid in seen: raise ValueError("duplicate official item id")
                seen.add(sid)
                ctx,ids=context(item); t,source_program=target_for(item)
                missing=set(t["evidence"])-set(ids)
                if missing: raise ValueError("missing evidence:"+repr(sorted(missing)))
                y=execute(t["program"])
                if not close(y,t["answer"]): raise ValueError(f"gold mismatch exec={y} answer={t['answer']}")
                displayed_answer=item["qa"].get("answer",item["qa"].get("exe_ans"))
                if not display_matches(displayed_answer,t["answer"],t["scale"]):
                    raise ValueError(f"scale contract mismatch display={display_number(displayed_answer)} raw={t['answer']} scale={t['scale']}")
            except Exception as e:
                gold_exceptions.append({"id":sid,"reason":str(e),"program":item.get("qa",{}).get("program")})
                continue
            supported+=1; p=prompt(ctx,item["qa"]["question"])
            target=json.dumps(t,separators=(",",":"),ensure_ascii=False)
            p_tokens=len(tok(p,add_special_tokens=True)["input_ids"])
            t_tokens=len(tok(target,add_special_tokens=False)["input_ids"])+1
            reason=None
            if p_tokens>a.max_input: reason=f"prompt_tokens={p_tokens}>max_input={a.max_input}"
            elif split=="train" and p_tokens+t_tokens>a.max_length:
                reason=f"prompt_plus_target={p_tokens+t_tokens}>max_length={a.max_length}"
            if reason:
                length_exclusions.append({"id":sid,"split":split,"reason":reason})
                continue
            row={"id":sid,"source_split":split,"prompt":p,"target":target,"gold":t,"allowed_evidence":ids,"context":ctx,
                 "source_program":source_program,"prompt_tokens":p_tokens,"target_tokens":t_tokens}
            kept.append(row)
            if split=="train": cpt_rows.append({"id":sid,"prompt":"","target":ctx})
        with (out/f"{split}.jsonl").open("w",encoding="utf-8") as f:
            for x in kept: f.write(json.dumps(x,ensure_ascii=False)+"\n")
        eligible_ids[split]=[x["id"] for x in kept]
        summary[split]={"raw":len(raw),"gold_supported":supported,"eligible":len(kept),
                        "gold_excluded":len(raw)-supported,"length_excluded":supported-len(kept)}
    with (out/"cpt_train.jsonl").open("w",encoding="utf-8") as f:
        for x in cpt_rows: f.write(json.dumps(x,ensure_ascii=False)+"\n")
    eligible_ids["cpt_train"]=[x["id"] for x in cpt_rows]
    (out/"gold_exceptions.json").write_text(json.dumps(gold_exceptions,indent=2,ensure_ascii=False),encoding="utf-8")
    (out/"length_exclusions.json").write_text(json.dumps(length_exclusions,indent=2,ensure_ascii=False),encoding="utf-8")
    (out/"eligible_ids.json").write_text(json.dumps(eligible_ids,indent=2),encoding="utf-8")
    tokenizer_names=("tokenizer.json","tokenizer_config.json","vocab.json","merges.txt","tokenizer.model",
                     "special_tokens_map.json","added_tokens.json","SOURCE_REVISION")
    tokenizer_sha={n:digest(Path(a.tokenizer)/n) for n in tokenizer_names if (Path(a.tokenizer)/n).is_file()}
    source_sha={f"{s}.json":digest(src/f"{s}.json") for s in ("train","dev","test")}
    artifacts={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="manifest.json"}
    manifest={"schema":"finexec-eligible-v1","source_commit":revision,"source_sha256":source_sha,
      "tokenizer":{"path":str(Path(a.tokenizer).resolve()),"revision":a.tokenizer_revision,"sha256":tokenizer_sha,
                   "probe_sha256":tokenizer_probe(tok)},
      "eligibility_rules":{"full_context_no_truncation":True,"max_input_tokens":a.max_input,
       "train_max_prompt_plus_target_tokens":a.max_length,"generation_max_new_tokens":a.max_new,
       "dev_test_length_selection":"prompt tokens only; never gold evidence/answer length"},
      "summary":summary,"eligible_ids_file":"eligible_ids.json","artifacts_sha256":artifacts,
      "answer_semantics":"raw executor result; scale checked separately",
      "gold_gate":"100% parse/execute/match only for explicitly listed supported subset"}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    assert all(v["eligible"]>0 for v in summary.values())
    print(json.dumps({"status":"PASS","summary":summary,"gold_exceptions":len(gold_exceptions),
                      "length_exclusions":len(length_exclusions),"manifest_sha":digest(out/"manifest.json")}))
if __name__=="__main__": main()
```

这里的两类排除是有意分开的：`gold_exceptions.json` 记录最小 DSL/官方标注不受支持，`length_exclusions.json` 记录冻结 tokenizer 后的长度门。`eligible_ids.json` 固定每个 split 的顺序与 ID，`manifest.json` 绑定源文件、tokenizer、规则和全部产物 SHA。dev/test 的长度选择只看完整 prompt token 数，绝不读取 gold evidence/answer 来删句、截断或挑样本；同一 eligible dev/test JSONL 必须供所有模型评测。若排除率高或分布偏移，结论标 `INCONCLUSIVE`，不能宣称覆盖全 FinQA。

### 2.3 两份完整配置

将紧随其后的代码块原样保存为 `configs/sft_05b.json`：

```json
{"mode":"sft","model_dir":"models/Qwen2.5-0.5B","model_id":"Qwen/Qwen2.5-0.5B",
 "model_revision":"060db6499f32faf8b98477b0a26969ef7d8b9987",
 "train_file":"data/processed/train.jsonl","data_manifest":"data/processed/manifest.json",
 "max_length":512,"per_device_batch":1,"grad_accum":8,"max_steps":300,"learning_rate":0.00002,
 "warmup_steps":20,"weight_decay":0.01,"lr_scheduler_type":"linear","max_grad_norm":1.0,
 "save_steps":100,"logging_steps":1,"seed":1207,"full_determinism":true,"fp16":true}
```

将紧随其后的代码块原样保存为 `configs/cpt_short.json`：

```json
{"mode":"cpt","model_dir":"models/Qwen2.5-0.5B","model_id":"Qwen/Qwen2.5-0.5B",
 "model_revision":"060db6499f32faf8b98477b0a26969ef7d8b9987",
 "train_file":"data/processed/cpt_train.jsonl","data_manifest":"data/processed/manifest.json",
 "max_length":512,"per_device_batch":1,"grad_accum":8,"max_steps":100,"learning_rate":0.00001,
 "warmup_steps":10,"weight_decay":0.01,"lr_scheduler_type":"linear","max_grad_norm":1.0,
 "save_steps":50,"logging_steps":1,"seed":1207,"full_determinism":true,"fp16":true}
```

### 2.4 `train.py`：完整无 packing 训练入口

将紧随其后的代码块原样保存为 `train.py`：

```python
# file: train.py
import argparse, hashlib, importlib.metadata, json, os, re
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed

class Records(Dataset):
    def __init__(self,path,tok,max_length):
        self.rows=[]; target_counts=[]; sequence_counts=[]
        for lineno,line in enumerate(open(path,encoding="utf-8"),1):
            x=json.loads(line); p=tok(x["prompt"],add_special_tokens=True)["input_ids"]
            t=tok(x["target"],add_special_tokens=False)["input_ids"]+[tok.eos_token_id]
            if len(p)+len(t)>max_length:
                raise RuntimeError(f"ineligible row {x.get('id',lineno)} has {len(p)+len(t)}>{max_length} tokens; rerun prepare")
            self.rows.append({"input_ids":p+t,"labels":[-100]*len(p)+t,"id":x["id"]})
            target_counts.append(len(t)); sequence_counts.append(len(p)+len(t))
        if not self.rows: raise RuntimeError("empty eligible training set")
        self.stats={"records":len(self.rows),"eligible_target_tokens":sum(target_counts),
                    "min_target_tokens":min(target_counts),"max_target_tokens":max(target_counts),
                    "max_sequence_tokens":max(sequence_counts),"target_truncation":0}
        print(json.dumps(self.stats))
    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return self.rows[i]
class Collate:
    def __init__(self,pad): self.pad=pad
    def __call__(self,rows):
        n=max(len(x["input_ids"]) for x in rows); ids=[]; labels=[]; mask=[]
        for x in rows:
            k=n-len(x["input_ids"]); ids.append(x["input_ids"]+[self.pad]*k)
            labels.append(x["labels"]+[-100]*k); mask.append([1]*len(x["input_ids"])+[0]*k)
        y={"input_ids":torch.tensor(ids),"attention_mask":torch.tensor(mask),"labels":torch.tensor(labels)}
        assert (y["labels"]!=-100).sum()>0
        return y
class StopAfter(TrainerCallback):
    def __init__(self,step): self.step=step
    def on_step_end(self,args,state,control,**kwargs):
        if self.step and state.global_step>=self.step:
            control.should_save=True; control.should_training_stop=True
        return control
class AuditTrainer(Trainer):
    def __init__(self,*args,**kwargs): self._last_scale=None; self._scale_backoffs=0; super().__init__(*args,**kwargs)
    def log(self,logs,*args,**kwargs):
        scaler=getattr(getattr(self,"accelerator",None),"scaler",None)
        skipped=bool(getattr(getattr(self,"accelerator",None),"optimizer_step_was_skipped",False))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            lo=torch.tensor(int(skipped),device=self.args.device); hi=lo.clone()
            torch.distributed.all_reduce(lo,op=torch.distributed.ReduceOp.MIN)
            torch.distributed.all_reduce(hi,op=torch.distributed.ReduceOp.MAX)
            if lo.item()!=hi.item(): raise RuntimeError("AMP skip decision diverged across ranks")
            skipped=bool(hi.item())
        if scaler is not None:
            cur=float(scaler.get_scale())
            if self._last_scale is not None and cur<self._last_scale:self._scale_backoffs+=1
            self._last_scale=cur; logs={**logs,"loss_scale":cur,"observed_scale_backoffs":self._scale_backoffs}
        logs={**logs,"optimizer_step_was_skipped":skipped}
        result=super().log(logs,*args,**kwargs)
        if skipped: raise FloatingPointError("AMP skipped an optimizer update; aborting because Trainer global_step is not a successful-update clock")
        return result
def file_sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def tree_sha(root):
    root=Path(root); h=hashlib.sha256(); files=sorted(p for p in root.rglob("*") if p.is_file())
    if not files: raise RuntimeError(f"empty model directory: {root}")
    for p in files:
        h.update(p.relative_to(root).as_posix().encode()); h.update(b"\0")
        with p.open("rb") as f:
            for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def object_sha(obj):
    raw=json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(str(path)+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding="utf-8")
    os.replace(tmp,path)
def checkpoint_payload_files(root):
    root=Path(root); seals={"CHECKPOINT_MANIFEST.json","COMPLETED"}
    return sorted((p for p in root.rglob("*") if p.is_file() and p.relative_to(root).as_posix() not in seals),
                  key=lambda p:p.relative_to(root).as_posix())
def require_checkpoint_components(names,world_size,precision):
    base={Path(name).name for name in names}
    for required in ("optimizer.pt","scheduler.pt","trainer_state.json"):
        if required not in base: raise RuntimeError(f"checkpoint missing required file: {required}")
    if not any(name.endswith(".safetensors") or Path(name).name=="pytorch_model.bin" for name in names):
        raise RuntimeError("checkpoint missing model weights")
    rng=[name for name in base if re.fullmatch(r"rng_state(?:_\d+)?\.pth",name)]
    if len(rng)<int(world_size):
        raise RuntimeError(f"checkpoint has {len(rng)} RNG files, expected at least {world_size}")
    if precision=="fp16" and "scaler.pt" not in base:
        raise RuntimeError("FP16 checkpoint missing scaler.pt")
def seal_checkpoint(path,effective):
    root=Path(path)
    if not root.is_dir() or not re.fullmatch(r"checkpoint-\d+",root.name):
        raise RuntimeError(f"invalid checkpoint directory: {root}")
    marker=root/"COMPLETED"; sidecar=root/"CHECKPOINT_MANIFEST.json"
    marker.unlink(missing_ok=True)
    Path(str(marker)+".tmp").unlink(missing_ok=True)
    Path(str(sidecar)+".tmp").unlink(missing_ok=True)
    payload=checkpoint_payload_files(root)
    if not payload: raise RuntimeError("empty checkpoint payload")
    names=[p.relative_to(root).as_posix() for p in payload]
    if any(p.stat().st_size==0 for p in payload): raise RuntimeError("checkpoint contains an empty payload file")
    require_checkpoint_components(names,effective["world_size"],effective["precision"])
    files={name:{"bytes":p.stat().st_size,"sha256":file_sha(p)} for name,p in zip(names,payload)}
    manifest={"format_version":1,"checkpoint_dir":root.name,"step":int(root.name.split("-",1)[1]),
      "effective_contract_sha":object_sha(effective),"code_sha":effective["code_sha"],
      "torch_version":str(torch.__version__),"transformers_version":importlib.metadata.version("transformers"),
      "world_size":effective["world_size"],"precision":effective["precision"],
      "required_files":names,"files":files}
    atomic_json(sidecar,manifest)
    atomic_json(marker,{"format_version":1,"step":manifest["step"],"manifest_sha256":file_sha(sidecar)})
    return manifest
def validate_sealed_checkpoint(path,effective):
    root=Path(path); sidecar=root/"CHECKPOINT_MANIFEST.json"; marker=root/"COMPLETED"
    if not root.is_dir() or not re.fullmatch(r"checkpoint-\d+",root.name):
        raise RuntimeError(f"invalid checkpoint directory: {root}")
    if not sidecar.is_file() or sidecar.stat().st_size==0: raise RuntimeError("checkpoint manifest missing/empty")
    if not marker.is_file() or marker.stat().st_size==0: raise RuntimeError("checkpoint COMPLETED marker missing/empty")
    try:
        m=json.loads(sidecar.read_text(encoding="utf-8")); done=json.loads(marker.read_text(encoding="utf-8"))
    except Exception as e: raise RuntimeError(f"invalid checkpoint seal JSON: {e}") from e
    step=int(root.name.split("-",1)[1])
    if (m.get("format_version")!=1 or done.get("format_version")!=1 or
        m.get("checkpoint_dir")!=root.name or m.get("step")!=step or done.get("step")!=step):
        raise RuntimeError("checkpoint seal schema/path mismatch")
    if done.get("manifest_sha256")!=file_sha(sidecar):
        raise RuntimeError("checkpoint manifest SHA mismatch; refuse resume")
    if m.get("effective_contract_sha")!=object_sha(effective):
        raise RuntimeError("effective resume contract mismatch")
    if (m.get("code_sha")!=effective["code_sha"] or m.get("torch_version")!=str(torch.__version__) or
        m.get("transformers_version")!=importlib.metadata.version("transformers")):
        raise RuntimeError("checkpoint code/runtime version mismatch; refuse resume")
    files=m.get("files"); required=m.get("required_files")
    if not isinstance(files,dict) or not files or not isinstance(required,list) or sorted(files)!=sorted(required):
        raise RuntimeError("checkpoint file manifest schema mismatch")
    actual=[p.relative_to(root).as_posix() for p in checkpoint_payload_files(root)]
    if sorted(files)!=sorted(actual): raise RuntimeError("checkpoint payload file set changed; refuse resume")
    for name,meta in files.items():
        rel=Path(name)
        if rel.is_absolute() or ".." in rel.parts: raise RuntimeError("unsafe checkpoint manifest path")
        p=root/rel
        if (not p.is_file() or p.stat().st_size==0 or not isinstance(meta,dict) or
            meta.get("bytes")!=p.stat().st_size or meta.get("sha256")!=file_sha(p)):
            raise RuntimeError(f"checkpoint payload hash/size mismatch: {name}")
    require_checkpoint_components(actual,m.get("world_size"),m.get("precision"))
    if m.get("world_size")!=effective["world_size"] or m.get("precision")!=effective["precision"]:
        raise RuntimeError("checkpoint world-size/precision mismatch")
    return m
class CheckpointSeal(TrainerCallback):
    def __init__(self,effective): self.effective=effective
    def on_save(self,args,state,control,**kwargs):
        distributed=torch.distributed.is_available() and torch.distributed.is_initialized()
        if distributed: torch.distributed.barrier()
        rank0=not distributed or torch.distributed.get_rank()==0; failure=[None]
        if rank0:
            try: seal_checkpoint(Path(args.output_dir)/f"checkpoint-{state.global_step}",self.effective)
            except Exception as e: failure[0]=repr(e)
        if distributed: torch.distributed.broadcast_object_list(failure,src=0)
        if failure[0] is not None: raise RuntimeError("checkpoint sealing failed: "+failure[0])
        return control
def tokenizer_probe(tok):
    probes=["[text_0] Revenue 12.5%",'{"evidence":["table_0"],"program":"divide(1,2)"}',"财报 question"]
    payload={"class":tok.__class__.__name__,"size":len(tok),"eos":tok.eos_token_id,
             "encodings":[tok(x,add_special_tokens=True)["input_ids"] for x in probes]}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def bind_data(train_file,manifest_file,tok):
    doc=json.loads(Path(manifest_file).read_text(encoding="utf-8")); actual=file_sha(train_file)
    expected=doc.get("artifacts_sha256",{}).get(Path(train_file).name)
    if expected!=actual: raise RuntimeError(f"training data is absent from or mismatched with manifest: {train_file}")
    if doc.get("tokenizer",{}).get("probe_sha256")!=tokenizer_probe(tok):
        raise RuntimeError("training tokenizer differs from eligibility tokenizer")
    return actual,file_sha(manifest_file)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--output-dir",required=True)
    ap.add_argument("--model-dir"); ap.add_argument("--max-steps",type=int); ap.add_argument("--grad-accum",type=int)
    ap.add_argument("--resume"); ap.add_argument("--stop-after",type=int); ap.add_argument("--fp32",action="store_true"); a=ap.parse_args()
    cfg=json.loads(Path(a.config).read_text()); set_seed(cfg["seed"]); model_dir=a.model_dir or cfg["model_dir"]
    tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,use_fast=True)
    if tok.eos_token_id is None: raise RuntimeError("EOS required")
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    source_marker=Path(model_dir)/"SOURCE_REVISION"
    expected_marker=f"{cfg['model_id']}@{cfg['model_revision']}"
    if not source_marker.is_file() or source_marker.read_text(encoding="utf-8").strip()!=expected_marker:
        raise RuntimeError("model SOURCE_REVISION missing or mismatched")
    data=Records(cfg["train_file"],tok,cfg["max_length"])
    model=AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,torch_dtype=torch.float32,
                                               attn_implementation="eager")
    model.config.use_cache=False
    steps=a.max_steps or cfg["max_steps"]; accum=a.grad_accum or cfg["grad_accum"]
    if a.stop_after and not 0<a.stop_after<steps: raise ValueError("stop-after must be between 1 and max-steps-1")
    if cfg["logging_steps"]!=1: raise ValueError("logging_steps must stay 1 so every AMP update is audited")
    args=TrainingArguments(output_dir=a.output_dir,overwrite_output_dir=False,max_steps=steps,
      per_device_train_batch_size=cfg["per_device_batch"],gradient_accumulation_steps=accum,
      learning_rate=cfg["learning_rate"],warmup_steps=cfg["warmup_steps"],weight_decay=cfg["weight_decay"],
      lr_scheduler_type=cfg["lr_scheduler_type"],max_grad_norm=cfg["max_grad_norm"],
      logging_steps=cfg["logging_steps"],save_steps=min(cfg["save_steps"],steps),save_total_limit=3,
      fp16=bool(cfg["fp16"] and not a.fp32),bf16=False,report_to=[],remove_unused_columns=False,
      dataloader_drop_last=True,seed=cfg["seed"],data_seed=cfg["seed"],save_safetensors=True,
      ddp_find_unused_parameters=False,full_determinism=cfg["full_determinism"])
    out=Path(a.output_dir); train_sha,data_manifest_sha=bind_data(cfg["train_file"],cfg["data_manifest"],tok)
    conflict=out.exists() and any(out.iterdir())
    distributed=torch.distributed.is_available() and torch.distributed.is_initialized()
    if distributed:
        flag=torch.tensor(int(conflict),device=args.device); torch.distributed.all_reduce(flag,op=torch.distributed.ReduceOp.MAX); conflict=bool(flag.item())
    if conflict and not a.resume: raise FileExistsError("non-resume run refuses non-empty output-dir")
    out.mkdir(parents=True,exist_ok=True)
    world_size=args.world_size; precision="fp16" if args.fp16 else "fp32"
    effective={"config_sha":file_sha(a.config),"code_sha":file_sha(__file__),"train_sha":train_sha,
      "data_manifest_sha":data_manifest_sha,"model_dir":str(Path(model_dir).resolve()),"model_tree_sha":tree_sha(model_dir),
      "model_id":cfg["model_id"],"model_revision":cfg["model_revision"],"mode":cfg["mode"],"max_length":cfg["max_length"],
      "per_device_batch":cfg["per_device_batch"],"gradient_accumulation_steps":accum,
      "global_batch":cfg["per_device_batch"]*accum*world_size,"max_steps":steps,
      "learning_rate":cfg["learning_rate"],"warmup_steps":cfg["warmup_steps"],"weight_decay":cfg["weight_decay"],
      "lr_scheduler_type":cfg["lr_scheduler_type"],"max_grad_norm":cfg["max_grad_norm"],
      "seed":cfg["seed"],"world_size":world_size,"precision":precision,"torch":str(torch.__version__),
      "transformers":importlib.metadata.version("transformers"),"full_determinism":cfg["full_determinism"],
      "tokenizer_probe_sha":tokenizer_probe(tok),
      "data_stats":data.stats}
    resume_seal=None
    if a.resume:
        rp=Path(a.resume).resolve()
        if (not rp.is_dir() or rp.parent!=out.resolve() or
            not re.fullmatch(r"checkpoint-\d+",rp.name)):
            raise RuntimeError("resume checkpoint must be checkpoint-N inside the same output-dir")
        previous_path=out/"run_manifest.json"
        if not previous_path.is_file(): raise RuntimeError("resume requires existing run_manifest.json")
        previous=json.loads(previous_path.read_text(encoding="utf-8"))
        if previous.get("effective_contract")!=effective: raise RuntimeError("effective resume contract mismatch")
        if previous.get("status")!="INTERRUPTED_FOR_RESUME": raise RuntimeError("resume requires a frozen INTERRUPTED_FOR_RESUME manifest")
        if Path(previous.get("stop_checkpoint","")).resolve()!=rp: raise RuntimeError("resume checkpoint differs from frozen stop checkpoint")
        resume_seal=validate_sealed_checkpoint(rp,effective)
        frozen_sha=previous.get("stop_checkpoint_manifest_sha")
        if not frozen_sha or frozen_sha!=file_sha(rp/"CHECKPOINT_MANIFEST.json"):
            raise RuntimeError("resume checkpoint manifest differs from frozen run manifest")
    manifest={"status":"STARTED_NOT_VALIDATED","config":cfg,"effective_contract":effective,
      "stop_after":a.stop_after,"resume":a.resume,"resume_sha":tree_sha(a.resume) if a.resume else None,
      "update_clock":"Trainer global_step is accepted only while every audited optimizer_step_was_skipped is false"}
    rank0=not distributed or torch.distributed.get_rank()==0
    if rank0: atomic_json(out/"run_manifest.json",manifest)
    if distributed: torch.distributed.barrier()
    callbacks=[CheckpointSeal(effective)]
    if a.stop_after: callbacks.insert(0,StopAfter(a.stop_after))
    trainer=AuditTrainer(model=model,args=args,train_dataset=data,data_collator=Collate(tok.pad_token_id),callbacks=callbacks)
    try: result=trainer.train(resume_from_checkpoint=a.resume)
    except Exception as e:
        if trainer.is_world_process_zero():
            manifest.update(status="FAIL_SYSTEM",observed_trainer_global_step=trainer.state.global_step,error=repr(e))
            atomic_json(out/"run_manifest.json",manifest)
        raise
    interrupted=bool(a.stop_after and trainer.state.global_step>=a.stop_after)
    if not interrupted and trainer.state.global_step!=steps:
        raise RuntimeError(f"training ended at {trainer.state.global_step}, expected {steps}")
    if torch.cuda.is_available():
        manifest["rank0_peak_allocated_gb"]=torch.cuda.max_memory_allocated(args.device)/1024**3
        manifest["rank0_peak_reserved_gb"]=torch.cuda.max_memory_reserved(args.device)/1024**3
    trainer.save_state()
    if interrupted:
        if trainer.is_world_process_zero():
            ckpt=out/f"checkpoint-{trainer.state.global_step}"
            if not ckpt.is_dir(): raise RuntimeError(f"StopAfter checkpoint missing: {ckpt}")
            sealed=validate_sealed_checkpoint(ckpt,effective)
            manifest.update(status="INTERRUPTED_FOR_RESUME",successful_steps=trainer.state.global_step,
                            attempted_steps=trainer.state.global_step,stop_checkpoint=str(ckpt.resolve()),
                            stop_checkpoint_manifest_sha=file_sha(ckpt/"CHECKPOINT_MANIFEST.json"),
                            stop_checkpoint_payload_files=len(sealed["files"]))
            (out/"partial_train_metrics.json").write_text(json.dumps(result.metrics,indent=2),encoding="utf-8")
            atomic_json(out/"run_manifest.json",manifest)
            print(json.dumps({"status":"INTERRUPTED_FOR_RESUME","successful_steps":trainer.state.global_step}))
        return
    trainer.save_model(out/"final_model")
    if trainer.is_world_process_zero():
        final_ckpt=out/f"checkpoint-{trainer.state.global_step}"
        final_seal=validate_sealed_checkpoint(final_ckpt,effective)
        tok.save_pretrained(out/"final_model")
        if source_marker.is_file():
            (out/"final_model"/"SOURCE_REVISION").write_text(source_marker.read_text(encoding="utf-8"),encoding="utf-8")
        (out/"final_model"/"MODEL_LINEAGE.json").write_text(json.dumps(effective,indent=2),encoding="utf-8")
        (out/"train_metrics.json").write_text(json.dumps(result.metrics,indent=2),encoding="utf-8")
        manifest.update(status="COMPLETED_PENDING_EVAL",successful_steps=trainer.state.global_step,
                        attempted_steps=trainer.state.global_step,final_model=str(out/"final_model"),
                        final_checkpoint=str(final_ckpt.resolve()),
                        final_checkpoint_manifest_sha=file_sha(final_ckpt/"CHECKPOINT_MANIFEST.json"),
                        final_checkpoint_payload_files=len(final_seal["files"]))
        atomic_json(out/"run_manifest.json",manifest)
        print(json.dumps({"status":"COMPLETED_PENDING_EVAL","successful_steps":trainer.state.global_step,**result.metrics}))
if __name__=="__main__": main()
```

这是固定 Transformers v4.46.3 的 API 路径。有效恢复合同绑定 config/code/data manifest/model tree、优化器超参、max steps、resolved precision 与 world size；只允许从同一 output-dir 的 checkpoint 恢复，`stop_after` 不属于训练语义。每次 `on_save` 在所有 rank 的 RNG 文件就绪后，由 rank 0 核对权重、optimizer、scheduler、trainer/RNG state（FP16 还包括 scaler），记录每个 payload 的字节数和 SHA-256，最后原子发布 `CHECKPOINT_MANIFEST.json` 与 `COMPLETED`。恢复必须在 Trainer/`torch.load` 前验证全部文件，并与上一次原子冻结在 `run_manifest.json` 的 manifest SHA 对上。StopAfter 返回后只标 `INTERRUPTED_FOR_RESUME`，不发布 `final_model`。完整结束才保存模型和 tokenizer，因此 CPT 的 `final_model` 可直接作为后续 SFT 的 `--model-dir`。代码逐步审计 AMP skip，任一 skip即跨 rank 一致失败，避免把 Trainer 增长的 `global_step` 误称 successful update。多 rank token 数不相等时 rank-mean 可能不等于严格 global-token mean，因此多卡本周只作系统练习；严谨等价实验要让每 rank target token 数一致或改为 global sum/count 自定义 loss，并单独回归。

### 2.5 `evaluate.py`：独立生成与 waterfall

将紧随其后的代码块原样保存为 `evaluate.py`：

```python
# file: evaluate.py
import argparse,hashlib,json,re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from evaluator import score,METRIC_KEYS
def file_sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def tree_sha(root):
    root=Path(root); h=hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        h.update(p.relative_to(root).as_posix().encode()); h.update(b"\0")
        with p.open("rb") as f:
            for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def tokenizer_probe(tok):
    probes=["[text_0] Revenue 12.5%",'{"evidence":["table_0"],"program":"divide(1,2)"}',"财报 question"]
    payload={"class":tok.__class__.__name__,"size":len(tok),"eos":tok.eos_token_id,
             "encodings":[tok(x,add_special_tokens=True)["input_ids"] for x in probes]}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--tokenizer",required=True)
    ap.add_argument("--data",required=True); ap.add_argument("--out",required=True); ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--manifest",required=True); ap.add_argument("--max-input",type=int,default=512)
    ap.add_argument("--max-new",type=int,default=128); a=ap.parse_args()
    manifest=json.loads(Path(a.manifest).read_text(encoding="utf-8")); data_sha=file_sha(a.data)
    if manifest.get("artifacts_sha256",{}).get(Path(a.data).name)!=data_sha:
        raise RuntimeError("eval data is absent from or mismatched with eligibility manifest")
    rules=manifest["eligibility_rules"]
    if a.max_input!=rules["max_input_tokens"] or a.max_new!=rules["generation_max_new_tokens"]:
        raise RuntimeError("generation limits differ from frozen eligibility manifest")
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True); tok.pad_token=tok.pad_token or tok.eos_token
    if manifest.get("tokenizer",{}).get("probe_sha256")!=tokenizer_probe(tok):
        raise RuntimeError("eval tokenizer differs from eligibility tokenizer")
    model_marker=Path(a.model)/"SOURCE_REVISION"
    if not model_marker.is_file(): raise RuntimeError("evaluated model lacks SOURCE_REVISION lineage")
    model_source=model_marker.read_text(encoding="utf-8").strip()
    model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.float32,
                                               attn_implementation="eager")
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(dev).eval()
    all_rows=[json.loads(x) for x in open(a.data,encoding="utf-8")]
    id_path=Path(a.manifest).parent/manifest["eligible_ids_file"]
    if manifest["artifacts_sha256"].get(id_path.name)!=file_sha(id_path): raise RuntimeError("eligible ID file hash mismatch")
    id_doc=json.loads(id_path.read_text(encoding="utf-8"))
    split=Path(a.data).stem
    if [x["id"] for x in all_rows]!=id_doc.get(split): raise RuntimeError("eval row IDs differ from eligible manifest")
    rows=all_rows[:a.limit or None]
    counts={k:0.0 for k in METRIC_KEYS}; preds=[]
    for x in rows:
        z=tok(x["prompt"],return_tensors="pt",truncation=False)
        if z.input_ids.shape[1]>a.max_input: raise RuntimeError("input exceeds frozen limit:"+x["id"])
        z={k:v.to(dev) for k,v in z.items()}
        with torch.no_grad(): y=model.generate(**z,do_sample=False,num_beams=1,max_new_tokens=a.max_new,
          eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id)
        text=tok.decode(y[0,z["input_ids"].shape[1]:],skip_special_tokens=True).strip()
        flags,err=score(text,x["gold"],x["allowed_evidence"])
        for k,v in flags.items(): counts[k]+=float(v)
        preds.append({"id":x["id"],"text":text,"flags":flags,"error":err})
    out=Path(a.out)
    if out.exists() and any(out.iterdir()): raise FileExistsError("eval refuses non-empty output directory")
    out.mkdir(parents=True,exist_ok=True)
    with (out/"predictions.jsonl").open("w",encoding="utf-8") as f:
        for x in preds:f.write(json.dumps(x,ensure_ascii=False)+"\n")
    ids_sha=hashlib.sha256("\n".join(x["id"] for x in rows).encode()).hexdigest()
    summary={"status":"EVALUATED_NOT_INTERPRETED","n":len(rows),"eligible_file_n":len(all_rows),
             "rates":{k:v/max(1,len(rows)) for k,v in counts.items()},"data_sha":data_sha,
             "eligibility_manifest_sha":file_sha(a.manifest),"evaluated_ids_sha":ids_sha,
             "model_tree_sha":tree_sha(a.model),"model_source_revision":model_source,
             "generation":{"do_sample":False,"num_beams":1,"max_input_tokens":a.max_input,"max_new_tokens":a.max_new}}
    (out/"metrics.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary))
if __name__=="__main__":main()
```

### 2.6 `tests/test_core.py`：30+ cases、安全与 mask

将紧随其后的代码块原样保存为 `tests/test_core.py`：

```python
# file: tests/test_core.py
import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluator import execute,close,score
from prepare import display_matches,target_for
class TestCore(unittest.TestCase):
  def test_30_numeric_cases(self):
    n=0
    for a in range(-3,4):
      for b in range(1,6):
        self.assertTrue(close(execute(f"add({a},{b})"),a+b)); n+=1
    self.assertGreaterEqual(n,30)
  def test_refs(self): self.assertEqual(execute("add(2,3), multiply(#0,4)"),20)
  def test_official_percent_conventions(self):
    fixtures=[
      # Official schema-style A leaves a ratio raw; display renders it as percent.
      {"qa":{"gold_inds":{"text_0":"fixture"},"program":"divide(2526,40678)",
             "exe_ans":execute("divide(2526,40678)"),"answer":"6.2%"}},
      # Official schema-style B already multiplies by const_100; do not multiply twice.
      {"qa":{"gold_inds":{"table_0":"fixture"},
             "program":"subtract(85936,83000), divide(#0,83000), multiply(#1,const_100)",
             "exe_ans":execute("subtract(85936,83000), divide(#0,83000), multiply(#1,const_100)"),
             "answer":"3.5%"}},
    ]
    for item in fixtures:
      target,_=target_for(item)
      self.assertTrue(close(execute(target["program"]),target["answer"]))
      self.assertTrue(display_matches(item["qa"]["answer"],target["answer"],target["scale"]))
    # A literal percentage operand is a fraction inside arithmetic.
    self.assertTrue(close(execute("multiply(200,1.25%)"),2.5))
  def test_reject(self):
    for x in ("__import__(x)","divide(1,0)","add(1)","add(#0,1)","open(a,b)","add(1,2","add(1,,2)"):
      with self.assertRaises((ValueError,ZeroDivisionError)):execute(x)
  def test_schema(self):
    gold={"answer":5,"evidence":["text_0"],"program":"add(2,3)","scale":"none"}
    txt=json.dumps({"evidence":["text_0"],"program":"add(2,3)","answer":5,"scale":"none"})
    f,e=score(txt,gold,["text_0"]); self.assertTrue(all(f.values()),(f,e))
  def test_hallucinated_evidence(self):
    txt=json.dumps({"evidence":["text_9"],"program":"add(2,3)","answer":5,"scale":"none"})
    f,_=score(txt,{"answer":5,"evidence":["text_0"],"program":"add(2,3)","scale":"none"},["text_0"])
    self.assertFalse(f["evidence_ids_valid"]); self.assertFalse(f["evidence"])
  def test_empty_evidence_and_wrong_scale_do_not_pass(self):
    gold={"answer":5,"evidence":["text_0"],"program":"add(2,3)","scale":"percent"}
    txt=json.dumps({"evidence":[],"program":"add(2,3)","answer":5,"scale":"none"})
    f,_=score(txt,gold,["text_0"])
    self.assertTrue(f["evidence_ids_valid"]); self.assertEqual(f["evidence_f1"],0.0)
    self.assertFalse(f["evidence"]); self.assertFalse(f["scale"])
  def test_reported_answer_must_be_json_number(self):
    gold={"answer":5,"evidence":["text_0"],"program":"add(2,3)","scale":"none"}
    for bad in ("5",True):
      txt=json.dumps({"evidence":["text_0"],"program":"add(2,3)","answer":bad,"scale":"none"})
      f,_=score(txt,gold,["text_0"]); self.assertFalse(f["schema"])
  def test_partial_evidence_f1(self):
    gold={"answer":5,"evidence":["text_0","table_0"],"program":"add(2,3)","scale":"none"}
    txt=json.dumps({"evidence":["text_0"],"program":"add(2,3)","answer":5,"scale":"none"})
    f,_=score(txt,gold,["text_0","table_0"])
    self.assertAlmostEqual(f["evidence_precision"],1.0); self.assertAlmostEqual(f["evidence_recall"],0.5)
    self.assertAlmostEqual(f["evidence_f1"],2/3); self.assertFalse(f["evidence"])
if __name__=="__main__":unittest.main()
```

## Day 1：来源冻结、adapter 与 gold evaluator gate

### 为什么做

先证明 oracle，而不是让错误 evaluator 给模型打分。

### 输入与前置检查

从 `week12-finexec` 执行；FinQA commit、LICENSE、三个 JSON 和依赖锁已存在。

### 本日要创建/修改的文件

使用 §2 完整创建 `evaluator.py/prepare.py/tests/test_core.py`；生成 `data/processed/*`、两类 exclusions、eligible IDs 与 manifest。

### 实现（固定代码路径）

只使用本文代码；官方原始仓库保持 detached/read-only，不修改。

### 执行命令

```bash
cd week12-finexec
python -m py_compile evaluator.py prepare.py train.py evaluate.py tests/test_core.py
python evaluator.py
python -m unittest -v tests/test_core.py
python prepare.py --source third_party/FinQA --out data/processed \
  --tokenizer models/Qwen2.5-0.5B \
  --tokenizer-revision 060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --max-length 512 --max-input 512 --max-new 128
python -m json.tool data/processed/manifest.json
wc -l data/processed/{train,dev,test,cpt_train}.jsonl
sha256sum evaluator.py prepare.py data/processed/* > data/manifests/processed.sha256
```

### 预期观测（估算/示例，不是实测）

单测至少 30 numeric cases；prepare 分别输出每 split 的 raw/gold_supported/eligible/gold_excluded/length_excluded。不能预填排除率；若 supported 或 eligible subset 很小，后续只能做 pipeline smoke。

### 验收条件

eligible 样本逐条 program 可执行并匹配 raw executor gold；gold/length 两类 exceptions 非静默；gold evidence ID 存在；split 不混；CPT 只有 eligible train context。`eligible_ids.json` 与三个 JSONL 行序完全一致，manifest 中 tokenizer/source/artifact SHA 均可复算；dev/test 没有基于 gold evidence 的筛句或选样。

### 若失败，按什么顺序查

首个 exception 原 program → top-level split/reference → number/percent → evidence ID 编号 → source JSON schema。先扩 evaluator tests，再重跑全 prepare；禁止用 Python eval。

### 当日证据清单

source/processed/tokenizer SHA、commit、LICENSE、unit 输出、eligible ID 清单、split 数量、两类 exceptions 分布、gold gate 标签。

## Day 2：模型离线加载、mask 与单 batch 前反向

### 为什么做

验证 tokenizer/EOS/label 边界与 eager attention 的 FP32 前反向；V100 FP16 兼容性在 Day 3 的独立 smoke 中验证，不先烧正式步数。

### 输入与前置检查

Day 1 PASS；snapshot 文件齐全；Transformers 版本恰为 4.46.3。

### 本日要创建/修改的文件

创建 §2 的 `train.py` 与两份 config；生成 `runs/day2_one_step`。

### 实现（固定代码路径）

`Records` 明确拒绝被截断 target；`Collate` 只在 target/EOS 留 labels；packing 关闭。

### 执行命令

```bash
cd week12-finexec
python -m json.tool configs/sft_05b.json
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python - <<'PY'
from transformers import AutoConfig,AutoTokenizer
p="models/Qwen2.5-0.5B"; c=AutoConfig.from_pretrained(p,local_files_only=True); t=AutoTokenizer.from_pretrained(p,local_files_only=True)
print({"model_type":c.model_type,"hidden":c.hidden_size,"vocab":c.vocab_size,"eos":t.eos_token_id})
PY
CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python train.py \
  --config configs/sft_05b.json --output-dir runs/day2_one_step --max-steps 1 --grad-accum 1 --fp32
```

### 预期观测（估算/示例，不是实测）

离线 load 不访问网络；Trainer 打印一个 finite `loss/grad_norm/learning_rate` 并产生 checkpoint/final_model。单步可能数分钟；显存以 `nvidia-smi`/PyTorch 现场记录。

### 验收条件

model/tokenizer revision 对应 snapshot；target token 数>0；prompt/pad mask；不发生 target truncation；eager attention；无 BF16/FA2。

### 若失败，按什么顺序查

`transformers-cli env` → config model_type → tokenizer EOS/pad → snapshot SHA → FP32 OOM 则 sequence 512→256 且作为新 config → 仍不能加载则 `MODEL-BLOCKED`，不升级 torch。

### 当日证据清单

离线 load stdout、config/data/model SHA、one-step manifest/metrics、峰值显存和异常。

## Day 3：128 样本过拟合、20-step FP16 与 resume

### 为什么做

teacher-forced loss 与生成都要在小集合上响应，且 checkpoint 要能续训。

### 输入与前置检查

Day 2 PASS；先用 PowerShell/Unix 均可读写的 Python 生成固定 128 子集。

### 本日要创建/修改的文件

`data/processed/train128.jsonl`、临时 config、`runs/day3_*`。

### 实现（固定官方/本地路径）

```bash
python - <<'PY'
import hashlib,json
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[json.loads(x) for x in open("data/processed/train.jsonl",encoding="utf-8")][:128]
with open("data/processed/train128.jsonl","w",encoding="utf-8") as f:
  for x in rows:f.write(json.dumps(x,ensure_ascii=False)+"\n")
json.dump({"train128":[x["id"] for x in rows]},open("data/processed/train128_eligible_ids.json","w"),indent=2)
parent=json.load(open("data/processed/manifest.json",encoding="utf-8"))
subset={"schema":"finexec-derived-eligible-v1","parent_manifest_sha":sha("data/processed/manifest.json"),
 "artifacts_sha256":{"train128.jsonl":sha("data/processed/train128.jsonl"),
                     "train128_eligible_ids.json":sha("data/processed/train128_eligible_ids.json")},
 "eligible_ids_file":"train128_eligible_ids.json","eligibility_rules":parent["eligibility_rules"],
 "tokenizer":parent["tokenizer"]}
json.dump(subset,open("data/processed/train128.manifest.json","w"),indent=2)
c=json.load(open("configs/sft_05b.json")); c["train_file"]="data/processed/train128.jsonl"
c["data_manifest"]="data/processed/train128.manifest.json"
json.dump(c,open("configs/sft_128.json","w"),indent=2)
PY
```

### 执行命令

```bash
cd week12-finexec
CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python train.py \
  --config configs/sft_128.json --output-dir runs/day3_continuous30 --max-steps 30 --grad-accum 1 --fp32
CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python train.py \
  --config configs/sft_128.json --output-dir runs/day3_resume --max-steps 30 --grad-accum 1 --stop-after 20 --fp32
CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python train.py \
  --config configs/sft_128.json --output-dir runs/day3_resume --max-steps 30 --grad-accum 1 \
  --resume runs/day3_resume/checkpoint-20 --fp32
python - <<'PY'
from pathlib import Path
from safetensors import safe_open
a=Path("runs/day3_continuous30/final_model"); b=Path("runs/day3_resume/final_model")
af=sorted(x.name for x in a.glob("*.safetensors")); bf=sorted(x.name for x in b.glob("*.safetensors"))
assert af and af==bf,(af,bf); delta=0.0
for name in af:
  with safe_open(a/name,framework="pt",device="cpu") as x, safe_open(b/name,framework="pt",device="cpu") as y:
    assert set(x.keys())==set(y.keys())
    for k in x.keys(): delta=max(delta,(x.get_tensor(k)-y.get_tensor(k)).abs().max().item())
print({"continuous_vs_resume_max_abs":delta}); assert delta<=1e-5
PY
# 独立的真实 FP16 smoke；不与上面的 FP32 continuous/resume 等价实验混算：
CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python train.py \
  --config configs/sft_128.json --output-dir runs/day3_fp16_smoke --max-steps 20 --grad-accum 1
CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python evaluate.py \
  --model runs/day3_resume/final_model --tokenizer models/Qwen2.5-0.5B \
  --data data/processed/train128.jsonl --manifest data/processed/train128.manifest.json \
  --limit 16 --out runs/day3_eval --max-input 512 --max-new 128
```

### 预期观测（估算/示例，不是实测）

FP32 连续 30-step 与“相同 30-step schedule，在 step 20 主动保存停止后恢复到 30”可比较；step 20 时 manifest 必须是 `INTERRUPTED_FOR_RESUME` 且没有 `final_model`，恢复完成后才是 `COMPLETED_PENDING_EVAL`。另一个独立 run 确实启用 FP16 跑 20 step。eval 产生 predictions 和 waterfall；这些短跑不保证 JSON 学会，是管线门而非效果门。

### 验收条件

FP16 loss/grad finite 且每步 `optimizer_step_was_skipped=false`，任一 skip 都应停止；FP32 resume checkpoint 含 model/optimizer/scheduler/trainer/rng，FP16 smoke 的 checkpoint 还含 scaler。resume 不从 step 0，恢复合同完全相同，同机 FP32 continuous-vs-resume final weight max-abs≤1e-5；生成切片不包含 prompt且绑定 `train128.manifest.json`。真正 overfit 可继续有界到 500–2000 step，但必须先估算卡时。

### 若失败，按什么顺序查

固定一条 record 可视化 decoded labels → 单 batch FP32 → FP16 → checkpoint 文件清单 → resume 日志 global_step → 生成 EOS/max_new。loss 降而 JSON 坏时先修 mask/template，不加步数。

### 当日证据清单

subset/parent manifest SHA、FP32 continuous/resume 20/30 step metrics 与权重 max-abs stdout、独立 FP16 20-step metrics、checkpoint 文件清单、16 条预测/waterfall、峰值显存/scale。

## Day 4：有界正式 SFT 与可选短 CPT→SFT

### 为什么做

完成主路径，并把 CPT 作为受控 rehearsal，而非“知识注入”宣传。

### 输入与前置检查

Day 3 pipeline PASS；根据 smoke 估算 300 step 卡时/磁盘；dev/test 未用于调参。

### 本日要创建/修改的文件

仅 runs；若 4 卡，复制 config 为新文件并记录 `grad_accum=2`，保持 global batch `1×2×4=8`。

### 实现（固定代码路径）

单卡主路径 `global batch=1×8×1=8`。四卡命令必须显式覆盖 accum=2；DDP 只提升系统吞吐，不分摊模型状态。

### 执行命令

```bash
cd week12-finexec
CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python train.py \
  --config configs/sft_05b.json --output-dir runs/S_base_to_sft
# 获批四卡替代路径（不要与上条同时算主 run）：
CUDA_VISIBLE_DEVICES=0,1,2,3 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 torchrun --standalone --nproc-per-node=4 \
  train.py --config configs/sft_05b.json --output-dir runs/S_base_to_sft_ws4 --grad-accum 2
# 可选 CPT；只有主 SFT 完成后：
CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python train.py \
  --config configs/cpt_short.json --output-dir runs/C_cpt
CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python train.py \
  --config configs/sft_05b.json --model-dir runs/C_cpt/final_model --output-dir runs/C_cpt_to_sft
```

### 预期观测（估算/示例，不是实测）

每个完整 run 有有界 max_steps、checkpoint、final_model、manifest、metrics；`final_model` 同时含模型、tokenizer 与 `MODEL_LINEAGE.json`，所以 CPT→SFT 的本地加载路径是自足的。不同 tokenizer 时不可直接比 raw loss；本处 tokenizer 固定。

### 验收条件

successful steps、eligible target-token 分布与 rank-0 peak memory 记录完整；FP16 无非有限/skip；save/resume 可用；CPT 只读 train context；SFT 配置除起始 checkpoint 外一致；不自动上传。

### 若失败，按什么顺序查

OOM：缩 micro/sequence并重算 global batch → NaN：FP32 固定 batch重放 → DDP hang：2-step `NCCL_DEBUG=INFO` → CPT 后坏：确认 SFT optimizer重建、model/tokenizer一致。达到止损即标 FAIL-SYSTEM/INCONCLUSIVE。

### 当日证据清单

每 run manifest、完整 config、source/data/model revision、训练指标、checkpoint SHA、world size、卡时估算/实测分栏。

## Day 5：独立 dev/test eval、错误 waterfall 与结论

### 为什么做

用同一生成协议比较 base、SFT、CPT→SFT，并把格式、证据、程序、执行和答案错误分开。

### 输入与前置检查

Day 4 至少主 SFT 完成；先 dev 选择，test 只对冻结最终 checkpoint 运行一次。

### 本日要创建/修改的文件

`runs/eval_*`、`reports/decision.md`、`reports/artifacts.sha256`。

### 实现（固定代码路径）

使用同一 `evaluate.py --max-input 512 --max-new 128`，greedy/no tool feedback。

将紧随其后的代码块保存为 `reports/decision.md`；评测后只用实际 metrics/SHA 替换 `NOT-RUN`：

```markdown
# FinExec decision
- status: NOT-RUN
- selected checkpoint: NOT-RUN
- eligible manifest/data/evaluated IDs SHA: NOT-RUN
- JSON / evidence F1 / scale / execution / answer: NOT-RUN
- CPT delta and conclusion: NOT-RUN
- limitations / failure bucket / cost: NOT-RUN
- safety: research only; no trading or investment advice
```

### 执行命令

```bash
cd week12-finexec
for spec in \
  "base models/Qwen2.5-0.5B" \
  "sft runs/S_base_to_sft/final_model"; do
  set -- $spec
  CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python evaluate.py \
    --model "$2" --tokenizer models/Qwen2.5-0.5B --data data/processed/dev.jsonl \
    --manifest data/processed/manifest.json --out "runs/eval_$1" --max-input 512 --max-new 128
done
# 仅冻结选择后执行一次 test：
CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python evaluate.py \
  --model runs/S_base_to_sft/final_model --tokenizer models/Qwen2.5-0.5B \
  --data data/processed/test.jsonl --manifest data/processed/manifest.json \
  --out runs/final_test --max-input 512 --max-new 128
find runs -name metrics.json -o -name train_metrics.json | sort
find configs data/processed runs -type f -print0 | sort -z | xargs -0 sha256sum > reports/artifacts.sha256
```

### 预期观测（估算/示例，不是实测）

每个 metrics 显示 JSON/schema/evidence ID validity/evidence exact、precision/recall/F1、scale/program/execution/answer/consistent rates，并写入 data、eligible manifest、evaluated IDs 与 model tree SHA；不预填提升数字。

### 验收条件

同一个 manifest 锁定的 prompt/eligible IDs/tokenizer/generation/evaluator；base/SFT 必须有相同 `data_sha` 与 `evaluated_ids_sha`。gold supported subset gate 清楚；抽查 30–50 error；valid JSON≥90% 只是目标，未达则如实归因；CPT 差异不足时标 neutral/negative/inconclusive。

### 若失败，按什么顺序查

先生成是否截断 → JSON parse → schema → evidence hallucination → program parser → executed vs reported → scale。系统错误修复后必须重跑整个 eval，不能手改预测。

### 当日证据清单

base/SFT/可选 CPT→SFT 的 metrics/predictions、artifact SHA、错误桶、唯一结论标签、非投资建议与数据边界声明。

## 3. 观测、止损与恢复速查

- train loss 下降只说明 teacher forcing；必须看 greedy waterfall。
- `grad_norm` 爆炸、loss scale 连续回退、NaN/Inf：立即停，固定 sample 以 FP32 重放。
- valid JSON 升、execution 不升：模型只学格式或程序/operand 仍错；不上 RL。
- evidence F1 降而 answer 偶中：reward hacking 风险；reported answer 不可替代 program。
- OOM：先 sequence/micro batch，DDP 加卡不会分摊每卡模型状态。
- 恢复只用完整 checkpoint；核对 model/tokenizer/data/config SHA、resolved precision/world size 和 global step。坏/缺 optimizer/scheduler/RNG 的目录只能用于评测；FP16 checkpoint 还必须有 scaler，FP32 checkpoint 不应被错误要求 scaler。
- 任何尚未执行项写 `NOT-RUN` 或 `ENV-BLOCKED`，不得用 0 或预计值伪装结果。
