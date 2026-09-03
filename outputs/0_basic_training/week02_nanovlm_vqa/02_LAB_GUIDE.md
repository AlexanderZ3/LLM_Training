# Week 02 实践篇：从空目录跑通 nanoVLM 风格 222M VQA

> **全篇待执行、待验证。** 本手册不声称任何模型、数据或命令已在用户机器运行。公司机器上的代码、数据、图片、日志、trace、checkpoint、路径与硬件数字不得导出；个人 5070 Ti 只允许从公开源独立重跑。公司环境复用既有 Conda，不创建 Conda/venv，不升级 PyTorch 2.1。

## Day 0：环境、磁盘、依赖与遥测预检

### 为什么做

nanoVLM v0.1 的官方默认包含 BF16、`torch.compile` 和 W&B，不适合 V100/PyTorch2.1。本日先证明现有 torch/torchvision ABI、Hub工具和磁盘边界，不运行官方 `train.py`。

### 输入与前置检查

估算磁盘：三个模型/tokenizer 快照及数据子集建议至少 8–15 GB；估算显存：222M projector-only 单卡通常可在 16/32 GB 内从 micro-batch 1 起步，但以实测为准；估算时间：下载取决于网络，128 样本 overfit 数十分钟量级，正式短训数小时量级。

`[公司 Linux]`：

```bash
which python
python -V
python -m pip --version
python -m pip check
nvidia-smi
df -h .
git --version
python - <<'PY'
import json, torch
d={"torch":torch.__version__,"cuda":torch.version.cuda,"cuda_ok":torch.cuda.is_available(),"gpus":torch.cuda.device_count()}
try:
 import torchvision; d["torchvision"]=torchvision.__version__
except Exception as e: d["torchvision_error"]=type(e).__name__
if torch.cuda.is_available(): d.update(gpu=torch.cuda.get_device_name(0),capability=torch.cuda.get_device_capability(0),bf16=torch.cuda.is_bf16_supported())
print(json.dumps(d))
PY
```

`[个人 PowerShell]`：

```powershell
Get-Command python
python -V
python -m pip check
nvidia-smi
Get-PSDrive -PSProvider FileSystem
python -c "import torch, torchvision; print(torch.__version__, torch.version.cuda, torchvision.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None"
```

### 本日要创建/修改的文件

`[公司 Linux]`：

```bash
mkdir -p week02-vlm/{artifacts,configs,data/images,manifests,src,tests,runs,third_party}
cd week02-vlm
touch src/__init__.py
python -m pip freeze > manifests/pip-before.txt
```

`[个人 PowerShell]`：

```powershell
New-Item -ItemType Directory -Force week02-vlm,week02-vlm/artifacts,week02-vlm/configs,week02-vlm/data/images,week02-vlm/manifests,week02-vlm/src,week02-vlm/tests,week02-vlm/runs,week02-vlm/third_party | Out-Null
Set-Location week02-vlm
New-Item -ItemType File -Force src/__init__.py | Out-Null
python -m pip freeze | Set-Content -Encoding utf8 manifests/pip-before.txt
```

目录职责：

```text
artifacts/{siglip,smollm2,tokenizer,nanovlm-v01}/  固定公开快照
data/{images,records.jsonl}/                    本地有界 VQA 子集
manifests/sources.json                         requested/resolved revision 与许可声明
manifests/                                      其余 hash/env/审批证据
src/prepare_data.py                             Hub流式抽取或合成 fixture
src/data.py                                     image processor、packing、mask
src/vlm.py                                      vision→shuffle→projector→LM
src/train_vlm.py                                train/eval/counterfactual/resume
tests/test_contracts.py                         mask/projector/normalization
tests/test_trainer.py                           token加权/tied-weight/lineage
runs/                                           仅本机证据
```

### 实现

兼容矩阵：

| 组件 | 公司约束/候选 | 决策 |
|---|---|---|
| torch | 已装 2.1.x | 禁止替换 |
| torchvision | 0.16.x，patch 版本与 torch 对齐 | 不匹配就停止并找管理员 |
| transformers | 候选 4.46.3 | 支持 SmolLM2；先 dry-run/import smoke |
| datasets | 候选 2.21.0 | 只用于 The Cauldron streaming |
| huggingface_hub | 候选 0.26.5 | 固定快照下载 |
| safetensors | 候选 0.4.5 | 模型权重 |
| Pillow/NumPy | Pillow 10.x、NumPy <2 | 避免旧栈 ABI 漂移 |

这些是教程验证候选，不是公司已安装事实。

### 执行命令

`[两者/Python]` 只解析依赖：

```bash
python -m pip install --dry-run "transformers==4.46.3" "datasets==2.21.0" "huggingface_hub==0.26.5" "safetensors==0.4.5" "Pillow>=10,<11" "numpy<2" "pytest<9"
```

只有公司镜像批准且输出不替换 torch/torchvision/CUDA包时才执行：

```bash
python -m pip install "transformers==4.46.3" "datasets==2.21.0" "huggingface_hub==0.26.5" "safetensors==0.4.5" "Pillow>=10,<11" "numpy<2" "pytest<9"
python -m pip check
python -c "import transformers,datasets,huggingface_hub,safetensors,PIL; print(transformers.__version__,datasets.__version__,huggingface_hub.__version__)"
```

若 resolver 要改变 torch/torchvision，停止；公司由管理员恢复 `pip-before.txt` 对应 lock，个人按个人 lock 重建。不要 `pip install nanoVLM`、不要运行 `wandb login`。

### 预期观测（估算/示例，不是实测）

公司 V100 通常 capability 7.0、BF16 false；torchvision 可正常 import；候选依赖不要求替换 torch。任何不同都记录为事实。

### 验收条件

`pip check` 通过、当前环境快照已保存、磁盘充足、torch/torchvision ABI正确、无外部遥测凭据。

### 若失败，按什么顺序查

解释器路径 → torch/torchvision版本对 → CUDA分配 → resolver变更集 → 公司批准 wheel/镜像。不得升级生产 CUDA 或关闭证书验证。

### 当日证据清单

`pip-before.txt`、兼容矩阵实测列、磁盘预算、GPU探针摘要（全部留本机）。

## Day 1：锁定代码、backbone 与有界公开数据

### 为什么做

把仓库、模型、tokenizer、数据 config 与许可绑定到不可变 revision；下载后用文件 hash 证明本地对象完整。

### 输入与前置检查

| 对象 | 固定标识 | 所需文件/范围 | 许可检查 | 完整性 |
|---|---|---|---|---|
| nanoVLM | `v0.1` / commit `6ba9082e16f1fc8c21a1f8d0c54b26c9233c8771` | 仓库，只读对照 | MIT LICENSE | `git rev-parse HEAD` |
| SigLIP | `google/siglip-base-patch16-224@7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` | config、processor、safetensors | model card/Apache-2.0 | snapshot文件清单+SHA |
| SmolLM2 | `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | config、safetensors | model card/Apache-2.0 | 同上 |
| tokenizer | `HuggingFaceTB/cosmo2-tokenizer@4ce2318a3628e77279c939ed6a9f3f03034402de` | tokenizer JSON/config | card/随附许可 | encode/EOS/pad smoke |
| The Cauldron | `HuggingFaceM4/the_cauldron@847a98a779b1652d65111daf20c972dfcd333605`，config `clevr` | 前 6000 个可用去重样本 | config card与上游许可 | stable IDs、image hash、schema |

公司必须先批准公开对象；The Cauldron 是集合，不能只看集合页就假定所有 config 同一许可。

### 本日要创建/修改的文件

先创建机器可读的固定来源合同；许可字段是待公司复核的来源声明，不等于审批已经完成：

```json
{
  "schema_version": 1,
  "verified_on": "2026-09-03",
  "objects": {
    "nanovlm": {"repo": "https://github.com/huggingface/nanoVLM", "requested": "v0.1", "resolved": "6ba9082e16f1fc8c21a1f8d0c54b26c9233c8771", "license_claim": "MIT"},
    "vision": {"repo": "google/siglip-base-patch16-224", "requested": "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed", "resolved": "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed", "license_claim": "Apache-2.0"},
    "lm": {"repo": "HuggingFaceTB/SmolLM2-135M", "requested": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2", "resolved": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2", "license_claim": "Apache-2.0"},
    "tokenizer": {"repo": "HuggingFaceTB/cosmo2-tokenizer", "requested": "4ce2318a3628e77279c939ed6a9f3f03034402de", "resolved": "4ce2318a3628e77279c939ed6a9f3f03034402de", "license_claim": "review model card"},
    "dataset": {"repo": "HuggingFaceM4/the_cauldron", "config": "clevr", "requested": "847a98a779b1652d65111daf20c972dfcd333605", "resolved": "847a98a779b1652d65111daf20c972dfcd333605", "license_claim": "review CLEVR upstream terms"}
  }
}
```

将该块原样保存为 `manifests/sources.json`；不得把 `license_claim` 改写成 `approved=true`，审批证据另存并留在执行机器。

创建数据抽取器：

```python
# src/prepare_data.py
import argparse, hashlib, io, json, random
from pathlib import Path
from PIL import Image, ImageDraw

def png_bytes(image):
    b=io.BytesIO(); image.convert("RGB").save(b,format="PNG",optimize=False); return b.getvalue()
def stable_id(image_sha,q): return hashlib.sha256((image_sha+"\0"+q).encode("utf-8")).hexdigest()
def split_for_image(image_sha): return "dev" if int(image_sha[:8],16)%10==0 else "train"
def write(rows,out_root,source):
    root=Path(out_root); image_dir=root/"images"; image_dir.mkdir(parents=True,exist_ok=True)
    records=[]; seen={}
    for image,q,a in rows:
        blob=png_bytes(image); iid=hashlib.sha256(blob).hexdigest(); sid=stable_id(iid,q)
        if sid in seen:
            if seen[sid]!=a: raise ValueError(f"conflicting gold for image/question {sid}")
            continue
        seen[sid]=a; rel=f"images/{iid}.png"
        image_path=root/rel
        if image_path.exists() and hashlib.sha256(image_path.read_bytes()).hexdigest()!=iid:
            raise ValueError(f"existing image hash mismatch: {image_path}")
        if not image_path.exists(): image_path.write_bytes(blob)
        records.append({"sample_id":sid,"image":rel,"question":q,"answer":a,
                        "split":split_for_image(iid),"image_sha256":iid})
    if not records or not any(x["split"]=="train" for x in records) or not any(x["split"]=="dev" for x in records):
        raise RuntimeError("empty data or missing train/dev split")
    train_images={r["image_sha256"] for r in records if r["split"]=="train"}
    dev_images={r["image_sha256"] for r in records if r["split"]=="dev"}
    if train_images & dev_images: raise RuntimeError("image leakage across train/dev")
    with (root/"records.jsonl").open("w",encoding="utf-8") as f:
        for r in sorted(records,key=lambda x:x["sample_id"]): f.write(json.dumps(r,ensure_ascii=False)+"\n")
    summary={"samples":len(records),"train":sum(r["split"]=="train" for r in records),
             "dev":sum(r["split"]=="dev" for r in records),"unique_images":len(train_images|dev_images),
             "image_overlap":0,"split_policy":"image_sha256_prefix_mod10_v1","source":source}
    (root/"data_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8"); print(summary)
def hub_rows(args):
    from datasets import load_dataset
    ds=load_dataset(args.repo,args.config,split="train",streaming=True,revision=args.revision)
    for row in ds.take(args.count):
        images=row.get("images"); texts=row.get("texts")
        if not images or not texts: continue
        t=texts[0]
        if not isinstance(t,dict) or "user" not in t or "assistant" not in t: continue
        yield images[0].convert("RGB"),str(t["user"]),str(t["assistant"])
def synthetic_rows(count,seed):
    rng=random.Random(seed); colors=[("red",(220,30,30)),("green",(30,180,50)),("blue",(30,60,220))]
    for index in range(count):
        name,rgb=rng.choice(colors); base=Image.new("RGB",(224,224),(255,255,255)); d=ImageDraw.Draw(base)
        if rng.random()<0.5: d.rectangle((64,64,160,160),fill=rgb); shape="square"
        else: d.ellipse((64,64,160,160),fill=rgb); shape="circle"
        desired="dev" if index%10==0 else "train"
        for nonce in range(10000):
            image=base.copy()
            image.putpixel((0,0),(index&255,(index>>8)&255,nonce&255))
            image.putpixel((1,0),((nonce>>8)&255,(index+nonce)&255,0))
            if split_for_image(hashlib.sha256(png_bytes(image)).hexdigest())==desired: break
        else: raise RuntimeError("could not construct deterministic synthetic split fixture")
        if rng.random()<0.5: q,a=f"What color is the {shape}?",name
        else: q,a="What shape is shown?",shape
        yield image,q,a
def main():
    p=argparse.ArgumentParser(); p.add_argument("--mode",choices=["hub","synthetic"],required=True)
    p.add_argument("--repo",default="HuggingFaceM4/the_cauldron"); p.add_argument("--config",default="clevr")
    p.add_argument("--revision",default="847a98a779b1652d65111daf20c972dfcd333605"); p.add_argument("--count",type=int,default=6000)
    p.add_argument("--seed",type=int,default=17); p.add_argument("--out",default="data"); a=p.parse_args()
    source={"mode":a.mode,"repo":a.repo if a.mode=="hub" else None,
            "config":a.config if a.mode=="hub" else None,
            "requested_revision":a.revision if a.mode=="hub" else None,
            "resolved_revision":a.revision if a.mode=="hub" else None,
            "count_requested":a.count,"seed":a.seed if a.mode=="synthetic" else None}
    write(hub_rows(a) if a.mode=="hub" else synthetic_rows(a.count,a.seed),a.out,source)
if __name__=="__main__": main()
```

### 实现

`sample_id` 只绑定规范化图像 hash 与 question，不读取 gold；split 只由 `image_sha256` 决定。同一图像即使有多个问题也只能落在同一 split，且写盘前显式断言 image-hash 交集为空。answer 只作为监督字段；同一 image/question 出现冲突 gold 时立即停止。synthetic fixture 在角落写两个无关语义的小像素，并仅按图像 hash 搜索预定 90/10 bucket，避免少数重复图恰好没有 dev；这个过程同样不读取 question/answer。这里构造的仍只是内部 dev，不声称官方 validation，合成模式只用于代码 gate。

### 执行命令

先固定参考仓库：

`[公司 Linux]` 或 `[个人 PowerShell]`：

```bash
git clone --filter=blob:none --no-checkout https://github.com/huggingface/nanoVLM.git third_party/nanoVLM-v0.1
git -C third_party/nanoVLM-v0.1 checkout 6ba9082e16f1fc8c21a1f8d0c54b26c9233c8771
git -C third_party/nanoVLM-v0.1 rev-parse HEAD
python -c "from huggingface_hub import HfApi; a=HfApi(); checks=[('model','google/siglip-base-patch16-224','7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed'),('model','HuggingFaceTB/SmolLM2-135M','93efa2f097d58c2a74874c7e644dbc9b0cee75a2'),('model','HuggingFaceTB/cosmo2-tokenizer','4ce2318a3628e77279c939ed6a9f3f03034402de'),('dataset','HuggingFaceM4/the_cauldron','847a98a779b1652d65111daf20c972dfcd333605')]; got=[(k,r,(a.model_info(r,revision=v) if k=='model' else a.dataset_info(r,revision=v)).sha,v) for k,r,v in checks]; assert all(x[2]==x[3] for x in got),got; print(got)"
```

网络获批时下载本地快照：

```bash
python -c "from huggingface_hub import snapshot_download as d; d('google/siglip-base-patch16-224',revision='7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed',local_dir='artifacts/siglip')"
python -c "from huggingface_hub import snapshot_download as d; d('HuggingFaceTB/SmolLM2-135M',revision='93efa2f097d58c2a74874c7e644dbc9b0cee75a2',local_dir='artifacts/smollm2')"
python -c "from huggingface_hub import snapshot_download as d; d('HuggingFaceTB/cosmo2-tokenizer',revision='4ce2318a3628e77279c939ed6a9f3f03034402de',local_dir='artifacts/tokenizer')"
python -m src.prepare_data --mode hub --repo HuggingFaceM4/the_cauldron --config clevr --revision 847a98a779b1652d65111daf20c972dfcd333605 --count 6000 --out data
```

离线/公司替代：管理员将这三个相同 revision 的完整快照置于已创建的 `artifacts/*`，The Cauldron 获批缓存置于数据目录；不得自行转运。等待数据时：

```bash
python -m src.prepare_data --mode synthetic --count 1000 --seed 17 --out data
```

`[公司 Linux]` hash：

```bash
find artifacts -type f -print0 | sort -z | xargs -0 sha256sum > manifests/artifacts.sha256
sha256sum data/records.jsonl > manifests/data.sha256
```

`[个人 PowerShell]` hash：

```powershell
Get-ChildItem artifacts -Recurse -File | Sort-Object FullName | Get-FileHash -Algorithm SHA256 | Format-Table -AutoSize | Out-File manifests/artifacts.sha256
Get-FileHash data/records.jsonl -Algorithm SHA256 | Format-List | Out-File manifests/data.sha256
```

### 预期观测（估算/示例，不是实测）

仓库 HEAD 精确匹配；三处本地快照有 config/processor/tokenizer/weights；records 有非零 train/dev、无重复 sample ID。具体数量可能少于6000（无效/重复被跳过），必须记录实值。

### 验收条件

revision/许可/文件清单/hash齐全；30条人工图文检查；train/dev sample ID 与 image hash overlap=0。合成数据状态保持 `INCONCLUSIVE`。

### 若失败，按什么顺序查

revision存在性 → LFS文件是否只是指针 → snapshot是否完整 → config schema → PIL坏图 → data许可/配额。不得 `trust_remote_code` 或 `ignore_mismatched_sizes` 绕过结构错误。

### 当日证据清单

`artifacts.sha256`、`data.sha256`、dataset/model cards、审批记录、`data_summary.json`、30条检查结论（图片本身不外传）。

## Day 2：实现 packing、projector 与 CPU 合同测试

### 为什么做

在加载222M前用小张量证明 label mask、pixel shuffle 与答案归一化，避免用大模型掩盖 off-by-one。

### 输入与前置检查

本地 tokenizer/image processor 快照完整；records schema 为 `sample_id,image,question,answer,split,image_sha256` 六个字段；最大文本长度固定79。

### 本日要创建/修改的文件

```python
# src/data.py
import json, re, unicodedata
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset

def pack_ids(prompt,answer,eos,pad,max_len):
    full=list(prompt)+list(answer)+[eos]
    if len(full)>max_len: raise ValueError(f"text length {len(full)} > {max_len}")
    labels=[-100]*len(prompt)+list(answer)+[eos]
    n=max_len-len(full)
    return full+[pad]*n, labels+[-100]*n, [1]*len(full)+[0]*n

def normalize_answer(s):
    s=unicodedata.normalize("NFKC",s).lower().strip()
    s=re.sub(r"[^\w\s.-]"," ",s); s=re.sub(r"\b(a|an|the)\b"," ",s)
    return " ".join(s.split())

class Records(Dataset):
    def __init__(self,path,split):
        root=Path(path).parent
        self.rows=[json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
        self.rows=[{**r,"image":str(root/r["image"])} for r in self.rows if r["split"]==split]
    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return self.rows[i]

class Collator:
    def __init__(self,tokenizer,image_processor,max_text=79):
        self.tok=tokenizer; self.proc=image_processor; self.max_text=max_text
        if self.tok.eos_token_id is None: raise ValueError("tokenizer needs EOS")
        if self.tok.pad_token_id is None: self.tok.pad_token=self.tok.eos_token
    def __call__(self,rows):
        ids=[]; labels=[]; masks=[]; images=[]
        for r in rows:
            prompt=self.tok.encode(f"Question: {r['question']} Answer:",add_special_tokens=False)
            answer=self.tok.encode(r["answer"],add_special_tokens=False)
            i,l,m=pack_ids(prompt,answer,self.tok.eos_token_id,self.tok.pad_token_id,self.max_text)
            ids.append(i); labels.append(l); masks.append(m); images.append(Image.open(r["image"]).convert("RGB"))
        pixels=self.proc(images=images,return_tensors="pt")["pixel_values"]
        return {"pixel_values":pixels,"input_ids":torch.tensor(ids),"labels":torch.tensor(labels),
                "attention_mask":torch.tensor(masks),"sample_ids":[r["sample_id"] for r in rows],
                "answers":[r["answer"] for r in rows]}
```

```python
# src/vlm.py
import math, torch
from torch import nn
from transformers import AutoModelForCausalLM, SiglipVisionModel

class PixelShuffleProjector(nn.Module):
    def __init__(self,vision_dim,language_dim,factor=2):
        super().__init__(); self.factor=factor; self.proj=nn.Linear(vision_dim*factor*factor,language_dim,bias=False)
    def forward(self,x):
        B,N,D=x.shape; side=math.isqrt(N)
        if side*side!=N or side%self.factor: raise ValueError(f"cannot shuffle N={N} factor={self.factor}")
        f=self.factor
        x=x.view(B,side,side,D).reshape(B,side//f,f,side//f,f,D).permute(0,1,3,2,4,5).contiguous()
        return self.proj(x.view(B,(side//f)**2,D*f*f))

class LocalVLM(nn.Module):
    def __init__(self,vision_path,lm_path,factor=2):
        super().__init__(); self.vision=SiglipVisionModel.from_pretrained(vision_path,local_files_only=True)
        self.lm=AutoModelForCausalLM.from_pretrained(lm_path,local_files_only=True)
        vd=self.vision.config.hidden_size; ld=self.lm.config.hidden_size
        self.projector=PixelShuffleProjector(vd,ld,factor); self.factor=factor
    def set_mode(self,mode):
        for p in self.parameters(): p.requires_grad=False
        for p in self.projector.parameters(): p.requires_grad=True
        if mode=="light":
            layers=getattr(self.lm.model,"layers",None)
            if layers is None: raise RuntimeError("LM layer path changed; audit fixed transformers version")
            for p in layers[-1].parameters(): p.requires_grad=True
            # SmolLM2-135M 固定配置 tie_word_embeddings=true；不解冻 lm_head，
            # 否则同一 Parameter 会把 input embeddings 一并解冻。
            if any(p.requires_grad for p in self.lm.get_input_embeddings().parameters()):
                raise RuntimeError("light mode must keep tied input/output embeddings frozen")
        elif mode!="projector": raise ValueError(mode)
    def visual(self,pixels):
        if any(p.requires_grad for p in self.vision.parameters()): h=self.vision(pixel_values=pixels).last_hidden_state
        else:
            with torch.no_grad(): h=self.vision(pixel_values=pixels).last_hidden_state
        return self.projector(h)
    def forward(self,pixel_values,input_ids,attention_mask,labels=None,zero_visual=False):
        v=self.visual(pixel_values)
        if zero_visual: v=torch.zeros_like(v)
        text=self.lm.get_input_embeddings()(input_ids); emb=torch.cat([v,text],dim=1)
        vm=torch.ones(v.shape[:2],dtype=attention_mask.dtype,device=attention_mask.device)
        mask=torch.cat([vm,attention_mask],dim=1); combined_labels=None
        if labels is not None:
            ignore=torch.full(v.shape[:2],-100,dtype=labels.dtype,device=labels.device)
            combined_labels=torch.cat([ignore,labels],dim=1)
        out=self.lm(inputs_embeds=emb,attention_mask=mask,labels=combined_labels,use_cache=False)
        return out,v
    @torch.no_grad()
    def greedy(self,pixels,ids,mask,eos,max_new=12,zero_visual=False):
        if ids.size(0)!=1: raise ValueError("reference greedy evaluator uses batch=1")
        length=int(mask[0].sum()); ids=ids[:,:length]; mask=mask[:,:length]
        for _ in range(max_new):
            out,_=self(pixels,ids,mask,None,zero_visual); nxt=out.logits[:,-1].argmax(-1,keepdim=True)
            ids=torch.cat([ids,nxt],1); mask=torch.cat([mask,torch.ones_like(nxt)],1)
            if int(nxt)==eos: break
        return ids[:,length:]
```

```python
# tests/test_contracts.py
import hashlib, json
import pytest, torch
from PIL import Image
from src.data import pack_ids,normalize_answer
from src.prepare_data import png_bytes, split_for_image, stable_id, write
from src.vlm import PixelShuffleProjector

def test_pack_answer_only():
    ids,labels,mask=pack_ids([10,11],[20,21],0,0,7)
    assert ids==[10,11,20,21,0,0,0]
    assert labels==[-100,-100,20,21,0,-100,-100]
    assert mask==[1,1,1,1,1,0,0]
    assert sum(x!=-100 for x in labels)==3
def test_projector_shape_grad():
    p=PixelShuffleProjector(8,6,2); x=torch.randn(2,196,8,requires_grad=True); y=p(x)
    assert y.shape==(2,49,6); y.square().mean().backward(); assert p.proj.weight.grad.abs().sum()>0
def test_normalize():
    assert normalize_answer(" The, RED square! ")=="red square"
def test_image_group_split_and_gold_independence(tmp_path):
    buckets={}
    for value in range(256):
        image=Image.new("RGB",(8,8),(value,0,0)); iid=hashlib.sha256(png_bytes(image)).hexdigest()
        buckets.setdefault(split_for_image(iid),(image,iid))
        if set(buckets)=={"train","dev"}: break
    assert set(buckets)=={"train","dev"}
    train_image,train_iid=buckets["train"]; dev_image,_=buckets["dev"]
    write([(train_image,"q1","a1"),(train_image,"q2","a2"),(dev_image,"q3","a3")],tmp_path/"ok",{"mode":"test"})
    rows=[json.loads(x) for x in (tmp_path/"ok/records.jsonl").read_text().splitlines()]
    train_images={r["image_sha256"] for r in rows if r["split"]=="train"}
    dev_images={r["image_sha256"] for r in rows if r["split"]=="dev"}
    assert train_iid in train_images and not (train_images & dev_images)
    assert stable_id(train_iid,"q1")==next(r["sample_id"] for r in rows if r["question"]=="q1")
    with pytest.raises(ValueError,match="conflicting gold"):
        write([(train_image,"same","a"),(train_image,"same","different")],tmp_path/"conflict",{"mode":"test"})
```

### 实现

HF causal LM 负责内部 shift，因此 labels 不手工移位。vision frozen 时可以 no_grad；LM frozen 时仍保留计算图到 inputs_embeds。projector 完整实现 196→49。

### 执行命令

`[两者/Python]`：

```bash
python -m py_compile src/prepare_data.py src/data.py src/vlm.py
python -m pytest -q tests/test_contracts.py
python -c "from transformers import AutoTokenizer,AutoImageProcessor; t=AutoTokenizer.from_pretrained('artifacts/tokenizer',local_files_only=True); p=AutoImageProcessor.from_pretrained('artifacts/siglip',local_files_only=True); print(t.eos_token_id,t.pad_token_id,p.size)"
```

### 预期观测（估算/示例，不是实测）

Day 2 的 pytest 4项通过；projector输出 `[2,49,6]`，image-group/gold-independence 测试通过；本地 tokenizer有EOS，processor尺寸能解析。任何 download 都不应在 `local_files_only=True` 时发生。

### 验收条件

mask有效数恰等于 answer+EOS；prompt/pad为-100；projector梯度非零；normalization测试明确。

### 若失败，按什么顺序查

本地 snapshot文件 → transformers版本 → tokenizer EOS/pad → 79 token截断 → 196是否含CLS → reshape/permute → HF内部shift。

### 当日证据清单

pytest输出、真实单样本 token长度/label有效数、processor config/hash、projector参数量。

## Day 3：完整 Trainer、单 batch 与 projector-only overfit

### 为什么做

证明视觉监督能到达 projector，冻结 backbone 不变化；128样本能过拟合前禁止正式训练。

### 输入与前置检查

Day 2测试通过；本地快照可离线加载；records至少128 train和16 dev。

### 本日要创建/修改的文件

创建本地训练入口。它故意不 import v0.1 的 BF16/W&B trainer：

```python
# src/train_vlm.py
import argparse, hashlib, json, math, os, platform, random, sys, time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import numpy as np, torch
import transformers
from PIL import Image
from transformers import AutoImageProcessor,AutoTokenizer
from src.data import Records,Collator,normalize_answer
from src.vlm import LocalVLM

def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def read_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def sha256_file(path):
    p=Path(path); h=hashlib.sha256()
    if not p.is_file(): raise FileNotFoundError(p)
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def file_fact(path):
    p=Path(path); return {"path":p.as_posix(),"bytes":p.stat().st_size,"sha256":sha256_file(p)}
def tree_fact(root,python_only=False):
    root=Path(root); files=[]; combined=hashlib.sha256(); total=0
    candidates=sorted((p for p in root.rglob("*") if p.is_file()),key=lambda p:p.as_posix())
    for p in candidates:
        rel=p.relative_to(root)
        if ".cache" in rel.parts or "__pycache__" in rel.parts: continue
        if python_only and p.suffix!=".py": continue
        f=file_fact(p); f["path"]=rel.as_posix(); files.append(f); total+=f["bytes"]
        combined.update(f["path"].encode()+b"\0"+f["sha256"].encode()+b"\n")
    if not files: raise RuntimeError(f"no files to hash under {root}")
    return {"root":root.as_posix(),"file_count":len(files),"bytes":total,
            "tree_sha256":combined.hexdigest(),"files":files}
def image_fact(records_path):
    records_path=Path(records_path); root=records_path.parent; rows=[json.loads(x) for x in records_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows: raise RuntimeError("empty records")
    seen={}; train_images=set(); dev_images=set(); combined=hashlib.sha256(); total=0
    for r in rows:
        if r["split"] not in {"train","dev"}: raise ValueError(f"invalid split: {r['split']}")
        p=root/r["image"]; digest=sha256_file(p)
        if digest!=r["image_sha256"]: raise ValueError(f"image hash drift: {p}")
        seen[p.as_posix()]={"path":p.as_posix(),"bytes":p.stat().st_size,"sha256":digest}
        (train_images if r["split"]=="train" else dev_images).add(digest)
    if train_images & dev_images: raise ValueError("image leakage across train/dev")
    for key in sorted(seen):
        f=seen[key]; total+=f["bytes"]; combined.update(key.encode()+b"\0"+f["sha256"].encode()+b"\n")
    return {"referenced_count":len(seen),"bytes":total,"tree_sha256":combined.hexdigest(),"split_overlap":0}
def runtime_fact():
    d={"python":platform.python_version(),"torch":str(torch.__version__),"numpy":np.__version__,
       "transformers":transformers.__version__,"datasets":version("datasets"),
       "huggingface_hub":version("huggingface_hub"),"pillow":version("Pillow"),
       "safetensors":version("safetensors"),"cuda_build":torch.version.cuda,
       "cuda_available":torch.cuda.is_available()}
    if d["cuda_available"]:
        d.update(gpu0=torch.cuda.get_device_name(0),capability0=list(torch.cuda.get_device_capability(0)))
    return d
def checked_lineage(a):
    sources_path=Path("manifests/sources.json"); summary_path=Path(a.records).parent/"data_summary.json"
    sources=read_json(sources_path); summary=read_json(summary_path); source=summary["source"]
    if source["mode"]=="hub":
        expected=sources["objects"]["dataset"]
        if (source["repo"],source["config"],source["resolved_revision"])!=(expected["repo"],expected["config"],expected["resolved"]):
            raise ValueError("dataset source/revision differs from sources.json")
    return {"schema_version":1,"formal_eligible":source["mode"]=="hub",
            "sources_manifest":file_fact(sources_path),"sources":sources["objects"],
            "data_source":source,"data_summary":file_fact(summary_path),"records":file_fact(a.records),
            "referenced_images":image_fact(a.records),"vision":tree_fact(a.vision),"lm":tree_fact(a.lm),
            "tokenizer":tree_fact(a.tokenizer),"code":tree_fact("src",python_only=True),
            "runtime":runtime_fact()}
def atomic_json(path,obj):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+"\n",encoding="utf-8"); os.replace(tmp,p)
def update_run_manifest(out,a,contract,state=None,status="RUNNING"):
    p=Path(out)/"run_manifest.json"; invocation={"utc":datetime.now(timezone.utc).isoformat(),
        "argv":sys.argv,"resume_checkpoint":a.resume}
    if p.exists():
        doc=read_json(p)
        if doc["contract"]!=contract: raise ValueError("run manifest lineage/contract changed")
        doc.setdefault("invocations",[]).append(invocation)
    else: doc={"schema_version":1,"status":status,"contract":contract,"invocations":[invocation]}
    doc["status"]=status
    if state is not None: doc["final_state"]=state
    else: doc.pop("final_state",None)
    atomic_json(p,doc)
class Stream:
    def __init__(self,data,seed):
        self.data=data; self.g=torch.Generator().manual_seed(seed); self.order=torch.randperm(len(data),generator=self.g); self.pos=0; self.epoch=0
    def next(self,n):
        if n<1 or not len(self.order): raise ValueError("positive batch and non-empty train split required")
        idx=[]
        while len(idx)<n:
            take=min(n-len(idx),len(self.order)-self.pos)
            idx.extend(self.order[self.pos:self.pos+take].tolist()); self.pos+=take
            if self.pos==len(self.order):
                self.epoch+=1; self.order=torch.randperm(len(self.data),generator=self.g); self.pos=0
        return [self.data[i] for i in idx]
    def state(self): return {"g":self.g.get_state(),"order":self.order,"pos":self.pos,"epoch":self.epoch}
    def load(self,s): self.g.set_state(s["g"]); self.order=s["order"]; self.pos=s["pos"]; self.epoch=s["epoch"]
def schedule(step,total,warmup=20):
    if step<warmup:return (step+1)/warmup
    r=min(1,(step-warmup)/max(1,total-warmup)); return 0.1+0.45*(1+math.cos(math.pi*r))
def target_token_weight(valid,total):
    if valid<1 or total<valid: raise ValueError("invalid target-token counts")
    return valid/total
def rng(): return {"py":random.getstate(),"np":np.random.get_state(),"cpu":torch.get_rng_state(),
                   "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
def set_rng(s):
    random.setstate(s["py"]); np.random.set_state(s["np"]); torch.set_rng_state(s["cpu"])
    if torch.cuda.is_available() and s["cuda"]: torch.cuda.set_rng_state_all(s["cuda"])
def save(path,m,opt,scaler,state,stream,contract):
    obj={"model":m.state_dict(),"optimizer":opt.state_dict(),"scaler":scaler.state_dict(),"state":state,
         "stream":stream.state(),"rng":rng(),"contract":contract}; p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".tmp"); torch.save(obj,tmp); os.replace(tmp,p)
def load_ck(path,m,opt=None,scaler=None,stream=None,expected_contract=None,expected_lineage=None):
    try: ck=torch.load(path,map_location="cpu",weights_only=False)
    except TypeError: ck=torch.load(path,map_location="cpu")
    if expected_contract is not None and ck.get("contract")!=expected_contract:
        raise ValueError("checkpoint lineage/contract changed")
    if expected_lineage is not None and ck.get("contract",{}).get("lineage")!=expected_lineage:
        raise ValueError("checkpoint lineage differs from eval inputs")
    m.load_state_dict(ck["model"])
    if opt is not None: opt.load_state_dict(ck["optimizer"]); scaler.load_state_dict(ck["scaler"]); stream.load(ck["stream"]); set_rng(ck["rng"])
    return ck
@torch.no_grad()
def val_loss(m,dev,collate,device,amp,n=64):
    m.eval(); total=0.; count=0
    for i in range(min(n,len(dev))):
        b=collate([dev[i]]); b={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in b.items()}
        with torch.cuda.amp.autocast(enabled=amp,dtype=torch.float16): out,_=m(b["pixel_values"],b["input_ids"],b["attention_mask"],b["labels"])
        valid=int((b["labels"]!=-100).sum()); total+=float(out.loss)*valid; count+=valid
    m.train(); return total/max(1,count)
def build(a):
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True); proc=AutoImageProcessor.from_pretrained(a.vision,local_files_only=True)
    return tok,Collator(tok,proc,a.max_text),LocalVLM(a.vision,a.lm)
def train(a):
    out=Path(a.output)
    if out.exists() and any(out.iterdir()) and not a.resume: raise RuntimeError("non-empty output; choose new run or --resume")
    if a.resume and Path(a.resume).resolve().parent!=out.resolve(): raise ValueError("resume checkpoint must be inside --output")
    if a.max_steps>a.schedule_steps: raise ValueError("max_steps cannot exceed frozen schedule_steps")
    if min(a.batch,a.accum,a.eval_every,a.save_every,a.max_consecutive_skips)<1: raise ValueError("batch/clock arguments must be positive")
    lineage=checked_lineage(a); seed_all(a.seed); device=torch.device("cuda")
    tok,collate,m=build(a); m.set_mode(a.mode); m.to(device)
    projector_ids={id(p) for p in m.projector.parameters()}; expected_ids=set(projector_ids)
    if a.mode=="light": expected_ids|={id(p) for p in m.lm.model.layers[-1].parameters()}
    actual_ids={id(p) for p in m.parameters() if p.requires_grad}
    if actual_ids!=expected_ids: raise RuntimeError("unexpected trainable parameter set")
    trainable_names=sorted(n for n,p in m.named_parameters() if p.requires_grad)
    train_ds=Records(a.records,"train"); dev=Records(a.records,"dev")
    if a.overfit: train_ds.rows=train_ds.rows[:a.overfit]
    if not train_ds.rows or not dev.rows: raise RuntimeError("empty train or dev split")
    stream=Stream(train_ds,a.seed+1); proj=list(m.projector.parameters()); other=[p for p in m.parameters() if p.requires_grad and id(p) not in projector_ids]
    groups=[{"params":proj,"lr":a.lr_projector}]+([{"params":other,"lr":a.lr_backbone}] if other else [])
    opt=torch.optim.AdamW(groups,weight_decay=0.01); scaler=torch.cuda.amp.GradScaler(enabled=a.precision=="fp16")
    training={k:getattr(a,k) for k in ["mode","seed","precision","max_text","lr_projector","lr_backbone",
              "schedule_steps","batch","accum","overfit","eval_every","save_every","max_consecutive_skips"]}
    training.update(world_size=1,global_batch_examples=a.batch*a.accum,reduction="global_target_token_mean",
                    trainable_parameters=trainable_names)
    contract={"lineage":lineage,"training":training}
    state={"step":0,"attempted":0,"effective_target_tokens":0,"attempted_target_tokens":0,
           "effective_samples":0,"attempted_samples":0,"consecutive_skips":0}
    if a.resume:
        ck=load_ck(a.resume,m,opt,scaler,stream,contract)
        state=ck["state"]
    out.mkdir(parents=True,exist_ok=True); update_run_manifest(out,a,contract); log=out/"metrics.jsonl"
    while state["step"]<a.max_steps:
        t0=time.perf_counter(); torch.cuda.reset_peak_memory_stats(device)
        factor=schedule(state["step"],a.schedule_steps)
        for i,g in enumerate(opt.param_groups): g["lr"]=(a.lr_projector if i==0 else a.lr_backbone)*factor
        opt.zero_grad(set_to_none=True); weighted_nll=0.; samples=[]; micro_batches=[]; target_tokens=0
        for _ in range(a.accum):
            b=collate(stream.next(a.batch)); samples+=b["sample_ids"]
            valid=int((b["labels"]!=-100).sum()); target_tokens+=valid; micro_batches.append((b,valid))
        if target_tokens<1: raise RuntimeError("accumulation window has zero supervised tokens")
        for b,valid in micro_batches:
            b={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in b.items()}
            with torch.cuda.amp.autocast(enabled=a.precision=="fp16",dtype=torch.float16): outp,v=m(b["pixel_values"],b["input_ids"],b["attention_mask"],b["labels"])
            if v.shape[1:]!=(49,m.lm.config.hidden_size): raise RuntimeError(f"visual shape {v.shape}")
            finite_loss=bool(torch.isfinite(outp.loss).item())
            if not finite_loss and a.precision=="fp32": raise FloatingPointError("non-finite FP32 loss")
            scaler.scale(outp.loss*target_token_weight(valid,target_tokens)).backward(); weighted_nll+=float(outp.loss.detach())*valid
        trainable=[p for p in m.parameters() if p.requires_grad]
        scaler.unscale_(opt); grad=float(torch.nn.utils.clip_grad_norm_(trainable,1.0,error_if_nonfinite=False))
        if not math.isfinite(grad) and a.precision=="fp32": raise FloatingPointError("non-finite FP32 grad norm")
        before=float(scaler.get_scale()); scaler.step(opt); scaler.update(); after=float(scaler.get_scale()); success=after>=before
        state["attempted"]+=1; state["attempted_target_tokens"]+=target_tokens; state["attempted_samples"]+=len(samples)
        if success:
            state["step"]+=1; state["effective_target_tokens"]+=target_tokens
            state["effective_samples"]+=len(samples); state["consecutive_skips"]=0
        else: state["consecutive_skips"]+=1
        torch.cuda.synchronize(device); elapsed=max(time.perf_counter()-t0,1e-9)
        nll=weighted_nll/target_tokens
        rec={**state,"train_nll":nll if math.isfinite(nll) else None,
             "grad_norm":grad if math.isfinite(grad) else None,"grad_finite":math.isfinite(grad),
             "loss_scale":after,"skipped":not success,"sample_ids":samples,
             "lr_projector":opt.param_groups[0]["lr"],"lr_backbone":opt.param_groups[1]["lr"] if len(opt.param_groups)>1 else None,
             "target_tokens_this_attempt":target_tokens,"step_ms":elapsed*1000,
             "effective_target_tokens_per_s":target_tokens/elapsed if success else 0.0,
             "effective_samples_per_s":len(samples)/elapsed if success else 0.0,
             "peak_allocated_mb":torch.cuda.max_memory_allocated(device)/2**20,
             "peak_reserved_mb":torch.cuda.max_memory_reserved(device)/2**20}
        if success and state["step"]%a.eval_every==0: rec["dev_nll"]=val_loss(m,dev,collate,device,a.precision=="fp16")
        with log.open("a",encoding="utf-8") as f:f.write(json.dumps(rec)+"\n")
        print(json.dumps(rec))
        if success and state["step"]%a.save_every==0: save(out/"last.pt",m,opt,scaler,state,stream,contract)
        if state["consecutive_skips"]>=a.max_consecutive_skips:
            save(out/"last.pt",m,opt,scaler,state,stream,contract)
            update_run_manifest(out,a,contract,state,status="FAILED_NUMERICS")
            raise FloatingPointError("consecutive GradScaler backoffs reached stop limit")
    save(out/"last.pt",m,opt,scaler,state,stream,contract)
    final_status="COMPLETED" if a.max_steps==a.schedule_steps else "PAUSED_AT_BOUNDARY"
    update_run_manifest(out,a,contract,state,status=final_status)
def evaluate(a):
    if a.eval_samples<1 or a.max_new<1: raise ValueError("eval-samples and max-new must be positive")
    lineage=checked_lineage(a); tok,collate,m=build(a); ck=load_ck(a.checkpoint,m,expected_lineage=lineage)
    device=torch.device("cuda"); m.to(device).eval(); dev=Records(a.records,"dev")
    if len({r["image_sha256"] for r in dev.rows})<2: raise RuntimeError("counterfactual eval needs at least two distinct dev images")
    result={}
    for condition in ["original","shuffled","blank","zero_visual"]:
        correct=invalid=0; total=min(a.eval_samples,len(dev))
        for i in range(total):
            text=dev[i]; image=text
            if condition=="shuffled":
                image=next(dev[(i+j)%len(dev)] for j in range(1,len(dev)) if dev[(i+j)%len(dev)]["image_sha256"]!=text["image_sha256"])
            with Image.open(image["image"]) as im:
                pixels=collate.proc(images=[im.convert("RGB")],return_tensors="pt")["pixel_values"].to(device)
            if condition=="blank": pixels.zero_()
            prompt=tok.encode(f"Question: {text['question']} Answer:",add_special_tokens=False)
            ids=torch.tensor([prompt],dtype=torch.long,device=device); mask=torch.ones_like(ids)
            gen=m.greedy(pixels,ids,mask,tok.eos_token_id,a.max_new,condition=="zero_visual")
            pred=tok.decode(gen[0],skip_special_tokens=True); gold=text["answer"]
            correct+=normalize_answer(pred)==normalize_answer(gold); invalid+=int(not pred.strip())
        result[condition]={"normalized_em":correct/max(1,total),"invalid_rate":invalid/max(1,total),"n":total}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result))
def parser():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
    for name in ["train","eval"]:
        q=sub.add_parser(name); q.add_argument("--vision",default="artifacts/siglip"); q.add_argument("--lm",default="artifacts/smollm2")
        q.add_argument("--tokenizer",default="artifacts/tokenizer"); q.add_argument("--records",default="data/records.jsonl"); q.add_argument("--max-text",type=int,default=79)
    t=sub.choices["train"]; t.add_argument("--mode",choices=["projector","light"],default="projector"); t.add_argument("--precision",choices=["fp32","fp16"],default="fp32")
    t.add_argument("--batch",type=int,default=2); t.add_argument("--accum",type=int,default=1); t.add_argument("--max-steps",type=int,required=True); t.add_argument("--schedule-steps",type=int,required=True)
    t.add_argument("--lr-projector",type=float,default=3e-4); t.add_argument("--lr-backbone",type=float,default=3e-5); t.add_argument("--seed",type=int,default=17)
    t.add_argument("--overfit",type=int,default=0); t.add_argument("--eval-every",type=int,default=50); t.add_argument("--save-every",type=int,default=50)
    t.add_argument("--max-consecutive-skips",type=int,default=8); t.add_argument("--resume"); t.add_argument("--output",required=True)
    e=sub.choices["eval"]; e.add_argument("--checkpoint",required=True); e.add_argument("--eval-samples",type=int,default=100); e.add_argument("--max-new",type=int,default=12); e.add_argument("--output",required=True)
    return p
if __name__=="__main__":
    a=parser().parse_args(); train(a) if a.cmd=="train" else evaluate(a)
```

创建训练时钟、lineage 与 tied-weight 回归测试：

```python
# tests/test_trainer.py
import pytest, torch
from torch import nn
from src.train_vlm import Stream, load_ck, target_token_weight, tree_fact
from src.vlm import LocalVLM

class DummyLM(nn.Module):
    def __init__(self):
        super().__init__(); self.model=nn.Module(); self.model.layers=nn.ModuleList([nn.Linear(4,4),nn.Linear(4,4)])
        self.embed=nn.Embedding(8,4); self.lm_head=nn.Linear(4,8,bias=False); self.lm_head.weight=self.embed.weight
    def get_input_embeddings(self): return self.embed

def dummy_vlm():
    m=LocalVLM.__new__(LocalVLM); nn.Module.__init__(m)
    m.vision=nn.Linear(4,4); m.projector=nn.Linear(4,4); m.lm=DummyLM(); return m

def test_global_target_token_mean_weights_gradients():
    short=torch.tensor(2.0,requires_grad=True); long=torch.tensor(4.0,requires_grad=True)
    loss=short*target_token_weight(1,4)+long*target_token_weight(3,4); loss.backward()
    assert float(loss)==3.5 and float(short.grad)==0.25 and float(long.grad)==0.75

def test_stream_order_does_not_depend_on_micro_batch_grouping():
    left=Stream(list(range(7)),19); right=Stream(list(range(7)),19)
    grouped=sum((left.next(2) for _ in range(7)),[]); singles=sum((right.next(1) for _ in range(14)),[])
    assert grouped==singles

def test_light_mode_keeps_tied_embeddings_frozen():
    m=dummy_vlm(); m.set_mode("light")
    assert not m.lm.embed.weight.requires_grad and not m.lm.lm_head.weight.requires_grad
    assert all(p.requires_grad for p in m.projector.parameters())
    assert all(p.requires_grad for p in m.lm.model.layers[-1].parameters())
    assert all(not p.requires_grad for p in m.lm.model.layers[0].parameters())

def test_lineage_hash_and_checkpoint_contract_guard(tmp_path):
    d=tmp_path/"tree"; d.mkdir(); (d/"x.bin").write_bytes(b"one"); first=tree_fact(d)
    (d/"x.bin").write_bytes(b"two"); assert tree_fact(d)["tree_sha256"]!=first["tree_sha256"]
    model=nn.Linear(2,2); ck=tmp_path/"last.pt"
    torch.save({"contract":{"revision":"fixed"},"model":model.state_dict()},ck)
    with pytest.raises(ValueError,match="contract changed"):
        load_ck(ck,model,expected_contract={"revision":"drifted"})
```

### 实现

`--schedule-steps` 固定整条实验 LR 时钟，所以分段 resume 不会因每段 `max_steps` 改变调度；`max_steps` 只是本次调用的停止边界，故意不进入不可变 contract。每个 accumulation window 先统计全部有效 answer+EOS token，再按 `n_i/Σn_i` 加权各 micro-batch mean loss，因此梯度是全局 target-token mean，而不是 micro-batch mean 的平均。FP16 非有限梯度交给 GradScaler 跳过并回退 scale，scheduler 只在成功 update 前进；连续 8 次回退才保存恢复点并止损。checkpoint/run manifest 会拒绝 code、data/images、模型/tokenizer文件、版本、batch、累积、精度、trainable集合或来源 revision 漂移。非空输出目录无 resume 会拒绝覆盖，Stream 保存 permutation、cursor 与 generator。独立生成评测只向模型提供 question prompt，绝不能把 collator 中含 gold answer 的训练序列送入生成器。

### 执行命令

先在 Week02 项目根目录重跑静态与 CPU 单测，再做单样本 GPU 前反向：

```bash
python -m py_compile src/prepare_data.py src/data.py src/vlm.py src/train_vlm.py tests/test_contracts.py tests/test_trainer.py
python -m pytest -q tests/test_contracts.py tests/test_trainer.py
```

然后用 1 step 检查真实 backbone 路径：

`[公司 Linux]`：

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train_vlm train --precision fp32 --mode projector --batch 1 --accum 1 --max-steps 1 --schedule-steps 1 --overfit 8 --eval-every 1 --save-every 1 --output runs/one_batch
CUDA_VISIBLE_DEVICES=0 python -m src.train_vlm train --precision fp32 --mode projector --batch 2 --accum 1 --max-steps 100 --schedule-steps 100 --overfit 128 --eval-every 20 --save-every 20 --output runs/projector_overfit
```

`[个人 PowerShell]`：

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
python -m src.train_vlm train --precision fp32 --mode projector --batch 1 --accum 1 --max-steps 1 --schedule-steps 1 --overfit 8 --eval-every 1 --save-every 1 --output runs/one_batch
python -m src.train_vlm train --precision fp32 --mode projector --batch 2 --accum 1 --max-steps 100 --schedule-steps 100 --overfit 128 --eval-every 20 --save-every 20 --output runs/projector_overfit
```

### 预期观测（估算/示例，不是实测）

visual shape `[B,49,576]`；projector grad finite/nonzero；冻结参数不应变化；JSONL 含 target-token-weighted `train_nll`、`step_ms`、有效 target tokens/s 和 allocated/reserved 显存；128样本 loss 应显著低于初值。绝对 loss/EM 不预填。

### 验收条件

单batch finite；有效 label数与answer+EOS一致；100 step loss较初始有清晰下降（建议至少50%，仅作为系统 gate）；projector参数变化、vision/LM参数 hash 不变。

### 若失败，按什么顺序查

单样本长度 → labels → visual shape → projector grad → LM是否被 no_grad → optimizer param groups → 固定同一batch。不要直接解冻全模型。

### 当日证据清单

one_batch/overfit JSONL、参数训练清单、冻结参数前后 hash、首批 shapes/label counts。

## Day 4：FP16、light-unfreeze、checkpoint/resume

### 为什么做

在 V100 的正确精度路径上完成200-step stability gate，并证明 optimizer/scaler/sampler/LR 能连续恢复。

### 输入与前置检查

projector overfit通过；显存从batch1开始；公司未运行W&B/Hub上传。

### 本日要创建/修改的文件

无需新增代码；在 `manifests/run-plan.json` 预注册两组唯一差异：mode、backbone LR 与为保持相同 global batch 所需的 micro-batch/accumulation。`src.train_vlm` 会自动把实际 trainable 参数名和全套 lineage 写进各 run 的 manifest。

### 实现

Run A projector-only；Run B projector + LM top block（不解冻 tied `lm_head`/input embedding）。两者同数据、seed、global batch、schedule steps，并使用相同的全局 target-token mean reduction。V100只用FP16；reserved显存止损线可设30.5GB，但以调度安全策略优先。

### 执行命令

`[公司 Linux]`：

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train_vlm train --precision fp16 --mode projector --batch 2 --accum 4 --max-steps 200 --schedule-steps 1000 --overfit 128 --eval-every 50 --save-every 50 --output runs/A_fp16_200
CUDA_VISIBLE_DEVICES=0 python -m src.train_vlm train --precision fp16 --mode light --batch 1 --accum 8 --max-steps 200 --schedule-steps 1000 --overfit 128 --eval-every 50 --save-every 50 --output runs/B_fp16_200
```

`[个人 PowerShell]` 同公开独立数据：

```powershell
python -m src.train_vlm train --precision fp16 --mode projector --batch 2 --accum 4 --max-steps 200 --schedule-steps 1000 --overfit 128 --eval-every 50 --save-every 50 --output runs/A_fp16_200
```

resume对照保持总 schedule=100：

`[两者/Python]`：

```bash
python -m src.train_vlm train --precision fp32 --mode projector --batch 1 --accum 1 --max-steps 100 --schedule-steps 100 --overfit 128 --eval-every 25 --save-every 50 --output runs/continuous100
python -m src.train_vlm train --precision fp32 --mode projector --batch 1 --accum 1 --max-steps 50 --schedule-steps 100 --overfit 128 --eval-every 25 --save-every 50 --output runs/resumed100
python -m src.train_vlm train --precision fp32 --mode projector --batch 1 --accum 1 --max-steps 100 --schedule-steps 100 --overfit 128 --eval-every 25 --save-every 50 --output runs/resumed100 --resume runs/resumed100/last.pt
```

### 预期观测（估算/示例，不是实测）

FP16 loss finite、scale不过度连续回退、warmup后skip比例低；单次溢出只形成 `skipped=true`，不会推进成功 step/LR 时钟，连续达到 `--max-consecutive-skips` 才止损。resume 的 step51 sample IDs、LR、scale与continuous一致。具体性能不预填。

### 验收条件

200 step无NaN/Inf；warmup后skip `<1%`；peak reserved不越预注册线；resume next sample IDs与状态连续；Run A/B只有冻结策略/LR/micro-batch-accum等已登记差异，global batch相同。

### 若失败，按什么顺序查

FP32同batch → CE/label有效数 → autocast是否FP16 → scaler/unscale/clip顺序 → micro-batch OOM → param groups → resume schedule/sample stream。V100禁用BF16与FA2。

### 当日证据清单

FP16 JSONL、skip统计、peak memory、A/B param groups、resume step51对照。

## Day 5：有界正式训练、独立 eval 与视觉反事实

### 为什么做

把“训练链能跑”与“模型确实使用视觉”分开检验。

### 输入与前置检查

公开 `clevr` 子集而非 synthetic；Day4 gate全过；run budget和输出目录获批。

### 本日要创建/修改的文件

只需手工新增每个 run 的 `decision.md`；`run_manifest.json` 由训练入口在首个 update 前原子生成，直接记录来源 requested/resolved revision、artifact/data/image/code SHA-256、环境版本、seed、precision、world_size=1、global batch、target-token reduction、max text 与 trainable 参数名。resume 在加载权重前比对同一 contract；只有到达完整 schedule 才标 `COMPLETED`，较早的稳定性边界标 `PAUSED_AT_BOUNDARY`。

### 实现

正式短训预算1000–5000成功updates，按可用时长提前预注册；不要事后按最好点改预算。独立eval在相同dev IDs上运行 original/shuffled/blank/zero_visual。

### 执行命令

`[公司 Linux]`：

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train_vlm train --precision fp16 --mode projector --batch 2 --accum 8 --max-steps 2000 --schedule-steps 2000 --eval-every 100 --save-every 100 --output runs/A_formal
CUDA_VISIBLE_DEVICES=0 python -m src.train_vlm train --precision fp16 --mode light --batch 1 --accum 16 --max-steps 2000 --schedule-steps 2000 --eval-every 100 --save-every 100 --output runs/B_formal
CUDA_VISIBLE_DEVICES=0 python -m src.train_vlm eval --checkpoint runs/A_formal/last.pt --eval-samples 500 --max-new 12 --output runs/A_formal/counterfactual.json
CUDA_VISIBLE_DEVICES=0 python -m src.train_vlm eval --checkpoint runs/B_formal/last.pt --eval-samples 500 --max-new 12 --output runs/B_formal/counterfactual.json
```

`[个人 PowerShell]` 缩小公开复现：

```powershell
python -m src.train_vlm train --precision fp16 --mode projector --batch 1 --accum 8 --max-steps 500 --schedule-steps 500 --eval-every 100 --save-every 100 --output runs/public_personal
python -m src.train_vlm eval --checkpoint runs/public_personal/last.pt --eval-samples 100 --max-new 12 --output runs/public_personal/counterfactual.json
```

### 预期观测（估算/示例，不是实测）

每个condition有normalized EM、invalid rate、n；training有answer NLL、grad/scale/LR、step time、有效 target tokens/s 与显存峰值。没有预设绝对EM。original若高于配对shuffle/blank才是视觉使用的初步证据。

### 验收条件

固定dev IDs、greedy decode和normalize；至少base/A/B与四种condition形成表；若original不优于反事实，结论写“未证明视觉依赖”；若只有synthetic，状态`INCONCLUSIVE`。

### 若失败，按什么顺序查

checkpoint meta与本地revision → prompt/EOS → sample/image置换映射 → normalized文本 → projector grad → 数据是否可由语言先验回答。错误抽30条分 object/attribute/count/spatial/OCR/reasoning/format/hallucination/question ignored。

### 当日证据清单

`metrics.jsonl`、`counterfactual.json`、固定sample ID清单、错误桶、full checkpoint、`decision.md`。公司端全部留公司机器。

## 最终止损与发布门

来源/许可/hash、gold-independent image-group split、packing单测、196→49shape、answer-only mask、全局 target-token mean、精确 trainable 集合、projector梯度/冻结不变、128 overfit、FP16有界跳步、lineage-gated resume、question-only original/shuffle/blank/zero对照缺一不可。任何命令未运行都标“待执行”；任何公开数据或模型缺失都不能用合成结果顶替正式结论。

技术链接核验日期：**2026-09-03**：[nanoVLM v0.1](https://github.com/huggingface/nanoVLM/releases/tag/v0.1)、[SigLIP](https://huggingface.co/google/siglip-base-patch16-224)、[SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)、[The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)、[PyTorch 2.1 AMP](https://pytorch.org/docs/2.1/notes/amp_examples.html)。
