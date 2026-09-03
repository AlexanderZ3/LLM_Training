# Week 01 实践篇：从空目录跑通 TinyStories → BPE → MiniGPT

> **全篇状态：待执行、待验证。** 任何“预期输出”都是契约示例，不是实测。公司硬件、拓扑和软件版本为用户自述，先探测再决定。公司机器上的代码、公开数据副本、日志、trace、图、checkpoint、路径和性能数字都不得导出；个人 5070 Ti 只能从公开源重新下载、重新运行。

## 0. 平台标签、预算与止损

- `[公司 Linux]`：主路径；复用已经激活且获批的 Conda 环境，不创建 Conda/venv，不升级公司 PyTorch 2.1，不绕过代理。
- `[个人 PowerShell]`：公开缩小复现；复用个人现有环境。5070 Ti 对 torch/CUDA wheel 的要求与 V100 不同，以个人探针和 PyTorch 官方矩阵为准。
- `[两者/Python]`：平台无关 Python 命令；Windows 在 PowerShell 中执行，Linux 在 bash 中执行。
- 估算磁盘：原始 TinyStories 与 token bins 需预留约 8–15 GB；1K BPE 教学子集可低至 1 GB。估算显存：20M 级模型、`T=256`、micro-batch 8 通常远低于 16 GB，但必须实测。估算时间：BPE 10–90 分钟、smoke 数分钟、正式短训 1–数小时，均受机器与配额影响。

立即止损：来源/license 未批准；下载内容像 HTML；`torch`/CUDA ABI 不通；依赖 dry-run 要替换 torch；CPU 单测失败；FP32 20 step 非有限；128-window 不能过拟合；resume 下一批 ID 不同；命令尝试外部上传。

## Day 0：冻结环境事实与空目录

### 为什么做

先证明现有解释器、torch 和磁盘可用，并保存变更前快照。没有这一步，后续报错无法区分代码与环境。

### 输入与前置检查

`[公司 Linux]` 在已经分配的作业/交互节点执行：

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
d={"torch":torch.__version__,"cuda_build":torch.version.cuda,
   "cuda_available":torch.cuda.is_available(),"gpu_count":torch.cuda.device_count()}
if torch.cuda.is_available():
    d.update(gpu=torch.cuda.get_device_name(0),
             capability=torch.cuda.get_device_capability(0),
             bf16=torch.cuda.is_bf16_supported())
print(json.dumps(d, ensure_ascii=False))
PY
```

`[个人 PowerShell]`：

```powershell
Get-Command python
python -V
python -m pip --version
python -m pip check
nvidia-smi
Get-PSDrive -PSProvider FileSystem
git --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None"
```

### 本日要创建/修改的文件

目录从零创建；下面命令先创建所有后续会引用的本地父路径。

`[公司 Linux]`：

```bash
mkdir -p week01-llm/{configs,data/raw,data/processed,src,tests,runs,manifests,third_party}
cd week01-llm
touch src/__init__.py
python -m pip freeze > manifests/pip-before.txt
```

`[个人 PowerShell]`：

```powershell
New-Item -ItemType Directory -Force week01-llm, week01-llm/configs, week01-llm/data/raw, week01-llm/data/processed, week01-llm/src, week01-llm/tests, week01-llm/runs, week01-llm/manifests, week01-llm/third_party | Out-Null
Set-Location week01-llm
New-Item -ItemType File -Force src/__init__.py | Out-Null
python -m pip freeze | Set-Content -Encoding utf8 manifests/pip-before.txt
```

最终目录职责：

```text
week01-llm/
  configs/tiny.json             完整模型/训练配置
  data/raw/                     固定 revision 的两个原始 txt
  data/processed/               tokenizer JSON 与 uint16 token stream
  manifests/                    来源、hash、环境与处理统计
  src/audit_data.py             UTF-8/大小/hash/schema 审计
  src/tokenizer.py              完整确定性 byte-BPE
  src/prepare.py                流式编码为 uint16 bin
  src/model.py                  decoder-only Transformer
  src/train.py                  train/eval/generate/checkpoint/resume
  tests/test_tokenizer.py       BPE round-trip/确定性/序列化
  tests/test_model.py           shape/causal/oracle/gradient
  tests/test_resume.py          lineage 漂移拒绝/下一批连续性
  runs/                         本机实验；公司产物绝不外传
```

### 实现

本日只建目录，不写训练代码。依赖最小集合为 `numpy, pytest, huggingface_hub`；PyTorch 已由当前环境提供。

### 执行命令

`[两者/Python]` 先只看 resolver，确认不替换 torch：

```bash
python -m pip install --dry-run "numpy<2" "pytest<9" "huggingface_hub==0.26.2"
```

公司只有在镜像/变更获批时才安装；个人也先看 dry-run。若缺包且解析不碰 torch，执行：

```bash
python -m pip install "numpy<2" "pytest<9" "huggingface_hub==0.26.2"
python -m pip check
```

回滚方法不是盲目 uninstall：对照 `manifests/pip-before.txt`，由公司管理员恢复批准 lock；个人环境按自己的 lock 重装。不得用 `--no-deps` 掩盖不满足的依赖。

### 预期观测（估算/示例，不是实测）

公司 V100 应报告 capability `(7,0)`、FP16 可用、BF16 通常为 false；若不是，记录事实，不改代码伪装。个人 5070 Ti 的 capability/torch 版本以实际输出为准。

### 验收条件

`pip check` 通过；磁盘足够；公司 torch 仍为 2.1.x；路径已创建；`pip-before.txt` 非空。

### 若失败，按什么顺序查

解释器是否为预期 Conda → `pip check` → torch import → driver/runtime → 调度是否分配 GPU → 管理员提供兼容 wheel。公司端不自行升级驱动/CUDA。

### 当日证据清单

`manifests/pip-before.txt`、终端环境摘要、可用空间、执行节点与日期（都留执行机器本地）。

## Day 1：锁定 TinyStories、下载并验证格式

### 为什么做

训练数据若是错误页、截断文件或漂移 revision，后续任何 loss 都不可解释。

### 输入与前置检查

固定对象：`roneneldan/TinyStories@f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`，只取：

| 文件 | 用途 | license 检查 | 完整性 |
|---|---|---|---|
| `TinyStoriesV2-GPT4-train.txt` | 训练 tokenizer 与 LM | 保存 dataset card；公司审批 CDLA-Sharing-1.0 | SHA-256、bytes、UTF-8、行/空行、首尾抽样 |
| `TinyStoriesV2-GPT4-valid.txt` | 独立 validation | 同上 | 同上，且 hash 不同 |

先解析远端 revision，不下载内容：

`[公司 Linux]` 或 `[个人 PowerShell]`：

```bash
git ls-remote https://huggingface.co/datasets/roneneldan/TinyStories refs/heads/main
```

输出若不是预期 hash，不要改用新 HEAD；仍按固定 revision 下载，或在 `source.json` 中走版本变更评审。

### 本日要创建/修改的文件

创建 `src/audit_data.py`：

```python
# src/audit_data.py
import argparse, hashlib, json
from pathlib import Path

def inspect(path: Path):
    h=hashlib.sha256(); size=0; lines=0; blank=0; head=b""; tail=b""
    decoder_ok=True
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            if not head: head=chunk[:256]
            tail=(tail+chunk)[-256:]
            h.update(chunk); size+=len(chunk); lines+=chunk.count(b"\n")
    try:
        with path.open("r", encoding="utf-8", errors="strict") as f:
            for line in f:
                blank += int(not line.strip())
    except UnicodeDecodeError:
        decoder_ok=False
    probe=head.lstrip().lower()
    return {"path":str(path),"sha256":h.hexdigest(),"bytes":size,
            "newlines":lines,"blank_lines":blank,"utf8":decoder_ok,
            "looks_like_html":probe.startswith(b"<!doctype") or probe.startswith(b"<html"),
            "head_hex":head[:32].hex(),"tail_hex":tail[-32:].hex()}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train", required=True); p.add_argument("--valid", required=True)
    p.add_argument("--revision", required=True); p.add_argument("--resolved-revision", required=True)
    p.add_argument("--out", required=True)
    a=p.parse_args(); train=inspect(Path(a.train)); valid=inspect(Path(a.valid))
    assert train["bytes"]>0 and valid["bytes"]>0
    assert train["utf8"] and valid["utf8"]
    assert not train["looks_like_html"] and not valid["looks_like_html"]
    assert train["sha256"] != valid["sha256"]
    if len(a.resolved_revision)!=40 or any(c not in "0123456789abcdef" for c in a.resolved_revision):
        raise ValueError("resolved revision must be a 40-character lowercase commit")
    if len(a.revision)==40 and a.revision!=a.resolved_revision:
        raise ValueError("requested commit did not resolve to itself")
    out={"dataset":"roneneldan/TinyStories","requested_revision":a.revision,
         "resolved_revision":a.resolved_revision,
         "files":{"train":train,"valid":valid}}
    Path(a.out).write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False))
if __name__=="__main__": main()
```

### 实现

`audit_data.py` 全量流式 hash，严格 UTF-8 解码，拒绝空文件与常见 HTML 错误页；不会打印真实文本。

### 执行命令

无论网络是否可达，先创建 Day 2 必用且不含外部数据的 toy oracle；这一步不是网络失败后的临时分支：

`[公司 Linux]`：

```bash
python -c 'from pathlib import Path; p=Path("data/raw"); (p/"toy-train.txt").write_text("Once upon a time, a cat ran.\nA child smiled.\n",encoding="utf-8"); (p/"toy-valid.txt").write_text("A dog slept.\n",encoding="utf-8")'
```

`[个人 PowerShell]`：

```powershell
python -c "from pathlib import Path; p=Path('data/raw'); (p/'toy-train.txt').write_text('Once upon a time, a cat ran.\nA child smiled.\n',encoding='utf-8'); (p/'toy-valid.txt').write_text('A dog slept.\n',encoding='utf-8')"
```

网络获批时，再解析并下载固定 revision：

`[公司 Linux]` 或 `[个人 PowerShell]`（在 `week01-llm` 根目录）：

```bash
python -c "from huggingface_hub import HfApi; r='f54c09fd23315a6f9c86f9dc80f725de7d8f9c64'; got=HfApi().dataset_info('roneneldan/TinyStories',revision=r).sha; assert got==r,(got,r); print(got)"
python -c "from huggingface_hub import hf_hub_download as d; [d(repo_id='roneneldan/TinyStories',repo_type='dataset',revision='f54c09fd23315a6f9c86f9dc80f725de7d8f9c64',filename=f,local_dir='data/raw') for f in ('TinyStoriesV2-GPT4-train.txt','TinyStoriesV2-GPT4-valid.txt')]"
python -m src.audit_data --train data/raw/TinyStoriesV2-GPT4-train.txt --valid data/raw/TinyStoriesV2-GPT4-valid.txt --revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 --resolved-revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 --out manifests/raw.json
```

公司网络不可达：只能让管理员把相同 revision 的两个公开文件与其已核验的 resolved revision 记录放入已经创建的 `data/raw/`，然后仍运行上面的 `python -m src.audit_data ...`。不得用个人网盘/U 盘/聊天转运。只有 toy 文件时明确只过代码门。

### 预期观测（估算/示例，不是实测）

`raw.json` 有两个不同的 64 位 hex SHA-256、非零 bytes、`utf8=true`、`looks_like_html=false`。具体 bytes/行数不得预填。

### 验收条件

公开文件审计全通过才可形成正式 TinyStories 结论；toy 只允许进入 Day 2–4 correctness，状态保持 `INCONCLUSIVE`。

### 若失败，按什么顺序查

DNS/TLS/403/timeout 分类 → 固定 revision 是否存在 → 文件是否 LFS 错误指针/HTML → 磁盘是否满 → 管理员镜像。绝不关闭 TLS 校验或绕代理。

### 当日证据清单

dataset card/许可审批记录、`manifests/raw.json`、下载工具版本、请求 revision 与 resolved revision。

## Day 2：完整实现、测试、训练并保存 byte-BPE

### 为什么做

Tokenizer 是训练所得的数据组件。今天要覆盖 byte 初始词表、pair 统计、稳定 tie-break、merge rank、encode/decode、special token、save/load 与 hash 稳定性，不能以 `AutoTokenizer` 代替。

### 输入与前置检查

确认 Day 1 的公开 train 文件存在；若只有 toy，命令切换相应路径并保留 `INCONCLUSIVE`。

### 本日要创建/修改的文件

创建完整实现：

```python
# src/tokenizer.py
import argparse, hashlib, json, re
from collections import Counter
from pathlib import Path

PATTERN=re.compile(r"\s+|[^\s]+")
SPECIAL="<|endoftext|>"

def merge_once(seq, pair, new_id):
    out=[]; i=0
    while i < len(seq):
        if i+1 < len(seq) and (seq[i],seq[i+1]) == pair:
            out.append(new_id); i+=2
        else:
            out.append(seq[i]); i+=1
    return tuple(out)

def valid_utf8_prefix(raw):
    while True:
        try: return raw.decode("utf-8")
        except UnicodeDecodeError as e:
            if e.end == len(raw): raw=raw[:e.start]
            else: raise

class ByteBPE:
    def __init__(self, merges=()):
        self.merges=[(int(a),int(b),int(n)) for a,b,n in merges]
        self.rank={(a,b):i for i,(a,b,_) in enumerate(self.merges)}
        self.new_id={(a,b):n for a,b,n in self.merges}
        self.vocab={i:bytes([i]) for i in range(256)}
        for a,b,n in self.merges:
            if a not in self.vocab or b not in self.vocab: raise ValueError("bad merge order")
            self.vocab[n]=self.vocab[a]+self.vocab[b]
        self.eot_id=256+len(self.merges)
    @property
    def vocab_size(self): return self.eot_id+1
    @classmethod
    def train_file(cls, path, target_base_vocab, max_bytes=None):
        if target_base_vocab < 256: raise ValueError("target_base_vocab must be >=256")
        with open(path,"rb") as f: raw=f.read() if max_bytes is None else f.read(max_bytes)
        text=valid_utf8_prefix(raw)
        freq=Counter()
        for segment in text.split(SPECIAL):
            for m in PATTERN.finditer(segment): freq[tuple(m.group(0).encode("utf-8"))]+=1
        merges=[]
        for new_id in range(256,target_base_vocab):
            counts=Counter()
            for piece,n in freq.items():
                for pair in zip(piece,piece[1:]): counts[pair]+=n
            if not counts: break
            best_count=max(counts.values())
            pair=min(p for p,c in counts.items() if c==best_count)
            updated=Counter()
            for piece,n in sorted(freq.items()): updated[merge_once(piece,pair,new_id)]+=n
            freq=updated; merges.append((pair[0],pair[1],new_id))
        return cls(merges)
    def _encode_plain(self, text):
        out=[]
        for m in PATTERN.finditer(text):
            ids=tuple(m.group(0).encode("utf-8"))
            while len(ids)>1:
                candidates=[(self.rank[p],p) for p in zip(ids,ids[1:]) if p in self.rank]
                if not candidates: break
                _,pair=min(candidates)
                ids=merge_once(ids,pair,self.new_id[pair])
            out.extend(ids)
        return out
    def encode(self, text, allowed_special=False, add_eot=False):
        if allowed_special:
            parts=text.split(SPECIAL); out=[]
            for i,part in enumerate(parts):
                out.extend(self._encode_plain(part))
                if i+1<len(parts): out.append(self.eot_id)
        else:
            out=self._encode_plain(text)
        if add_eot: out.append(self.eot_id)
        return out
    def decode(self, ids, skip_special=False):
        pieces=[]
        for i in ids:
            i=int(i)
            if i==self.eot_id:
                if not skip_special: pieces.append(SPECIAL.encode("utf-8"))
            elif i in self.vocab: pieces.append(self.vocab[i])
            else: raise ValueError(f"unknown token id {i}")
        return b"".join(pieces).decode("utf-8",errors="strict")
    def save(self,path):
        obj={"format":"week01-byte-bpe-v1","pretokenizer":r"\s+|[^\s]+",
             "merges":[list(x) for x in self.merges],
             "special":{"text":SPECIAL,"id":self.eot_id}}
        Path(path).write_text(json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n",encoding="utf-8")
    @classmethod
    def load(cls,path):
        obj=json.loads(Path(path).read_text(encoding="utf-8"))
        if obj["format"]!="week01-byte-bpe-v1": raise ValueError("unsupported format")
        tok=cls(obj["merges"])
        if obj["special"]!={"text":SPECIAL,"id":tok.eot_id}: raise ValueError("special contract mismatch")
        return tok

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
    t=sub.add_parser("train"); t.add_argument("--input",required=True); t.add_argument("--vocab-size",type=int,required=True)
    t.add_argument("--max-bytes",type=int); t.add_argument("--output",required=True)
    a=sub.add_parser("audit"); a.add_argument("--model",required=True); a.add_argument("--input",required=True); a.add_argument("--max-chars",type=int,default=200000)
    args=p.parse_args()
    if args.cmd=="train":
        tok=ByteBPE.train_file(args.input,args.vocab_size,args.max_bytes); tok.save(args.output)
        print(json.dumps({"base_vocab":tok.eot_id,"vocab_with_eot":tok.vocab_size,"sha256":sha(args.output)}))
    else:
        tok=ByteBPE.load(args.model)
        text=Path(args.input).read_text(encoding="utf-8")[:args.max_chars]
        ids=tok.encode(text,allowed_special=True); restored=tok.decode(ids)
        assert restored==text
        print(json.dumps({"chars":len(text),"utf8_bytes":len(text.encode()),"tokens":len(ids),
                          "tokens_per_char":len(ids)/max(1,len(text)),"roundtrip":True,"sha256":sha(args.model)}))
if __name__=="__main__": main()
```

创建测试：

```python
# tests/test_tokenizer.py
from pathlib import Path
from src.tokenizer import ByteBPE, SPECIAL, sha

CASES=["", "a", "abababab", "hello world\n", "中文与 emoji🙂", "123  45\tX", SPECIAL]

def train(tmp_path):
    p=tmp_path/"toy.txt"; p.write_text("abab abab\nhello hello\n中文🙂\n",encoding="utf-8")
    return p,ByteBPE.train_file(p,270)

def test_roundtrip(tmp_path):
    _,tok=train(tmp_path)
    for s in CASES:
        ids=tok.encode(s,allowed_special=(SPECIAL in s))
        assert tok.decode(ids)==s
        assert all(0<=i<tok.vocab_size for i in ids)

def test_save_load_and_determinism(tmp_path):
    p,t1=train(tmp_path); t2=ByteBPE.train_file(p,270)
    a=tmp_path/"a.json"; b=tmp_path/"b.json"; t1.save(a); t2.save(b)
    assert sha(a)==sha(b)
    loaded=ByteBPE.load(a)
    for s in CASES: assert loaded.encode(s,allowed_special=(SPECIAL in s))==t1.encode(s,allowed_special=(SPECIAL in s))

def test_merge_example(tmp_path):
    p=tmp_path/"x.txt"; p.write_text("abababab",encoding="utf-8")
    tok=ByteBPE.train_file(p,258)
    assert tok.decode(tok.encode("abab"))=="abab"
```

### 实现

实现采用“唯一 piece 频数 + 每轮重算 pair”的 correctness-first 算法；不会持有全量 token 序列，但大词表仍可能慢。先 toy 270，再公开子集 1K；只有 1K 正确后才在有界子集训练 4096。`--max-bytes` 与输入 SHA 都属于 tokenizer 身份。

### 执行命令

`[两者/Python]`：

```bash
python -m pytest -q tests/test_tokenizer.py
python -m src.tokenizer train --input data/raw/toy-train.txt --vocab-size 270 --output data/processed/toy-bpe.json
python -m src.tokenizer audit --model data/processed/toy-bpe.json --input data/raw/toy-valid.txt
python -m src.tokenizer train --input data/raw/TinyStoriesV2-GPT4-train.txt --vocab-size 1024 --max-bytes 8388608 --output data/processed/bpe1k.json
python -m src.tokenizer audit --model data/processed/bpe1k.json --input data/raw/TinyStoriesV2-GPT4-valid.txt
python -m src.tokenizer train --input data/raw/TinyStoriesV2-GPT4-train.txt --vocab-size 4096 --max-bytes 67108864 --output data/processed/bpe4096.json
python -m src.tokenizer audit --model data/processed/bpe4096.json --input data/raw/TinyStoriesV2-GPT4-valid.txt
```

前两条 toy 命令要求 Day 1 已创建 toy 文件；若公开数据已就绪仍保留 toy oracle。4096/64 MiB 是有界教学配置，不声称等价于全语料 tokenizer；若决定全量训练，去掉 `--max-bytes`，重新生成 hash 并重新编码所有 bins。

### 预期观测（估算/示例，不是实测）

pytest 全绿；audit 输出 `roundtrip:true`、OOV 隐含为 0、tokens/char；两次相同训练生成相同 hash。具体压缩率不得预填。

### 验收条件

所有 CASES 可逆；EOT 既可显式编码又不参与 merge；save/load IDs 一致；相同输入重复训练 hash 相同；4096 tokenizer hash 落 manifest。

### 若失败，按什么顺序查

UTF-8 原始审计 → 截断是否切到多字节尾部 → pre-tokenizer 是否 train/encode 相同 → tie-break → merge rank 应用顺序 → JSON 是否稳定排序 → special 是否被当普通 bytes。

### 当日证据清单

测试输出、1K/4096 tokenizer JSON 与 SHA-256、训练输入 hash/max_bytes、validation tokens/char（全部留本机）。

## Day 3：流式编码、模型实现与 CPU 前反向

### 为什么做

把 tokenizer 与模型通过明确的 uint16/vocab contract 接起来，并在 GPU 长训前用 CPU 证明 shift、causal attention、shape 与梯度正确。

### 输入与前置检查

`bpe4096.json`、两个公开 txt 已通过前两日 gate；`vocab_with_eot` 应为 4097，且小于 65536。

### 本日要创建/修改的文件

创建流式编码器：

```python
# src/prepare.py
import argparse, array, hashlib, json, sys
from pathlib import Path
from src.tokenizer import ByteBPE, sha

def encode_file(tok, src, dst):
    n=0; max_id=-1; Path(dst).parent.mkdir(parents=True,exist_ok=True)
    with open(src,"r",encoding="utf-8",errors="strict",newline="") as fi, open(dst,"wb") as fo:
        for line in fi:
            ids=tok.encode(line,allowed_special=True)
            if ids:
                if max(ids)>=65536: raise ValueError("uint16 overflow")
                a=array.array("H",ids)
                if sys.byteorder!="little": a.byteswap()
                a.tofile(fo); n+=len(ids); max_id=max(max_id,max(ids))
    return {"path":str(dst),"tokens":n,"max_id":max_id,"dtype":"uint16-le","sha256":sha(dst)}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--tokenizer",required=True); p.add_argument("--train",required=True)
    p.add_argument("--valid",required=True); p.add_argument("--out-dir",required=True); p.add_argument("--manifest",required=True)
    a=p.parse_args(); tok=ByteBPE.load(a.tokenizer); out=Path(a.out_dir)
    result={"tokenizer_sha256":sha(a.tokenizer),"vocab_size":tok.vocab_size,
            "train":encode_file(tok,a.train,out/"train.bin"),"valid":encode_file(tok,a.valid,out/"valid.bin")}
    assert result["train"]["tokens"]>257 and result["valid"]["tokens"]>257
    Path(a.manifest).write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result))
if __name__=="__main__": main()
```

创建模型：

```python
# src/model.py
import math, torch
from torch import nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self,d_model,n_head,max_seq,dropout):
        super().__init__(); assert d_model%n_head==0
        self.n_head=n_head; self.dh=d_model//n_head
        self.qkv=nn.Linear(d_model,3*d_model); self.proj=nn.Linear(d_model,d_model)
        self.drop=nn.Dropout(dropout)
        self.register_buffer("causal",torch.tril(torch.ones(max_seq,max_seq,dtype=torch.bool)),persistent=False)
    def forward(self,x):
        B,T,D=x.shape
        q,k,v=self.qkv(x).view(B,T,3,self.n_head,self.dh).permute(2,0,3,1,4).unbind(0)
        scores=(q@k.transpose(-2,-1))/math.sqrt(self.dh)
        prob=scores.masked_fill(~self.causal[:T,:T],float("-inf")).softmax(-1)
        y=(self.drop(prob)@v).transpose(1,2).contiguous().view(B,T,D)
        return self.drop(self.proj(y))

class Block(nn.Module):
    def __init__(self,c):
        super().__init__(); d=c["d_model"]
        self.ln1=nn.LayerNorm(d); self.attn=CausalSelfAttention(d,c["n_head"],c["context_length"],c["dropout"])
        self.ln2=nn.LayerNorm(d); self.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d),nn.Dropout(c["dropout"]))
    def forward(self,x):
        x=x+self.attn(self.ln1(x))
        return x+self.mlp(self.ln2(x))

class TinyGPT(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=c
        self.tok=nn.Embedding(c["vocab_size"],c["d_model"]); self.pos=nn.Embedding(c["context_length"],c["d_model"])
        self.drop=nn.Dropout(c["dropout"]); self.blocks=nn.ModuleList([Block(c) for _ in range(c["n_layer"])])
        self.ln=nn.LayerNorm(c["d_model"]); self.head=nn.Linear(c["d_model"],c["vocab_size"],bias=False)
        self.head.weight=self.tok.weight; self.apply(self._init)
    def _init(self,m):
        if isinstance(m,(nn.Linear,nn.Embedding)): nn.init.normal_(m.weight,mean=0.0,std=0.02)
        if isinstance(m,nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self,ids,labels=None):
        B,T=ids.shape
        if T>self.c["context_length"]: raise ValueError("sequence too long")
        x=self.drop(self.tok(ids)+self.pos(torch.arange(T,device=ids.device))[None])
        for block in self.blocks: x=block(x)
        logits=self.head(self.ln(x)); loss=None
        if labels is not None: loss=F.cross_entropy(logits.float().reshape(-1,logits.size(-1)),labels.reshape(-1))
        return logits,loss
```

创建配置文件 `configs/tiny.json`：

```json
{
  "seed": 1337,
  "vocab_size": 4097,
  "context_length": 256,
  "n_layer": 6,
  "d_model": 384,
  "n_head": 6,
  "dropout": 0.0,
  "micro_batch": 8,
  "grad_acc_steps": 4,
  "lr": 0.0003,
  "min_lr": 0.00003,
  "warmup_steps": 100,
  "weight_decay": 0.1,
  "grad_clip": 1.0,
  "eval_batches": 20,
  "eval_interval": 100,
  "save_interval": 100,
  "train_bin": "data/processed/train.bin",
  "valid_bin": "data/processed/valid.bin",
  "tokenizer": "data/processed/bpe4096.json"
}
```

创建模型测试：

```python
# tests/test_model.py
import math, torch
from src.model import CausalSelfAttention, TinyGPT

def cfg(): return {"vocab_size":300,"context_length":8,"n_layer":2,"d_model":24,"n_head":3,"dropout":0.0}

def test_shape_grad_and_causality():
    torch.manual_seed(0); m=TinyGPT(cfg()).eval()
    a=torch.randint(0,300,(2,8)); b=a.clone(); b[:,5:]=torch.randint(0,300,(2,3))
    la,loss=m(a,a); lb,_=m(b,b)
    assert la.shape==(2,8,300) and torch.isfinite(loss)
    torch.testing.assert_close(la[:,:5],lb[:,:5],atol=1e-6,rtol=1e-6)
    loss.backward(); assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

def test_attention_matches_manual():
    torch.manual_seed(1); a=CausalSelfAttention(12,3,7,0.0).eval(); x=torch.randn(2,7,12)
    got=a(x); B,T,D=x.shape
    q,k,v=a.qkv(x).view(B,T,3,3,4).permute(2,0,3,1,4).unbind(0)
    s=q@k.transpose(-2,-1)/math.sqrt(4); p=s.masked_fill(~a.causal[:T,:T],float("-inf")).softmax(-1)
    ref=a.proj((p@v).transpose(1,2).contiguous().view(B,T,D))
    torch.testing.assert_close(got,ref,atol=1e-6,rtol=1e-6)
    assert torch.all(p.masked_select(torch.triu(torch.ones(T,T,dtype=torch.bool),1)[None,None]))==0
```

### 实现

`prepare.py` 逐行编码，保留原换行，且因 BPE 不跨 whitespace/non-whitespace piece，行边界不会改变 merge。`.bin` 固定 little-endian uint16；模型用显式 causal mask 和 FP32 CE。

### 执行命令

`[两者/Python]`：

```bash
python -m src.prepare --tokenizer data/processed/bpe4096.json --train data/raw/TinyStoriesV2-GPT4-train.txt --valid data/raw/TinyStoriesV2-GPT4-valid.txt --out-dir data/processed --manifest manifests/processed.json
python -m pytest -q tests/test_tokenizer.py tests/test_model.py
python -m py_compile src/audit_data.py src/tokenizer.py src/prepare.py src/model.py
```

### 预期观测（估算/示例，不是实测）

`processed.json` 中 train/valid token 数均大于 257、最大 ID `<4097`、hash 不同；pytest 全绿；参数梯度 finite。

### 验收条件

配置 vocab 与 tokenizer 完全一致；attention oracle `<1e-5`；未来 token 改变不影响此前 logits；train/valid bins 独立。

### 若失败，按什么顺序查

tokenizer hash → uint16 endian/max ID → config vocab → x/y 手工窗口 → causal mask 方向 → QKV permute → FP32 loss。

### 当日证据清单

`manifests/processed.json`、pytest 输出、参数量（由下一日脚本实算）、一个脱敏 toy batch shape。

## Day 4：完整训练器、AMP 与等价 resume

### 为什么做

把“能 backward”升级成具备独立 RNG、token 加权 eval、全状态 checkpoint、原子保存、AMP 和本地 JSONL 的训练状态机。

### 输入与前置检查

Day 3 全绿；配置 JSON 已去掉说明注释；公开 bins 与 tokenizer hash 已冻结。

### 本日要创建/修改的文件

创建完整入口：

```python
# src/train.py
import argparse, hashlib, json, math, os, platform, random, sys, time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
from src.model import TinyGPT
from src.tokenizer import ByteBPE

def load_cfg(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def sha256_file(path):
    path=Path(path); h=hashlib.sha256()
    if not path.is_file(): raise FileNotFoundError(path)
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def file_fact(path):
    p=Path(path)
    return {"path":p.as_posix(),"bytes":p.stat().st_size,"sha256":sha256_file(p)}
def code_fact(root="src"):
    root=Path(root); files=[]; combined=hashlib.sha256()
    for p in sorted(root.glob("*.py"),key=lambda x:x.as_posix()):
        f=file_fact(p); files.append(f)
        combined.update(p.relative_to(root).as_posix().encode()+b"\0"+f["sha256"].encode()+b"\n")
    if not files: raise RuntimeError("no src/*.py files to bind")
    return {"tree_sha256":combined.hexdigest(),"files":files}
def runtime_fact():
    d={"python":platform.python_version(),"torch":str(torch.__version__),"numpy":np.__version__,
       "huggingface_hub":version("huggingface_hub"),"cuda_build":torch.version.cuda,
       "cuda_available":torch.cuda.is_available()}
    if d["cuda_available"]:
        d.update(gpu0=torch.cuda.get_device_name(0),capability0=list(torch.cuda.get_device_capability(0)))
    return d
def checked_lineage(config_path,c):
    raw_path=Path("manifests/raw.json"); processed_path=Path("manifests/processed.json")
    raw=load_cfg(raw_path); processed=load_cfg(processed_path)
    fixed="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
    if raw.get("dataset")!="roneneldan/TinyStories" or raw.get("resolved_revision")!=fixed:
        raise ValueError("TinyStories source/revision drift")
    for split,f in raw["files"].items():
        if sha256_file(f["path"])!=f["sha256"]: raise ValueError(f"raw {split} hash drift")
    expected={"tokenizer_sha256":sha256_file(c["tokenizer"]),
              "train":sha256_file(c["train_bin"]),"valid":sha256_file(c["valid_bin"])}
    if processed["tokenizer_sha256"]!=expected["tokenizer_sha256"]: raise ValueError("tokenizer hash drift")
    for split in ("train","valid"):
        if processed[split]["sha256"]!=expected[split]: raise ValueError(f"{split}.bin hash drift")
    source={k:raw[k] for k in ("dataset","requested_revision","resolved_revision")}
    return {"schema_version":1,"source":source,"config":file_fact(config_path),"raw_manifest":file_fact(raw_path),
            "processed_manifest":file_fact(processed_path),"tokenizer":file_fact(c["tokenizer"]),
            "train_bin":file_fact(c["train_bin"]),"valid_bin":file_fact(c["valid_bin"]),
            "code":code_fact(),"runtime":runtime_fact()}
def atomic_json(path,obj):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+"\n",encoding="utf-8"); os.replace(tmp,p)
def update_run_manifest(out,a,contract,state=None,status="RUNNING"):
    p=Path(out)/"run_manifest.json"; invocation={"utc":datetime.now(timezone.utc).isoformat(),
        "argv":sys.argv,"resume_checkpoint":a.resume}
    if p.exists():
        doc=load_cfg(p)
        if doc["contract"]!=contract: raise ValueError("run manifest lineage/contract changed")
        doc.setdefault("invocations",[]).append(invocation)
    else:
        doc={"schema_version":1,"status":status,"contract":contract,"invocations":[invocation]}
    doc["status"]=status
    if state is not None: doc["final_state"]={k:int(v) for k,v in state.items() if isinstance(v,(int,bool))}
    else: doc.pop("final_state",None)
    atomic_json(p,doc)
def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def lr_at(step,c,total):
    if step<c["warmup_steps"]: return c["lr"]*(step+1)/max(1,c["warmup_steps"])
    r=min(1.0,(step-c["warmup_steps"])/max(1,total-c["warmup_steps"]))
    return c["min_lr"]+0.5*(c["lr"]-c["min_lr"])*(1+math.cos(math.pi*r))

class Batches:
    def __init__(self,path,T,seed,overfit_windows=0):
        self.data=np.memmap(path,dtype="<u2",mode="r"); self.T=T; self.g=torch.Generator().manual_seed(seed)
        if len(self.data)<=T: raise ValueError("token stream shorter than context")
        self.fixed=torch.linspace(0,len(self.data)-T-1,max(1,overfit_windows),dtype=torch.long) if overfit_windows else None
        self.cursor=0
    def get(self,B,device):
        if self.fixed is None: starts=torch.randint(0,len(self.data)-self.T,(B,),generator=self.g)
        else:
            starts=torch.stack([self.fixed[(self.cursor+i)%len(self.fixed)] for i in range(B)]); self.cursor=(self.cursor+B)%len(self.fixed)
        rows=[np.asarray(self.data[int(i):int(i)+self.T+1],dtype=np.int64) for i in starts]
        z=torch.from_numpy(np.stack(rows)); return z[:,:-1].to(device),z[:,1:].to(device),starts
    def state(self): return {"generator":self.g.get_state(),"cursor":self.cursor}
    def load(self,s): self.g.set_state(s["generator"]); self.cursor=s["cursor"]

def optimizer_for(model,c):
    decay=[p for _,p in model.named_parameters() if p.requires_grad and p.dim()>=2]
    no_decay=[p for _,p in model.named_parameters() if p.requires_grad and p.dim()<2]
    return torch.optim.AdamW([{"params":decay,"weight_decay":c["weight_decay"]},
                              {"params":no_decay,"weight_decay":0.0}],lr=c["lr"],betas=(0.9,0.95))

@torch.no_grad()
def evaluate(model,source,c,device,amp):
    model.eval(); total=0.0; count=0
    for _ in range(c["eval_batches"]):
        x,y,_=source.get(c["micro_batch"],device)
        with torch.cuda.amp.autocast(enabled=amp,dtype=torch.float16): logits,_=model(x)
        nll=torch.nn.functional.cross_entropy(logits.float().reshape(-1,logits.size(-1)),y.reshape(-1),reduction="sum")
        total+=float(nll); count+=y.numel()
    model.train(); nll=total/count
    return {"valid_nll":nll,"ppl":math.exp(nll) if nll<20 else None,"valid_tokens":count}

def rng_state():
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
            "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
def set_rng(s):
    random.setstate(s["python"]); np.random.set_state(s["numpy"]); torch.set_rng_state(s["torch"])
    if torch.cuda.is_available() and s["cuda"]: torch.cuda.set_rng_state_all(s["cuda"])

def save(path,model,opt,scaler,c,state,train_src,val_src,contract):
    obj={"model":model.state_dict(),"optimizer":opt.state_dict(),"scaler":scaler.state_dict(),"config":c,
         "state":state,"rng":rng_state(),"train_source":train_src.state(),"val_source":val_src.state(),
         "contract":contract}
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    torch.save(obj,tmp); os.replace(tmp,path)

def load_resume(path,model,opt,scaler,train_src,val_src,expected_contract):
    try: ck=torch.load(path,map_location="cpu",weights_only=False)
    except TypeError: ck=torch.load(path,map_location="cpu")
    if ck.get("contract")!=expected_contract: raise ValueError("checkpoint lineage/contract changed")
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
    scaler.load_state_dict(ck["scaler"]); train_src.load(ck["train_source"]); val_src.load(ck["val_source"]); set_rng(ck["rng"])
    return ck["state"]

def run_train(a):
    c=load_cfg(a.config); seed_all(c["seed"]); device=torch.device(a.device)
    if a.precision=="fp16" and device.type!="cuda": raise ValueError("fp16 requires CUDA")
    schedule_steps=a.schedule_steps or a.max_steps
    if a.max_steps>schedule_steps: raise ValueError("max_steps cannot exceed schedule_steps")
    out=Path(a.output)
    if out.exists() and any(out.iterdir()) and not a.resume: raise FileExistsError(f"refusing to overwrite {out}")
    if a.resume and Path(a.resume).resolve().parent!=out.resolve(): raise ValueError("resume checkpoint must be inside --output")
    lineage=checked_lineage(a.config,c)
    contract={"lineage":lineage,"precision":a.precision,"schedule_steps":schedule_steps,
              "overfit_windows":a.overfit_windows,"device_type":device.type,"world_size":1}
    amp=a.precision=="fp16"; model=TinyGPT(c).to(device); opt=optimizer_for(model,c)
    scaler=torch.cuda.amp.GradScaler(enabled=amp)
    train_src=Batches(c["train_bin"],c["context_length"],c["seed"]+1,a.overfit_windows)
    val_src=Batches(c["valid_bin"],c["context_length"],c["seed"]+2)
    state={"successful_step":0,"attempted_step":0,"effective_tokens":0,"attempted_tokens":0,
           "schedule_steps":schedule_steps}
    if a.resume:
        state=load_resume(a.resume,model,opt,scaler,train_src,val_src,contract)
        if state.get("schedule_steps")!=schedule_steps: raise ValueError("schedule_steps changed across resume")
    out.mkdir(parents=True,exist_ok=True); update_run_manifest(out,a,contract); log=out/"metrics.jsonl"
    print(json.dumps({"parameters":sum(p.numel() for p in model.parameters()),"trainable":sum(p.numel() for p in model.parameters() if p.requires_grad)}))
    while state["successful_step"]<a.max_steps:
        step=state["successful_step"]; lr=lr_at(step,c,schedule_steps)
        for g in opt.param_groups: g["lr"]=lr
        opt.zero_grad(set_to_none=True); loss_sum=0.0; id_hash=hashlib.sha256(); t0=time.perf_counter()
        if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
        for _ in range(c["grad_acc_steps"]):
            x,y,starts=train_src.get(c["micro_batch"],device); id_hash.update(starts.numpy().tobytes())
            with torch.cuda.amp.autocast(enabled=amp,dtype=torch.float16): _,loss=model(x,y)
            if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite loss at attempted {state['attempted_step']}")
            scaler.scale(loss/c["grad_acc_steps"]).backward(); loss_sum+=float(loss)
        scaler.unscale_(opt); grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),c["grad_clip"]))
        scale_before=float(scaler.get_scale()); scaler.step(opt); scaler.update(); scale_after=float(scaler.get_scale())
        success=scale_after>=scale_before; state["attempted_step"]+=1
        tokens=c["micro_batch"]*c["context_length"]*c["grad_acc_steps"]; state["attempted_tokens"]+=tokens
        if success: state["successful_step"]+=1; state["effective_tokens"]+=tokens
        if device.type=="cuda": torch.cuda.synchronize(device)
        rec={**state,"train_nll":loss_sum/c["grad_acc_steps"],"lr":lr,"grad_norm":grad,
             "loss_scale":scale_after,"step_skipped":not success,"batch_id_hash":id_hash.hexdigest(),
              "step_ms":1000*(time.perf_counter()-t0),
              "effective_tokens_per_s":tokens/max(time.perf_counter()-t0,1e-9) if success else 0.0,
             "peak_allocated_mb":torch.cuda.max_memory_allocated(device)/2**20 if device.type=="cuda" else 0,
             "peak_reserved_mb":torch.cuda.max_memory_reserved(device)/2**20 if device.type=="cuda" else 0}
        if success and state["successful_step"]%c["eval_interval"]==0: rec.update(evaluate(model,val_src,c,device,amp))
        with log.open("a",encoding="utf-8") as f: f.write(json.dumps(rec)+"\n")
        print(json.dumps(rec))
        if success and state["successful_step"]%c["save_interval"]==0: save(out/"last.pt",model,opt,scaler,c,state,train_src,val_src,contract)
    save(out/"last.pt",model,opt,scaler,c,state,train_src,val_src,contract)
    final_status="COMPLETED" if a.max_steps==schedule_steps else "PAUSED_AT_BOUNDARY"
    update_run_manifest(out,a,contract,state,status=final_status)

def load_for_infer(ckpt,device):
    try: ck=torch.load(ckpt,map_location="cpu",weights_only=False)
    except TypeError: ck=torch.load(ckpt,map_location="cpu")
    model=TinyGPT(ck["config"]).to(device); model.load_state_dict(ck["model"]); model.eval(); return ck,model

def run_eval(a):
    ck,m=load_for_infer(a.checkpoint,a.device); c=ck["config"]; src=Batches(c["valid_bin"],c["context_length"],c["seed"]+999)
    result=evaluate(m,src,c,torch.device(a.device),a.precision=="fp16")
    Path(a.output).write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result))

@torch.no_grad()
def run_generate(a):
    ck,m=load_for_infer(a.checkpoint,a.device); c=ck["config"]; tok=ByteBPE.load(c["tokenizer"]); device=torch.device(a.device)
    ids=tok.encode(a.prompt); x=torch.tensor(ids,dtype=torch.long,device=device)[None]
    for _ in range(a.max_new_tokens):
        logits,_=m(x[:,-c["context_length"]:]); nxt=logits[:,-1].argmax(-1,keepdim=True); x=torch.cat([x,nxt],1)
        if int(nxt)==tok.eot_id: break
    text=tok.decode(x[0].tolist()); Path(a.output).write_text(text,encoding="utf-8"); print(text)

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
    t=sub.add_parser("train"); t.add_argument("--config",required=True); t.add_argument("--device",default="cuda")
    t.add_argument("--precision",choices=["fp32","fp16"],default="fp32"); t.add_argument("--max-steps",type=int,required=True)
    t.add_argument("--schedule-steps",type=int,help="固定 LR schedule 总步数；分段 resume 时各段必须相同")
    t.add_argument("--output",required=True); t.add_argument("--resume"); t.add_argument("--overfit-windows",type=int,default=0)
    e=sub.add_parser("eval"); e.add_argument("--checkpoint",required=True); e.add_argument("--device",default="cuda")
    e.add_argument("--precision",choices=["fp32","fp16"],default="fp32"); e.add_argument("--output",required=True)
    g=sub.add_parser("generate"); g.add_argument("--checkpoint",required=True); g.add_argument("--device",default="cuda")
    g.add_argument("--prompt",required=True); g.add_argument("--max-new-tokens",type=int,default=80); g.add_argument("--output",required=True)
    a=p.parse_args(); {"train":run_train,"eval":run_eval,"generate":run_generate}[a.cmd](a)
if __name__=="__main__": main()
```

再创建 resume/lineage 回归测试：

```python
# tests/test_resume.py
import numpy as np, pytest, torch
from src.model import TinyGPT
from src.train import Batches, load_resume, optimizer_for, save, seed_all, sha256_file

def config(path):
    return {"seed":7,"vocab_size":64,"context_length":4,"n_layer":1,"d_model":8,"n_head":2,
            "dropout":0.0,"weight_decay":0.0,"lr":1e-3,"min_lr":1e-4,"warmup_steps":1,
            "grad_clip":1.0,"micro_batch":2,"grad_acc_steps":1,"eval_batches":1,
            "eval_interval":1,"save_interval":1,"train_bin":str(path),"valid_bin":str(path),
            "tokenizer":"unused-in-this-unit-test"}

def test_resume_next_batch_and_contract_guard(tmp_path):
    bin_path=tmp_path/"tokens.bin"; (np.arange(128,dtype=np.uint16)%63).astype("<u2").tofile(bin_path)
    c=config(bin_path); seed_all(c["seed"]); model=TinyGPT(c); opt=optimizer_for(model,c)
    scaler=torch.cuda.amp.GradScaler(enabled=False)
    train=Batches(bin_path,4,11); valid=Batches(bin_path,4,12)
    train.get(2,"cpu"); valid.get(2,"cpu")
    contract={"lineage":{"train_sha256":sha256_file(bin_path)},"schedule_steps":4}
    ckpt=tmp_path/"last.pt"; state={"successful_step":1,"attempted_step":1,
        "effective_tokens":8,"attempted_tokens":8,"schedule_steps":4}
    save(ckpt,model,opt,scaler,c,state,train,valid,contract)
    _,_,expected_train=train.get(2,"cpu"); _,_,expected_valid=valid.get(2,"cpu")

    model2=TinyGPT(c); opt2=optimizer_for(model2,c); scaler2=torch.cuda.amp.GradScaler(enabled=False)
    train2=Batches(bin_path,4,999); valid2=Batches(bin_path,4,999)
    got=load_resume(ckpt,model2,opt2,scaler2,train2,valid2,contract)
    _,_,actual_train=train2.get(2,"cpu"); _,_,actual_valid=valid2.get(2,"cpu")
    assert got==state and torch.equal(actual_train,expected_train) and torch.equal(actual_valid,expected_valid)
    assert train2.g.get_state().device.type=="cpu"
    with pytest.raises(ValueError,match="contract changed"):
        load_resume(ckpt,model2,opt2,scaler2,train2,valid2,{"lineage":{"train_sha256":"changed"}})
```

### 实现

这是本周可执行的核心链，不依赖假想 `src.*`。`successful_step` 只在 scaler 未回退时前进；`batch_id_hash` 用于 resume 对照；checkpoint 包括两套 batch generator 与所有 RNG。

### 执行命令

先静态检查与 CPU 单 batch/3 step：

`[两者/Python]`：

```bash
python -m py_compile src/train.py tests/test_resume.py
python -m pytest -q tests/test_resume.py
python -m src.train train --config configs/tiny.json --device cpu --precision fp32 --max-steps 3 --overfit-windows 128 --output runs/cpu3
```

再做单卡 FP32 20 step：

`[公司 Linux]`：

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train train --config configs/tiny.json --device cuda --precision fp32 --max-steps 20 --overfit-windows 128 --output runs/fp32_20
```

`[个人 PowerShell]`：

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
python -m src.train train --config configs/tiny.json --device cuda --precision fp32 --max-steps 20 --overfit-windows 128 --output runs/fp32_20
```

然后 FP16 200 step：

`[公司 Linux]`：

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train train --config configs/tiny.json --device cuda --precision fp16 --max-steps 200 --overfit-windows 128 --output runs/fp16_200
```

`[个人 PowerShell]`：

```powershell
python -m src.train train --config configs/tiny.json --device cuda --precision fp16 --max-steps 200 --overfit-windows 128 --output runs/fp16_200
```

resume A/B 要在两个干净输出目录、相同配置运行。为了 20 step 保存，将配置副本中的 `save_interval` 暂时改为 20，并把该副本另存 `configs/resume.json` 后执行：

`[两者/Python]`：

```bash
python -c "import json; p='configs/tiny.json'; c=json.load(open(p)); c['save_interval']=20; c['eval_interval']=20; open('configs/resume.json','w').write(json.dumps(c,indent=2)+'\n')"
python -m src.train train --config configs/resume.json --device cuda --precision fp32 --max-steps 40 --schedule-steps 40 --overfit-windows 128 --output runs/continuous40
python -m src.train train --config configs/resume.json --device cuda --precision fp32 --max-steps 20 --schedule-steps 40 --overfit-windows 128 --output runs/resumed40
python -m src.train train --config configs/resume.json --device cuda --precision fp32 --max-steps 40 --schedule-steps 40 --overfit-windows 128 --output runs/resumed40 --resume runs/resumed40/last.pt
```

### 预期观测（估算/示例，不是实测）

JSONL 每行包含 step、NLL、LR、grad norm、loss scale、skip、batch hash、时间和显存。128-window NLL 应明显下降；具体数值不预填。continuous 与 resumed 在下一批 hash 相同后，loss 相对差目标 `<1%`；确定性 CPU 路径应更接近。

### 验收条件

CPU 与 FP32 finite；128-window 明显过拟合；FP16 200 step finite，warmup 后 skipped `<1%`；unscale 在 clip 前；resume next-batch hash、LR、scale、step 连续。

### 若失败，按什么顺序查

CPU fixed batch → 首个 x/y → 参数 grad → FP32 GPU → autocast/loss float → scale/grad norm → checkpoint batch generator → RNG 保存时点。OOM 先降 `micro_batch`，再用 accumulation 保持 token budget。

### 当日证据清单

`runs/*/metrics.jsonl`、`last.pt`、配置副本、batch hash 对照、FP16 skip 比例与 peak memory；公司产物只留公司机器。

## Day 5：有界正式训练、独立 eval、生成与 Profiler

### 为什么做

把 correctness 证据和有限预算结果汇合；不以无限长跑或主观故事样例代替 validation。

### 输入与前置检查

Day 4 所有 gate 通过；公开 TinyStories train/valid 完整；运行配额获批；`runs/formal_fp16` 尚不存在或已明确是新 run。

### 本日要创建/修改的文件

无需新增核心代码。`src.train` 会在任何 optimizer update 前原子创建 `run_manifest.json`，绑定 config、`raw.json`/`processed.json`、tokenizer、train/valid bin、所有 `src/*.py`、Python/torch/NumPy/CUDA build、seed、precision、device type、schedule 与 world size=1；resume 会在加载权重前逐项重算并拒绝漂移。调用正常结束后，只有到达完整 schedule 才标 `COMPLETED`，较早预注册边界标 `PAUSED_AT_BOUNDARY`；异常中断仍为 `RUNNING`，不能伪装成成功。

### 实现

正式预算固定 `max_steps=5000`，不是效果承诺。FP32 reference 用相同代码跑 500 step，只用于数值/性能参考，不能和不同 token budget 的质量直接比较。

### 执行命令

`[公司 Linux]`：

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train train --config configs/tiny.json --device cuda --precision fp32 --max-steps 500 --output runs/reference_fp32
CUDA_VISIBLE_DEVICES=0 python -m src.train train --config configs/tiny.json --device cuda --precision fp16 --max-steps 5000 --output runs/formal_fp16
CUDA_VISIBLE_DEVICES=0 python -m src.train eval --checkpoint runs/formal_fp16/last.pt --device cuda --precision fp16 --output runs/formal_fp16/eval.json
CUDA_VISIBLE_DEVICES=0 python -m src.train generate --checkpoint runs/formal_fp16/last.pt --device cuda --prompt "Once upon a time" --max-new-tokens 80 --output runs/formal_fp16/generation.txt
```

`[个人 PowerShell]`：

```powershell
python -m src.train train --config configs/tiny.json --device cuda --precision fp16 --max-steps 1000 --output runs/public_personal
python -m src.train eval --checkpoint runs/public_personal/last.pt --device cuda --precision fp16 --output runs/public_personal/eval.json
python -m src.train generate --checkpoint runs/public_personal/last.pt --device cuda --prompt "Once upon a time" --max-new-tokens 80 --output runs/public_personal/generation.txt
```

个人命令是独立公开缩小复现，不接收公司 checkpoint 或数字。

短 Profiler 使用内置 PyTorch API时，先复制训练入口并仅在 10 个 step 包裹 `torch.profiler.profile`；本手册核心代码没有虚构 `--profile` 参数，因此不提供不可执行命令。最小可运行独立 kernel 观测：

`[两者/Python]`：

```bash
python -c "import torch; from torch.profiler import profile,ProfilerActivity; x=torch.randn(2048,2048,device='cuda'); acts=[ProfilerActivity.CPU,ProfilerActivity.CUDA]; p=profile(activities=acts,record_shapes=True,profile_memory=True); p.__enter__(); y=x@x; torch.cuda.synchronize(); p.__exit__(None,None,None); print(p.key_averages().table(sort_by='self_cuda_time_total',row_limit=10))"
```

公司端只打印本地表，不导出 trace。

### 预期观测（估算/示例，不是实测）

train/valid NLL 曲线、LR warmup/cosine、loss scale、skip、step p50/p95、tokens/s、peak allocated/reserved；generation 仅作错误分析。正常不等于达到特定 PPL。

### 验收条件

- eval 是固定 valid bin 的 token-weighted NLL/PPL；
- 训练/评测 tokenizer 和 hash 相同；
- 生成至少用固定 10 prompts（可重复调用），按重复、语法、角色漂移、提前 EOS、训练式复述分桶；
- 任何缺失正式 run/公开 valid 都标 `INCONCLUSIVE`；
- 公司数据与产物没有离开公司边界。

### 若失败，按什么顺序查

正式与 smoke config diff → data/tokenizer hash → first failing step → FP32 fixed batch → OOM/非有限分流 → 从最后一个完整 `last.pt` 恢复。损坏 `.tmp` 不当 checkpoint；原始数据 hash 改变则停止 resume。

### 当日证据清单

```text
runs/formal_fp16/
  run_manifest.json     config/code/data/tokenizer/env/seed/precision/world_size
  metrics.jsonl         原始逐 step 记录
  last.pt               等价 resume 全状态
  eval.json             token-weighted valid NLL/PPL
  generation.txt        固定 prompt 输出
  decision.md           PASS / FAIL-SYSTEM / FAIL-MODEL / INCONCLUSIVE
```

## 6. 命令参数速查

| 参数 | 含义 | 改动风险 |
|---|---|---|
| `--max-bytes` | tokenizer 只看 train 前缀字节数 | 改动即改变 tokenizer 身份，必须重做 bins |
| `--vocab-size` | 不含 EOT 的 base vocab 上限 | 模型 vocab 应等于实际 base+1 |
| `context_length` | 每个监督窗口 token 数 | attention 计算/激活近二次增长 |
| `micro_batch` | 单次 forward 的 batch | 首要 activation/OOM 旋钮 |
| `grad_acc_steps` | 每次更新累计次数 | 改变每步有效 tokens 和 LR 语义 |
| `--schedule-steps` | LR 曲线的冻结总步数 | 分段训练时不能随当前停止点改变 |
| `--overfit-windows` | 固定有限起点集合 | 只用于 correctness gate，不代表泛化 |
| `--resume` | 加载全状态 checkpoint | config/data/tokenizer 必须一致 |

## 7. 最终发布门

只有以下全满足才写 PASS：来源与许可可追溯；BPE round-trip/save-load/hash 稳定；train-only tokenizer；bins 合同正确；attention oracle `<1e-5`；128-window overfit；FP32→FP16 顺序正确；FP16 200 step finite 且 warmup 后 skip `<1%`；resume 下一 batch 与状态连续；公开 valid 独立 eval；证据全在允许边界内。本文未声称任何一项已经实机完成。

技术链接核验日期：**2026-09-03**：[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)、[CS336 A1](https://github.com/stanford-cs336/assignment1-basics)、[minBPE](https://github.com/karpathy/minbpe)、[PyTorch 2.1 AMP](https://pytorch.org/docs/2.1/notes/amp_examples.html)。
