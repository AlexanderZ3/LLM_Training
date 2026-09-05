# Week M01 附录 — 数据集与奖励模型完整下载指南

> 生成日期：2026-09-05 · 生成方：cc  
> 适用：个人机器（5070 Ti 工作站或任何有网络与磁盘的机器）。**公司机器不适用**，公司环境的下载走公司流程。  
> 全部体积与版本号来自 2026-09-05 的 HuggingFace 官方 API（`已确认`）。

## 0. 一句话

在 conda 环境 `ResearchAgentPy310` 下跑一条 Python 命令，整夜把 MiniMind 的**全部数据集**（11 个 jsonl，23.62 GB）下到 `datasets/`，断线自动重试，中断后重跑同一条命令继续。

默认**不含模型权重**。GRPO 用的 InternLM2 1.8B 奖励模型是可选项，要下得显式加 `--tier all`，理由见第 1 节。

```powershell
cd d:\zz\00_RealProjects\0_LLM_Training\outputs\1_cc_coaching\track_minimind\week01_minimind_5070ti
& "D:\Software\Large\Anconda\envs\ResearchAgentPy310\python.exe" lab\scripts\download_datasets.py --overnight
```

或者直接双击这个周目录下的 **`下载数据集.bat`**，效果一样。

脚本会自己确认解释器：即使你用别的 Python 启动它，它也会自动切换到 `ResearchAgentPy310` 再继续。在 conda terminal 里 `conda activate ResearchAgentPy310` 之后直接 `python lab/scripts/download_datasets.py --overnight` 同样可以。

## 1. 你会下到什么（默认 23.62 GB，11 个文件）

两个来源，各自锁定了版本号。下载器只从这两个固定版本取文件，不会因为上游更新而拿到不同内容。

| 来源 | 仓库 | 锁定版本 | 体积 | 默认下载 | 许可 |
| --- | --- | --- | --- | --- | --- |
| 数据集 | `jingyaogong/minimind_dataset` | `312afb4f76391145c6902f765bb51691c09a12f5` | 23.62 GB（11 个 jsonl） | **是** | `apache-2.0` 与 `cc-by-nc-2.0` |
| 奖励模型权重 | `internlm/internlm2-1_8b-reward` | `25f3593492ab4625ce00fce8c5e67802d6e702ca` | 3.17 GB（含 2 个 safetensors） | 否，需 `--tier all` | 标注为 `other` |

**为什么奖励模型不在默认里**：它是**模型权重，不是数据集**。MiniMind 的 `train_grpo.py` 默认会加载它给回答打分，但 Week M01 Day 5 刻意不走这条路——16 GB 显存同时放策略模型、参考模型和这个 1.8B 模型有溢出风险，所以 Day 5 默认用 `lab/src/mm_probe/rule_reward.py` 的规则奖励。**按现在的课程设计，你不下它也能完整走完 Day 0 到 Day 5。** 它只是后面想换成模型打分时的升级路径。

**许可提醒（必读）**：数据集卡片同时标了 `apache-2.0` 和 `cc-by-nc-2.0`，后者是**非商用**条款。自己学习和本地实验没问题；如果之后要把训练出的模型或衍生数据放进公开作品集、或用于任何商业场景，先自己去读一遍数据集卡片确认边界。奖励模型的许可标注是 `other`，用之前读它的模型卡。cc 不替你做许可判断。

### 数据集文件清单

| 文件 | 字节 | 层级 | 什么时候用 |
| --- | ---: | --- | --- |
| `pretrain_t2t_mini.jsonl` | 1,241,043,656 | mini | Day 2 预训练主数据，字段 `{"text"}` |
| `sft_t2t_mini.jsonl` | 1,739,201,170 | mini | Day 1 逐 token 分析、Day 3 全参微调，字段 `{"conversations"}` |
| `dpo.jsonl` | 53,653,322 | mini | Day 4 偏好对，字段 `{"chosen","rejected"}` |
| `rlaif.jsonl` | 23,754,740 | mini | Day 5 强化学习的提示词来源 |
| `pretrain_t2t.jsonl` | 8,275,074,893 | full | 全量预训练，是 mini 的超集；跑长训练才需要 |
| `sft_t2t.jsonl` | 14,096,018,369 | full | 全量微调，是 mini 的超集 |
| `agent_rl.jsonl` | 82,036,930 | full | 智能体强化学习轨迹，多轮工具调用 |
| `agent_rl_math.jsonl` | 18,372,683 | full | 智能体强化学习的数学子集 |
| `lora_medical.jsonl` | 34,002,385 | full | 低秩微调医疗领域，Day 3 可选扩展 |
| `lora_exam.jsonl` | 24,650,716 | full | 低秩微调考试领域 |
| `lora_identity.jsonl` | 22,789 | full | 低秩微调身份认知；只有 22 KB，验证链路最快 |

### 奖励模型文件清单

权重两片共 3.17 GB，加上分片索引、配置、三个远程代码文件和分词器。远程代码文件必须一起下，加载时要传 `trust_remote_code=True`。

**Day 5 不一定需要它。** 16 GB 显存上同时放策略模型、参考模型和这个 1.8B 奖励模型有溢出风险，所以 Day 5 默认走 `lab/src/mm_probe/rule_reward.py` 的规则奖励。奖励模型是可选升级路径，先下着，用不用后面再定。

## 2. 开跑前确认三件事

**磁盘。** 27 GB 下载 + 解压过程中 HuggingFace 缓存会有临时副本，留 35 GB 余量比较稳。下载器开跑前会自己算一遍，空间不够直接退出而不是下到一半塞满盘。

```powershell
Get-PSDrive D | Select-Object @{n='剩余GB';e={[math]::Round($_.Free/1GB,1)}}
```

**依赖。** 只需要 `huggingface_hub`，本机 conda 环境已经装好。5070 Ti 那台机器上如果没装：

```powershell
& "<你的python路径>" -m pip install huggingface_hub
```

加速包 `hf_xet` 与 `hf_transfer` 已在本机环境装好（`已确认`，2026-09-05）。缺了只是慢一点，`huggingface_hub` 会打一条建议安装的提示——那是提示不是错误，脚本不会因此中断。5070 Ti 那台机器上建议一并装：

```powershell
& "<你的python路径>" -m pip install huggingface_hub hf_xet hf_transfer
```

**网络。** 直连 HuggingFace 在这台机器上实测可达（`已确认`，2026-09-05）。如果你的网络访问不畅，用镜像：

```powershell
& "D:\Software\Large\Anconda\envs\ResearchAgentPy310\python.exe" lab\scripts\download_datasets.py --overnight --endpoint https://hf-mirror.com
```

镜像和官方源的文件字节数一致，下载器照样按字节数校验，走哪个源都不影响结果可信度。

## 3. 整夜下载怎么跑

全部逻辑在一个 Python 脚本里：`lab/scripts/download_datasets.py`。没有 shell 编排，没有外层封装，重试和循环都在 Python 内部完成。

先在周目录下打开 conda terminal 并激活环境，或者直接用绝对路径调解释器。下面的例子都用绝对路径，复制即可用。

### 先看一眼计划，不下载

```powershell
& "D:\Software\Large\Anconda\envs\ResearchAgentPy310\python.exe" lab\scripts\download_datasets.py --dry-run
```

它会列出全部 22 个文件、每个的字节数和用途、当前状态、磁盘够不够、加速包有没有装。确认无误再正式跑。

### 正式跑（这就是整夜那条命令）

```powershell
& "D:\Software\Large\Anconda\envs\ResearchAgentPy310\python.exe" lab\scripts\download_datasets.py --overnight
```

`--overnight` 做三件事：把重扫轮数提到 20，每轮之间等 60 秒，直到全部就绪或轮数用尽。

行为：

- 文件按体积**从小到大**下载。网络不稳时先把便宜的拿到手，最坏情况也只损失大文件的进度。
- 单个文件失败最多重试 6 次，间隔递增到 60 秒。
- 一轮结束后仍有未完成的就等 60 秒再来一轮，最多 20 轮。整夜足够。
- 每个文件下完立刻按**精确字节数**校验，不匹配算失败并重试。
- 全部完成后计算 sha256，写 `datasets/SHA256SUMS.txt` 和 `datasets/DOWNLOAD_MANIFEST.json`。

日志写在 `datasets/download.log`，由 Python 以 UTF-8 直接写入，不经过控制台，所以不会出现代码页导致的乱码。第二天早上看日志末尾就知道结果。

### 想让它在后台跑，关掉终端也不停

```powershell
Start-Process -FilePath "D:\Software\Large\Anconda\envs\ResearchAgentPy310\python.exe" `
  -ArgumentList 'lab\scripts\download_datasets.py','--overnight' `
  -WorkingDirectory (Get-Location) -WindowStyle Hidden
```

Git Bash 或 WSL：

```bash
nohup "D:\Software\Large\Anconda\envs\ResearchAgentPy310\python.exe" lab/scripts/download_datasets.py --overnight > /dev/null 2>&1 &
```

### 只要 Week M01 主线用得到的（3.06 GB，十几分钟）

如果你想今晚就开始 Day 1 而不是等全量：

```powershell
& "D:\Software\Large\Anconda\envs\ResearchAgentPy310\python.exe" lab\scripts\download_datasets.py --tier mini
```

四个层级：`mini` 是 3.06 GB 的四个文件，`full` 是全部 11 个 jsonl（23.62 GB），`reward` 是奖励模型（3.17 GB），`all` 是前两者之和（27.00 GB，默认）。

层级可以叠加着跑：先 `mini` 开工，白天再补 `--overnight`。已经下好的文件不会重下。

### 常用参数

| 参数 | 作用 |
| --- | --- |
| `--overnight` | 整夜模式：20 轮，轮间 60 秒 |
| `--tier mini\|full\|reward\|all` | 选择下载层级，默认 `all` |
| `--dry-run` | 只列清单不下载 |
| `--verify-only` | 只校验已有文件 |
| `--force` | 字节数正确也重下 |
| `--no-hash` | 跳过 sha256（27 GB 哈希在机械盘上要十几分钟） |
| `--endpoint https://hf-mirror.com` | 换镜像源 |
| `--out <目录>` | 换下载目标目录 |
| `--rounds N` / `--sleep S` | 手动指定轮数与轮间等待 |
| `--no-reexec` | 不要自动切换到 conda 解释器 |

## 4. 中断、续传与校验

**Ctrl-C 随时可停。** 已下载的部分保留，重跑同一条命令继续。

**断电或崩溃后**也一样，重跑即可。下载器每次开跑先扫一遍目标目录，字节数正确的直接跳过。

**只校验不下载**：

```powershell
& "D:\Software\Large\Anconda\envs\ResearchAgentPy310\python.exe" lab\scripts\download_datasets.py --verify-only
```

**怀疑某个文件坏了**：删掉它重跑，或者用 `--force` 强制重下全部。字节数不对的文件下载器会自动识别为"部分"并重新获取，这一点已经实测验证过。

**跳过哈希计算**：27 GB 算 sha256 在机械盘上要十几分钟。急着用可以加 `--no-hash`，字节数校验仍然做。

## 5. 下完之后怎么接进训练

数据落在 `week01_minimind_5070ti/datasets/` 下的两个子目录：

```text
datasets/
├── minimind_dataset/          # 11 个 jsonl
│   ├── pretrain_t2t_mini.jsonl
│   ├── sft_t2t_mini.jsonl
│   └── ...
├── internlm2_1_8b_reward/     # 奖励模型
│   ├── model-00001-of-00002.safetensors
│   └── ...
├── DOWNLOAD_MANIFEST.json     # 版本、字节数、哈希、用途
├── SHA256SUMS.txt
└── download.log
```

有两种接法，选一种即可。

**接法一：命令行直接指路径。** MiniMind 的训练脚本和本 lab 的 `bounded_train.py` 都接受数据路径参数，不需要文件在特定位置。

```powershell
$Data = "<周目录绝对路径>\datasets\minimind_dataset"
& $Py lab\scripts\mmp.py inspect_dataset --stage sft --data-path "$Data\sft_t2t_mini.jsonl" --n 1
```

**接法二：把 `datasets/minimind_dataset` 挂到 MiniMind 仓库的 `dataset/` 下。** 这样原仓库脚本的默认路径 `../dataset/xxx.jsonl` 直接可用。Windows 上用目录联接，不需要管理员权限：

```powershell
cmd /c mklink /J "$env:MINIMIND_ROOT\dataset" "<周目录绝对路径>\datasets\minimind_dataset"
```

Linux 或 WSL：

```bash
ln -s "<周目录绝对路径>/datasets/minimind_dataset" "$MINIMIND_ROOT/dataset"
```

如果 `$MINIMIND_ROOT\dataset` 已经存在且非空，先确认里面没有你要的东西再处理，不要直接覆盖。

**验证接通了**，跑一条不需要 GPU 的命令：

```powershell
& $Py lab\scripts\mmp.py inspect_dataset --stage sft --data-path "<...>\sft_t2t_mini.jsonl" --n 1
```

它会打印一条真实样本的逐 token 表，告诉你哪些位置进损失、哪些是 -100。这既是 Day 1 的主线，也是数据接通的证明。

## 6. 出问题按这个顺序查

| 症状 | 先查什么 | 怎么办 |
| --- | --- | --- |
| 开跑就报空间不足并退出 | 日志里的"目标盘可用" | 换 `--out` 到别的盘，或先 `--tier mini` |
| 连接超时、`ConnectionError` | 能不能访问 huggingface.co | 加 `--endpoint https://hf-mirror.com` |
| 某文件反复"字节数不符" | 是不是被代理或防火墙插了错误页 | 删掉该文件重跑；换端点；查代理设置 |
| `ModuleNotFoundError: huggingface_hub` | 用的是哪个 python | 用绝对路径调 `ResearchAgentPy310` 的 python.exe；脚本也会自动切换 |
| 大文件下到一半停住不动 | 日志最后一行的时间戳 | Ctrl-C 后重跑，续传会接上；网络实在差就先 `--tier mini` |
| 下完了但训练脚本说找不到数据 | 传的路径对不对 | 回第 5 节，两种接法选一种 |
| 磁盘满了导致中途失败 | 剩余空间 | 清空间后重跑；只 `mini` 层的话 3.06 GB 就够 |

## 7. 本机验证状态

以下是在写这份文档的机器上**真实执行过**的（Windows，conda 环境 `ResearchAgentPy310`，Python 3.10.20，无 GPU，2026-09-05）：

| 检查 | 结果 |
| --- | --- |
| `download_datasets.py` 语法编译 | 通过 |
| `--help` 与 `--dry-run` | 正常，22 个文件 27.00 GB 清单正确 |
| 真实下载 8 个小文件（最大 22 KB） | 全部成功，字节数校验通过，哈希与清单正常写出 |
| 重跑同一命令 | 全部识别为"已就绪"并跳过，退出码 0 |
| 人为破坏一个文件后重跑 | 识别为"部分"，只重下那一个，恢复正常 |
| 解释器自动切换 | 用 `torchdiff` 环境启动，脚本自行切到 `ResearchAgentPy310` |
| **真实下载奖励模型全层（3.17 GB，含两个 GB 级文件）** | 11/11 成功，字节数全部校验通过，均速约 18 MB/s，全程无中断 |
| 直连 huggingface.co | 可达，API 响应 0.4 秒 |

**未验证**：完整 27 GB 的端到端下载没跑过（那正是今晚要做的事）；`hf-mirror.com` 镜像端点没实测；真实断网重连的续传没构造过，它依赖 `huggingface_hub` 自身的 blob 续传，加上本脚本的重试与重扫外壳。

**一次已修复的真实故障（2026-09-05）**：首版用 PowerShell 包一层做重试循环，里面对原生命令用了 `2>&1`。Windows PowerShell 5.1 会把原生命令的每一行标准错误包装成 `ErrorRecord`，脚本开头的 `$ErrorActionPreference = "Stop"` 于是被 `huggingface_hub` 一条无害的"建议安装 hf_xet"提示触发，整个下载在第 10 个文件处终止。现在重试、循环、日志、编码、磁盘检查全部在 Python 内部完成，不依赖任何 shell 语义；两个 shell 封装已删除。教训写进了 `CLAUDE.md` 第 3.2 节。

所有体积数字来自官方 API，不是估算。

## 8. 来源

- [MiniMind 数据集](https://huggingface.co/datasets/jingyaogong/minimind_dataset)，版本 `312afb4f7639`，核验 2026-09-05
- [InternLM2 1.8B 奖励模型](https://huggingface.co/internlm/internlm2-1_8b-reward)，版本 `25f3593492ab`，核验 2026-09-05
- [MiniMind 仓库](https://github.com/jingyaogong/minimind)，锁定 commit `7a6fddd63a30c06b2fdd5fac4089922b29bc841b`，Apache-2.0
