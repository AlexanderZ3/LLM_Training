# Week M02 · Day 0 — 把公司 8×V100 的真实能力和 MiniMind 的 torch 2.1 兼容性同时钉死

| 字段 | 值 |
| --- | --- |
| 主要产物 | `<公司内路径>/day0/compat_audit.md`（同一天产出的 `<公司内路径>/day0/probe.json` 是它的环境前提） |
| 估时 | `90 分钟`（步骤估时之和） |
| 环境 | `公司 V100` |
| AI 辅助等级要求 | `A2`（命令照抄，但四个计数和探针的每一条 `[FAIL]` 必须由你自己解释） |
| 前置 | 无（本周第一张卡）；机器上已有 Conda + PyTorch 2.1.0，且你有权限在公司内部路径写文件 |
| 本卡对应门 | `静态检查` |
| 可裁剪项 | 步骤 5（本机 CPU 单测可挪到 Day 1 前做）；步骤 1–4、6 不可裁 |

## 为什么做（三句以内）

本周后面五天的每一条命令都建立在两个还没被验证的假设上：这台机器真的是 8 张 sm70、`is_bf16_supported()` 真的是 False，以及 MiniMind 那份代码在 torch 2.1 上没有调用不存在的 API。这两件事不确认，Day 1 的 fp16 等价实验一旦失败，你分不清是数值问题还是环境问题。今天把它们变成 13 个可提交的抽象字段。

## 步骤

### 1. 锁定 MiniMind 代码基线（10 分钟）· `模板`

```bash
git clone https://github.com/jingyaogong/minimind.git
cd minimind && git checkout 7a6fddd63a30c06b2fdd5fac4089922b29bc841b && git rev-parse HEAD && cd ..
export MINIMIND_ROOT=<MiniMind 仓库绝对路径>
export PYTHONPATH=$PWD/lab/src:$PYTHONPATH
```

- 预期：`git rev-parse HEAD` 打印 `7a6fddd63a30c06b2fdd5fac4089922b29bc841b`；`echo $MINIMIND_ROOT` 指向刚克隆的目录。
- 不对时先查：公司机器无法直连 GitHub 时，用公司内部已有的镜像或已下载副本，仍然 `git checkout` 到这个 commit；commit 不一致会让步骤 4 的四个计数全部对不上。

### 2. 记录"没有改动公司环境"的基线（10 分钟）· `可直接执行`

```bash
mkdir -p <公司内路径>/day0
conda env list
python -c "import torch, sys; print(sys.prefix, torch.__version__)"
pip freeze > <公司内路径>/day0/pipfreeze_before.txt
```

- 预期：`torch.__version__` 打印 `2.1.0`；`conda env list` 的行数记下来，本周结束时应当一模一样。
- 不对时先查：`torch.__version__` 不是 `2.1.0` → 先看 `sys.prefix` 是不是激活错了环境；本周不升级、不新建环境，版本不符就在证据里记 `torch_matches_target=false`，不要就地改环境。

### 3. 跑环境探针（15 分钟）· `模板`

```bash
python lab/scripts/probe_company.py --out <公司内路径>/day0/probe.json
echo "probe exit code: $?"
```

- 预期：`probe.json` 出现；`devices.device_count` == 8、每卡 `capability` == `"7.0"`、`devices.homogeneous` == true、`devices.bf16_supported` == **false**、`arch.has_sm_70` == true、`minimind.commit_matches` == true；`sdpa.functional` 里 `math` 与 `mem_efficient` 为 true、`flash` 大概率为 false；退出码 0 表示六项检查全过，退出码 1 表示至少一项 `[FAIL]`。
- 不对时先查：`arch.has_sm_70` 为 false → 这份 wheel 根本不支持 V100，立刻停下并把它记成本周的阻塞项，不要继续往 Day 1 走；其余 `[FAIL]` 逐条抄进证据字段的 `notes`，不要就地装包或改环境。

### 4. 跑兼容审计并核对四个计数（20 分钟）· `模板`

```bash
python lab/scripts/audit_minimind_compat.py --out <公司内路径>/day0/compat_audit.md
```

- 预期：报告有 6 节。第 1 节列出 **9 个默认 `--dtype bfloat16` 的 trainer 脚本**（`train_agent / train_distillation / train_dpo / train_full_sft / train_grpo / train_grpo_rule / train_lora / train_ppo / train_pretrain`）；第 2 节结论行是 **4 个用了 autocast 但没有 GradScaler 的脚本**（`train_agent / train_grpo / train_grpo_rule / train_ppo`）；第 3 节列出 **8 处 `torch.compile` 调用点**；第 4 节是 **0 处 torch 2.1 不存在的 API**。这四个数是写课程的机器上对同一 commit 静态扫描出来的，公司机器上应当一字不差。
- 不对时先查：任一计数不等于 9 / 4 / 8 / 0 → 先 `git rev-parse HEAD` 核对 commit（步骤 1），静态扫描的输入只有代码，commit 一致就不该有差异；确认 commit 一致后把差异记进证据 `notes`。

### 5. 过一遍 CPU 单测，确认代码包本身是好的（15 分钟）· `可直接执行`

```bash
python -m pytest lab/tests -q
```

- 预期：`73 passed, 1 skipped`，约 90 秒。唯一的 skip 是 `test_full_fsdp_reshard_requires_gpu`（FSDP 需要 CUDA 设备）。
- 不对时先查：全部用例被 skip 且提示找不到 MiniMind → `MINIMIND_ROOT` 没生效（步骤 1）；报 `PermissionError` 指向临时目录 → 加 `--basetemp <一个可写目录>`。

### 6. 第一次跑 `run_*` 脚本：先用 `MAX_STEPS=2` 试线路（15 分钟）· `模板`

`lab/scripts/` 里的 6 组 `run_*.ps1/.sh` 加 `fault_kill_rank.ps1/.sh` 中的 `torchrun` 命令，在写课程的那台机器上**没有端到端跑过**（那台机器无 GPU，且它的 torch wheel 缺 libuv，`torchrun` 起不来）。所以每个 `run_*` 脚本在公司机器上的第一次运行都先压到 2 步，只验证参数拼装和三次 `torchrun` 能不能起来：

```bash
OUT_DIR=<公司内路径>/day0/wiring MAX_STEPS=2 \
  bash lab/scripts/run_ddp_equiv.sh
```

- 预期：三次 `torchrun`（world_size = 1 / 2 / 8）都进入训练循环并各写 2 行；`<公司内路径>/day0/wiring/` 下出现 `equiv_ws1_rank0.jsonl`、`equiv_ws2_rank0.jsonl`、`equiv_ws8_rank0.jsonl`；结尾两条 `--compare-json` 各打印一行判定。tiny 配置只是拿来验证线路，它的 loss 数值不作为任何结论。
- 不对时先查：`torchrun` 起不来 → 先看 rendezvous 报错里有没有 `use_libuv`（那是写课程那台 Windows 机器的问题，Linux 上不该出现）；卡住不动 → 把 `NCCL_TIMEOUT_S` 压到 60 再跑一次，让它快点失败。

### 7. 把 13 个抽象字段填进证据块（5 分钟）· `可直接执行`

对着 `02_LAB_GUIDE.md` 第 6.1 节逐项抄：`torch_matches_target`、`device_count`、`capability_uniform`、`bf16_supported`、`has_sm_70`、`sdpa_functional`（三个 bool）、`topology_link_types`、`topology_fully_symmetric`、`minimind_commit_matches`、`bf16_default_scripts`、`autocast_without_scaler`、`torch_compile_sites`、`newer_api_hits`。

- 预期：13 个字段都有值，全部是 bool、int 或短字符串列表，没有一个是绝对性能数字或路径。
- 不对时先查：`topology_link_types` 写成了矩阵形状 → 探针的 `--redact` 默认开启，它只写连接类型的计数直方图与"是否全对称"，矩阵在任何情况下都不进 `probe.json`；你要填的只有类型列表和那个布尔值。

## 公司路径差异

- 本卡全部在公司 8×V100 上执行，`--out-dir` / `--out` 一律指向公司内部路径。
- **fp16 不是 bf16**：MiniMind 的 9 个 trainer 脚本默认 `--dtype bfloat16`，在 V100 上会在进入 autocast 时抛 `RuntimeError`。从 Day 1 起本周所有训练命令一律显式写 `--dtype float16`；步骤 6 的 `--equiv-check` 会强制 fp32，那是等价实验的要求，不是这条规则的例外。
- 日志、`probe.json` 原文、`compat_audit.md` 原文、`nvidia-smi topo -m` 矩阵、任何绝对性能数字全部留在公司内；离开公司机器的只有下面证据字段里的抽象值（比值、布尔、计数）。

## 今日自测（闭卷，2 题，只记录能/不能）

1. `M02-R-01`（bf16 在 V100 报错的位置与异常类型、五个 trainer 的 `--dtype` 默认值、哪一个没有 GradScaler）
2. `M02-R-02`（torch 2.1 flash 后端的 compute capability 区间、V100 会落到哪些 SDPA 后端）

## 证据字段（填完即可归档到 memory/cc/evidence/）

```json
{
  "date": "YYYY-MM-DD",
  "week": 2,
  "day": 0,
  "card": "day0",
  "skill": "env_probe_compat_audit",
  "env": "v100",
  "ai_level": "A2",
  "config_id": "minimind@7a6fddd + torch2.1.0",
  "primary_artifact": "<公司内路径>/day0/compat_audit.md",
  "observations": {
    "torch_matches_target": "true | false",
    "device_count": 8,
    "capability_uniform": "true | false",
    "bf16_supported": "false（预期）",
    "has_sm_70": "true | false",
    "sdpa_functional": "flash=? mem_efficient=? math=?",
    "topology_link_types_count": "<int>（互联链路类型的种类数；类型名本身属于拓扑细节，不出公司）",
    "topology_fully_symmetric": "true | false",
    "minimind_commit_matches": "true | false",
    "bf16_default_scripts": 9,
    "autocast_without_scaler": 4,
    "torch_compile_sites": 8,
    "newer_api_hits": 0,
    "wiring_smoke_three_runs_started": "true | false",
    "pytest_passed": 73,
    "pytest_skipped": 1
  },
  "status": "PASS | FAIL-SYSTEM | INCONCLUSIVE | BLOCKED",
  "failure_class": "none | env | measurement | system | insufficient",
  "self_check": {"M02-R-01": "能 | 不能", "M02-R-02": "能 | 不能"},
  "time_spent_min": 0,
  "notes": "<一句话：探针的每条 [FAIL] 与四个计数的偏差，不含主机名与路径>"
}
```
