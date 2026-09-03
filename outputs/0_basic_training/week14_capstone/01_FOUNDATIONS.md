# Week 14 基础篇：可复现、可扩展、可恢复的 Capstone

> 状态：发布规格，尚未在用户机器执行；所有门槛、性能与时间值均为预注册目标或估算。  
> 核验日期：2026-09-03。公司 8×V100 32GB、拓扑和 PyTorch 2.1 为用户自述待核验。

## 1. 本周不是“再做一个项目”

Capstone 的问题是：你能否在没有 Codex/Claude/联网帮助、没有手改源码的情况下，从干净 output 目录完成：

```text
validate-data → train → checkpoint → resume → eval
              → 1/4/8 卡 correctness/scaling → profile → failure injection → report
```

只选一个已有可信载荷：

- 推荐：Week 11 Mini-WAM `A_action_only vs B_action_world`；
- 备选：Week 12 FinExec `base→SFT vs short-CPT→SFT`。

周一冻结后不换题。若前周 data/evaluator/smoke 未通过，Capstone 直接 `NOT-READY`；不能用本周重新造一个更简单任务掩盖缺口。

最终产物是自包含 `week14-capstone/`、唯一顶层 CLI、两个 frozen config、数据/代码/模型 lineage、完整/坏 checkpoint 处理、独立 eval、1/4/8 卡表、6–10页报告和闭卷答辩证据。

## 2. 环境与资产边界

| 环境 | 职责 | 精度 | 资产规则 |
|---|---|---|---|
| 公司 Linux 8×V100 | 正式 1/4/8、recovery、profile | FP32 tiny reference；FP16 AMP | 代码输入须审批；数据/模型/log/trace/checkpoint/指标/拓扑不得导出 |
| 个人 5070 Ti | 用公开源从头复建缩小版 | 探针后 FP16/FP32 | 必须重新下载/生成并重跑；不能复制公司任何产物或数字 |
| 可选 H100 | 单独硬件 transfer | 可预注册 BF16 | 独立 run_id/config，不冒充 V100 结果 |

V100 支持 FP16，通常无原生 BF16；不使用官方 FA2 CUDA kernel。公司 PyTorch 2.1 是冻结约束。本周不学习/安装新框架，不创建 Conda/venv，不更改 CUDA/torch。

## 3. 输入来源、版本与许可继承

Capstone 不跟随远端 `main`，而是复制前周已冻结并重验 SHA 的对象。

| 载荷 | 数据 | 模型/代码 | 固定 revision/许可 |
|---|---|---|---|
| Mini-WAM | ToyPush-v1；可选 `lerobot/pusht` | Week11 `run.py`；可选 DINOv2-small | PushT `7628202a2180972f291ba1bc6723834921e72c19` MIT；DINO `ed25f3a31f01632728cabb09d1542f84ab7b0056` Apache-2.0；Toy 由代码/data SHA固定 |
| FinExec | FinQA supported subset | Qwen2.5-0.5B + Week12 evaluator/train/eval | FinQA `0f16e2867befa6840783e58be38c9efb9229d742` MIT；Qwen `060db6499f32faf8b98477b0a26969ef7d8b9987` Apache-2.0 |

选择真实公开/获批数据时保存 LICENSE/README/source revision。Toy-only Mini-WAM 可证明系统，却不能获得真实机器人模型结论；FinQA supported subset 若排除率高也必须限制外推。公司离线环境只用获批 snapshot，不教绕过网络。

## 4. 一键入口与命令合同

唯一顶层接口：

```text
python run_experiment.py validate-data --config configs/reference.json --run-id validate --world-size 1
python run_experiment.py train --config configs/reference.json --run-id ref_ws1 --world-size 1
python run_experiment.py resume --config configs/reference.json --run-id ref_resume --world-size 1 --checkpoint runs/ref_ws1/workload/step_000100.pt
python run_experiment.py eval --config configs/reference.json --run-id ref_eval --world-size 1 --checkpoint runs/ref_resume/workload/step_000300.pt
python run_experiment.py profile --config configs/treatment.json --run-id trt_profile --world-size 8
```

入口只负责：加载 frozen config、验证本地路径/SHA、构造参数数组、启动已复制的 workload、写 manifest。它不能通过 shell string 执行，也不能联网、上传、自动修改 source。每个子命令必须有最大步数/时间或有限数据边界。

## 5. Lineage：结果到底来自什么

一个结果节点定义为：

\[
R=f(C_{code},C_{config},D_{data},M_{model},s,precision,world,ranks,checkpoint,eval).
\]

manifest 至少包含下列经过机器校验的字段；hash 字段只能是实际计算出的 64 位小写十六进制值，不能预填示例值：

```json
{
  "required": ["run_id", "config_sha256", "frozen_manifest_sha256", "seed", "precision",
               "world_size", "command", "status", "successful_updates", "successful_samples"],
  "hash_contract": {"algorithm":"sha256", "encoding":"lowercase-hex", "length":64},
  "status_at_launch":"STARTED_NOT_VALIDATED"
}
```

`run_id` 不能代替 hashes；文件名相同也不能证明内容相同。评测 JSON 必须回指唯一 checkpoint SHA、config/data/evaluator SHA。

冻结清单必须是运行时白名单，而不只是“若干代表文件”：顶层 runner、artifact guard、A/B verifier、publication verifier、README/test、pre-registration、两份 capstone config、实际 workload core 与 evidence adapter、两份 workload config、数据生成代码、data 与 meta 都要登记。runner 必须拒绝任何未登记的 `--config` 或 config 内未登记的 code/config/data 路径；publication 还要逐项重算 frozen SHA，不能只相信 run 中抄入的 manifest SHA。冻结 manifest 与最终 `artifacts.sha256` 都不得把自身纳入 hash 集，否则内容变化会形成不可满足的自引用。

## 6. 预注册：先决定什么算赢

运行前冻结：treatment 唯一差异、primary/secondary metric、数值容差、FP16门槛、global batch、scaling定义、resume门槛、最大steps/GPU小时/磁盘和结论标签。

A/B 的“同预算”是可执行合同：同一 workload code/data、相同模型结构与 seed 产生相同初始权重 hash；同 world size/global batch/precision、相同 successful updates、successful samples（文本载荷还须相同 target tokens）、相同 successful-step sample stream；最后用同一 evaluator、split、样本 ID 顺序与 batch 参数。解析 config 后抄入预计算 SHA 不叫观测：训练必须在实际 `indices→batch` 调用点记录每 rank 完整 IDs，eval 必须从实际 batch 返回值生成 ordered-ID SHA/n，再由 publication 与合同及另一 arm 比较。Mini-WAM 没有 token 序列，必须显式记录 `target_tokens_seen=0 (not_applicable)`，并以相等的 successful samples 作为预算单位，不能把 token 字段悄悄省略。AMP attempted update 不属于模型预算；只有所有 rank 均未 skip 的 successful update 才计入。

Mini-WAM primary 可用 dev action MSE/预注册 rollout success，world relative improvement 为机制指标；FinExec primary 可用 dev execution accuracy，valid JSON/evidence F1 为次指标。看 test 后改变 primary metric 等同无效实验。

结论四分：

- `PASS`：系统门与预注册模型门都过；
- `FAIL-MODEL`：系统可信但 treatment 不达门；
- `FAIL-SYSTEM`：数据/数值/并行/checkpoint/eval不可信；
- `INCONCLUSIVE`：样本/seed/资源不足，无法区分。

## 7. 单卡到多卡的数学等价

DDP global batch：

\[
B_g=B_{micro}\times A_{accum}\times N_{world}.
\]

例如固定 `B_g=64`：1卡 `micro=8,accum=8`；4卡 `micro=4,accum=4`；8卡 `micro=2,accum=4`。实际配置可不同，但乘积必须对。

DDP 默认平均各 rank 梯度。若每 rank loss 是等量元素的均值，它对应 global mean；若 FinExec target token 数不同，必须用全局 token `sum/count` 或明确 rank-mean 偏差。不能再除 world size。

correctness 对比必须固定样本 ID/order、初始化、successful optimizer steps，并比较 loss、grad norm、parameter checksum 与 primary eval。FP16/NCCL 非确定性下使用预注册容差，例如前20 step相对偏差<2%，不是要求 bitwise equality。

## 8. Scaling 与 profiler

弱扩展固定每卡 micro batch；强扩展固定 global batch。吞吐 `T_N`、speedup `S_N=T_N/T_1`、效率：

\[
E_N=\frac{S_N}{N}=\frac{T_N}{N T_1}.
\]

只有同一口径的 successful samples/target tokens、同 warmup/measure window 才能比较。checkpoint/eval/profile step 不进入训练计时。

计时分层：DataLoader wait、H2D、forward、backward、optimizer、checkpoint；CUDA 异步计时需 event 或 synchronize。Profiler/Nsight trace 只做短窗口并留公司环境。通信比例不能从 GPU utilization 猜，要看 NCCL kernel/collective时间与 critical path。

已给定课程诊断阈值（300M+计算密集弱扩展才适用）：`E4≥65%`、`E8≥50%`；它们不是硬件保证。小 Toy 模型低效率不能直接判 NCCL 坏。

## 9. checkpoint 与故障恢复

完整训练状态至少：

```text
model, optimizer, scheduler, GradScaler,
Python/NumPy/Torch/CUDA RNG,
successful step, epoch/sampler offset,
EMA（若有）, config/data/code/model hashes,
world size/strategy/wrap policy
```

发布协议：写临时目录 → 各 rank 写完 → 文件非空/SHA manifest → barrier → rank0 做结构检查 → 原子写最后的 `COMPLETE.json` → 更新 last pointer。对 pickle checkpoint，首次登记只能来自已批准 runner：runner 成功结束后先写绑定 artifact SHA、wrapper/workload manifest SHA、frozen manifest SHA 与 runner SHA 的 producer receipt；登记器在反序列化前验证 receipt、批准 runs 路径和 trust code，再建立并复验候选 hash marker，之后才允许 `torch.load` 做 model/optimizer/scaler/双时钟/逐 rank RNG/contract 语义检查，通过后原子发布最终 marker。后续 resume/eval 同样先验 producer receipt 与最终 marker/status/逐文件 hash，之后才允许加载。`.tmp`、候选 marker、缺状态、零字节、hash错一律拒绝。hash 证明完整性而非来源可信，不能登记任意外来 pickle。

resume 等价不是“程序没报错”。连续 run 与中断恢复 run 必须固定总 schedule 与 successful-step sample stream，在同一 fixed next batch 上比较完整 sample IDs、LR、scale、loss、grad norm；终点除参数 state 外还要比较 optimizer、scaler 与逐 rank RNG，确保“再走一步”仍等价。课程门槛可预注册 next-batch loss/grad相对偏差<1%、Mini-WAM 终态 max-abs≤1e-6。恢复实验是独立系统门，不能让 A 走 continuous、B 走 resume 后再把两者当纯模型 A/B。

发布证据也必须通过机器一致性门：报告中的 reference/treatment run IDs 要各自回指并重新验证 frozen manifest；两边 actual successful updates/samples、实际完整 sample-stream SHA 与实际 eval ordered-ID SHA/n 必须相等；eval JSON、训练 manifest、逐 rank metrics/ID audit 都要重算其记录 SHA，eval 的 checkpoint SHA 必须等于 wrapper manifest 声明的 parent checkpoint SHA；所有被引用 checkpoint producer receipt/marker 都要再次验证。缺一项只能 `FAIL-SYSTEM/INCONCLUSIVE`，不得发布模型结论。

FSDP sharded checkpoint 还绑定 world-size/strategy；除非专门验证 optimizer reshard，不承诺跨 world-size恢复。

## 10. 两个载荷的 shape/schema边界

### Mini-WAM worked example

`B=8,H_a=4,A=2,D=64,H=W=32`：image `[8,3,32,32] uint8→float`，action `[8,4,2] fp32`，latent `[8,64]`，valid `[8] bool`。future target stop-grad；B/C只有 action permutation差异。

### FinExec worked example

`B=2,S=512,V=151936`：IDs/labels `[2,512] int64`，logits `[2,512,V] fp16`，prompt/pad labels为 `-100`。生成 schema：

```json
{"evidence":["table_2"],"program":"divide(10,4)","answer":2.5,"scale":"none"}
```

程序必须经安全 evaluator；reported answer 与 executed answer分开。输出明确“研究任务、非投资建议”。

## 11. 最小可运行 lineage 示例

```python
import hashlib,json
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
files={"config":"configs/reference.json","code":"workload/run.py","data":"data/input.bin"}
missing=[p for p in files.values() if not Path(p).is_file()]
if missing: raise SystemExit("missing:"+repr(missing))
manifest={k+"_sha256":sha(v) for k,v in files.items()}
manifest["status"]="STARTED_NOT_VALIDATED"
Path("runs/demo").mkdir(parents=True,exist_ok=True)
Path("runs/demo/manifest.json").write_text(json.dumps(manifest,indent=2))
```

这是可运行机制示例；文件路径由 Lab 在调用前创建。`STARTED_NOT_VALIDATED` 不能提前写 PASS。

## 12. 正确性不变量

1. 周一后 workload/reference/treatment/primary metric不变；
2. 每个本地路径已创建，所有实际使用的 source/data/model/config 均在白名单且 hash重验；未登记 config 被拒；
3. `validate-data` 在 train 前；test 从不调参；
4. FP32 tiny reference、FP16 single smoke先于多卡；
5. fixed global batch correctness先于 scaling；
6. attempted 与 successful step分开；rank间非有限一致处理；A/B successful updates/samples/sample stream相等；
7. eval为独立进程并绑定 checkpoint；A/B 使用同一 frozen eval protocol；
8. incomplete/corrupt checkpoint必须在 `torch.load` 前拒绝；continuous/resume 比较 next loss/state；
9. profile只在固定窗口，预计值/实测值分栏；publication consistency gate 通过；
10. SHA manifest 不自包含，公司资产绝不通过报告、截图、复制、云logger或本仓库外传。

## 13. 失败诊断树

| 症状 | 最小检查 | 根因候选 | 修复 | 回归 |
|---|---|---|---|---|
| clean-room命令找不到模块 | `find`+manifest | 未复制跨周文件、相对路径依赖 | 自包含复制并重写config | 新shell从README执行 |
| 单卡reference不可复现 | sample IDs/RNG | 数据顺序、随机增强、非确定kernel | 固定RNG/数据；记录容差 | 两次20-step |
| 4/8卡loss偏离 | global batch/IDs | sampler、grad normalization、skip不一致 | 修分区/分母/collective | 前20 successful step |
| 吞吐低 | 分段计时/trace | 数据、通信、checkpoint、小模型 | 隔离瓶颈再优化 | 相同窗口复测 |
| resume step对但loss跳 | fixed next batch | optimizer/scaler/RNG/sampler缺失 | 完整状态 | continuous/resume |
| 坏checkpoint被加载 | marker/SHA | loader只看目录名 | completion+hash验证 | 删除一个文件再测 |
| treatment指标好但系统门坏 | gate表 | 假改善 | 先标FAIL-SYSTEM | 修复后全流程重跑 |

## 14. 闭卷自检与 teach-back

1. run_id、config hash与checkpoint hash分别解决什么？
2. 写出一个固定global batch的1/4/8卡组合。
3. 为什么rank平均loss对变长target token可能不等于global token mean？
4. 何时 E8<50% 不说明NCCL故障？
5. `COMPLETE.json` 为什么必须最后写？
6. 什么是“仅权重checkpoint”，为何不能等价resume？
7. 如何区分 FAIL-MODEL 与 FAIL-SYSTEM？
8. 用5分钟从raw sample讲到optimizer update、DDP collective、checkpoint、独立eval和最终结论。

## 15. 官方资料

- [PyTorch DDP 2.1](https://pytorch.org/docs/2.1/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch FSDP 2.1](https://pytorch.org/docs/2.1/fsdp.html)
- [PyTorch AMP 2.1](https://pytorch.org/docs/2.1/notes/amp_examples.html)
- [PyTorch Profiler](https://pytorch.org/docs/2.1/profiler.html)
- [FinQA](https://github.com/czyssrs/FinQA)
- [LeRobot PushT](https://huggingface.co/datasets/lerobot/pusht)

链接于2026-09-03核验。Capstone使用前周固定 revisions，不因远端更新而漂移；任何升级都形成新的 manifest 和实验，不覆盖原证据。
