# Week 05 基础篇：PushT 数据合同、时间对齐、泄漏与可复现

> 本篇是待执行的数据工程手册，不是实测报告。公司机器、V100数量/拓扑和软件均为用户自述待核验；公司PyTorch2.1不升级。公司数据副本、视频、代码、日志、hash清单、路径、统计、图、checkpoint都不得导出。个人5070 Ti只能从公开仓库独立下载/重建。

## 1. 本周问题与最终产物

训练loss只有在“每个tensor代表什么、来自何时、如何切分”都可信时才有意义。本周固定唯一Task-P：公开 `lerobot/pusht`，不在周中换benchmark。

最终产物：

```text
固定 revision 的 LeRobot PushT v3.0 metadata/parquet/（可选视频）
→ source/task/data card
→ schema validator + corruption suite
→ episode安全的window/padding/mask
→ observation/action lag proxy与已知lag oracle
→ episode级train/dev/OOD显式ID
→ train-only normalization
→ 文件/episode/split/preprocess manifest
→ sampler/RNG/resume可复现实验
```

Week04的TinyFlow/BC可消费本周windows；Week06以后所有profiling必须绑定本周manifest，否则“同一数据”不可证。Task-P是公开代理任务，不等于用户真实Task-D；二者只能共享schema/工具，不能声称动作空间或结果可部署。

## 2. 固定公开对象、版本与许可

技术链接核验日期：**2026-09-03**。

| 对象 | 固定标识 | 用途 | 许可/边界 |
|---|---|---|---|
| LeRobot PushT dataset | `lerobot/pusht@7628202a2180972f291ba1bc6723834921e72c19` | 本周唯一Task-P | dataset card标MIT；公司审批并保存card |
| dataset format | `meta/info.json`声明`codebase_version=v3.0` | parquet/video schema | 以固定snapshot内metadata为真，不按current docs猜 |
| LeRobot source | `fbb811fca92504439792b97d216f0d00c2268382` | current format/action文档只读参考 | Apache-2.0；当前包要求新torch/Python，不装进PyTorch2.1环境 |
| Diffusion Policy | `5ba07ac6661db573af695b419a7947ecb704690f` | PushT收集顺序与action chunk背景 | MIT；只读参考 |
| RoboTwin unified | `lerobot/robotwin_unified@1287871839fae2296bc27b88a5457c3e1eba8e1f` | 后续可选，不在本周混入 | 独立审批；不能与PushT拼接后仍称单任务 |

固定card在源端声明：206 episodes、25,650 frames、10 FPS、只有train split、图像96×96×3、state/action各2维。它们是“上游metadata声明”，仍需本地validator复核；不能写成用户已下载/已测事实。

主教程只下载metadata与parquet即可完成state/action合同。图像-动作人工对齐需要获批下载视频，或结论明确为 `IMAGE_ALIGNMENT_UNVERIFIED`。公司网络不可达时，只能用获批镜像/预置snapshot；Week04 `point-mass-v1` 可跑validator oracle，但不能替代PushT数据结论。

## 3. Task Card：先分已知、待核验与禁止推断

| 字段 | 固定/待核验内容 |
|---|---|
| task | Push-T：圆形agent推动T形block到目标区；以固定dataset card/上游环境为准 |
| observation image | `observation.image`, uint8 HWC 96×96×3，视频10FPS（metadata声明） |
| observation state | `[2] float32`；LeRobot环境映射名为`agent_pos`，坐标单位/原点/范围需由上游源码+数据统计核验 |
| action | `[2] float32`；LeRobot默认语义为absolute target position，但本snapshot必须用源码/轨迹验证，不能称力/速度 |
| timing | row k先记录observation/state与对应action，action随后作用于环境；用上游收集代码和lag proxy核验 |
| control period | 0.1s（10FPS metadata声明）；timestamp实测容差另定 |
| action horizon | 教程固定H=16，不改变原数据 |
| success | `next.success [1] bool`；需验证与episode结束关系，不能凭最后frame推断 |
| ID/OOD | episode级；OOD只选初始agent x位于最高20%的单轴 covariate shift |
| excluded | 损坏episode、schema不符、未经授权数据；所有排除规则在看test outcome前冻结 |

若action绝对/相对语义不能从固定源码和数值验证，本周状态是 `CONTRACT_UNRESOLVED`，禁止训练模型掩盖。

## 4. Canonical schema与具体shape

源row至少：

```text
episode_index: int64 scalar
frame_index: int64 scalar
timestamp: float32 seconds scalar
observation.state: float32 [2]
action: float32 [2]
next.done: bool scalar
next.success: bool scalar
index: int64 global scalar
task_index: int64 scalar
observation.image: video frame 96×96×3（可选解码）
```

Canonical batch/window：`B=4,H=16,Ds=2,Da=2`：

- observation at k：`state [4,2] float32`；
- action target k..k+15：`actions [4,16,2] float32`；
- `action_mask [4,16] bool`；
- `episode_id/frame_index [4] int64`；
- 可选image `uint8 [4,96,96,3]`，processor后float `[4,3,96,96]`。

source dtype、模型dtype与物理语义是三层。存储float32不等于训练必须FP32；FP16 AMP也不允许改变单位/坐标系。

## 5. Episode、边界与window不变量

同一episode内：frame index从0连续；timestamp严格递增；相邻dt接近0.1s；global index唯一；尾frame的done/success关系可解释。不同episode绝不能因为parquet文件排序而拼接。

长度3、H=4的动作 `[10,11,12]`：

```text
k0 [10,11,12,PAD], mask [1,1,1,0]
k1 [11,12,PAD,PAD], mask [1,1,0,0]
k2 [12,PAD,PAD,PAD], mask [1,0,0,0]
```

padding值可任意替换而不改变masked loss。processed window必须记录`episode_index,k,source global index`，从模型错误能追回原row；processed manifest还要直接绑定source/raw manifest，不能只靠目录邻接猜lineage。

## 6. 时间对齐与lag符号

本周定义正lag `ℓ>0`：命令 `action[k]` 与未来 `state[k+ℓ]` 对齐。若action是absolute target position，可比较：

$$E(\ell)=\operatorname{mean}_{e,k}\|a_{e,k}-s_{e,k+\ell}\|_2^2,$$

仅在state/action共享坐标/单位已确认时使用。若动作不是绝对position，这个proxy无效，应改用`state velocity/delta`与action的cross-correlation。

扫描`ℓ∈[-3,3]`必须保证索引仍在同一episode。先对合成序列注入已知lag，恢复误差≤1 step；真实PushT如果曲线平坦或邻近lag差异小，结论`INCONCLUSIVE`，不强选0。

lag proxy只是统计证据；还要人工检查30段视频/轨迹、上游收集顺序和小BC probe。最终lag显式写config，不能藏在loader里。

## 7. Split、泄漏与 OOD

源dataset只有train split。本周建立“内部开发split”，不能称官方benchmark test。

1. 以episode为不可分组；
2. 按每个episode初始`state[0]`的x排序，最高20%作为单轴OOD；
3. 剩余episode按`sha256('week05:'+episode_id)`稳定分到train/dev；
4. 显式写`train_ids.txt/dev_ids.txt/test_ids.txt`；
5. normalization只用train rows；
6. 不因看见test success而调整规则。

检查：episode ID、global index、exact row hash不跨split；近重复trajectory另报。图像内容hash需要解码/固定frame bytes；只下载parquet时明确该项未完成。

OOD只改变“initial agent x”这个轴。它不是标准PushT OOD benchmark，也不能保证difficulty单调。

## 8. Normalization与泄漏

对train有效row计算每维`mean/std/min/max/p01/p99`：

$$\hat x=(x-\mu_{train})/(\sigma_{train}+10^{-6}).$$

dev/test只应用train stats。若先用全量估计，模型虽未见label也已经利用holdout分布信息。stats JSON和hash进入data/model checkpoint；动作表示从absolute变relative后必须重算stats和所有windows。

异常值处理规则（clip/drop/keep）须在看test outcome前写入card，并报告影响row/episode数。不得静默删除官方重复后仍声称原benchmark。

## 9. Validator是可失败程序

立即阻断的硬规则：required keys、metadata与Parquet physical dtype/shape、finite、逐interval timestamp、frame连续、episode boundary、done suffix、global index唯一、task index范围、split overlap、train-only stats。state/action范围与camera key稳定性需先从固定snapshot统计/metadata预注册阈值；阈值尚未冻结时只能报告分位数或`IMAGE_ALIGNMENT_UNVERIFIED`，不能假装已通过硬门。

corruption suite每次只注入一种错误：缺文件、零行Parquet、坏metadata version、缺列、错误Arrow dtype、NaN action、错shape、重复timestamp、倒序frame、跨episodewindow、重复global index、bad mask、test row进入stats。validator必须对干净fixture全过，并对每项断言具体错误码，不能用另一条偶然失败冒充覆盖。

## 10. Reproducibility是分层合同

需要固定：Python/NumPy/torch CPU/torch CUDA all devices、model init、DataLoader generator、worker init、sampler permutation/epoch/cursor、augmentation、lag/window config、代码/data/stats hash。

同seed不保证跨GPU/torch版本bitwise。目标分层：

| 模式 | 目标 |
|---|---|
| 纯Python/CPU validator | bitwise稳定manifest/IDs |
| 单GPU deterministic probe | sample IDs完全相同，loss在预注册容差内 |
| 高性能CUDA | 记录非确定算子/容差，不伪称bitwise |
| resume | 下一batch IDs、augmentation/noise RNG、LR/scaler连续 |

checkpoint保存时点要统一：完成成功optimizer update、推进sampler/step之后再保存，恢复后下一batch应与continuous一致。

## 11. 数据与infra联动

- 视频decode可能成为CPU/I/O瓶颈；先用parquet-only区分模型与video成本；
- 多worker会放大文件句柄、随机性与prefetch resume问题；从`num_workers=0`建立oracle；
- episode长度不均使random window过采长episode；报告episode-uniform与window-uniform差异；
- H增大线性增加action target与mask，后续Flow attention/MLP成本也增加；
- hash全量大文件成本高但只在manifest构建执行，不放训练hot path；
- 公司Profiler/数据统计包含内部环境事实，即使数据公开也不能导出。

## 12. 正确性不变量

1. revision/card/license/local hash齐全；
2. metadata声明与本地parquet统计逐项核验；
3. state/action每维语义、单位、坐标系、时刻已解决，否则停训；
4. clean validator 100%，corruption suite全抓；
5. windows不跨episode、mask边界手算通过；
6. synthetic已知lag恢复≤1 step，真实lag配置或INCONCLUSIVE；
7. split ID/global index overlap=0；
8. normalization只用train；
9. manifest能唯一重建source/split/preprocess/stats；
10. 同seed IDs相同，resume下一batch相同。

## 13. 失败树

| 症状 | 第一检查 | 根因 | 修复/回归 |
|---|---|---|---|
| parquet读不到 | LFS指针/size/schema | snapshot不完整、pyarrow版本 | 固定文件清单与hash；管理员重导 |
| frame/dt不连续 | 按episode打印邻差 | 排序错、drop frame、源异常 | 按index稳定排序；异常策略写card |
| loss好但rollout不动 | action/state统计和表示 | absolute/delta反、lag、静止占比 | 语义核验+lag+反归一化测试 |
| lag最优在边界±3 | 扩扫描前先查符号/单位 | estimator错、真实lag更大 | synthetic oracle；预注册后扩区间 |
| train/dev重复 | episode/global hash集合 | frame随机split、重复源 | 重建显式ID，废弃旧结果 |
| test stats泄漏 | stats provenance | 全量fit、缓存复用 | train IDs重算，hash回归 |
| 同seed IDs不同 | sampler/file排序 | glob无排序、worker RNG | stable sort/stateful sampler |
| resume下一批不同 | cursor/prefetch/RNG | 保存时点或worker队列 | num_workers=0 oracle，再扩展 |
| current LeRobot安装冲突 | resolver dry-run | 其要求新torch/Python | 不安装；直接读固定parquet |

## 14. Teach-back（先闭卷）

1. 为什么frame随机split会造成轨迹泄漏？
2. 写出本周正lag定义，并手算`ℓ=1`的有效索引。
3. action是absolute target时与delta时，lag proxy应如何不同？
4. 为什么normalization只能fit train？
5. 手算长度3、H4的三个window/mask。
6. 解释同seed为何不自动等于bitwise。
7. 列出sampler可resume的最小状态。
8. 哪些证据缺失时必须标 `CONTRACT_UNRESOLVED`？

## 15. 官方一手资料

技术链接核验日期：**2026-09-03**。

- [LeRobot PushT dataset card](https://huggingface.co/datasets/lerobot/pusht)
- [LeRobotDataset v3.0（LeRobot v0.4.3）](https://huggingface.co/docs/lerobot/v0.4.3/en/lerobot-dataset-v3)
- [LeRobot action representations](https://huggingface.co/docs/lerobot/main/en/action_representations)
- [LeRobot official repository](https://github.com/huggingface/lerobot)
- [Diffusion Policy official repository](https://github.com/real-stanford/diffusion_policy)
- [RoboTwin official repository](https://github.com/RoboTwin-Platform/RoboTwin)
- [PyTorch 2.1 reproducibility](https://pytorch.org/docs/2.1/notes/randomness.html)
