# 第 05 周执行手册：Task-P 数据合同、时间对齐与可复现

> 对应原 12 周主线第 3 周。标准投入 7.5 小时。  
> 主载荷：一个固定公共机器人 Task-P；辅助复用第 01 周文本数据管线。  
> 本周目标：证明模型看到的任务含义正确。数据不可信时，训练 loss 没有解释价值。

每日时间盒：10 分钟写数据假设，50–60 分钟实现/检查，15–20 分钟记录异常和处理；第 2 小时只补一个 validator 或对照。

## 1. 本周先做一个选择

本周到第 11 周只保留一个 Task-P，不随意换 benchmark。

推荐决策：

| 条件 | 选择 |
|---|---|
| 能访问 LeRobot 的 RoboTwin 统一数据且磁盘充足 | 从 lerobot/robotwin_unified 固定一个单任务 |
| RoboTwin 下载/仿真依赖超出一周预算 | 使用 LeRobot 公开 PushT 数据 |
| Hugging Face 数据访问受限 | 使用第 04 周合成 episode，但保留完全相同 schema |

RoboTwin 2.0 官方资源：

- 代码：https://github.com/RoboTwin-Platform/RoboTwin
- 官方文档：https://robotwin-platform.github.io/doc/
- LeRobot benchmark 文档：https://huggingface.co/docs/lerobot/en/robotwin
- LeRobot 统一数据：https://huggingface.co/datasets/lerobot/robotwin_unified

本周默认只使用数据，不安装完整 SAPIEN/RoboTwin 仿真环境。安装仿真器是独立工程，不应挤占数据正确性练习。

## 2. Task Card 必须先于代码

周一开始前创建 task_card.md：

| 字段 | 必须写清 |
|---|---|
| task name/version | 唯一任务与数据 revision |
| observation | 相机、state、语言字段 |
| action | 维度、物理含义、绝对/增量、单位 |
| control rate | Hz 与 control period |
| action horizon | H |
| success | 明确可计算条件 |
| ID split | 训练分布 |
| OOD axis | 只选一个，如物体位置/光照 |
| excluded data | 测试 episode、损坏样本、未经授权数据 |

如果选择 RoboTwin，需要明确它的双臂 14D joint-space action 与你未来 Task-D 的动作空间并不相同；它是公共试验场，不是直接可部署动作 head。

## 3. 目录

    week05-data/
      task_card.md
      data_card.md
      manifests/
        source.json
        train_ids.txt
        dev_ids.txt
        test_ids.txt
      src/
        schema.py
        inspect_data.py
        validate.py
        build_windows.py
        estimate_lag.py
        split.py
        sampler.py
        hash_manifest.py
      tests/
        test_episode_boundaries.py
        test_units.py
        test_lag.py
        test_splits.py
        test_sampler_resume.py
      reports/decision.md

统一 episode schema 建议：

    episode_id: string
    frame_index: int
    timestamp: float seconds
    images: dict camera_name → uint8 H×W×3
    state: float32 Ds
    action: float32 Da
    instruction: string
    success: optional bool
    metadata: task/object/seed/domain fields

不要在 schema 里把所有 rotation 都写成“pose”。明确 quaternion、Euler、axis-angle 或 6D representation，并写顺序和坐标系。

## 4. 每日安排

### 周一：获取、只读检查与数据字典

目标：不训练，先知道每个字段是什么。

网络和磁盘预检：

    df -h
    python -c "import datasets, huggingface_hub; print('imports ok')"
    git ls-remote https://github.com/huggingface/lerobot.git HEAD

若用 Hub：

- 先读取 dataset card、license、config、split 和文件总量；
- 先 streaming/metadata，不直接下载全量；
- 固定 revision；
- cache 路径使用公司批准的本地盘；
- 不使用个人网盘转运。

当天至少人工检查 30 个 episode/window：

- 相机内容是否与 state/action 时间一致；
- camera name/order 是否稳定；
- action 是否有明显限位/异常尖峰；
- episode 开始/结束是否正确；
- success/termination 含义；
- instruction 是否缺失或恒定；
- 帧率是否稳定。

终端统计：

| 类别 | 指标 |
|---|---|
| episode | 数量、长度 p1/p50/p99、空 episode |
| time | dt p1/p50/p99、倒序/重复 timestamp |
| image | resolution、dtype、坏帧、camera 缺失率 |
| state/action | shape、dtype、NaN/Inf、各维 p1/p50/p99 |
| behavior | action near-zero 比例、限位比例、success 比例 |

当天 PASS：

- task/data card 初稿完成；
- 每个 state/action 维度有业务含义；
- 数据 revision 与本地 hash 记录；
- 未把 test 数据用于 normalization。

可选第二小时：对第 01 周 tokenized text 做同样 Data Card，写出文本 token stream 与机器人 episode 的共同不变量和不同点。

### 周二：validator、单位、边界与窗口

目标：把隐含假设变成可失败的程序。

validator 至少检查：

1. schema required keys；
2. dtype/shape；
3. NaN/Inf；
4. timestamp 单调；
5. frame_index 连续；
6. state/action 范围；
7. quaternion norm 或 rotation 合法性；
8. camera 数量/order；
9. episode boundary；
10. success/termination 一致性；
11. control period 异常；
12. duplicate sample id。

window builder：

- observation index k；
- 可选 history K；
- action target k...k+H-1；
- future target k+offset；
- padding 与 mask；
- 绝不跨 episode；
- 明确 action 是在 image k 之前还是之后执行。

必须用人工微型 episode 测试：

    episode actions = [10, 11, 12]
    horizon = 4

逐个 k 写出期望 window、padding 和 mask，再让单测比较。不能只在真实数据上目测。

验收：

- validator 对干净合成集 100% PASS；
- 每种故障注入都能被对应规则抓到；
- 真实数据中的异常有数量和处理策略；
- window 边界单测全过；
- normalization 只基于 train。

可选第二小时：生成含 12 类故障的 corruption suite，作为以后每次数据改动的回归测试。

### 周三：camera/action lag 与动作表示

目标：确定 observation 与 action 真正对应的时刻。

至少实现一种 lag proxy：

- state velocity 与 action 的 cross-correlation；
- end-effector delta 与 command action correlation；
- 视觉运动幅度与 action norm correlation；
- 对合成数据注入已知 lag，先验证 estimator。

扫描：

    lag ∈ {-3, -2, -1, 0, 1, 2, 3} control steps

对每个 lag，保持模型/seed/steps 相同，跑极小 BC probe，比较 validation error。不要用完整 TinyFlow 长训做 lag sweep。

动作表示 A/B 只允许选一个关键对照：

- absolute action；
- delta action；
- 或 joint action 与简化 EEF representation。

记录：

- control period；
- 最优 lag；
- 相邻 lag 差异；
- estimator 置信度；
- 是否需要在 dataset 配置中显式补偿；
- 边界 padding 如何变化。

PASS：

- 合成已知 lag 能正确恢复到误差不超过 1 step；
- 真实数据选出 lag 或明确 INCONCLUSIVE；
- 最终 lag 进入 config，不在 loader 中隐藏；
- 时间对齐误差目标小于一个 control period。

### 周四：split、泄漏、统计量与数据 hash

目标：建立 point-in-time 式的数据纪律。

split 必须以 episode 为基本单位，并尽量以 object instance/scene seed 分组。检查：

- episode_id overlap；
- frame/sample hash overlap；
- object instance overlap；
- simulation seed overlap；
- near-duplicate image/trajectory；
- train-derived normalization 泄漏；
- 手工过滤规则是否看过 test label。

本周固定：

    train/dev/test = explicit ID lists

不要只记录 random seed 后每次重切。manifest 记录：

- upstream repo/dataset revision；
- selected task/config；
- file/episode hashes；
- split ID list hash；
- preprocessing code commit；
- normalization stats hash；
- camera order；
- lag/action representation。

OOD split 只选一个轴：

- object pose range；
- lighting；
- background；
- object instance；
- instruction paraphrase。

不同时混多个轴，否则错误无法归因。

验收：

- episode overlap = 0；
- object/seed overlap 符合预注册规则；
- test 数据不参与 normalization/tokenizer tuning；
- manifest 能唯一重建本周数据版本；
- 发现官方重复时明确记录，不静默删除后改变 benchmark。

可选第二小时：实现 Bloom filter 或 hash index 加速大数据重复检查，但先保证透明正确。

### 周五：sampler/RNG、同 seed 重现与 Data Card

目标：知道“可复现”具体到哪一层。

需要固定：

- Python random；
- NumPy；
- torch CPU；
- torch CUDA all devices；
- DataLoader generator；
- worker_init_fn；
- sampler seed/epoch；
- model initialization；
- augmentation randomness；
- cuDNN deterministic 选择及性能代价。

实验：

1. 同环境同 seed 跑两次 TinyFlow/BC probe，各 50 step；
2. 比较 sampled episode/window IDs；
3. 比较前 20 step loss；
4. 不同 seed 跑一次，确认差异不是被错误固定；
5. step 25 checkpoint，恢复 sampler/RNG 后继续；
6. 比较下一批 sample IDs 与 fixed-batch loss。

容差分层：

| 模式 | 目标 |
|---|---|
| CPU/严格 deterministic 单测 | 尽可能 bitwise |
| 单 GPU deterministic 配置 | 前 20 step 极小误差 |
| 高性能 CUDA 配置 | 预注册数值容差，记录非确定算子 |
| 多 worker 数据增强 | sample IDs 可追踪，统计可重复 |

Data Card 最终包含：

1. 来源/license/revision；
2. schema；
3. task 与 action 含义；
4. 采样/窗口；
5. lag；
6. split/OOD；
7. normalization；
8. validator 结果；
9. 已知偏差/缺失；
10. 数据不能用于什么结论。

## 5. 本周硬验收

| 项目 | PASS |
|---|---|
| schema | 每个 tensor 的单位、坐标系、shape、时刻明确 |
| validator | clean 100%；corruption suite 能被正确抓到 |
| windows | 不跨 episode；padding/mask 单测全过 |
| lag | 合成恢复误差 ≤1 step；真实值已配置或明确不确定 |
| split | episode overlap 0；object/seed 规则明确 |
| stats | normalization 只用 train |
| reproducibility | 同 seed sample IDs 一致；前 20 step 在容差内 |
| resume | sampler/RNG 恢复后下一批次一致 |
| report | Task Card + Data Card + manifest 齐全 |

## 6. 常见错误

### 训练 loss 很好，rollout 不动

- action 中静止比例是否过高；
- absolute/delta 含义是否反；
- normalization 是否反归一化；
- observation/action lag；
- gripper/joint 维度顺序；
- padding 是否被当成真实动作。

### 同 seed 仍不同

- worker seed；
- augmentation；
- sampler epoch；
- CUDA 非确定算子；
- 恢复时 RNG 保存点；
- 数据文件遍历顺序。

不要以“GPU 本来就不确定”结束；先定位差异出现在哪一步。

### Hub 数据太大

- 先 dataset card 和 metadata；
- 只选单 task/config；
- 使用 streaming 或官方小样本；
- 估算磁盘再下载；
- 无批准不迁移外部硬盘；
- 保留合成 schema fallback。

## 7. 闭卷口试

1. 为什么按 frame 随机 split 会泄漏？
2. action lag 的正负号怎样定义？
3. absolute 与 delta action 各自需要哪些归一化？
4. loss mask 和 attention mask 有何不同？
5. 为什么 normalization 必须只用 train？
6. 同 seed 不等于 bitwise deterministic 的原因有哪些？
7. DataLoader worker 的 RNG 如何影响 resume？
8. 数据 hash 为什么比只记 dataset 名更可靠？
9. Task-P 与 Task-D 哪些组件能共享、哪些不能？
10. 什么情况下数据问题应标记 FAIL-SYSTEM 而非 FAIL-MODEL？
