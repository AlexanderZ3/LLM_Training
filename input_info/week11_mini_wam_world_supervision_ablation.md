# 第 11 周执行手册：Mini-WAM——动作条件未来表征与三组消融

> 对应原 12 周主线第 9 周。标准投入 7.5 小时。  
> 主问题：future latent loss 真能提供与动作相关的世界监督，还是只增加一个容易下降的辅助 loss？  
> 推荐卡数：单卡 correctness；2/4 卡正式短训；8 卡非必要。

每日时间盒：10 分钟写因果假设，50–60 分钟核心实现/运行，15–20 分钟核对对照；第 2 小时只补一个 seed 或 future offset。

## 1. 最小研究设计

冻结或固定视觉 encoder，定义：

    z_t = encoder(image_t)
    z_future = stop_gradient(encoder(image_{t+k}))
    z_pred = world_head(z_t, action_chunk, state, k)

目标：

    L_total = L_action + lambda_world × L_future

其中：

    L_future = normalized masked distance(z_pred, z_future)

三组必须做：

| Run | action loss | world loss | action conditioning |
|---|---|---|---|
| A | yes | no | normal |
| B | yes | yes | correct |
| C | yes | yes | shuffled across batch/episode |

baseline：

    z_pred = z_t

即 last-frame baseline。只有 B 优于 last-frame 且优于 C，才有证据说明 future predictor 利用了动作条件。

## 2. 资源

| 资源 | 用途 | 地址 |
|---|---|---|
| LeRobot VLA-JEPA | 世界表征监督的公开参考 | https://huggingface.co/docs/lerobot/main/en/vla_jepa |
| LeRobot SmolVLA | 可复用视觉/动作模块 | https://huggingface.co/docs/lerobot/main/en/smolvla |
| JEPA 论文/资源 | representation prediction 背景 | https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/ |
| 本计划第 04/09/10 周 | TinyFlow、objective、公平训练基础 | 本地手册 |

本周不照搬大模型 WAM。只做 20M–100M world head，保证每个因果对照能在一周内完成。

## 3. 数据和 split

复用第 05 周 Task-P：

- camera order；
- state/action；
- lag；
- normalization；
- train/dev/test；
- OOD axis；
- episode IDs。

新增：

- future offset k；
- 当前/未来图像必须在同一 episode；
- episode 尾部 valid_future mask；
- action chunk 覆盖 t 到 t+k 的相关动作；
- future frame 不可从 augmentation 泄漏；
- train/eval encoder preprocessing 完全一致。

本周只固定一个 offset，例如 t+8；t+4/t+16 只能作为 stretch。

## 4. 目录

    week11-miniwam/
      configs/action_only.yaml
      configs/action_world.yaml
      configs/action_world_shuffle.yaml
      src/future_dataset.py
      src/latent_cache.py
      src/world_head.py
      src/losses.py
      src/train.py
      src/eval_future.py
      src/eval_action.py
      src/error_analysis.py
      tests/test_future_alignment.py
      tests/test_action_shuffle.py
      tests/test_stop_gradient.py
      reports/pre_registration.md
      reports/decision.md

## 5. 每日安排

### 周一：future target、last-frame baseline 与防泄漏

目标：证明 target 是真正未来，不是数据索引错误。

任务：

1. 画出 t、action chunk、t+k 的时序；
2. 人工 episode 单测 future index；
3. episode 尾部 mask；
4. 固定 encoder.eval；
5. target 在 no_grad/stop_gradient 下生成；
6. 计算 last-frame baseline；
7. 统计 latent 每维 mean/std/norm；
8. 决定 cosine、normalized MSE 或其他单一 metric。

防止伪改善：

- 不让 predictor 直接读 future image；
- 不把相同 frame 的增强 view 当未来；
- 不让 target encoder被 L_future 一起更新并塌缩；
- 不用 train latent 统计处理 test；
- 不在 split 之间共享相邻 frames。

测试：

- 人工 episode 的 t+k 精确；
- future 不跨 episode；
- target tensor requires_grad=false；
- 将 future image 替换后 target 改变；
- last-frame metric 在全量 dev 可重复。

PASS：

- future alignment tests 全过；
- baseline 固定；
- latent 非全常数/非零；
- pre_registration 写出成功/失败判据。

可选第二小时：比较 patch tokens 与 pooled latent，只做 100 样本 memory/metric，不开始网格。

### 周二：action-conditioned world head 与 256 样本过拟合

目标：证明 world head 有能力拟合，且 action path 真的被使用。

建议结构：

- z_t projection；
- action chunk encoder；
- state encoder；
- offset embedding；
- 4–8 层小 Transformer/MLP；
- output shape 与 z_future 对齐。

如果 z 是 patch tokens：

- 明确 token 数；
- 可预测 pooled/selected tokens 降成本；
- mask；
- 不用过大 world head 抢占主 policy。

先固定 256 windows：

1. FP32；
2. world-only overfit；
3. correct action；
4. zero action；
5. shuffled action；
6. 检查 action encoder grad；
7. 与 last-frame 比较。

shuffle 必须：

- 在 batch 内形成无 fixed point 或统计 fixed points；
- 尽量跨 episode；
- 保持 action shape/边缘分布不变；
- 不打乱 z_t/z_future 配对；
- seed 固定。

PASS：

- correct-action world head 可过拟合；
- action encoder 有非零梯度；
- shuffled/zero 不应与 correct 完全相同；
- predictor 输出非塌缩；
- FP16 20-step smoke。

若 correct/shuffle 相同，先检查数据中 action 是否对未来真的有信息，不能直接增加模型。

### 周三：A/B/C 等预算正式训练

目标：完成核心三组消融。

公平约束：

- 同 policy initialization；
- B/C world head initialization 相同；
- 同 train windows/global batch/optimizer steps；
- 同 action optimizer/hyperparameters；
- 同 precision；
- 同 eval selection rule；
- 同两个 seeds，时间不足先所有组三百 step，再扩主 seed；
- 记录 successful steps/samples seen。

lambda_world：

1. 用 100-step calibration 看 L_action 与 lambda×L_future 的梯度量级；
2. 只选择一个 lambda；
3. 在 pre_registration 记录；
4. 不根据 test OOD 反复调。

需要分别记录：

- L_action；
- L_future；
- policy grad norm；
- world-head grad norm；
- shared encoder grad；
- loss scale/skips；
- throughput；
- peak memory。

多卡：

- 单卡 overfit 后；
- 2/4 卡 fixed global batch correctness；
- 正式可用四卡 0–3；
- A/B/C 使用同卡数；
- world head 额外参数可能改变通信，报告成本。

PASS：

- 三组均有 checkpoint；
- samples seen 差异 <1%；
- FP16 稳定；
- B/C 除 shuffle 外一致；
- A 的 policy 容量与 B/C action path一致。

### 周四：future、action、ID/OOD 与失败相关性

目标：不只看 L_future。

指标：

#### 世界预测

- future latent error；
- last-frame relative improvement；
- correct vs shuffled gap；
- 按运动幅度分桶；
- 按 offset/episode phase 分桶；
- prediction variance/collapse indicators。

#### 动作

- validation action error；
- endpoint/jerk；
- action saturation；
- ID rollout success；
- 一个 OOD axis success；
- inference latency。

#### 失败分数

研究 world error 是否与失败相关：

- 每 episode 的 early world error；
- success/failure label；
- AUROC 或 rank correlation；
- 不能用训练集同数据拟合后再评；
- 样本少时只作探索。

核心门槛：

- B 相对 last-frame future error 改善目标 ≥10%；
- B 优于 C；
- B 的 ID success 相对 A 不下降超过 3 个百分点；
- 至少一个 OOD proxy/rollout 指标改善，才称初步正结果。

如果只改善 world error、动作完全无改善：

    “world target 可预测，但尚未证明对 policy 有用”

不能写成 WAM 成功。

### 周五：成本-收益、错误桶与结论

目标：决定是否继续 world supervision。

成本：

- trainable params；
- peak memory；
- step time；
- communication ratio；
- training FLOPs proxy；
- inference 是否需要 world head；
- latent cache 磁盘/生成成本。

结果表：

| Run | future err | vs last | action err | ID success | OOD success | step ms | peak GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | N/A | N/A | | | | | |
| B | | | | | | | |
| C | | | | | | | |

错误桶：

- static future；
- large-motion；
- contact transition；
- occlusion；
- wrong action conditioning；
- latent collapse；
- policy no-motion；
- overshoot；
- recovery；
- system。

结论标签：

- PASS-WORLD：B 通过 future/action/OOD 预注册门槛；
- REPRESENTATION-ONLY：future 改善但 policy 无收益；
- FAIL-MODEL：B 不优于 last-frame/C，且系统正确；
- FAIL-SYSTEM：alignment/mask/train 不可信；
- INCONCLUSIVE：seed/rollout 不足。

## 6. 本周硬验收

| 项 | PASS |
|---|---|
| target | t+k 对齐、同 episode、stop-gradient |
| baseline | last-frame 在 fixed dev 上可重复 |
| overfit | 256 windows correct-action world head 可过拟合 |
| causal control | B correct 优于 C shuffled |
| fairness | A/B/C samples/steps/config 受控 |
| stability | FP16 无 NaN；skipped <1% |
| future | 目标相对 last-frame ≥10%，否则记录失败 |
| action | ID 不下降 >3pp；OOD 是否改善明确 |
| cost | 参数/显存/吞吐/通信额外成本齐全 |
| conclusion | 不把 auxiliary loss 下降等同 policy 改善 |

## 7. 闭卷口试

1. 为什么 target encoder 要 stop-gradient？
2. last-frame baseline 能排除什么？
3. shuffled-action 对照为什么比 no-action 更强？
4. future offset 太短/太长各有什么问题？
5. latent MSE 下降为何不保证动作提升？
6. lambda 如何改变 shared policy gradient？
7. 如何检测 latent collapse？
8. 什么是 condition leakage？
9. world error 做失败分数需要什么独立评测？
10. 哪种结果应标记 REPRESENTATION-ONLY？
