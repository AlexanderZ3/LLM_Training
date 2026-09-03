# Week 11 参考答案：Mini-WAM

> 先闭卷完成 03_ORAL_EXAM.md，再使用本文自评。参考答案不是唯一措辞；每题按 0–3 分评分。若不能回答追问，不应给满分。

## Recall

### W11-Q01

参考答案：patch 路径中 z_t、z_future、z_pred 通常为 B×V×N×D；pool 后为 B×D；action_chunk 为 B×H_a×A。k 是从 t 到 target frame 的时间偏移，H_a 是输入动作段长度，两者不必数值相等，但必须由控制频率、采样率和动作覆盖区间给出确定映射。

评分点：shape；k/H_a 区分；时间合同。  
误区/追问：把 k 当 token 数。追问：动作降采样两倍后怎样改映射？

### W11-Q02

参考答案：L_total=L_action+λ_world L_future。masked MSE 应对有效元素求和，再除以有效样本/token/维度总数。固定除以 B 会让 episode 尾部 padding 比例改变 loss 尺度和梯度。

评分点：联合损失；mask；有效分母。  
误区/追问：先 mean 再乘 mask。追问：全 batch 无有效 future 时如何处理？

### W11-Q03

参考答案：stop-gradient 切断 L_future 到 target encoder 参数的反向路径；encoder.eval 只切换 dropout、BatchNorm 等训练行为，不会关闭梯度。冻结还需 requires_grad=false 或 no_grad。

评分点：梯度路径；eval 语义；正确冻结。  
误区/追问：认为 eval 等于 no_grad。追问：EMA target 需要保存什么状态？

### W11-Q04

参考答案：A 只有 action loss，是策略基线；B 加 world loss 且用正确动作，是处理组；C 与 B 完全相同但动作错配，用于检验 predictor 是否真正利用动作。B 对 A 测 utility，B 对 C 测 condition causality。

评分点：三组；唯一变量；两类比较。  
误区/追问：C 同时换初始化。追问：A 是否仍应保留相同 action-path 容量？

## Explain

### W11-Q05

参考答案：慢变化视频中 z_t 天然接近 z_t+k；复杂模型即使没学动作也能得到低 error。last-frame 给出“世界保持不变”的强基线，模型必须超过它才说明预测有增量信息。

评分点：静态先验；增量标准；不能只看绝对 loss。  
误区/追问：认为 baseline 越弱越好。追问：按运动幅度分桶会看到什么？

### W11-Q06

参考答案：shuffled-action 保留动作 shape、幅值和边缘分布，只破坏动作与未来的配对；zero/no-action 同时改变分布和网络路径，可能造成 OOD 输入或容量变化。因此 shuffle 更接近受控反事实。

评分点：保持边缘分布；破坏对应；控制变量。  
误区/追问：同时 shuffle future。追问：fixed point 会怎样稀释效应？

### W11-Q07

参考答案：L_future 是表征预测 proxy。它可能由静态背景、易预测 latent 或独立 world head 完成；也可能与 action 梯度冲突。闭环成功还受动作分布、误差累积、控制频率和环境反馈影响。

评分点：proxy/目标区别；共享路径；闭环因素。  
误区/追问：辅助 loss 下降即成功。追问：什么组合才允许 PASS-WORLD？

### W11-Q08

参考答案：共享梯度为 g_action+λg_world。同向时可能形成正迁移，反向时会损害 policy；λ 控制的是梯度贡献而不是仅控制 loss 数值。应测两者 norm 和 cosine，并看 action dev 指标。

评分点：公式；方向；证据。  
误区/追问：让两个 loss 数字相等。追问：无共享参数时 world loss 如何影响 policy？

## Apply

### W11-Q09

参考答案：z 为 8×2×196×768，action 为 8×16×7。若 valid_future 为 8，可扩成 8×1×1×1；若还有 token mask，应组合成 8×2×196×1，并以有效元素数归一化。

评分点：精确 shape；广播；分母。  
误区/追问：mask 只乘 loss 后仍对全部 token mean。追问：相机缺失 mask 如何加入？

### W11-Q10

参考答案：反复采样 randperm 并拒绝任何 perm[i]=i，有限次数后用循环位移兜底。batch=1 无法形成组内反事实，应增大 batch、用跨 micro-batch 缓冲或将该样本排除并标记限制。

评分点：derangement；确定性 seed；batch=1 处理。  
误区/追问：roll 但忘记跨 episode。追问：为何要报告 fixed-point ratio？

### W11-Q11

参考答案：至少包括 alignment/mask tests、target 无梯度、world-only loss 显著下降、action encoder grad 非零、correct/zero/shuffle 输出敏感、prediction 不塌缩、last-frame 被超过、checkpoint 可恢复、20-step FP16 finite。

评分点：模型容量；条件路径；数值/恢复。  
误区/追问：只要求训练 loss 下降。追问：过拟合失败先加模型还是先查数据？

### W11-Q12

参考答案：zero_grad；autocast 中 forward；敏感 loss/reduction 转 FP32；scaler.scale(loss).backward；scaler.unscale_(optimizer)；记录 clip 前 norm；clip；scaler.step；scaler.update；记录是否跳步。clip 必须在 unscale 后。

评分点：完整顺序；unscale→clip；跳步观测。  
误区/追问：先 clip 缩放梯度。追问：为何不能只 model.half？

## Debug

### W11-Q13

参考答案：先确认 action 真的进入 forward 且 action encoder 有梯度；再比较 correct/zero/shuffle 的同 batch 输出；核对 shuffle 无 fixed point 且只打乱 action；核对数据时间对齐；按运动量分桶；最后才考虑动作对未来信息弱或模型容量问题。

评分点：代码路径→对照实现→数据→任务信号。  
误区/追问：立刻加大 λ。追问：什么结果说明数据几乎静止？

### W11-Q14

参考答案：检查 target 和 prediction 各自样本间 std/norm；检查 valid mask 数量与 loss 分母；将 target 替换为不同 future 看 hash/latent 是否变化；按运动幅度看 last-frame error。target 有方差而 prediction 无方差是模型塌缩；有效数为零是 mask bug；二者都有低变化可能是静态数据。

评分点：区分三假设；干预测试；分桶。  
误区/追问：只看一条 loss。追问：归一化 σ 接近零怎么办？

### W11-Q15

参考答案：数据类包括 split preprocessing、camera order、train-only stats、episode 泄漏；模型类包括 train-mode dropout、过拟合、checkpoint 选择；系统类包括错 checkpoint、不同 config/hash、eval dtype 或 rank 聚合。用 manifest、固定样本输出和单进程 eval 逐层排除。

评分点：三类根因；对应证据；先固定 eval。  
误区/追问：直接归因 domain shift。追问：如何验证不是错误 checkpoint？

### W11-Q16

参考答案：在固定 batch 复现；关闭 world loss 验证异常是否消失；保持模型不变切 FP32；对 world head 子模块加 finite hook；检查 latent 极值和归一化；缩小到首个非有限 op；修复敏感 reduction/输入范围，而不是只降 batch。

评分点：固定复现；二分模块；FP32 reference；首异常。  
误区/追问：不断降低 scaler 初值而不定位。追问：attempted 与 successful step 如何记录？

## Design

### W11-Q17

参考答案：运行前冻结假设、主/次指标、A/B/C、一个 λ、一个 offset、seed、split、checkpoint 选择、容差、算力上限、停止条件和结论标签；文件做 hash。只有预先列出的系统错误允许修复并形成新版本。

评分点：指标；变量；资源/停止；版本不可抵赖。  
误区/追问：只写“看效果”。追问：发现 evaluator bug 后如何更新注册？

### W11-Q18

参考答案：cache key 至少含 dataset revision、sample/episode ID、encoder model/revision/weights hash、preprocess/augmentation hash、camera order、resolution、dtype、latent shape、split、代码 commit。任一项改变即失效；test stats 不能进入 train cache 归一化。

评分点：模型、数据、预处理、shape、失效。  
误区/追问：只用文件名区分。追问：随机增强还能否缓存？

### W11-Q19

参考答案：冻结 world head 和阈值；在未用于训练/调参的 episode 上取早期窗口 world error；与独立 success/failure label 对齐；报告 AUROC/AUPRC、置信区间和按任务分桶；与简单基线比较，禁止同一数据拟合又评测。

评分点：独立集；时序无未来泄漏；合适指标；基线。  
误区/追问：用全 episode error 预测早期失败。追问：类别极不平衡为何看 AUPRC？

### W11-Q20

参考答案：先单卡同 batch reference，再固定 global batch 做四卡 correctness；通过 micro-batch×accumulation×world-size 保持总 batch；DistributedSampler 固定 seed/epoch；比较 successful updates 和 samples seen；A/B/C 用相同卡数、rank mapping和 logging/eval 周期。

评分点：global batch 公式；sampler；successful updates；同硬件。  
误区/追问：每卡 batch 不变却称算法公平。追问：弱扩展能回答什么？

## Trade-off

### W11-Q21

参考答案：patch latent 保留空间/局部动态，预测能力强，但 token 多导致 activation、world attention、显存和通信成本高；pooled latent 便宜稳定，适合先证伪，但可能丢失接触点和局部运动。先 pooled 建闭环，再按错误桶决定是否 patch。

评分点：表达能力；系统成本；阶段化选择。  
误区/追问：默认更细一定更好。追问：可否只选部分 token？

### W11-Q22

参考答案：太短时 last-frame 已很强、动作效应小；太长时多模态不确定性、有效窗口减少、action chunk 覆盖不足、误差累积。预注册一个与控制周期有业务含义的 k，其他 offset 只做后续 stretch，不用 test 选 k。

评分点：短/长风险；数据量；预注册。  
误区/追问：一次扫描后挑最好并报主结果。追问：如何按 episode phase 分桶？

### W11-Q23

参考答案：数据少、算力有限、要避免 target collapse 或建立因果基线时冻结；有充足数据、独立 target/EMA 机制且 frozen 表征确实限制任务时才联合训练。进入联合训练前需冻结版通过、collapse 指标完备、学习率分组和新消融预注册。

评分点：冻结理由；联合收益；门槛。  
误区/追问：联合训练总会更好。追问：EMA target 的更新应在 optimizer 前还是后？

### W11-Q24

参考答案：标记 REPRESENTATION-ONLY，而非 PASS-WORLD。说明动作条件表征预测成立，但尚无 policy utility 且系统成本较高。下一步只做一个针对性实验，例如定位梯度是否传到共享策略、降低成本或在预注册更敏感 OOD 上复测；若仍无益则停止该分支。

评分点：正确标签；证据边界；成本纳入；单一下一步。  
误区/追问：因 future 指标好就宣称成功。追问：什么新证据会升级为 PASS-WORLD？
