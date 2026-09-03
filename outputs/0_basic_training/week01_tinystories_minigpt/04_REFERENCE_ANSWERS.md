# Week 01 回答篇：TinyStories MiniGPT

> **先闭卷完成 `03_ORAL_EXAM.md`，再打开本篇。** 每题 4 分；“评分点”是最小充分集，不要求逐字一致。答对结论却不能解释 shape/证据，最多 2 分。

## Recall

### W01-R01
参考答案：`x,y:[B,T]`，`y[:,i]=原序列[:,i+1]`，logits `[B,T,V]`；交叉熵前常展平为 `[BT,V]` 与 `[BT]`。评分点：三组 shape 2；shift 1；边界窗口 1。常见误区：把 labels 与 input 完全相同。追问：EOS 跨文档时如何处理？

### W01-R02
参考答案：model、optimizer、scheduler、scaler、step/tokens、Python/NumPy/torch CPU/CUDA RNG、sampler、config、数据/tokenizer hash。评分点：训练状态 2；随机/数据状态 1；版本契约 1。误区：只存 weights。追问：为何 scheduler 可“重建”仍建议校验 state？

### W01-R03
参考答案：线性得 `[B,T,3D]`，view `[B,T,3,Nh,Dh]`，permute/unbind 得各 `[B,Nh,T,Dh]`，`QKᵀ` 得 `[B,Nh,T,T]`。评分点：每阶段 1。误区：把 head 与 time 轴交换后不 contiguous。追问：输出如何回 `[B,T,D]`？

### W01-R04
参考答案：zero grad；autocast forward/loss；scale(loss).backward；unscale；finite/grad norm；clip；scaler.step；update；成功后 scheduler。评分点：顺序 3；成功 step 语义 1。误区：先 clip scaled grad。追问：accumulation 时何时 unscale？

## Explain

### W01-E01
参考答案：mask 前置把未来 logits 变为 `-∞`，softmax 后概率严格为 0 且剩余位置重归一；后乘 0 会使行和小于 1。评分点：归一化 2；信息泄漏 1；数值实现 1。误区：只说“防作弊”。追问：全 mask 行怎么办？

### W01-E02
参考答案：byte 基础符号覆盖任意字节，所以无 OOV；但语料预处理、pair tie-break、遍历顺序、special token、merge rank 任一变化都会改变编码。评分点：覆盖原理 2；复现因素 2。误区：无 OOV 等于稳定 token IDs。追问：hash 哪些文件？

### W01-E03
参考答案：大词表通常缩短 T，降低 attention 的 `T²` 成本；却增大 `[V,D]` embedding/head 与 softmax `V` 维成本，数据稀疏性也变。评分点：两种相反作用各 2。误区：只谈参数量。追问：weight tying 改变哪部分？

### W01-E04
参考答案：L2 梯度进入 Adam 的动量与按坐标预条件；AdamW 把 `-ηλθ` 从梯度更新中解耦，缩放不同。评分点：更新式 2；预条件差异 1；参数分组 1。误区：在 SGD 直觉上推广。追问：norm/bias 为何常不衰减？

## Apply

### W01-A01
参考答案：`Dh=64`；Q/K/V `[8,6,256,64]`；scores `[8,6,256,256]`；logits `[8,256,8192]`。主要 attention 序列二次张量是 scores/probability。评分点：四项各 1。误区：QKV 写成 `[8,256,6]`。追问：哪项随 V 最大？

### W01-A02
参考答案：每 step 64 序列，共 12,800 序列；最多 `12,800×256=3,276,800` token；scheduler 走 200 optimizer steps。评分点：三数各 1；区分有效 token 1。误区：scheduler 走 800。追问：DDP world size=4 呢？

### W01-A03
参考答案：固定 128 窗口、seed、dropout=0、顺序和模型；FP32 训练至几乎记忆，监控同一集合 NLL/accuracy；失败先查 shift/mask/optimizer/梯度，再谈容量。评分点：冻结 1；指标 1；判据 1；诊断 1。误区：扩大模型掩盖 bug。追问：固定 batch 与固定 dataset 有何不同？

### W01-A04
参考答案：累加所有非 ignore token 的 NLL sum 与 token count，末尾 `sum/count`，PPL 最后一次 exp；不能先对 batch PPL 平均。评分点：sum 1；count 1；短 batch 1；exp 时点 1。误区：batch mean 等权。追问：跨 rank 如何 reduce？

## Debug

### W01-D01
参考答案：先打印一个 x/y 验 shift；检查 causal mask；在 1 batch overfit；确认梯度/optimizer param group/LR；检查 padding；最后才调超参。评分点：顺序与可证伪检查 4。误区：直接加 epochs。追问：为何 `ln(V)` 是线索？

### W01-D02
参考答案：记录未缩放前后的 finite、按模块 grad norm、logit/activation 极值、loss 分项、scale/skip、LR；同 fixed batch 跑 FP32，二分模块或关闭 autocast 局部定位。评分点：日志 2；对照 1；定位策略 1。误区：只把 init scale 调小。追问：overflow 与 underflow 如何区分？

### W01-D03
参考答案：先验证 step21 batch IDs；再比 RNG/sampler；再查 optimizer/scheduler/scaler/global step；确认 dropout、zero_grad 与保存时点；最后核对非确定算子和容差。评分点：五层中四层。误区：权重能 load 就算 resume。追问：怎样做 state diff？

### W01-D04
参考答案：验证 token-weighted NLL、split/hash 与 prompt 拼接；检查训练重复；用 greedy 固定解码隔离 sampling；查看 repetition/EOS 错误桶，并与小样本过拟合输出对照。评分点：评测 1；数据 1；解码 1；模型证据 1。误区：直接调 temperature。追问：PPL 与生成质量为何非一一对应？

## Design

### W01-G01
参考答案：单入口支持 config/max_steps/seed/precision/resume/profile/output；终端 JSON 输出状态、NLL、吞吐、显存、scale/skip；本机留 manifest/checkpoint/eval/decision，外部只保留不含内部事实的学习总结。评分点：入口 1；观测 1；证据链 1；边界 1。误区：依赖云端 dashboard。追问：异常如何脱敏求助？

### W01-G02
参考答案：保存 raw/data preprocessing hash、训练 corpus revision、vocab、ordered merges、special IDs、normalization、实现版本和 tokenizer hash；checkpoint 引用 hash，不同 hash 默认不可直接比 PPL。评分点：四类契约各 1。误区：只记 vocab size。追问：新增 special token 会怎样？

### W01-G03
参考答案：固定公开数据/hash、模型、global batch、steps/tokens、FP16、eval；分别报告硬件/软件与 throughput，不把速度当算法；对质量只比较同训练预算，必要时在同一硬件重跑关键对照。评分点：控制变量 2；分层结论 1；公开边界 1。误区：搬公司日志到个人机。追问：随机种子如何处理？

### W01-G04
参考答案：构造极小 B/T/D；用 Python loop 只遍历允许的 `j≤i` 计算 softmax 加权和；对比向量化输出 `<1e-5`，并断言未来概率 0、行和 1、梯度 finite。评分点：oracle 2；mask 1；梯度/容差 1。误区：用同一实现互相验证。追问：dropout 要怎么处理？

## Trade-off

### W01-T01
参考答案：learned position 简单、适合固定 T 的一周闭环；RoPE 外推与相对位置性质更强但实现/测试成本更高。本周以单变量和 ownership 为先，可把 RoPE 留作隔离实验。评分点：双方特性 2；本周决策 1；验证方式 1。误区：把 RoPE 当无成本升级。追问：超过训练长度会怎样？

### W01-T02
参考答案：共享 `[V,D]` 减参数/可能改善统计效率；要求维度兼容并耦合输入输出表示。检查对象 identity/storage、state_dict 和梯度只更新一次。评分点：收益 1；限制 1；实现 1；测试 1。误区：复制数值冒充 tying。追问：参数量少多少？

### W01-T03
参考答案：大 micro-batch 可能提高 GPU 利用但吃显存；accumulation 保持显存却增加 micro-step/同步开销；更多 optimizer steps 改变优化预算。必须用 global batch、tokens seen、optimizer steps 三轴描述。评分点：三者各 1；统一口径 1。误区：只报 epochs。追问：LR 是否线性缩放？

### W01-T04
参考答案：FP32 是数值/实现 oracle，先隔离数据与模型 bug；FP16 同时引入 autocast、scale/skip 与 kernel 差异。先通过 reference 才能把偏离归因于精度。评分点：隔离变量 2；风险 1；验收 1。误区：FP16 能跑就正确。追问：哪些 reduction 保持 FP32？

## 自评落盘

记录日期、总分、六级分项、AI 辅助等级 A0–A4、证据、错误类型和回归日期。单日高分不构成 mastery；核心 Gate 需另一天 A0/A1 transfer/debug 证据。
