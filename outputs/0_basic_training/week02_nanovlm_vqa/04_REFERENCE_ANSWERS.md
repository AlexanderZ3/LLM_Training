# Week 02 回答篇：nanoVLM / VQA

> 请先闭卷。每题 4 分：正确结论 1、shape/机制 1、证据 1、边界或追问 1。

## Recall

### W02-R01
答案：`[B,3,H,W]→[B,Nv,Dv]→[B,Nv',Dl]`，文本 `[B,T]→[B,T,Dl]`，拼接 `[B,Nv'+T,Dl]`。评分点：四段齐。误区：把 `Nv` 当 channel。追问：processor 如何改变 Nv？

### W02-R02
答案：attention mask 控制可见/有效 key-token；label mask 控制哪些位置计 CE。问题/视觉可见但不计 loss，padding 两者相应屏蔽。评分点：两语义 2；例子 2。误区：用 `-100` 当 attention mask。追问：causal mask 又是什么？

### W02-R03
答案：freeze 决定参数梯度；no_grad 关闭 graph；eval 改 dropout/BN 行为。评分点：三项 3；组合使用 1。误区：`eval()` 自动禁梯度。追问：冻结 encoder forward 为何可 no_grad？

### W02-R04
答案：推理权重可只含 model/projector 与 config；等价 resume 还需 optimizer groups、scheduler、scaler、RNG、step、best metric、processor/tokenizer/data revisions、冻结清单。评分点：两类边界 2；状态 2。误区：只存 trainable weights 却声称 resume。追问：如何校验 param groups？

## Explain

### W02-E01
答案：decoder 接收最后维 `Dl`，视觉输出是 `Dv`；projector既匹配维度又适配表征尺度/语义，才能与文本 embeddings 同序列计算。评分点：shape 2；尺度/语义 1；梯度桥梁 1。误区：只要 pad 零即可。追问：若 Dv=Dl 仍需什么审计？

### W02-E02
答案：顺序 `[visual,question,answer]` 配合下三角 causal mask；answer 位置可看左侧 prefix，prefix 看不到右侧未来答案。评分点：顺序 1；mask 2；边界 1。误区：视觉 token 需要双向看答案。追问：视觉 token 之间能否互看？

### W02-E03
答案：小集合可能被 projector/LM 记忆，question 本身可能泄露答案；需 validation 上 original 相对 shuffle/blank/question-only 的差异证明对应图像提供信息。评分点：两种伪因 2；反事实 2。误区：train loss 低即 grounding。追问：shuffle 要如何配对？

### W02-E04
答案：prompt 已作为条件给定；对其计 loss 会把优化预算用在复述输入，并按 prompt 长度重加权样本。answer-only 与任务目标一致。评分点：目标 2；权重偏差 1；mask 1。误区：question 不计 loss就不参与 attention。追问：chat template special tokens 怎么处理？

## Apply

### W02-A01
答案：vision `[4,196,768]`，projected `[4,196,576]`，text `[4,64,576]`，combined `[4,260,576]`，labels `[4,260]`；总长度 260。评分点：每错一类扣 1。误区：相加 Nv 与 Dl。追问：logits shape？

### W02-A02
答案：覆盖大小写/空白/标点/冠词、整数小数、EOS/PAD 截断、空输出、超长、非 UTF-8/非法模板；每例有固定 normalized target 与 valid flag。评分点：normalization 2；生成边界 1；确定 oracle 1。误区：看模型错例后改规则。追问：多答案 gold 如何聚合？

### W02-A03
答案：按命名筛 projector、LM 顶层，断言互斥且并集等于 trainable；optimizer 只接这些参数；step 前后 hash/最大差检查 frozen，检查其 grad None。评分点：构造 1；断言 1；权重证据 1；grad 1。误区：LR=0 冒充 freeze。追问：weight decay 会怎样？

### W02-A04
答案：固定 dataset revision/config；以 stable sample ID/hash 排序/白名单，写 explicit ID lists 与 hash；每次先验证源 revision 和内容 hash。评分点：revision 1；ID 1；manifest 1；验证 1。误区：只保存 seed。追问：源数据修订后如何处理？

## Debug

### W02-D01
答案：确认 loss labels 有效；确认 combined embeds 真用 projector 输出；检查意外 detach/no_grad；检查 optimizer group/`requires_grad`；用 hook 看输出/梯度。评分点：四层。误区：直接解冻 LM。追问：输出有 grad 但参数无 grad说明什么？

### W02-D02
答案：视觉路径未连接；问题语言偏置足够；shuffle 配对错误/仍是同图；projector 未训练；evaluator 不敏感；所选样本不需视觉。每个解释配 hook、question-only、需视觉子集或 checksum。评分点：四个带实验的解释。误区：直接归因数据太少。追问：blank 下降而 shuffle 不降如何解释？

### W02-D03
答案：核对 nanoVLM release/commit、222M config、vision/LM revisions、transformers 版本和 state dict key；禁止 `strict=False`/`ignore_mismatched_sizes` 静默吞掉核心 mismatch。评分点：核对 3；禁用 1。误区：current main 与 v0.1 混用。追问：哪些 missing keys可能合理？

### W02-D04
答案：对齐 train/generate prompt；检查 EOS/PAD/BOS IDs、answer start、max_new_tokens；确认 decode 是否切掉 prompt；先 greedy 看逐 token logits，再查 checkpoint/processor。评分点：四层。误区：先升 temperature。追问：为什么 loss 可正常？

## Design

### W02-G01
答案：三模型各测 original/shuffle/blank/question-only；同 split、decode、seed、processor，报 EM/invalid/类型桶；主要看 original 相对反事实差与 A/B 相对 base。评分点：矩阵 2；控制变量 1；解释 1。误区：只报最好格。追问：何时结论 INCONCLUSIVE？

### W02-G02
答案：先各自 processor/tokenize，再 pad；记录 answer span；visual prefix 扩展 attention；labels 初始化 -100，仅复制答案目标；断言有效 label count、长度与 CE oracle。评分点：collate 2；mask 1；断言 1。误区：右 padding 被计 loss。追问：left padding 怎样变？

### W02-G03
答案：本机保存 manifest/config/checkpoint/eval/trace；终端打印不含样本的聚合表、PASS 标签与错误类别计数；求助只给抽象 shape/错误类型/最小公开代码，不外传任何图或数值。评分点：证据链 2；边界 2。误区：截图 dashboard。追问：个人作品集如何做？

### W02-G04
答案：固定模型、文本长度、batch、precision、冻结策略；只改变 resolution或token pooling；warmup 后重复测 step p50/p95、allocated/reserved、Nv、CUDA top ops；达到安全显存余量前停止。评分点：控制 2；观测 1；止损 1。误区：同时改 batch。追问：为何近似二次？

## Trade-off

### W02-T01
答案：patch tokens保留空间细节但序列长、显存/attention贵；pooled便宜却形成信息瓶颈。评分点：表达 2；系统 2。误区：pooling 永远更差。追问：如何用任务类型验证？

### W02-T02
答案：projector-only隔离接口并便宜稳定；轻量解冻允许任务适配但增加显存、遗忘和过拟合风险。先前者通过链路 Gate，再用后者做受控收益测试。评分点：双方 2；顺序 1；证据 1。误区：解冻越多越好。追问：双 LR 如何设起点？

### W02-T03
答案：降 batch影响利用率/统计；降分辨率损失视觉细节并减 Nv；缩文本可能截断问题/答案；少解冻降低适配容量。选择应由 OOM 阶段与任务信息决定。评分点：四项。误区：只用 accumulation 解决所有 OOM。追问：activation OOM 与 optimizer OOM如何辨别？

### W02-T04
答案：V100 有 FP16 Tensor Core但通常无原生 BF16；官方 FA2 CUDA路径要求更新架构。原生 attention + AMP 可在现有 PyTorch 2.1审计和复现，避免 ABI/kernel 风险。评分点：硬件 2；软件约束 1；fallback 1。误区：“V100 不支持 FP16”。追问：如何从探针证实？

## 自评

落盘日期、24 题得分、六级分项、AI 辅助等级、关键错误与 48–72 小时回归题。一次看答案后的高分不算独立证据。
