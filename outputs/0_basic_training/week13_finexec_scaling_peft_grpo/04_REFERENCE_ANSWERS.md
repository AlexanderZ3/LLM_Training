# Week 13 参考答案：Scaling、FSDP、LoRA 与 GRPO

> 先闭卷完成 03_ORAL_EXAM.md。每题 0–3 分；只会复述而不能回答追问，最多 2 分。

## Recall

### W13-Q01

参考答案：包括低精度参数、梯度、Adam 一阶/二阶矩、可能的 FP32 master 参数，再加 activation、临时 buffer 和 allocator。12–16 B/P 取决于 dtype、optimizer 和实现，只能做排除明显不可行配置的估算。

评分点：至少四项状态；实现依赖；非参数项。  
误区/追问：把 activation 算进固定 B/P。追问：logits 与 vocab 怎样影响峰值？

### W13-Q02

参考答案：W'=W+(α/r)BA；A 为 r×in，B 为 out×r；可训练参数 r(in+out)，不计 bias/额外保存模块。base W 冻结。

评分点：公式；shape；参数量；scale。  
误区/追问：参数量 r×in×out。追问：target 多个 Linear 时怎样求和？

### W13-Q03

参考答案：forward 前按 wrap unit all-gather 参数；backward 需要相应参数 materialize；梯度用 reduce-scatter；optimizer 更新本 rank shard。具体 reshard/prefetch 依配置。

评分点：all-gather；reduce-scatter；shard update。  
误区/追问：说成 DDP all-reduce。追问：wrap 太细的代价？

### W13-Q04

参考答案：A_i=(r_i-组内均值)/(组内标准差+ε)。实际目标还常有概率 ratio、clip 和 KL/reference control；全组 reward 相同则信号接近零。

评分点：公式；相对信号；零方差。  
误区/追问：直接用 reward 当 advantage。追问：group size=2 有何噪声？

## Explain

### W13-Q05

参考答案：DDP 在每个 rank 上放完整模型和通常完整 optimizer state，只同步梯度；增加卡数提高数据并行吞吐但不切 base 参数，所以单卡仍需容纳 4B base。

评分点：复制；同步；单卡约束。  
误区/追问：8 卡显存自动合并。追问：什么策略才分摊参数？

### W13-Q06

参考答案：FSDP 主要 shard 参数、梯度和 optimizer state。activation 由 local micro-batch、sequence、层和 checkpointing 决定；临时 all-gather、logits/KV 也可能主导峰值。

评分点：shard 范围；activation 决定因素；临时峰值。  
误区/追问：Full Shard 把所有显存除 N。追问：activation checkpointing 的代价？

### W13-Q07

参考答案：tokenizer 改变 token 序列长度、词表与预测难度，cross-entropy 的事件空间不同；模型/数据模板也可能不同。应在同一冻结 evaluator 上比较生成任务指标，并报告 target tokens 和成本。

评分点：事件空间；序列；共同 evaluator。  
误区/追问：loss 更低必然更好。追问：perplexity 能跨 tokenizer 比吗？

### W13-Q08

参考答案：总 reward 会掩盖分项退化；模型可通过变长、复制格式、伪造 evidence、利用容差或固定高频答案得分。必须看分项、adversarial cases、KL/entropy/length 和下游 execution。

评分点：具体漏洞；分项；行为指标。  
误区/追问：reward 是确定函数所以不会错。追问：哪种依赖顺序防答案捷径？

## Apply

### W13-Q09

参考答案：rank 8 为 8×(4096+4096)=65,536；rank 16 为 131,072。若有相同 shape 的 L 层与多个 target projection，再乘对应模块数。

评分点：两个数；线性关系；扩展。  
误区/追问：把 α 算进参数量。追问：FP32 Adam 状态约占多少字节？

### W13-Q10

参考答案：固定 train/dev/test IDs、prompt/schema、label mask/packing、evaluator、generation、dev selection、seed、successful updates或target tokens、eval频率。记录 tokenizer 差异、允许 micro-batch/accumulation 调整但保持 global batch。

评分点：至少八项；允许项；token budget。  
误区/追问：只固定 steps。追问：overflow 跳步怎样影响预算？

### W13-Q11

参考答案：环境/模型 audit→纸面预算→单卡或两卡 1-step→50-step finite→trainable/strategy检查→峰值/吞吐→checkpoint save→新进程 load→greedy 10 条 evaluator。全部通过后才正式 200 step。

评分点：前置；数值；恢复；任务 eval。  
误区/追问：跑完 50 step 即 PASS。追问：FSDP save 为何需所有 rank 参与？

### W13-Q12

参考答案：format、evidence、program valid、execution matches gold、reported answer 与 executed 一致、scale。依赖顺序应防止无效 program 仅凭答案获满分；每项保留、归一化范围冻结。

评分点：六项；依赖 gate；分项记录。  
误区/追问：只优化 total。追问：evidence F1 与 execution 哪个先 gate？

## Debug

### W13-Q13

参考答案：列出具体 trainable 名称而非百分比；检查目标模块确实参与 forward；固定 batch 后看每层 grad；检查 optimizer 参数集合；一步更新前后比较 adapter hash；核对 labels/mask；验证 adapter load 和 dtype。最后才调 LR/rank。

评分点：名称→forward→grad→optimizer→update。  
误区/追问：直接升 rank。追问：tied module 名称匹配有何风险？

### W13-Q14

参考答案：可能实际是 NO_SHARD/错误策略；wrap 太粗导致大 all-gather；full state dict 暂存在 GPU；activation/logits 主导；use_orig_params/optimizer 创建错误；每 rank 都保存 full checkpoint；allocator reserved/碎片；测量包含 generation。

评分点：至少五个；配置与时间线证据。  
误区/追问：只降 batch。追问：怎样分离 allocated 与 reserved？

### W13-Q15

参考答案：这是典型目标错配/reward hacking 信号。立即停止长跑，冻结 checkpoint；按分项和 adversarial case 找漏洞，增加 evidence gate/长度约束或修 reward 后从稳定 SFT 重新 smoke；标记 RL-NOT-READY，不把 program 指标单独报喜。

评分点：判定；止损；修 reward；重启基线。  
误区/追问：继续训等 evidence 回升。追问：为何不能只提高 evidence 权重？

### W13-Q16

参考答案：用 ranges 分 data/H2D/forward/backward/optimizer/generation/checkpoint；看 memory timeline 判断 activation；FSDP collective 与 idle gap 判断通信；关闭周期 eval/checkpoint测 steady state；固定 batch/sequence，报告 p50/p95。

评分点：分段；受控开关；时间/显存。  
误区/追问：只看 GPU utilization。追问：为什么 generation 单独测？

## Design

### W13-Q17

参考答案：表中需有参数/梯度/Adam/master、shard degree，也要列 activation、attention、logits、all-gather/prefetch、通信 bucket、KV cache、temporary cast、checkpoint staging、allocator余量；给 expected 与 measured 两列。

评分点：状态；非线性项；峰值；实测回填。  
误区/追问：只用参数量乘 16。追问：sequence 翻倍哪些项接近二次增长？

### W13-Q18

参考答案：记录 adapter method/config/rank/alpha/targets、adapter weights hash、base ID/revision/hash、tokenizer ID/revision/special tokens、PEFT/Transformers版本、dtype、modules_to_save、训练config/data hash。load 时逐项校验并拒绝意外 missing/unexpected keys。

评分点：adapter；base；tokenizer；版本；严格加载。  
误区/追问：只保存 adapter.bin。追问：lm_head/embedding tying 怎么处理？

### W13-Q19

参考答案：空 JSON、复制 prompt、伪造 evidence、答案对/program无效、program对/reported错、除零/NaN、超长响应、固定高频答案、容差边界、重复 evidence、深嵌套。每例断言各 reward 分量，不只 total。

评分点：至少八种；预期分项；资源攻击。  
误区/追问：只测正常 gold。追问：极长但正确回答应怎样计分？

### W13-Q20

参考答案：用完全公开的数据/代码独立部署；先在各机探针；主算法对照固定模型、precision、global batch、tokens、seed、版本可比项；若 H100 用 BF16或软件不同，另列硬件实验，不与 FP16 结果合并。公司结果不导出，跨机只能重跑同公开 recipe。

评分点：公开重跑；变量控制；精度分支；安全边界。  
误区/追问：复制公司 checkpoint 到 H100。追问：如何比较 throughput 而非算法？

## Trade-off

### W13-Q21

参考答案：若目标是学习 sharding、全参更新且 1.7B 在现有环境兼容，选 FSDP；若想研究更大 base 且 4B 可驻留、PEFT 稳定，选 LoRA。比较学习目标、单卡峰值、通信、checkpoint复杂度和预期任务收益，只选一个核心路径。

评分点：目标；内存；复杂度；单选。  
误区/追问：更大参数必选。追问：什么时候 LoRA+FSDP 值得增加复杂度？

### W13-Q22

参考答案：rank 16 比 8 的 adapter 参数、梯度、optimizer和部分通信约翻倍，表达能力可能更强，也更易过拟合且吞吐略降。先以 rank 8 建 reference，只有错误显示容量不足且成本可接受时预注册 rank 16 对照。

评分点：线性成本；容量；证据驱动。  
误区/追问：rank 越高必越好。追问：target modules 与 rank 哪个先调？

### W13-Q23

参考答案：只有量化库已批准、版本锁定、SM70 kernel smoke/数值/checkpoint 全过且确实受 base 显存限制时做 QLoRA。否则编译/升级会污染生产栈并吞掉学习时间；回退 FP16 LoRA 或较小全参 FSDP仍能回答核心训练问题。

评分点：准入；V100兼容；fallback。  
误区/追问：为了省显存必须 QLoRA。追问：量化 compute dtype 为何显式 FP16？

### W13-Q24

参考答案：若 0.6pp 未超过预注册方差/置信区间，结论是 STOP-SCALE 或 INCONCLUSIVE；即使显著，也需判断2.3×显存和65%吞吐损失是否符合目标。下一步优先分析 evidence/data 错误、改 evaluator/SFT，而不是继续加模型或 RL。

评分点：方差；成本收益；标签；下一步。  
误区/追问：指标正增长就 CONTINUE-SCALE。追问：什么证据支持 CONTINUE-DATA？
