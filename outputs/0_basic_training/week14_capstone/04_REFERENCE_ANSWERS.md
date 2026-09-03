# Week 14 参考答案：Capstone 答辩

> 先闭卷完成 03_ORAL_EXAM.md。每题 0–3 分；满分需要机制、证据、不变量和取舍，不能只背定义。

## Recall

### W14-Q01

参考答案：repeatability 是同环境重复在容差内；reproducibility 是依靠冻结材料从 clean output 重建；determinism 是给定状态产生逐位或强一致结果。GPU 并行可不逐位，但仍需预注册容差和可追踪随机状态。

评分点：三者区分；GPU现实；容差。  
误区/追问：不 bitwise 就不可复现。追问：第二 seed 属于哪类证据？

### W14-Q02

参考答案：model、optimizer、scheduler、GradScaler、global/attempted/successful step、Python RNG、NumPy RNG、CPU/CUDA RNG、sampler epoch/offset/generator、EMA/target encoder、config/data/source hash、precision、world size/strategy/wrap、evaluator/generation version。

评分点：至少十类；随机/采样；lineage。  
误区/追问：只存模型权重。追问：sampler offset 丢失会怎样？

### W14-Q03

参考答案：B_global=B_micro×accumulation×world_size；E_N=throughput_N/(N×throughput_1)；relative_error=|a-b|/max(|a|,|b|,ε)。要说明 throughput 单位和 weak/strong scaling。

评分点：三公式；单位；scaling类型。  
误区/追问：把 speedup 当 efficiency。追问：target tokens/s 何时优于 samples/s？

### W14-Q04

参考答案：run_id、status、source revision/hash、config hash、data revision/hash、seed、world size/rank mapping、precision、实际软件版本、checkpoint ID/hash、eval command/config/evaluator version。

评分点：代码/配置/数据；随机/资源；checkpoint/eval。  
误区/追问：只记命令文本。追问：命令为什么也应 hash？

## Explain

### W14-Q05

参考答案：load 只证明反序列化；正确 resume 还要求 optimizer moments、LR schedule、scaler、RNG、sampler、step/EMA 连续。应在独立进程用相同 next batch 比 loss/grad/update，并检查后续无重复/漏样。

评分点：状态；fixed batch；数据连续。  
误区/追问：loss 大致接近就行。追问：dropout RNG 丢失有何表现？

### W14-Q06

参考答案：floating reduction 顺序和 kernel 可产生小差异，所以不必逐位；但 fixed global batch 能隔离 world-size 对优化问题的改变，发现 sampler、reduction、LR、overflow和初始化错误。容差应由 FP32/单卡 reference 预注册。

评分点：浮点来源；为何仍测；容差依据。  
误区/追问：所有偏差都是浮点噪声。追问：一步梯度偏差 10% 合理吗？

### W14-Q07

参考答案：小模型每卡计算少，all-reduce latency、launch、data和同步成为主导；8 卡增加通信而没有足够计算隐藏它。需对照 nccl-tests、通信比例和更大 workload，不能仅以 E8 低判 NCCL 故障。

评分点：计算通信比；证据；对照。  
误区/追问：强制 ring 即可修。追问：弱扩展会怎样改变比例？

### W14-Q08

参考答案：Capstone 首要验收是数据、数值、多卡、profile、恢复和结论归因。若这些可信，treatment 无提升就是有效 FAIL-MODEL 证据；比用系统错误制造漂亮结果更有价值。

评分点：系统所有权；负结果；正确标签。  
误区/追问：主指标没涨即整周失败。追问：什么情况应为 INCONCLUSIVE？

## Apply

### W14-Q09

参考答案：单卡 global=8×4=32。四卡可 micro=2、accumulation=4；或 micro=1、accumulation=8。八卡可 micro=1、accumulation=4。还要固定 sampler 和 successful updates。

评分点：正确组合；公式；其他不变量。  
误区/追问：四卡仍 micro=8、acc=4。追问：micro 太小时性能如何？

### W14-Q10

参考答案：claim 指向 evaluation summary；summary 指向 predictions/metric、checkpoint hash、eval command/config/evaluator；checkpoint 指向 training run；run 指向 source/config/data hash、seed、precision、world size和环境。任何断链使 claim 降级为未知。

评分点：完整反向链；hash；断链处理。  
误区/追问：报告中写文件名即可。追问：best checkpoint selection 放在哪一层？

### W14-Q11

参考答案：在 runs/test_fixtures 新建合成目录，只放部分假 shard或缺 marker，写独立 manifest；调用 loader，断言非零退出和明确错误。绝不删除、截断或改真实 checkpoint。

评分点：隔离 fixture；预期拒绝；不破坏真实数据。  
误区/追问：复制真实 checkpoint 后删 shard。追问：hash 错误怎样单独测？

### W14-Q12

参考答案：单元/单卡FP32→单卡FP16→四卡fixed batch→另一四卡组smoke→八卡smoke→八卡fixed batch→scaling profile。每层需 finite、样本正确、rank一致、loss/grad容差、资源安全；前层失败不进入后层。

评分点：顺序；每层Gate；correctness先于性能。  
误区/追问：直接8卡省时间。追问：treatment何时加入多卡？

## Debug

### W14-Q13

参考答案：先比较 next batch ID；核对 model/optimizer/scheduler/scaler和step；核对 RNG/sampler/augmentation；检查 train/eval mode、EMA；核对 config/data/source hash；固定 FP32 一步复现。不要先调 LR。

评分点：数据→状态→模式→lineage→最小复现。  
误区/追问：归因正常 warmup。追问：LR相同但Adam moments丢失会怎样？

### W14-Q14

参考答案：rank 内一致只说明同步完成，不说明目标与单卡相同。检查 global batch、loss reduction、gradient accumulation、DistributedSampler样本、LR scaling、初始权重、AMP skipped update、padding/target-token分母。

评分点：全局语义；至少四根因；固定batch证据。  
误区/追问：参数一致即正确。追问：DDP平均与loss sum怎样交互？

### W14-Q15

参考答案：100-step周期首先怀疑 checkpoint/eval/log flush或barrier。用range对齐尖峰，分别关闭这些周期任务测steady state；报告p50/p95和端到端，不把尖峰平均隐藏。

评分点：周期关联；受控关闭；两种报告。  
误区/追问：GPU降频。追问：异步写盘仍可能在哪里同步？

### W14-Q16

参考答案：先固定 checkpoint/hash、model.eval、greedy/seed/max tokens；固定样本ID/order和evaluator；比较环境/tokenizer版本；核对source/config/eval command hash；逐样本diff找首次分歧。生成采样未固定不能声称 evaluator 不稳定。

评分点：checkpoint；generation；data；environment/lineage。  
误区/追问：只比较总指标。追问：dropout没关会是什么证据？

## Design

### W14-Q17

参考答案：写临时目录；各rank关闭文件并生成大小/hash；barrier；写manifest；再次校验；写COMPLETE marker；原子rename；last pointer最后更新。loader只接受 marker、manifest、hash和strategy兼容的目录。

评分点：临时写；校验；marker；原子发布；loader拒绝。  
误区/追问：先更新last再写文件。追问：对象存储无rename时怎么办？

### W14-Q18

参考答案：冻结假设、唯一变量、primary/secondary、checkpoint selection、容差、seed、样本、generation/evaluator、scaling、资源/停止和结论标签，并 hash。test 只在最终 checkpoint 运行；bug 修复必须新版本并重评所有组。

评分点：变量/指标；test隔离；资源；版本化。  
误区/追问：先看曲线再选 primary。追问：探索性指标怎样报告？

### W14-Q19

参考答案：新 shell、clean output；只给锁定代码/依赖清单/本地数据manifest/README；不联网、不问AI、不手改；一条命令 validate/dry run，自动生成manifest和终端PASS/FAIL；再独立eval加载checkpoint。

评分点：环境隔离；说明完整；唯一入口；独立eval。  
误区/追问：复用旧runs目录。追问：依赖wheel不可获取如何记录？

### W14-Q20

参考答案：公司产物从不离开公司；个人设备从官方源重新下载公开代码/数据/权重，以独立run ID和lineage运行；不复制配置、日志、数字、图或拓扑。公开报告只引用个人结果，公司经历只做无敏感数字的抽象能力总结。

评分点：物理隔离；重新获取；独立lineage；公开声明。  
误区/追问：手抄公司指标不算导出。追问：哪些抽象总结可以安全保留？

## Trade-off

### W14-Q21

参考答案：按已有data/evaluator可信度、系统bug是否清零、单卡reference、1/4/8可行、恢复状态、评测自动化和一周闭环概率评分。选择证据链完整者，不选更炫或模型更大者；周一后不换。

评分点：可执行Gate；证据优先；冻结。  
误区/追问：优先选最新模型。追问：两条都不完整怎么办？

### W14-Q22

参考答案：严格determinism便于debug，但可能禁用高性能kernel或降低吞吐；多卡归约仍有顺序差异。开发期对tiny fixed batch用强determinism，性能run用预注册统计容差，并固定seed/manifest，分别报告正确性与性能。

评分点：收益；成本；分阶段策略。  
误区/追问：性能run也必须bitwise。追问：何时必须逐位？

### W14-Q23

参考答案：DDP简单时rank0完整checkpoint便于迁移；FSDP大模型优先sharded checkpoint避免rank0 OOM。world-size迁移只在当前PyTorch 2.1 API、optimizer reshard和wrap策略有测试支持时做；否则声明只支持同world-size。

评分点：格式选择；OOM；迁移Gate；显式限制。  
误区/追问：所有shard都能任意卡数加载。追问：full state生成的峰值如何控制？

### W14-Q24

参考答案：报告系统Gate逐项PASS，然后将模型结论写为FAIL-MODEL：在预注册预算、seed和容差内未观察到收益；给效应量/方差、成本、限制和替代解释。下一步只选一个最高价值实验，或停止该算法分支。

评分点：负结果；证据边界；成本/限制；单一下一步。  
误区/追问：换指标找正结果。追问：哪些缺失会迫使改成INCONCLUSIVE？
