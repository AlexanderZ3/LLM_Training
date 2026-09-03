# Week 09 回答篇：Diffusion vs Flow Matching

> 先闭卷。每题3分：核心、公式/证据、边界各1分。以下公式限定于本文预注册路径，不代表所有scheduler/flow path。资料核验日期：2026-09-03。

## Recall

### W09-R01

参考答案：noisy=alpha_t×clean+sigma_t×noise；epsilon target=noise；loss是prediction与noise在有效动作元素上的masked MSE。alpha/sigma由锁定scheduler定义。

评分点：三项；mask；scheduler边界。  
常见误区：target写成clean但sampler仍按epsilon。  
追问：如何从epsilon_hat估clean？

### W09-R02

参考答案：x_t=(1-t)noise+t clean；target velocity=clean-noise；Euler为x_next=x+delta_t v_theta(x,t,c)，从t=0噪声积分到1。

评分点：插值；符号；更新方向。  
常见误区：target=noise-clean却仍正向积分。  
追问：非线性path会改什么？

### W09-R03

参考答案：actions/prediction/target [B,A,Da]；time [B]并广播[B,1,1]；mask [B,A]或[B,A,1]，扩展到动作维；condition常为[B,Scond,D]。

评分点：核心shape；广播；mask。  
常见误区：time为单一scalar导致batch无法独立采样。  
追问：多摄像头condition如何形成Scond？

### W09-R04

参考答案：数据revision/split/lag/normalization、模型容量/condition encoder/trainable params、初始化、global batch、optimizer/LR、windows seen、precision/accumulation、mask、预处理、seed、checkpoint selection、评测noise/episodes。

评分点：至少八项；横跨数据/模型/预算/评测；唯一变量。  
常见误区：只说“超参一致”。  
追问：哪一项允许作为objective配套差异？

## Explain

### W09-E01

参考答案：两者target尺度、timestep难度与sampling distribution不同；diffusion回归noise，flow回归clean-noise velocity。loss只在各自目标内有意义，共同比较应回到反归一化action、rollout和latency。

评分点：target差；time分布；共同指标。  
常见误区：谁MSE低谁更好。  
追问：同objective不同seed的loss可否比较？

### W09-E02

参考答案：batch、drop_last、AMP skip、world size/accum和有效mask会让每step实际windows不同。应按successful updates和effective windows seen对齐，目标差<1%。

评分点：至少三原因；正确口径；阈值。  
常见误区：attempted steps即有效更新。  
追问：skip时scheduler是否推进？

### W09-E03

参考答案：sampling steps是推理时网络调用/积分次数；训练通常只采一个t并做一次forward/backward，不运行完整sampler。训练吞吐由模型、batch、backward和通信决定。

评分点：train/inference分离；调用差；profile。  
常见误区：Euler 5步所以训练也快2倍。  
追问：objective实现为何仍可能改变训练成本？

### W09-E04

参考答案：MSE平均局部误差，不覆盖闭环分布偏移、接触阈值、时序、恢复和多模态平均。小坐标误差可能跨越成败边界；反之高MSE轨迹也可能成功。

评分点：闭环；任务阈值；错误桶。  
常见误区：完全否定offline metric。  
追问：如何联合使用两类指标？

## Apply

### W09-A01

参考答案：input=(1-0.25)(-1)+0.25×2=-0.75+0.5=-0.25；velocity=2-(-1)=3。

评分点：input；target；方向。  
常见误区：input=0.25。  
追问：Euler delta=0.1且预测准确时下一点？

### W09-A02

参考答案：mask_exp=mask.unsqueeze(-1).expand(B,A,Da)；loss=sum(mask_exp×(pred-target)²)/max(mask_exp.sum(),1)。若用mask.sum×Da，需确认每动作维同权。

评分点：广播；numerator；denominator。  
常见误区：先对全张量mean再乘mask。  
追问：每动作维有独立validity时怎样改？

### W09-A03

参考答案：D-s17、F-s17、D-s29、F-s29；共同config，只有objective/scheduler/sampler白名单差异；对齐successful updates/windows；每run按预注册同一validation metric在固定step或规则选checkpoint，不能各挑最好。

评分点：矩阵；预算；selection。  
常见误区：第二seed只跑输的一方。  
追问：预算不足怎样降级仍公平？

### W09-A04

参考答案：同GPU、checkpoint、validation inputs与initial noise，batch=1，warmup，100+ samples；各K=5/10/20，CUDA event测完整sampler loop的p50/p95。排除权重加载、数据下载和环境rendering；预后处理是否包含要写明且一致。

评分点：固定项；计时边界；重复统计。  
常见误区：用第一次推理。  
追问：solver每step网络调用不同时怎么报告？

## Debug

### W09-D01

参考答案：不从loss判优。先核对target/mask reference、反归一化与sampler匹配，再看相同K/initial noise、checkpoint selection、windows、两个seed和rollout错误桶。可能是真实质量差，也可能实现/评测不公平。

评分点：loss不可比；实现审计；共同指标。  
常见误区：调高flow训练步数。  
追问：何时判FAIL-MODEL？

### W09-D02

参考答案：确认每次重置torch/CUDA/random RNG、initial noise完全相同；model eval模式、dropout关闭；scheduler/timestep顺序固定；数据augment关闭；deterministic kernel限制与异步状态；processor无随机性。

评分点：noise；model模式；scheduler/数据RNG。  
常见误区：只设置一次全局seed。  
追问：完全bitwise deterministic是否必要？

### W09-D03

参考答案：先查AMP skips、drop_last/loader耗尽、global batch与mask有效数。补跑到相同effective windows或按预注册截断较多一方；差异>1%未校正则不作主比较，标FAIL-SYSTEM/INCONCLUSIVE。

评分点：根因；校正；不硬比较。  
常见误区：按attempted step直接作图。  
追问：wall-time等预算会回答不同问题吗？

### W09-D04

参考答案：审计optional dependency与attn配置，选择同一eager/经验证PyTorch2.1 SDPA兼容后端，两个objective一致；不安装未知SM70 fork、不升级torch。若锁定commit硬依赖FA2，换经批准兼容commit或同一surrogate并降级结论。

评分点：共同fallback；版本纪律；结论边界。  
常见误区：只让flow回退eager。  
追问：如何证明没有偷偷走不同backend？

## Design

### W09-G01

参考答案：固定small tensors与t=0、0.25、1；分别手算input/target，断言shape和值；用完美prediction验证loss=0，用反号prediction确保loss>0；测试mask全/部分/padding；flow从noise到clean的Euler小步方向应正确。

评分点：端点；符号负例；mask。  
常见误区：只检查shape。  
追问：diffusion scheduler怎样做golden test？

### W09-G02

参考答案：固定ID split和一个预注册OOD轴（如物体位置）；两个objectives/seed用同episode seeds，各30–50+；记录success interval和错误桶。reset failure/crash单列为system denominator，不静默计模型失败或删除。

评分点：同seed；OOD；system单列。  
常见误区：失败episode全丢弃。  
追问：system failure过高时如何裁决？

### W09-G03

参考答案：自动规范化并diff resolved configs。允许objective、prediction_type、noise/timestep schedule、与之配套solver字段；模型、数据、batch、optimizer、seed policy、precision、mask、steps/windows、logging/eval等差异直接阻断。

评分点：白名单；阻断项；保存diff。  
常见误区：认为官方推荐超参可以各自不同。  
追问：若研究“各自最佳配置”该如何另立实验？

### W09-G04

参考答案：公司V100原始产物完全留内部；个人5070Ti用公开数据从零生成且独立manifest；H100仅授权补充。三个硬件分别报告torch、precision、backend，不搬数值、不合并曲线、不从新GPU推断V100。

评分点：隔离；独立生成；硬件标注。  
常见误区：把公司图重新绘制后公开。  
追问：哪些方法论可公开描述而不含内部事实？

## Trade-off

### W09-T01

参考答案：同为10次离散更新、相同checkpoint/硬件/输入时可比较端到端质量和延迟；但DDIM与Euler积分对象、单步运算、scheduler开销及训练objective不同，差异是“objective+sampler系统”，不能只归因solver名称。

评分点：共同条件；剩余差异；结论单位。  
常见误区：10=10即完全相同计算。  
追问：如何测每次network evaluation成本？

### W09-T02

参考答案：第二seed检验训练结论的随机稳定性，通常比在同checkpoint多一个K更关键；若主要产品问题是实时延迟拐点，5/10/20 sweep也必要。最低先2seed与主K=10，再扩K。

评分点：证据层次；任务依赖；最低顺序。  
常见误区：用很多K替代seed。  
追问：什么时候第三seed优先？

### W09-T03

参考答案：例如flow在两seed latency/jerk更好，diffusion在ID/OOD success或endpoint更好，且差异有稳定信号；目标权重未预注册时不能硬合成总分，应报告Pareto与适用场景。

评分点：多指标冲突；跨seed；不事后加权。  
常见误区：挑一个喜欢的metric判胜。  
追问：如何预注册业务优先级？

### W09-T04

参考答案：同一冻结特征可绕过下载/kernel阻塞，保留objective、mask、sampler和训练系统练习；但结论只适用于surrogate feature条件，不能外推完整视觉端到端VLA。两组特征和缓存必须一致。

评分点：收益；一致；外推边界。  
常见误区：把surrogate胜负写成SmolVLA结论。  
追问：冻结缓存会改变数据瓶颈吗？

## 复训

低于44/72：重做W09-A01、W09-A02、W09-G03的closed-book测试，并在toy上证明两个sampler deterministic。安全/硬件边界错误优先复训。
