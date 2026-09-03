# Week 04 回答篇：TinyFlowPolicy

> 先闭卷。每题 4 分，必须包含机制、shape/式、证据、边界。

## Recall

### W04-R01
答案：`x0,x1,xt,ut,vθ:[B,H,Da]`，`t:[B,1]`广播为`[B,1,1]`；x0噪声、x1数据、xt插值、ut目标速度。评分点：shape2；语义2。误区：x0=数据。追问：condition shape？

### W04-R02
答案：`xt=(1-t)x0+t x1`，`ut=x1-x0`；loss为有效位置 squared error总和除`mask.sum()*Da`。评分点：三式3；FP32聚合1。误区：固定BHD分母。追问：mask全0怎么办？

### W04-R03
答案：`x_{k+1}=x_k+Δt vθ(x_k,t_k,c)`，从标准高斯在0积分到1。评分点：更新2；方向1；初值1。误区：1→0直接套同式。追问：非均匀grid呢？

### W04-R04
答案：EMA weights/decay/update count；model/optimizer/scheduler/scaler、RNG、sampler、step、normalization/config/data hash。评分点：EMA2；其余2。误区：只存shadow weights。追问：eval swap如何恢复？

## Explain

### W04-E01
答案：线性conditional path给定样本时`dxt/dt=x1-x0`可直接监督速度场；ODE仅在生成时把噪声推到数据。评分点：导数2；训练/推理分工2。误区：每次训练都采样完整轨迹。追问：换非线性path会怎样？

### W04-E02
答案：t表示路径位置/噪声程度，同一xt可能来自不同阶段且目标场不同；无t会把不同向量场平均。评分点：歧义2；实现1；实验1。误区：t只是学习率。追问：time embedding怎么测？

### W04-E03
答案：action chunk作为同时生成的整体，不存在“未来动作标签泄漏”式自回归约束；双向可建模chunk内部关系。语言next-token必须遮未来真实token。评分点：任务因果2；mask2。误区：机器人一定要causal。追问：在线autoregressive action时呢？

### W04-E04
答案：分布多峰时均值动作可能不可执行；误差会闭环累积；MSE未计稳定性、碰撞、latency。评分点：三因3；需rollout1。误区：offline最好必然部署最好。追问：哪些proxy可先用？

## Apply

### W04-A01
答案：`200×7=1400` 个有效标量。评分点：答案2；解释mask timestep扩展Da2。误区：8×32×7。追问：每batch有效比不同如何聚合？

### W04-A02
答案：k0 `[10,11,12,P]`,mask `[1,1,1,0]`; k1 `[11,12,P,P]`,`[1,1,0,0]`; k2 `[12,P,P,P]`,`[1,0,0,0]`。评分点：每行1；不跨episode1。误区：用下个episode补齐。追问：是否包含终止action？

### W04-A03
答案：令`v(x,t,c)=a`，任意x0、任意网格总Δt=1，Euler应得`x0+a`，误差接近浮点容差且不依赖步数。评分点：场1；oracle1；网格1；容差1。误区：拿训练模型作oracle。追问：线性场如何测收敛阶？

### W04-A04
答案：固定split/windows、normalization、seed、optimizer steps/seen samples、global batch、容量/optimizer、eval episodes；Flow固定solver/initial noise。评分点：四类控制。误区：只固定epoch。追问：推理预算怎么公平？

## Debug

### W04-D01
答案：核对预测参数化与target；时间方向/Δt；恒定场solver test；normalization/inverse；train t范围；EMA checkpoint；最后扫steps。评分点：有序四层。误区：先增1000步。追问：fixed noise有何价值？

### W04-D02
答案：构造相同valid区域但不同padding值/长度，正确loss应相同；检查mask扩为`[B,H,1]`且分母`mask.sum()*Da`。评分点：对照2；修复2。误区：padding填0就安全。追问：attention mask仍可能有什么bug？

### W04-D03
答案：condition未接入/被mask；encoder无grad；数据action可由时间/边际预测；shuffle实现仍配原样本；模型容量/训练不足。用hooks、grad、checksum、需condition子集验证。评分点：四个解释带实验。误区：直接加condition dropout。追问：zero差但shuffle不差？

### W04-D04
答案：检查EMA weights/decay/update count、是否在skip时更新、保存时raw/EMA对应、eval swap还原、scaler成功step语义。评分点：四层。误区：只恢复model。追问：scheduler偏离如何影响EMA？

## Design

### W04-G01
答案：Stage1固定128 windows+x0+t，验证映射可记忆；Stage2同windows但随机x0/t，验证分布训练；两级都用FP32、固定seed/无dropout，失败分别定位实现与覆盖难度。评分点：两阶段2；控制1；归因1。误区：直接正式长训。追问：BC gate放哪里？

### W04-G02
答案：固定checkpoint/condition/initial noise/eval episodes，只改Euler steps 5/10/20/50；报告latency p50/p95、offline/rollout、数值oracle；选择Pareto点。评分点：控制2；指标1；决策1。误区：每steps换seed。追问：warmup为何重要？

### W04-G03
答案：打印episode数、success/Wilson区间、endpoint/jerk分位数、错误桶计数、ID/OOD、seed/config/hash；轨迹和原始数据留本机。评分点：统计2；证据链1；边界1。误区：只报平均success。追问：无图怎样查振荡？

### W04-G04
答案：统一`ConditionBatch`含typed tensors/masks/schema；encoder输出固定`[B,Nc,D]`或global embedding；policy只依赖接口，manifest记录camera/language/state与normalization。评分点：接口2；mask1；版本1。误区：在train loop写死state concat。追问：缺失模态怎么表达？

## Trade-off

### W04-T01
答案：BC一步快稳但多峰平均；Flow可建模多峰、采样可控，但需噪声/t训练、多步推理与更多失败面。评分点：建模2；系统2。误区：Flow必胜。追问：何时BC更合理？

### W04-T02
答案：prepend透明但增序列；FiLM/AdaLN高效但隐式；cross-attention灵活多token但实现/算力高。由condition粒度和诊断需要选。评分点：三者3；原则1。误区：接口可随意互换不影响参数量。追问：图像patch更适合哪种？

### W04-T03
答案：高decay平滑强但响应慢；低decay跟随快但噪声大；只按成功step更新；FP32 shadow稳但增内存。评分点：三权衡3；记录count1。误区：每micro-step更新。追问：早期bias如何处理？

### W04-T04
答案：H增大会增加训练输出/attention/activation并改变任务；solver steps主要线性增加推理调用/延迟，不改训练数据shape。二者都可能影响闭环但不能混扫。评分点：两轴各2。误区：把solver step当horizon。追问：如何做2D实验仍可归因？

## 自评

记录日期、分项分、AI辅助等级、关键错误、对应最小单测与回归日。看答案后复述不算 A0/A1 证据。
