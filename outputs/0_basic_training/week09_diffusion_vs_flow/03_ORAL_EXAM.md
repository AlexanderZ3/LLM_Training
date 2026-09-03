# Week 09 拷问篇：Diffusion vs Flow Matching

> 先闭卷，不得打开回答篇。建议 55 分钟；所有推导使用 clean_actions/noise 等语义名，不能只写含混的 x0/x1。资料核验日期：2026-09-03。

## Recall

### W09-R01

写出 epsilon-prediction diffusion 的扰动公式、target 与 loss。

### W09-R02

写出线性路径 flow matching 的插值、target velocity 与 Euler 更新。

### W09-R03

给出 actions、time、mask、prediction 的 shape。

### W09-R04

列出本周 objective 外必须固定的至少八个变量。

## Explain

### W09-E01

为什么 diffusion 与 flow 的 train MSE 数值不能直接比较？

### W09-E02

为什么相同 training steps 不一定是相同预算？

### W09-E03

为什么 sampling steps 少不推出训练吞吐更高？

### W09-E04

为什么 offline action MSE 与 rollout success 可能方向相反？

## Apply

### W09-A01

clean=2、noise=-1、t=0.25，计算线性 flow 的 input 与 target velocity。

### W09-A02

对 prediction shape [B,A,Da] 和 mask [B,A] 写出正确 masked MSE。

### W09-A03

设计 2 objectives×2 seeds 的等预算训练矩阵与 checkpoint selection。

### W09-A04

设计 5/10/20 sampling steps 的延迟测量，明确包含和排除什么。

## Debug

### W09-D01

flow loss 比 diffusion 低一半，但 rollout 更差。你先查什么？

### W09-D02

diffusion sampler 每次用同 seed 仍产生不同输出。列出定位路径。

### W09-D03

四个 run 的 attempted steps 相同，但 flow 的 windows seen 少 7%。怎样处理？

### W09-D04

V100 上 LeRobot import 试图加载 flash-attn 并失败。如何回退且保持公平？

## Design

### W09-G01

设计人工 tensor 的 target/reference 单测，能抓住时间方向与 target 符号错误。

### W09-G02

设计一个 ID 与一个 OOD rollout 评测，如何处理系统 crash？

### W09-G03

设计 config diff gate，哪些差异允许，哪些直接 FAIL-SYSTEM？

### W09-G04

设计公司 V100、个人 5070 Ti、可选 H100 的结果隔离。

## Trade-off

### W09-T01

比较 DDIM 10 steps 与 Euler 10 steps，什么相同，什么仍不可直接归因？

### W09-T02

比较多跑一个 seed 与多跑一个 sampling step 点的证据价值。

### W09-T03

什么证据支持 TRADEOFF，而不是强行判一方获胜？

### W09-T04

完整 encoder 不兼容时，用冻结特征 surrogate 的收益与结论边界是什么？

## 评分

每题 0–3，共72。58+可独立；44–57需小提示；30–43公式或实验控制薄弱；低于30重做toy。

任何外带公司代码/数据/日志/trace/图片/checkpoint/指标、在V100上强装官方FA2/BF16、或用单seed最低train loss宣称胜出，相关题0分。
