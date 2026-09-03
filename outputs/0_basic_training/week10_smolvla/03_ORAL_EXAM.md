# Week 10 拷问篇：SmolVLA 微调闭环

> 先闭卷，禁止查看回答篇。建议60分钟；必须画数据与模型shape，区分锁定revision与当前main。资料核验日期：2026-09-03。

## Recall

### W10-R01

画出SmolVLA从图像、state、instruction到action chunk的高层数据流。

### W10-R02

写出action、time、mask与flow velocity target的shape和公式。

### W10-R03

列出版本四元组及strict load需要检查的三类key问题。

### W10-R04

列出完整resume必须恢复的状态。

## Explain

### W10-E01

为什么processor属于模型正确性，而不是外围预处理？

### W10-E02

camera order错误时，为什么loss仍可能下降？

### W10-E03

为什么8卡不能解决V100不兼容的kernel？

### W10-E04

freeze vision/VLM后，哪些显存会下降，哪些仍存在？

## Apply

### W10-A01

给出images/state/text/actions/mask的batch shape审计表。

### W10-A02

设计batch 1/2/4/8 memory scan与安全停止条件。

### W10-A03

设计128样本overfit和condition shuffle对照。

### W10-A04

设计base vs fine-tuned的固定评测协议。

## Debug

### W10-D01

import时尝试编译flash-attn并失败，如何定位和回退？

### W10-D02

checkpoint出现missing/unexpected keys，但strict=false后forward能跑。能否继续？

### W10-D03

单卡正常，四卡loss异常。给出排查顺序。

### W10-D04

resume后第一batch loss差8%，step编号正确。还要查哪些状态？

## Design

### W10-G01

设计compatibility.md，使另一个工程师能判断SM70会走哪条attention/precision路径。

### W10-G02

设计一个版本适配薄层，如何避免吞掉LeRobot未知flag？

### W10-G03

设计单卡→四卡→可选八卡gate及每层证据。

### W10-G04

设计Model Card，使系统失败、模型失败和证据不足可区分。

## Trade-off

### W10-T01

比较action-expert-only、unfreeze top block与full fine-tune的显存、适应性和风险。

### W10-T02

比较锁定stable commit与追main的收益和风险；公司PyTorch2.1下如何选择？

### W10-T03

比较四卡长训与八卡topology sample的GPU-hours价值。

### W10-T04

offline action error改善但rollout不改善时，如何裁决与下一步？

## 评分

每题0–3，共72。58+可独立；44–57需review；30–43只能调用框架；低于30回到processor/overfit。

任何建议外带公司代码、数据、日志、trace、图片、checkpoint或指标，强升公司torch/CUDA，在V100用BF16/官方FA2，或用strict=false忽略核心head，相关题0分并触发复训。
