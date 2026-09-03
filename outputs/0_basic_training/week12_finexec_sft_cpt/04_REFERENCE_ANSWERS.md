# Week 12 参考答案：FinExec Evaluator、SFT 与 CPT

> 必须先闭卷完成 03_ORAL_EXAM.md。每题按 0–3 分评分；能复述答案但不能通过追问，最多 2 分。

## Recall

### W12-Q01

参考答案：evidence 是存在于当前文档的去重 ID 列表；program 是白名单 DSL 字符串；answer 是可比较的十进制结果；scale 是固定枚举。四者可独立检查，reported answer 必须与 program 执行并应用 scale 后的值一致。

评分点：字段/类型；存在性；一致性。  
误区/追问：把 evidence 写成自由文本。追问：空 evidence 何时允许？

### W12-Q02

参考答案：JSON parse、schema、evidence existence、program parse、op/arity/type、finite execution、scale、gold executed answer、reported consistency、evidence F1。first failure 用于互斥 waterfall；全部 flags 保留一条样本的多重错误。

评分点：顺序；至少八项；两种记录。  
误区/追问：只统计成功样本。追问：execution_accuracy 的分母是什么？

### W12-Q03

参考答案：L=-Σm_t log p(y_t|x_<t)/max(Σm_t,1)。prompt 与 padding 的 m=0/label=-100，目标 JSON 与 EOS 的 m=1。按有效 target token 归一化。

评分点：公式；mask；EOS；分母。  
误区/追问：prompt 也监督。追问：框架内部 shift 如何验证？

### W12-Q04

参考答案：program result 是 DSL 原始执行值；scaled result 是依据 scale 做单位转换后的值；reported answer 是模型显式输出。三者分开能发现 scale 错误和“答案碰巧对、过程错”。

评分点：三层；scale；一致性。  
误区/追问：只比较 answer 与 gold。追问：0.3223 和 32.23 如何关联？

## Explain

### W12-Q05

参考答案：evaluator 是训练选择、评测乃至 reward 的测量仪器；它错会把模型行为系统性误标。100% 指明确清洗、记录 exceptions 后的有效 gold set，在 parse、execute、answer match 上全部通过，不代表数据无歧义或模型正确。

评分点：测量因果；范围；exceptions。  
误区/追问：删除失败样本不留记录。追问：新增 op 后要做什么？

### W12-Q06

参考答案：同一数学程序可有交换顺序、等价嵌套或格式差异，字符串 exact match 会误罚等价程序。execution 比较语义结果；但仍要配合 op/evidence/scale 检查，防止偶然答案。

评分点：语义等价；exact match 局限；仍需结构。  
误区/追问：execution 对就忽略 evidence。追问：如何处理容差？

### W12-Q07

参考答案：模型可能抄到答案、利用数据捷径或输出不可执行推理；这无法复核、无法做稳定 reward，也不能证明过程正确。应将 answer 和 execution 分项报告。

评分点：捷径；可审计；分项。  
误区/追问：最终数字对就算成功。追问：反过来 program 对、reported answer 错如何记？

### W12-Q08

参考答案：0.5M–2M token 太少且分布窄，任何变化都可能来自格式适应、优化噪声或遗忘；没有独立知识基准和足够 seed。它只能验证数据、mask、checkpoint 迁移与成本。

评分点：预算；替代解释；证据边界。  
误区/追问：loss 降即知识增加。追问：怎样设计更强的知识迁移证据？

## Apply

### W12-Q09

参考答案：前 40 个 prompt token label=-100；接下来的 20 个 JSON token使用对应 token ID；EOS 使用 eos_token_id 并计 loss；最后 4 个 pad=-100。具体 shift 由模型 loss 实现决定，先用 logits/labels 最小样例确认。

评分点：40/20/4；EOS；shift 警惕。  
误区/追问：EOS/pad 都监督。追问：left padding 会改变哪些索引？

### W12-Q10

参考答案：根是 divide，左子树 subtract(128.4,97.1)，右子为 97.1；原始结果约 0.32235；scale=percent 后乘 100，按冻结规则舍入为约 32.23；reported answer 再与该值比较。

评分点：AST；执行次序；scale/舍入。  
误区/追问：先把每个 operand 当 percent。追问：old=0 时返回什么错误？

### W12-Q11

参考答案：至少覆盖正/负数、逗号、货币、括号负数、percent、thousand/million、nested op、wrong arity、unknown op、divide zero、NaN/Inf、rounding、absolute/relative tolerance、scale mismatch、invalid evidence、answer inconsistency、深度/长度限制。

评分点：正常、边界、恶意、单位四类。  
误区/追问：只有 happy path。追问：property-based test 可验证什么？

### W12-Q12

参考答案：固定 SFT train/dev IDs、prompt/schema、mask/packing、有效 target tokens 或更新预算、LR 等 SFT 配置、generation、evaluator、dev checkpoint 选择、seed。Run C 唯一主要差异是从短 CPT checkpoint 初始化，并记录 optimizer 是否重置。

评分点：数据；训练；评测；唯一差异。  
误区/追问：Run C 增加 SFT steps。追问：tokenizer 不同还能怎样比较？

## Debug

### W12-Q13

参考答案：先 decode supervised labels 查 mask；验证 causal shift；核对 train/generate prompt 模板、prompt slicing；检查 EOS/max_new_tokens、JSON escaping 和 truncation；关闭 packing；最后才调 LR/模型。

评分点：labels→shift→生成→packing→优化。  
误区/追问：直接训练更久。追问：如何看出只学复制 prompt？

### W12-Q14

参考答案：用两个内容完全不同的短样本，分别单独与 packed forward；检查 B 的 token 表示/logits 是否随 A target 改变；可视化 attention/segment mask；验证 boundary labels、position_ids、EOS。若不能证明 block 隔离，关闭 packing。

评分点：干预测试；mask/position/labels；安全回退。  
误区/追问：吞吐高所以保留 packing。追问：causal mask 为什么仍可能让 B 看见 A？

### W12-Q15

参考答案：不能进入正式训练。输出 first-failure IDs，人工核对 parser、operator、scale、normalization、rounding与原标注；真实异常写 exceptions manifest 和理由，在冻结后的有效 set 达到 100%，不静默修改原始文件。

评分点：停止；根因；显式 exceptions；重新冻结。  
误区/追问：98.7% 足够大。追问：如何避免清洗规则看 test 后过拟合？

### W12-Q16

参考答案：数据：sampler 样本顺序/重复、global batch、padding/token 分母；数学：loss reduction 与 DDP averaging、accumulation/no_sync、LR 缩放；系统：不同权重/config、AMP skips、rank divergence。先固定同一批样本和 FP32 一步梯度，再扩到 20 step。

评分点：三类；一步 oracle；逐步扩展。  
误区/追问：归因浮点误差。追问：怎样比较更新后参数？

## Design

### W12-Q17

参考答案：自写 tokenizer/递归下降 parser 产生 AST；只允许固定 op、常数、reference；限制字符、长度、深度、arity、dtype；context lookup 只读；Decimal/有限检查；超时/异常转明确错误。执行器无文件、网络、shell、import 能力。

评分点：白名单；资源限制；只读上下文；错误类型。  
误区/追问：用 ast.literal_eval 就能执行任意 DSL。追问：如何防超深嵌套？

### W12-Q18

参考答案：记录 dataset/repo/commit/license、原始文件 hash、adapter code hash、official split、sample/document/evidence ID 映射、normalization/scale 规则、duplicates、长度/截断、exclusions、生成时间。manifest 本身做 hash。

评分点：来源；split；变换；排除；hash。  
误区/追问：只记文件路径。追问：为何 normalized value 与 raw text 都保留？

### W12-Q19

参考答案：预注册 dev 主指标，如 execution accuracy 加格式 gate；固定 eval 周期和 greedy config；只按 dev 选 best；选择后冻结 checkpoint hash；test 只运行一次，报告全部 waterfall 与限制。发现测量 bug 必须版本化重评所有模型。

评分点：dev selection；冻结；test 一次；bug 处理。  
误区/追问：每轮看 test 选模型。追问：多个 dev 指标如何组合？

### W12-Q20

参考答案：同时画 target loss、valid JSON、valid program、execution、answer、evidence F1、scale、reported consistency、response length。若格式指标升而 program/execution/evidence 不升，即只学格式；按 op/operand/evidence 错误桶进一步验证。

评分点：分层面板；诊断模式；错误桶。  
误区/追问：只看 answer accuracy。追问：format token 比例如何影响 loss？

## Trade-off

### W12-Q21

参考答案：严格 schema 带来确定解析、分项错误和可靠 reward，但增加格式 token、截断与 teacher-forcing 易学捷径。若 0.6B 长期卡在格式且 schema 含冗余字段，可在训练前简化表示；不能看 test 后删难字段。

评分点：可测性；训练负担；预注册简化。  
误区/追问：自由文本更自然所以更好。追问：如何保留 process audit？

### W12-Q22

参考答案：关闭 packing 损失有效 tokens/s 和显存利用率，但保住样本隔离、mask 可解释性和可靠 reference。只有 block attention、position、label boundary 单测与 packed/unpacked fixed-batch 等价通过后再开。

评分点：吞吐；正确性；重新开启 gate。  
误区/追问：只比较 train loss。追问：packed 与 unpacked 怎样固定有效 token？

### W12-Q23

参考答案：若模型和 optimizer 单卡可容纳，先单卡 correctness；DDP 用于复制模型并扩吞吐，简单；FSDP 用于降低 model-state 显存，但引入 all-gather、reduce-scatter 和 checkpoint 复杂。只有显存压力或学习 sharding 目标明确时用 FSDP。

评分点：显存语义；复杂度；分阶段。  
误区/追问：DDP 汇总 8 卡显存。追问：activation OOM 时 FSDP一定解决吗？

### W12-Q24

参考答案：定性为格式学习明显、推理收益弱且 evidence 退化，不能称任务成功。先看错误 waterfall 和训练目标是否被格式 token 主导，检查 evidence mask/数据与截断；下一步优先改数据/evaluator或目标权重，而不是直接放大/RL。

评分点：证据边界；退化识别；正确优先级。  
误区/追问：valid JSON 大涨就进入 GRPO。追问：什么 gate 才允许 Week 13 RL smoke？
