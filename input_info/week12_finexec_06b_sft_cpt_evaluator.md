# 第 12 周执行手册：FinExec 0.6B——evaluator、SFT、短 CPT 与完整 LLM 后训练

> 对应原 12 周主线第 10 周。标准投入 7.5 小时。  
> 本周是第二次深入 LLM 训练：把第 01–08 周的 tokenizer/mask/AMP/DDP/FSDP 用到可程序验证的金融数值推理。  
> 不做 Agent、RAG、交易、PnL reward；只做 evidence → program → answer。

每日时间盒：10 分钟写 evaluator/训练假设，50–60 分钟核心实现，15–20 分钟跑 gold/生成测试并记录；第 2 小时只做一个错误桶或规模点。

## 1. 固定任务

输出 schema：

    {
      "evidence": ["table:r3c2", "text:p4s1"],
      "program": "divide(subtract(128.4,97.1),97.1)",
      "answer": 32.23,
      "scale": "percent"
    }

程序由 deterministic executor 执行。LLM 负责：

- 选择证据；
- 生成受限程序；
- 输出 scale；
- 输出答案。

确定性代码负责：

- parse；
- type check；
- program execution；
- unit/scale；
- tolerance；
- error classification。

## 2. 公开资源

| 资源 | 用途 | 地址 |
|---|---|---|
| FinQA 官方仓库 | 财报表格/文本、程序监督、split | https://github.com/czyssrs/FinQA |
| TAT-QA 官方仓库 | 表格+文本金融问答 | https://github.com/NExTplusplus/TAT-QA |
| Qwen 小模型 | 0.5B–0.6B base 候选 | https://huggingface.co/Qwen |
| Transformers | tokenizer/model/loading | https://github.com/huggingface/transformers |
| CS336 A1/A5 | LM 训练与 SFT/RL 思路 | https://cs336.stanford.edu/spring2025/ |

模型选择 gate：

1. 首选已批准且能在现有 transformers/torch 上加载的 0.5B–0.6B base checkpoint；
2. 可选择 Qwen3-0.6B-Base；若版本依赖冲突，退回 Qwen1.5-0.5B；
3. 不为了一个新模型升级 torch/CUDA；
4. 必须使用 base 而非 chat，除非明确研究 chat template；
5. 记录 model revision/license/tokenizer；
6. 若权重下载受限，可用第 01 周 MiniGPT 扩到 100M 做 pipeline correctness，但最终效果标记 INCONCLUSIVE。

## 3. 获取

    git clone --depth 1 https://github.com/czyssrs/FinQA.git third_party/FinQA
    git clone --depth 1 https://github.com/NExTplusplus/TAT-QA.git third_party/TAT-QA
    git clone --depth 1 https://github.com/huggingface/transformers.git third_party/transformers-reference

仅第三个用于读源码，不从 main 安装。训练使用锁定的 pip transformers。

模型示例：

    hf download Qwen/Qwen3-0.6B-Base --local-dir artifacts/qwen3-06b-base

若 repo 名或可用性与当时 Hub 不一致，先在 Qwen 官方组织页确认，不猜测第三方镜像。公开权重下载需要按公司网络政策；不把个人 token 写入命令历史。

## 4. 目录

    week12-finexec/
      data/
        raw/
        processed/
        manifests/
      src/
        schema.py
        finqa_adapter.py
        tatqa_adapter.py
        evidence_ids.py
        program_parser.py
        executor.py
        evaluator.py
        collator.py
        packer.py
        train_sft.py
        train_cpt.py
        generate_eval.py
        error_waterfall.py
      tests/
        test_executor.py
        test_scale.py
        test_evidence.py
        test_label_mask.py
        test_packing.py
        test_gold_eval.py
      configs/
        sft_06b.yaml
        cpt_short.yaml
      reports/
        data_card.md
        model_card.md
        decision.md

## 5. 先定义错误 waterfall

每条生成只落入可组合但顺序明确的检查：

1. output parse success；
2. JSON schema valid；
3. evidence IDs exist；
4. program parses；
5. operation/arity valid；
6. execution finite；
7. scale valid；
8. executed answer matches gold；
9. reported answer matches executed result；
10. evidence precision/recall/F1。

至少报告：

    valid_json_rate
    valid_program_rate
    execution_rate
    execution_accuracy
    answer_accuracy
    scale_accuracy
    evidence_f1
    reported_vs_executed_consistency

## 6. 每日安排

### 周一：gold evaluator 必须 100%

目标：在训练前建立不可争辩的 reward/eval oracle。

实现受限 DSL，操作符先只支持数据中必要子集：

- add；
- subtract；
- multiply；
- divide；
- exp；
- greater；
- table lookup/reference；
- percent scale；
- constant。

parser 不允许 eval/exec 任意代码。用 tokenizer/AST/递归下降解析器。安全和确定性优先。

单测：

- 正负数；
- comma/currency；
- percent；
- divide by zero；
- nested ops；
- wrong arity；
- unknown op；
- NaN/Inf；
- rounding/tolerance；
- scale mismatch；
- evidence ID 不存在；
- reported answer 与执行结果不一致。

在 official gold program 上运行：

    gold_parse_rate
    gold_execution_rate
    gold_answer_accuracy

进入训练的硬门槛：

    gold evaluator = 100%

如果官方标注有少数异常：

- 人工核查；
- 建 exceptions manifest；
- 不静默改 gold；
- 100% 指的是对清洗后、明确列出的有效 gold set。

当天输出：

- evaluator spec；
- 30+ unit cases；
- gold error list；
- tolerance/rounding 规则；
- error waterfall。

可选第二小时：property-based tests，验证 add commutativity 等适用性质。

### 周二：数据适配、prompt、answer mask 与 packing

目标：把 FinQA/TAT-QA 转成统一、无泄漏的监督。

统一输入：

    document:
      table with stable row/column IDs
      text sentences with stable paragraph/sentence IDs
    question
    output JSON

要求：

- table cell ID 稳定；
- sentence ID 稳定；
- prompt 中 evidence ID 与 output 可对应；
- 数字原始格式与 normalized value 分开；
- unit/scale 明确；
- split 沿用官方；
- 不将 dev/test 放进 CPT/SFT train；
- 记录 dataset commit/hash。

label mask：

- prompt tokens：ignore；
- padding：ignore；
- output JSON tokens：计算 loss；
- EOS：计算 loss；
- packing boundary：样本间不得互相 attention，或明确使用文档支持的 block mask；
- 如果实现不能正确隔离 packed samples，本周先不 packing。

单测：

1. 1 条样本不 packing；
2. 2 条不同长度 padding；
3. 2 条 packing；
4. 超长 truncation；
5. JSON 被截断时样本应丢弃或安全重构；
6. labels decode 只得到目标；
7. position_ids/attention_mask 正确。

统计：

- input/output token length p50/p95/max；
- truncation rate；
- valid evidence ID rate；
- program op distribution；
- answer/scale distribution；
- FinQA/TAT-QA 占比；
- duplicates across split。

PASS：

- label mask tests 全过；
- packing 不泄漏；不确定则关闭；
- official split 保持；
- truncation 不破坏 JSON；
- Data Card 完成。

### 周三：128 样本 overfit 与单卡 FP16 SFT

目标：先证明 0.6B 模型能学习格式、证据和程序。

overfit set 要覆盖：

- 不同操作符；
- table/text evidence；
- percent/none/thousand/million；
- nested programs；
- 正负数；
- FinQA/TAT-QA。

训练：

1. FP32 或 FP16 单 batch 20-step sanity；
2. FP16 AMP；
3. 128 样本 500–2k step；
4. greedy generation；
5. evaluator；
6. 查看格式/程序/答案分别何时学会。

检查：

- only target tokens loss；
- tokenizer special IDs；
- padding side；
- EOS；
- gradient accumulation；
- loss normalization 按有效 target tokens；
- GradScaler；
- unscale 后 clip；
- trainable params；
- checkpoint/resume。

overfit PASS：

- valid JSON 目标接近 100%；
- valid program 显著提高；
- execution accuracy 在 128 train 上显著提高；
- 不能只看 teacher-forced loss；
- 生成使用 greedy，固定 max_new_tokens；
- fixed-batch resume 相对 loss 偏差 <1%。

如果 loss 降而生成 JSON 无效：

- train/generate prompt mismatch；
- EOS；
- label shift；
- decode prompt slicing；
- max_new_tokens；
- JSON quotes/escaping；
- packing boundary。

### 周四：正式 SFT 与短 CPT rehearsal

目标：完成后训练主线，同时把 CPT 作为流程实验而非知识神话。

#### Run S：base → SFT

- 0.5B–0.6B；
- 全参；
- 单卡或 2/4/8 卡 DDP/FSDP；
- 固定 global batch；
- 200–500 step minimum，时间允许更长；
- best checkpoint 按 dev execution accuracy 或预注册组合指标；
- 不能按 test 选择；
- 每次 eval 使用固定 generation config。

#### Run C：base → short CPT → SFT

CPT corpus：

- 只使用 train split 中可合法使用的金融文本/表格线性化；
- 0.5M–2M tokens；
- next-token objective；
- 不含 dev/test answers；
- tokenizer 不在 test 上重训；
- 保存 CPT checkpoint；
- 再用与 Run S 相同 SFT config。

CPT 只回答：

- 数据/packing/mask 管线是否正确；
- checkpoint 从 CPT 到 SFT 是否可迁移；
- 短程是否出现遗忘/数值问题；
- 系统成本。

不能因少量 tokens 宣称“获得金融知识”。

多卡选择：

- 如果单卡 0.6B fit，先用 DDP 练 throughput；
- 若 optimizer/model state 压力大，用第 08 周 FSDP；
- 1/2/4/8 只做固定 global batch correctness/短 profile；
- 不为用 8 卡扩大 global batch 污染 SFT 对照。

记录：

- target tokens seen；
- attempted/successful updates；
- train/dev loss；
- valid JSON/program/execution/answer/evidence；
- scale/skips/grad；
- throughput/peak memory/communication；
- checkpoint selection。

### 周五：base、SFT、CPT→SFT 的独立评测与 error waterfall

目标：判断收益来自哪里。

固定三个模型：

    base
    SFT
    short CPT → SFT

固定 generation：

- greedy；
- same prompt template；
- same max_new_tokens；
- same tokenizer/revision；
- same test split；
- no tool feedback during generation；
- executor 只在生成后评测。

结果：

| model | valid JSON | valid program | exec acc | answer acc | evidence F1 | scale acc |
|---|---:|---:|---:|---:|---:|---:|
| base | | | | | | |
| SFT | | | | | | |
| CPT→SFT | | | | | | |

错误 waterfall：

| stage | base failures | SFT failures | CPT→SFT failures |
|---|---:|---:|---:|
| parse | | | |
| schema | | | |
| evidence | | | |
| program parse | | | |
| execution | | | |
| scale | | | |
| answer | | | |

抽样 30–50 条：

- wrong evidence；
- correct evidence/wrong op；
- correct op/wrong operands；
- scale；
- arithmetic；
- hallucinated ID；
- format；
- truncation；
- ambiguous gold。

有效结论：

- SFT pipeline PASS：gold evaluator 100%，valid JSON ≥90% 目标，执行指标可重复；
- CPT value positive：最终 SFT 核心指标提升至少约 2 个百分点且无明显通用/格式回归；
- CPT neutral/negative：如实报告；
- INCONCLUSIVE：训练步数/评测样本太少。

## 7. 本周硬验收

| 项 | PASS |
|---|---|
| evaluator | 清洗后的 gold programs 100% parse/execute/match |
| security | 不用 Python eval/exec 执行模型文本 |
| schema | evidence/program/answer/scale 可独立检查 |
| data | official split；无 dev/test 进入 train/CPT |
| mask | prompt/padding ignore；target/EOS 计 loss |
| packing | 边界隔离正确；否则关闭 |
| overfit | 128 样本生成指标显著提升 |
| FP16/resume | 无 NaN；skipped <1%；fixed-batch resume <1% |
| SFT | valid JSON 目标 ≥90%，或有清晰失败归因 |
| CPT | 只作 rehearsal；不夸大短 token 结果 |
| evaluation | base/SFT/CPT→SFT 同生成设置与 waterfall |

## 8. 常见故障

### evaluator 不是 100%

不训练。先查：

- operator mapping；
- scale；
- table normalization；
- rounding；
- gold 异常；
- nested parse；
- operand index。

### train loss 降、execution 不升

- teacher forcing 与 autoregressive exposure；
- 格式 token 占比过大；
- evidence/program 权重；
- truncation；
- answer 泄漏；
- evaluator/program DSL mismatch；
- 训练只学 JSON 模板。

### packing 后异常

- 样本 attention 泄漏；
- labels crossing boundary；
- EOS/position IDs；
- loss 按 token 权重；
- 先关闭 packing 建 reference。

### CPT 后 SFT 变差

- CPT LR 太高；
- tokens 太少且分布窄；
- optimizer state 是否重置；
- tokenizer/embedding；
- forgetting；
- SFT config 是否真正相同。

## 9. 闭卷口试

1. 为什么程序执行正确率比字符串 exact match 更合理？
2. reported answer 与 executed answer 为什么要分别检查？
3. prompt mask 错误会造成什么伪提升？
4. packing 如何引入跨样本泄漏？
5. gold evaluator 100% 为什么必须先于训练？
6. SFT checkpoint 应按哪个 dev 指标选择？
7. CPT 与 SFT 的 objective/数据分别是什么？
8. 0.5M–2M token CPT 为什么不能证明知识提升？
9. 为什么不能用交易 PnL 作本周 reward？
10. 什么结果能说明模型只学会格式、没学会推理？
