# Week NN · Day D — <一句话标题>

| 字段 | 值 |
| --- | --- |
| 主要产物 | `<一个文件路径 / 一个测试命令的通过输出 / 一段可复述的解释>` |
| 估时 | `<30–120> 分钟`（步骤估时之和） |
| 环境 | `个人 5070 Ti` / `公司 V100` / `CPU 即可` |
| AI 辅助等级要求 | `A0` / `A1` / `A2` / `A3` |
| 前置 | 前一张卡的证据：`<字段名 = 期望值>` |
| 本卡对应门 | `静态检查` / `dry run` / `overfit` / `FP32 ref` / `FP16` / `resume` / `多卡等价` / `scaling` / `eval` |
| 可裁剪项 | 时间不够时可跳过的步骤编号（主要产物不可裁） |

## 为什么做（三句以内）

<这一步在整周因果链里的位置；不做会导致后面哪一步无法判定。>

## 步骤

### 1. <动作名>（<估时> 分钟）· `可直接执行`

```bash
<确切命令，路径相对 lab/>
```

- 预期：`<输出片段 / 文件出现 / 数值范围（标注估算）>`
- 不对时先查：`<一个检查点：哪个文件、哪个变量、哪条日志>`

### 2. <动作名>（<估时> 分钟）· `模板`

写文件 `lab/src/<pkg>/<file>.py` 中的函数 `<name>`：

- 输入/输出 shape：`<...>`
- 必须满足的不变量：`<...>`
- 测试命令：`pytest lab/tests/test_<x>.py -k <name>`
- 预期：`<N passed>`
- 不对时先查：`<...>`

### 3–7. （同上格式；总步骤 ≤ 7）

## 公司路径差异（仅当本卡涉及 V100/多卡时保留本节）

- 只写与个人路径不同的命令/参数；
- 重申：日志、trace、checkpoint、图片留在公司内，只提交下面证据字段中的抽象值。

## 今日自测（闭卷，2 题，只记录能/不能）

1. <Recall 或 Explain 题，来自本周 04_ORAL_EXAM.md 的 ID>
2. <...>

## 证据字段（填完即可归档到 memory/cc/evidence/）

```json
{
  "date": "YYYY-MM-DD",
  "week": NN,
  "day": D,
  "card": "dayD",
  "skill": "<本卡技能标签>",
  "env": "5070ti | v100 | cpu | h100",
  "ai_level": "A0 | A1 | A2 | A3 | A4",
  "config_id": "<配置文件名或 hash 的非敏感标识>",
  "primary_artifact": "<主要产物路径或测试名>",
  "observations": {
    "<字段1>": "<值>",
    "<字段2>": "<值>"
  },
  "status": "PASS | FAIL-MODEL | FAIL-SYSTEM | INCONCLUSIVE",
  "failure_class": "none | measurement | data_contract | numeric | system | model | insufficient",
  "self_check": {"q1": "能 | 不能", "q2": "能 | 不能"},
  "time_spent_min": 0,
  "notes": "<不含敏感信息的一句话>"
}
```
