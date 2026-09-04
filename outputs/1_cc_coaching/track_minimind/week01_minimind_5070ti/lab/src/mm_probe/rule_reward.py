"""规则 reward：当 16 GB 放不下 internlm2-1_8b-reward 时的替代，接口与 MiniMind `LMForRewardModel` 相同。

`RuleRewardModel(model_path, device, dtype)` 参数只为保持签名兼容，全部忽略。
`get_score(messages, response)` 返回 [-3, 3] 的确定性分数：
  +1.0  回答长度在 [20, 800] 字符（与 train_grpo.py 里的长度奖励同区间）
  +1.0  回答与最后一条 user 问题的字符 2-gram 有重叠（粗略"答非所问"检测）
  -1.0  回答里 3-gram 重复率 > 20%
  -1.0  回答为空或只有空白
分数是规则给的，只能用来跑通 GRPO 管线与观察 advantage 统计，不能当作模型质量信号。
"""
from __future__ import annotations

import re
from typing import Dict, List


def _ngrams(s: str, n: int) -> List[str]:
    s = re.sub(r"\s+", "", s)
    return [s[i:i + n] for i in range(max(len(s) - n + 1, 0))]


class RuleRewardModel:
    def __init__(self, model_path: str = None, device: str = "cpu", dtype=None):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype

    def get_score(self, messages: List[Dict[str, str]], response: str) -> float:
        resp = response or ""
        if not resp.strip():
            return -1.0
        score = 0.0
        n = len(resp.strip())
        score += 1.0 if 20 <= n <= 800 else 0.0
        question = messages[-1]["content"] if messages else ""
        q2 = set(_ngrams(question, 2))
        r2 = set(_ngrams(resp, 2))
        if q2 and r2 and (q2 & r2):
            score += 1.0
        g3 = _ngrams(resp, 3)
        if g3:
            rep = 1.0 - len(set(g3)) / len(g3)
            if rep > 0.2:
                score -= 1.0
        return max(min(score, 3.0), -3.0)
