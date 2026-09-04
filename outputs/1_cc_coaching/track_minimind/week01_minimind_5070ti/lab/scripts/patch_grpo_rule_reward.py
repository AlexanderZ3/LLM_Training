"""生成 `$MINIMIND_ROOT/trainer/train_grpo_rule.py`：把 1.8B reward 模型换成 lab 的规则 reward（16 GB 放不下时的替代）。

不修改原 train_grpo.py；只复制一份并替换一行（commit 7a6fddd 第 279 行）：
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
→   reward_model = RuleRewardModel(...)   # 来自 lab/src/mm_probe/rule_reward.py
幂等：重复运行结果相同。用法：python scripts/patch_grpo_rule_reward.py [--minimind-root DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "src"))

from mm_probe.minimind_env import find_minimind_root  # noqa: E402

ORIGINAL_LINE = "    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="生成 train_grpo_rule.py（规则 reward 版 GRPO）")
    p.add_argument("--minimind-root", default=None)
    args = p.parse_args(argv)
    root = find_minimind_root(args.minimind_root)
    src = root / "trainer" / "train_grpo.py"
    dst = root / "trainer" / "train_grpo_rule.py"
    text = src.read_text(encoding="utf-8")
    if ORIGINAL_LINE not in text:
        print(f"未在 {src} 找到预期的 reward_model 实例化行；MiniMind commit 可能不是锁定值。", file=sys.stderr)
        return 1
    lab_src = str(LAB / "src").replace("\\", "/")
    replacement = (
        f"    sys.path.insert(0, {lab_src!r})\n"
        "    from mm_probe.rule_reward import RuleRewardModel  # lab 规则 reward，替代 1.8B reward 模型\n"
        "    reward_model = RuleRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)\n"
    )
    new_text = text.replace(ORIGINAL_LINE, replacement)
    header = "# 由 lab/scripts/patch_grpo_rule_reward.py 生成；与 train_grpo.py 唯一区别是 reward_model 换成规则 reward。\n"
    dst.write_text(header + new_text, encoding="utf-8")
    print(f"written {dst}")
    print("运行：cd $MINIMIND_ROOT/trainer && python train_grpo_rule.py --batch_size 1 --num_generations 2 --max_seq_len 256 --max_gen_len 256 --loss_type grpo --debug_mode --debug_interval 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
