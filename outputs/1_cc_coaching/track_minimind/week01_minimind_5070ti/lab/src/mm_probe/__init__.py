"""mm_probe — 挂在固定 commit 的 MiniMind 之上的仪表包（Week M01）。

所有模块通过 `minimind_env.find_minimind_root()` 定位用户克隆的 MiniMind 仓库
（环境变量 `MINIMIND_ROOT` 或 `--minimind-root`），不复制 MiniMind 源码。

纯函数模块（不需要 MiniMind、不需要 tokenizer）：
    hooks, mask_fault, grpo_stats, parse_log, dpo_check(数学部分), inspect_dataset.generate_labels
依赖 MiniMind 的模块：
    bounded_train, eval_generate, inspect_dataset(CLI), dpo_check(CLI)
"""

__version__ = "0.1.0"

PINNED_MINIMIND_COMMIT = "7a6fddd63a30c06b2fdd5fac4089922b29bc841b"
