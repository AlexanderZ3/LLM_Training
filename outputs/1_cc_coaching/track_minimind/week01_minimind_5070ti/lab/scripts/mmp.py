"""mm_probe 统一入口：`python scripts/mmp.py <module> [args...]`，免设 PYTHONPATH。

<module> ∈ inspect_dataset | bounded_train | parse_log | mask_fault | dpo_check | grpo_stats | eval_generate
等价于 `PYTHONPATH=lab/src python -m mm_probe.<module> [args...]`。
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

MODULES = ("inspect_dataset", "bounded_train", "parse_log", "mask_fault", "dpo_check", "grpo_stats", "eval_generate")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in MODULES:
        print(__doc__)
        print("modules:", ", ".join(MODULES))
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    mod = importlib.import_module(f"mm_probe.{argv[0]}")
    return int(mod.main(argv[1:]) or 0)


if __name__ == "__main__":
    sys.exit(main())
