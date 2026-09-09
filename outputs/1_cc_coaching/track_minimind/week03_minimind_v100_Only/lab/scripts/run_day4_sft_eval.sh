#!/usr/bin/env bash
# Day 4：有界 SFT + 训练前后独立 eval（不看 loss）
#
# 这个文件只做一件事：用对解释器调一次 lab/scripts/run_day.py。
# 顺序、失败停止、日志、编码全部在那个 Python 里（见 CLAUDE.md 3.2）。
# 这里不设 -e、不写循环、不对原生命令用 2>&&1。
#
# 首次试跑（每个训练步骤只跑 2 步）：
#   MAX_STEPS=2 bash lab/scripts/run_day4_sft_eval.sh
#
# 换解释器：
#   MM_PYTHON=/path/to/conda/envs/xxx/bin/python bash lab/scripts/run_day4_sft_eval.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${MM_PYTHON:-}"
if [ -z "$PY" ]; then
  if command -v python3 >/dev/null 2>/dev/null; then PY="python3"; else PY="python"; fi
fi

"$PY" "$HERE/run_day.py" --day 4 --config v100_768 --device cuda:0 --dtype float16 --global-batch 64 "$@"
RC=$?

# 显式检查退出码，不依赖 set -e。
if [ "$RC" -ne 0 ]; then
  echo "[run_day4] 失败，退出码 $RC。上面最后一个 rc!=0 的步骤就是失败点。"
fi
exit "$RC"
