#!/usr/bin/env bash
# Day 1：数据契约 + fp32/fp16 误差账
#
# 这个文件只做一件事：用对解释器调一次 lab/scripts/run_day.py。
# 顺序、失败停止、日志、编码全部在那个 Python 里（见 CLAUDE.md 3.2）。
# 这里不设 -e、不写循环、不对原生命令用 2>&&1。
#
# 首次试跑（每个训练步骤只跑 2 步）：
#   MAX_STEPS=2 bash lab/scripts/run_day1_numeric.sh
#
# 换解释器：
#   MM_PYTHON=/path/to/conda/envs/xxx/bin/python bash lab/scripts/run_day1_numeric.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${MM_PYTHON:-}"
if [ -z "$PY" ]; then
  if command -v python3 >/dev/null 2>/dev/null; then PY="python3"; else PY="python"; fi
fi

# 用 v100_768_accum 而不是 v100_768：两者全局批都是 64，但 accum=4 让 micro=16。
# 误差账那一步要在同一张卡上同时驻留 fp32 参考模型与 fp16 被测模型，
# micro=64 时 fp32 那一路的激活按估算约 16 GB，单卡放不下。
"$PY" "$HERE/run_day.py" --day 1 --config v100_768_accum --device cuda:0 --dtype float16 "$@"
RC=$?

# 显式检查退出码，不依赖 set -e。
if [ "$RC" -ne 0 ]; then
  echo "[run_day1] 失败，退出码 $RC。上面最后一个 rc!=0 的步骤就是失败点。"
fi
exit "$RC"
