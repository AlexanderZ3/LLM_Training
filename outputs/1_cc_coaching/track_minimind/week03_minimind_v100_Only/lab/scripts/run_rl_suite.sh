#!/usr/bin/env bash
# 唯一职责：选对解释器，调一次 run_rl_suite.py。别往这里加逻辑。
#
# 为什么这么薄：见 CLAUDE.md 3.2 与 run_quant_suite.sh 的同段注释。
# 重试、循环、日志、编码、目录创建全部在 Python 里；这里连输出重定向都不做。
#
# 用法：
#   bash lab/scripts/run_rl_suite.sh --dry-run
#   bash lab/scripts/run_rl_suite.sh --config lab/configs/rl_v100.json --device cuda
#   MM_PYTHON=/path/to/python bash lab/scripts/run_rl_suite.sh --experiments r3,r5
#
# 参数原样透传给 Python，用 --help 看全部选项。

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${MM_PYTHON:-python}"

"$PY" "${SCRIPT_DIR}/run_rl_suite.py" "$@"
status=$?

if [ "$status" -ne 0 ]; then
  echo "run_rl_suite.py 退出码 ${status}（0=全部成功，1=有实验失败，2=用法/路径错误）" >&2
fi
exit "$status"
