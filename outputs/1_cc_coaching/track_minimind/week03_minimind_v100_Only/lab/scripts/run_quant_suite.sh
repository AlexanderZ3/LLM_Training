#!/usr/bin/env bash
# 唯一职责：选对解释器，调一次 run_quant_suite.py。别往这里加逻辑。
#
# 为什么这么薄（CLAUDE.md 3.2）：重试、循环、日志、编码、目录创建全部在 Python 里。
# shell 里包原生命令会引入平台特有的坑（PowerShell 5.1 把原生命令的 stderr 包成
# ErrorRecord，配合 ErrorActionPreference=Stop 曾经中断过一次 27 GB 的下载）。
# 这里连日志重定向都不做——run_quant_suite.py 自己写 UTF-8 的 JSON/JSONL。
#
# 用法：
#   bash lab/scripts/run_quant_suite.sh --dry-run
#   bash lab/scripts/run_quant_suite.sh --config lab/configs/quant_v100.json --device cuda
#   MM_PYTHON=/path/to/python bash lab/scripts/run_quant_suite.sh --experiments q3,q5
#
# 参数原样透传给 Python，用 --help 看全部选项。

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${MM_PYTHON:-python}"

"$PY" "${SCRIPT_DIR}/run_quant_suite.py" "$@"
status=$?

if [ "$status" -ne 0 ]; then
  echo "run_quant_suite.py 退出码 ${status}（0=全部成功，1=有实验失败，2=用法/路径错误）" >&2
fi
exit "$status"
