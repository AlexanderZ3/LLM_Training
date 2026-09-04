#!/usr/bin/env bash
# 命令等级：可直接执行（改 PYTHON 与 OUT 即可）
# 作用：整夜无人值守下载 MiniMind 全量数据集 + 奖励模型；断线自动重试，可随时 Ctrl-C 后重跑续传。
# 用法：
#   bash lab/scripts/download_all.sh
#   TIER=mini bash lab/scripts/download_all.sh
#   HF_ENDPOINT=https://hf-mirror.com bash lab/scripts/download_all.sh
#   nohup bash lab/scripts/download_all.sh > /dev/null 2>&1 &     # 后台整夜跑
set -uo pipefail

TIER="${TIER:-all}"
PYTHON="${PYTHON:-python}"
OUTER_ROUNDS="${OUTER_ROUNDS:-20}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-60}"
NO_HASH="${NO_HASH:-0}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEEK_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${OUT:-$WEEK_DIR/datasets}"
DL="$SCRIPT_DIR/download_datasets.py"

command -v "$PYTHON" >/dev/null 2>&1 || { echo "找不到解释器：$PYTHON（用 PYTHON= 指定）"; exit 1; }
[[ -f "$DL" ]] || { echo "找不到 download_datasets.py：$DL"; exit 1; }

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

ARGS=("$DL" --tier "$TIER" --out "$OUT" --retries 6 --rounds 2)
[[ "$NO_HASH" == "1" ]] && ARGS+=(--no-hash)

if [[ "$DRY_RUN" == "1" ]]; then
  "$PYTHON" "${ARGS[@]}" --dry-run
  exit $?
fi

mkdir -p "$OUT"
RUN_LOG="$OUT/download_run_$(date +%Y%m%d_%H%M%S).log"
echo "[download_all] tier=$TIER"
echo "[download_all] out=$OUT"
echo "[download_all] 本轮日志=$RUN_LOG"
echo "[download_all] 累积日志=$OUT/download.log"
echo "[download_all] 最多 $OUTER_ROUNDS 轮，每轮间隔 ${SLEEP_BETWEEN}s。Ctrl-C 可随时停止，重跑本脚本续传。"
echo

exit_code=1
for ((i = 1; i <= OUTER_ROUNDS; i++)); do
  echo "===== 外层第 $i/$OUTER_ROUNDS 轮  $(date +%H:%M:%S) ====="
  echo "===== 外层第 $i/$OUTER_ROUNDS 轮  $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$RUN_LOG"
  "$PYTHON" "${ARGS[@]}" 2>&1 | tee -a "$RUN_LOG"
  exit_code="${PIPESTATUS[0]}"
  if [[ "$exit_code" -eq 0 ]]; then
    echo
    echo "[download_all] 全部完成（外层第 $i 轮）。"
    break
  fi
  if [[ "$exit_code" -eq 2 ]]; then
    echo "[download_all] 磁盘空间不足，停止。换 OUT= 到别的盘，或用 TIER=mini。"
    break
  fi
  if [[ "$i" -lt "$OUTER_ROUNDS" ]]; then
    echo "[download_all] 本轮仍有未完成项，${SLEEP_BETWEEN}s 后再来一轮…"
    sleep "$SLEEP_BETWEEN"
  fi
done

echo
if [[ "$exit_code" -eq 0 ]]; then
  echo "[download_all] 校验清单：$OUT/DOWNLOAD_MANIFEST.json"
  [[ "$NO_HASH" == "1" ]] || echo "[download_all] 哈希清单：$OUT/SHA256SUMS.txt"
else
  echo "[download_all] 退出码 $exit_code —— 还有文件未就绪。重跑本脚本即可继续（已完成的会跳过）。"
fi
exit "$exit_code"
