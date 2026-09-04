#!/usr/bin/env bash
# 命令等级：模板（先替换 OUT_DIR / MINIMIND_ROOT）
#
# Day 5 故障演练：训练到第 AFTER_STEP 步时 kill 掉 KILL_RANK，观察其余 rank 的行为。
#
# 跑两遍，只差一个环境变量：
#   A) ASYNC=1（默认）：watchdog 在 NCCL_TIMEOUT_S 秒后让其余 rank 崩溃，torchrun 退出非 0
#   B) ASYNC=0        ：其余 rank 静默阻塞到进程组 timeout
# 计时是这个练习的主要观测：从被杀那一刻到 torchrun 退出的墙钟时间。
#
# 注意：NCCL_TIMEOUT_S 默认压到 60 秒，就是为了让 B) 也能在 10 分钟预算内结束。
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${LAB_DIR}/src:${PYTHONPATH:-}"

: "${MINIMIND_ROOT:?请先 export MINIMIND_ROOT=<MiniMind 仓库绝对路径>}"
OUT_DIR="${OUT_DIR:-${LAB_DIR}/runs/fault}"        # 公司环境请改成公司内部路径
CONFIG="${CONFIG:-${LAB_DIR}/configs/v100_dense.json}"
NPROC="${NPROC:-8}"
KILL_RANK="${KILL_RANK:-3}"
AFTER_STEP="${AFTER_STEP:-5}"
MAX_STEPS="${MAX_STEPS:-20}"
GLOBAL_BATCH="${GLOBAL_BATCH:-64}"
DTYPE="${DTYPE:-float16}"
NCCL_TIMEOUT_S="${NCCL_TIMEOUT_S:-60}"
PY="${PY:-python}"

mkdir -p "${OUT_DIR}"
echo "[fault_kill_rank] 先看一眼当前的 NCCL 相关环境变量与建议值"
"${PY}" -m mm_dist.faults env --nccl-timeout-s "${NCCL_TIMEOUT_S}"

run_case () {  # $1 = async_error_handling (0/1)
  echo ""
  echo "=== 用例：TORCH_NCCL_ASYNC_ERROR_HANDLING=$1 ==="
  local t0 t1
  t0="$(date +%s)"
  set +e
  TORCH_NCCL_ASYNC_ERROR_HANDLING="$1" NCCL_ASYNC_ERROR_HANDLING="$1" \
  torchrun --nproc_per_node "${NPROC}" -m mm_dist.faults demo \
    --config "${CONFIG}" --out-dir "${OUT_DIR}" --dtype "${DTYPE}" \
    --global-batch "${GLOBAL_BATCH}" --accum 1 --max-steps "${MAX_STEPS}" \
    --kill-rank "${KILL_RANK}" --after-step "${AFTER_STEP}" \
    --async-error-handling "$1" --overwrite-env 1 \
    --nccl-timeout-s "${NCCL_TIMEOUT_S}" \
    --log-name "fault_async$1"
  local rc=$?
  set -e
  t1="$(date +%s)"
  echo "[fault_kill_rank] async=$1 torchrun 退出码=${rc} 墙钟=$((t1 - t0))s"
}

run_case 1
run_case 0

echo ""
echo "[fault_kill_rank] 证据字段（抽象值）："
echo "  killed_at_step（应等于 AFTER_STEP=${AFTER_STEP}）"
echo "  两个用例的 torchrun 退出码"
echo "  两个用例从被杀到退出的墙钟秒数之比"
echo "  存活 rank 的日志最后一条 step（说明它们停在哪一步）"
echo "本脚本不写 checkpoint。恢复回归请单独跑 run_reshard.sh（它自己保存再以 LOAD_NPROC 卡加载并 verify）。"
