#!/usr/bin/env bash
# 命令等级：模板（先替换 OUT_DIR / MINIMIND_ROOT，其余可直接跑）
#
# Day 1：fixed-global-batch 等价。同一个 --global-batch，跑 1 / 2 / 8 卡三次，
# 逐步比 loss。判定：max|delta| < 1e-4（float32 等价模式）。
#
# 预算：TINY 配置每次 < 1 分钟；V100 dense 配置按 MAX_STEPS=20 估计每次 2-4 分钟，
# 三次合计 < 10 分钟（估算，以实际为准）。
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${LAB_DIR}/src:${PYTHONPATH:-}"

: "${MINIMIND_ROOT:?请先 export MINIMIND_ROOT=<MiniMind 仓库绝对路径>}"
OUT_DIR="${OUT_DIR:-${LAB_DIR}/runs/ddp_equiv}"   # 公司环境请改成公司内部路径
CONFIG="${CONFIG:-${LAB_DIR}/configs/v100_dense.json}"
GLOBAL_BATCH="${GLOBAL_BATCH:-64}"
MAX_STEPS="${MAX_STEPS:-20}"                       # 上限：每次运行 <= 10 分钟
NCCL_TIMEOUT_S="${NCCL_TIMEOUT_S:-600}"
PY="${PY:-python}"

# 超时后崩溃而不是静默 hang；不覆盖公司已设的值
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

mkdir -p "${OUT_DIR}"
echo "[run_ddp_equiv] out=${OUT_DIR} config=${CONFIG} G=${GLOBAL_BATCH} steps=${MAX_STEPS}"

run_one () {  # $1=nproc  $2=accum
  echo "--- world_size=$1 accum=$2 (micro=$((GLOBAL_BATCH / ($1 * $2))))"
  torchrun --nproc_per_node "$1" -m mm_dist.train_ddp \
    --config "${CONFIG}" --out-dir "${OUT_DIR}" \
    --global-batch "${GLOBAL_BATCH}" --accum "$2" \
    --max-steps "${MAX_STEPS}" --equiv-check \
    --nccl-timeout-s "${NCCL_TIMEOUT_S}" \
    --log-name "equiv_ws$1"
}

run_one 1 8
run_one 2 4
run_one 8 1

echo "--- 比对 ws1 vs ws2"
"${PY}" -m mm_dist.train_ddp --compare-json \
  "${OUT_DIR}/equiv_ws1_rank0.jsonl" "${OUT_DIR}/equiv_ws2_rank0.jsonl" \
  --compare-tol 1e-4
echo "--- 比对 ws1 vs ws8"
"${PY}" -m mm_dist.train_ddp --compare-json \
  "${OUT_DIR}/equiv_ws1_rank0.jsonl" "${OUT_DIR}/equiv_ws8_rank0.jsonl" \
  --compare-tol 1e-4

echo "[run_ddp_equiv] 证据字段：max|delta|、steps_compared、micro_batch（三次必须相同）"
