#!/usr/bin/env bash
# 命令等级：模板（先替换 OUT_DIR / MINIMIND_ROOT）
#
# Day 3：EP=1/2/4/8 扫描。EP=8 需要 num_experts=8（约束 ep_size<=E 且整除），
# 脚本用 --num-experts 覆盖 config，不改 config 文件本身。
#
# 先跑一次 equiv（单进程、无通信）确认六步流水在数值上等于原始 MOEFeedForward，
# 再跑各 EP 规模的 --time-breakdown。
# 预算：equiv < 30 秒；每个 EP 规模 MAX_STEPS=10 估计 1-2 分钟，合计 < 10 分钟。
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${LAB_DIR}/src:${PYTHONPATH:-}"

: "${MINIMIND_ROOT:?请先 export MINIMIND_ROOT=<MiniMind 仓库绝对路径>}"
OUT_DIR="${OUT_DIR:-${LAB_DIR}/runs/ep_scan}"      # 公司环境请改成公司内部路径
CONFIG="${CONFIG:-${LAB_DIR}/configs/v100_moe.json}"
NPROC="${NPROC:-8}"
NUM_EXPERTS="${NUM_EXPERTS:-8}"                    # EP=8 需要 8 个专家
BATCH="${BATCH:-8}"
SEQ_LEN="${SEQ_LEN:-512}"
MAX_STEPS="${MAX_STEPS:-10}"
DTYPE="${DTYPE:-float16}"
CAPACITY_FACTOR="${CAPACITY_FACTOR:-0}"            # 0 = dropless
NCCL_TIMEOUT_S="${NCCL_TIMEOUT_S:-600}"
PY="${PY:-python}"

export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

mkdir -p "${OUT_DIR}"
echo "[run_ep_scan] nproc=${NPROC} E=${NUM_EXPERTS} batch=${BATCH} seq=${SEQ_LEN} steps=${MAX_STEPS}"

echo "--- 门 1：EP=1 数值等价（单进程，无集合通信）"
"${PY}" -m mm_dist.ep_moe equiv \
  --config "${CONFIG}" --out-dir "${OUT_DIR}" --num-experts "${NUM_EXPERTS}" \
  --batch 2 --seq-len 64 --tol 1e-5 --force-cpu --dtype float32

for EP in 1 2 4 8; do
  if [ "${EP}" -gt "${NUM_EXPERTS}" ]; then
    echo "--- 跳过 EP=${EP}：ep_size 不能大于 num_experts=${NUM_EXPERTS}"
    continue
  fi
  echo "--- EP=${EP}"
  torchrun --nproc_per_node "${NPROC}" -m mm_dist.ep_moe run \
    --config "${CONFIG}" --out-dir "${OUT_DIR}" --dtype "${DTYPE}" \
    --ep-size "${EP}" --num-experts "${NUM_EXPERTS}" \
    --capacity-factor "${CAPACITY_FACTOR}" --time-breakdown \
    --batch "${BATCH}" --seq-len "${SEQ_LEN}" --max-steps "${MAX_STEPS}" \
    --nccl-timeout-s "${NCCL_TIMEOUT_S}" --log-name "ep${EP}"
done

echo "[run_ep_scan] 证据字段（抽象值）："
echo "  equiv 的 max_abs_diff 与 pass"
echo "  每个 EP：time_ms 里 dispatch_a2a / combine_a2a / expert_compute 的占比"
echo "  wait_before_dispatch 在各 rank 之间的极差（木桶效应）"
echo "  recv_tokens 的最大/最小比值（负载不均程度）"
