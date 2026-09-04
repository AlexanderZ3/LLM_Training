#!/usr/bin/env bash
# 命令等级：模板（先替换 OUT_DIR / MINIMIND_ROOT；需要 8 张 V100，CPU 上跑不了 FSDP）
#
# Day 2：同 global batch 下 DDP 与 FSDP 三种策略的对照。
# 看四个数：step_time_s、peak_mem_mb、collectives_total、loss 是否与 DDP 一致。
# 预期（估算，非实测）：64M 模型上 FULL_SHARD 的 step time 高于 DDP，
# 因为每层 all_gather 的延迟无法被这么小的计算量掩盖；显存则应低于 DDP。
#
# 预算：4 次运行 × MAX_STEPS=20，估计合计 8-12 分钟；超时先把 MAX_STEPS 降到 10。
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${LAB_DIR}/src:${PYTHONPATH:-}"

: "${MINIMIND_ROOT:?请先 export MINIMIND_ROOT=<MiniMind 仓库绝对路径>}"
OUT_DIR="${OUT_DIR:-${LAB_DIR}/runs/fsdp_vs_ddp}"   # 公司环境请改成公司内部路径
CONFIG="${CONFIG:-${LAB_DIR}/configs/v100_dense.json}"
NPROC="${NPROC:-8}"
GLOBAL_BATCH="${GLOBAL_BATCH:-64}"
MAX_STEPS="${MAX_STEPS:-20}"
DTYPE="${DTYPE:-float16}"          # V100 没有 bf16；改成 bfloat16 会被显式拒绝
AC="${AC:-0}"                      # 1 = 打开 activation checkpointing
NCCL_TIMEOUT_S="${NCCL_TIMEOUT_S:-600}"
PY="${PY:-python}"

export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

mkdir -p "${OUT_DIR}"
echo "[run_fsdp_vs_ddp] nproc=${NPROC} G=${GLOBAL_BATCH} steps=${MAX_STEPS} dtype=${DTYPE} ac=${AC}"

echo "--- 基线：DDP"
torchrun --nproc_per_node "${NPROC}" -m mm_dist.train_ddp \
  --config "${CONFIG}" --out-dir "${OUT_DIR}" --dtype "${DTYPE}" \
  --global-batch "${GLOBAL_BATCH}" --accum 1 --max-steps "${MAX_STEPS}" \
  --nccl-timeout-s "${NCCL_TIMEOUT_S}" --log-name "ddp_ws${NPROC}"

for STRATEGY in full_shard shard_grad_op no_shard; do
  echo "--- FSDP ${STRATEGY}"
  torchrun --nproc_per_node "${NPROC}" -m mm_dist.train_fsdp \
    --config "${CONFIG}" --out-dir "${OUT_DIR}" --dtype "${DTYPE}" \
    --sharding "${STRATEGY}" --activation-checkpointing "${AC}" \
    --global-batch "${GLOBAL_BATCH}" --accum 1 --max-steps "${MAX_STEPS}" \
    --nccl-timeout-s "${NCCL_TIMEOUT_S}" --log-name "fsdp_${STRATEGY}_ws${NPROC}"
done

echo "--- loss 一致性：DDP vs FULL_SHARD（fp16 下容差放宽到 1e-2）"
"${PY}" -m mm_dist.train_ddp --compare-json \
  "${OUT_DIR}/ddp_ws${NPROC}_rank0.jsonl" \
  "${OUT_DIR}/fsdp_full_shard_ws${NPROC}_rank0.jsonl" --compare-tol 1e-2 || true

echo "[run_fsdp_vs_ddp] 证据字段（只带出比值，不带出绝对数）："
echo "  step_time_ratio = fsdp.step_time_s / ddp.step_time_s"
echo "  mem_ratio       = fsdp.peak_mem_mb / ddp.peak_mem_mb"
echo "  collectives_total（DDP 与三种 FSDP 各一个整数）、fsdp_units（应为层数+1）"
