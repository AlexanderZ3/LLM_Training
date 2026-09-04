#!/usr/bin/env bash
# 命令等级：模板（先替换 OUT_DIR / CKPT_DIR / MINIMIND_ROOT；需要 GPU，CPU 上 FSDP 跑不了）
#
# Day 2 下半场：8 卡训练 → 保存分片 checkpoint → 4 卡恢复 → 验证 loss 连续。
# 判定：verify 输出的 abs_delta < --tol（fp16 建议 1e-2，fp32 建议 1e-4），
# 并且 converted_step == step * saved_ws / current_ws。
#
# 预算：save 约 MAX_STEPS 步 + verify 一次前向，估计 3-6 分钟。
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${LAB_DIR}/src:${PYTHONPATH:-}"

: "${MINIMIND_ROOT:?请先 export MINIMIND_ROOT=<MiniMind 仓库绝对路径>}"
OUT_DIR="${OUT_DIR:-${LAB_DIR}/runs/reshard}"        # 公司环境请改成公司内部路径
CKPT_DIR="${CKPT_DIR:-${OUT_DIR}/ckpt_sharded}"
CONFIG="${CONFIG:-${LAB_DIR}/configs/v100_dense.json}"
SAVE_NPROC="${SAVE_NPROC:-8}"
LOAD_NPROC="${LOAD_NPROC:-4}"
GLOBAL_BATCH="${GLOBAL_BATCH:-64}"
MAX_STEPS="${MAX_STEPS:-20}"
DTYPE="${DTYPE:-float16}"
FORMAT="${FORMAT:-sharded}"          # sharded | full（full 是 rank0_only 回退）
TOL="${TOL:-1e-2}"                   # fp16 用 1e-2；把 DTYPE 换成 float32 时用 1e-4
NCCL_TIMEOUT_S="${NCCL_TIMEOUT_S:-600}"

export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

mkdir -p "${OUT_DIR}"
echo "[run_reshard] ${SAVE_NPROC} 卡保存 -> ${LOAD_NPROC} 卡恢复  format=${FORMAT} tol=${TOL}"

echo "--- 保存（world_size=${SAVE_NPROC}）"
torchrun --nproc_per_node "${SAVE_NPROC}" -m mm_dist.ckpt_reshard save \
  --config "${CONFIG}" --ckpt-dir "${CKPT_DIR}" --out-dir "${OUT_DIR}" \
  --dtype "${DTYPE}" --format "${FORMAT}" --sharding full_shard \
  --global-batch "${GLOBAL_BATCH}" --accum 1 --max-steps "${MAX_STEPS}" \
  --nccl-timeout-s "${NCCL_TIMEOUT_S}"

echo "--- 恢复并验证（world_size=${LOAD_NPROC}）"
torchrun --nproc_per_node "${LOAD_NPROC}" -m mm_dist.ckpt_reshard verify \
  --config "${CONFIG}" --ckpt-dir "${CKPT_DIR}" --out-dir "${OUT_DIR}" \
  --dtype "${DTYPE}" --format "${FORMAT}" --sharding full_shard \
  --global-batch "${GLOBAL_BATCH}" --accum $((SAVE_NPROC / LOAD_NPROC)) \
  --max-steps "${MAX_STEPS}" --tol "${TOL}" \
  --result-json "${OUT_DIR}/verify_${SAVE_NPROC}to${LOAD_NPROC}.json" \
  --nccl-timeout-s "${NCCL_TIMEOUT_S}"

echo "[run_reshard] 证据字段：pass、abs_delta、tol、saved_world_size、current_world_size"
echo "              （只带出这五个抽象值，checkpoint 与日志留在公司内）"
