#!/usr/bin/env bash
# 命令等级：模板（先替换 OUT_DIR / MINIMIND_ROOT）
#
# Day 4：热点专家（router bias）× aux loss 系数 的二维扫描，输出一张 CSV。
# 自变量：--bias-list（给 --expert-id 那个专家的 logit 加常数）
#         --aux-coef-list（覆盖 config 的 router_aux_loss_coef）
# 因变量：max_load_ratio、router_entropy、mean_logits_loss、mean_wait_ms
#
# 预期读法（估算）：bias 增大 -> max_load_ratio 上升、entropy 下降、wait_ms 分化；
# aux_coef 增大 -> max_load_ratio 回落到 ~1，但 mean_logits_loss 可能变差（第 7 条失败模式）。
#
# 预算：组合数 × MAX_STEPS。默认 5×4=20 组 × 10 步，估计 6-10 分钟；
# 超预算先把 bias-list 砍到 0,1,4。
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${LAB_DIR}/src:${PYTHONPATH:-}"

: "${MINIMIND_ROOT:?请先 export MINIMIND_ROOT=<MiniMind 仓库绝对路径>}"
OUT_DIR="${OUT_DIR:-${LAB_DIR}/runs/moe_sweep}"     # 公司环境请改成公司内部路径
CONFIG="${CONFIG:-${LAB_DIR}/configs/v100_moe.json}"
NPROC="${NPROC:-8}"
BIAS_LIST="${BIAS_LIST:-0,0.5,1,2,4}"
AUX_COEF_LIST="${AUX_COEF_LIST:-0,0.001,0.01,0.1}"
EXPERT_ID="${EXPERT_ID:-0}"
GLOBAL_BATCH="${GLOBAL_BATCH:-64}"
MAX_STEPS="${MAX_STEPS:-10}"
DTYPE="${DTYPE:-float16}"
NCCL_TIMEOUT_S="${NCCL_TIMEOUT_S:-600}"

export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

mkdir -p "${OUT_DIR}"
echo "[run_moe_sweep] bias=${BIAS_LIST} aux=${AUX_COEF_LIST} expert=${EXPERT_ID} steps=${MAX_STEPS}"

torchrun --nproc_per_node "${NPROC}" -m mm_dist.router_stats sweep \
  --config "${CONFIG}" --out-dir "${OUT_DIR}" --dtype "${DTYPE}" \
  --expert-id "${EXPERT_ID}" \
  --bias-list "${BIAS_LIST}" --aux-coef-list "${AUX_COEF_LIST}" \
  --global-batch "${GLOBAL_BATCH}" --accum 1 --max-steps "${MAX_STEPS}" \
  --csv "${OUT_DIR}/moe_sweep.csv" \
  --nccl-timeout-s "${NCCL_TIMEOUT_S}"

echo "--- 单点复现（bias 最大的那一档，写逐步 JSONL 便于看曲线）"
LAST_BIAS="${BIAS_LIST##*,}"
torchrun --nproc_per_node "${NPROC}" -m mm_dist.router_stats stats \
  --config "${CONFIG}" --out-dir "${OUT_DIR}" --dtype "${DTYPE}" \
  --expert-id "${EXPERT_ID}" --bias "${LAST_BIAS}" --aux-coef 0.0005 \
  --global-batch "${GLOBAL_BATCH}" --accum 1 --max-steps "${MAX_STEPS}" \
  --log-name "hot_expert" --nccl-timeout-s "${NCCL_TIMEOUT_S}"

echo "[run_moe_sweep] 证据字段（抽象值）：CSV 里的 bias、aux_coef、max_load_ratio、"
echo "  router_entropy、mean_logits_loss、mean_wait_ms；不带出 CSV 原文与日志。"
