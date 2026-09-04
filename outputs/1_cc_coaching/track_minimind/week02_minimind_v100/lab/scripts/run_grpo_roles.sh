#!/usr/bin/env bash
# 命令等级：模板（先替换 OUT_DIR / MINIMIND_ROOT）
#
# Day 5：GRPO 三种角色放置的对照。
#   replicate    每卡全量（policy+rollout+reward），DDP 包 policy
#   policy_fsdp  policy 分片到 8 卡，rollout/reward 仍每卡一份（需要 GPU）
#   split_roles  4 卡 policy / 2 卡 rollout / 2 卡 reward，每轮 broadcast 同步权重
#
# 看三个数：peak_mem_mb、step_time_s、role_idle_ms。
# 权重同步正确性看 param_checksum：split_roles 下 broadcast 之后所有 rank 必须相同。
#
# 预算：3 种放置 × MAX_STEPS=5 × (生成 64 token)，估计 6-10 分钟。
# 时间不够时优先保 policy_fsdp（Gate 依赖它），split_roles 允许 INCONCLUSIVE。
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${LAB_DIR}/src:${PYTHONPATH:-}"

: "${MINIMIND_ROOT:?请先 export MINIMIND_ROOT=<MiniMind 仓库绝对路径>}"
OUT_DIR="${OUT_DIR:-${LAB_DIR}/runs/grpo_roles}"    # 公司环境请改成公司内部路径
CONFIG="${CONFIG:-${LAB_DIR}/configs/v100_dense.json}"
NPROC="${NPROC:-8}"
N_POLICY="${N_POLICY:-4}"
N_ROLLOUT="${N_ROLLOUT:-2}"
N_REWARD="${N_REWARD:-2}"
N_PROMPTS="${N_PROMPTS:-4}"
GROUP_SIZE="${GROUP_SIZE:-4}"
PROMPT_LEN="${PROMPT_LEN:-32}"
MAX_GEN_LEN="${MAX_GEN_LEN:-64}"
MAX_STEPS="${MAX_STEPS:-5}"
DTYPE="${DTYPE:-float16}"
REWARD="${REWARD:-rule}"            # rule | model
REWARD_PATH="${REWARD_PATH:-}"      # --reward model 时指向公司内部已有的 HF 目录
NCCL_TIMEOUT_S="${NCCL_TIMEOUT_S:-600}"

export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

REWARD_ARGS=(--reward "${REWARD}")
if [ "${REWARD}" = "model" ]; then
  : "${REWARD_PATH:?--reward model 需要 REWARD_PATH=<公司内部 HF 模型目录>}"
  REWARD_ARGS+=(--reward-path "${REWARD_PATH}")
fi

mkdir -p "${OUT_DIR}"
echo "[run_grpo_roles] nproc=${NPROC} 角色=${N_POLICY}/${N_ROLLOUT}/${N_REWARD} "\
"prompts=${N_PROMPTS}x${GROUP_SIZE} gen=${MAX_GEN_LEN} steps=${MAX_STEPS}"

for PLACEMENT in replicate policy_fsdp split_roles; do
  echo "--- placement=${PLACEMENT}"
  torchrun --nproc_per_node "${NPROC}" -m mm_dist.grpo_roles \
    --config "${CONFIG}" --out-dir "${OUT_DIR}" --dtype "${DTYPE}" \
    --placement "${PLACEMENT}" \
    --n-policy "${N_POLICY}" --n-rollout "${N_ROLLOUT}" --n-reward "${N_REWARD}" \
    --n-prompts "${N_PROMPTS}" --group-size "${GROUP_SIZE}" \
    --prompt-len "${PROMPT_LEN}" --max-gen-len "${MAX_GEN_LEN}" \
    --max-steps "${MAX_STEPS}" "${REWARD_ARGS[@]}" \
    --nccl-timeout-s "${NCCL_TIMEOUT_S}" --log-name "grpo_${PLACEMENT}"
done

echo "[run_grpo_roles] 证据字段（抽象值）："
echo "  mem_ratio_fsdp_vs_replicate = policy_fsdp.peak_mem_mb / replicate.peak_mem_mb"
echo "  step_time_ratio（三种放置两两比值）"
echo "  split_roles 各角色的 role_idle_ms 占 step_time 的比例"
echo "  broadcast 后各 rank 的 param_checksum 是否一致（布尔值）"
