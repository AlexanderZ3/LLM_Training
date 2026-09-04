#!/usr/bin/env bash
# 命令等级：模板 —— 需要 MINIMIND_ROOT、MM_PYTHON（缺省 python）、Day 3 的 out/full_sft_768.pth；
#   REWARD=internlm（默认）还需 reward 模型：hf download internlm/internlm2-1_8b-reward --local-dir "$MINIMIND_ROOT/../internlm2-1_8b-reward"
#   REWARD=rule：16 GB 放不下时用 scripts/patch_grpo_rule_reward.py 生成的 train_grpo_rule.py（规则 reward）。
# Day 5：GRPO smoke。原脚本无步数上限，用 timeout 终止（MAX_MINUTES）；参数按周卡：batch 1 / gen 2 / seq 256 / gen_len 256 / loss grpo。
set -euo pipefail
: "${MINIMIND_ROOT:?请先 export MINIMIND_ROOT}"
PYTHON="${MM_PYTHON:-python}"
REWARD="${REWARD:-internlm}"
NUM_GEN="${NUM_GEN:-2}"
MAX_MINUTES="${MAX_MINUTES:-15}"
DTYPE="${DTYPE:-bfloat16}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-../../internlm2-1_8b-reward}"
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$MINIMIND_ROOT/out/full_sft_768.pth" ]] || { echo "缺少 out/full_sft_768.pth：先运行 run_sft_bounded.sh"; exit 1; }
RUN_DIR="$LAB/runs/grpo_smoke_$REWARD"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/grpo_stdout.log"

if [[ "$REWARD" == "rule" ]]; then
  "$PYTHON" "$LAB/scripts/patch_grpo_rule_reward.py"
  SCRIPT=train_grpo_rule.py; RM_PATH=unused
else
  SCRIPT=train_grpo.py; RM_PATH="$REWARD_MODEL_PATH"
fi
set +e
( cd "$MINIMIND_ROOT/trainer" && timeout "${MAX_MINUTES}m" "$PYTHON" "$SCRIPT" --batch_size 1 --num_generations "$NUM_GEN" --max_seq_len 256 --max_gen_len 256 \
    --loss_type grpo --accumulation_steps 1 --dtype "$DTYPE" --epochs 1 --log_interval 1 --save_interval 5 --debug_mode --debug_interval 1 --num_workers 0 \
    --from_weight full_sft --reward_model_path "$RM_PATH" --data_path ../dataset/rlaif.jsonl 2>&1 | tee "$LOG" )
code=${PIPESTATUS[0]}
set -e
[[ $code -eq 124 ]] && echo "[grpo] 达到 $MAX_MINUTES 分钟上限，已终止（smoke 目的已达成；checkpoint 每 5 步在 ../out 与 ../checkpoints）"
[[ $code -ne 0 && $code -ne 124 ]] && echo "[grpo] 退出码 $code：OOM / reward 模型加载失败 见 02_LAB_GUIDE 第 5 节"
"$PYTHON" "$LAB/scripts/mmp.py" grpo_stats --input "$LOG" --format debug-log --csv "$RUN_DIR/grpo_groups.csv" --json "$RUN_DIR/grpo_stats.json"
"$PYTHON" "$LAB/scripts/mmp.py" parse_log --input "$LOG" --format minimind --csv "$RUN_DIR/grpo_steps.csv" --png "$RUN_DIR/grpo_steps.png" --fields reward,kl_ref,adv_std,actor_loss,avg_response_len
