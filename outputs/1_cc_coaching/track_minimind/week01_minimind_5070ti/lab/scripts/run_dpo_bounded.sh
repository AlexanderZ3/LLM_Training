#!/usr/bin/env bash
# 命令等级：模板 —— 需要 MINIMIND_ROOT、MM_PYTHON（缺省 python）、Day 3 产出的 $MINIMIND_ROOT/out/full_sft_768.pth。
# Day 4：DPO。原 train_dpo.py 无步数上限（只有 --epochs）：MODE=bounded（默认）用 lab bounded_train.py --stage dpo；MODE=epoch 原脚本整轮。
set -euo pipefail
: "${MINIMIND_ROOT:?请先 export MINIMIND_ROOT}"
PYTHON="${MM_PYTHON:-python}"
CONFIG="${CONFIG:-sft_5070ti}"
MAX_STEPS="${MAX_STEPS:-200}"
SAVE_EVERY="${SAVE_EVERY:-50}"
DTYPE="${DTYPE:-bfloat16}"
MODE="${MODE:-bounded}"
RESUME="${RESUME:-}"
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FROM_WEIGHT="$MINIMIND_ROOT/out/full_sft_768.pth"
[[ -f "$FROM_WEIGHT" ]] || { echo "缺少 $FROM_WEIGHT：先运行 run_sft_bounded.sh（MASK_FAULT=none）"; exit 1; }
RUN_DIR="$LAB/runs/dpo_$CONFIG"
mkdir -p "$RUN_DIR"

"$PYTHON" "$LAB/scripts/mmp.py" dpo_check --config "$CONFIG" --policy "$FROM_WEIGHT" --ref same --n 2 --beta 0.15 --check-ref-frozen --out "$RUN_DIR/dpo_check_before.json"

if [[ "$MODE" == "bounded" ]]; then
  args=(--config "$CONFIG" --stage dpo --max-steps "$MAX_STEPS" --save-every "$SAVE_EVERY" --dtype "$DTYPE"
        --from-weight "$FROM_WEIGHT" --save-dir "$RUN_DIR" --log-jsonl "$RUN_DIR/log_dpo.jsonl" --export-pth "$MINIMIND_ROOT/out/dpo_768.pth")
  [[ -n "$RESUME" ]] && args+=(--resume "$RESUME")
  "$PYTHON" "$LAB/scripts/mmp.py" bounded_train "${args[@]}"
  "$PYTHON" "$LAB/scripts/mmp.py" parse_log --input "$RUN_DIR/log_dpo.jsonl" --csv "$RUN_DIR/log_dpo.csv" --png "$RUN_DIR/log_dpo.png" --fields loss,reward_margin,dpo_accuracy,lr,grad_norm
else
  ( cd "$MINIMIND_ROOT/trainer" && "$PYTHON" train_dpo.py --epochs 1 --batch_size 4 --learning_rate 4e-8 --max_seq_len 1024 --beta 0.15 --dtype "$DTYPE" \
      --from_weight full_sft --log_interval 20 --save_interval 500 --num_workers 4 --data_path ../dataset/dpo.jsonl 2>&1 | tee "$RUN_DIR/minimind_dpo_stdout.log" )
  "$PYTHON" "$LAB/scripts/mmp.py" parse_log --input "$RUN_DIR/minimind_dpo_stdout.log" --format minimind --csv "$RUN_DIR/minimind_dpo.csv" --png "$RUN_DIR/minimind_dpo.png"
fi
AFTER="$MINIMIND_ROOT/out/dpo_768.pth"
# 训练后：policy = dpo 权重，ref = full_sft 权重 → margin 应为正、loss < ln2
"$PYTHON" "$LAB/scripts/mmp.py" dpo_check --config "$CONFIG" --policy "$AFTER" --ref "$FROM_WEIGHT" --n 2 --beta 0.15 --out "$RUN_DIR/dpo_check_after.json"
"$PYTHON" "$LAB/scripts/mmp.py" eval_generate --config "$CONFIG" --checkpoint "$AFTER" --mode chat --out "$RUN_DIR/eval.jsonl" --tag dpo
