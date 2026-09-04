#!/usr/bin/env bash
# 命令等级：模板 —— 需要 MINIMIND_ROOT、MM_PYTHON（缺省 python）、Day 2 产出的 $MINIMIND_ROOT/out/pretrain_768.pth。
# Day 3：有界 full SFT + mask 故障注入对照。MODE=bounded（默认）| epoch；MASK_FAULT=none | assistant_all_ignored | user_in_loss。
# 例：MAX_STEPS=300 bash run_sft_bounded.sh ; MASK_FAULT=user_in_loss MAX_STEPS=300 bash run_sft_bounded.sh
set -euo pipefail
: "${MINIMIND_ROOT:?请先 export MINIMIND_ROOT}"
PYTHON="${MM_PYTHON:-python}"
CONFIG="${CONFIG:-sft_5070ti}"
MAX_STEPS="${MAX_STEPS:-300}"
OVERFIT_N="${OVERFIT_N:-0}"
SAVE_EVERY="${SAVE_EVERY:-100}"
DTYPE="${DTYPE:-bfloat16}"
MODE="${MODE:-bounded}"
MASK_FAULT="${MASK_FAULT:-none}"
RESUME="${RESUME:-}"
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FROM_WEIGHT="$MINIMIND_ROOT/out/pretrain_768.pth"
[[ -f "$FROM_WEIGHT" ]] || { echo "缺少 $FROM_WEIGHT：先运行 run_pretrain_bounded.sh"; exit 1; }
RUN_DIR="$LAB/runs/sft_${CONFIG}_${MASK_FAULT}"
mkdir -p "$RUN_DIR"

if [[ "$MODE" == "bounded" ]]; then
  args=(--config "$CONFIG" --stage sft --max-steps "$MAX_STEPS" --overfit-n "$OVERFIT_N" --save-every "$SAVE_EVERY" --dtype "$DTYPE"
        --from-weight "$FROM_WEIGHT" --mask-fault "$MASK_FAULT" --save-dir "$RUN_DIR" --log-jsonl "$RUN_DIR/log_sft.jsonl")
  [[ "$MASK_FAULT" == "none" ]] && args+=(--export-pth "$MINIMIND_ROOT/out/full_sft_768.pth")
  [[ -n "$RESUME" ]] && args+=(--resume "$RESUME")
  set +e; "$PYTHON" "$LAB/scripts/mmp.py" bounded_train "${args[@]}"; code=$?; set -e
  "$PYTHON" "$LAB/scripts/mmp.py" parse_log --input "$RUN_DIR/log_sft.jsonl" --csv "$RUN_DIR/log_sft.csv" --png "$RUN_DIR/log_sft.png"
  "$PYTHON" "$LAB/scripts/mmp.py" eval_generate --config "$CONFIG" --checkpoint "$RUN_DIR/latest.pt" --mode chat --max-new-tokens 64 --out "$RUN_DIR/eval.jsonl" --tag "sft_$MASK_FAULT"
  [[ $code -ne 0 ]] && echo "bounded_train 退出码 $code（2 = 最后一步 loss 非有限；assistant_all_ignored 故障下这是预期症状）"
else
  ( cd "$MINIMIND_ROOT/trainer" && "$PYTHON" train_full_sft.py --epochs 1 --batch_size 8 --accumulation_steps 2 --max_seq_len 768 --learning_rate 1e-5 --dtype "$DTYPE" \
      --from_weight pretrain --log_interval 20 --save_interval 500 --num_workers 4 --data_path ../dataset/sft_t2t_mini.jsonl 2>&1 | tee "$RUN_DIR/minimind_sft_stdout.log" )
  "$PYTHON" "$LAB/scripts/mmp.py" parse_log --input "$RUN_DIR/minimind_sft_stdout.log" --format minimind --csv "$RUN_DIR/minimind_sft.csv" --png "$RUN_DIR/minimind_sft.png"
  "$PYTHON" "$LAB/scripts/mmp.py" eval_generate --config "$CONFIG" --checkpoint "$MINIMIND_ROOT/out/full_sft_768.pth" --mode chat --out "$RUN_DIR/eval.jsonl" --tag sft_epoch
fi
