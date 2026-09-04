#!/usr/bin/env bash
# 命令等级：模板 —— 需要先 export MINIMIND_ROOT=<clone>；MM_PYTHON 指向装好 cu128 torch 的解释器（缺省 python）。
# Day 2：有界 pretrain。MODE=bounded（默认）用 lab 的 bounded_train.py（原脚本无步数上限，只能整轮）；MODE=epoch 调原 trainer/train_pretrain.py 跑 1 个 epoch。
# 例：MAX_STEPS=20 CONFIG=smoke_5070ti bash run_pretrain_bounded.sh
#     MAX_STEPS=300 OVERFIT_N=128 bash run_pretrain_bounded.sh
set -euo pipefail
: "${MINIMIND_ROOT:?请先 export MINIMIND_ROOT}"
PYTHON="${MM_PYTHON:-python}"
CONFIG="${CONFIG:-pretrain_5070ti}"
MAX_STEPS="${MAX_STEPS:-200}"
OVERFIT_N="${OVERFIT_N:-0}"
SAVE_EVERY="${SAVE_EVERY:-100}"
DTYPE="${DTYPE:-bfloat16}"
MODE="${MODE:-bounded}"
RESUME="${RESUME:-}"
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$LAB/runs/pretrain_$CONFIG"
mkdir -p "$RUN_DIR" "$MINIMIND_ROOT/out"

if [[ "$MODE" == "bounded" ]]; then
  args=(--config "$CONFIG" --stage pretrain --max-steps "$MAX_STEPS" --overfit-n "$OVERFIT_N" --save-every "$SAVE_EVERY" --dtype "$DTYPE"
        --save-dir "$RUN_DIR" --log-jsonl "$RUN_DIR/log_pretrain.jsonl" --export-pth "$MINIMIND_ROOT/out/pretrain_768.pth")
  [[ -n "$RESUME" ]] && args+=(--resume "$RESUME")
  "$PYTHON" "$LAB/scripts/mmp.py" bounded_train "${args[@]}"
  "$PYTHON" "$LAB/scripts/mmp.py" parse_log --input "$RUN_DIR/log_pretrain.jsonl" --csv "$RUN_DIR/log_pretrain.csv" --png "$RUN_DIR/log_pretrain.png"
else
  # 官方默认：batch 32, accumulation 8, max_seq_len 340, lr 5e-4, bf16；checkpoint 写到 ../out 与 ../checkpoints
  ( cd "$MINIMIND_ROOT/trainer" && "$PYTHON" train_pretrain.py --epochs 1 --batch_size 32 --accumulation_steps 8 --max_seq_len 340 --dtype "$DTYPE" \
      --log_interval 20 --save_interval 500 --num_workers 4 --data_path ../dataset/pretrain_t2t_mini.jsonl 2>&1 | tee "$RUN_DIR/minimind_pretrain_stdout.log" )
  "$PYTHON" "$LAB/scripts/mmp.py" parse_log --input "$RUN_DIR/minimind_pretrain_stdout.log" --format minimind --csv "$RUN_DIR/minimind_pretrain.csv" --png "$RUN_DIR/minimind_pretrain.png"
fi
