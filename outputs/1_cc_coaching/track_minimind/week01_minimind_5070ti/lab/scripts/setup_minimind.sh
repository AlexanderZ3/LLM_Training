#!/usr/bin/env bash
# 命令等级：模板 —— 先改 ROOT（克隆目录）与 PYTHON（装好 torch/huggingface_hub 的解释器）再运行。
# 作用：克隆 MiniMind → checkout 锁定 commit → 用 huggingface_hub 下载周卡列出的 4 个 jsonl 到 dataset/ → 写 dataset/SHA256SUMS.txt
# 用法：ROOT=$HOME/minimind PYTHON=python bash lab/scripts/setup_minimind.sh [--skip-data]
set -euo pipefail
ROOT="${ROOT:-$HOME/minimind}"
PYTHON="${PYTHON:-python}"
COMMIT="7a6fddd63a30c06b2fdd5fac4089922b29bc841b"
REPO="https://github.com/jingyaogong/minimind"
FILES=(pretrain_t2t_mini.jsonl sft_t2t_mini.jsonl dpo.jsonl rlaif.jsonl)
SKIP_DATA=0
[[ "${1:-}" == "--skip-data" ]] && SKIP_DATA=1

if [[ ! -d "$ROOT/.git" ]]; then
  echo "[setup] cloning $REPO -> $ROOT"
  git clone --filter=blob:none "$REPO" "$ROOT"
fi
git -C "$ROOT" fetch --all --quiet
git -C "$ROOT" checkout --quiet "$COMMIT"
HEAD="$(git -C "$ROOT" rev-parse HEAD)"
[[ "$HEAD" == "$COMMIT" ]] || { echo "checkout 失败：HEAD=$HEAD 期望=$COMMIT"; exit 1; }
echo "[setup] MiniMind at $COMMIT"

DATA_DIR="$ROOT/dataset"
if [[ "$SKIP_DATA" -eq 0 ]]; then
  # 数据集：HF jingyaogong/minimind_dataset（约 3 GB；需要网络；国内可 export HF_ENDPOINT=https://hf-mirror.com）
  "$PYTHON" - "$DATA_DIR" "${FILES[@]}" <<'EOF'
from huggingface_hub import hf_hub_download
import sys
for f in sys.argv[2:]:
    p = hf_hub_download(repo_id='jingyaogong/minimind_dataset', filename=f, repo_type='dataset', local_dir=sys.argv[1])
    print('downloaded', p)
EOF
fi

: > "$DATA_DIR/SHA256SUMS.txt"
for f in "${FILES[@]}"; do
  p="$DATA_DIR/$f"
  if [[ -f "$p" ]]; then
    h="$(sha256sum "$p" | awk '{print $1}')"
    s="$(stat -c %s "$p" 2>/dev/null || stat -f %z "$p")"
    echo "$h  $f  $s" >> "$DATA_DIR/SHA256SUMS.txt"
    echo "[setup] $f sha256=$h size=$s"
  else
    echo "[setup] 缺少 $p"
  fi
done
echo "[setup] written $DATA_DIR/SHA256SUMS.txt"
echo "[setup] done. 设置：export MINIMIND_ROOT=$ROOT"
