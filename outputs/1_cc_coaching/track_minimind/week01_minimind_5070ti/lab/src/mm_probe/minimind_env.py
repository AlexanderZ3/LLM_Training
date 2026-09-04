"""定位并接入用户本地的 MiniMind 仓库（固定 commit）。

约定：
- `MINIMIND_ROOT` 环境变量或 `--minimind-root` 指向克隆目录；
- tokenizer 在 `$MINIMIND_ROOT/model/`；数据在 `$MINIMIND_ROOT/dataset/`；
- 本模块只把仓库根加进 `sys.path`，不复制任何 MiniMind 源码。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PINNED_COMMIT = "7a6fddd63a30c06b2fdd5fac4089922b29bc841b"
MINIMIND_GIT_URL = "https://github.com/jingyaogong/minimind"
HF_DATASET_REPO = "jingyaogong/minimind_dataset"

# 周卡核验过的 4 个数据文件（文件名 → 官方记录的字节数）
DATA_FILES: Dict[str, int] = {
    "pretrain_t2t_mini.jsonl": 1_241_043_656,
    "sft_t2t_mini.jsonl": 1_739_201_170,
    "dpo.jsonl": 53_653_322,
    "rlaif.jsonl": 23_754_740,
}

LAB_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = LAB_ROOT / "configs"


class MiniMindNotFound(RuntimeError):
    """MINIMIND_ROOT 未设置或目录不完整。"""


def find_minimind_root(cli_value: Optional[str] = None, require: bool = True) -> Optional[Path]:
    """按 `--minimind-root` → `MINIMIND_ROOT` 顺序解析仓库根目录。

    require=True 时目录缺失会抛 MiniMindNotFound，并说明如何修复。
    """
    candidate = cli_value or os.environ.get("MINIMIND_ROOT")
    if not candidate:
        if require:
            raise MiniMindNotFound(
                "MINIMIND_ROOT 未设置。先运行 lab/scripts/setup_minimind.(ps1|sh) 克隆 MiniMind，"
                "再 `set MINIMIND_ROOT=<clone dir>`（PowerShell: $env:MINIMIND_ROOT=...）或传 --minimind-root。"
            )
        return None
    root = Path(candidate).expanduser().resolve()
    needed = [root / "model" / "model_minimind.py", root / "model" / "tokenizer.json", root / "dataset" / "lm_dataset.py"]
    missing = [str(p) for p in needed if not p.exists()]
    if missing and require:
        raise MiniMindNotFound(f"MINIMIND_ROOT={root} 缺少文件: {missing}。请确认它是 MiniMind 仓库根目录。")
    return root


def add_minimind_to_path(root: Path) -> Path:
    """把仓库根加入 sys.path，使 `from model.model_minimind import ...` 可用。"""
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def git_head(root: Path) -> Optional[str]:
    """返回 `git rev-parse HEAD`；不是 git 仓库或 git 不可用时返回 None。"""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def sha256_file(path: Path, chunk_mb: int = 8) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_mb * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tokenizer(root: Path):
    """加载 MiniMind 自带 tokenizer（vocab 6400，bos=<|im_start|>=1，eos=<|im_end|>=2，pad=<|endoftext|>=0）。"""
    from transformers import AutoTokenizer  # 延迟导入：纯函数模块不需要 transformers

    return AutoTokenizer.from_pretrained(str(root / "model"))


def load_config(path_or_name: str) -> Dict[str, Any]:
    """读取 lab/configs/*.json；既接受文件路径也接受配置名（如 `tiny_cpu`）。"""
    p = Path(path_or_name)
    if not p.exists():
        p = CONFIG_DIR / f"{path_or_name}.json"
    if not p.exists():
        raise FileNotFoundError(f"配置不存在: {path_or_name}（也不在 {CONFIG_DIR}）")
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("name", p.stem)
    cfg["_path"] = str(p)
    return cfg


def build_model_config(model_cfg: Dict[str, Any]):
    """由配置 JSON 的 `model` 段构造 MiniMindConfig（需先 add_minimind_to_path）。"""
    from model.model_minimind import MiniMindConfig  # type: ignore

    kwargs = dict(model_cfg)
    hidden_size = int(kwargs.pop("hidden_size", 768))
    num_hidden_layers = int(kwargs.pop("num_hidden_layers", 8))
    use_moe = bool(kwargs.pop("use_moe", False))
    return MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, use_moe=use_moe, **kwargs)


def build_model(model_cfg: Dict[str, Any], device: str = "cpu"):
    """构造 MiniMindForCausalLM 并放到 device。"""
    from model.model_minimind import MiniMindForCausalLM  # type: ignore

    lm_config = build_model_config(model_cfg)
    model = MiniMindForCausalLM(lm_config)
    return model.to(device), lm_config


def resolve_data_path(spec: str, root: Optional[Path]) -> str:
    """把配置里的 `$MINIMIND_ROOT/dataset/x.jsonl` 展开为实际路径；`fixture` 原样返回。"""
    if spec == "fixture":
        return spec
    if "$MINIMIND_ROOT" in spec:
        if root is None:
            raise MiniMindNotFound("数据路径含 $MINIMIND_ROOT 但未设置 MINIMIND_ROOT")
        spec = spec.replace("$MINIMIND_ROOT", str(root))
    return str(Path(spec).expanduser())


def load_state_dict_any(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """读取两种 checkpoint：MiniMind 的 `{weight}_{hidden}.pth`（纯 state_dict，half）
    或 bounded_train 的 `*.pt`（含 'model' 键）。返回 float32 state_dict。"""
    import torch

    obj = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        sd = obj["model"]
    else:
        sd = obj
    return {k: (v.float() if hasattr(v, "dtype") and v.is_floating_point() else v) for k, v in sd.items()}


def utf8_stdout() -> None:
    """Windows 控制台默认 GBK；统一改为 UTF-8，避免打印中文 token 时报错。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
