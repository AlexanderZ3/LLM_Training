#!/usr/bin/env python
"""检查一个 HuggingFace 格式的权重目录是否完整、且能在**离线模式**下被加载。

内网上最常见的两种失败，这个脚本各堵一个：

1. **拷漏文件。** 只拷了 safetensors，漏了 tokenizer 或 `trust_remote_code` 要用的
   那几个 .py。报错发生在加载时，信息通常是一个找不到模块的 ImportError，
   不会告诉你"你少拷了 modeling_xxx.py"。
2. **代码悄悄联网。** transformers 在缓存未命中时会去连 huggingface.co。内网上
   这会挂住直到超时，看起来像"卡住了"而不是"没网"。所以这里显式把离线开关打开，
   让它**立刻失败并说清楚原因**，而不是等超时。

不加载权重张量本身（那要几十 GB 显存/内存），只做文件盘点 + config/tokenizer 解析。

退出码：0 通过；1 有问题；2 用法错误。

用法
----
  python lab/scripts/check_weights_layout.py --dir $MM_WEIGHTS_ROOT/qat_base
  python lab/scripts/check_weights_layout.py --name qat_base
  python lab/scripts/check_weights_layout.py --name qat_base --tokenizer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mm_v100 import paths  # noqa: E402
from mm_v100.console import use_utf8  # noqa: E402

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pth", ".pt")
TOKENIZER_ANY = (
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "spiece.model",
    "merges.txt",
)


def human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024 or unit == "TiB":
            return f"{x:,.1f} {unit}" if unit != "B" else f"{int(x):,} B"
        x /= 1024
    return f"{x:.1f} TiB"


def check_dir(d: Path, want_tokenizer: bool) -> List[str]:
    """返回问题列表。空列表表示通过。"""
    problems: List[str] = []

    if not d.is_dir():
        return [
            f"目录不存在：{d}\n"
            f"    下一步：内网下不动权重，要在外网机器上下好再拷进来，见 weights/README.md。"
        ]

    files = sorted(p for p in d.rglob("*") if p.is_file())
    names = {p.relative_to(d).as_posix() for p in files}
    total = sum(p.stat().st_size for p in files)

    print(f"目录：{d}")
    print(f"文件数：{len(files)}，合计 {human(total)}")
    print("")

    weights = [p for p in files if p.suffix in WEIGHT_SUFFIXES]
    if not weights:
        problems.append(
            f"没有任何权重文件（{'/'.join(WEIGHT_SUFFIXES)}）。\n"
            f"    下一步：确认拷贝时没有只拷到 config 和 tokenizer。"
        )
    else:
        print(f"  权重分片 {len(weights)} 个：")
        for p in weights[:8]:
            print(f"    {p.relative_to(d).as_posix():<48}{human(p.stat().st_size):>14}")
        if len(weights) > 8:
            print(f"    …… 还有 {len(weights) - 8} 个")
        print("")

    # 分片索引与实际分片必须对得上。
    index_files = [p for p in files if p.name.endswith(".index.json")]
    for idx in index_files:
        try:
            with open(idx, "r", encoding="utf-8") as fh:
                index = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"分片索引读不了：{idx.name}：{exc}")
            continue
        referenced = set(index.get("weight_map", {}).values())
        missing = sorted(s for s in referenced if s not in names)
        if missing:
            problems.append(
                f"{idx.name} 引用了 {len(referenced)} 个分片，其中这些不在目录里：\n"
                f"    {', '.join(missing)}\n"
                f"    下一步：把漏掉的分片补拷进来。分片不全时加载会报一个含糊的 KeyError。"
            )
        else:
            print(f"  {idx.name}：引用的 {len(referenced)} 个分片都在。")

    cfg = d / "config.json"
    if not cfg.is_file():
        problems.append(
            "缺 config.json。\n    下一步：这个文件很小，最容易在挑文件拷的时候被漏掉。"
        )
    else:
        try:
            with open(cfg, "r", encoding="utf-8") as fh:
                conf = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"config.json 解析失败：{exc}")
            conf = {}
        arch = conf.get("architectures") or ["<未写>"]
        print(f"  config.json：architectures={arch}，model_type={conf.get('model_type', '<未写>')}")

        # 远程代码模型：auto_map 引用的 .py 必须一起拷。
        auto_map = conf.get("auto_map") or {}
        if auto_map:
            needed = set()
            for target in auto_map.values():
                targets = target if isinstance(target, (list, tuple)) else [target]
                for t in targets:
                    if isinstance(t, str) and "--" not in t and "." in t:
                        needed.add(t.split(".")[0] + ".py")
            missing_py = sorted(m for m in needed if m not in names)
            if missing_py:
                problems.append(
                    f"config.json 的 auto_map 指向远程代码，但这些 .py 不在目录里：\n"
                    f"    {', '.join(missing_py)}\n"
                    f"    下一步：把它们一起拷进来，并且加载时传 trust_remote_code=True。"
                )
            else:
                print(f"  auto_map：{len(needed)} 个远程代码文件都在，加载时记得 trust_remote_code=True。")

    tok_present = sorted(n for n in names if n in TOKENIZER_ANY)
    if tok_present:
        print(f"  tokenizer 文件：{', '.join(tok_present)}")
    else:
        msg = f"没有任何 tokenizer 文件（{'/'.join(TOKENIZER_ANY)}）。"
        if want_tokenizer:
            problems.append(msg + "\n    下一步：tokenizer 文件很小但必须一起拷。")
        else:
            print(f"  提示：{msg} 如果这个目录只放权重、tokenizer 在别处，可以忽略。")

    if want_tokenizer:
        problems.extend(try_load_tokenizer(d))

    return problems


def try_load_tokenizer(d: Path) -> List[str]:
    """在强制离线模式下真的加载一次 tokenizer。这是最有信息量的一步。"""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print("")
    print("  离线加载 tokenizer（HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1）……")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        print(f"  跳过：这个环境没有 transformers（{exc}）。")
        return []
    try:
        tok = AutoTokenizer.from_pretrained(str(d), trust_remote_code=True, local_files_only=True)
    except Exception as exc:  # 加载失败的原因太多，这里要的是把原因原样报出来
        return [
            f"离线加载 tokenizer 失败：{type(exc).__name__}: {exc}\n"
            f"    下一步：如果错误里提到连接或 offline，说明还有文件没拷全；\n"
            f"    如果提到找不到模块，说明缺 trust_remote_code 需要的那几个 .py。"
        ]
    print(f"  成功：vocab_size={len(tok)}，类型={type(tok).__name__}")
    return []


def main(argv: List[str]) -> int:
    use_utf8()
    ap = argparse.ArgumentParser(description="检查 HuggingFace 权重目录是否完整且可离线加载")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dir", type=str, help="权重目录的完整路径")
    g.add_argument("--name", type=str, help="MM_WEIGHTS_ROOT 下的子目录名，例如 qat_base")
    ap.add_argument(
        "--tokenizer",
        action="store_true",
        help="额外真的离线加载一次 tokenizer（需要 transformers）",
    )
    args = ap.parse_args(argv)

    target: Optional[Path]
    if args.dir:
        target = Path(args.dir).expanduser().resolve()
    else:
        try:
            target = paths.weights_dir(args.name, must_exist=False)
        except paths.PathContractError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    problems = check_dir(target, args.tokenizer)

    print("")
    if problems:
        print(f"有 {len(problems)} 个问题：")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")
        return 1
    print("通过。这个目录可以在离线模式下使用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
