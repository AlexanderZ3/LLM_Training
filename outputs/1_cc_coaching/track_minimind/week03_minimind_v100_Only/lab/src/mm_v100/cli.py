"""scripts/ 里每个入口都要做的几件小事：切 UTF-8、定 out-dir、写 JSON/文本。

单独抽出来是因为做错任何一件的表现都很难查：
- 没切 UTF-8：Windows 控制台 cp1252 下 print 中文直接 UnicodeEncodeError，
  脚本在打印第一行日志时就崩了，而崩的位置看起来跟业务逻辑毫无关系；
- out-dir 各写各的：产物散落，证据对不上号；
- 写 JSON 忘了 ensure_ascii=False：证据文件里全是 \\u4e2d\\u6587。
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from .console import use_utf8


def setup() -> None:
    """每个脚本 main() 的第一行。重复调用无副作用。"""
    use_utf8()


def resolve_out_dir(explicit: Optional[str], run_tag: str) -> str:
    """--out-dir 优先；没给就用 paths.run_dir(run_tag)（即 $MM_RUNS_ROOT/<tag>）。"""
    from . import paths as P

    if explicit:
        os.makedirs(explicit, exist_ok=True)
        return os.path.abspath(explicit)
    return str(P.run_dir(run_tag))


def write_json(obj: Any, path: str) -> str:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=True)
    return os.path.abspath(path)


def write_text(text: str, path: str) -> str:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
        if not text.endswith("\n"):
            fh.write("\n")
    return os.path.abspath(path)


def add_src_to_path(script_file: str) -> str:
    """给 scripts/*.py 用：把 lab/src 塞进 sys.path 并返回它。

    脚本自己也要先做一次这件事才能 import 到本模块，所以这个函数只是
    让「万一被别处 import」时行为一致，不是 scripts 的第一道工序。
    """
    import sys

    here = os.path.dirname(os.path.abspath(script_file))
    src = os.path.join(os.path.dirname(here), "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return src
