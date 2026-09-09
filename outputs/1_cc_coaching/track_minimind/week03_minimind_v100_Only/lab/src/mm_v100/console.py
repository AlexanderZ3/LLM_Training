"""控制台输出的 UTF-8 兜底。每个带 CLI 的脚本入口第一行调用 use_utf8()。

为什么需要：这些脚本在两个地方跑。V100 是 Linux，locale 一般是 UTF-8，中文直接能打。
cc 这台 Windows 机器的控制台默认代码页是 cp1252/cp936，print 中文会直接抛
UnicodeEncodeError——不是输出难看，是整个脚本崩掉。开环交付的代码如果在本机
连自检都跑不起来，就没法在拷到 V100 之前发现任何问题。

所以统一在入口把 stdout/stderr 重绑到 UTF-8，并且 errors='replace'：宁可某个字符
显示成问号，也不要因为一条日志把训练脚本打断。这是 CLAUDE.md 3.2 节那次
27 GB 下载中断的同类教训——日志路径上的异常不该影响主流程。
"""

from __future__ import annotations

import io
import sys

__all__ = ["use_utf8"]

_DONE = False


def use_utf8() -> None:
    """把 stdout/stderr 切成 UTF-8。重复调用无副作用。"""
    global _DONE
    if _DONE:
        return
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
                continue
            except (ValueError, OSError):
                # 某些被重定向的流不支持 reconfigure，走下面的包装路径。
                pass
        buffer = getattr(stream, "buffer", None)
        if buffer is not None:
            setattr(sys, name, io.TextIOWrapper(buffer, encoding="utf-8", errors="replace"))
    _DONE = True
