#!/usr/bin/env python
"""核对 MM_DATA_ROOT 下的数据布局是否与 DATA_MANIFEST.json 一致。

拷完数据在 V100 上跑的第一条命令。它回答一个问题：**cc 写的代码里那些路径，
在这台机器上是不是真的都指得到东西。**

只打印文件名、字节数和判定结果，不读取也不打印任何样本内容，
所以它的输出可以直接贴出来给 cc 看，不涉及数据外流。

退出码：0 全部通过；1 有缺失或字节数不符；2 用法/清单错误。

用法
----
  python lab/scripts/check_data_layout.py
  python lab/scripts/check_data_layout.py --group minimind_dataset
  python lab/scripts/check_data_layout.py --sha256
  python lab/scripts/check_data_layout.py --json > /tmp/layout.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mm_v100 import paths  # noqa: E402
from mm_v100.console import use_utf8  # noqa: E402

SHA_PREFIX_LEN = 16
READ_CHUNK = 8 * 1024 * 1024

STATUS_OK = "OK"
STATUS_MISSING = "MISSING"
STATUS_SIZE = "SIZE-MISMATCH"
STATUS_SHA = "SHA-MISMATCH"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(READ_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_full_sums(root: Path) -> Dict[str, str]:
    """如果用户把外网那台机器的 SHA256SUMS.txt 一起拷进来了，就用全量哈希比对。

    格式按 sha256sum 的输出：'<64 位哈希>  <相对路径>'。
    """
    f = root / "SHA256SUMS.txt"
    if not f.is_file():
        return {}
    out: Dict[str, str] = {}
    with open(f, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            digest, name = parts[0].lower(), parts[1].strip().lstrip("*")
            if len(digest) == 64:
                out[name.replace("\\", "/")] = digest
    return out


def group_has_content(gdir: Path) -> bool:
    """可选组是否算"用户拷进来了"。

    只看目录存在是不够的：本仓库里为了固定路径名，预先建了空目录并放了 .gitkeep。
    那个占位目录不该让可选组被判成缺文件。所以要求至少有一个非隐藏的普通文件。
    """
    if not gdir.is_dir():
        return False
    for child in gdir.iterdir():
        if child.is_file() and not child.name.startswith("."):
            return True
    return False


def select_entries(group: Dict[str, Any], tier: str,
                   used_by: Optional[str]) -> List[Dict[str, Any]]:
    """按层级 / 按哪一天用得到，挑出这次要校验的文件。

    为什么需要这个：23.83 GiB 的全量数据里，跑完 Day 1–5 只需要 mini 层的
    4 个文件（2.85 GiB）。内网拷贝很贵，学员合理地只带需要的那部分进去。
    如果校验脚本一律按 11 个文件报，只带核心数据的人会看到 7 个 MISSING，
    而那 7 个**本来就不该在那里**——一个把正确状态报成失败的检查，
    比没有检查更糟，因为它会训练人忽略这个检查。
    """
    entries = list(group["files"])
    if tier != "all":
        entries = [e for e in entries if e.get("tier", "full") == tier]
    if used_by:
        entries = [e for e in entries if used_by in (e.get("used_by") or [])]
    return entries


def check_group(
    group_name: str,
    group: Dict[str, Any],
    root: Path,
    do_sha: bool,
    full_sums: Dict[str, str],
    entries: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    gdir = root / group_name
    for entry in (group["files"] if entries is None else entries):
        rel = entry["path"]
        expected_bytes = int(entry["bytes"])
        p = gdir / rel
        row: Dict[str, Any] = {
            "group": group_name,
            "path": f"{group_name}/{rel}",
            "expected_bytes": expected_bytes,
            "actual_bytes": None,
            "status": STATUS_MISSING,
            "detail": "",
        }
        if not p.is_file():
            rows.append(row)
            continue
        actual = p.stat().st_size
        row["actual_bytes"] = actual
        if actual != expected_bytes:
            row["status"] = STATUS_SIZE
            row["detail"] = f"差 {actual - expected_bytes:+,} B"
            rows.append(row)
            continue
        row["status"] = STATUS_OK
        if do_sha:
            key = f"{group_name}/{rel}"
            digest = sha256_of(p)
            want_full = full_sums.get(key) or full_sums.get(rel)
            if want_full:
                if digest != want_full:
                    row["status"] = STATUS_SHA
                    row["detail"] = "全量 sha256 不符"
                else:
                    row["detail"] = "全量 sha256 一致"
            else:
                want_prefix = str(entry.get("sha256_prefix", "")).lower()
                if want_prefix:
                    if digest[:SHA_PREFIX_LEN] != want_prefix:
                        row["status"] = STATUS_SHA
                        row["detail"] = f"前 {SHA_PREFIX_LEN} 位不符"
                    else:
                        row["detail"] = f"前 {SHA_PREFIX_LEN} 位一致"
                else:
                    row["detail"] = "清单没有哈希，仅核字节数"
        rows.append(row)
    return rows


def main(argv: List[str]) -> int:
    use_utf8()
    ap = argparse.ArgumentParser(
        description="核对数据布局与 DATA_MANIFEST.json 是否一致（不读取样本内容）"
    )
    ap.add_argument(
        "--group",
        action="append",
        default=None,
        help="只检查某个组，可重复。默认检查所有非可选组；可选组存在才检查。",
    )
    ap.add_argument(
        "--sha256",
        action="store_true",
        help="同时校验哈希。23 GB 大约几分钟。有 SHA256SUMS.txt 就全量比，否则比前 16 位。",
    )
    ap.add_argument("--json", action="store_true", help="输出机器可读的 JSON 而不是表格")
    ap.add_argument(
        "--all-optional",
        action="store_true",
        help="强制检查可选组（例如奖励模型），缺失时算失败",
    )
    ap.add_argument(
        "--tier",
        choices=["mini", "full", "all"],
        default="all",
        help="只校验某一层。mini = 跑完 Day 1-5 需要的 4 个文件（2.85 GiB）；"
             "默认 all = 清单里全部 11 个（23.83 GiB）",
    )
    ap.add_argument(
        "--used-by",
        type=str,
        default=None,
        metavar="TAG",
        help="只校验某一天用得到的文件，例如 --used-by day1",
    )
    args = ap.parse_args(argv)

    try:
        manifest = paths.load_manifest()
    except paths.PathContractError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    root = paths.data_root()
    groups: Dict[str, Any] = manifest["groups"]

    selected: List[str] = []
    for name, g in groups.items():
        if args.group is not None:
            if name in args.group:
                selected.append(name)
            continue
        optional = bool(g.get("optional", False))
        if not optional or args.all_optional or group_has_content(root / name):
            selected.append(name)

    if args.group is not None:
        unknown = [g for g in args.group if g not in groups]
        if unknown:
            print(f"清单里没有这些组：{', '.join(unknown)}", file=sys.stderr)
            print(f"清单里有的组：{', '.join(groups)}", file=sys.stderr)
            return 2

    full_sums = load_full_sums(root) if args.sha256 else {}

    rows: List[Dict[str, Any]] = []
    skipped_by_filter = 0
    for name in selected:
        g = groups[name]
        entries = select_entries(g, args.tier, args.used_by)
        skipped_by_filter += len(g["files"]) - len(entries)
        rows.extend(check_group(name, g, root, args.sha256, full_sums, entries))

    bad = [r for r in rows if r["status"] != STATUS_OK]
    ok_count = len(rows) - len(bad)

    if args.json:
        print(
            json.dumps(
                {
                    "data_root": str(root),
                    "groups_checked": selected,
                    "sha256": args.sha256,
                    "full_sums_used": bool(full_sums),
                    "ok": ok_count,
                    "bad": len(bad),
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1 if bad else 0

    print(f"MM_DATA_ROOT = {root}")
    print(f"检查的组：{', '.join(selected) if selected else '（无）'}")
    if args.tier != "all" or args.used_by:
        crit = []
        if args.tier != "all":
            crit.append(f"tier={args.tier}")
        if args.used_by:
            crit.append(f"used_by={args.used_by}")
        # 不做静默截断：明确说清这次跳过了多少个，避免「只查了一部分」
        # 被读成「全部通过」。
        print(f"筛选：{'，'.join(crit)}（按此跳过 {skipped_by_filter} 个文件，"
              f"它们不在本次校验范围内）")
    if args.sha256:
        print(f"哈希模式：{'全量（用 SHA256SUMS.txt）' if full_sums else f'前 {SHA_PREFIX_LEN} 位（用清单）'}")
    print("")
    width = max((len(r["path"]) for r in rows), default=10)
    for r in rows:
        actual = f"{r['actual_bytes']:,}" if r["actual_bytes"] is not None else "-"
        line = f"  {r['status']:<14}{r['path']:<{width + 2}}{actual:>18}"
        if r["detail"]:
            line += f"   {r['detail']}"
        print(line)
    print("")
    total_expected = sum(r["expected_bytes"] for r in rows)
    print(f"合计 {len(rows)} 个文件，期望 {total_expected:,} B")
    print(f"通过 {ok_count}，有问题 {len(bad)}")

    if bad:
        print("")
        print("下一步：")
        if any(r["status"] == STATUS_MISSING for r in bad):
            print("  · MISSING —— 文件没拷到位。对照 datasets/README.md 第 2 节的目录树补齐，")
            print("    或者 export MM_DATA_ROOT 指向你实际放数据的那个父目录。")
        if any(r["status"] == STATUS_SIZE for r in bad):
            print("  · SIZE-MISMATCH —— 多半是传输被截断，或者用了文本模式传输改了换行。")
            print("    用二进制方式（tar / rsync / scp）重拷这几个文件。")
        if any(r["status"] == STATUS_SHA for r in bad):
            print("  · SHA-MISMATCH —— 字节数对但内容不对，说明拿到的不是同一个版本。")
            print("    确认外网下载时用的 revision 与清单里登记的一致。")
        return 1

    print("")
    print("全部通过。这一行可以作为 Day 0 的证据字段 data_layout=PASS。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
