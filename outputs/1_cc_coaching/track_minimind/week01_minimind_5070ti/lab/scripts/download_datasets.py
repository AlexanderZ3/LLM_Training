#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MiniMind 数据集与奖励模型的一次性全量下载器（可整夜无人值守运行）。

设计目标
--------
1. **可中断可续传**：`huggingface_hub` 自身按 blob 续传；本脚本在其之上再套一层
   重试与"整轮重扫"循环，网络断了直接重跑同一条命令，已完成的文件会被跳过。
2. **只信字节数与哈希**：每个文件下载后校验 *精确字节数*（清单里的数字来自
   2026-09-05 的官方 API），不匹配就判失败并重试；全部完成后写
   ``SHA256SUMS.txt`` 与 ``DOWNLOAD_MANIFEST.json``。
3. **不静默覆盖**：已存在且字节数正确的文件默认跳过（``--force`` 才重下）。
4. **磁盘先算后下**：开跑前按所选层级估算需求，空间不足直接退出，不会下到一半塞满盘。

固定版本（`已确认`，2026-09-05 由 HuggingFace API 取得）
---------------------------------------------------------
- 数据集 ``jingyaogong/minimind_dataset`` @ ``312afb4f76391145c6902f765bb51691c09a12f5``
  许可 ``apache-2.0`` 与 ``cc-by-nc-2.0``——**含非商用条款，作品集/商用前先自行确认**。
- 奖励模型 ``internlm/internlm2-1_8b-reward`` @ ``25f3593492ab4625ce00fce8c5e67802d6e702ca``
  许可标注为 ``other``，使用前读它的模型卡。

用法
----
    # 看清单与体积，不下载
    python download_datasets.py --tier all --dry-run

    # 整夜全量下载（数据集 + 奖励模型，约 29 GB）
    python download_datasets.py --tier all --out D:/.../datasets

    # 只下 Week M01 主线需要的 4 个文件（约 3.06 GB）
    python download_datasets.py --tier mini --out D:/.../datasets

    # 事后只校验，不下载
    python download_datasets.py --tier all --out D:/.../datasets --verify-only

层级
----
``mini``   Week M01 Day 1–5 直接用到的 4 个 jsonl（3.06 GB）
``full``   数据集仓库全部 jsonl（含 22.4 GB 的两个全量文件，25.59 GB）
``reward`` 奖励模型 internlm2-1_8b-reward（3.40 GB）
``all``    ``full`` + ``reward``（28.99 GB）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DATASET_REPO = "jingyaogong/minimind_dataset"
DATASET_REV = "312afb4f76391145c6902f765bb51691c09a12f5"
REWARD_REPO = "internlm/internlm2-1_8b-reward"
REWARD_REV = "25f3593492ab4625ce00fce8c5e67802d6e702ca"

# (相对路径, 精确字节数, 层级标签, 一句话用途)
DATASET_FILES: List[Tuple[str, int, str, str]] = [
    ("pretrain_t2t_mini.jsonl", 1_241_043_656, "mini", "Day 2 预训练主数据（{\"text\"}）"),
    ("sft_t2t_mini.jsonl", 1_739_201_170, "mini", "Day 1/3 SFT 主数据（conversations）"),
    ("dpo.jsonl", 53_653_322, "mini", "Day 4 偏好对（chosen/rejected）"),
    ("rlaif.jsonl", 23_754_740, "mini", "Day 5 GRPO 的 prompt 来源"),
    ("pretrain_t2t.jsonl", 8_275_074_893, "full", "全量预训练（mini 的超集，跑长训练才需要）"),
    ("sft_t2t.jsonl", 14_096_018_369, "full", "全量 SFT（mini 的超集）"),
    ("agent_rl.jsonl", 82_036_930, "full", "Agentic RL 轨迹（多轮工具调用）"),
    ("agent_rl_math.jsonl", 18_372_683, "full", "Agentic RL 数学子集"),
    ("lora_medical.jsonl", 34_002_385, "full", "LoRA 医疗领域（Day 3 可选扩展）"),
    ("lora_exam.jsonl", 24_650_716, "full", "LoRA 考试领域"),
    ("lora_identity.jsonl", 22_789, "full", "LoRA 身份认知（极小，验证 LoRA 链路最快）"),
]

REWARD_FILES: List[Tuple[str, int, str, str]] = [
    ("model-00001-of-00002.safetensors", 1_981_392_544, "reward", "奖励模型权重 1/2"),
    ("model-00002-of-00002.safetensors", 1_417_790_344, "reward", "奖励模型权重 2/2"),
    ("model.safetensors.index.json", 13_682, "reward", "分片索引"),
    ("config.json", 813, "reward", "模型配置"),
    ("configuration_internlm2.py", 9_042, "reward", "remote code：配置类"),
    ("modeling_internlm2.py", 91_364, "reward", "remote code：模型实现"),
    ("tokenization_internlm2.py", 8_806, "reward", "remote code：tokenizer"),
    ("tokenization_internlm2_fast.py", 7_805, "reward", "remote code：fast tokenizer"),
    ("tokenizer.model", 1_477_754, "reward", "SentencePiece 词表"),
    ("tokenizer_config.json", 2_950, "reward", "tokenizer 配置"),
    ("special_tokens_map.json", 809, "reward", "特殊 token 映射"),
]

TIER_ORDER = {"mini": 0, "full": 1, "reward": 2}


def selected(tier: str) -> List[Tuple[str, str, str, int, str, str]]:
    """返回 [(repo, repo_type, filename, size, tier, note), ...]，按体积从小到大。

    小文件先下：网络不稳时先把便宜的拿到手，坏掉也只损失大文件的进度。
    """
    out: List[Tuple[str, str, str, int, str, str]] = []
    want_data = tier in ("mini", "full", "all")
    want_reward = tier in ("reward", "all")
    for name, size, t, note in DATASET_FILES:
        if not want_data:
            continue
        if tier == "mini" and t != "mini":
            continue
        out.append((DATASET_REPO, "dataset", name, size, t, note))
    if want_reward:
        for name, size, t, note in REWARD_FILES:
            out.append((REWARD_REPO, "model", name, size, t, note))
    out.sort(key=lambda r: r[3])
    return out


def rev_of(repo: str) -> str:
    return DATASET_REV if repo == DATASET_REPO else REWARD_REV


def subdir_of(repo: str) -> str:
    return "minimind_dataset" if repo == DATASET_REPO else "internlm2_1_8b_reward"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.2f} PB"


def log(msg: str, logfile: Optional[Path]) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if logfile is not None:
        with logfile.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def sha256_of(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def local_path(out: Path, repo: str, filename: str) -> Path:
    return out / subdir_of(repo) / filename


def size_ok(path: Path, expect: int) -> bool:
    return path.is_file() and path.stat().st_size == expect


def download_one(repo: str, repo_type: str, filename: str, expect: int,
                 out: Path, logfile: Optional[Path], retries: int,
                 token: Optional[str]) -> Tuple[bool, str]:
    """下载单个文件并校验字节数。返回 (成功, 说明)。"""
    from huggingface_hub import hf_hub_download

    dest_dir = out / subdir_of(repo)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename

    for attempt in range(1, retries + 1):
        try:
            got = hf_hub_download(
                repo_id=repo,
                filename=filename,
                repo_type=repo_type,
                revision=rev_of(repo),
                local_dir=str(dest_dir),
                token=token,
            )
            gp = Path(got)
            if gp.resolve() != dest.resolve() and gp.is_file():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(gp, dest)
            actual = dest.stat().st_size if dest.is_file() else -1
            if actual == expect:
                return True, "ok"
            msg = f"字节数不符：期望 {expect:,}，实际 {actual:,}"
            log(f"  [retry {attempt}/{retries}] {filename}: {msg}", logfile)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - 网络层异常种类很多，统一重试
            log(f"  [retry {attempt}/{retries}] {filename}: {type(exc).__name__}: {exc}", logfile)
        if attempt < retries:
            wait = min(60, 5 * attempt)
            log(f"  等待 {wait}s 后重试…", logfile)
            time.sleep(wait)
    return False, "重试用尽"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="下载 MiniMind 数据集与奖励模型到本地 datasets 目录（可续传）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", default=None,
                    help="下载目标目录；缺省是本脚本所在周目录下的 datasets/")
    ap.add_argument("--tier", choices=("mini", "full", "reward", "all"), default="all",
                    help="下载哪一层（默认 all = 全部数据集 + 奖励模型）")
    ap.add_argument("--dry-run", action="store_true", help="只列清单与体积，不下载")
    ap.add_argument("--verify-only", action="store_true", help="只校验已存在的文件，不下载")
    ap.add_argument("--force", action="store_true", help="即使字节数正确也重新下载")
    ap.add_argument("--retries", type=int, default=6, help="单文件重试次数（默认 6）")
    ap.add_argument("--rounds", type=int, default=3,
                    help="整轮重扫次数：一轮结束后若仍有失败项就再来一轮（默认 3）")
    ap.add_argument("--no-hash", action="store_true",
                    help="跳过 sha256 计算（29 GB 全量哈希在机械盘上可能要十几分钟）")
    ap.add_argument("--endpoint", default=None,
                    help="覆盖 HF 端点，例如 https://hf-mirror.com（等价于设 HF_ENDPOINT）")
    ap.add_argument("--token", default=None, help="HF token（这两个仓库都不需要，留空即可）")
    ap.add_argument("--min-free-gb", type=float, default=5.0,
                    help="下载完成后至少要剩多少 GB，低于此值直接退出（默认 5）")
    args = ap.parse_args(argv)

    here = Path(__file__).resolve()
    week_dir = here.parents[2]          # lab/scripts/x.py -> lab -> week01_...
    out = Path(args.out).resolve() if args.out else (week_dir / "datasets").resolve()

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    out.mkdir(parents=True, exist_ok=True)
    logfile = out / "download.log"

    items = selected(args.tier)
    total = sum(it[3] for it in items)

    log("=" * 78, logfile)
    log(f"MiniMind 数据下载器  tier={args.tier}  目标={out}", logfile)
    log(f"数据集 {DATASET_REPO} @ {DATASET_REV[:12]}  (apache-2.0 + cc-by-nc-2.0)", logfile)
    log(f"奖励模型 {REWARD_REPO} @ {REWARD_REV[:12]}  (license: other)", logfile)
    log(f"端点 HF_ENDPOINT={os.environ.get('HF_ENDPOINT', 'https://huggingface.co (默认)')}", logfile)
    log(f"共 {len(items)} 个文件，合计 {human(total)}", logfile)
    log("=" * 78, logfile)

    # 已有多少
    have = sum(it[3] for it in items if size_ok(local_path(out, it[0], it[2]), it[3]))
    todo = total - have
    log(f"已就绪 {human(have)}，还需下载约 {human(todo)}", logfile)

    # 磁盘检查
    free = shutil.disk_usage(out).free
    log(f"目标盘可用 {human(free)}", logfile)
    if not args.dry_run and not args.verify_only:
        after = free - todo
        if after < args.min_free_gb * (1024 ** 3):
            log(f"空间不足：下载后仅剩 {human(after)}，低于阈值 {args.min_free_gb} GB。", logfile)
            log("对策：换 --out 到别的盘，或先跑 --tier mini（3.06 GB）。", logfile)
            return 2

    # 清单
    log("", logfile)
    log(f"{'层':<7}{'字节':>16}  {'状态':<8}文件 / 用途", logfile)
    for repo, _rt, name, size, tier, note in items:
        p = local_path(out, repo, name)
        state = "已就绪" if size_ok(p, size) else ("部分" if p.exists() else "待下载")
        log(f"{tier:<7}{size:>16,}  {state:<8}{subdir_of(repo)}/{name}  — {note}", logfile)
    log("", logfile)

    if args.dry_run:
        log("--dry-run：到此为止，未下载任何内容。", logfile)
        return 0

    # 下载
    failed: List[str] = []
    if not args.verify_only:
        for rnd in range(1, args.rounds + 1):
            pending = [it for it in items
                       if args.force or not size_ok(local_path(out, it[0], it[2]), it[3])]
            if not pending:
                log(f"第 {rnd} 轮：无待下载项。", logfile)
                break
            log(f"—— 第 {rnd}/{args.rounds} 轮，待下载 {len(pending)} 个文件，"
                f"{human(sum(p[3] for p in pending))} ——", logfile)
            for idx, (repo, rt, name, size, _tier, _note) in enumerate(pending, 1):
                log(f"[{idx}/{len(pending)}] {subdir_of(repo)}/{name}  ({human(size)})", logfile)
                t0 = time.time()
                ok, why = download_one(repo, rt, name, size, out, logfile, args.retries, args.token)
                dt = max(time.time() - t0, 1e-6)
                if ok:
                    log(f"    完成，用时 {dt:.0f}s，均速 {human(size / dt)}/s", logfile)
                else:
                    log(f"    失败：{why}", logfile)
            args.force = False  # 只在第一轮尊重 --force，后续轮次按字节数跳过
        failed = [f"{subdir_of(it[0])}/{it[2]}" for it in items
                  if not size_ok(local_path(out, it[0], it[2]), it[3])]

    # 校验与清单落盘
    log("", logfile)
    log("—— 校验 ——", logfile)
    manifest: Dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tier": args.tier,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REV,
        "reward_repo": REWARD_REPO,
        "reward_revision": REWARD_REV,
        "files": [],
    }
    sums: List[str] = []
    bad = 0
    for repo, _rt, name, size, tier, note in items:
        p = local_path(out, repo, name)
        rel = f"{subdir_of(repo)}/{name}"
        if not p.is_file():
            log(f"  缺失   {rel}", logfile)
            bad += 1
            manifest["files"].append({"path": rel, "expected_bytes": size, "status": "missing"})
            continue
        actual = p.stat().st_size
        if actual != size:
            log(f"  字节错 {rel}  期望 {size:,} 实际 {actual:,}", logfile)
            bad += 1
            manifest["files"].append({"path": rel, "expected_bytes": size,
                                      "actual_bytes": actual, "status": "size_mismatch"})
            continue
        digest = ""
        if not args.no_hash:
            digest = sha256_of(p)
            sums.append(f"{digest}  {rel}  {actual}")
        log(f"  OK     {rel}  {actual:,}" + (f"  sha256={digest[:16]}…" if digest else ""), logfile)
        manifest["files"].append({"path": rel, "expected_bytes": size,
                                  "actual_bytes": actual, "sha256": digest or None,
                                  "tier": tier, "note": note, "status": "ok"})

    (out / "DOWNLOAD_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if sums:
        (out / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")

    log("", logfile)
    log(f"清单已写入 {out / 'DOWNLOAD_MANIFEST.json'}", logfile)
    if sums:
        log(f"哈希已写入 {out / 'SHA256SUMS.txt'}", logfile)
    if bad or failed:
        log(f"仍有 {bad} 个文件未就绪。重跑同一条命令即可续传（已完成的会跳过）。", logfile)
        return 1
    log("全部文件字节数校验通过。", logfile)
    log("", logfile)
    log("下一步：把数据接进 MiniMind 或本 lab —— 见 06_DATASET_DOWNLOAD.md 第 5 节。", logfile)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[中断] 已下载的部分保留，重跑同一条命令可续传。", flush=True)
        sys.exit(130)
