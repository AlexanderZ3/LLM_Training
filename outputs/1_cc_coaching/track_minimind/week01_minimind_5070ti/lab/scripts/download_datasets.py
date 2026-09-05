#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MiniMind 数据集与奖励模型下载器——纯 Python，一条命令跑整夜。

为什么全部逻辑都在 Python 里
----------------------------
最初版本用 PowerShell 包一层做重试循环，结果踩了 Windows PowerShell 5.1 的坑：
对原生命令用 ``2>&1`` 会把**每一行标准错误**包装成 ErrorRecord，配合脚本开头的
``$ErrorActionPreference = "Stop"``，`huggingface_hub` 一条无害的
"建议安装 hf_xet" 提示就把整个下载终止在第 10 个文件。
所以现在重试、日志、编码、磁盘检查、整夜循环**全部在 Python 内部完成**，
不依赖任何 shell 语义。

设计要点
--------
1. **可中断可续传**：``huggingface_hub`` 自身按 blob 续传；本脚本在其之上再套
   单文件重试与整轮重扫。断线、断电、Ctrl-C 之后重跑同一条命令即可继续。
2. **只信字节数**：每个文件下载后校验*精确字节数*（清单里的数字取自
   2026-09-05 的官方 API），不匹配判失败并重试。
3. **不静默覆盖**：字节数正确的文件默认跳过（``--force`` 才重下）。
4. **磁盘先算后下**：按所选层级估算需求，空间不足直接退出。
5. **认准 conda 解释器**：默认强制在 ``ResearchAgentPy310`` 下运行；用别的
   解释器启动会自动 re-exec 过去（``--no-reexec`` 可关掉）。

固定版本（`已确认`，2026-09-05 由 HuggingFace API 取得）
---------------------------------------------------------
- 数据集 ``jingyaogong/minimind_dataset`` @ ``312afb4f76391145c6902f765bb51691c09a12f5``
  许可 ``apache-2.0`` 与 ``cc-by-nc-2.0``——**含非商用条款**。
- 奖励模型 ``internlm/internlm2-1_8b-reward`` @ ``25f3593492ab4625ce00fce8c5e67802d6e702ca``
  许可标注为 ``other``。

用法
----
    # 整夜下载全部数据集（23.62 GB）——这就是唯一需要记住的命令
    python download_datasets.py --overnight

    # 只看计划不下载
    python download_datasets.py --dry-run

    # 只下 Week M01 主线需要的 4 个文件（3.06 GB）
    python download_datasets.py --tier mini

    # 事后只校验
    python download_datasets.py --verify-only

层级（默认 ``full``，**不含模型权重**）
------------------------------------------
``mini``   Week M01 Day 1–5 直接用到的 4 个 jsonl（3.06 GB）
``full``   数据集仓库全部 11 个 jsonl（23.62 GB）——**默认**
``reward`` InternLM2 1.8B 奖励模型权重（3.17 GB）。这是**模型不是数据集**，
           而且 Week M01 Day 5 默认走 ``mm_probe/rule_reward.py`` 的规则奖励，
           16 GB 显存同时放策略、参考和这个 1.8B 模型有溢出风险，所以它是
           可选升级路径，需要显式 ``--tier reward`` 或 ``--tier all`` 才会下。
``all``    ``full`` + ``reward``（26.79 GB）
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

# ---------------------------------------------------------------- conda 解释器

CONDA_ENV_NAME = "ResearchAgentPy310"
CONDA_PYTHON = r"D:\Software\Large\Anconda\envs\ResearchAgentPy310\python.exe"


def ensure_conda_interpreter(argv: List[str]) -> None:
    """若当前解释器不是项目约定的 conda 环境，就 re-exec 过去。

    约定见 CLAUDE.md 第 3.1 节：本机所有 Python 一律走 ``ResearchAgentPy310``。
    裸 ``python`` 在这台机器上是 Windows 商店占位程序，退出码 9009。
    """
    target = Path(CONDA_PYTHON)
    if not target.is_file():
        return  # 换了机器就不强制，继续用当前解释器
    try:
        same = Path(sys.executable).resolve() == target.resolve()
    except OSError:
        same = False
    if same:
        return
    print(f"[conda] 当前解释器 {sys.executable}")
    print(f"[conda] 切换到     {target}（环境 {CONDA_ENV_NAME}）")
    os.execv(str(target), [str(target), os.path.abspath(__file__), *argv])


# ---------------------------------------------------------------- 固定清单

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


def selected(tier: str) -> List[Tuple[str, str, str, int, str, str]]:
    """返回 [(repo, repo_type, filename, size, tier, note), ...]，按体积从小到大。

    小文件先下：网络不稳时先把便宜的拿到手，最坏情况只损失大文件的进度。
    """
    out: List[Tuple[str, str, str, int, str, str]] = []
    want_data = tier in ("mini", "full", "all")
    want_reward = tier in ("reward", "all")
    if want_data:
        for name, size, t, note in DATASET_FILES:
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


# ---------------------------------------------------------------- 工具

def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{int(n)} B" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


class Logger:
    """同时写 stdout 与 UTF-8 日志文件。日志编码不受控制台代码页影响。"""

    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str = "") -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}" if msg else ""
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            # 控制台代码页装不下中文时退化成 ascii，不让它中断下载
            print(line.encode("ascii", "replace").decode("ascii"), flush=True)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as fh:
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


# ---------------------------------------------------------------- 下载

def download_one(repo: str, repo_type: str, filename: str, expect: int,
                 out: Path, log: Logger, retries: int,
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
            if gp.is_file() and gp.resolve() != dest.resolve():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(gp, dest)
            actual = dest.stat().st_size if dest.is_file() else -1
            if actual == expect:
                return True, "ok"
            log(f"  [retry {attempt}/{retries}] {filename}: "
                f"字节数不符，期望 {expect:,} 实际 {actual:,}")
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - 网络层异常种类很多，统一重试
            log(f"  [retry {attempt}/{retries}] {filename}: {type(exc).__name__}: {exc}")
        if attempt < retries:
            wait = min(60, 5 * attempt)
            log(f"  等待 {wait}s 后重试…")
            time.sleep(wait)
    return False, "重试用尽"


def download_round(items, out: Path, log: Logger, retries: int,
                   token: Optional[str], force: bool) -> int:
    """跑一轮下载，返回本轮结束后仍未就绪的文件数。"""
    pending = [it for it in items if force or not size_ok(local_path(out, it[0], it[2]), it[3])]
    if not pending:
        return 0
    log(f"待下载 {len(pending)} 个文件，{human(sum(p[3] for p in pending))}")
    for idx, (repo, rt, name, size, _tier, _note) in enumerate(pending, 1):
        log(f"[{idx}/{len(pending)}] {subdir_of(repo)}/{name}  ({human(size)})")
        t0 = time.time()
        ok, why = download_one(repo, rt, name, size, out, log, retries, token)
        dt = max(time.time() - t0, 1e-6)
        if ok:
            log(f"    完成，用时 {dt:.0f}s，均速 {human(size / dt)}/s")
        else:
            log(f"    失败：{why}")
    return sum(1 for it in items if not size_ok(local_path(out, it[0], it[2]), it[3]))


# ---------------------------------------------------------------- 校验与清单

def verify_and_write(items, out: Path, log: Logger, tier: str,
                     do_hash: bool) -> int:
    log()
    log("—— 校验 ——")
    manifest: Dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tier": tier,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REV,
        "reward_repo": REWARD_REPO,
        "reward_revision": REWARD_REV,
        "files": [],
    }
    sums: List[str] = []
    bad = 0
    for repo, _rt, name, size, t, note in items:
        p = local_path(out, repo, name)
        rel = f"{subdir_of(repo)}/{name}"
        if not p.is_file():
            log(f"  缺失   {rel}")
            bad += 1
            manifest["files"].append({"path": rel, "expected_bytes": size, "status": "missing"})
            continue
        actual = p.stat().st_size
        if actual != size:
            log(f"  字节错 {rel}  期望 {size:,} 实际 {actual:,}")
            bad += 1
            manifest["files"].append({"path": rel, "expected_bytes": size,
                                      "actual_bytes": actual, "status": "size_mismatch"})
            continue
        digest = ""
        if do_hash:
            digest = sha256_of(p)
            sums.append(f"{digest}  {rel}  {actual}")
        log(f"  OK     {rel}  {actual:,}" + (f"  sha256={digest[:16]}…" if digest else ""))
        manifest["files"].append({"path": rel, "expected_bytes": size,
                                  "actual_bytes": actual, "sha256": digest or None,
                                  "tier": t, "note": note, "status": "ok"})

    (out / "DOWNLOAD_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if sums:
        (out / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    log()
    log(f"清单已写入 {out / 'DOWNLOAD_MANIFEST.json'}")
    if sums:
        log(f"哈希已写入 {out / 'SHA256SUMS.txt'}")
    return bad


# ---------------------------------------------------------------- 主流程

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="下载 MiniMind 数据集与奖励模型（纯 Python，可整夜无人值守）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", default=None,
                    help="下载目标目录；缺省是本脚本所在周目录下的 datasets/")
    ap.add_argument("--tier", choices=("mini", "full", "reward", "all"), default="full",
                    help="下载哪一层。默认 full = **只下数据集**（11 个 jsonl，23.62 GB）。"
                         "mini = Week M01 主线的 4 个文件（3.06 GB）。"
                         "reward = 只下 InternLM2 1.8B 奖励模型权重（3.17 GB，可选，"
                         "Day 5 默认走规则奖励并不需要它）。all = 数据集 + 奖励模型（27 GB）")
    ap.add_argument("--overnight", action="store_true",
                    help="整夜模式：最多 --rounds 轮，每轮间隔 --sleep 秒，直到全部就绪")
    ap.add_argument("--rounds", type=int, default=3,
                    help="整轮重扫次数（--overnight 时默认提升到 20）")
    ap.add_argument("--sleep", type=int, default=60, help="轮间等待秒数（默认 60）")
    ap.add_argument("--retries", type=int, default=6, help="单文件重试次数（默认 6）")
    ap.add_argument("--dry-run", action="store_true", help="只列清单与体积，不下载")
    ap.add_argument("--verify-only", action="store_true", help="只校验已存在的文件，不下载")
    ap.add_argument("--force", action="store_true", help="即使字节数正确也重新下载")
    ap.add_argument("--no-hash", action="store_true",
                    help="跳过 sha256（27 GB 全量哈希在机械盘上要十几分钟）")
    ap.add_argument("--endpoint", default=None,
                    help="覆盖 HF 端点，例如 https://hf-mirror.com")
    ap.add_argument("--token", default=None, help="HF token（这两个仓库都不需要）")
    ap.add_argument("--min-free-gb", type=float, default=5.0,
                    help="下载完成后至少要剩多少 GB，低于此值直接退出（默认 5）")
    ap.add_argument("--no-reexec", action="store_true",
                    help="不要自动切换到 conda 解释器")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    here = Path(__file__).resolve()
    week_dir = here.parents[2]  # lab/scripts/x.py -> lab -> week01_...
    out = Path(args.out).resolve() if args.out else (week_dir / "datasets").resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / "download.log")

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
    if args.overnight and args.rounds == 3:
        args.rounds = 20

    items = selected(args.tier)
    total = sum(it[3] for it in items)

    # 加速包检测：缺了只是慢一点，绝不是错误
    import importlib.util as _u
    accel = [m for m in ("hf_xet", "hf_transfer") if _u.find_spec(m)]

    log("=" * 78)
    log(f"MiniMind 数据下载器  tier={args.tier}"
        f"{'  [整夜模式]' if args.overnight else ''}")
    log(f"解释器 {sys.executable}")
    log(f"目标   {out}")
    log(f"数据集 {DATASET_REPO} @ {DATASET_REV[:12]}  (apache-2.0 + cc-by-nc-2.0)")
    log(f"奖励模型 {REWARD_REPO} @ {REWARD_REV[:12]}  (license: other)")
    log(f"端点   {os.environ.get('HF_ENDPOINT', 'https://huggingface.co (默认)')}")
    log(f"加速   {'、'.join(accel) if accel else '无（可 pip install hf_xet hf_transfer 提速）'}")
    log(f"共 {len(items)} 个文件，合计 {human(total)}")
    log("=" * 78)

    have = sum(it[3] for it in items if size_ok(local_path(out, it[0], it[2]), it[3]))
    todo_bytes = total - have
    log(f"已就绪 {human(have)}，还需下载约 {human(todo_bytes)}")

    free = shutil.disk_usage(out).free
    log(f"目标盘可用 {human(free)}")
    if not args.dry_run and not args.verify_only:
        after = free - todo_bytes
        if after < args.min_free_gb * (1024 ** 3):
            log(f"空间不足：下载后仅剩 {human(after)}，低于阈值 {args.min_free_gb} GB。")
            log("对策：换 --out 到别的盘，或先跑 --tier mini（3.06 GB）。")
            return 2

    log()
    log(f"{'层':<7}{'字节':>16}  {'状态':<8}文件 / 用途")
    for repo, _rt, name, size, tier, note in items:
        p = local_path(out, repo, name)
        state = "已就绪" if size_ok(p, size) else ("部分" if p.exists() else "待下载")
        log(f"{tier:<7}{size:>16,}  {state:<8}{subdir_of(repo)}/{name}  — {note}")
    log()

    if args.dry_run:
        log("--dry-run：到此为止，未下载任何内容。")
        return 0

    if not args.verify_only:
        force = args.force
        remaining = -1
        for rnd in range(1, args.rounds + 1):
            log(f"—— 第 {rnd}/{args.rounds} 轮 ——")
            remaining = download_round(items, out, log, args.retries, args.token, force)
            force = False  # 只有第一轮尊重 --force
            if remaining == 0:
                log(f"第 {rnd} 轮结束：全部就绪。")
                break
            if rnd < args.rounds:
                log(f"仍有 {remaining} 个文件未就绪，{args.sleep}s 后再来一轮…")
                time.sleep(args.sleep)

    bad = verify_and_write(items, out, log, args.tier, not args.no_hash)

    log()
    if bad:
        log(f"仍有 {bad} 个文件未就绪。重跑同一条命令即可续传（已完成的会跳过）。")
        return 1
    log("全部文件字节数校验通过。")
    log("下一步：把数据接进 MiniMind 或本 lab —— 见 06_DATASET_DOWNLOAD.md 第 5 节。")
    return 0


if __name__ == "__main__":
    _argv = sys.argv[1:]
    if "--no-reexec" not in _argv:
        ensure_conda_interpreter(_argv)
    try:
        sys.exit(main(_argv))
    except KeyboardInterrupt:
        print("\n[中断] 已下载的部分保留，重跑同一条命令可续传。", flush=True)
        sys.exit(130)
