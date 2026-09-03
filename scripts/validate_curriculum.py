#!/usr/bin/env python3
"""Deterministic, read-only structural checks for the 14-week curriculum."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


WEEKS = {
    1: "week01_tinystories_minigpt",
    2: "week02_nanovlm_vqa",
    3: "week03_v100_fp16_topology",
    4: "week04_tinyflowpolicy",
    5: "week05_data_contract_repro",
    6: "week06_single_gpu_profiling",
    7: "week07_ddp_scaling",
    8: "week08_fsdp_checkpoint",
    9: "week09_diffusion_vs_flow",
    10: "week10_smolvla",
    11: "week11_mini_wam",
    12: "week12_finexec_sft_cpt",
    13: "week13_finexec_scaling_peft_grpo",
    14: "week14_capstone",
}

FILES = {
    "01_FOUNDATIONS.md": (5750, ("##", "shape", "失败")),
    "02_LAB_GUIDE.md": (14000, ("验收", "证据", "预期")),
    "03_ORAL_EXAM.md": (1000, ("Recall", "Debug", "Trade-off")),
    "04_REFERENCE_ANSWERS.md": (1800, ("评分", "误区")),
}

QUESTION_ID = re.compile(
    r"(?<![A-Z0-9])(?:W(?:0[1-9]|1[0-4])[-_](?:Q|[A-Z]+)[-_]?\d{1,3}|Q\d{2,3})(?![A-Z0-9])",
    re.IGNORECASE,
)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def normalized_ids(text: str) -> set[str]:
    return {match.group(0).upper().replace("_", "-") for match in QUESTION_ID.finditer(text)}


def check_local_links(path: Path, text: str) -> list[str]:
    errors: list[str] = []
    prose_lines: list[str] = []
    inside_fence = False
    for line in text.splitlines():
        if re.match(r"^\s*(?:```|~~~)", line):
            inside_fence = not inside_fence
            continue
        if inside_fence or line.startswith(("    ", "\t")):
            continue
        prose_lines.append(line)
    for raw_target in MARKDOWN_LINK.findall("\n".join(prose_lines)):
        target = raw_target.strip().strip("<>").split("#", 1)[0].split("?", 1)[0]
        if not target or re.match(r"^(?:https?://|mailto:)", target, re.IGNORECASE):
            continue
        candidate = (path.parent / target).resolve()
        if not candidate.exists():
            errors.append(f"broken local link: {path}: {raw_target}")
    return errors


def validate(root: Path) -> tuple[list[str], list[str], int]:
    errors: list[str] = []
    warnings: list[str] = []
    checked = 0
    course_root = root / "outputs" / "0_basic_training"

    for week, dirname in WEEKS.items():
        week_dir = course_root / dirname
        if not week_dir.is_dir():
            errors.append(f"missing week directory: {week_dir}")
            continue

        texts: dict[str, str] = {}
        for filename, (minimum, required_terms) in FILES.items():
            path = week_dir / filename
            if not path.is_file():
                errors.append(f"missing file: {path}")
                continue
            checked += 1
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                errors.append(f"not UTF-8: {path}")
                continue
            texts[filename] = text
            if len(text.strip()) < minimum:
                errors.append(f"content too short ({len(text.strip())} < {minimum}): {path}")
            for term in required_terms:
                if term.lower() not in text.lower():
                    errors.append(f"missing required term '{term}': {path}")
            if filename == "01_FOUNDATIONS.md" and not re.search(r"失败|故障|风险|OOM|hang|NaN|不一致", text, re.IGNORECASE):
                errors.append(f"foundations lacks failure/fault/risk discussion: {path}")
            if not text.lstrip().startswith("#"):
                errors.append(f"missing Markdown title: {path}")

            if filename == "01_FOUNDATIONS.md":
                foundation_groups = {
                    "data/model object": r"数据|dataset|模型|model",
                    "source/version policy": r"官方|来源|revision|commit|tag|版本",
                    "algorithm/formula": r"公式|推导|算法|loss|目标函数",
                    "shape/dtype worked example": r"shape|dtype|float(?:16|32|64)|int(?:32|64)",
                    "invariant/gate": r"不变量|Gate|门槛|正确性",
                    "infrastructure coupling": r"infra|显存|吞吐|通信|I/O|数值",
                    "self check": r"自检|Teach-back|闭卷|思考题",
                }
                for label, pattern in foundation_groups.items():
                    if not re.search(pattern, text, re.IGNORECASE):
                        errors.append(f"foundations lacks {label}: {path}")
                heading_count = len(re.findall(r"^#{2,4}\s+", text, re.MULTILINE))
                if heading_count < 8:
                    errors.append(f"foundations has only {heading_count} substantive headings; need >= 8: {path}")

            if filename == "02_LAB_GUIDE.md":
                lab_groups = {
                    "environment preflight": r"环境|前置检查|preflight",
                    "download or declared offline data": r"下载|download|clone|合成数据|无需下载|不下载外部",
                    "version pinning": r"revision|commit|tag|版本|hash|SHA-256",
                    "license/data authorization": r"许可|license|授权|审批|批准|无需外部数据",
                    "directory/files": r"目录|文件",
                    "implementation": r"实现|代码",
                    "training path": r"train|训练",
                    "incremental smoke path": r"smoke|冒烟|probe|探针|最小.{0,8}运行|短.{0,8}运行|50-step",
                    "independent evaluation/validation": r"eval|评测|验证|validator|审计",
                    "checkpoint/resume": r"checkpoint|resume|恢复",
                    "failure diagnosis": r"若失败|故障|诊断|排查",
                    "not-run evidence label": r"待执行|不是实测|未实测|尚未.*运行",
                }
                for label, pattern in lab_groups.items():
                    if not re.search(pattern, text, re.IGNORECASE):
                        errors.append(f"lab guide lacks {label}: {path}")
                day_count = len(re.findall(r"^##\s+(?:Day|第\s*\d+\s*天)", text, re.MULTILINE | re.IGNORECASE))
                if day_count < 3:
                    errors.append(f"lab guide has only {day_count} day/phase sections; need >= 3: {path}")
                action_count = len(
                    re.findall(
                        r"^\s*(?:python(?:3)?\s|CUDA_VISIBLE_DEVICES=|torchrun\s|git\s+(?:clone|ls-remote)|hf\s+download|nvidia-smi|nsys\s|ncu\s)",
                        text,
                        re.MULTILINE | re.IGNORECASE,
                    )
                )
                if action_count < 8:
                    errors.append(f"lab guide has only {action_count} concrete command lines; need >= 8: {path}")
                if re.search(r"^\s*(?:TODO\b|pass\s*(?:#.*)?$)|此处省略|请自行补全|剩余代码自行", text, re.MULTILINE | re.IGNORECASE):
                    errors.append(f"lab guide contains an unfinished implementation marker: {path}")
            errors.extend(check_local_links(path, text))

            for line_no, line in enumerate(text.splitlines(), start=1):
                if re.search(r"V100.*不支持.*FP16", line, re.IGNORECASE) and not re.search(
                    r"笔误|错误|纠正|误解|误区|并非|不是|原句|输入", line
                ):
                    errors.append(f"likely false V100/FP16 claim: {path}:{line_no}")
                if re.search(r"(?:已实测通过|已经实机通过|已在用户.*运行通过)", line):
                    warnings.append(f"possible unsupported run claim: {path}:{line_no}")

        oral = texts.get("03_ORAL_EXAM.md", "")
        answers = texts.get("04_REFERENCE_ANSWERS.md", "")
        if oral and answers:
            oral_ids = normalized_ids(oral)
            answer_ids = normalized_ids(answers)
            if len(oral_ids) < 20:
                errors.append(f"week {week:02d}: only {len(oral_ids)} unique oral-exam IDs; need >= 20")
            missing_answers = sorted(oral_ids - answer_ids)
            extra_answers = sorted(answer_ids - oral_ids)
            if missing_answers:
                errors.append(f"week {week:02d}: questions without answers: {', '.join(missing_answers)}")
            if extra_answers:
                errors.append(f"week {week:02d}: answer IDs without questions: {', '.join(extra_answers)}")
            if re.search(r"(?:参考答案|标准答案|答案[:：])", oral):
                warnings.append(f"week {week:02d}: oral exam may leak an answer marker")

        lab = texts.get("02_LAB_GUIDE.md", "")
        if lab:
            if not re.search(r"公司|V100", lab, re.IGNORECASE):
                errors.append(f"week {week:02d}: lab guide lacks company/V100 boundary")
            if not re.search(r"5070\s*Ti|5070Ti", lab, re.IGNORECASE):
                errors.append(f"week {week:02d}: lab guide lacks personal 5070 Ti lane")
            if not re.search(
                r"(?:不得|禁止|不能|严禁|不).{0,30}(?:导出|上传|外传|外带|带出)"
                r"|(?:导出|上传|外传|外带|带出).{0,30}(?:不得|禁止|不能|严禁)"
                r"|(?:公司|原始|所有|全部).{0,60}(?:留公司|留在公司|留.*内部|留.*本机|公司机器)",
                lab,
            ):
                errors.append(f"week {week:02d}: lab guide lacks explicit company artifact boundary")

    expected = set(WEEKS.values())
    if course_root.is_dir():
        unexpected = sorted(
            item.name for item in course_root.iterdir() if item.is_dir() and item.name.startswith("week") and item.name not in expected
        )
        if unexpected:
            warnings.append(f"unexpected week directories: {', '.join(unexpected)}")

    return errors, warnings, checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root (defaults to the parent of scripts/).",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    errors, warnings, checked = validate(root)

    print(f"Curriculum root: {root}")
    print(f"Checked files: {checked}/56")
    for warning in warnings:
        print(f"WARN: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print(f"RESULT: FAIL ({len(errors)} error(s), {len(warnings)} warning(s))")
        return 1
    print(f"RESULT: PASS (0 errors, {len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
