"""Day 2：parse_log 解析 bounded_train JSONL 与 MiniMind 三种 stdout 格式（格式抄自 commit 7a6fddd 源码）。"""
from __future__ import annotations

import csv
import json

from mm_probe.parse_log import ascii_summary, detect_format, load_rows, parse_minimind_line, write_csv

PRETRAIN_LINE = "Epoch:[1/2](100/38783), loss: 5.1234, logits_loss: 5.1234, aux_loss: 0.0000, lr: 0.00049000, epoch_time: 12.0min"
DPO_LINE = "Epoch:[1/1](100/2000), loss: 0.6931, dpo_loss: 0.6931, aux_loss: 0.0000, learning_rate: 0.00000004, epoch_time: 3.000min"
GRPO_LINE = ("Epoch:[1/1](3/500), Reward: 1.2345, KL_ref: 0.0012, Adv Std: 0.9000, Adv Mean: 0.0000, "
             "Actor Loss: -0.0123, Avg Response Len: 120.50, Learning Rate: 0.00000030")


def test_parse_pretrain_sft_line():
    r = parse_minimind_line(PRETRAIN_LINE)
    assert r["epoch"] == 1 and r["epochs"] == 2 and r["step"] == 100 and r["iters"] == 38783
    assert r["loss"] == 5.1234 and r["logits_loss"] == 5.1234 and r["aux_loss"] == 0.0
    assert r["lr"] == 0.00049 and r["epoch_time"] == 12.0


def test_parse_dpo_line():
    r = parse_minimind_line(DPO_LINE)
    assert r["dpo_loss"] == 0.6931 and r["learning_rate"] == 4e-8 and r["epoch_time"] == 3.0


def test_parse_grpo_line():
    r = parse_minimind_line(GRPO_LINE)
    assert r["reward"] == 1.2345 and r["kl_ref"] == 0.0012 and r["adv_std"] == 0.9 and r["adv_mean"] == 0.0
    assert r["actor_loss"] == -0.0123 and r["avg_response_len"] == 120.5 and r["learning_rate"] == 3e-7


def test_non_log_lines_ignored():
    assert parse_minimind_line("Model Params: 25.83M") is None
    assert parse_minimind_line("[DEBUG] gen[0] reward=1.0") is None


def test_stdout_file_and_csv(tmp_path):
    p = tmp_path / "stdout.log"
    p.write_text("Model Params: 25.83M\n" + PRETRAIN_LINE + "\n" + PRETRAIN_LINE.replace("(100/", "(200/").replace("5.1234", "4.9") + "\n", encoding="utf-8")
    assert detect_format(str(p)) == "minimind"
    rows = load_rows(str(p))
    assert [r["step"] for r in rows] == [100, 200]
    out = tmp_path / "o.csv"
    keys = write_csv(rows, str(out))
    assert "loss" in keys
    with open(out, encoding="utf-8", newline="") as f:
        recs = list(csv.DictReader(f))
    assert len(recs) == 2 and float(recs[1]["loss"]) == 4.9
    s = ascii_summary(rows)
    assert "loss" in s and "rows=2" in s


def test_jsonl_format(tmp_path):
    p = tmp_path / "log.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for i in range(3):
            f.write(json.dumps({"step": i + 1, "loss": 3.0 - i, "lr": 1e-4, "grad_norm": 0.5, "scaler_scale": None,
                                "tokens_per_s": 100.0, "peak_mem_mb": None}) + "\n")
    assert detect_format(str(p)) == "jsonl"
    rows = load_rows(str(p))
    assert len(rows) == 3 and rows[-1]["loss"] == 1.0
    s = ascii_summary(rows, ["loss", "tokens_per_s"])
    assert "tokens_per_s" in s
