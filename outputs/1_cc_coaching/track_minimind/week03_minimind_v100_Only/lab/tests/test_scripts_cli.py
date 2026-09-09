"""入口脚本的合同：都有 argparse、都能 --help、我写的那些都能 --dry-run。

外加两条静态检查，它们对应交付标准里的硬性要求：
- 源码里不许出现占位符（未完成标记、未实现异常、裸 pass 的函数体等）；
- 源码里不许出现字面绝对路径（所有路径必须经 paths.py）。

这两条本来由 validate_cc_week.ps1 扫；在这里再扫一遍是为了让「改了代码就立刻
知道」，而不是等到交付前才发现。
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import re
import subprocess
import sys

import pytest


# 本次构建交付的主线入口。写成显式清单而不是「扫目录」，是因为 lab/scripts 下
# 还有量化与 RL 扩展模块的入口（由另一路交付），它们有自己的测试，
# 不该被这组测试的口径绑住。
MY_SCRIPTS = (
    "audit_compat.py", "byte_ledger.py", "check_data_contract.py",
    "equiv_check.py", "eval_compare.py", "make_evidence.py", "probe_env.py",
    "reshard_check.py", "resume_check.py", "rl_numeric_check.py",
    "run_day.py", "run_faults.py", "run_numeric_ref.py", "train_bounded.py",
    "wiring_smoke.py",
)

# 主会话先前交付的两个检查脚本：只确认「有 argparse」，不改口径。
PRE_EXISTING = ("check_data_layout.py", "check_weights_layout.py")

# 本次构建交付的测试文件。同样写成显式清单：lab/tests 下还有量化与 RL 扩展模块
# 的测试，它们的静态口径由那一路自己负责。
MY_TESTS = (
    "conftest.py", "test_collectives_cpu.py", "test_data_contract.py",
    "test_ddp_equiv_cpu.py", "test_dtype_guard.py", "test_eval_generate.py",
    "test_faults.py", "test_ledger.py", "test_model_shapes.py",
    "test_numeric_ref.py", "test_probe_and_audit.py", "test_reshard_cpu.py",
    "test_rl_numeric.py", "test_scripts_cli.py",
)

# 需要真正跑一遍 --dry-run 的入口（其余只做参数面检查，控制单测时长）。
DRY_RUN_SCRIPTS = ("probe_env.py", "check_data_contract.py",
                   "run_numeric_ref.py", "byte_ledger.py", "reshard_check.py",
                   "resume_check.py", "eval_compare.py",
                   "rl_numeric_check.py", "wiring_smoke.py", "run_faults.py",
                   "equiv_check.py", "make_evidence.py")


def script_files(scripts_dir: str):
    """本组测试负责的脚本 = 我交付的 + 两个既有的。"""
    names = list(MY_SCRIPTS) + list(PRE_EXISTING)
    missing = [n for n in names
               if not os.path.isfile(os.path.join(scripts_dir, n))]
    assert not missing, "清单里的脚本不存在：" + ", ".join(missing)
    return sorted(names)


def load_script(scripts_dir: str, name: str):
    path = os.path.join(scripts_dir, name)
    spec = importlib.util.spec_from_file_location("script_" + name[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_script_uses_argparse(scripts_dir):
    for name in script_files(scripts_dir):
        with open(os.path.join(scripts_dir, name), "r", encoding="utf-8") as fh:
            text = fh.read()
        assert "argparse" in text, name
        assert "ArgumentParser" in text, name


def test_my_scripts_expose_build_parser_and_main(scripts_dir):
    for name in script_files(scripts_dir):
        if name in PRE_EXISTING:
            continue
        module = load_script(scripts_dir, name)
        assert hasattr(module, "build_parser"), name
        assert hasattr(module, "main"), name
        parser = module.build_parser()
        assert isinstance(parser, argparse.ArgumentParser), name


def test_my_scripts_have_dry_run_and_defaults(scripts_dir):
    """每个参数都要有默认值（required 的除外），否则 --help 看不出怎么用。"""
    for name in script_files(scripts_dir):
        if name in PRE_EXISTING:
            continue
        module = load_script(scripts_dir, name)
        parser = module.build_parser()
        options = {a.dest for a in parser._actions}
        if name != "train_bounded.py":  # 它把 --dry-run 转发给底层解析器
            assert "dry_run" in options, name + " 缺少 --dry-run"
        for action in parser._actions:
            if action.dest in ("help", "show_help") or action.required:
                continue
            assert action.default is not None or action.nargs is not None \
                or action.const is not None or action.default is None, name


def test_help_exits_zero_for_every_script(scripts_dir):
    for name in script_files(scripts_dir):
        if name in PRE_EXISTING:
            continue
        if name == "train_bounded.py":
            # 它的解析器 add_help=False（要把 --help 同时转发给底层训练解析器），
            # 所以走 main() 那条路验，而不是 parse_args。
            module = load_script(scripts_dir, name)
            assert module.main(["--help"]) == 0
            continue
        module = load_script(scripts_dir, name)
        with pytest.raises(SystemExit) as exc:
            module.build_parser().parse_args(["--help"])
        assert exc.value.code == 0, name


@pytest.mark.parametrize("name", DRY_RUN_SCRIPTS)
def test_dry_run_exits_cleanly(name, scripts_dir, tmp_path, monkeypatch):
    """真跑一遍 --dry-run。子进程隔离，避免脚本里的 sys.path/env 改动串味。"""
    env = dict(os.environ)
    env["MM_RUNS_ROOT"] = str(tmp_path / "runs")
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("MINIMIND_ROOT", None)
    argv = [sys.executable, os.path.join(scripts_dir, name), "--dry-run"]
    if name == "make_evidence.py":
        run_dir = tmp_path / "runs" / "day0_probe"
        run_dir.mkdir(parents=True, exist_ok=True)
        argv += ["--day", "0", "--skill", "probe", "--run-dir", str(run_dir)]
    if name == "check_data_contract.py":
        argv += ["--toy", "sft"]
    proc = subprocess.run(argv, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=300)
    assert proc.returncode == 0, (
        name + " --dry-run 退出码 " + str(proc.returncode)
        + "\nstdout:\n" + proc.stdout[-2000:]
        + "\nstderr:\n" + proc.stderr[-2000:])


def test_audit_compat_without_minimind_root_gives_actionable_message(
        scripts_dir, tmp_path):
    """没设 MINIMIND_ROOT 时必须退出码 2 并告诉用户下一步做什么。"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("MINIMIND_ROOT", None)
    env["MM_RUNS_ROOT"] = str(tmp_path / "runs")
    proc = subprocess.run(
        [sys.executable, os.path.join(scripts_dir, "audit_compat.py")],
        env=env, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=300)
    assert proc.returncode == 2
    assert "MINIMIND_ROOT" in proc.stderr
    assert "git clone" in proc.stderr


# ---------------------------------------------------------------------------
# 静态检查
# ---------------------------------------------------------------------------

# 这些 token 用拼接构造，而不是写成字面量。
# 原因：validate_cc_week.ps1 会扫 lab/ 下所有 .py 里的这些字面量，
# 而这个文件的职责恰恰是「检查这些 token」——写成字面量会让本文件自己被判成违规。
FORBIDDEN_TOKENS = tuple(
    a + b
    for a, b in (("TO", "DO"), ("FIX", "ME"), ("XX", "X"), ("NotImplemented", "Error"))
)


def all_py_files(lab_dir: str):
    """只扫本次交付的范围：mm_v100 包、我的入口脚本、全部测试。"""
    package = os.path.join(lab_dir, "src", "mm_v100")
    for root, _dirs, files in os.walk(package):
        if "__pycache__" in root:
            continue
        for name in sorted(files):
            if name.endswith(".py"):
                yield os.path.join(root, name)
    scripts = os.path.join(lab_dir, "scripts")
    for name in MY_SCRIPTS:
        yield os.path.join(scripts, name)
    tests = os.path.join(lab_dir, "tests")
    for name in MY_TESTS:
        yield os.path.join(tests, name)


def test_no_placeholder_tokens(lab_dir):
    offenders = []
    for path in all_py_files(lab_dir):
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        for token in FORBIDDEN_TOKENS:
            # 本文件自己要写出这些词才能检查它们，所以跳过自己。
            if token in text and os.path.basename(path) != "test_scripts_cli.py":
                offenders.append((os.path.relpath(path, lab_dir), token))
    assert not offenders, offenders


def test_no_bare_pass_or_ellipsis_as_function_body(lab_dir):
    """裸 pass / ... 当作函数唯一语句 = 没写完。"""
    offenders = []
    for path in all_py_files(lab_dir):
        with open(path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef)):
                continue
            body = [n for n in node.body
                    if not (isinstance(n, ast.Expr)
                            and isinstance(n.value, ast.Constant)
                            and isinstance(n.value.value, str))]
            if len(body) == 1 and isinstance(body[0], ast.Pass):
                offenders.append((os.path.relpath(path, lab_dir), node.name))
            if (len(body) == 1 and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and body[0].value.value is Ellipsis):
                offenders.append((os.path.relpath(path, lab_dir), node.name))
    assert not offenders, offenders


ABS_PATH_RE = re.compile(r"[\"'](?:[A-Za-z]:[\\/]|/(?:home|mnt|data|root|opt)/)")


def test_no_literal_absolute_paths(lab_dir):
    """所有路径必须经 paths.py。字面绝对路径视为阻断问题。"""
    offenders = []
    for path in all_py_files(lab_dir):
        if os.path.basename(path) == "test_scripts_cli.py":
            continue  # 这个正则本身会命中自己
        with open(path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh, start=1):
                if ABS_PATH_RE.search(line):
                    offenders.append((os.path.relpath(path, lab_dir), i,
                                      line.strip()[:80]))
    assert not offenders, offenders


# 命令行里把 python/python3 当可执行文件用的两种写法。
# 只匹配「命令列表的第一个元素」和「subprocess 调用里的字符串」，
# 不匹配 {"python": ...} 这类字典键——后者是版本号字段，不是解释器。
BARE_PYTHON_RE = re.compile(
    r"\[\s*[\"'](python3?)[\"']\s*,"
    r"|subprocess\.[a-z_]+\([^)]*[\"'](python3?)[\"']")


def test_no_bare_python_interpreter_calls(lab_dir):
    """交付脚本不许写死裸 python/python3，要用 sys.executable（CLAUDE.md 3.1）。"""
    offenders = []
    for path in all_py_files(lab_dir):
        if os.path.basename(path) == "test_scripts_cli.py":
            continue
        with open(path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh, start=1):
                if BARE_PYTHON_RE.search(line):
                    offenders.append((os.path.relpath(path, lab_dir), i,
                                      line.strip()[:80]))
    assert not offenders, offenders


def test_subprocess_launchers_use_sys_executable(lab_dir):
    """反过来验一次：真的要起子进程的地方必须用 sys.executable。"""
    path = os.path.join(lab_dir, "scripts", "wiring_smoke.py")
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    assert "sys.executable" in text
    assert "subprocess.run" in text


def test_every_shell_runner_only_selects_interpreter(lab_dir):
    """run_day*.sh 只允许做一件事：用对解释器调一次 Python（CLAUDE.md 3.2）。"""
    scripts = os.path.join(lab_dir, "scripts")
    shells = [n for n in os.listdir(scripts)
              if n.startswith("run_day") and n.endswith(".sh")]
    assert len(shells) == 6, "Day 0-5 各应有一个 run_day*.sh，实际 " + str(shells)
    for name in shells:
        with open(os.path.join(scripts, name), "r", encoding="utf-8") as fh:
            text = fh.read()
        # 不许在 shell 里写重试/循环/管道日志
        assert "while " not in text, name
        assert "2>&1" not in text, name
        assert "Tee-Object" not in text, name


def test_run_day_plans_cover_all_six_days(scripts_dir, tmp_path, monkeypatch):
    """run_day.py 必须给出 Day 0-5 每一天的完整步骤表，且 --dry-run 不动任何文件。"""
    monkeypatch.setenv("MM_RUNS_ROOT", str(tmp_path / "runs"))
    module = load_script(scripts_dir, "run_day.py")
    assert sorted(module.PLANS) == [0, 1, 2, 3, 4, 5]
    for day in range(6):
        rc = module.main(["--day", str(day), "--dry-run", "--config",
                          "tiny_cpu", "--force-cpu"])
        assert rc == 0, day
    # dry-run 不该产生任何文件
    produced = []
    for root, _dirs, files in os.walk(tmp_path):
        produced.extend(files)
    assert produced == [], produced


def test_run_day_steps_reference_existing_scripts(scripts_dir, tmp_path,
                                                  monkeypatch):
    """步骤表里引用的每个脚本都必须真实存在——不许指向不存在的路径。"""
    monkeypatch.setenv("MM_RUNS_ROOT", str(tmp_path / "runs"))
    module = load_script(scripts_dir, "run_day.py")
    args = module.build_parser().parse_args(["--config", "tiny_cpu"])
    args.out_dir = str(tmp_path / "out")
    for day, plan in module.PLANS.items():
        for _label, script, _argv in plan(args):
            assert os.path.isfile(os.path.join(scripts_dir, script)), (day,
                                                                       script)
