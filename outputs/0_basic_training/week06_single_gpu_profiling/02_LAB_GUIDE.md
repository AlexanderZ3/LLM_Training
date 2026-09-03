# Week 06 实践篇：从空目录到可信单卡 Profile

> 所有命令状态：待执行。所有“预计输出”是示例/判据，不是实测。复用已有 Conda 环境，不创建新 Conda 或 venv，不强升公司 PyTorch 2.1/CUDA。核验日期：2026-09-03。

## 0. 执行范围、预算与回滚

主路径在公司 Linux 的一张 V100 上；8×V100、32GB 与拓扑均为用户自述待核验。个人 5070 Ti 可用相同合成数据公开重跑；H100 仅作可选附录。公司代码、数据、日志、trace、图片、checkpoint、配置和性能数字不得离开公司环境。

估算资源，不是实测：

- 空闲磁盘至少 2GB；若保存 profiler trace，建议 5GB；
- smoke 预计 5–15 分钟，正式扫描/三次 A/B 预计 2–4 小时；
- smoke 默认小模型；约 100M 配置从 batch=1 开始；
- reserved 到 28–30GB 停止上探，正式配置必须 <30.5GB。

依赖变更回滚：本周核心仅用现有 torch 和标准库，不应安装任何包。若管理员批准依赖变更，先保存 pip freeze.before.txt；出现冲突时按组织流程恢复原镜像/环境，不在共享环境随意 pip uninstall。

## 1. 对象登记

| 对象 | 官方 ID/仓库 | 固定方法 | 所需文件 | 许可 | 完整性 |
|---|---|---|---|---|---|
| PyTorch | pytorch/pytorch | 实机记录 torch.__version__；目标 2.1.x | 已安装 wheel/环境 | BSD-style | import、version、CUDA probe |
| 合成 token | 本文 src/lab.py | seed=17 与 CLI 写入 metrics | 无下载 | 非第三方数据 | sample_count、shape、seed |
| TinyLM | 本文完整代码 | lab.py SHA-256 + CLI config | src/lab.py | 项目内部教学代码 | py_compile、CPU unit、参数量 |

离线替代就是核心路径本身：所有样本本机生成。不得把它称为真实语料效果实验。

## 2. 最终目录

从准备放实验的空目录执行：

    week06-profile/
      src/lab.py
      manifests/
        pip-freeze.before.txt
        environment.txt
        code.sha256
      runs/

src/lab.py：生成合成 token、构建 TinyLM，执行 unit/train/profile/eval、保存完整 checkpoint 与 metrics。  
manifests：环境、代码与依赖证据。  
runs：每次唯一输出；程序拒绝覆盖非空目录。

## Day 1：环境与边界 Gate

### 为什么做

先证明设备、版本、磁盘和工具满足前置条件；性能数据一旦在错误环境产生就不可解释。

### 输入与前置检查

在空目录的父目录执行以下待执行命令：

    mkdir -p week06-profile/src week06-profile/manifests week06-profile/runs
    cd week06-profile
    {
      date -Iseconds
      pwd
      which python
      python --version
      python -m pip --version
      git --version
      nvidia-smi
      nvidia-smi topo -m
      df -h .
      df -i .
      python -c "import torch; print('torch',torch.__version__); print('cuda_build',torch.version.cuda); print('cuda_available',torch.cuda.is_available()); print('device',torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('capability',torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None); print('bf16',torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None)"
      nsys --version || true
      ncu --version || true
    } | tee manifests/environment.txt
    python -m pip freeze | sort | tee manifests/pip-freeze.before.txt
    python -m pip check

### 本日要创建/修改的文件

environment.txt 与 pip-freeze.before.txt 已由上述命令创建。

### 实现

本日无 Python 实现。公司环境禁止上传日志；tee 文件留本机。

### 执行命令

依赖计划只读检查；本周无 requirements：

    python -c "from torch.profiler import profile, schedule, ProfilerActivity; print('profiler_import_ok')"
    python -c "import torch; assert torch.__version__.startswith('2.1.'), torch.__version__"

### 预期观测（估算/示例，不是实测）

应看到 cuda_available=True；V100 预期 capability=(7,0)、bf16=False；torch 应为 2.1.x。实际输出写入内部证据。

### 验收条件

- Python、torch、CUDA 可导入；
- 设备和调度分配一致；
- 至少 2GB 可用磁盘；
- pip check 成功；
- 若公司版本并非 2.1.x，标 ENV-MISMATCH，不自行升级。

### 若失败，先看什么，再改什么

先查调度/可见设备，再查已有模块/镜像，最后提交管理员；不下载未知 wheel、不绕过网络。

### 当日证据清单

environment.txt、pip-freeze.before.txt、设备与版本结论（待验证）。

## Day 2：创建完整训练器

### 为什么做

后续每条命令只引用本日完整创建的 src/lab.py，不假定任何旧周脚本存在。

### 输入与前置检查

    cd week06-profile
    test -d src
    test -f manifests/environment.txt

### 本日要创建/修改的文件

创建 src/lab.py。不得保留空实现或省略实现。

### 实现

本日完整创建文件：`src/lab.py`。

    cat > src/lab.py <<'PY'
    import argparse
    import contextlib
    import hashlib
    import json
    import math
    import random
    import time
    from pathlib import Path

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.profiler import ProfilerActivity, profile, record_function, schedule
    from torch.utils.data import DataLoader, Dataset


    class SyntheticTokens(Dataset):
        def __init__(self, samples, seq, vocab, seed, decode_ms=0.0, id_offset=0):
            g = torch.Generator().manual_seed(seed)
            self.tokens = torch.randint(0, vocab, (samples, seq + 1), generator=g)
            self.decode_s = decode_ms / 1000.0
            self.id_offset = id_offset

        def __len__(self):
            return self.tokens.size(0)

        def __getitem__(self, index):
            if self.decode_s:
                time.sleep(self.decode_s)
            row = self.tokens[index]
            return row[:-1], row[1:], torch.tensor(self.id_offset + index, dtype=torch.long)


    class TinyLM(nn.Module):
        def __init__(self, vocab, seq, dim, heads, layers):
            super().__init__()
            if dim % heads:
                raise ValueError("dim must be divisible by heads")
            self.vocab = vocab
            self.seq = seq
            self.token = nn.Embedding(vocab, dim)
            self.position = nn.Parameter(torch.zeros(1, seq, dim))
            self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=heads,
                    dim_feedforward=4 * dim,
                    dropout=0.0,
                    batch_first=True,
                    norm_first=True,
                    activation="gelu",
                )
                for _ in range(layers)
            ])
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, vocab, bias=False)
            self.head.weight = self.token.weight
            causal = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
            self.register_buffer("causal_mask", causal, persistent=False)

        def forward(self, ids):
            x = self.token(ids) + self.position[:, : ids.size(1)]
            for i, block in enumerate(self.blocks):
                with record_function("transformer_block"):
                    x = block(x, src_mask=self.causal_mask[: ids.size(1), : ids.size(1)])
            return self.head(self.norm(x))


    def seed_all(seed):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    def atomic_json(path, obj):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)


    def sha256_file(path):
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()


    def checkpoint_manifest_path(path):
        return Path(str(path) + ".manifest.json")


    def validate_checkpoint_file(path):
        path = Path(path)
        manifest_path = checkpoint_manifest_path(path)
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"checkpoint missing or empty: {path}")
        if not manifest_path.is_file() or manifest_path.stat().st_size <= 0:
            raise RuntimeError(f"checkpoint manifest missing or empty: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_code = sha256_file(__file__)
        if manifest.get("format_version") != 1:
            raise RuntimeError("unsupported checkpoint format_version")
        if manifest.get("checkpoint_file") != path.name:
            raise RuntimeError("checkpoint manifest filename mismatch")
        if manifest.get("bytes") != path.stat().st_size or manifest.get("sha256") != sha256_file(path):
            raise RuntimeError("checkpoint size/hash mismatch before torch.load")
        if manifest.get("torch_version") != str(torch.__version__):
            raise RuntimeError("checkpoint torch version mismatch before torch.load")
        if manifest.get("code_sha256") != expected_code:
            raise RuntimeError("checkpoint code SHA mismatch before torch.load")
        return manifest


    def finite_grads(model):
        return all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())


    def trajectory_config(args):
        keys = ("device", "precision", "samples", "batch", "seq", "vocab", "dim", "heads",
                "layers", "workers", "decode_ms", "lr", "clip", "seed")
        return {key: getattr(args, key) for key in keys}


    def save_checkpoint(path, model, optimizer, scaler, step, args):
        code_sha = sha256_file(__file__)
        state = {
            "format_version": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "config": vars(args),
            "trajectory_config": trajectory_config(args),
            "torch_version": str(torch.__version__),
            "code_sha256": code_sha,
        }
        tmp = Path(str(path) + ".tmp")
        torch.save(state, tmp)
        checkpoint_sha = sha256_file(tmp)
        checkpoint_bytes = tmp.stat().st_size
        tmp.replace(path)
        atomic_json(checkpoint_manifest_path(path), {
            "format_version": 1,
            "checkpoint_file": Path(path).name,
            "bytes": checkpoint_bytes,
            "sha256": checkpoint_sha,
            "step": step,
            "torch_version": str(torch.__version__),
            "code_sha256": code_sha,
        })


    def load_checkpoint(path, model, optimizer, scaler, device, args):
        manifest = validate_checkpoint_file(path)
        state = torch.load(path, map_location=device)
        if (state.get("format_version") != manifest["format_version"] or
                state.get("step") != manifest["step"] or
                state.get("code_sha256") != manifest["code_sha256"]):
            raise RuntimeError("checkpoint payload disagrees with validated manifest")
        if state.get("torch_version") != str(torch.__version__):
            raise RuntimeError(
                f"torch version mismatch: saved={state.get('torch_version')} "
                f"current={torch.__version__}"
            )
        saved = state.get("trajectory_config")
        current = trajectory_config(args)
        if saved != current:
            raise ValueError(f"resume trajectory mismatch: saved={saved}, current={current}")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        return int(state["step"])


    def percentile(values, q):
        if not values:
            return None
        x = torch.tensor(values, dtype=torch.float64)
        return float(torch.quantile(x, q).item())


    def make_loader(args, device, seed, id_offset, decode_ms):
        dataset = SyntheticTokens(
            args.samples, args.seq, args.vocab, seed, decode_ms, id_offset=id_offset
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch,
            shuffle=False,
            drop_last=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
            generator=torch.Generator().manual_seed(seed + 1_000_003),
        )
        return dataset, loader


    def build(args, device):
        dataset, loader = make_loader(
            args, device, seed=args.seed, id_offset=0, decode_ms=args.decode_ms
        )
        model = TinyLM(args.vocab, args.seq, args.dim, args.heads, args.layers).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        use_amp = device.type == "cuda" and args.precision == "fp16"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        return dataset, loader, model, optimizer, scaler, use_amp


    def unit_test():
        seed_all(17)
        args = argparse.Namespace(
            samples=16, seq=8, vocab=32, seed=17, decode_ms=0.0,
            batch=2, workers=0, dim=16, heads=4, layers=2, lr=1e-3,
            precision="fp32",
        )
        _, loader, model, optimizer, scaler, use_amp = build(args, torch.device("cpu"))
        x, y, ids = next(iter(loader))
        logits = model(x)
        assert logits.shape == (2, 8, 32)
        loss = F.cross_entropy(logits.reshape(-1, 32), y.reshape(-1))
        loss.backward()
        assert torch.isfinite(loss) and finite_grads(model)
        optimizer.step()
        print(json.dumps({"status": "PASS", "shape": list(logits.shape),
                          "ids": ids.tolist(), "loss_finite": True}))


    def evaluate(args, device, model):
        if not args.resume:
            raise ValueError("--mode eval requires --resume")
        if args.eval_batches < 1:
            raise ValueError("--eval-batches must be >= 1")
        train_id_max = args.samples - 1
        if args.eval_id_offset <= train_id_max:
            raise ValueError(
                f"eval IDs overlap training IDs: eval_start={args.eval_id_offset}, "
                f"train_end={train_id_max}"
            )
        _, loader = make_loader(
            args, device, seed=args.eval_seed, id_offset=args.eval_id_offset, decode_ms=0.0
        )
        model.eval()
        losses = []
        observed_ids = []
        with torch.no_grad():
            for i, (x, y, sample_ids) in enumerate(loader):
                if i == args.eval_batches:
                    break
                observed_ids.extend(int(x) for x in sample_ids)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda" and args.precision == "fp16",
                                             dtype=torch.float16):
                    logits = model(x)
                    loss = F.cross_entropy(logits.float().reshape(-1, args.vocab), y.reshape(-1))
                losses.append(float(loss))
        if not losses or not observed_ids:
            raise RuntimeError("eval loader produced no complete batch; increase --samples or reduce --batch")
        mean_loss = sum(losses) / len(losses)
        return {"status": "PASS" if math.isfinite(mean_loss) else "FAIL-MODEL",
                "eval_loss": mean_loss, "eval_batches": len(losses),
                "eval_seed": args.eval_seed, "eval_id_min": min(observed_ids),
                "eval_id_max": max(observed_ids), "train_id_range": [0, train_id_max],
                "disjoint_from_train": min(observed_ids) > train_id_max}


    def compare_nested(left, right, path="root", atol=1e-6, rtol=1e-5):
        if torch.is_tensor(left) and torch.is_tensor(right):
            if left.shape != right.shape or left.dtype != right.dtype:
                raise AssertionError(f"{path}: tensor metadata differs")
            if left.is_floating_point():
                if not torch.allclose(left, right, atol=atol, rtol=rtol):
                    delta = float((left.double() - right.double()).abs().max())
                    raise AssertionError(f"{path}: max_abs={delta}")
                return float((left.double() - right.double()).abs().max()) if left.numel() else 0.0
            if not torch.equal(left, right):
                raise AssertionError(f"{path}: integer/RNG tensor differs")
            return 0.0
        if isinstance(left, dict) and isinstance(right, dict):
            if left.keys() != right.keys():
                raise AssertionError(f"{path}: keys differ")
            return max((compare_nested(left[k], right[k], f"{path}.{k}", atol, rtol)
                        for k in left), default=0.0)
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right):
                raise AssertionError(f"{path}: length differs")
            return max((compare_nested(x, y, f"{path}[{i}]", atol, rtol)
                        for i, (x, y) in enumerate(zip(left, right))), default=0.0)
        if left != right:
            raise AssertionError(f"{path}: {left!r} != {right!r}")
        return 0.0


    def compare_checkpoints(args):
        if not args.checkpoint_a or not args.checkpoint_b:
            raise ValueError("--mode compare requires --checkpoint-a and --checkpoint-b")
        manifest_a = validate_checkpoint_file(args.checkpoint_a)
        manifest_b = validate_checkpoint_file(args.checkpoint_b)
        first = torch.load(args.checkpoint_a, map_location="cpu")
        second = torch.load(args.checkpoint_b, map_location="cpu")
        if first.get("step") != manifest_a["step"] or second.get("step") != manifest_b["step"]:
            raise RuntimeError("comparison payload step disagrees with validated manifest")
        current_torch = str(torch.__version__)
        if first.get("torch_version") != current_torch or second.get("torch_version") != current_torch:
            raise RuntimeError(
                f"comparison requires exact torch version {current_torch}; "
                f"saved={[first.get('torch_version'), second.get('torch_version')]}"
            )
        if first["trajectory_config"] != second["trajectory_config"]:
            raise AssertionError("trajectory configs differ")
        max_abs = compare_nested(
            {k: first[k] for k in ("model", "optimizer", "scaler", "python_rng", "torch_rng", "cuda_rng", "step")},
            {k: second[k] for k in ("model", "optimizer", "scaler", "python_rng", "torch_rng", "cuda_rng", "step")},
            atol=args.compare_atol, rtol=args.compare_rtol,
        )
        cfg = first["trajectory_config"]
        if args.eval_id_offset <= cfg["samples"] - 1:
            raise ValueError("fixed comparison sample must be outside the training ID range")
        eval_data = SyntheticTokens(
            1, cfg["seq"], cfg["vocab"], args.eval_seed, id_offset=args.eval_id_offset
        )
        x, y, sample_id = eval_data[0]
        x, y = x.unsqueeze(0), y.unsqueeze(0)
        next_losses = []
        for state in (first, second):
            model = TinyLM(cfg["vocab"], cfg["seq"], cfg["dim"], cfg["heads"], cfg["layers"])
            model.load_state_dict(state["model"], strict=True)
            model.eval()
            with torch.no_grad():
                logits = model(x)
                next_losses.append(float(F.cross_entropy(logits.reshape(-1, cfg["vocab"]), y.reshape(-1))))
        if not math.isclose(next_losses[0], next_losses[1], abs_tol=args.compare_atol, rel_tol=args.compare_rtol):
            raise AssertionError(f"next fixed-batch losses differ: {next_losses}")
        print(json.dumps({"status": "RESUME_EQUIVALENT", "step": first["step"],
                          "max_state_abs": max_abs, "next_eval_id": int(sample_id),
                          "next_loss_a": next_losses[0], "next_loss_b": next_losses[1]}))


    def train(args):
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device(args.device)
        seed_all(args.seed)
        out = Path(args.output)
        if out.exists() and any(out.iterdir()) and not args.resume:
            raise FileExistsError(f"refusing to overwrite {out}")
        out.mkdir(parents=True, exist_ok=True)
        dataset, loader, model, optimizer, scaler, use_amp = build(args, device)
        start_step = 0
        if args.resume:
            start_step = load_checkpoint(args.resume, model, optimizer, scaler, device, args)
        data_iter = iter(loader)
        for _ in range(start_step % len(loader)):
            next(data_iter)

        if args.mode == "eval":
            metrics = evaluate(args, device, model)
            atomic_json(out / "metrics.json", metrics)
            print(json.dumps(metrics))
            return
        if args.steps <= start_step:
            raise ValueError(f"training target --steps={args.steps} must exceed resume step={start_step}")

        prof = None
        prof_context = contextlib.nullcontext()
        if args.mode == "profile":
            activities = [ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(ProfilerActivity.CUDA)
            prof = profile(
                activities=activities,
                schedule=schedule(wait=1, warmup=2, active=8, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(out / "trace")),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            prof_context = prof

        losses, data_ms, events = [], [], []
        skipped = 0
        attempted_updates = start_step
        successful_updates = start_step
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        steady_start_step = max(start_step, args.warmup)
        steady_wall_start = None
        with prof_context:
            for step in range(start_step, args.steps):
                if step == steady_start_step:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    steady_wall_start = time.perf_counter()
                attempted_updates += 1
                t0 = time.perf_counter()
                try:
                    x, y, sample_ids = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    x, y, sample_ids = next(data_iter)
                data_elapsed = 1000.0 * (time.perf_counter() - t0)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                start_event = end_event = None
                if device.type == "cuda":
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                with record_function("forward_and_loss"):
                    with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                        logits = model(x)
                        loss = F.cross_entropy(
                            logits.float().reshape(-1, args.vocab), y.reshape(-1)
                        )
                        if args.inject_nan and step == args.warmup:
                            loss = loss * torch.tensor(float("nan"), device=device)
                with record_function("backward"):
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite loss at step {step}")
                    scaler.scale(loss).backward()
                with record_function("unscale_clip_optimizer"):
                    scaler.unscale_(optimizer)
                    if not finite_grads(model):
                        raise FloatingPointError(f"non-finite grad at step {step}")
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    did_skip = scaler.get_scale() < old_scale
                    skipped += int(did_skip)
                    if did_skip:
                        raise FloatingPointError(f"AMP skipped optimizer step at step {step}")
                    successful_updates += 1
                if device.type == "cuda":
                    end_event.record()
                    if step >= args.warmup:
                        events.append((start_event, end_event))
                if step >= args.warmup:
                    losses.append(float(loss.detach()))
                    data_ms.append(data_elapsed)
                if prof is not None:
                    prof.step()
                if step == start_step or (step + 1) % args.log_every == 0:
                    print(json.dumps({
                        "step": step + 1,
                        "loss": float(loss.detach()),
                        "grad_norm": float(grad_norm),
                        "scale": float(scaler.get_scale()),
                        "attempted_updates": attempted_updates,
                        "successful_updates": successful_updates,
                        "sample_id_first": int(sample_ids[0]),
                    }))

        if device.type == "cuda":
            torch.cuda.synchronize()
        steady_wall_seconds = (
            time.perf_counter() - steady_wall_start if steady_wall_start is not None else None
        )
        steady_steps = max(0, args.steps - steady_start_step)
        steady_tokens = steady_steps * args.batch * args.seq
        step_ms = [s.elapsed_time(e) for s, e in events]
        params = sum(p.numel() for p in model.parameters())
        metrics = {
            "status": "PASS",
            "mode": args.mode,
            "start_step": start_step,
            "end_step": args.steps,
            "seed": args.seed,
            "precision": args.precision,
            "batch": args.batch,
            "seq": args.seq,
            "parameters": params,
            "loss_last": losses[-1] if losses else None,
            "loss_scale_final": float(scaler.get_scale()),
            "skipped_steps": skipped,
            "attempted_updates": attempted_updates,
            "successful_updates": successful_updates,
            "step_ms_p50": percentile(step_ms, 0.50),
            "step_ms_p95": percentile(step_ms, 0.95),
            "data_ms_p50": percentile(data_ms, 0.50),
            "data_ms_p95": percentile(data_ms, 0.95),
            "e2e_timed_steps": steady_steps,
            "e2e_wall_seconds": steady_wall_seconds,
            "e2e_tokens_per_s": (
                steady_tokens / steady_wall_seconds
                if steady_wall_seconds is not None and steady_wall_seconds > 0 else None
            ),
            "peak_allocated_gb": (
                torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
            ),
            "peak_reserved_gb": (
                torch.cuda.max_memory_reserved() / 1e9 if device.type == "cuda" else 0.0
            ),
            "torch_version": str(torch.__version__),
        }
        if metrics["peak_reserved_gb"] >= 30.5:
            metrics["status"] = "FAIL-SYSTEM"
        save_checkpoint(out / "checkpoint.pt", model, optimizer, scaler, args.steps, args)
        atomic_json(out / "metrics.json", metrics)
        if prof is not None:
            sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
            print(prof.key_averages().table(sort_by=sort_key, row_limit=20))
        print(json.dumps(metrics, sort_keys=True))


    def parse_args():
        p = argparse.ArgumentParser()
        p.add_argument("--mode", choices=["unit", "train", "profile", "eval", "compare"], default="train")
        p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
        p.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
        p.add_argument("--output", default="runs/run")
        p.add_argument("--resume")
        p.add_argument("--steps", type=int, default=40)
        p.add_argument("--warmup", type=int, default=10)
        p.add_argument("--eval-batches", type=int, default=10)
        p.add_argument("--eval-seed", type=int, default=1000017)
        p.add_argument("--eval-id-offset", type=int, default=10000000)
        p.add_argument("--samples", type=int, default=8192)
        p.add_argument("--batch", type=int, default=2)
        p.add_argument("--seq", type=int, default=128)
        p.add_argument("--vocab", type=int, default=8192)
        p.add_argument("--dim", type=int, default=256)
        p.add_argument("--heads", type=int, default=8)
        p.add_argument("--layers", type=int, default=4)
        p.add_argument("--workers", type=int, default=0)
        p.add_argument("--decode-ms", type=float, default=0.0)
        p.add_argument("--lr", type=float, default=3e-4)
        p.add_argument("--clip", type=float, default=1.0)
        p.add_argument("--seed", type=int, default=17)
        p.add_argument("--log-every", type=int, default=10)
        p.add_argument("--inject-nan", action="store_true")
        p.add_argument("--checkpoint-a", default="")
        p.add_argument("--checkpoint-b", default="")
        p.add_argument("--compare-atol", type=float, default=1e-6)
        p.add_argument("--compare-rtol", type=float, default=1e-5)
        return p.parse_args()


    if __name__ == "__main__":
        arguments = parse_args()
        if arguments.mode == "unit":
            unit_test()
        elif arguments.mode == "compare":
            compare_checkpoints(arguments)
        else:
            train(arguments)
    PY

### 执行命令

文件保存后：

    cd week06-profile
    python -m py_compile src/lab.py
    sha256sum src/lab.py | tee manifests/code.sha256

### 预期观测（估算/示例，不是实测）

py_compile 无输出且返回 0；code.sha256 有一行 hash 和路径。

### 验收条件

代码无空实现或省略段；hash 已保存。

### 若失败，先看什么，再改什么

用 python -m py_compile 的行号修复语法；再检查是否复制了全部缩进；不进入 GPU。

### 当日证据清单

src/lab.py、code.sha256（待验证）。

## Day 3：CPU→单 batch→smoke→resume

### 为什么做

按低成本顺序证明 shape、loss、FP16 和完整恢复，不把正式训练当调试器。

### 输入与前置检查

    cd week06-profile
    test -f src/lab.py
    python -m py_compile src/lab.py

### 本日要创建/修改的文件

无；运行时创建 runs 下目录。

### 实现

使用 Day 2 已完整创建且 hash 固定的 src/lab.py。

### 执行命令

纯 CPU 单元测试：

    python src/lab.py --mode unit --device cpu

单 batch/最小 GPU 前反向：

    CUDA_VISIBLE_DEVICES=0 python src/lab.py --mode train --device cuda --precision fp16 --steps 1 --warmup 0 --batch 1 --seq 32 --vocab 256 --dim 64 --heads 4 --layers 1 --samples 32 --output runs/gpu_one

20-step smoke：

    CUDA_VISIBLE_DEVICES=0 python src/lab.py --mode train --device cuda --precision fp16 --steps 20 --warmup 5 --batch 2 --seq 128 --vocab 8192 --dim 256 --heads 8 --layers 4 --output runs/smoke20

resume 等价路径必须同时有 uninterrupted control；三条训练命令除分段边界/输出目录外轨迹配置完全相同：

    CUDA_VISIBLE_DEVICES=0 python src/lab.py --mode train --device cuda --precision fp16 --steps 20 --warmup 5 --batch 2 --seq 128 --vocab 8192 --dim 256 --heads 8 --layers 4 --output runs/equiv_control
    CUDA_VISIBLE_DEVICES=0 python src/lab.py --mode train --device cuda --precision fp16 --steps 10 --warmup 5 --batch 2 --seq 128 --vocab 8192 --dim 256 --heads 8 --layers 4 --output runs/equiv_resume
    CUDA_VISIBLE_DEVICES=0 python src/lab.py --mode train --device cuda --precision fp16 --steps 20 --warmup 5 --batch 2 --seq 128 --vocab 8192 --dim 256 --heads 8 --layers 4 --resume runs/equiv_resume/checkpoint.pt --output runs/equiv_resume
    python src/lab.py --mode compare --device cpu --checkpoint-a runs/equiv_control/checkpoint.pt --checkpoint-b runs/equiv_resume/checkpoint.pt | tee manifests/resume-equivalence.json

第三条允许写入同一非空目录，因为提供了 `--resume`；compare 不训练，逐层比较 model/optimizer/scaler/RNG/step，并在独立固定样本上比较 next loss。

### 预期观测（估算/示例，不是实测）

每 10 step 打印 step、loss、grad_norm、scale、sample_id_first；目录出现 metrics.json 和 checkpoint.pt。数值不得预填。

### 验收条件

- CPU 输出 status=PASS 与 logits [2,8,32]；
- GPU loss/grad 有限，skipped 不持续；
- smoke metrics status=PASS；
- resume 的 start_step=10、end_step=20，compare 输出 `RESUME_EQUIVALENT`；
- checkpoint 包含 model/optimizer/scaler/RNG/step/config。

### 若失败，先看什么，再改什么

CUDA 错先查 CUDA_VISIBLE_DEVICES/nvidia-smi；shape 错回 CPU unit；非有限用同配置 fp32；resume 错用：

    python - <<'PY'
    import torch
    from src.lab import validate_checkpoint_file
    path="runs/resume/checkpoint.pt"
    validate_checkpoint_file(path)
    s=torch.load(path,map_location="cpu")
    print(sorted(s),s["step"],s["config"])
    PY

### 当日证据清单

CPU 输出、gpu_one/smoke20、control/resume 的 metrics、checkpoint key 审计与 `manifests/resume-equivalence.json`（均待验证）。

## Day 4：短 Profile 与显存 envelope

### 为什么做

先定位 top op，再只改 batch 或 sequence 建立安全边界。

### 输入与前置检查

Day 3 smoke 必须 PASS；检查内部磁盘：

    cd week06-profile
    test -f runs/smoke20/metrics.json
    df -h .

### 本日要创建/修改的文件

无；程序创建 trace 与 metrics。

### 实现

复用 Day 2 profiler schedule：wait=1、warmup=2、active=8。正式吞吐另跑 train 模式。

### 执行命令

    CUDA_VISIBLE_DEVICES=0 python src/lab.py --mode profile --steps 15 --warmup 3 --batch 2 --seq 256 --vocab 8192 --dim 512 --heads 8 --layers 6 --output runs/profile_short
    CUDA_VISIBLE_DEVICES=0 python src/lab.py --mode train --steps 120 --warmup 20 --batch 2 --seq 256 --vocab 8192 --dim 512 --heads 8 --layers 6 --output runs/profile_off

显存单变量点：

    CUDA_VISIBLE_DEVICES=0 python src/lab.py --mode train --steps 15 --warmup 5 --batch 1 --seq 256 --dim 512 --heads 8 --layers 6 --output runs/mem_b1_t256
    CUDA_VISIBLE_DEVICES=0 python src/lab.py --mode train --steps 15 --warmup 5 --batch 2 --seq 256 --dim 512 --heads 8 --layers 6 --output runs/mem_b2_t256
    CUDA_VISIBLE_DEVICES=0 python src/lab.py --mode train --steps 15 --warmup 5 --batch 1 --seq 512 --dim 512 --heads 8 --layers 6 --output runs/mem_b1_t512

只有前三点安全，才继续 batch=4 或 seq=1024。约 100M 正式候选从 batch=1：

    CUDA_VISIBLE_DEVICES=0 python src/lab.py --mode train --steps 40 --warmup 20 --batch 1 --seq 256 --vocab 16384 --dim 768 --heads 12 --layers 12 --output runs/model_100m_probe

### 预期观测（估算/示例，不是实测）

profile_short 打印 top 20 op 并产生 trace 子目录；metrics 同时有 GPU-only `step_ms_p50/p95`、端到端 `e2e_wall_seconds/e2e_tokens_per_s` 与 peak allocated/reserved。约 100M 只是参数量级估算，实际 parameters 字段为准。

### 验收条件

- trace 仅 8 active steps；
- profiler on/off 分开；
- 至少一个 batch 轴和一个 sequence 轴；
- reserved 达 28–30GB 即停止，不制造 OOM；
- top op 可映射到 transformer_block/forward/backward。

### 若失败，先看什么，再改什么

trace 过大先删减 active（按公司内部清理规范，不在此手册下删除）；OOM 从最后安全点恢复；reserved≫allocated 看：

    python - <<'PY'
    import torch
    print(torch.cuda.memory_summary())
    PY

### 当日证据清单

profile on/off metrics、内部 trace、显存三点、参数量与停止原因（待验证）。

## Day 5：有界正式运行、独立 eval、A/B 与失败注入

### 为什么做

把“定位”闭环到因果 A/B、可加载 checkpoint 和已验证失败保护。

### 输入与前置检查

选择 Day 4 最后安全配置；下列示例使用小配置，100M 候选需先通过 probe。

### 本日要创建/修改的文件

人工创建 runs/decision.md，内容含假设、唯一改动、不变量、结果、状态。

### 实现

A/B 用 decode_ms=1 模拟数据开销，只改变 workers。正式真实数据时该开关应替换为已确认的数据瓶颈变量。

先创建结论记录；运行后只用实际证据替换 `PENDING`：

    cat > runs/decision.md <<'MD'
    # Week06 单卡优化决策
    - 假设：当每样本解码延迟固定为 1ms 时，4 workers 可隐藏部分 data time。
    - 唯一主变量：workers=0 vs workers=4。
    - 固定项：seed=17、模型/数据公式、batch/seq/precision、120 steps、20 warmup。
    - 三次重复的 metrics 路径：runs/ab_A_1..3 与 runs/ab_B_1..3。
    - 正式判定字段：e2e_tokens_per_s；GPU-only step_ms 与 data_ms 只用于归因。
    - 实际中位数/方差：PENDING。
    - 状态：PENDING；未运行前不得写 PASS。
    MD

### 执行命令

有界正式训练与独立 eval：

    CUDA_VISIBLE_DEVICES=0 python src/lab.py --mode train --steps 300 --warmup 20 --batch 2 --seq 256 --vocab 8192 --dim 512 --heads 8 --layers 6 --seed 17 --output runs/formal_s17
    CUDA_VISIBLE_DEVICES=0 python src/lab.py --mode eval --precision fp16 --batch 2 --seq 256 --vocab 8192 --dim 512 --heads 8 --layers 6 --seed 17 --eval-seed 1000017 --eval-id-offset 10000000 --resume runs/formal_s17/checkpoint.pt --output runs/eval_s17

A/B 各三次：

    for r in 1 2 3; do
      CUDA_VISIBLE_DEVICES=0 python src/lab.py --mode train --steps 120 --warmup 20 --log-every 120 --workers 0 --decode-ms 1 --seed 17 --output "runs/ab_A_${r}"
      CUDA_VISIBLE_DEVICES=0 python src/lab.py --mode train --steps 120 --warmup 20 --log-every 120 --workers 4 --decode-ms 1 --seed 17 --output "runs/ab_B_${r}"
    done

安全失败注入：

    CUDA_VISIBLE_DEVICES=0 python src/lab.py --mode train --steps 12 --warmup 5 --batch 1 --seq 32 --vocab 256 --dim 64 --heads 4 --layers 1 --inject-nan --output runs/inject_nan

该命令预期非零退出，不是 PASS run。

### 预期观测（估算/示例，不是实测）

formal/eval 均生成 metrics；eval 应报告 ID 范围与训练范围不相交。A/B 从六个 `e2e_tokens_per_s` 计算中位数与方差，不能拿 GPU-only `step_ms` 代表 workers 改动后的吞吐；inject_nan 预计报 non-finite loss/grad at step 5，且不生成成功 checkpoint。

### 验收条件

- 正式 run 有界在 300 steps；
- eval 能 strict load、loss finite、`disjoint_from_train=true`；
- A/B 三次仅 workers 不同，吞吐提升 ≥10% 或形成可信负结果；
-任何 loss/grad/skip/显存回归则回滚；
- NaN 注入被即时阻断；
-结论明确标“待用户运行验证”直到真实执行。

### 若失败，先看什么，再改什么

先比六个 resolved CLI/metrics 的不变量和端到端计时步数，再用 GPU event/data p95 归因；eval 失败查 checkpoint config、eval seed 与 ID 区间；NaN 未捕获则回 Day 2 finite_grads。

### 当日证据清单

formal checkpoint/metrics、独立 eval、A/B 六组、失败注入 stderr、decision.md（待验证）。

## 3. 结果阅读与故障恢复

- loss 缓慢下降：只说明合成 next-token 训练可优化；不代表语言能力。
- loss 突升且 scale 下降：检查 overflow、输入和 grad norm。
- p95≫p50：检查 workers、共享负载、周期 I/O。
- peak reserved≥30.5GB：FAIL-SYSTEM，回最后安全 batch/seq。
- OOM：停止上探，从最后完整 checkpoint 以更安全配置重启；不能靠循环 empty_cache 掩盖。
- 中断：仅从原子 checkpoint.pt 恢复；若只有 .tmp，拒绝加载。
- trace 损坏：不影响 checkpoint，但 profiler 结论为 INCONCLUSIVE，重跑短 trace。

## 4. 发布验收

只有以下证据齐全才 PASS：

1. 环境、pip freeze、代码 SHA；
2. CPU unit、单 batch、20-step smoke；
3. checkpoint/resume 与独立 eval；
4. profiler on/off；
5. 两轴显存 envelope；
6. 三次单变量 A/B；
7. NaN 失败注入；
8. 公司产物未外带。

官方参考：[PyTorch 2.1 Profiler](https://docs.pytorch.org/docs/2.1/profiler.html)、[AMP](https://docs.pytorch.org/docs/2.1/notes/amp_examples.html)、[CUDA memory](https://docs.pytorch.org/docs/2.1/notes/cuda.html#cuda-memory-management)。核验日期：2026-09-03。
