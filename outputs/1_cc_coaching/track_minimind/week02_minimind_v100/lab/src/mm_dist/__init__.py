"""mm_dist — Week M02 的 MiniMind 分布式教学包。

目标运行环境：公司 8x V100 + PyTorch 2.1.0（只用 2.1 已有的 API）。
本机（Windows / CPU / torch 2.14）只用于 py_compile、gloo 2 进程逻辑验证与 CPU 单测。

模块：
    common       —— 分布式初始化、dtype 守卫、seed、模型/数据构建、JSONL 日志
    train_ddp    —— 单卡/多卡 DDP 统一入口 + fixed-global-batch 等价检查 + 日志比对
    train_fsdp   —— FSDP1 三种 sharding 策略对照（需要 CUDA）
    ckpt_reshard —— 分片 checkpoint 保存 / 跨 world_size 恢复 / loss 连续性 verify
    ep_moe       —— 教学版专家并行（all_to_all_single dispatch/combine）
    router_stats —— router bias 注入、每专家负载统计、bias / aux_coef 扫描
    grpo_roles   —— GRPO 三种角色放置（replicate / policy_fsdp / split_roles）
    faults       —— 故障注入（kill rank）与 NCCL 超时设置助手
"""

__all__ = [
    "common",
    "train_ddp",
    "train_fsdp",
    "ckpt_reshard",
    "ep_moe",
    "router_stats",
    "grpo_roles",
    "faults",
]

__version__ = "0.1.0"

# 目标机版本；probe/README/日志都引用这两个常量
TARGET_TORCH_VERSION = "2.1.0"
MINIMIND_COMMIT = "7a6fddd63a30c06b2fdd5fac4089922b29bc841b"
