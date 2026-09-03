# Week 03 拷问篇：V100 / Topology / NCCL / FP16

> 闭卷作答，候选拓扑一律先标“待现场核验”。不得查看答案篇。

## Recall

1. **W03-R01**：driver、`torch.version.cuda`、`nvcc` 分别是什么？
2. **W03-R02**：V100 的架构/compute capability、FP16 与 BF16 能力边界是什么？
3. **W03-R03**：解释 NV1、NV2、PIX、SYS 的拓扑含义。
4. **W03-R04**：allocated 与 reserved 显存分别是什么？

## Explain

5. **W03-E01**：为什么 8×32GB 的 DDP 不是单进程可透明使用的 256GB？
6. **W03-E02**：为什么一个 SYS pair 不推出 8 卡 all-reduce 必须走 SYS？
7. **W03-E03**：algbw 与 busbw 各回答什么，为什么不能只看一个？
8. **W03-E04**：为什么硬件支持 FP16 仍可能不比 FP32 快？

## Apply

9. **W03-A01**：`CUDA_VISIBLE_DEVICES=4,6` 时 local rank 0/1 如何映射？你会打印哪些证据？
10. **W03-A02**：设计 NV2/NV1/SYS pair 的公平 collective 测试表。
11. **W03-A03**：ring all-reduce 在 N=8、algbw=假设值 X 时，写 busbw 换算式；说明适用边界。
12. **W03-A04**：设计对齐/未对齐 hidden shape 的 FP16 Tensor Core 实验。

## Debug

13. **W03-D01**：`nvidia-smi` 可见 8 卡，但 GPU5 FP16 correctness 失败；下一步是什么？
14. **W03-D02**：NCCL 8 卡 hang，强制禁用 P2P 后能跑；为什么不能就此关闭 P2P？
15. **W03-D03**：两个四卡组带宽差 25%，如何区分拓扑、CPU/DataLoader、共享负载与测量错误？
16. **W03-D04**：出现 hwloc sysfs 警告但 NVLink 测试正常，应怎样记录与处置？

## Design

17. **W03-G01**：设计一份未来 11 周可引用的 Hardware Card schema。
18. **W03-G02**：设计从单卡到八卡的 correctness→performance 晋级门。
19. **W03-G03**：在禁止导出日志/拓扑的环境中，如何让结论可复核又不外泄？
20. **W03-G04**：设计一个安全的 GradScaler overflow/recovery 故障实验。

## Trade-off

21. **W03-T01**：让 NCCL 自动选算法与手工固定 Ring/Tree 的权衡是什么？
22. **W03-T02**：nccl-tests 与真实 Transformer benchmark 分别能证明什么、不能证明什么？
23. **W03-T03**：性能模式的非确定性与严格 deterministic 模式如何取舍？
24. **W03-T04**：为何公司 PyTorch 2.1 不强行升级，即使新版本 profiler/kernel 更好？

## 评分

每题 0–4，总分 96；`≥77` 且所有“先 correctness、再性能”的题无安全性错误才通过。把用户自述拓扑说成实测、声称 V100 不支持 FP16、建议越权读取/导出任何一项均直接退回整改。

