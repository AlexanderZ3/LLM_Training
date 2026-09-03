# Week 04 基础篇：TinyFlowPolicy 的目标场、动作窗口与闭环证据

> 本篇是待执行教程，不是实测报告。公司 `V100 32GB`、个人 `5070 Ti 16GB` 为用户自述待核验；公司PyTorch 2.1不升级。V100主路径是FP16 AMP + GradScaler，通常不支持原生BF16，禁用BF16/TF32/官方FlashAttention-2。公司代码、数据、日志、轨迹、图片、trace、checkpoint和性能数字不得导出。

## 1. 本周问题与最终产物

本周从离散token生成跨到连续动作生成：

```text
episode → observation k + action chunk k:k+H + padding mask
→ train-only normalization
→ x0~N(0,I), t~U(0,1), xt=(1-t)x0+t x1
→ conditional velocity vθ(xt,t,condition)
→ masked flow-matching MSE
→ Euler ODE 0→1 生成动作chunk
→ offline action误差 + synthetic closed-loop rollout
```

最终交付：确定性point-mass数据生成器、window/mask、同预算MSE behavior-cloning baseline、约10M参数TinyFlowPolicy、FP32 overfit、FP16、EMA、完整resume、Euler采样、ID/OOD rollout和condition shuffle/zero对照。

Week05会复用episode/schema/window并把隐含假设变成数据合同；Week09会用本周的公平预算比较diffusion与flow。

## 2. 数据与模型来源、许可和版本

技术链接核验日期：**2026-09-03**。

| 对象 | 精确版本 | 本周用途 | 许可/边界 |
|---|---|---|---|
| `point-mass-v1` | 本教程 `make_data.py` + seed + config SHA-256 | 主数据；零下载、可恢复已知动力学 | 用户本地生成，无外部数据许可；只能证明系统/玩具控制，不代表机器人 |
| TinyFlowPolicy | 本教程代码从随机初始化 | 主模型；无外部checkpoint | 代码身份由文件hash记录 |
| Flow Matching论文 | arXiv `2210.02747` | 线性probability path定义 | 论文引用；不作为运行依赖 |
| LeRobot PushT | `lerobot/pusht@7628202a2180972f291ba1bc6723834921e72c19`，`meta/info.json` 声明 format `v3.0` | 可选schema transfer，不是本周主训练 | dataset card标MIT；公司审批；只下metadata/parquet可避免视频 |
| LeRobot代码 | commit `fbb811fca92504439792b97d216f0d00c2268382` | 只读当前格式参考 | Apache-2.0；当前版本要求较新Python/torch，**不能安装进公司PyTorch2.1环境** |
| Diffusion Policy | commit `5ba07ac6661db573af695b419a7947ecb704690f` | action chunk/PushT背景 | MIT；只读参考 |

主路径选合成数据是为了让从空目录可执行，不把仿真器/视频编解码依赖混入算法门。使用公开PushT只构成transfer结果；其观测/action物理语义必须从固定dataset card与环境源码核验，不能把2D向量随口写成“力”或“速度”。

## 3. point-mass-v1 明确合同

每个episode由seed确定，长度24–64：

- `state=[px,py,vx,vy]`，float32，位置/速度为无量纲toy坐标；
- `goal=[gx,gy]`，float32；
- `action=[ax,ay]`，float32，是截断到`[-1,1]`的加速度命令；
- `dt=0.1`；动力学 `v_{k+1}=0.8v_k+0.2a_k`，`p_{k+1}=p_k+dt·v_{k+1}`；
- expert `a_k=clip(2(goal-p_k)-0.3v_k,-1,1)`；
- success：episode任一步 `||p-goal||<0.15`；
- episode ID、seed、length进入manifest；train/dev/OOD按episode分，不按window随机切。

这一定义让action lag、单位、动力学和success都可计算。它只是toy，不映射真实机器人关节。

## 4. Window、padding、mask 与 normalization

在时刻k，condition取`state[k]`和goal，目标chunk为`action[k:k+H]`。episode尾部不足H时右pad 0，`mask[h]=1`表示真实动作。

worked example：episode actions `[10,11,12]`、`H=4`：

```text
k=0: [10,11,12,0], mask=[1,1,1,0]
k=1: [11,12,0,0], mask=[1,1,0,0]
k=2: [12,0,0,0], mask=[1,0,0,0]
```

绝不跨episode补后续动作。`actions [B,H,Da]`、`mask [B,H]`，loss mask扩到`[B,H,1]`广播到Da。

所有均值/标准差只用train episode的真实位置：

$$\hat a=(a-\mu_a)/(\sigma_a+\epsilon).$$

padding在normalize前后都不应影响统计或loss。推理输出必须反归一化后才进入toy环境。stats连同data hash进入checkpoint；换stats不能继续同一run。

## 5. Flow matching 的线性路径

令数据动作chunk `x1∈R^{H×Da}`，噪声 `x0~N(0,I)`，`t~Uniform(0,1)`：

$$x_t=(1-t)x_0+t x_1,$$

对t求导：

$$u_t=\frac{dx_t}{dt}=x_1-x_0.$$

模型接收 `xt,t,condition`，预测velocity `vθ`：

$$\mathcal L=\frac{\sum_{b,h,d}m_{b,h}(v_{bhd}-u_{bhd})^2}
{D_a\sum_{b,h}m_{b,h}}.$$

分母是有效标量元素数，不是固定`BHDa`。否则短episode padding比例会改变loss scale。`target=(x1-x0).detach()`；不需要对ODE反传，也不需要在训练时求解轨迹。

具体shape：`B=4,H=16,Da=2,Ds=4,Dg=2`：

| tensor | shape |
|---|---:|
| state/goal condition | `[4,6]` |
| x0/x1/xt/target/pred | `[4,16,2]` |
| t | `[4,1]`，广播为`[4,1,1]` |
| mask | `[4,16]`→`[4,16,1]` |
| flattened action input | `[4,32]` |

若mask有效步数为`[16,10,4,1]`，分母是`(31)×2=62`，而不是`4×16×2=128`。

## 6. 时间与条件表示

模型必须知道t；同一个空间点在不同时刻的合理velocity可不同。本周使用固定sin/cos embedding：

$$\phi(t)=[\sin(2\pi f_i t),\cos(2\pi f_i t)]_{i=1}^{K}.$$

condition为normalized state+goal。输入拼成 `[condition, flatten(mask⊙xt), mask, φ(t)]`，投影到hidden，再过残差MLP blocks，输出 `[B,H,Da]`。把padding latent归零仍不够：mask本身必须进入模型，才能区分“真实零动作”和“无效位置”。正式配置hidden=1024、5个每个含两层1024线性的residual block，总参数据代码实算，目标约10M。

动作chunk一次性联合生成，MLP没有causal attention；若换Transformer，action token通常可以双向交互。loss mask与attention padding mask是两种不同语义。

condition对照：correct、在batch内跨episodeshuffle、全零。若三者误差近似相同，模型可能忽略condition；不得仅靠更大模型掩盖。

## 7. ODE采样器

推理从 `z0~N(0,I)` 开始解：

$$\frac{dx}{dt}=v_\theta(x,t,c),\quad t:0\rightarrow1.$$

Euler用N步、`Δt=1/N`：

$$x_{i+1}=x_i+\Delta t\,v_\theta(x_i,t_i,c).$$

恒定velocity `v=c` 的N步结果应在浮点容差内等于`x0+c`；线性field可用解析/高步数oracle。比较5/10/20/50步必须固定初始noise和condition；否则随机差异会伪装solver差异。

更多ODE步通常降低离散误差，却线性增加模型forward/延迟；训练loss低不保证Euler少步就好。

## 8. 为什么需要 MSE BC baseline

BC直接预测chunk：`fθ(condition)→x1`，同样的window、normalization、mask、train/dev episode、optimizer update预算。它是数据/head的最小oracle：若BC连128 windows都不能overfit，flow更复杂的noise/time不是首要怀疑对象。

公平比较至少锁：train windows、seed集合、成功updates、global batch、模型容量量级、evaluation IDs、动作反归一化和rollout动力学。Flow额外报告ODE steps和推理延迟。

## 9. EMA、AMP 与成功更新时钟

EMA：`θema←βθema+(1-β)θ`，只在optimizer实际成功step后更新；shadow保持FP32，保存weights/decay/update_count。eval必须显式选raw或EMA，用临时state或独立模型，不能永久覆盖训练权重。

V100 FP16：autocast算model forward，masked MSE用`pred.float()`与`target.float()`；`scale→backward→unscale→clip→step→update`。若scale回退，本次不推进scheduler、EMA、successful_step或effective samples。

显存由约10M参数/grad/Adam、activations、batch与ODE eval组成。训练一步只有一次model forward；eval一次chunk含N次forward，因此评测也可能成为时间瓶颈。

## 10. Checkpoint/resume合同

保存：model、optimizer、LR时钟、GradScaler、EMA、successful/attempted step、samples seen、Python/NumPy/torch/CUDA RNG、window sampler permutation/cursor、noise/t generator、normalization stats、dataset/config/code hash。

`40 continuous`与`20+resume+20`必须使用同一总`schedule_steps=40`；固定step21 window IDs、x0、t的hash后比较loss。若分段第一段将总步数错误设20，cosine LR已不同，不能称resume测试。

## 11. Offline与closed-loop不是同一问题

offline masked action MSE只说明demo分布上动作接近。closed-loop状态由模型动作改变，误差会进入新状态；小MSE也可能导致overshoot/oscillation/wrong direction。

toy rollout固定100个episode seed，报告success与Wilson 95%区间：

$$\hat p_W=\frac{\hat p+z^2/(2n)}{1+z^2/n},\quad
h=\frac{z}{1+z^2/n}\sqrt{\frac{\hat p(1-\hat p)}n+\frac{z^2}{4n^2}},\ z=1.96.$$

OOD只改一个轴，如goal范围从`[-1,1]`扩到`[-1.5,1.5]`；不要同时改动力学与初始状态。

## 12. 正确性不变量

1. episode/window不跨界，mask/padding微型例子全过；
2. normalization只看train有效值；
3. 手工x0/x1/t的xt/target准确；
4. masked loss分母为有效标量数，改padding value不影响loss；
5. 恒定field Euler测试通过；
6. BC先overfit 128 windows；
7. Flow fixed noise/time后random noise/time两级overfit；
8. correct condition优于shuffle/zero，否则“未证明条件依赖”；
9. FP32 smoke后才FP16，200 step finite且skip受控；一次独立受控overflow必须证明GradScaler skip后仍能恢复；
10. EMA仅随成功update，受控skip时参数/EMA不变，resume后raw/EMA/RNG/sampler连续。

## 13. 失败树

| 症状 | 第一检查 | 根因 | 修复/回归 |
|---|---|---|---|
| BC不overfit | 同128 IDs、mask/normalize | window/head/LR错误 | 固定batch，逐项打印，BC gate |
| fixed flow可、random不可 | x0/t hash与范围 | time embed/noise scale | 手工target与t端点测试 |
| flow loss降、sample差 | velocity/noise目标与0→1方向 | solver符号/反归一化/EMA | 恒定field+固定noise对照 |
| 短episodeloss异常 | denominator与mask维 | 用BHD固定分母 | padding替换不变性测试 |
| condition对照无差 | shuffle是否跨episode | 数据先验/condition断路 | grad与输入hash；换有条件必要性的子集 |
| FP16 overflow | 同batch FP32 | loss reduction FP16/scale/grad | FP32 masked MSE、正确clip顺序 |
| EMA漂移 | successful step计数 | skip也update | 人为overflow后EMA hash不变 |
| resume偏离 | step21 IDs/x0/t/LR | sampler/noise RNG/总schedule漏存 | 补全状态，重跑40对照 |
| offline好rollout差 | first divergence state | distribution shift/控制累积 | 错误桶与OOD，不改metric掩盖 |

## 14. Teach-back（先闭卷）

1. 从`xt=(1-t)x0+tx1`推导velocity target。
2. 手算有效步数31、Da=2时masked MSE分母。
3. 解释模型为何必须看t。
4. 写恒定field的Euler结果。
5. 为什么先BC overfit，再flow fixed-noise，再random-noise？
6. EMA为何不能在overflow skip时更新？
7. offline MSE低为何不保证rollout success？
8. 列出resume必须冻结的三类RNG状态。

## 15. 官方一手资料

技术链接核验日期：**2026-09-03**。

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [LeRobot official repository](https://github.com/huggingface/lerobot)
- [LeRobot PushT dataset](https://huggingface.co/datasets/lerobot/pusht)
- [Diffusion Policy official repository](https://github.com/real-stanford/diffusion_policy)
- [PyTorch 2.1 AMP examples](https://pytorch.org/docs/2.1/notes/amp_examples.html)
