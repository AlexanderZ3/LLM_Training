# 00_INPUT.md - 高频具体项目任务入口

## 1. 最新输入需求区

```text
处理 00_INPUT.md 顶部最新输入需求块。
```

路由说明：本仓库有两套独立的智能体系统。`AGENT: codex` 由 Codex 按 `AGENTS.md` 处理；`AGENT: cc` 由 Claude Code 按 `CLAUDE.md` 处理（命令 `/cc-input`）；`AGENT: both` 两边各自处理并各自归档。缺省为 `codex`。归档标题以 `[codex]` / `[cc]` 标明处理方。

### 新需求填写区

```text
<!-- VOICE_ROUTER:INPUT:BEGIN -->
TYPE: TASK
DEPTH: AUTO
MODE: COACH
AGENT: codex

<!-- 在这里填写下一条需求 -->
<!-- VOICE_ROUTER:INPUT:END -->
```

---

## 2. 历史归档区

### 2026-09-08 [cc] - Week M03 纯 V100 周包 + 量化/RL 扩展 + 路径合同

- 处理状态：已完成。结构校验 `PASS`（0 error 0 warning，口试 27 题/27 答）；`pytest lab/tests -q` → `368 passed, 6 skipped`；`compileall` 退出码 0。
- 需求 1–2（开环分工）：`CLAUDE.md` 第 2、3.1、7 节改写——本机 conda 环境改为用户指定的 `rfm`，删除已失效的 `ResearchAgentPy310`/`D:` 盘路径与 5070 Ti、H100 执行路径，写明"唯一 GPU 环境是公司内网 8×V100，cc 交付是开环的"。
- 需求 3（路径合同）：不下载任何数据。生成 `datasets/`（22 文件精确字节清单 + 两个数据组目录名）、`weights/`、`runs/` 三份 README 与目录骨架，代码全部经 `lab/src/mm_v100/paths.py` 取路径，另给 `check_data_layout.py` / `check_weights_layout.py` 两个校验脚本（四种状态与两种内网失败模式均已在本机构造验证）。
- 需求 4（作废两步走）：新建 `outputs/1_cc_coaching/track_minimind/week03_minimind_v100_Only/`，自包含、不依赖 M01/M02 的 Gate 或产物；`00_TRACK_README.md` 把 M01/M02 标注为已作废。
- 需求 5（QAT 扩展）：`06_QUANT_LOWBIT.md` + `lab/src/mm_quant/`，Q1–Q6 六个实验。按联网核验的硬件事实定位为"误差预算 + 内存带宽"，不做提速类实验。
- 需求 6（RL 扩展）：`07_RL_LOWPRECISION.md` + `lab/src/mm_rl/`，R1–R6 六个实验，围绕"低精度下两个几乎相等的对数概率相减"。
- 需求 7（内网提醒）：`08_INTRANET_SETUP.md`，含以 torch 2.1.0 为固定点的版本相容矩阵、constraints 钉死写法、离线 wheelhouse 四参数约束、HF 离线变量的当前正确写法、七行踩坑速查。
- 详情见 `01_PROGRESS.md` 的 2026-09-08 [cc] 记录（含构建期发现并修复的 5 处交付缺陷）。

原始输入：

```text
TYPE: TASK
DEPTH: AUTO
MODE: COACH
AGENT:

针对这个文件夹中的代码， 以后会这样安排：  
1， 因为这台电脑可以连接到外网和cc，所以代码分析和修改会是在这里， 但是这里的电脑没有gpu， 无法验证， 只能是开环的一个输出； 
2， 之后的跑代码和采集profiling等等都在另外的v100上面做，那个服务器只能接受，不能输出，   
3， 数据集我已经下载好了， 我会将其拷贝到服务器上的相应位置， 在这个文件夹中我现在希望你做的是不用针对去下载数据集，但是在相应位置你要生成这些文件夹或者文件的名字， 这样你这边的程序中的路径引用就是和V100服务器上完全对应的。  
4， 现在我无法touch到5070ti的资源了， 所以你在 C:\Users\zzz_7893\Desktop\0_Projects\LLM_Training\outputs\1_cc_coaching\track_minimind  这个文件夹中不用管之前的两个week的两部走计划，   
直接给我出一个 week03_minimind_v100_Only子文件夹，内容也和名字一样，就是只在v100上验证现在所有的要掌握的内容， 当然除非v100架构没有的就不用说了， 基本该学习的都要学到位； 
5， 我之后接下来要做QAT， 一个超大模型的8比特量化， 所以最好给我一些这方面的扩展也蛮好的，
6， 之前规划的RL的后训练啥的也尽量多扩展一些， 我在那台V100上慢慢配置环境就好， 我之后也有用RL后训练的需求，很可能还是会低精度+RL一起 训练； 
7， 一些补充， 我的那台V100是一个内网， 可以pip intsall  也可以git clone  但是有些包不是那么容易，尤其有依赖的那种，下载权重也得从外面下好再考进来， 总之你你注意一下，有些环境配置需要提醒我注意的，可以稍微提醒一下。   

现在综合来看重新出一个周计划。

<!-- 在这里填写下一条需求 -->
```


### 2026-09-05 [cc] - conda 环境 + MiniMind 两周课程

- 处理状态：已完成。
- 需求 1：本机 Python 统一走 conda 环境 `ResearchAgentPy310`，规则写入 `CLAUDE.md` 3.1 节，入口 `.claude/scripts/cc_py.ps1`，已装 CPU 版 torch 与 pytest。
- 需求 2：新建轨道 `outputs/1_cc_coaching/track_minimind/`，两个周包共 103 个文件、1.3 MB。M01 单机 5070 Ti 全链路，M02 公司八卡 V100 分布式。两周结构校验 `PASS`；M01 测试 42 通过，M02 测试 73 通过 1 跳过。
- 详情见 `01_PROGRESS.md` 的 2026-09-05 [cc] 记录。

原始输入：

```text
TYPE: TASK
DEPTH: AUTO
MODE: COACH
AGENT: claude  code

1. 以后使用python就用conda环境ResearchAgentPy310
路径是   D:\Software\Large\Anconda\envs\ResearchAgentPy310

2. 根据D:\zz\00_RealProjects\0_LLM_Training\input_info\minimind_5070ti_v100.md
文档，打造两周课程，一个给单机5070ti的训练， 第二周上八卡V100
```

### 2026-09-03 - 14 周 LLM 训练教程与陪跑教练

- 处理状态：已完成；文档/静态发布门 PASS，目标环境执行门 PENDING。
- 补充要求已并入验收：Foundations/Lab 必须写成详细操作指南，具体到数据、模型、下载、训练、完整代码、观测、排错与 Gate，不能只是 roadmap。
- 主要交付：`outputs/0_basic_training/`、`outputs/00_2026-09-03_教练系统实施规划.md`、`outputs/01_2026-09-03_技术事实与兼容性核验.md`、`outputs/02_2026-09-03_详细教程质量审查报告.md`。

原始输入：

```text
TYPE: TASK
DEPTH: AUTO
MODE: COACH

这个项目主要是记录和学习顶尖开源的LLM大模型架构：
 算法层分析，以及从算法和infra联合的视角分析。

现在我要从零开始花14周时间巩固最基础的深度学习和大模型训练，
从一次类似nanpGPT这样的小项目完全做起， 要理解其中的每一项，
我希望你是这样的一个私人教练， 帮助我去成为一个训练专家，理解
每一个训练的设计，以及log， training loss 曲线怎么看等等。 我要知道最细的细节，
不只是宏观架构上在别人的代码基础上可以修改部分算法设计， 而是从宏观架构上成为一个资深的
大模型训练的架构师， 我要之后能从这个视角去深度的掌握模型训练，进而帮助真正的高价值垂直领域
类似金融，投行这些领域训练专有的大模型。

我现在的输入是  一份12周的overall 学习计划，D:\zz\00_RealProjects\0_LLM_Training\input_info\Overall_robot_wam_va_vla_finance_12_week_plan_v100_fp16_2026-09.md，
然后之后我在这个基础上增加了两周的训练， 总共还有十四周单独的拆分计划存在  D:\zz\00_RealProjects\0_LLM_Training\input_info

我希望你能把十四周的每一种计划都扩充成完全可以照着做的教程，每一份教程包含： 一份详细的基础知识点，
一份根据实践清单（包含执行哪个项目，安装哪些包， 使用哪些工具看loss曲线或者分析算子性能等等，一定要
非常细节和可操作性），一份拷问我的思考清单（模拟资深工程师，架构师问我各种细节为什么那样设计，
有什么优劣）， 以及一份相应的回答文档。     这四份文档都保存在一个子文件夹中，按周顺序命名，然后这
十四周的教学内容都放在  D:\zz\00_RealProjects\0_LLM_Training\outputs\0_basic_training中

我的硬件训练资源大概是公司的8卡V100， 不支持fp16（这些文档中也有介绍）, 
有conda 环境和pytorch2.1.0 和相应的 cuda环境，可以支持pip install必要的包（这个不需要再建虚拟环境了
文档中更新的不及时，你可以改掉）
然后我的公司电脑的数据和代码  图片都不能传出来，只能传出一些总结性的text文字

其次，在家中我有一台 5070ti   16G的台式工作站，可以用于之后验证blackwell架构和复现一些工作，
这个我有所有的权限。之后必要时候我可以租4卡H100或者8卡H100，这也有完全权限。

大概这样背景， 你需要做两件事情：
1， 实现规划分析一下，完成一个这样的全能陪跑教练，需要构建多少codex的智能体，然后新建这些智能体和
workflow， 同时更新掉现在的AGENT.md   README.md   现在的这两个是给FDE陪跑教练设计的

2. 基于上述智能体和我上述那么多的背景和十几个文档，完成最终我想要的文件，并且输出到指定位置。

最后完成之后记录所有有用内容， 并更新相关的文档
```
