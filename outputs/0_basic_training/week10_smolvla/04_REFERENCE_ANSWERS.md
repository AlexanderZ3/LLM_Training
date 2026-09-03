# Week 10 回答篇：SmolVLA 微调闭环

> 先闭卷再校准。每题3分：核心、证据/shape、边界各1分。实现细节以锁定LeRobot/model revision为准。资料核验日期：2026-09-03。

## Recall

### W10-R01

参考答案：多camera图像进入视觉/VLM编码，state经投影、instruction经tokenizer/VLM形成context；context条件化action expert，expert从noise经flow updates生成action chunk。processor位于输入与输出语义边界。

评分点：三输入；context；action expert/flow。  
常见误区：称其仅为图像到动作CNN。  
追问：KV cache/attention模式为何需看revision？

### W10-R02

参考答案：clean/noise/pred/target为[B,A,Da]，time [B]广播[B,1,1]，mask [B,A]或[B,A,1]。线性路径input=(1-t)noise+t clean，velocity=clean-noise，做masked MSE。

评分点：shape；公式；mask denominator。  
常见误区：time方向或velocity符号反。  
追问：推理Euler一步如何写？

### W10-R03

参考答案：LeRobot SHA/tag、model revision、processor/config revision、dataset revision/hash。strict load检查missing keys、unexpected keys、mismatched shapes，并对白名单作显式记录。

评分点：四元组；三类问题；核心head不静默跳过。  
常见误区：只有模型仓库名。  
追问：transformers/torch版本放在哪里？

### W10-R04

参考答案：model、optimizer、scheduler、GradScaler、Python/NumPy/torch CPU/CUDA RNG、sampler epoch/offset、global step、successful updates/samples、processor/config和全部revision。

评分点：训练状态；随机/数据；processor。  
常见误区：只恢复weights和step。  
追问：EMA存在时怎么办？

## Explain

### W10-E01

参考答案：processor定义camera字段、resize/range、tokenization、state/action顺序、normalization、padding和inverse；这些决定tensor每一维的语义。即便shape合法，处理不一致也相当于给模型不同任务。

评分点：至少四项；语义；train/eval一致。  
常见误区：processor只影响速度。  
追问：normalization统计为何只来自train？

### W10-E02

参考答案：模型仍能拟合错误但一致的camera映射，或从state/其他camera捷径降低train loss；loss不验证物理语义。需schema断言、可视/统计检查、camera shuffle/ablation和rollout。

评分点：错误映射仍可拟合；捷径；检测。  
常见误区：loss下降证明camera对。  
追问：怎样防止camera dict迭代顺序变化？

### W10-E03

参考答案：DDP每进程仍执行同一个kernel；V100 SM70缺少官方FA2要求的更新架构能力，卡数只增加副本/吞吐，不改变单进程指令兼容。应选eager/兼容SDPA或锁定兼容commit。

评分点：每rank独立；硬件边界；fallback。  
常见误区：8卡共同模拟一个兼容GPU。  
追问：FSDP能否解决import失败？

### W10-E04

参考答案：frozen参数仍占权重显存；其梯度和Adam moments通常消失。若正确no_grad/切图，内部saved activation可减少，但输出/context和下游action expert activation仍在；具体AMP副本与cache实测。

评分点：四类状态区分；activation条件；实测。  
常见误区：冻结模块占零显存。  
追问：unfreeze一个top block增加哪些状态？

## Apply

### W10-A01

参考答案：images按camera映射为[B,3,H,W]或堆叠[B,C,3,H,W]；state[B,Ds]；token ids/mask[B,Ttext]；actions[B,A,Da]；valid mask[B,A]/[B,A,1]。同时打印dtype、range、字段顺序与normalization revision。

评分点：所有shape；语义元数据；不假定唯一camera布局。  
常见误区：只打印shape不打印字段名。  
追问：padding到max_action_dim怎样审计？

### W10-A02

参考答案：固定horizon、C、resolution、freeze、precision，依次batch1/2/4/8；每点3warmup+10measure，首次optimizer后记录allocated/reserved。预测到30GB或实测28–30GB停；主配置reserved<30.5GB且数值finite。

评分点：单变量；测量规则；阈值。  
常见误区：以真实OOM为终点。  
追问：optimizer lazy init如何影响点1？

### W10-A03

参考答案：固定128–256 windows、augment/dropout关闭或固定，先tiny FP32再FP16，训练action expert到loss显著下降；B组只shuffle observation/instruction配对，其他不变。若B无退化，condition可能未使用或数据可由动作先验解决。

评分点：overfit；单变量shuffle；解释。  
常见误区：把shuffle组也换seed。  
追问：怎样区分instruction和vision捷径？

### W10-A04

参考答案：同test/ID+OOD、processor、camera、normalization、horizon、flow steps/solver、initial noise、raw/EMA和episode seeds；比较action/endpoint/jerk/saturation/latency/rollout区间及错误桶，system crash单列。

评分点：固定项；多指标；系统失败。  
常见误区：base和FT各用推荐推理参数。  
追问：checkpoint selection如何固定？

## Debug

### W10-D01

参考答案：用pip dry-run和rg定位optional extra、import与attn_implementation；选择不含flash extra的批准安装/锁定commit，配置eager或经PyTorch2.1/V100短测的SDPA。不要编SM70 fork或升级torch，记录fallback。

评分点：定位；安全回退；版本记录。  
常见误区：尝试多个非官方wheel。  
追问：如何证明两个run后端一致？

### W10-D02

参考答案：不能直接继续。forward可跑可能表示核心head随机初始化或部分权重未加载。比对四元组，逐key分类；只有预注册无害buffer可白名单，核心VLM/action expert/processor mismatch必须阻断。

评分点：阻断；分类；核心key。  
常见误区：strict=false即兼容方案。  
追问：mismatched shape通常指向哪些schema？

### W10-D03

参考答案：先global batch/micro/accum与LR；sample IDs/sampler；loss reduction/mask denominator；frozen/unused参数；processor随机augment；scale/skip；no_sync；rank参数checksum。用单卡固定global batch reference逐步比较。

评分点：数学先；数据；AMP/参数。  
常见误区：先增大NCCL timeout。  
追问：当前LeRobot的batch语义怎样从banner核对？

### W10-D04

参考答案：查optimizer moments、scheduler position、GradScaler、Python/NumPy/CPU/CUDA RNG、sampler offset、augment、processor/normalization和实际fixed batch IDs；step相同不足以证明状态连续。

评分点：至少六项；fixed IDs；不接受8%。  
常见误区：把差异归因浮点误差。  
追问：怎样建立uninterrupted reference？

## Design

### W10-G01

参考答案：记录GPU capability、torch/CUDA/transformers/LeRobot/model revision；rg命中位置；precision、autocast/scaler；attention API、实际kernel证据；FA2是否依赖；fallback；one-forward结果；依赖dry-run及阻断项。

评分点：版本；代码+运行证据；处理结论。  
常见误区：只写“V100兼容”。  
追问：哪些内容不能从公司导出？

### W10-G02

参考答案：内部统一schema经显式映射表生成锁定版本CLI/config；启动前解析--help或导入config类，未知字段报错；保存resolved config与diff，不用kwargs静默丢弃。每个revision有契约测试。

评分点：显式映射；fail closed；resolved config。  
常见误区：catch异常后忽略flag。  
追问：main/stable接口同时维护的成本？

### W10-G03

参考答案：单卡gate=strict load、forward、overfit、200+ FP16、resume、peak；四卡gate=20 smoke→100 timed、样本/参数/loss/scale正确与E4/comm；八卡仅四卡通过且负载足够，50–100 profile。任一失败不越级。

评分点：三级顺序；每层证据；八卡可选。  
常见误区：直接8卡跑长任务。  
追问：四卡第二拓扑域何时复测？

### W10-G04

参考答案：列base/revision、data/schema、processor、backend/precision/freeze、trainable params、train/resume/eval config、memory/performance、base-vsFT指标、错误桶、限制与公司边界；结论分别用FAIL-SYSTEM、FAIL-MODEL、INCONCLUSIVE。

评分点：可复现；结果/限制；状态区分。  
常见误区：只写最佳成功率。  
追问：真实checkpoint无法获取时怎么写？

## Trade-off

### W10-T01

参考答案：expert-only训练状态最省、稳定但适应视觉/语言域能力有限；unfreeze top block增加梯度/moments/activation，可能更适应任务；full FT成本与灾难遗忘/数值风险最高。先expert-only，以shuffle/OOD证据决定是否解冻。

评分点：状态显存；能力；递进gate。  
常见误区：参数越多训练越好。  
追问：LoRA应作为哪种独立实验？

### W10-T02

参考答案：stable/pinned可复现且更可能与2.1兼容；main有新功能/修复但依赖和flags漂移，当前多GPU/FSDP可能需要更新栈。公司优先已有批准commit，dry-run与短测；main只作阅读或独立兼容项目，不强升。

评分点：收益风险；2.1选择；pin。  
常见误区：网页命令比本机help权威。  
追问：模型revision与代码commit怎样配对验证？

### W10-T03

参考答案：若450M在四卡后通信主导，八卡增加GPU-hours却不缩time-to-target；四卡长训更值。八卡50–100step仍可验证topology/NCCL与扩展边界；只有correctness和E8/负载合理才长训。

评分点：GPU-hours；sample价值；门槛。  
常见误区：资源空闲就必须用满。  
追问：冻结比例如何改变扩展？

### W10-T04

参考答案：结论为离线指标改善但闭环收益未证实，通常TRADEOFF或INCONCLUSIVE；检查区间、错误桶、processor/inverse、时序/jerk、OOD与system failures。下一步针对失败桶做单一数据或控制改动，不夸大模型收益。

评分点：诚实裁决；诊断；单变量下一步。  
常见误区：只发布离线最好值。  
追问：何时可判FAIL-MODEL？

## 复训

低于44/72：重做W10-A01、W10-D02、W10-G03并完成一次strict-load/processor契约白板复测。任何数据外带、版本强升、V100 BF16/FA2或silent key skip错误，先做安全与兼容性复训。
