# Week 05 回答篇：数据合同 / 对齐 / 可复现

> **先闭卷。** 每题4分：定义、机制、证据、边界各1分；参考答案不是可背诵的运行结论。

## Recall

### W05-R01
答案：episode_id字符串、frame_index整数、timestamp秒、images为camera→uint8 HWC、state float `[Ds]`、action float `[Da]`、instruction字符串、可选success、task/object/seed/domain metadata。评分点：四类完整。误区：只写“pose”。追问：rotation还需什么？

### W05-R02
答案：Task Card定义任务/成功/ID-OOD；Data Card描述来源/schema/统计/偏差/用途限制；manifest记录revision/hash/split/preprocess以机器复建。评分点：三职责3；关系1。误区：三个文件复制同内容。追问：哪一个先冻结？

### W05-R03
答案：Python/NumPy/torch CPU/CUDA all RNG；DataLoader generator、worker seed、sampler seed/epoch/cursor、augmentation、model init与deterministic flags。评分点：四类。误区：只`torch.manual_seed`。追问：persistent workers呢？

### W05-R04
答案：keys、dtype/shape、finite、timestamp、frame、range、rotation、camera、episode boundary、termination、dt、duplicate ID。评分点：每三类1分。误区：validator只查NaN。追问：如何测试validator自身？

## Explain

### W05-E01
答案：同episode相邻帧高度相似并共享场景/对象/策略，跨split使模型近邻记忆，验证不再估计新episode泛化。评分点：相关性2；指标偏置1；group split1。误区：数据多就无所谓。追问：object泛化还需什么group？

### W05-E02
答案：test/dev统计向训练暴露评测分布，属于信息泄漏；train冻结mean/std后原样应用。评分点：泄漏2；流程1；hash1。误区：均值不是label所以可用。追问：clipping threshold呢？

### W05-E03
答案：非确定CUDA kernel/并行归约、worker调度、文件遍历顺序、依赖/硬件、augmentation和保存时点都可改变执行。评分点：四因。误区：“GPU天然随机”停止分析。追问：最先比较什么？

### W05-E04
答案：上游同名数据可修订，seed算法/输入顺序可变；需revision、文件/episode hash、explicit IDs、preprocess commit、stats hash。评分点：原因2；补救2。误区：记下载日期即可。追问：streaming shard如何hash？

## Apply

### W05-A01
答案：本教程定义`o_k`预测`a_{k+ℓ}`；ℓ=+2即observation k对齐未来第2个控制步action。时间线明确k,k+1,k+2及边界padding。评分点：定义2；图/边界2。误区：只说“延迟2帧”无方向。追问：cross-correlation实现符号为何易反？

### W05-A02
答案：k0 `[a0,a1,a2,P]` mask1110；k1 `[a1,a2,P,P]`1100；k2 `[a2,P,P,P]`1000；是否包含terminal command由Task Card定义。评分点：三行3；终止1。误区：拼下个episode。追问：history K边界呢？

### W05-A03
答案：按episode构建group，先将目标pose区间完全留作OOD；同object instance/scene seed不可跨train与OOD；写explicit lists并检查hash/near duplicate。评分点：单轴1；三级group2；manifest1。误区：同时换光照/对象。追问：ID dev怎么取？

### W05-A04
答案：从clean fixture复制12份，每份只注入一种错误；validator输出唯一stable code，断言目标code存在且非目标code不误报；另测多故障组合。评分点：隔离2；oracle1；组合1。误区：只在真实数据目测。追问：range异常如何定阈值？

## Debug

### W05-D01
答案：查静止action比例、absolute/delta语义、inverse normalization、lag、维度顺序/单位、padding计loss，再查模型。评分点：六项中四项且有顺序。误区：先换更大policy。追问：哪个终端统计最先看？

### W05-D02
答案：记录每batch IDs与augmentation params；workers=0对照；固定文件排序；检查worker_init_fn/generator；检查sampler cursor/epoch与prefetch造成的保存边界。评分点：四层。误区：提高seed位数。追问：persistent workers如何恢复？

### W05-D03
答案：先确认合成lag oracle；预注册proxy/符号/选择规则；看bootstrap/跨episode置信区间；用同预算BC probe与领域时序定义交叉验证；仍不清则INCONCLUSIVE。评分点：四步。误区：选最低val error后当真值。追问：相邻lag差小意味着什么？

### W05-D04
答案：量化并标记upstream重复；根据预注册benchmark政策决定保留并分层报告或建立去重派生split；版本/ID/hash全记录，不静默删除也不冒充原split。评分点：透明2；决策1；命名1。误区：为好分保留且不披露。追问：旧结果是否作废？

## Design

### W05-G01
答案：upstream repo/dataset revision、task/config、file/shard/episode hashes、explicit split list hash、preprocess commit/config、schema/camera order、lag/action representation、normalization hash、生成时间/工具版本。评分点：四组。误区：只记dataset URL。追问：哪些值变化应拒绝resume？

### W05-G02
答案：在定义好的batch边界保存RNG全套、sampler epoch/cursor/permutation、worker/augmentation生成器、prefetch语义、global step；恢复先构建相同loader再load state，断言step26 IDs/params。评分点：状态2；时点1；断言1。误区：仅恢复模型RNG。追问：多worker预取如何简化Gate？

### W05-G03
答案：终端输出数量/长度/dt/缺失/finite/分位数/违规code/duplicate/split overlap/hash与PASS状态；只保留聚合和内部sample locator，原图不输出。评分点：覆盖2；可追踪1；边界1。误区：截图30张图外传review。追问：如何人工检查仍留审计证据？

### W05-G04
答案：共享typed observation/action batch、mask、time/schema/version接口与trainer/eval hooks；Task-D另有adapter，强制声明dimension/unit/frame/rate/representation/success，契约不匹配硬失败。评分点：共享2；隔离1；硬校验1。误区：shape相同就直接复用head。追问：14D joint到EEF能否无损？

## Trade-off

### W05-T01
答案：absolute易表达全局目标但怕标定/坐标偏移；delta局部范围稳但积分漂移、依赖初值。各自统计与inverse/部署积分不同。评分点：两方2；normalization1；部署1。误区：delta只是absolute做差无需契约。追问：gripper离散维怎么办？

### W05-T02
答案：strict便于bitwise归因但可能禁快kernel/降低吞吐；高性能更真实但需容差、ID追踪与重复。单测/resume先strict，性能结论再冻结高性能模式。评分点：双方2；分阶段2。误区：全程只选一种。追问：如何记录容差？

### W05-T03
答案：streaming省磁盘、早查看metadata，但受shard顺序/网络/revision影响且随机访问/resume复杂；本地冻结可hash/随机访问，耗磁盘与复制时间。评分点：三轴3；选择1。误区：streaming天然可复现。追问：cache eviction怎么办？

### W05-T04
答案：RoboTwin更接近复杂双臂但数据/依赖大；PushT协议轻且公开；合成数据最可控、无下载但外部有效性弱。按一周的schema/lag/repro目标与批准资源选，不能因“更真实”扩scope。评分点：三者3；决策1。误区：合成PASS等于真实Task-P结论。追问：何时升级到RoboTwin？

## 自评

落盘日期、总分/六级分项、AI辅助等级、证据文件、错误类型和回归日期。Data Card完整不等于数据正确；只有validator、leakage、lag与复现实验通过才构成本周Gate证据。
