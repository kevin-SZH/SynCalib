# ExtrinsicManifoldData 开发指南

## 项目目标

本 workspace 用于构建面向相机-LiDAR 在线外参标定的新型训练数据。
核心任务不是在已经正确对齐的数据上独立采样人工扰动，而是：给定一套产品或
图纸提供的名义 LiDAR-to-camera 外参 `T0`，从仿真数据中构造或筛选出真实外参
`Ti` 位于 `T0` 局部 `SE(3)` 邻域内的大量、物理一致、几何可观测的数据对。

当前实现从原始 TartanGround cubemap RGB、packed depth 和真值位姿开始，负责
scene split、帧对重叠筛选、双边相机/LiDAR 姿态构造、邻域覆盖、传感器生成、
质量过滤、事务式导出和审计。网络集成遵循 workspace 的 CalibNet-first 政策，
在数据审计通过前不修改训练语义。

## 核心数学约定

统一使用列向量和 LiDAR-to-camera 外参：

```text
p_camera = T_lidar_to_camera @ p_lidar
```

每个任务必须显式给出完整名义外参：

```text
T0 = [R0, t0]
```

第 `i` 个物理一致样本具有真实外参：

```text
Ti = [Ri, ti]
```

训练初始状态统一使用 `T0`，正确几何关系使用 `Ti`：

```text
p_initial = T0 @ p_lidar
p_aligned = Ti @ p_lidar
```

网络应补偿的完整左残差定义为：

```text
delta_Ti = Ti @ inverse(T0)
p_aligned = delta_Ti @ p_initial
```

禁止用 `ti - t0` 代替 `delta_Ti` 的平移分量。若
`delta_Ti = [delta_Ri, delta_ti]`，则：

```text
delta_Ri = Ri @ inverse(R0)
delta_ti = ti - delta_Ri @ t0
```

旋转距离使用 `SO(3)` 对数映射或等价的测地角：

```text
omega_i = log(delta_Ri)
rotation_distance = ||omega_i||
```

平移和旋转使用不同单位和显式范围，不允许直接对齐欧拉角、比较齐次矩阵元素，
或在未定义米/弧度权重时对六维向量计算原始欧氏距离。

## 邻域与扰动政策

邻域围绕完整 `T0` 定义，而不是只围绕基线长度、平移方向或旋转矩阵的某个分量。
第一版不要求把平移分解为径向和切向分量。物理基线
`b = -R.T @ t` 可以作为审计指标记录，但不是主要采样坐标。

测地球任务的平移范围与名义外参平移尺度成比例：

```text
effective_max_t = max(alpha_t * ||t0||, t_min_floor)
```

CMRNext Direct-Euler 任务由用户明确采用逐分量盒：平移每轴 `[-0.2, 0.2] m`，
Euler XYZ 每轴 `[-10, 10] deg`。两种采样域不得混用覆盖坐标。物理深度/基线
缩放范围必须同时考虑原始数据帧对的真值相对位姿和当前目标外参 `Ti`，不得脱离
实际几何关系统一套用固定区间。不同任务、传感器或目标外参范围可以采用不同的
尺度区间；不得只根据 `T0`、单一基线长度或统一经验常数确定所有任务的范围。

测地球候选用 `[delta_t/effective_max_t, log(delta_R)/max_rotation_rad]` 描述，
按半径壳层和球面方向审计。Direct-Euler 候选使用原始独立采样坐标
`[tx, ty, tz, roll, pitch, yaw]`，各自除以 `0.2 m` 或 `10 deg` 后每轴分 5 桶，
形成 `5^6` 稀疏覆盖报告。固定模式只验证残差为零。归一化坐标只用于统计，
不表示米与度具有相同物理单位。

深度缩放只能发生在帧对通过双向重叠门之后。确定当前目标 `Ti` 后，必须结合原始
帧对的真值相对位姿求解或选择物理尺度；同一候选中的源/目标深度、相对平移和最终
点云必须使用一致尺度。相机和 LiDAR 姿态使用 SO(3) 旋转矩阵构造，LiDAR 倾斜
必须旋转 beam 后重新查询 cubemap，禁止只修改外参矩阵或旋转最终点云来伪造
“物理一致”样本。图像、点云和 `Ti` 必须来自同一套缩放后的仿真几何关系。

## 三阶段系统架构

### 1. 帧对池 `pair_pool.py`

帧对池负责扫描 TartanGround 六面 RGB、packed float depth 和 pano 真值位姿，先按
完整场景建立 scene-holdout 清单，再按原始位姿中心严格小于 `2 m` 的空间邻域提议
有向帧对；时间 gap 不参与筛选，以保留回环并消除运动速度影响。帧对必须通过双向几何
z-buffer 同表面球面区域 overlap 后才能进入生成阶段；四份正式任务均以两个方向较小值
不低于 `0.30` 作为候选丰富性下限，不使用帧对点数、tile 或遮挡率门。

帧对池保存和恢复可复现的合格候选及其可审计身份，包括场景、轨迹、源帧、目标帧和
原始文件路径。缓存恢复必须校验配置、输入身份和内容完整性，但不得复制原始数据，也
不得把机器特定的不可迁移绝对路径写入受版本控制的配置。

### 2. 样本生成 `sample_generation.py`

固定模式令 `Ti = T0`；随机模式采样完整左残差并令 `Ti = delta_Ti @ T0`。生成器从
帧对池读取 RGB、packed depth 和真值位姿，在帧对通过双向重叠门并确定目标 `Ti` 后，
结合原始帧对的真值相对位姿求解物理尺度，并对源/目标深度、相对平移和点云应用
一致缩放。

Direct-Euler 模式下同一 `Ti` 最多接收 4 个物理实现，或累计尝试 100 次后刷新；
接收样本不得清零该 `Ti` 的累计尝试数。场景连续 500 次尝试没有接收新样本时保留
已验收的部分结果并结束该场景；场景计数在接收新样本后清零。
相机光轴纬度带通过 `sampling.camera_optical_axis_latitude_band.enabled` 显式开关；
启用时四份正式任务使用 `max_abs_latitude_deg=10.0`。

相机和 LiDAR 使用耦合的物理姿态。从 cubemap 重新渲染相机时必须符合任务提供的
rectified pinhole 内参；模拟 HDL-64E 或 MID-360 时必须沿旋转后的 beam 重新查询
深度。候选以迭代器逐个输出，避免先大规模生成再过滤或忽略外参覆盖。

### 3. 质量过滤与导出 `quality_export.py`

每个样本必须分别在真实 `Ti` 和名义 `T0` 下评分。两套检查用途不同，不能相互
替代。

真实外参质量门验证数据和真值是否可靠：

- 检查 `Ri` 有限、正交且行列式接近 1。
- 仅保留相机前方、图像范围内的投影点。
- 使用最终针孔相机轴向 GT 深度在 LiDAR 投影的同一像素判断可见性。
- 分别用全部图像内投影点和 visible 投影点计算 Alpha Complex 面积。
- 全部投影面积 `A_all` 和可见投影面积 `A_visible` 均为独立构造的连续 Alpha
  Complex 面积；以 `A_all/(H*W)` 作为覆盖率，以
  `(A_all-A_visible)/A_all` 作为遮挡面积比例。生成质量
  阶段不使用 z-buffer owner、点数比例或固定图像网格。
- 检查与 RGB 边缘一致的深度断层、物体轮廓、表面法向多样性和二维梯度方向，
  排除只有单个平滑平面的低可观测样本。
- 优先复用 SynCalib `auto_filter` 的 A/B/C/D 指标与配置。正式训练候选保留 A/B，
  评估候选默认只保留 A；阈值必须位于 YAML，不得散落在实现代码中。

名义外参质量门验证数据在目标训练任务中是否可用：

- 随机模式用 `T0` 额外计算全部投影 Alpha 面积和 visible Alpha 面积比例，
  验证网络初始投影可用；固定 `Ti=T0` 模式只执行一次正式投影质量门。
- `T0` 投影面积不足或 visible 区域比例不足时丢弃样本。
- 真实投影质量很高但 `T0` 下输入接近空白的样本必须丢弃。

所有质量指标都写入 per-sample metadata。质量过滤算法、阈值或传感器配置变化后，
必须使旧评分缓存失效并重新评分。

通过质量门的候选必须审计六维残差覆盖、场景贡献以及同一源点云、帧对和目标视角
的重复使用。Direct-Euler 的 `5^6` 报告区分 proposed residual、accepted unique
residual 和 accepted sample，只报告有限样本占用与均匀偏差，不强制 15625 格全部
非空；geodesic 正式覆盖不足仍须失败并输出缺口报告。

正常生成必须在每个课程级别为 overall 和实际存在的 split 默认输出独立统计目录。
复用统计同时区分接受样本复用次数和每个 source/target 环视帧关联的唯一有向帧对数；
外参分布图按 fixed、Direct-Euler 或 geodesic 采样域生成，禁止用混合米/度的六维
欧氏距离或单张不可解释的六维散点图代替结构化覆盖报告。JSON 是权威统计，PNG 由
相同报告派生并在任务完成前验证可解码且非空白。

质量与导出模块同时负责进度记录、恢复复用、失败报告、staging、完整性审计和事务式
提交。选择过程必须保留随机种子、候选排序规则、覆盖统计和所有丢弃原因。

### KITTI 导出与审计

第一版导出 KITTI 风格的文件布局，并使用明确的数据语义：

```text
<sequence>/
  image_2/
  velodyne/
  calib.txt
  poses.txt
  nominal_extrinsic.txt
  residuals.txt
  metadata.json
```

- `poses.txt` 每行保存真实 `Ti` 的 3x4 row-major 矩阵。
- `nominal_extrinsic.txt` 保存该任务的完整 `T0`。
- `residuals.txt` 每行保存 `delta_Ti = Ti @ inverse(T0)`。
- `metadata.json` 保存坐标语义、数据来源、传感器 profile、邻域配置、筛选配置、
  六维覆盖统计、per-sample 质量指标、丢弃统计、随机种子和配置/输入指纹。
- 点云必须是 `Nx4 float32`，第 4 列是 dummy `0.0`；训练时必须保持
  `use_reflectance=false`。
- 图像、点云、`poses.txt`、`residuals.txt` 和 metadata 样本数量必须完全一致。

Metadata 必须明确写入：

```text
training_semantics: nominal_to_true_residual_v1
current_cmrnext_loader_compatible: false
```

`current_cmrnext_loader_compatible: false` 是输出语义声明，不构成旧训练链路兼容承诺。
训练端必须显式实现并验证 `nominal_to_true_residual_v1` 语义后才能使用这些输出。

导出必须使用 staging 目录、完整验证和事务式提交。失败构建不能留下看似完整的正式
数据目录。原始数据、生成数据、质量缓存、报告图片和本机路径不能提交到 Git。

## 传感器配置政策

已有 HDL 任务继续使用固定白名单 profile；MID-360 任务独立提供真实相机内参、完整
LiDAR-to-camera `T0` 和设备量程，只复用 KITTI 导出文件格式：

- KITTI Odometry 04-12 rectified camera：`1226x370`，
  `fx=fy=707.0912`，`cx=601.8873`，`cy=183.1104`。
- Velodyne HDL-64E：`64` rings，`2083` points/ring，垂直 FOV
  `[-24.9, 2.0]` degrees。
- Livox MID-360 扫描模型：10 Hz，每帧连续发射 20000 条 beam；从 4 秒、
  800000 条且不首尾
  相接的 `mid360.csv` 非重复 pattern 中按 seed 选择连续窗口。方位角/天顶角按
  `+X 前、+Y 左、+Z 上` 右手坐标解释，四路交织扫描用于质量邻接。

KITTI 风格目录和文件名不代表复用 KITTI 内参、外参或量程。正式输出必须从原始
RGB/depth/pose 重新渲染相机并重新模拟 LiDAR。

MID-360 的 20000 表示发射 beam 数；无效深度或超量程回波允许丢弃。当前每帧使用
一个静态 TartanGround cubemap，不模拟 0.1 秒内的平台运动畸变。

相机或 LiDAR profile 改动后，必须重新运行所有几何相关质量筛选，并重新生成 split
manifest。传感器参数放在 YAML 中，禁止在多个脚本中复制硬编码常量。

## 数据划分与防泄漏

必须先按完整场景或轨迹组确定 train/validation/test 归属，再生成虚拟视角、点云和
其他派生样本。同一源帧、帧对、目标视角或其任何派生版本不得跨 split。

默认使用 scene-holdout 评估。训练集和评估集可以覆盖相同的 `T0` 邻域范围，但必须
来自不相交场景。Manifest 必须记录 split 策略、场景列表、种子、过滤模式、传感器
profile 和数据指纹。禁止在导出完成后对所有帧做普通随机切分。

## 代码组织与工程规范

新增 Python 包使用四空格缩进、PEP 8 命名、类型提示和简短 docstring。函数和文件
使用 `snake_case`，类使用 `PascalCase`，常量使用 `UPPER_SNAKE_CASE`。

正式业务只允许以下三个顶层模块，避免增加平行业务链路或单体流水线：

```text
src/extrinsic_manifold_data/
  pair_pool.py
  sample_generation.py
  quality_export.py
  generation.py
  cli.py
  sources/
config/
  fixed_t0.yaml
  random_ti.yaml
scripts/
  generate_dataset.py
```

`generation.py` 只公开 `GenerationOptions` 和 `generate_dataset(options)`，并按加载
配置、建立或恢复帧对池、生成候选、过滤导出的固定顺序编排。底层允许保留必要的小型
配置、坐标变换、覆盖计算和 TartanGround 读取工具，但不得形成额外生成入口。

优先导入和复用 SynCalib 已验证的变换、投影、LiDAR 模拟和自动质量评分逻辑，不复制
其实现。若必须改变通用逻辑，应先在 SynCalib 中形成有测试的公共接口，再由本项目
调用。结构化数据使用 YAML/JSON/SQLite 等结构化 API，不用脆弱的文本拼接。

所有命令必须支持显式 seed、dry-run 或仅报告模式、可审计日志和清晰退出码。默认
不得覆盖已有正式输出；覆盖、清理或替换操作必须显式请求，并在提交前验证全部文件。

### 默认工程约束

默认不新增 hash、冻结 contract、baseline 或 gate。只有能明确给出一个具体失败场景，
并说明 Git、版本号、主键、事务、唯一约束、类型和普通测试为什么不足时，才允许加入。
不得为了简化而删除已有安全措施。门禁只放在不可逆、跨系统、安全或正式发布边界；
前置检查不得挤掉真正的代码执行、模拟或测量。

### Git 提交纪律

任何修改只要涉及功能模块的新增、删除、拆分、合并或职责变化，就必须形成一次独立
Git commit。提交说明必须使用中文，清楚描述本次改动的功能边界，不得把互不相关的
模块变化混入同一提交。提交前必须运行与改动风险相匹配的测试和差异检查。

若当前执行者没有 Git 写权限，或用户希望自行提交，则必须在该阶段结束时明确提醒
需要提交，并给出建议的中文提交说明和应包含的文件范围；不得在未提醒的情况下继续
堆积下一项独立功能修改。纯调查、只读审查和没有修改仓库文件的操作不要求提交。

### 中文注释与可读性

每个新增函数都必须提供中文 docstring，说明函数作用、所有参数的含义与单位、返回值
以及调用者需要知道的异常或副作用。新增类及其公共方法遵循同一规则。对已有函数做
实质性修改时，如果原函数缺少上述说明，应在同一修改中补齐。

每个新增模块级全局变量、常量或共享状态都必须有中文注释，说明其语义、单位、坐标
约定、允许范围或生命周期。函数内部对几何变换、中间坐标系、阈值、索引、缓存状态
和其他不直观变量添加中文说明；不要给显而易见的赋值添加逐行复述式注释。注释必须
与当前实现保持一致，修改行为时同步更新，不允许用过期注释制造错误理解。

## 运行验收不变量

本项目不在仓库中维护独立测试代码。功能变更必须通过编译、导入、两份正式 YAML
校验以及真实 TartanGround 小规模生成进行验收。临时配置和输出必须位于仓库外，
验收结束后删除，不得把测试缓存或生成产物提交到 Git。

真实数据验收至少检查以下内容：

- `Ti @ inverse(T0)` 的组合、求逆和恢复恒等式。
- 平移范围按 `||t0||` 比例计算以及 `t_min_floor` 退化保护。
- 尺度决策同时使用原始帧对的真值相对位姿和目标 `Ti`，且源/目标深度、相对平移
  和点云实际使用同一尺度。
- 坐标约定和 `P_camera = T @ P_lidar` 投影方向一致。
- 帧对池 z-buffer 同表面 overlap、针孔深度可见性和 Alpha 区域覆盖指标实际执行。
- 固定与随机模式均能读取帧对、生成传感器数据并进入质量过滤。
- KITTI 图像尺寸、HDL-64E 或 MID-360 pattern 和 `Nx4 float32` 点云符合配置；
  MID-360 额外检查连续 20000 条窗口、合法边界、复现 seed 和四路邻接。
- pose、residual、图像、点云和 metadata 样本数量完全一致。
- scene-holdout 下源场景、帧和派生样本不存在跨 split 泄漏。

验收报告仍须包含筛选前后样本数、按原因丢弃统计、六维残差覆盖、最近邻距离
分布、场景贡献、质量分布以及 `T0` 公共监督点分布。只报告总样本数不视为完成
邻域构建。
