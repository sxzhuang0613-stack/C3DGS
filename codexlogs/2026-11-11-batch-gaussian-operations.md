# Stage1 批量高斯操作与模型解耦

- 目标：用 3DGS 风格的批量启发式操作替代逐候选结构搜索，解除结构模型对 noop 模型的依赖。
- `Gaussians/GaussianModel_structural.py` 改为独立类，维护属性、初始化、PLY、Adam 与学习率。批量梯度阈值选点、小点 Clone、大点 Split、无效点/可选弱幅值 Prune；不再使用 Birth/Merge/Swap、Top-K 梯度排序、局部响应、SVD/坐标下降、额外渲染、逐候选验收或回滚。
- 全部操作基于同一旧模型，各参数和 Adam 状态统一提交一次。Clone 均分强度并施加对称扰动；Split 保持未加 jitter、未截断内禀核的有符号积分、中心及二阶矩。未改旧点保留动量，新点清零；Clone 父点重置位置/强度动量；组级 step 和固定正则参考容量保留。
- `GetInitialGaussian.py` 移除无消费者的半径聚合；统计改为体素单位位置梯度。旧搜索器 checkpoint 因统计单位/算法变化拒绝完整续训，仍支持独立实验 PLY 热启动。训练采样、损失及更新顺序保持不变。
- `config_train.yaml/config_utils.py` 更新并验证批量参数：正梯度中位数 2 倍、绝对下限 0（待标定）、最低观测 8 次、面内投影 sigma 门限 2 像素、每周期最多增长 5%。弱幅值裁剪默认关闭。`README.md` 与 `docs/batch_gaussian_thresholds.md` 说明单位、校准、误差上界及近似限制。

实际验证（已有 cardiac-3dgs 环境，未安装依赖或重编译扩展）：

- 全新进程确认结构模型基类仅 object，未导入 GaussianModel_noop；实际渲染包装和 `.so` 仍来自 site-packages，API 3。
- 13 项结构回归通过：包含无额外渲染/求解/Top-K、加性 Clone、Split 积分/矩、混合一次提交、容量和随机映射、Adam、checkpoint/RNG、尺度单位不变性，以及短 CUDA 渲染→更新→结构调整→更新。
- 9 项 CPU 几何/初始化回归通过；CUDA 初始化伴随与 backward 检查也通过。
- 配置/接口回归 14 项中 13 项通过；未修改旧 noop 模型的容量测试仍失败（8 != 6），不扩大修复范围。
- 默认后端的合成训练/续训/评估集成未稳定通过：一次在评估阶段、一次在训练 backward 出现 CUDA illegal memory access；不能据此认定本次引入或修复该错误。单独诊断进程设置 `torch.backends.cudnn.enabled=False` 后，该合成集成测试通过；没有改变工程后端或环境设置，也不据此确定根因。
- 5 个修改 Python 文件 AST 检查、git diff --check 通过。

限制：未做真实病例、完整训练、全仓库测试或前后性能对比；默认 GPU 端到端路径仍需排查。阈值是工程初值，未证明最优；结构调整不保证目标下降或后续训练必然补偿误差。没有修改 noop、Stage2、评估入口或 CUDA 渲染器。
