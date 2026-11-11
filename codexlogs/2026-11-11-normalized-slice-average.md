# 切片库改为归一化高斯平均

- 目标：修改 `diff-gaussian-rasterization_slice`，移除 opacity 与遮挡合成；当前主项目使用的另一个库及训练入口不变。
- 前向：对保留的三维高斯权重累加 `sum(w*color)` 和 `sum(w)`，有效覆盖时严格相除，否则返回背景；空高斯也返回背景。
- 保留：原有轴向坐标、候选 tile 范围、排序和 `w >= 1/255` 裁剪；分母保护阈值为 `1e-8`。
- 反向：使用 `dL/dw = dot(dL/dI, color-I)/sum(w)`，继续传播至均值、协方差、尺度、旋转与 SH；opacity 返回零梯度。
- 接口：公开 Python 参数及 `(image, radii, invdepth)` 返回结构不变；内部缓存改存权重和，Python 保存最终图像供原生反向使用；原生扩展必须配套重编译。
- 同步修正：前后向像素核使用实际选中的预计算协方差；可选空张量显式转换为空指针，避免误入尺度或颜色分支。
- 文件：主要涉及 `forward.cu`、`backward.cu`、`rasterizer_impl.cu/.h`、相关声明与桥接文件、Python 包装；补充 README 与 `tests/test_normalized_average.py`。

## 验证

- 环境：`cardiac-3dgs`，PyTorch 2.4.1+cu121，A100；使用已有 CUDA 12.1、gcc12-tools 与 Ninja，临时编译为 sm_80，没有安装依赖或覆盖已安装库。
- 系统 GCC 13 与 CUDA 12.1 不兼容，cardiac-3dgs 的 GCC 7 又低于 PyTorch 要求，最终使用已有 GCC 12.4。
- 最终扩展：`/tmp/slice-normalized-build-wkqa_33r/_C_G3D.so`；构建脚本 `/tmp/build_normalized_slice.py`，最终编译日志 `/tmp/slice-normalized-build-final.log`。
- 最终 8 项 CUDA 回归测试全部通过：CPU float64 独立前向/梯度参考、三个切片方向、两种协方差路径、opacity 无关性、同色/重复高斯、排列、多 tile/多批次、背景与 SH。
- 测试命令：`SLICE_NORMALIZED_EXTENSION=/tmp/slice-normalized-build-wkqa_33r/_C_G3D.so /home/exouser/miniconda3/envs/cardiac-3dgs/bin/python -B diff-gaussian-rasterization_slice/tests/test_normalized_average.py -v`。
- 最终测试日志：`/tmp/slice-normalized-tests-final.log`。编译仍有上游 `Tensor.data<T>()` 弃用及未使用深度变量警告，未扩大范围清理。
- 对最终扩展运行 Compute Sanitizer 12.1 memcheck，同一组 8 项测试全部通过，`ERROR SUMMARY: 0 errors`；日志 `/tmp/slice-normalized-memcheck.log`。这不代表首轮瞬态异常已定位或长期稳定性得到证明。

## 限制

- 首轮使用 CUDA 参考计算时，多批次测试曾在 `torch.linalg.inv` 处报告一次非法访问，进程后续测试受到污染；独立同步重跑没有复现同一异常，但参考比较有一个像素超差。改为 CPU 双精度参考后测试通过；首轮异常根因未定位，不能据此声明环境或长期运行稳定。
- 梯度参考样例避开裁剪边界，使用 `scale_modifier=1` 和条件良好的协方差；不构成对所有数值范围的验证。
- 单高斯主导时归一化会抵消空间衰减，几何梯度可能很弱；无覆盖处不向高斯传播梯度。未修改训练损失、初始化或增密策略。
- 未运行完整训练、真实病例或性能基准，未安装扩展；使用时需要重编译并重启进程。
- 该目录在任务开始时即为 Git 未跟踪状态；修改前快照位于 `/tmp/slice-normalized-before-a5kojko8`。保留了原有配置和其他工作区修改。
