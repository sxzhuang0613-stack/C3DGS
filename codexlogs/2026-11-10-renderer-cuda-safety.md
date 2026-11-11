# 渲染库 CUDA 缓存与索引安全检查

- 目标：按已确认的分析方案修改 `diff-gaussian-slice-rendering`，拦截缓存错配和调试模式下的非法索引，改善 CUDA 错误定位；保持公开函数签名、渲染数学和梯度定义。
- `rasterizer_impl.h/.cu`：三块缓存增加 magic、布局版本、调用标识、N/W/H/instances 和缓存类型。backward 在零实例提前返回和候选读取前核对元数据，拒绝等尺寸的跨 forward 缓存混用。缓存头通过按值 kernel 写入，避免异步复制引用失效的主机栈数据。
- `forward.cu`、`backward.cu`、`auxiliary.h`、`rasterizer.h`：传递 debug 标志；调试模式在读取候选范围、Gaussian ID、分桶偏移及 tile ID 前检查索引，并逐阶段同步、报告阶段名。非法输入抛错，不返回部分结果。默认模式使用不含逐索引检查的 kernel 特化。
- `rasterize_points.cu`：分配图像之前按 CUDA Runtime 查询的设备 grid 限制校验尺寸；backward 实例数上限与 forward 一致。
- `ext.cpp`、`setup.py`、Python 包装及 README：扩展 API 更新为 4，缓存布局版本为 1，保留四元数修订标记 1；导出编译时源码 SHA-256。Python 拒绝旧扩展。
- 新增 `tests/test_renderer_safety.py` 六项回归测试：缓存混用（含零点）、损坏头/短缓存/缓存类型、尺寸与实例数错配、损坏范围和 ID、非默认 stream 的有效前后向，以及设备 grid 超限。索引损坏测试还检查抛错后原缓存可继续使用。

实际验证：

- 使用已有 cardiac-3dgs Python/PyTorch、CUDA 12.1、gcc12，在 `/tmp/c3dgs-renderer-safety-1ib9eu0r` 独立编译成功。测试仅通过临时 PYTHONPATH 加载复制的 Python 包及新扩展，确认版本 `4/1/1`；源码指纹与当前工作区一致：`3c1bbf8051d22bddcb889bbeb041ef8f7bcffa67d220a83748e765b3ef7b25a3`。
- 24 项定向测试全部通过：新增安全测试 6 项、既有 slice_response 16 项、既有四元数 CPU 梯度参考及原生非单位输入拒绝测试 2 项。日志：临时构建目录的 `tests.log`。
- 已有 Compute Sanitizer 对新增六项测试执行 memcheck，六项通过，退出码 0，`ERROR SUMMARY: 0 errors`；没有按 kernel 名过滤。日志：同目录 `memcheck.log`、`memcheck-tests.log`。
- 三个修改/新增 Python 文件 AST 检查和 `git diff --check` 通过。

限制与运行状态：

- 缓存头校验在普通 backward 也增加一次当前 stream 同步；未做性能测量。完整索引检查及额外阶段同步仅在 `debug=True` 启用。元数据检查不能验证任意伪造内容，也不能识别整套替换为另一合法 forward 的缓存。
- 未执行真实病例训练、600 步训练测试或全仓库测试，不能据此认定原始间歇性 CUDA 错误已完全解决。
- 未安装或覆盖 site-packages，当前正常导入的已安装扩展仍为 API 3；重新安装并重启 Python 进程后才会使用修改。没有修改训练入口、策略或配置，也没有覆盖任务开始前已有的 build 产物和 `temp.py` 改动。
