# 恢复使用 pip 安装的渲染库

- 用户要求：解释 loss.backward 与渲染器 backward 的关系，取消应用入口对工作区渲染库的强制导入。
- 修改 `Renderer.py`：移除工作区路径的 sys.path 插入及来源拒绝检查，恢复标准的 `from diff_gaussian_slice_rendering import ...`。
- 未修改渲染数学、训练、CUDA 扩展、Dataloader 或已安装包，未安装或重编译依赖。库自身的源码专项测试不属于本次应用入口修改范围。
- 验证：使用 cardiac-3dgs 环境的 Python 3.10，在全新进程导入 Renderer，Python 包与原生 `_C_G3D.so` 均来自该环境的 site-packages；Renderer 导入前后 sys.path 相同，公开渲染函数与安装包对象一致。相关 diff 空白检查通过。
- 解释核对：安装包的 `_RasterizeGaussians` 继承 torch.autograd.Function；loss.backward 启动全图反向传播，经过渲染节点时调用其 backward，由 `_C.rasterize_gaussians_backward` 返回局部梯度，再沿计算图继续传播。
- 限制：本次验证导入来源与接口，没有运行 GPU 渲染、反向或完整训练。已启动的 Python 进程可能仍缓存旧模块，需重启相应训练进程再使用新导入方式。
