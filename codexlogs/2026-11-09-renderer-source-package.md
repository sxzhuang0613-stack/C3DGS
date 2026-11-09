# 渲染库恢复为待安装源码

- 用户要求：源码库保持待安装形式，取消未安装即可调用的工作区加载方式；本次不执行安装。
- 库的 `__init__.py` 改为正常加载包内 `_C_G3D`，删除 build/lib 搜索、C3DGS_EXTENSION_DIR 旁加载及缺扩展时的降级导入。保留 CUDA 数学实现和接口版本校验，将版本错误提示改为 pip 重装。
- 将 build、egg-info、包内就地编译 `.so` 和字节码移至 `/tmp/c3dgs-renderer-build-backup-20261109-1lxddk18`，保留 manifest，未删除源码或覆盖用户修改。部分生成文件原被 Git 跟踪，因此会显示为删除。
- 同步移除 tests 中指向渲染库源码/独立扩展目录的路径覆盖，以及 benchmark 的 package-root/extension-dir 旁加载参数；项目根目录的普通测试导入路径保留。README 统一为 pip 安装后调用。
- 实际验证：8 个相关 Python 文件 AST 语法检查通过，相关 diff 空白检查通过；源码目录直接 import 按预期因缺少 `_C_G3D` 失败；全新进程从项目入口导入，Python 包和原生扩展仍来自 cardiac-3dgs 环境的 site-packages。源码目录无 `.so`、`.o` 或 `.pyc` 残留。
- 没有 pip 安装、卸载、重编译、GPU 渲染或训练；已安装环境未修改。后续用户自行 pip 安装后需重新启动相关 Python 进程。
