# CUDA Kernel Profiling Report Guide

在 AutoDL 容器环境中，由于 GPU 性能计数器权限受限，`ncu` 无法直接使用。
本文档记录可用的分析工具及完整的报告生成与获取流程。

---

## 工具对比

| 工具 | CLI 命令 | 报告格式 | 容器可用 | 查看工具 |
|------|---------|---------|---------|---------|
| Nsight Systems | `nsys` | `.nsys-rep` | 可用 | Nsight Systems UI |
| Nsight Compute | `ncu` | `.ncu-rep` | 权限受限 | Nsight Compute UI |
| NVIDIA Visual Profiler（已废弃） | `nvprof` | `.nvvp` | 可用 | nvvp（已停止分发） |

**推荐使用 `nsys`**：不依赖 GPU 性能计数器，容器内无需特权即可运行。

---

## 方案一：使用 nsys（推荐）

### 1. 编译程序

```bash
cd /root/Learn-CUDA
cmake -B build && cmake --build build
```

### 2. 生成报告

`nsys` 不在默认 PATH 中，需使用完整路径：

```bash
/opt/nvidia/nsight-compute/2024.1.1/host/target-linux-x64/nsys profile \
  -o /root/Learn-CUDA/kernel_profiling/<报告名> \
  /root/Learn-CUDA/build/kernel_profiling/<可执行文件名>
```

示例：

```bash
/opt/nvidia/nsight-compute/2024.1.1/host/target-linux-x64/nsys profile \
  -o /root/Learn-CUDA/kernel_profiling/combined_memory_nsys \
  /root/Learn-CUDA/build/kernel_profiling/combined_memory_access
```

生成文件：`combined_memory_nsys.nsys-rep`

### 3. 下载报告到本地

在**本地终端**执行：

```bash
scp user@<服务器IP>:/root/Learn-CUDA/kernel_profiling/<报告名>.nsys-rep ./
```

也可以在 VSCode Remote Explorer 中右键文件选择「下载」。

### 4. 本地打开报告

1. 下载安装 Nsight Systems：https://developer.nvidia.com/nsight-systems
2. 打开 Nsight Systems，File → Open → 选择 `.nsys-rep` 文件

可查看内容：
- GPU/CPU 执行时间线
- CUDA kernel 调用与耗时
- 内存拷贝（HtoD / DtoH）时间
- CUDA API 调用记录

---

## 方案二：使用 ncu（需要宿主机权限）

容器内默认报错 `ERR_NVGPUCTRPERM`，需在宿主机执行以下命令后方可使用：

```bash
# 宿主机上临时开放权限（重启后失效）
sudo sh -c 'echo 0 > /proc/driver/nvidia/params/RestrictProfilingToAdminUsers'
```

开放权限后生成报告：

```bash
ncu --set full -o /root/Learn-CUDA/kernel_profiling/<报告名> \
  /root/Learn-CUDA/build/kernel_profiling/<可执行文件名>
```

下载 `.ncu-rep` 文件后，用 Nsight Compute 打开：https://developer.nvidia.com/nsight-compute

可额外查看内容（相比 nsys 更深入）：
- 单 kernel 硬件指标（Warp 占用率、内存带宽利用率等）
- Roofline Model
- Memory Chart

---

## 快速参考

```bash
# 添加 nsys 到当前会话 PATH（可写入 ~/.bashrc 永久生效）
export PATH="/opt/nvidia/nsight-compute/2024.1.1/host/target-linux-x64:$PATH"

# 之后直接使用
nsys profile -o <报告名> <可执行文件>
```
