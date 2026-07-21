# CUDA 项目构建与运行说明

本项目使用 CMake 构建 CUDA 代码。以下记录在本机（Tesla V100，CUDA 12.8）上从零构建并运行 `cuda_reduce/reduce_v0_global_memory.cu` 的完整步骤。

## 环境信息

| 项目 | 版本 / 值 |
| --- | --- |
| GPU | Tesla V100-PCIE-32GB（架构 sm_70）|
| NVIDIA 驱动 | 580.65.06 |
| CUDA Toolkit | 12.8（安装于 `/usr/local/cuda-12.8`）|
| CMake | 3.22.1 |
| 操作系统 | Linux |

## 前置准备

### 1. 把 nvcc 加入 PATH

系统默认 PATH 里没有 `nvcc`，需要先加入：

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
```

> 建议把这行写进 `~/.bashrc`，避免每次手动设置。

### 2. 解决 BLAS 依赖

`cuda_reduce/CMakeLists.txt` 里有 `find_package(BLAS REQUIRED)`，但本机只装了运行时库 `libblas.so.3`，缺少开发用的软链，且 apt 无法安装 `libblas-dev`。

解决办法：配置时直接把库路径传给 CMake（不改动系统或构建脚本）：

```bash
-DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libblas.so.3
```

## 构建步骤

在项目根目录 `/root/Learn-CUDA` 下执行：

```bash
# 确保 nvcc 在 PATH 中
export PATH=/usr/local/cuda-12.8/bin:$PATH

# 1. 配置（生成 build 目录）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libblas.so.3

# 2. 编译
cmake --build build
```

## 运行

可执行文件生成在 `build/cuda_reduce/` 下：

```bash
./build/cuda_reduce/reduce_v0_global_memory
```

预期输出：

```
hello cuda
```

## 常见问题

- **`Could NOT find BLAS`**：见上文「解决 BLAS 依赖」，配置时加 `-DBLAS_LIBRARIES=...`。
- **`nvcc: command not found`**：未把 `/usr/local/cuda-12.8/bin` 加入 PATH。
- **`nvcc warning ... architectures prior to sm_75 will be removed`**：仅为弃用警告，V100（sm_70）当前仍可正常编译运行，可忽略。

## 清理重建

```bash
rm -rf build
```

删除后重复上面的「构建步骤」即可。
