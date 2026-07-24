# cuda_sgemm CPU 验证：构建步骤与注意事项

本文件记录如何用 CMake 把 `cuda_sgemm/` 下的矩阵乘法（SGEMM）代码组织起来，
并完成 **CPU 朴素实现的正确性验证**（对拍 CBLAS 参考实现）。供日后参考。

---

## 一、文件职责

| 文件 | 作用 |
| --- | --- |
| `test_mmult.cpp` | 测试驱动 `main`：生成随机矩阵 → 调参考实现 → 调待测实现 → 比较结果 |
| `ref_mmult.cpp` | `REF_MMult`：调用 CBLAS `cblas_sgemm` 作为**标准答案** |
| `sgemm_baseline_cpu.cpp` | `MY_MMult`：**待验证的 CPU 三重循环朴素实现** |
| `random_matrix.cpp` | `random_matrix`：用 `drand48` 填充 [-1,1] 随机矩阵 |
| `compare_metrics.cpp` | `compare_matrices`：逐元素比较，返回最大绝对误差 |
| `parameters.h` | 测试规模参数（矩阵尺寸范围、重复次数） |
| `sgemm_v0_global_memory.cu` | GPU 版本骨架（当前仅占位，后续填充 kernel） |

数据布局约定：所有矩阵 **行主序（row-major）**，`X(i,j) = x[i*ldX + j]`。
驱动中 `lda=k, ldb=n, ldc=n`（C = A·B，A 是 m×k，B 是 k×n，C 是 m×n）。

---

## 二、构建前需要处理的 4 个问题（关键！）

代码开箱即用会有几个坑，已在提交中修复，原理如下：

### 1. CPU 实现的索引 bug（逻辑错误）

原始 `sgemm_baseline_cpu.cpp` 写成 `C[r*k+c]` 且用 `A[r*k+i]`、`B[i*n+c]` 硬编码维度。
- C 是 m×n，其列步长应是 **n（=ldc）**，写 `r*k+c` 在 `k≠n` 时越界/算错。
- 应统一使用传入的 leading dimension，改为：
  ```cpp
  temp += A[r * lda + i] * B[i * ldb + c];
  C[r * ldc + c] = temp;
  ```
- 当前测试恰好 m=n=k（方阵），bug 被掩盖；但非方阵时必错。**永远用 ld 参数，不要硬编码维度。**

### 2. 缺少 `cblas.h` 头文件

本机只装了运行时库 `libblas.so.3`（`nm -D` 可见它导出了 `cblas_sgemm` 符号），
但没有开发头 `cblas.h`，`apt` 也装不了 `libblas-dev`。
- 解决：在 `ref_mmult.cpp` 里用 `extern "C"` **自行声明** `CBLAS_ORDER`、`CBLAS_TRANSPOSE`
  枚举和 `cblas_sgemm` 原型（取值遵循 CBLAS 标准），不 `#include <cblas.h>`。
- 链接期由 `libblas.so.3` 提供符号即可。

### 3. 缺少 `helper_cuda.h`（`checkCudaErrors`）

`test_mmult.cpp` 用了 CUDA samples 的 `checkCudaErrors`，但本机没装该头文件。
- 解决：在 `test_mmult.cpp` 顶部自定义一个等价宏（检查 `cudaError_t`，出错打印文件行号并退出）。

### 4. 测试规模过大

原 `parameters.h`：`PFIRST=1024, PLAST=4096, NREPEATS=20`。
CPU 朴素三重循环 O(n³)，跑 4096³ × 20 次要几十分钟。
- 初步验证阶段调小为 `PFIRST=256, PLAST=512, PINC=128, NREPEATS=3`，秒级完成。
- 正式性能测试再调回大尺寸。

---

## 三、CMakeLists 组织方式

`cuda_sgemm/CMakeLists.txt` 定义两个目标：
- **`sgemm_cpu`**：聚合上述 5 个 `.cpp`，链接 `CUDA::cudart`（驱动用到 cudaEvent 计时/设备查询）
  和 BLAS 库。
- `sgemm_v0_global_memory`：GPU 骨架，保留原样。

BLAS 定位逻辑（关键）：本机 `libblas.so.3` 没有 `.so` 软链，`find_library` 默认找不到，
所以先 `find_library(NAMES openblas cblas blas)`，失败则 fallback 到
`/usr/lib/x86_64-linux-gnu/libblas.so.3`，并允许 `-DSGEMM_BLAS_LIBRARY=...` 覆盖。

> 注意：根目录 `CMakeLists.txt` 目前**没有** `find_package(BLAS REQUIRED)`，
> 所以不再需要 BUILD.md 里提到的 `-DBLAS_LIBRARIES=...`（那是旧状态）。

---

## 四、构建与运行步骤

在项目根目录 `/root/Learn-CUDA` 执行：

```bash
# 1. 确保 nvcc 在 PATH（本机 CUDA 装在 /usr/local/cuda）
export PATH=/usr/local/cuda/bin:$PATH

# 2. 配置
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# 3. 只编译 CPU 验证目标
cmake --build build --target sgemm_cpu

# 4. 运行
./build/cuda_sgemm/sgemm_cpu
```

### 预期输出

```
GPU Device 0: "NVIDIA GeForce RTX 2080 Ti" with compute capability 7.5

256 1.66 0.000000e+00
384 1.62 0.000000e+00
512 1.35 0.000000e+00
];
```

三列含义：`矩阵尺寸 p`、`GFLOPS`、`与参考实现的最大误差 diff`。
**`diff = 0.000000e+00` 表示 CPU 实现与 CBLAS 结果完全一致，验证通过。**
驱动内还有阈值检查：`diff > 0.5` 会打印 `diff too big !` 并退出。

---

## 五、环境信息（本机实测）

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 2080 Ti（sm_75）|
| CUDA Toolkit | 12.4（`/usr/local/cuda`）|
| BLAS | `/usr/lib/x86_64-linux-gnu/libblas.so.3`（仅运行时库，无头文件）|

---

## 六、后续 TODO

- 在 `sgemm_v0_global_memory.cu` 实现真正的 GPU kernel，并把 `test_mmult.cpp` 里
  `#define USE_CPU` 注释掉，走 GPU 分支（`MY_MMult(m,n,k,d_A,...,d_C,n)` 直接对设备指针操作）。
- GPU 分支需要一个接收 device 指针的 `MY_MMult` 实现（含 kernel 启动），
  并在算完后 `cudaMemcpy` 回 `cold` 再对拍。
- 恢复 `parameters.h` 大尺寸做性能对比。
