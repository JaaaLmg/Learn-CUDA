# ============================================================
# cuda_sgemm: 矩阵乘法（SGEMM）学习工程
# ============================================================

# ---------- 定位 BLAS 库（提供 cblas_sgemm）----------
# 优先查找 openblas / cblas / blas 的开发软链；
# 本机只装了运行时库 libblas.so.3（无 .so 软链，find_library 找不到），
# 因此提供 fallback，并允许命令行用 -DSGEMM_BLAS_LIBRARY=... 覆盖。
#
# 注意：不要用 if(NOT DEFINED ...) 包住 find_library。find_library 失败时会把
# NOTFOUND 写进缓存，下次 reconfigure（如 VSCode 复用同一 build 目录）该变量已
# DEFINED，会跳过 fallback 而直接拿到 NOTFOUND。正确做法是每次都判断变量真值，
# 并在 fallback 时用 CACHE ... FORCE 覆盖被污染的缓存。
find_library(SGEMM_BLAS_LIBRARY NAMES openblas cblas blas)
if(NOT SGEMM_BLAS_LIBRARY)
    set(_blas_fallback /usr/lib/x86_64-linux-gnu/libblas.so.3)
    if(EXISTS ${_blas_fallback})
        set(SGEMM_BLAS_LIBRARY ${_blas_fallback}
            CACHE FILEPATH "BLAS library providing cblas_sgemm" FORCE)
    else()
        message(FATAL_ERROR
            "未找到 BLAS 库。请用 -DSGEMM_BLAS_LIBRARY=/path/to/libblas.so.3 指定。")
    endif()
endif()
message(STATUS "cuda_sgemm 使用的 BLAS 库: ${SGEMM_BLAS_LIBRARY}")


# test_mmult 是测试驱动，它调用：
#   - REF_MMult  (ref_mmult.cpp)        : CBLAS 参考实现，作为标准答案
#   - MY_MMult   (sgemm_baseline_cpu.cpp): 待验证的 CPU 朴素实现
#   - random_matrix / compare_matrices  : 生成随机矩阵、比较结果
# 驱动里用到 cudaEvent 计时和设备查询，因此仍需链接 CUDA runtime。
add_executable(sgemm_cpu
    test_mmult.cpp
    ref_mmult.cpp
    sgemm_baseline_cpu.cpp
    random_matrix.cpp
    compare_metrics.cpp
)
target_include_directories(sgemm_cpu PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(sgemm_cpu PRIVATE
    CUDA::cudart
    ${SGEMM_BLAS_LIBRARY}
)
