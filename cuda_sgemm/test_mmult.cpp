#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parameters.h"

#include <chrono>

// CUDA runtime
#include <cublas_v2.h>
#include <cuda_runtime.h>

// 本机没有 CUDA samples 的 helper_cuda.h，这里提供一个等价的轻量宏：
// 检查 CUDA runtime 调用返回值，出错时打印位置并退出。
#define checkCudaErrors(call)                                               \
    do                                                                      \
    {                                                                       \
        cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess)                                           \
        {                                                                   \
            fprintf(stderr, "CUDA error %s:%d: '%s'\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err__));                             \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)


#define FILE_ROOT "/root/Learn-CUDA/cuda_sgemm/results"

// #define USE_CPU
#define USE_CUDA_V0

#if defined(USE_CUDA_V0)
#define SGEMM_VERSION "v0"
#endif


void REF_MMult(int, int, int, float *, int, float *, int, float *, int);
void MY_MMult_baseline(int, int, int, float *, int, float *, int, float *, int);
void MY_MMult_v0(int, int, int, float *, int, float *, int, float *, int);
void MY_MMult(int, int, int, float *, int, float *, int, float *, int);
// void copy_matrix(int, int, float *, int, float *, int);
void random_matrix(int, int, float *, int);
float compare_matrices(int, int, float *, int, float *, int);

double dclock();

int main()
{
    // print gpu info
    cudaDeviceProp deviceProp;
    int devID = 0;
    checkCudaErrors(cudaSetDevice(devID));
    auto error = cudaGetDeviceProperties(&deviceProp, devID);
    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error,
               __LINE__);
        exit(EXIT_FAILURE);
    }
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID,
           deviceProp.name, deviceProp.major, deviceProp.minor);

    // 仅在 GPU 路径下将结果写入对应版本的文件中
    FILE *result_file = NULL;
#ifdef SGEMM_VERSION
    char result_filename[256];
    snprintf(result_filename, sizeof(result_filename),
             "%s/result_%s.txt", FILE_ROOT, SGEMM_VERSION);
    result_file = fopen(result_filename, "w");
    if (result_file == NULL)
    {
        fprintf(stderr, "Failed to open result file: %s\n", result_filename);
        exit(EXIT_FAILURE);
    }
    fprintf(result_file, "p gflops diff\n");
#endif

    int p, m, n, k, rep;

    double dtime, dtime_best, gflops, diff;

    float *a, *b, *c, *cref, *cold;

    /* Time the "optimized" implementation */
    cudaEvent_t start, stop;
    // Allocate CUDA events that we'll use for timing
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    for (p = PFIRST; p <= PLAST; p += PINC)
    {
        m = (M == -1 ? p : M);
        n = (N == -1 ? p : N);
        k = (K == -1 ? p : K);

        const int lda = k, ldb = n, ldc = n;

        /* Allocate space for the matrices */
        /* Note: I create an extra column in A to make sure that
           prefetching beyond the matrix does not cause a segfault */
        const size_t mem_size_A = m * k * sizeof(float);
        const size_t mem_size_B = k * n * sizeof(float);
        const size_t mem_size_C = m * n * sizeof(float);
        a = (float *)malloc(mem_size_A);
        b = (float *)malloc(mem_size_B);
        c = (float *)malloc(mem_size_C);
        cold = (float *)malloc(mem_size_C);
        cref = (float *)malloc(mem_size_C);

        /* Generate random matrices A, B, Cold */
        random_matrix(m, k, a, m);
        random_matrix(k, n, b, k);
        random_matrix(m, n, cold, n);
        memset(cold, 0, mem_size_C);
        memset(cref, 0, mem_size_C);

        /* Init device matrix*/
        float *d_A, *d_B, *d_C;
        checkCudaErrors(cudaMalloc((void **)&d_A, mem_size_A));
        checkCudaErrors(cudaMalloc((void **)&d_B, mem_size_B));
        checkCudaErrors(cudaMemcpy(d_A, a, mem_size_A, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_B, b, mem_size_B, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMalloc((void **)&d_C, mem_size_C));

        /* Run the reference implementation so the answers can be compared */

        REF_MMult(m, n, k, a, lda, b, ldb, cref, ldc);

        float msecTotal = 0.0f;

#ifdef USE_CPU
        // CPU 计算不在 GPU stream 上，用 std::chrono 的墙钟时间才准确。
        auto cpu_start = std::chrono::steady_clock::now();
        for (rep = 0; rep < NREPEATS; rep++)
        {
            MY_MMult_baseline(m, n, k, a, k, b, n, cold, n);
        }
        auto cpu_stop = std::chrono::steady_clock::now();
        msecTotal =
            std::chrono::duration<float, std::milli>(cpu_stop - cpu_start).count();
#endif

#ifdef USE_CUDA_V0
        // GPU 路径用 CUDA event 计时（记录 stream 上的时间戳）。
        checkCudaErrors(cudaEventRecord(start, NULL));
        for (rep = 0; rep < NREPEATS; rep++)
        {
            /* Time your implementation */
            MY_MMult_v0(m, n, k, d_A, k, d_B, n, d_C, n);
        }
        checkCudaErrors(cudaEventRecord(stop, NULL));
        // Wait for the stop event to complete
        checkCudaErrors(cudaEventSynchronize(stop));
        checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
#endif


        // Compute and print the performance
        float msecPerMatrixMul = msecTotal / NREPEATS;
        double flopsPerMatrixMul = 2.0 * m * k * n;
        gflops =
            (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);

#ifndef USE_CPU
        // copy result from device to host
        checkCudaErrors(cudaMemcpy(cold, d_C, mem_size_C, cudaMemcpyDeviceToHost));
#endif

        diff = compare_matrices(m, n, cold, ldc, cref, ldc);
        if (diff > 0.5f || diff < -0.5f)
        {
            printf("diff too big !\n");
            exit(-1);
        }
        printf("%d %.2f %le \n", p, gflops, diff);

        if (result_file != NULL)
        {
            fprintf(result_file, "%d %.2f %le\n", p, gflops, diff);
        }

        free(a);
        free(b);
        free(c);
        free(cold);
        free(cref);

        checkCudaErrors(cudaFree(d_A));
        checkCudaErrors(cudaFree(d_B));
        checkCudaErrors(cudaFree(d_C));
    }

    if (result_file != NULL)
    {
        fclose(result_file);
    }

    return 0;
}