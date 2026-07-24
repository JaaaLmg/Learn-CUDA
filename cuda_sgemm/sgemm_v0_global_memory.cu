#include <stdlib.h>

// CUDA runtime
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <int BLOCK>
__global__ void sgemm(int m, int n, int k, float *A, int lda, float *B, int ldb, float *C, int ldc)
{
    float *A_start = A + BLOCK * blockIdx.y * lda;
    float *B_start = B + BLOCK * blockIdx.x;
    unsigned int x = BLOCK * blockIdx.x + threadIdx.x;
    unsigned int y = BLOCK * blockIdx.y + threadIdx.y;

    float temp = 0.0f;

    for (int i = 0; i < k; i++)
    {
        temp += A_start[threadIdx.y * lda + i] * B_start[i * ldb + threadIdx.x];
    }
    C[y * ldc + x] = temp;
}

void MY_MMult_v0(int m, int n, int k, float *A, int lda, float *B, int ldb, float *C, int ldc)
{
    constexpr int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    sgemm<BLOCK_SIZE><<<grid, block>>>(m, n, k, A, lda, B, ldb, C, ldc);
}