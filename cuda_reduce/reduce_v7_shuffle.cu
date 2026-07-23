#include <cstdio>
#include <cmath>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32


template <unsigned int blockSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (blockSize >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (blockSize >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
    if (blockSize >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);  // 0-4, 1-5, 2-6, etc.
    if (blockSize >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
    if (blockSize >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1);  // 0-1, 2-3, 4-5, etc.
    return sum;
}

template <int blockSize, int num_per_thread>
__global__ void reduce7(float *d_in, float *d_out) {
    float sum = 0;

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int i = bid * blockSize * num_per_thread + tid;

    #pragma unroll
    for(int iter=0; iter<num_per_thread; iter++){
        sum += d_in[i+iter*blockSize];
    }
    
    // allocate shared memory
    static __shared__ float warpLevelSums[WARP_SIZE]; 
    const int laneId = tid % WARP_SIZE;
    const int warpId = tid / WARP_SIZE;

    sum = warpReduceSum<blockSize>(sum);

    if(laneId == 0) warpLevelSums[warpId] = sum;
    __syncthreads();

    // read from shared memory only if that warp existed
    sum = (tid < blockDim.x / WARP_SIZE) ? warpLevelSums[laneId] : 0;

    // Final reduce using first warp
    if (warpId == 0) sum = warpReduceSum<blockSize/WARP_SIZE>(sum); 

    // write result for this block to global mem
    if (tid == 0) d_out[blockIdx.x] = sum;
}

bool check(float *out, float *res, int block_num) {
    for(int i=0; i<block_num; i++) {
        if(fabs(out[i] - res[i]) > 1e-4) return false;
    }
    return true;
}

int main()
{
    // allocate memory for data
    const int N = 32*1024*1024;
    float *arr = (float *)malloc(N*sizeof(float));
    float *d_in;
    cudaMalloc((void **)&d_in, N*sizeof(float));

    const int block_num = 1024;
    const int num_per_block = N / block_num;
    const int num_per_thread = num_per_block / THREAD_PER_BLOCK;
    float *out = (float *)malloc(block_num*sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out, block_num*sizeof(float));

    float *res = (float *)malloc(block_num*sizeof(float));

    for(int i=0; i<N; i++) {    // init data
        arr[i] = 2.0 * (float)drand48() - 1.0;
    }

    // calculate on cpu (use double to avoid large sequential float accumulation error)
    for(int i=0; i<block_num; i++) {
        double curr = 0;
        for(int j=0; j<num_per_block; j++) {
            curr += arr[i*num_per_block+j];
        }
        res[i] = (float)curr;
    }

    // calculate on gpu
    cudaMemcpy(d_in, arr, N*sizeof(float), cudaMemcpyHostToDevice);
    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);
    reduce7<THREAD_PER_BLOCK, num_per_thread><<<Grid, Block>>>(d_in, d_out);
    cudaMemcpy(out, d_out, block_num*sizeof(float), cudaMemcpyDeviceToHost);

    // print result
    if(check(out, res, block_num)) printf("answers are all the same\n");
    else printf("some answer went wrong\n");

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}