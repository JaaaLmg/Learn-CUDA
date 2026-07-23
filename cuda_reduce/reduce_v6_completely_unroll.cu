#include <cstdio>
#include <cmath>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256


// unroll the loop for the last warp
__device__ void warpReduce(volatile float *sdata, int tid) {     // sdata must be marked as volatile!
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

template <int blockSize>
__global__ void reduce6(float *d_in, float *d_out) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int i = bid * blockDim.x * 2 + tid;
    
    // allocate shared memory
    __shared__ float sdata[THREAD_PER_BLOCK];
    sdata[tid] = d_in[i] + d_in[i+THREAD_PER_BLOCK];
    __syncthreads();

    // do reduction in shared memory, try not to cause bank conflict
    // break the loop
    if(blockSize >= 512) {
        if(tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if(blockSize >= 256) {
        if(tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if(blockSize >= 128) {
        if(tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }

    if(tid < 32) warpReduce(sdata, tid);    // reduce in a single warp, no need to sychronize

    // output result
    if(tid==0) d_out[bid] = sdata[0];
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

    int num_per_block = THREAD_PER_BLOCK * 2;   // num of data reduced per block
    int block_num = N/num_per_block;
    float *out = (float *)malloc(block_num*sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out, block_num*sizeof(float));

    float *res = (float *)malloc(block_num*sizeof(float));

    for(int i=0; i<N; i++) {    // init data
        arr[i] = 2.0 * (float)drand48() - 1.0;
    }

    // calculate on cpu
    for(int i=0; i<block_num; i++) {
        float curr = 0;
        for(int j=0; j<num_per_block; j++) {
            curr += arr[i*num_per_block+j];
        }
        res[i] = curr;
    }

    // calculate on gpu
    cudaMemcpy(d_in, arr, N*sizeof(float), cudaMemcpyHostToDevice);
    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);
    reduce6<THREAD_PER_BLOCK><<<Grid, Block>>>(d_in, d_out);
    cudaMemcpy(out, d_out, block_num*sizeof(float), cudaMemcpyDeviceToHost);

    // print result
    if(check(out, res, block_num)) printf("answers are all the same\n");
    else printf("some answer went wrong\n");

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}