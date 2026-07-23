#include <cstdio>
#include <cmath>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256


__global__ void reduce2(float *d_in, float *d_out) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int i = bid * blockDim.x + tid;
    
    // allocate shared memory
    __shared__ float sdata[THREAD_PER_BLOCK];
    sdata[tid] = d_in[i];
    __syncthreads();

    // do reduction, try not to cause divergence branch
    for(int r=1; r<blockDim.x; r*=2) {
        if(tid < blockDim.x / (2 * r)) {
            int dataIdx = tid * (2 * r);
            sdata[dataIdx] += sdata[dataIdx + r];
        }
        __syncthreads();
    }

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

    int block_num = N/THREAD_PER_BLOCK;
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
        for(int j=0; j<THREAD_PER_BLOCK; j++) {
            curr += arr[i*THREAD_PER_BLOCK+j];
        }
        res[i] = curr;
    }

    // calculate gpu
    cudaMemcpy(d_in, arr, N*sizeof(float), cudaMemcpyHostToDevice);
    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);
    reduce2<<<Grid, Block>>>(d_in, d_out);
    cudaMemcpy(out, d_out, block_num*sizeof(float), cudaMemcpyDeviceToHost);

    // print result
    if(check(out, res, block_num)) printf("answers are all the same\n");
    else printf("some answer went wrong\n");

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}