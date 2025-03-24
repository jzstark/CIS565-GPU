#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// You cannot change this file type to CUDA/C++; there will be errors. 

/*

Matrix sizes:
MxK * KxN = MxN

*/


/* Kernel 1: Naive */

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float* A,
    const float* B, float beta, float* C) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // if statement is necessary to make things work under tile quantization
    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        // C = α*(A@B)+β*C
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}


/* Kernel 2: GMEM Coalescing */

#include <cassert>

template <const unsigned int BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
    const float* A, const float* B,
    float beta, float* C) {
    const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    // if statement is necessary to make things work under tile quantization
    if (cRow < M && cCol < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[cRow * K + i] * B[i * N + cCol];
        }
        C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
    }
}


/* Kernel 3: Shared Memory Cache-Blocking */

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
    const float* A, const float* B,
    float beta, float* C) {
    // the output block that we want to compute in this threadblock
    const uint32_t cRow = blockIdx.x;
    const uint32_t cCol = blockIdx.y;

    // allocate buffer for current block in fast shared mem
    // shared mem is shared between all threads in a block
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // the inner row & col that we're accessing in this thread
    const uint32_t threadCol = threadIdx.x % BLOCKSIZE;
    const uint32_t threadRow = threadIdx.x / BLOCKSIZE;

    // advance pointers to the starting positions
    A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
    B += cCol * BLOCKSIZE;                        // row=0, col=cCol
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

    float tmp = 0.0;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        // Have each thread load one of the elements in A & B
        // Make the threadCol (=threadIdx.x) the consecutive index
        // to allow global memory access coalescing
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

        // block threads in this block until cache is fully populated
        __syncthreads();
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        // execute the dotproduct on the currently cached block
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[threadRow * BLOCKSIZE + dotIdx] *
                Bs[dotIdx * BLOCKSIZE + threadCol];
        }
        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        __syncthreads();
    }
    C[threadRow * N + threadCol] =
        alpha * tmp + beta * C[threadRow * N + threadCol];
}