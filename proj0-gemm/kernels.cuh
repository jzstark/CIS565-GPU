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

/* Kernel 4. */

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,
    const float* A, const float* B, float beta,
    float* C) {
    // If we flip x and y here we get ~30% less performance for large matrices.
    // The current, 30% faster configuration ensures that blocks with sequential
    // blockIDs access columns of B sequentially, while sharing the same row of A.
    // The slower configuration would share columns of A, but access into B would
    // be non-sequential. So the faster configuration has better spatial locality
    // and hence a greater L2 hit rate.
    const uint32_t cRow = blockIdx.y;
    const uint32_t cCol = blockIdx.x;

    // each warp will calculate 32*TM elements, with 32 being the columnar dim.
    const int threadCol = threadIdx.x % BN;
    const int threadRow = threadIdx.x / BN;

    // allocate space for the current blocktile in SMEM
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // todo: adjust this to each thread to load multiple entries and
    // better exploit the cache sizes
    assert(BM * BK == blockDim.x);
    assert(BN * BK == blockDim.x);
    const uint32_t innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
    const uint32_t innerRowA = threadIdx.x / BK;
    const uint32_t innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
    const uint32_t innerRowB = threadIdx.x / BN;

    // allocate thread-local cache for results in registerfile
    float threadResults[TM] = { 0.0 };

    // outer loop over block tiles
    for (uint32_t bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // populate the SMEM caches
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();

        // advance blocktile
        A += BK;
        B += BK * N;

        // calculate per-thread results
        for (uint32_t dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // we make the dotproduct loop the outside loop, which facilitates
            // reuse of the Bs entry, which we can cache in a tmp var.
            float tmpB = Bs[dotIdx * BN + threadCol];
            for (uint32_t resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] +=
                    As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }

    // write out the results
    for (uint32_t resIdx = 0; resIdx < TM; ++resIdx) {
        C[(threadRow * TM + resIdx) * N + threadCol] =
            alpha * threadResults[resIdx] +
            beta * C[(threadRow * TM + resIdx) * N + threadCol];
    }
}



/* Kernel 10: Warptiling */
//TODO: this implementatio works like shit on 4090. ~150 GFlops at most. Using CuBLAS can reach 5.3TFLOPS, out of the theoretical 8TFLOPS. 
// It will be interesting to investigate the reason. 

const int WARPSIZE = 32; // warpSize is not constexpr

namespace wt {
    template <const int BM, const int BN, const int BK, const int rowStrideA,
        const int rowStrideB>
    __device__ void loadFromGmem(int N, int K, const float* A, const float* B,
        float* As, float* Bs, int innerRowA, int innerColA,
        int innerRowB, int innerColB) {
        for (uint32_t offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            const float4 tmp = reinterpret_cast<const float4*>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            // float4 tmp;
            // asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
            //     : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
            //     : "l"(&A[(innerRowA + offset) * K + innerColA * 4]));
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        for (uint32_t offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            reinterpret_cast<float4*>(
                &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                reinterpret_cast<const float4*>(
                    &B[(innerRowB + offset) * N + innerColB * 4])[0];
            // asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
            //     : "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 0]),
            //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 1]),
            //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 2]),
            //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 3])
            //     : "l"(&B[(innerRowB + offset) * N + innerColB * 4]));
        }
    }

    template <const int BM, const int BN, const int BK, const int WM, const int WN,
        const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
        const int TM, const int TN>
    __device__ void
        processFromSmem(float* regM, float* regN, float* threadResults, const float* As,
            const float* Bs, const uint32_t warpRow, const uint32_t warpCol,
            const uint32_t threadRowInWarp, const uint32_t threadColInWarp) {
        for (uint32_t dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // populate registers for whole warptile
            for (uint32_t wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (uint32_t i = 0; i < TM; ++i) {
                    regM[wSubRowIdx * TM + i] =
                        As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
                        threadRowInWarp * TM + i];
                }
            }
            for (uint32_t wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                for (uint32_t i = 0; i < TN; ++i) {
                    regN[wSubColIdx * TN + i] =
                        Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
                        threadColInWarp * TN + i];
                }
            }

            // execute warptile matmul
            for (uint32_t wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (uint32_t wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    // calculate per-thread results
                    for (uint32_t resIdxM = 0; resIdxM < TM; ++resIdxM) {
                        for (uint32_t resIdxN = 0; resIdxN < TN; ++resIdxN) {
                            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                (wSubColIdx * TN) + resIdxN] +=
                                regM[wSubRowIdx * TM + resIdxM] *
                                regN[wSubColIdx * TN + resIdxN];
                        }
                    }
                }
            }
        }
    }

} // namespace wt

/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMITER The number of subwarp tiling steps in M dimension.
 * @tparam WNITER The number of subwarp tiling steps in N dimension.
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 */
template <const int BM, const int BN, const int BK, const int WM, const int WN,
    const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
sgemmWarptiling(int M, int N, int K, float alpha, float* A, float* B,
    float beta, float* C) {
    const uint32_t cRow = blockIdx.y;
    const uint32_t cCol = blockIdx.x;

    // Placement of the warp in the threadblock tile
    const uint32_t warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
    const uint32_t warpCol = warpIdx % (BN / WN);
    const uint32_t warpRow = warpIdx / (BN / WN);

    // size of the warp subtile
    constexpr uint32_t WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint32_t WSUBM = WM / WMITER; // 64/2=32
    constexpr uint32_t WSUBN = WN / WNITER; // 32/2=16

    // Placement of the thread in the warp subtile
    const uint32_t threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]
    const uint32_t threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
    const uint32_t threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4

    // allocate space for the current blocktile in SMEM
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    // Move C_ptr to warp's output tile
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step
    const uint32_t innerRowA = threadIdx.x / (BK / 4);
    const uint32_t innerColA = threadIdx.x % (BK / 4);
    constexpr uint32_t rowStrideA = (NUM_THREADS * 4) / BK;
    const uint32_t innerRowB = threadIdx.x / (BN / 4);
    const uint32_t innerColB = threadIdx.x % (BN / 4);
    constexpr uint32_t rowStrideB = NUM_THREADS / (BN / 4);

    // allocate thread-local cache for results in registerfile
    float threadResults[WMITER * TM * WNITER * TN] = { 0.0 };
    // we cache into registers on the warptile level
    float regM[WMITER * TM] = { 0.0 };
    float regN[WNITER * TN] = { 0.0 };

    // outer-most loop over block tiles
    for (uint32_t bkIdx = 0; bkIdx < K; bkIdx += BK) {
        wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
            N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
        __syncthreads();
        wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
            TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol,
                threadRowInWarp, threadColInWarp);
        A += BK;     // move BK columns to right
        B += BK * N; // move BK rows down
        __syncthreads();
    }

    // write out the results
    for (uint32_t wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint32_t wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            // move C pointer to current warp subtile
            float* C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
            for (uint32_t resIdxM = 0; resIdxM < TM; resIdxM += 1) {
                for (uint32_t resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    // load C vector into registers
                    float4 tmp = reinterpret_cast<float4*>(
                        &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                        threadColInWarp * TN + resIdxN])[0];
                    // perform GEMM update in reg
                    const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
                    tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
                    tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
                    tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
                    tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
                    // write back
                    reinterpret_cast<float4*>(
                        &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                        threadColInWarp * TN + resIdxN])[0] = tmp;
                }
            }
        }
    }
};