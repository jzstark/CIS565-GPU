#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }


        __global__ void kern_scan_up(int n, int* odata, const int* idata, int d) {
			// TODO
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= n || index < 0) return;

            if (index % int(pow(2, d+1)) == 0) {
                //printf("(%d) index is %d: %d -- %d \n", d, index, index + int(pow(2, d + 1)) - 1, index + int(pow(2, d)) - 1);
                odata[index + int(pow(2, d + 1)) - 1] = idata[index + int(pow(2, d + 1)) - 1] + idata[index + int(pow(2, d)) - 1];
			}
        
        }

        __global__ void kern_scan_down(int n, int* odata, const int* idata, int d) {
            // TODO
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n || index < 0) return;

            int i, j; 
            if (index % int(pow(2,d+1)) == 0) {
                i = index + int(pow(2, d)) - 1;
                j = index + int(pow(2, d + 1)) - 1;
				//t = idata[i];
                odata[i] = idata[j];
                odata[j] = idata[i] + idata[j];
            }
        }

        __global__ void setElement(int* arr, int targetIdx, int value) {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;

            if (tid == 0) {
                arr[targetIdx] = value;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO

			const int n_power = nextPow2(n);

            int* d_odata;
            int* d_tmp_a;
            int* d_tmp_b;

            int* d_swap;
            const int D = ilog2ceil(n);
            cudaMalloc((void**)&d_odata, n * sizeof(int));
            cudaMalloc((void**)&d_tmp_a, n * sizeof(int));
            cudaMalloc((void**)&d_tmp_b, n * sizeof(int));

            // Copy data to gpu
            cudaMemcpy(d_tmp_a, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_tmp_b, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // Launch kernel
            int blockSize = 256;
            int numBlocks = (n + blockSize - 1) / blockSize;


            for (int d = 0; d <= D - 1; d++) {
                kern_scan_up <<< numBlocks, blockSize >>> (n, d_tmp_b, d_tmp_a, d);
                cudaMemcpy(d_tmp_a, d_tmp_b, n * sizeof(int), cudaMemcpyDeviceToDevice);
            }

			setElement <<< 1, 1 >>> (d_tmp_a, n - 1, 0);

            for (int d = D - 1; d >= 0; d--) {
                kern_scan_down <<< numBlocks, blockSize >> > (n, d_tmp_b, d_tmp_a, d);
                cudaMemcpy(d_tmp_a, d_tmp_b, n * sizeof(int), cudaMemcpyDeviceToDevice);
            }

            cudaMemcpy(odata, d_tmp_b, n * sizeof(int), cudaMemcpyDeviceToHost);

            //TODO: performance limit
            for (int i = 0; i < n - 1; i++) {
                odata[i] = odata[i + 1];
            }

            // Free gpu memory
            //cudaFree(d_idata);
            cudaFree(d_odata);
            cudaFree(d_tmp_b);
            cudaFree(d_tmp_a);
            timer().endGpuTimer();
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
