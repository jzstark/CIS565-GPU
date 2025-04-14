#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
		__global__ void kern_scan(int n, int* odata, const int* idata, int d) {
			// TODO
            int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= n) return;
            /* if (index == 0) {
                odata[0] == idata[0];
                return;
            } */

            if (index >= pow(2, d-1)) {
                odata[index] = idata[index] + idata[index - int(pow(2, d - 1))];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
			// Allocate gpu memory
			
            //int* d_idata;
			int* d_odata;
            int* d_tmp_a;
            int* d_tmp_b;

            int* d_swap;
			const int D = ilog2ceil(n);
			//cudaMalloc((void**)&d_idata, n * sizeof(int));
			cudaMalloc((void**)&d_odata, n * sizeof(int));
            cudaMalloc((void**)&d_tmp_a, n * sizeof(int));
            cudaMalloc((void**)&d_tmp_b, n * sizeof(int));

			// Copy data to gpu
			// cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_tmp_a, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_tmp_b, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // initialize d_tmp_b to 0s
            // cudaMemset(d_tmp_b, 0, n * sizeof(int));

			// Launch kernel
			int blockSize = 256;
			int numBlocks = (n + blockSize - 1) / blockSize;
            
            // d_tmp_b[0] = d_tmp_a[0];
            // int flag = 0; 
            for (int d = 1; d <= D; d++) {
                kern_scan <<< numBlocks, blockSize >> > (n, d_tmp_b, d_tmp_a, d);

                //print odata
                /* cudaMemcpy(odata, d_tmp_b, n * sizeof(int), cudaMemcpyDeviceToHost);

                printf("d_tmp_b # %d: ", d);
                for (int i = 0; i < n; i++) {
                    printf("%d ", odata[i]);
                }
                printf("\n"); */
                
                // swap(d_odata, d_idata)
                // d_swap = d_tmp_a;
                // d_tmp_a = d_tmp_b;
                // d_tmp_b = d_swap;

				//TODO: I think this is very inefficient to copy the whole array back and forth
                cudaMemcpy(d_tmp_a, d_tmp_b, n * sizeof(int), cudaMemcpyDeviceToDevice);
                //flag += 1;
            }

            /* if (flag % 2 == 0) {
                // Copy result back to cpu
                cudaMemcpy(odata, d_tmp_b, n * sizeof(int), cudaMemcpyDeviceToHost);
            } else {
                cudaMemcpy(odata, d_tmp_a, n * sizeof(int), cudaMemcpyDeviceToHost);
            } */

            cudaMemcpy(odata, d_tmp_b, n * sizeof(int), cudaMemcpyDeviceToHost);

            //print odata
			/* printf("odata: ");
            for (int i = 0; i < n; i++) {
				printf("%d ", odata[i]);
            }
			printf("\n"); */

			// Free gpu memory
			//cudaFree(d_idata);
			cudaFree(d_odata);
            cudaFree(d_tmp_b);
            cudaFree(d_tmp_a);
            timer().endGpuTimer();
        }

    }
}
