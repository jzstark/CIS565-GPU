#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
			odata[0] = idata[0];
			for (int i = 1; i < n; i++) {
				odata[i] = idata[i] + odata[i-1];
			}
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int counter = 0;
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					odata[counter] = idata[i];
					counter += 1;
				}
			}
            timer().endCpuTimer();
            return counter - 1;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            //timer().startCpuTimer();
            // TODO
            int* tmp1 = new int[n];
            int* tmp2 = new int[n];

            for (int i = 0; i < n; i++) {
				tmp1[i] = idata[i] == 0 ? 0 : 1; 
            }
            scan(n, tmp2, tmp1);

            int idx = -1; 
            for (int i = 0; i < n; i++) {
                if (tmp1[i] == 1) {
                    idx = tmp2[i] - 1;
                    odata[idx] = idata[i];
                }
            }

            delete[] tmp1;
			delete[] tmp2;
            //timer().endCpuTimer();
            return idx;
        }
    }
}
