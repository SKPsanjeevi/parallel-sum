*** PARALLEL SUM ***

Performance data on RTX3050Mobile for a vector of size N. In our case, N = 16 million elements (4096^2).

The max ceiling performance of a RTX3050 Mobile is 5.501 TFLOPS (for FP32) and global memory bandwidth of 192 GBytes/s.
Source : https://www.techpowerup.com/gpu-specs/geforce-rtx-3050-mobile.c3788

Commands for building and profiling:

`nvcc par-sum.cu -o par-sum -arch=sm_80 -lcublas`

`nsys profile -o nsys_par-sum --stats=true ./par-sum`

`ncu -o ncu_par-sum  -f ./par-sum`

The time-complexity of this reduction operation  is O(N) and there are N floating point operations to be done for this case. Given theoretical peak performance (or the ROOF in roofline model), we can compute relative performance if TIME taken by the kernel is known using

$$ \text{AGAINSTROOF [PERCENT]} = \frac{N \text{[FLOP]} } {\text{TIME [s] *  GPUPEAKPERFORMANCE [FLOPs]}} * 100 [\text{PERCENT}] $$

CUBLAS time taken = 0.415 ms (uses `asum_kernel` twice as seen from `nsys` data.). In this case, CUBLAS uses about roughly 0.735 % of the roofline performance for this GPU. It has to be noted that the execution times of CUBLAS kernel is actually used rather than the time taken by the CUBLAS function itself. The CUBLAS functions for the reduction operations take much longer time due to branching out and to call the specific kernel needed.

Regarding bandwidth, N amount of floats are transferred to the SMs but resulting in only one float output. Therefore, the bandwidth can be approximated as N * 4 [BYTE] / (TIME [s]).


|KERNEL    		|BANDWIDTH (GB/s)    	|TIME (ms) 	|AGAINST_CUBLAS (%)
| --------------------- | --------------------- | ------------- | ----------------
|NAIVE (GPU serial)	|1.34			|49.78		|0.83
|Shared mem		|35.13			|1.91		|21.72
|Halve the blocks (1/2)	|67.92			|0.988		|42.00
|Even fewer blocks (1/8)|169.46			|0.396		|104.7

