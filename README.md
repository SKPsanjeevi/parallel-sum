*** PARALLEL SUM ***

Performance data on RTX3050Mobile for a vector of size N. In our case, N = 16 million elements (4096^2).

The max theoretical performance of a RTX3050 Mobile is 5.501 TFLOPS (for FP32) and global memory bandwidth of 192 GBytes/s.
Source : https://www.techpowerup.com/gpu-specs/geforce-rtx-3050-mobile.c3788

Commands for building and profiling:

`nvcc par-sum.cu -o par-sum -arch=sm_80 -lcublas`

`nsys profile -o nsys_par-sum --stats=true ./par-sum`

`ncu -o ncu_par-sum  -f ./par-sum`

The time-complexity of this reduction operation  is O(N) and there are N floating point operations to be done for this case. Given theoretical peak performance (or the ROOF in roofline model), we can compute relative performance if TIME taken by the kernel is known using

$$ \text{AGAINSTROOF [PERCENT]} = \frac{N \text{[FLOP]} } {\text{TIME [s] *  GPUPEAKPERFORMANCE [FLOPs]}} * 100 [\text{PERCENT}] $$

CUBLAS time taken = 0.4 ms (uses `asum_kernel` as seen from `nsys` data). In this case, CUBLAS is about roughly 0.73% of the roofline performance for this GPU.


|KERNEL    		|BANDWIDTH     	|TIME (ms) 	|AGAINST_CUBLAS (%)
| --------------------- | ------------- | ------------- | ----------------

