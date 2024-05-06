*** PARALLEL SUM ***

Performance data on RTX3050Mobile for a vector of size N. In our case, N = 16 million elements (4096^2).

The max ceiling performance of a RTX3050 Mobile is 5.501 TFLOPS (for FP32) and global memory bandwidth of 192 GBytes/s.
Source : https://www.techpowerup.com/gpu-specs/geforce-rtx-3050-mobile.c3788


The time-complexity of this reduction operation  is O(N) and there are N floating point operations to be done for this case. Given theoretical peak performance (or the ROOF in roofline model), we can compute relative performance if TIME taken by the kernel is known using

$$ \text{AGAINSTROOF [PERCENT]} = \frac{N \text{[FLOP]} } {\text{TIME [s] *  GPUPEAKPERFORMANCE [FLOPS]}} * 100 [\text{PERCENT}] $$


CUBLAS functions typically have various custom kernels that get called even depending on the GPU specifications. It has to be noted that the execution times of the final CUBLAS kernel is actually used rather than the time taken by the CUBLAS function itself. The CUBLAS functions take much longer time to call the specific kernel needed. This itself motivates users to write custom kernels.


In a real world scenario, the GPU adaptively chooses varying frequency which is typically higher than the base frequency. We DO NOT pin the clock frequency to base frequency for all kernel measurements as it might measure close to real-time execution speeds. The unpinned clock frequency can be achieved in `ncu` using `--clock-control none` option. By default, `ncu` pins the kernels to base frequncy whereas `nsys` seems to be working on unpinned frequencies.

PINNED CLOCK FREQUENCY : CUBLAS kernel time taken = 0.541 ms (uses `asum_kernel` twice as seen from `nsys` data.). In this case, CUBLAS uses about roughly 0.564 % of the roofline performance for this GPU.

UNPINNED CLOCK FREQUENCY : CUBLAS kernel time taken = 0.415 ms (uses `asum_kernel` twice as seen from `nsys` data.). In this case, CUBLAS uses about roughly 0.735 % of the roofline performance for this GPU.


Regarding bandwidth, N amount of floats are transferred to the SMs but resulting in only one float output. Therefore, the bandwidth can be approximated as N * 4 [BYTE] / (TIME [s]).

Commands for building and profiling:

`nvcc par-sum.cu -o par-sum -arch=sm_80 -lcublas`

`nsys profile -o nsys_par-sum --stats=true ./par-sum`

`ncu -o ncu_par-sum -f --clock-control none  ./par-sum`

`ncu-ui ncu_par-sum.ncu-rep`



VERSION	|DESCRIPTION    	  	|BANDWIDTH (GB/s)    	|TIME (ms) 	|AGAINST_CUBLAS (%)
-------	| ------------------------- 	| --------------------- | ------------- | ----------------
1	|Naive (GPU serial)		|1.34			|49.78		|0.83
2	|Shared memory			|35.13			|1.91		|21.72
3	|Halve the blocks (1/2)		|67.92			|0.988		|42.00
4	|Even fewer blocks (1/8)	|170.33			|0.394		|105.3
5	|Unroll the last warp   	|169.46			|0.396		|104.7
6	|Use FLOAT4		   	|170.33			|0.394		|105.3

