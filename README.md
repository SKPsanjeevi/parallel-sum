*** PARALLEL SUM ***

Performance data on RTX3050Mobile for a vector of size N. In our case, N = 16.78 million elements (4096^2).

The max ceiling performance of a RTX3050 Mobile is 5.501 TFLOPS (for FP32) and global memory bandwidth of 192 GBytes/s.
Source : https://www.techpowerup.com/gpu-specs/geforce-rtx-3050-mobile.c3788


The time-complexity of this reduction operation  is O(N) and there are N floating point operations to be done for this case. Given theoretical peak performance (or the ROOF in roofline model), we can compute relative performance if TIME taken by the kernel is known using

$$ \text{AGAINSTROOF [PERCENT]} = \frac{N \text{[FLOP]} } {\text{TIME [s] *  GPUPEAKPERFORMANCE [FLOPS]}} * 100 [\text{PERCENT}] $$


CUBLAS functions typically have various custom kernels that get called even depending on the GPU specifications. It has to be noted that the execution times of the final CUBLAS kernel is actually used rather than the time taken by the CUBLAS function itself. The CUBLAS functions take much longer time to call the specific kernel needed. This itself motivates users to write custom kernels.


In a real world scenario, the GPU adaptively chooses varying frequency which is typically higher than the base frequency. We DO NOT pin the clock frequency to base frequency for all kernel measurements as it will measure the actual execution speeds. By default, Nsight Compute (`ncu`) pins the kernels to base frequncy whereas Nsight System `nsys` seems to be working on unpinned frequencies. The unpinned clock frequency can be achieved in `ncu` using `--clock-control none` option. 

PINNED CLOCK FREQUENCY : CUBLAS kernel time taken = 0.541 ms (uses `asum_kernel` twice as seen from `nsys` data.). In this case, CUBLAS uses about roughly **0.564 %** of the roofline performance for this GPU.

UNPINNED CLOCK FREQUENCY : CUBLAS kernel time taken = 0.415 ms (uses `asum_kernel` twice as seen from `nsys` data.). In this case, CUBLAS uses about roughly **0.735 %** of the roofline performance for this GPU.


Regarding bandwidth, N amount of floats are transferred to the SMs but resulting in only one float output. Therefore, the bandwidth can be approximated as N * 4 [BYTE] / (TIME [s]).

Commands for building and profiling:

`nvcc par-sum.cu -o par-sum -arch=sm_80 -lcublas`

`nsys profile -o nsys_par-sum --stats=true ./par-sum`

`ncu -o ncu_par-sum -f --clock-control none  ./par-sum`

`ncu-ui ncu_par-sum.ncu-rep`



VERSION	|DESCRIPTION    	  	|BANDWIDTH (GB/s)    	|TIME (ms) 	|AGAINST_CUBLAS* (%)
-------	| ------------------------- 	| --------------------- | ------------- | ----------------
1	|Naive (GPU serial)		|1.34			|49.78		|0.83
2	|Shared memory			|35.13			|1.91		|21.72
3	|Halve the blocks (1/2)		|67.92			|0.988		|42.00
4	|Even fewer blocks (1/8)	|170.33			|0.394		|105.3
5	|Unroll the last warp   	|169.46			|0.396		|104.7
6	|Vectorized loads (FLOAT4)   	|170.33			|0.394		|105.3


* - 100% implies CUBLAS performance


[Roofline model analysis](https://en.wikipedia.org/wiki/Roofline_model)
As it can be seen, the parallel sum algorithm is heavily memory bound. The highest memory bandwidth achieved (170.33 GB/s) is already about 89% of the global memory bandwidth of 192 GB/s with a plenty of compute FLOPS left on the table. This is expected as every data is 4 byte long for a single FLOP and therefore, the [arithmetic intensity](https://en.wikipedia.org/wiki/Roofline_model#Arithmetic_intensity) is 0.25.

Compute work load [FLOPS] = Arithmetic Intensity [FLOPs/byte] * Bandwdith [byte/s] = 0.25 * 170.33e9 = 42.58e9 FLOPs

Against GPUPEAKPERFORMANCE [%] = Compute work load [FLOPS] / GPUPEAKPERFORMANCE [FLOPS] * 100 [%] = 42.58e9/5.501e12 * 100 = 0.77% (of peak GPU capacity)

This, the memory boundness of the problem, explains the reason why the CUBLAS performance (0.735%) and our performance (0.77%) are severely underutilized compared the available peak compute in the GPU. With a compute heavy problem like [general matrix multiply - GEMM](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3), there are several strategies available to increase the arithmetic intensity (FLOPs/byte). I have explored them [here](https://github.com/SKPsanjeevi/sgemm).
