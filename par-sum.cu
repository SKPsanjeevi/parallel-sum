#include<iostream>
#include<chrono>
#include"helper.h"

#define size 4096*4096

// GPU serial addition
__global__
void kernelV1(float* A, float* result){
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(result, A[tx]);
}


// GPU parallel reduction and using shared memory
__global__
void kernelV2(float* A, float* result){
    extern __shared__ float sA[];
    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int bx = blockDim.x;
    sA[tx] = A[tId];
    __syncthreads();

    // now perform a parallel reduction until active threads (bx) reduce to 1
    while(bx > 1){
        bx /= 2;
        if(tx < bx) {
            sA[tx] += sA[tx + bx];
        }
        __syncthreads();
    }

    // add the partial result saved in every shared memory 0th position to the final result
    if(tx == 0){
        atomicAdd(result, sA[0]);
    }
}


// Reduce (Halve) the number of blocks and perform addition during first load
__global__
void kernelV3(float* A, float* result){
    extern __shared__ float sA[];
    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int bx = blockDim.x;
    
    // perform an addition with elements at an offset and load into shared memory
    sA[tx] = A[tId] + A[gridDim.x * blockDim.x + tId];
    __syncthreads();

    // now perform a parallel reduction until active threads (bx) reduce to 1
    while(bx > 1){
        bx /= 2;
        if(tx < bx) {
            sA[tx] += sA[tx + bx];
        }
        __syncthreads();
    }

    // add the partial result saved in every shared memory 0th position to the final result    
    if(tx == 0){
        atomicAdd(result, sA[0]);
    }
}


// Even fewer blocks; do multiple adds during first shared mem load
__global__
void kernelV4(float* A, float* result){
    extern __shared__ float sA[];
    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int bx = blockDim.x;

    // perform an addition with multiple elements at offsets and load into shared memory
    sA[tx] = A[tId] + A[gridDim.x * blockDim.x + tId] 
            + A[2 * gridDim.x * blockDim.x + tId] + A[3 * gridDim.x * blockDim.x + tId]
            + A[4 * gridDim.x * blockDim.x + tId] + A[5 * gridDim.x * blockDim.x + tId]
            + A[6 * gridDim.x * blockDim.x + tId] + A[7 * gridDim.x * blockDim.x + tId];
    __syncthreads();

    // parallel reduction
    while(bx > 8){
        bx >>= 3;        // log base 8 time complexity
        if(tx < bx) {
            sA[tx] += (sA[tx + bx] + sA[tx + 2*bx] + sA[tx + 3*bx] + sA[tx + 4*bx] +
                       sA[tx + 5*bx] + sA[tx + 6*bx] + sA[tx + 7*bx]);
        }
        __syncthreads();
    }

    // parallel reduction - final 8 elements would be left out from previous operation
    while(bx > 1){
        bx >>= 1;        // log base 2
        if(tx < bx) {
            sA[tx] += sA[tx + bx];
        }
        __syncthreads();
    }

    if(tx == 0){
        atomicAdd(result, sA[0]);
    }
}


// within warp level, activities are synchronous and do not need explicit synchronization
__device__
void warpReduce(volatile float* sA, int tx){
    sA[tx] += sA[tx + 16];
    sA[tx] += sA[tx + 8];
    sA[tx] += sA[tx + 4];
    sA[tx] += sA[tx + 2];
    sA[tx] += sA[tx + 1];
}

// remove explicit synchronization with in warp (final 32 elements)
__global__
void kernelV5(float* A, float* result){
    extern __shared__ float sA[];
    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int bx = blockDim.x;

    // perform an addition with multiple elements at offsets and load into shared memory
    sA[tx] = A[tId] + A[gridDim.x * blockDim.x + tId] 
            + A[2 * gridDim.x * blockDim.x + tId] + A[3 * gridDim.x * blockDim.x + tId]
            + A[4 * gridDim.x * blockDim.x + tId] + A[5 * gridDim.x * blockDim.x + tId]
            + A[6 * gridDim.x * blockDim.x + tId] + A[7 * gridDim.x * blockDim.x + tId];
    __syncthreads();

    // parallel reduction
    while(bx > 32){
        bx >>= 3;        // log base 8 time complexity
        if(tx < bx) {
            sA[tx] += (sA[tx + bx] + sA[tx + 2*bx] + sA[tx + 3*bx] + sA[tx + 4*bx] +
                       sA[tx + 5*bx] + sA[tx + 6*bx] + sA[tx + 7*bx]);
        }
        __syncthreads();
    }

    // parallel reduction - final 32 elements fitting in the warp
    warpReduce(sA, tx);

    if(tx == 0){
        atomicAdd(result, sA[0]);
    }
}


__device__
float4 float4Sum(float4& v0, float4& v1, float4& v2, float4& v3, float4& v4, float4& v5, float4& v6, float4& v7){
    return {v0.w+v1.w+v2.w+v3.w+v4.w+v5.w+v6.w+v7.w, 
            v0.x+v1.x+v2.x+v3.x+v4.x+v5.x+v6.x+v7.x,
            v0.y+v1.y+v2.y+v3.y+v4.y+v5.y+v6.y+v7.y,
            v0.z+v1.z+v2.z+v3.z+v4.z+v5.z+v6.z+v7.z};
}

// kernelV4 with FLOAT4
__global__
void kernelV6(float* A, float* result){
    extern __shared__ float sA[];
    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int bx = blockDim.x;

    // // perform an addition with multiple elements at offsets and load into shared memory
    sA[tx] = A[tId] + A[gridDim.x * blockDim.x + tId] 
            + A[2 * gridDim.x * blockDim.x + tId] + A[3 * gridDim.x * blockDim.x + tId]
            + A[4 * gridDim.x * blockDim.x + tId] + A[5 * gridDim.x * blockDim.x + tId]
            + A[6 * gridDim.x * blockDim.x + tId] + A[7 * gridDim.x * blockDim.x + tId];
    __syncthreads();

    // parallel reduction
    while(bx >= 32){
        bx >>= 3;
        if(tx % 4 == 0 && tx < bx) {
            int offset = bx/4, tx_4 = tx / 4;
            float4 v0 = reinterpret_cast<float4*>(sA)[tx_4];
            float4 v1 = reinterpret_cast<float4*>(sA)[tx_4 + offset];
            float4 v2 = reinterpret_cast<float4*>(sA)[tx_4 + 2*offset];
            float4 v3 = reinterpret_cast<float4*>(sA)[tx_4 + 3*offset];
            float4 v4 = reinterpret_cast<float4*>(sA)[tx_4 + 4*offset];
            float4 v5 = reinterpret_cast<float4*>(sA)[tx_4 + 5*offset];
            float4 v6 = reinterpret_cast<float4*>(sA)[tx_4 + 6*offset];
            float4 v7 = reinterpret_cast<float4*>(sA)[tx_4 + 7*offset];            
            float4* resultPtr = reinterpret_cast<float4*>(&sA[tx]);
            *resultPtr = float4Sum(v0, v1, v2, v3, v4, v5, v6, v7);
        }
        __syncthreads();
    }

    while(bx > 1){
        bx >>= 1;        // log base 2
        if(tx < bx) {
            sA[tx] += sA[tx + bx];
        }
        __syncthreads();
    }    

    if(tx == 0){
        atomicAdd(result, sA[0]);
    }
}


int main(){
    int N = size;

    // host memory allocation
    float *hA = (float *) malloc(N * sizeof(float));

    float hSumFromGPU, hSumFromCUBLAS;

    // device memory allocation
    float *A = (float *)fixed_cudaMalloc(N * sizeof(float));
    float *result = (float *)fixed_cudaMalloc(sizeof(float));
    cudaMemset(A, 0., N * sizeof(float));
    cudaMemset(&result, 0., sizeof(float));    
    
    // host memory initialization
    srand (static_cast <unsigned> (time(0)));  // seed for random initialization
    intializeMatrix(hA, N);

    // copy host to device memory
    gpuErrchk(cudaMemcpy(A, hA, N*sizeof(float), cudaMemcpyHostToDevice));

    // CPU compute of sum
    float hSumFromCPU = computeSum(hA, N);  // NOTE: hA gets modified due to parallel sum but unused henceforth

    // // kernel 1
    // kernelV1<<<size/256, 256>>>(A, result);

    // // kernel 2
    // int NUM_THREADS = 256;
    // kernelV2<<<CEIL_DIV(size, NUM_THREADS), NUM_THREADS, NUM_THREADS*sizeof(float)>>>(A, result);

    // //  Kernel 3
    // int NUM_THREADS = 2;
    // kernelV3<<<CEIL_DIV(size, (NUM_THREADS * 2)), NUM_THREADS, NUM_THREADS*sizeof(float)>>>(A, result);

    // //  Kernel 4 - kernel3 with fewer blocks
    // int NUM_THREADS = 256;
    // kernelV4<<<CEIL_DIV(size, (NUM_THREADS * 8)), NUM_THREADS, NUM_THREADS*sizeof(float)>>>(A, result);

    // //  Kernel 5
    // int NUM_THREADS = 256;
    // kernelV5<<<CEIL_DIV(size, (NUM_THREADS * 8)), NUM_THREADS, NUM_THREADS*sizeof(float)>>>(A, result);

    //  Kernel 6
    int NUM_THREADS = 128;
    kernelV6<<<CEIL_DIV(size, (NUM_THREADS*8)), NUM_THREADS, NUM_THREADS*sizeof(float)>>>(A, result);

    // copy device to host memory
    gpuErrchk(cudaMemcpy(&hSumFromGPU, result, sizeof(float), cudaMemcpyDeviceToHost));    

    // cuBLAS sum function
    cuBLASSUM(A, N, &hSumFromCUBLAS);


    float epsilon = 1e-6f;
    cout << "Result from CPU : " << hSumFromCPU << endl;
    cout << "Result from GPU : " << hSumFromGPU << endl;
    cout << "Result from cuBLAS : " << hSumFromCUBLAS << endl;
    cout << "Absolute difference : " << fabs(hSumFromGPU - hSumFromCUBLAS) << endl;
    cout << "Approximately equal? : " << (boolalpha) << approximatelyEqual(hSumFromGPU, hSumFromCUBLAS, epsilon) 
                                    << "  (epsilon = " << epsilon << ")" << endl;    

    // Free the memory
    cudaFree(A);
    free(hA);

    return 0;
}
