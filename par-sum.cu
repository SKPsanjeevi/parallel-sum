#include<iostream>
#include<chrono>
#include"helper.h"

#define size 4096*4096

// GPU serial version
__global__
void reductionNaive(float* A, float* result){
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(result, A[tx]);
}


// GPU parallel version using shared memory
__global__
void reductionShared(float* A, float* result){
    extern __shared__ float sA[];
    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int bx = blockDim.x;
    sA[tx] = A[tId];
    __syncthreads();

    while(bx > 1){
        bx /= 2;
        if(tx < bx) {
            sA[tx] += sA[tx + bx];
        }
        __syncthreads();
    }
    if(tx == 0){
        atomicAdd(result, sA[0]);
    }
}


// Halve the blocks and perform addition during first load
__global__
void kernelV3(float* A, float* result){
    extern __shared__ float sA[];
    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int bx = blockDim.x;
    sA[tx] = A[tId] + A[gridDim.x * blockDim.x + tId];
    __syncthreads();

    while(bx > 1){
        bx /= 2;
        if(tx < bx) {
            sA[tx] += sA[tx + bx];
        }
        __syncthreads();
    }
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
    sA[tx] = A[tId] + A[gridDim.x * blockDim.x + tId] 
            + A[2 * gridDim.x * blockDim.x + tId] + A[3 * gridDim.x * blockDim.x + tId]
            + A[4 * gridDim.x * blockDim.x + tId] + A[5 * gridDim.x * blockDim.x + tId]
            + A[6 * gridDim.x * blockDim.x + tId] + A[7 * gridDim.x * blockDim.x + tId];
    __syncthreads();

    while(bx > 1){
        bx /= 4;
        if(tx < bx) {
            sA[tx] += sA[tx + bx] + sA[tx + 2*bx] + sA[tx + 3*bx];
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

    float hSumfromGPU, hSumfromCUBLAS;

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
    float hResult = computeMV(hA, N);

    // // GPU kernel launch
    // reductionNaive<<<size/256, 256>>>(A, result);

    // // Shared memory optimization
    // int NUM_THREADS = 256;
    // reductionShared<<<CEIL_DIV(size, NUM_THREADS), NUM_THREADS, NUM_THREADS*sizeof(float)>>>(A, result);

    // //  Kernel 3
    // int NUM_THREADS = 256;
    // kernelV3<<<CEIL_DIV(size, (NUM_THREADS * 2)), NUM_THREADS, NUM_THREADS*sizeof(float)>>>(A, result);

    //  Kernel 4 - even fewer blocks
    int NUM_THREADS = 256;
    kernelV4<<<CEIL_DIV(size, (NUM_THREADS * 8)), NUM_THREADS, NUM_THREADS*sizeof(float)>>>(A, result);

    // copy device to host memory
    gpuErrchk(cudaMemcpy(&hSumfromGPU, result, sizeof(float), cudaMemcpyDeviceToHost));    

    // cuBLAS sum function
    cuBLASSUM(A, N, &hSumfromCUBLAS);


    float epsilon = 1e-6f;
    cout << "Result from GPU : " << hSumfromGPU << endl;
    cout << "Result from cuBLAS : " << hSumfromCUBLAS << endl;
    cout << "Absolute difference : " << fabs(hSumfromGPU - hSumfromCUBLAS) << endl;
    cout << "Approximately equal? : " << (boolalpha) << approximatelyEqual(hSumfromGPU, hSumfromCUBLAS, epsilon) 
                                    << "  (epsilon = " << epsilon << ")" << endl;    

    // Free the memory
    cudaFree(A);
    free(hA);

    return 0;
}
