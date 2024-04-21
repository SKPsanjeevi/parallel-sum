#include<iostream>
#include<chrono>
#include"helper.h"

#define size 4096*4096


__global__
void mvNaive(float *A, int N, float* result){

}


int main(){
    int N = size;

    // host memory allocation
    float *hA = (float *) malloc(N * sizeof(float));

    float hSumfromGPU, hSumfromCUBLAS;

    // device memory allocation
    float *A = (float *)fixed_cudaMalloc(N * sizeof(float));
    
    // host memory initialization
    srand (static_cast <unsigned> (time(0)));  // seed for random initialization
    intializeMatrix(hA, N);

    // CPU compute of sum
    float hResult = computeMV(hA, N);
   
    // copy vector from host to device
    gpuErrchk(cudaMemcpy(A, hA, N * sizeof(float), cudaMemcpyHostToDevice));

    // cuBLAS sum function
    cuBLASSUM(A, N, &hSumfromCUBLAS);

    // Free the memory
    cudaFree(A);
    free(hA);

    return 0;
}
