#include <cstdlib>  // header for rand()
#include <ctime>    // header for time-seed for rand()
#include <limits>   // get the smallest increment for a datatype
#include <cublas_v2.h>

using namespace std;

#define CEIL_DIV(M, N) ((M) + (N)-1) / (N)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void intializeMatrix(float *A, int size){
    for(int i = 0; i < size; i++){
//        A[i] = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
          A[i] = 1.;
    }
}

void *fixed_cudaMalloc(size_t len)
{
    void *p;
    if (cudaMalloc(&p, len) == cudaSuccess) return p;
    return 0;
}

template<class T>
bool approximatelyEqual(T a, T b, T epsilon)
{
    return fabs(a - b) <= ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

template<class T>
void compareResultsMV(T *A, T *B, int M){
    int factor = 100;
    T epsilon = std::numeric_limits<T>::epsilon() * factor;
    std::cout<<"Epsilon : " << epsilon << std::endl;
    for(int i = 0; i < M; i++){
        if( !approximatelyEqual(A[i], B[i], epsilon) ){
            printf("Outside tolerance at indices at i : %d, P : %f, Q : %f\n",
                                            i, A[i], B[i]);
            exit(0);
        }
    }
}

int computeMV(float *hA, int N){

return 0;
}

void cuBLASSUM(float *A, int N, float* result){
    cublasHandle_t handle;
    cublasCreate(&handle);

    int incx = 1; // stride between consecutive elements

    cublasSasum(handle, N, A, incx, result);
    cout << "result from cuBLAS : "<< *result << endl;

    cublasDestroy(handle);
}