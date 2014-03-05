// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Winter Semester 2013/2014, March 3 - April 4
// ###
// ###
// ### Evgeny Strekalovskiy, Maria Klodt, Jan Stuehmer, Mohamed Souiai
// ###
// ###
// ###


#include <cuda_runtime.h>
#include <iostream>
using namespace std;



// cuda error checking
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(string file, int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        cout << endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << endl;
        exit(1);
    }
}

__device__ float add(float a, float b)
{
    return a + b;
}

__global__ void add_arrays(float *a, float *b, float *c, int n)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n)
        c[idx] = add(a[idx], b[idx]);
}

int main(int argc, char **argv)
{
    // alloc and init input arrays on host (CPU)
    int n = 20;
    float *a = new float[n];
    float *b = new float[n];
    float *c = new float[n];
    for(int i=0; i<n; i++)
    {
        a[i] = i;
        b[i] = (i%5)+1;
        c[i] = 0;
    }

    // CPU computation
    for(int i=0; i<n; i++) c[i] = a[i] + b[i];

    // print result
    cout << "CPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << " + " << b[i] << " = " << c[i] << endl;
    cout << endl;
    // init c
    for(int i=0; i<n; i++) c[i] = 0;
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n*sizeof(float));
    cudaMalloc(&d_b, n*sizeof(float));
    cudaMalloc(&d_c, n*sizeof(float));
    cudaMemcpy(d_a, a, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid((n + block.x - 1) / block.x);
    add_arrays<<<grid, block>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c, d_c, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // ### Notes:
    // ### 2. Always use the macro CUDA_CHECK after each CUDA call, e.g. "cudaMalloc(...); CUDA_CHECK;"

    // print result
    cout << "GPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << " + " << b[i] << " = " << c[i] << endl;
    cout << endl;

    // free CPU arrays
    delete[] a;
    delete[] b;
    delete[] c;
}



