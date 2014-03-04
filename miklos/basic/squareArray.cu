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


__device__ float square(float x)
{
    return x * x;
}

__global__ void square_array(float *arr, int n)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n)
        arr[idx] = square(arr[idx]);
}

int main(int argc,char **argv)
{
    // alloc and init input arrays on host (CPU)
    int n = 10;
    float *a = new float[n];
    for(int i=0; i<n; i++) a[i] = i;

    // CPU computation
    for(int i=0; i<n; i++)
    {
        float val = a[i];
        val = val*val;
        a[i] = val;
    }

    // print result
    cout << "CPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << endl;
    cout << endl;
    


    // GPU computation
    // reinit data
    for(int i=0; i<n; i++) a[i] = i;

    float *d_a;
    cudaMalloc(&d_a, n*sizeof(float));
    cudaMemcpy(d_a, a, n*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid((n + block.x - 1) / block.x);
    square_array<<<grid, block>>>(d_a, n);

    cudaMemcpy(a, d_a, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a);

    // print result
    cout << "GPU:" << endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << endl;
    cout << endl;

    // free CPU arrays
    delete[] a;
}



