#include "kernel.h"
#include <algorithm>

template<typename T>
__device__ __host__ T min(T a, T b)
{
    return (a < b) ? a : b;
}

template<typename T>
__device__ __host__ T max(T a, T b)
{
    return (a > b) ? a : b;
}

template<typename T>
__device__ __host__ T clamp(T m, T x, T M)
{
    return max(m, min(x, M));
}


__global__ void calculate_F(float *U, float *F, int w, int h, float c1, float c2, float lambda)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x < w && y < h) {
        size_t i = x + (size_t)w*y;
        F[i] = lambda * ((c1 - U[i])*(c1 - U[i]) - (c2 - U[i])*(c2 - U[i]));
    }
}

__device__ float diff_i(float *M, int w, int h, int x, int y)
{
    size_t i = x + (size_t)w*y;
    return (x+1 < w) ? (M[i + 1] - M[i]) : 0.f;
}

__device__ float diff_j(float *M, int w, int h, int x, int y)
{
    size_t i = x + (size_t)w*y;
    return (y+1 < h) ? (M[i + w] - M[i]) : 0.f;
}

__global__ void update_Xij(float *Xi, float *Xj, float *T, float *U, int w, int h, float sigma)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x < w && y < h) {
        size_t i = x + (size_t)w*y;
        float xi = Xi[i] - sigma * (2 * diff_i(U, w, h, x, y) - diff_i(T, w, h, x, y));
        float xj = Xj[i] - sigma * (2 * diff_j(U, w, h, x, y) - diff_j(T, w, h, x, y));
        float dn = max(1.f, sqrtf(xi*xi + xj*xj));
        Xi[i] = xi / dn;
        Xj[i] = xj / dn;
    }
}

__device__ float divergence(float *X, float *Y, int w, int h, int x, int y)
{
    size_t i = x + (size_t)w*y;
    float dx_x = ((x+1 < w) ? X[i] : 0.f) - ((x > 0) ? X[i - 1] : 0.f);
    float dy_y = ((y+1 < h) ? Y[i] : 0.f) - ((y > 0) ? Y[i - w] : 0.f);
    return dx_x + dy_y;
}

__global__ void update_U(float *T, float *Xi, float *Xj, float *F, float *U, int w, int h, float tau)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x < w && y < h) {
        size_t i = x + (size_t)w*y;
        U[i] = clamp(0.f, T[i] - tau * (divergence(Xi, Xj, w, h, x, y) + F[i]), 1.f);
    }
}

__global__ void update_Output(uchar4* output, float *U, int w, int h) {

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    size_t i = x + (size_t)w*y;
    size_t idx = x + (size_t) w*(h-1 - y);
    unsigned char temp_res = roundf((U[i] * 255.f));
    output[idx].x = temp_res;
    output[idx].y = temp_res;
    output[idx].z = temp_res;
    output[idx].w = 255;

}

inline int div_ceil(int n, int b) { return (n + b - 1) / b; }


__global__ void createVertices(float *in, uchar4* pixel, int w, int h)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned char intensity = roundf(255 * in[y * w + x]);

    // Write positions
    size_t i = x + w*(h-1 - y);
    pixel[i].x = intensity;
    pixel[i].y = intensity;
    pixel[i].z = intensity;
    pixel[i].w = 255;
}

void executeKernel(void *d_in, void *d_out, size_t w, size_t h)
{
    float *d_U = reinterpret_cast<float *>(d_in);
    uchar4 *pixel = reinterpret_cast<uchar4 *>(d_out);

    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(w / dimBlock.x, h / dimBlock.y, 1);

    // set parameters manually here
    float lambda = 1.0;
    float sigma = 0.4;
    float tau = 0.4;
    int N = 160;
    float c1 = 1.0;
    float c2 = 0.00;

    float *d_T, *d_F, *d_Xi, *d_Xj;
    size_t imageBytes = w*h*sizeof(float);
    cudaMalloc(&d_T, imageBytes);
    cudaMalloc(&d_F, imageBytes);
    cudaMalloc(&d_Xi, imageBytes);
    cudaMalloc(&d_Xj, imageBytes);
    cudaMemcpy(d_T, d_U, imageBytes, cudaMemcpyDeviceToDevice);
    cudaMemset(d_Xi, 0, imageBytes);
    cudaMemset(d_Xj, 0, imageBytes);

    calculate_F<<< dimGrid, dimBlock >>>(d_U, d_F, w, h, c1, c2, lambda);

    for (int n = 0; n < N; n++) {
        update_Xij<<< dimGrid, dimBlock >>>(d_Xi, d_Xj, d_T, d_U, w, h, sigma);
        std::swap(d_U, d_T);
        update_U<<< dimGrid, dimBlock >>>(d_T, d_Xi, d_Xj, d_F, d_U, w, h, tau);
    }
    update_Output<<< dimGrid, dimBlock >>>(pixel, d_U, w, h);
    cudaFree(d_T);
    cudaFree(d_F);
    cudaFree(d_Xi);
    cudaFree(d_Xj);

//    createVertices<<<dimGrid, dimBlock>>>(in, pixel, w, h);
}
