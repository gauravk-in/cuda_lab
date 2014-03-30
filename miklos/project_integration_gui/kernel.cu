#include "kernel.h"
// #include "timer.h"

#include <algorithm>
#include <stdio.h>

texture<float,2,cudaReadModeElementType> texRef_Xi;
texture<float,2,cudaReadModeElementType> texRef_Xj;

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
        float temp_ui = U[i];
        F[i] = lambda * ((c1 - temp_ui)*(c1 - temp_ui) - (c2 - temp_ui)*(c2 - temp_ui));
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
		size_t i = x + (size_t) w * y;
		float xi = Xi[i] - sigma * (2 * diff_i(U, w, h, x, y) - diff_i(T, w, h, x, y));
		float xj = Xj[i] - sigma * (2 * diff_j(U, w, h, x, y) - diff_j(T, w, h, x, y));
		float dn = max(1.f, sqrtf(xi * xi + xj * xj));
		Xi[i] = xi / dn;
		Xj[i] = xj / dn;
    }
}

__device__ float divergence(float *X, float *Y, int w, int h, int x, int y)
{
    float dx_x = tex2D(texRef_Xi, x + 0.5f , y + 0.5f) - tex2D(texRef_Xi, x - 0.5f , y + 0.5f);
    float dy_y = tex2D(texRef_Xj, x + 0.5f , y + 0.5f) - tex2D(texRef_Xj, x + 0.5f , y - 0.5f);

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
    if( x < w && y < h ) {
        size_t i = x + (size_t)w*y;
        size_t idx = x + (size_t) w*(h-1 - y);
        unsigned char temp_res = int(U[i] + 0.5f) * 255;
        output[idx].x = temp_res;
        output[idx].y = temp_res;
        output[idx].z = temp_res;
        output[idx].w = 255;
    }
}

inline int div_ceil(int n, int b) { return (n + b - 1) / b; }

inline dim3 make_grid(dim3 whole, dim3 block)
{
    return dim3(div_ceil(whole.x, block.x),
                div_ceil(whole.y, block.y),
                div_ceil(whole.z, block.z));
}

static float *d_T, *d_F, *d_Xi, *d_Xj;


void allocate_device_memory(float *d_in, size_t w, size_t h)
{
	size_t imageBytes = w*h*sizeof(float);
	cudaMalloc(&d_T, imageBytes);
	cudaMalloc(&d_F, imageBytes);
	cudaMalloc(&d_Xi, imageBytes);
	cudaMalloc(&d_Xj, imageBytes);

	// Define texture attributes
    texRef_Xi.addressMode[0] = cudaAddressModeBorder;             // clamp x to border
    texRef_Xi.addressMode[1] = cudaAddressModeBorder;            // clamp y to border
    texRef_Xi.filterMode = cudaFilterModeLinear;            // linear interpolation
    texRef_Xi.normalized = false;
    cudaChannelFormatDesc desc_Xi = cudaCreateChannelDesc<float>();
    cudaBindTexture2D(NULL, &texRef_Xi, d_Xi, &desc_Xi, w, h, w*sizeof(d_Xi[0]));

	// Define texture attributes
    texRef_Xj.addressMode[0] = cudaAddressModeBorder;             // clamp x to border
    texRef_Xj.addressMode[1] = cudaAddressModeBorder;            // clamp y to border
    texRef_Xj.filterMode = cudaFilterModeLinear;            // linear interpolation
    texRef_Xj.normalized = false;
    cudaChannelFormatDesc desc_Xj = cudaCreateChannelDesc<float>();
    cudaBindTexture2D(NULL, &texRef_Xj, d_Xj, &desc_Xj, w, h, w*sizeof(d_Xj[0]));

}

void executeKernel(float *d_U, void *d_out, size_t w, size_t h, float lambda, float sigma, float tau, int N, float c1, float c2)
{
    // float *d_U = reinterpret_cast<float *>(d_in);
    uchar4 *pixel = reinterpret_cast<uchar4 *>(d_out);

//    static Timer timer;
//    timer.end();
//    printf("time: %.2fms (%.2f FPS)\n", timer.get() * 1E3F, 1.F / timer.get());
//    timer.start();

    dim3 dimBlock(32, 16);
    dim3 dimGrid = make_grid(dim3(w, h, 1), dimBlock);

    size_t imageBytes = w*h*sizeof(float);
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
}
