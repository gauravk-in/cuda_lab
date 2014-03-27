#include "kernel.h"

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
    float *in = reinterpret_cast<float *>(d_in);
    uchar4 *pixel = reinterpret_cast<uchar4 *>(d_out);
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(w / dimBlock.x, h / dimBlock.y, 1);
    createVertices<<<dimGrid, dimBlock>>>(in, pixel, w, h);
}
