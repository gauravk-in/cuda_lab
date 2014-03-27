#include "kernel.h"

__global__ void createVertices(float *in, uchar4* positions, int width, int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned char intensity = roundf(255 * in[y * width + x]);

    // Write positions
    positions[y * width + x].x = intensity;
    positions[y * width + x].y = intensity;
    positions[y * width + x].z = intensity;
    positions[y * width + x].w = 255;
}

void executeKernel(float *d_in, void *positions_, int width, int height, float time)
{
    uchar4 *positions = (uchar4 *)positions_;
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
    createVertices<<<dimGrid, dimBlock>>>(d_in, positions, width, height);
}
