#ifndef KERNEL_H
#define KERNEL_H

#include <stdlib.h>

extern "C" void allocate_device_memory(float *d_in, size_t width, size_t height);
extern "C" void executeKernel(float *d_U, void *d_out, size_t w, size_t h, float lambda, float sigma, float tau, int N, float c1, float c2);

#endif // KERNEL_H
