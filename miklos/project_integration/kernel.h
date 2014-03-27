#ifndef KERNEL_H
#define KERNEL_H

extern "C" void allocate_device_memory(float *d_in, size_t width, size_t height);
extern "C" void executeKernel(float *d_in, void *d_out, size_t width, size_t height);

#endif // KERNEL_H
