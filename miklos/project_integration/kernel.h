#ifndef KERNEL_H
#define KERNEL_H

extern "C" void executeKernel(void *d_in, void *d_out, size_t width, size_t height);

#endif // KERNEL_H
