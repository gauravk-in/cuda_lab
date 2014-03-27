#ifndef _KERNEL_H
#define _KERNEL_H

extern "C" void executeKernel(float *d_in, void *positions_, int width, int height, float time);

#endif // _KERNEL_H
