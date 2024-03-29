#include "glwidget.h"
#include "camera.h"
#include "kernel.h"
#include "timer.h"

#include <QGLFunctions>
#include <iostream>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

static GLuint pixelsVBO;
static struct cudaGraphicsResource* pixelsVBO_CUDA;

GlWidget::GlWidget(QWidget *parent)
    : QGLWidget(QGLFormat(), parent), gl(context())
{
    lambda = 1.0;
    sigma = 0.4;
    tau = 0.4;
    N = 160;
    c1 = 1.0;
    c2 = 0.00;
}

GlWidget::~GlWidget()
{
	cudaFree(d_in);
}

QSize GlWidget::sizeHint() const
{
    return QSize(camera.width(), camera.height());
}

void GlWidget::initializeGL()
{
    // Explicitly set device 0
    cudaGLSetGLDevice(0);

    // Create buffer object and register it with CUDA
    gl.glGenBuffers(1, &pixelsVBO);
    gl.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelsVBO);
    size_t size = camera.width() * camera.height() * 4 * sizeof(unsigned char);
    gl.glBufferData(GL_PIXEL_UNPACK_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&pixelsVBO_CUDA, pixelsVBO, cudaGraphicsMapFlagsWriteDiscard);

    size_t inBytes = camera.width() * camera.height() * sizeof(float);
    cudaMalloc((void **)&d_in, inBytes);
}

void GlWidget::paintGL()
{
    // Map buffer object for writing from CUDA
    void *d_out;
    cudaGraphicsMapResources(1, &pixelsVBO_CUDA, 0);
    size_t size; 
    cudaGraphicsResourceGetMappedPointer(&d_out, &size,  pixelsVBO_CUDA);

    size_t inBytes = camera.width() * camera.height() * sizeof(float);
    {
        QMutexLocker locker(&camera.mutex);
        cudaMemcpy(d_in, camera.data(), inBytes, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        camera.frameCopied.wakeAll();
    }

    // Execute kernel
    executeKernel(d_in, d_out, camera.width(), camera.height(), lambda, sigma, tau, N, c1, c2);

    // Unmap buffer object
    cudaGraphicsUnmapResources(1, &pixelsVBO_CUDA, 0);

    // Render from buffer object
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawPixels(camera.width(), camera.height(), GL_RGBA, GL_UNSIGNED_BYTE, 0);
    timer.end();
    fps = 1.F / timer.get();
    std::cout << "FPS = "<<  fps << std::endl;
    emit fps_updated(fps);
    timer.start();
}
