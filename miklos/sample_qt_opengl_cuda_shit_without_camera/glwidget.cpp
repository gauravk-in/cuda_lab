#include "glwidget.h"
#include "kernel.h"

#include <QGLFunctions>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

GLuint positionsVBO;
struct cudaGraphicsResource* positionsVBO_CUDA;

GlWidget::GlWidget(QWidget *parent)
    : QGLWidget(QGLFormat(), parent), func(context()), start(time(NULL))
{
}

GlWidget::~GlWidget()
{
}

QSize GlWidget::sizeHint() const
{
    return QSize(640, 480);
}

void GlWidget::initializeGL()
{
    makeCurrent();

    // Explicitly set device 0
    cudaGLSetGLDevice(0);

    // Create buffer object and register it with CUDA
    func.glGenBuffers(1, &positionsVBO);
    func.glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
    size_t size = (size_t)640 * 480 * 4 * sizeof(float); // TODO
    func.glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    func.glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA, positionsVBO, cudaGraphicsMapFlagsWriteDiscard);
}

void GlWidget::paintGL()
{
    // Map buffer object for writing from CUDA
    float4* positions;
    cudaGraphicsMapResources(1, &positionsVBO_CUDA, 0);
    size_t num_bytes; 
    cudaGraphicsResourceGetMappedPointer((void**)&positions, &num_bytes,  positionsVBO_CUDA);

    // Execute kernel
    executeKernel(positions, 640, 480, float(time(0) - start) * 0.1);

    // Unmap buffer object
    cudaGraphicsUnmapResources(1, &positionsVBO_CUDA, 0);

    // Render from buffer object
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    func.glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_POINTS, 0, 640 * 480); // TODO
    glDisableClientState(GL_VERTEX_ARRAY);
}
