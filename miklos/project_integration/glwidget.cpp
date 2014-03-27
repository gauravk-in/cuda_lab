#include "glwidget.h"
#include "kernel.h"

#include <QGLFunctions>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <cstdio>

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
    func.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, positionsVBO);
    size_t size = (size_t)640 * 480 * 4 * sizeof(unsigned char); // TODO
    func.glBufferData(GL_PIXEL_UNPACK_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    //func.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);  // TODO: what is it?
    cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA, positionsVBO, cudaGraphicsMapFlagsWriteDiscard);
}

void GlWidget::paintGL()
{
    // Map buffer object for writing from CUDA
    unsigned char *positions;
    cudaGraphicsMapResources(1, &positionsVBO_CUDA, 0);
    size_t num_bytes; 
    cudaGraphicsResourceGetMappedPointer((void**)&positions, &num_bytes,  positionsVBO_CUDA);

    float *d_in;
    cudaMalloc((void **)&d_in, (size_t)640*480*sizeof(float));

    extern cv::VideoCapture camera;
    cv::Mat mIn;
    camera >> mIn;
    printf("width: %d; height: %d;\n", mIn.cols, mIn.rows);
    cvtColor(mIn, mIn, CV_BGR2GRAY);

    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn, CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;

    cudaMemcpy(d_in, mIn.data, (size_t)640*480*sizeof(float), cudaMemcpyHostToDevice);

    // Execute kernel
    executeKernel(d_in, positions, 640, 480, float(time(0) - start) * 0.1);

    cudaFree(d_in);

    // Unmap buffer object
    cudaGraphicsUnmapResources(1, &positionsVBO_CUDA, 0);

    // Render from buffer object
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //func.glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);  // what is it?
    glDrawPixels( 640, 480, GL_RGBA, GL_UNSIGNED_BYTE, 0 );  // TODO
}
