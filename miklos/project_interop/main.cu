// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Winter Semester 2013/2014, March 3 - April 4
// ###
// ###
// ### Evgeny Strekalovskiy, Maria Klodt, Jan Stuehmer, Mohamed Souiai
// ###
// ###
// ###



// ###
// ###
// ### Miklos Homolya, miklos.homolya@tum.de, p056
// ### Ravikishore Kommajosyula, r.kommajosyula@tum.de, p057
// ### Gaurav Kukreja, gaurav.kukreja@tum.de, p058
// ###
// ###

#define GL_GLEXT_PROTOTYPES
#include <GL/glut.h>
#include "cuda_gl_interop.h"

#include "aux.h"
#include <iostream>
using namespace std;

/************************************************************************
 ***        GLOBAL VARIABLES                                        *****
 ************************************************************************/
int repeats;
bool gray;
float lambda;
float tau;
int N;
float c1;
float c2;
float sigma;

cv::VideoCapture camera(0);
cv::Mat mIn;

int w;
int h;
int nc;

// uncomment to use the camera
#define CAMERA

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
        F[i] = lambda * ((c1 - U[i])*(c1 - U[i]) - (c2 - U[i])*(c2 - U[i]));
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
        size_t i = x + (size_t)w*y;
        float xi = Xi[i] - sigma * (2 * diff_i(U, w, h, x, y) - diff_i(T, w, h, x, y));
        float xj = Xj[i] - sigma * (2 * diff_j(U, w, h, x, y) - diff_j(T, w, h, x, y));
        float dn = max(1.f, sqrtf(xi*xi + xj*xj));
        Xi[i] = xi / dn;
        Xj[i] = xj / dn;
    }
}

__device__ float divergence(float *X, float *Y, int w, int h, int x, int y)
{
    size_t i = x + (size_t)w*y;
    float dx_x = ((x+1 < w) ? X[i] : 0.f) - ((x > 0) ? X[i - 1] : 0.f);
    float dy_y = ((y+1 < h) ? Y[i] : 0.f) - ((y > 0) ? Y[i - w] : 0.f);
    return dx_x + dy_y;
}

__global__ void update_U(uchar4* output, float *T, float *Xi, float *Xj, float *F, float *U, int w, int h, float tau)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x < w && y < h) {
        size_t i = x + (size_t)w*y;
        U[i] = clamp(0.f, T[i] - tau * (divergence(Xi, Xj, w, h, x, y) + F[i]), 1.f);
        uchar temp_res = (uchar)(U[i] * 255.f);
        output[w*h-i-1].x = temp_res;
        output[w*h-i-1].y = temp_res;
        output[w*h-i-1].z = temp_res;
        output[w*h-i-1].w = 255;

    }
}

inline int div_ceil(int n, int b) { return (n + b - 1) / b; }

inline dim3 make_grid(dim3 whole, dim3 block)
{
    return dim3(div_ceil(whole.x, block.x),
                div_ceil(whole.y, block.y),
                div_ceil(whole.z, block.z));
}

GLuint bufferObj;
cudaGraphicsResource * resource;

#define HEIGHT 480
#define WIDTH 640

static void key_func( unsigned char key, int x, int y ) {
    switch (key) {
        case 27:
        // clean up OpenGL and CUDA

        cudaGraphicsUnregisterResource( resource );
        glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
        glDeleteBuffers( 1, &bufferObj );
        exit(0);
    }
}

static void draw_func( void ) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Get camera image
    camera >> mIn;
    if(gray)
        cvtColor(mIn, mIn, CV_BGR2GRAY);
    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;

    uchar4* d_output;
    size_t size;

    // allocate raw input image array
    float *imgIn  = new float[(size_t)w*h*nc];
    size_t imageBytes = (size_t)w*h*nc*sizeof(float);

    cudaGraphicsMapResources (1, &resource, NULL);
    cudaGraphicsResourceGetMappedPointer( (void**) &d_output, &size, resource);

    // Init raw input image array
    // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
    // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
    // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
    convert_mat_to_layered (imgIn, mIn);

    dim3 block(32, 16);
    dim3 grid = make_grid(dim3(w, h, 1), block);

    Timer timer; timer.start();
    float *d_T, *d_U, *d_F, *d_Xi, *d_Xj;
    cudaMalloc(&d_T, imageBytes);
    cudaMalloc(&d_U, imageBytes);
    cudaMalloc(&d_F, imageBytes);
    cudaMalloc(&d_Xi, imageBytes);
    cudaMalloc(&d_Xj, imageBytes);
    cudaMemcpy(d_T, imgIn, imageBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, d_T, imageBytes, cudaMemcpyDeviceToDevice);
    cudaMemset(d_Xi, 0, imageBytes);
    cudaMemset(d_Xj, 0, imageBytes);

    calculate_F<<< grid, block >>>(d_U, d_F, w, h, c1, c2, lambda);

    for (int n = 0; n < N; n++) {
        update_Xij<<< grid, block >>>(d_Xi, d_Xj, d_T, d_U, w, h, sigma);
        std::swap(d_U, d_T);
        update_U<<< grid, block >>>(d_output, d_T, d_Xi, d_Xj, d_F, d_U, w, h, tau);
    }

   // cudaMemcpy(imgOut, d_U, imageBytes, cudaMemcpyDeviceToHost);
    cudaGraphicsUnmapResources(1, &resource, NULL);
    cudaFree(d_T);
    cudaFree(d_U);
    cudaFree(d_F);
    cudaFree(d_Xi);
    cudaFree(d_Xj);

    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "time: " << t*1000 << " ms" << endl;

    // show input image
    // showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    glDrawPixels( WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
    glutSwapBuffers();
    glutPostRedisplay();
}

int main(int argc, char **argv)
{
#ifdef CAMERA
    cudaGLSetGLDevice(0);   CUDA_CHECK;

    // these GLUT calls need to be made before the other GL calls
     glutInit( &argc, argv );
     glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
     glutInitWindowSize( WIDTH, HEIGHT );
     glutCreateWindow( "bitmap" );

     glGenBuffers(1, &bufferObj);
     glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
     glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, WIDTH * HEIGHT * 4, NULL, GL_DYNAMIC_DRAW_ARB);
     cudaGraphicsGLRegisterBuffer( &resource, bufferObj, cudaGraphicsMapFlagsNone);
#endif

    // Before the GPU can process your kernels, a so called "CUDA context" must be initialized
    // This happens on the very first call to a CUDA function, and takes some time (around half a second)
    // We will do it right here, so that the run time measurements are accurate
    cudaDeviceSynchronize();  CUDA_CHECK;

    // Reading command line parameters:
    // getParam("param", var, argc, argv) looks whether "-param xyz" is specified, and if so stores the value "xyz" in "var"
    // If "-param" is not specified, the value of "var" remains unchanged
    //
    // return value: getParam("param", ...) returns true if "-param" is specified, and false otherwise

#ifdef CAMERA
#else
    // input image
    string image = "";
    bool ret = getParam("i", image, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> [-repeats <repeats>] [-gray]" << endl; return 1; }
#endif

    // number of computation repetitions to get a better run time measurement
    repeats = 1;
    getParam("repeats", repeats, argc, argv);
    cout << "repeats: " << repeats << endl;

    // load the input image as grayscale if "-gray" is specifed
    gray = true;
    // always true: getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;

    // ### Define your own parameters here as needed
    lambda = 1.0;
    getParam("lambda", lambda, argc, argv);
    cout << "λ: " << lambda << endl;

    sigma = 0.4;
    getParam("sigma", sigma, argc, argv);
    cout << "σ: " << sigma << endl;

    tau = 0.4;
    getParam("tau", tau, argc, argv);
    cout << "τ: " << tau << endl;

    N = 160;
    getParam("N", N, argc, argv);
    cout << "N: " << N << endl;

    c1 = 1.0;
    getParam("c1", c1, argc, argv);
    cout << "c1: " << c1 << endl;

    c2 = 0.00;
    getParam("c2", c2, argc, argv);
    cout << "c2: " << c2 << endl;

    // Init camera / Load input image
#ifdef CAMERA

    // Init camera
  	if(!camera.isOpened()) { cerr << "ERROR: Could not open camera" << endl; return 1; }
    int camW = 640;
    int camH = 480;
  	camera.set(CV_CAP_PROP_FRAME_WIDTH,camW);
  	camera.set(CV_CAP_PROP_FRAME_HEIGHT,camH);
    // read in first frame to get the dimensions

    camera >> mIn;
    if(gray)
        cvtColor(mIn, mIn, CV_BGR2GRAY);

#else

    // Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
    mIn = cv::imread(image.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
    // check
    if (mIn.data == NULL) { cerr << "ERROR: Could not load image " << image << endl; return 1; }

#endif

    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
    // get image dimensions
    w = mIn.cols;         // width
    h = mIn.rows;         // height
    nc = mIn.channels();  // number of channels
    cout << "image: " << w << " x " << h << endl;

    // Set the output image format
    cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
    // ### Define your own output images here as needed



    // For camera mode: Make a loop to read in camera frames
#ifdef CAMERA
    glutKeyboardFunc (key_func);
    glutDisplayFunc (draw_func);
    glutMainLoop();
#else

    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    float *imgIn  = new float[(size_t)w*h*nc];
    size_t imageBytes = (size_t)w*h*nc*sizeof(float);

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[(size_t)w*h*mOut.channels()];

    // Init raw input image array
    // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
    // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
    // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
    convert_mat_to_layered (imgIn, mIn);

    dim3 block(32, 16);
    dim3 grid = make_grid(dim3(w, h, 1), block);

    Timer timer; timer.start();
    float *d_U, *d_S, *d_vx, *d_vy;
    cudaMalloc(&d_U, imageBytes);
    cudaMalloc(&d_S, imageBytes);
    cudaMalloc(&d_vx, imageBytes);
    cudaMalloc(&d_vy, imageBytes);
    cudaMemcpy(d_U, imgIn, imageBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S, imgIn, imageBytes, cudaMemcpyHostToDevice);

    calculate_S<<< grid, block >>>(d_U, d_S, w, h, c1, c2);
    float *S = new float[(size_t)w*h];
    cudaMemcpy(S, d_S, imageBytes, cudaMemcpyDeviceToHost);
    float S_max = 0.0;
    for (size_t i = 0; i < (size_t)w*h; i++)
        S_max = max(S_max, fabs(S[i]));  // TODO: CPU thing
    delete[] S;
    float alpha = 0.5 * lambda * S_max;

    for (int n = 0; n < N; n++) {
        norm_grad<<< grid, block >>>(d_U, d_vx, d_vy, w, h);
        update<<< grid, block >>>(d_U, d_S, d_vx, d_vy, w, h, lambda, alpha, tau);
    }

    cudaMemcpy(imgOut, d_U, imageBytes, cudaMemcpyDeviceToHost);
    cudaFree(d_U);
    cudaFree(d_S);
    cudaFree(d_vx);
    cudaFree(d_vy);
    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "time: " << t*1000 << " ms" << endl;

    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, imgOut);
    showImage("Output", mOut, 100+w+40, 100);
    // ### Display your own output images here as needed

    // wait for key inputs
    cv::waitKey(0);

    // save input and result
    cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
    cv::imwrite("image_result.png",mOut*255.f);

    // free allocated arrays
    delete[] imgIn;
    delete[] imgOut;

#endif

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



