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


#include "aux.h"
#include <iostream>

// #include <GLTools.h>            // OpenGL toolkit
// #include <GLShaderManager.h>    // Shader Manager Class

#define GL_GLEXT_PROTOTYPES

#include <GL/gl.h>
#include <GL/glut.h>            // Windows FreeGlut equivalent
#include <GL/glext.h>
#include "cuda_gl_interop.h"

using namespace std;

// uncomment to use the camera
#define CAMERA

/*********************************************************************
 *** Global Variables                                           ******
 *********************************************************************/

// Pointers to Device Memory for Input, Output and Intermediate images
float *d_F, *d_U, *d_vx, *d_vy;

// Variables for OpenGL Interoperability
GLuint outputVBO;
struct cudaGraphicsResource* outputVBO_CUDA;

int repeats;
bool gray;
float lambda;
float tau;
int N;

cv::VideoCapture camera(0);

cv::Mat mIn;
float *imgIn;

int w, h, nc;


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




/**
 * Computes the normalized gradient.
 *
 * @param U input image (single-channel)
 * @param vx x-coordinate of result
 * @param vy y-coordinate of result
 * @param w width of image (pixels)
 * @param h height of image (pixels)
 */
__global__ void norm_grad(float *U, float *vx, float *vy, int w, int h)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x < w && y < h) {
        size_t i = x + (size_t)w*y;
        float ux = ((x+1 < w) ? (U[i + 1] - U[i]) : 0);
        float uy = ((y+1 < h) ? (U[i + w] - U[i]) : 0);
        float gn = sqrtf(ux*ux + uy*uy);
        if (fabsf(gn) < FLT_EPSILON) {
            // Prevent division by zero. Result will be null vector.
            vx[i] = 0.0;
            vy[i] = 0.0;
        } else {
            vx[i] = ux / gn;
            vy[i] = uy / gn;
        }
    }
}

/**
 * Update approximation.
 *
 * @param output
 * @param U approximation of solution (single-channel)
 * @param F original input image (single-channel)
 * @param vx normalized gradient of U (x-coordinate)
 * @param vy normalized gradient of U (y-coordinate)
 * @param w width of image (pixels)
 * @param h height of image (pixels)
 * @param lambda weight of 'similarity' energy component
 * @param tau update coefficient
 */
__global__ void update(float* output, float *U, float *F, float *vx, float *vy,
                       int w, int h, float lambda, float tau)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x < w && y < h) {
        size_t i = x + (size_t)w*y;

        // similarity to input image (functional derivative of energy)
        float d = F[i] - U[i];
        if (fabsf(d) < FLT_EPSILON * U[i])
            d = 0.f;
        else
            d = ((d > 0) ? 1.f : -1.f);

        // smoothness (functional derivative of energy)
        float dx_vx = ((x+1 < w) ? vx[i] : 0) - ((x > 0) ? vx[i - 1] : 0);
        float dy_vy = ((y+1 < h) ? vy[i] : 0) - ((y > 0) ? vy[i - w] : 0);
        float div_v = dx_vx + dy_vy;

        // explicit Euler update rule
        output[i] = U[i] + tau * (lambda * d + div_v);
    }
}

inline int div_ceil(int n, int b) { return (n + b - 1) / b; }

inline dim3 make_grid(dim3 whole, dim3 block)
{
    return dim3(div_ceil(whole.x, block.x),
                div_ceil(whole.y, block.y),
                div_ceil(whole.z, block.z));
}

void display();

void bye() {
    cout << "bye!\n";
}

int main(int argc, char **argv)
{
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
    lambda = 0.8;
    getParam("lambda", lambda, argc, argv);
    cout << "λ: " << lambda << endl;

    tau = 0.01;
    getParam("tau", tau, argc, argv);
    cout << "τ: " << tau << endl;

    N = 2000;
    getParam("N", N, argc, argv);
    cout << "N: " << N << endl;

    // Init camera / Load input image
#ifdef CAMERA
    // Init camera
  	// cv::VideoCapture camera(0);
  	if(!camera.isOpened()) { cerr << "ERROR: Could not open camera" << endl; return 1; }
    int camW = 640;
    int camH = 480;
  	camera.set(CV_CAP_PROP_FRAME_WIDTH,camW);
  	camera.set(CV_CAP_PROP_FRAME_HEIGHT,camH);
    // read in first frame to get the dimensions
    // cv::Mat mIn;
    camera >> mIn;
#else
    
    // Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
    // cv::Mat mIn = cv::imread(image.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
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




    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    imgIn  = new float[(size_t)w*h*nc];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[(size_t)w*h*mOut.channels()];

#ifdef CAMERA

    // Initialize OpenGL and GLUT for device 0
    // and make the OpenGL context current
    glutInit(&argc, argv);

    cout << __func__ << ": "<< __LINE__ << ": " << endl;

    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100,100);
    glutInitWindowSize(640,480);
    glutCreateWindow("Project");
    glutDisplayFunc(&display);

    cout << __func__ << ": "<< __LINE__ << ": " << endl;

    // Explicitly set device 0
    cudaGLSetGLDevice(0);

    cout << __func__ << ": "<< __LINE__ << ": " << endl;

    // Create buffer object and register it with CUDA
    glGenBuffers(1, &outputVBO);
    glBindBuffer(GL_ARRAY_BUFFER, outputVBO);
    unsigned int size = w * h * nc * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&outputVBO_CUDA,
                                 outputVBO,
                                 cudaGraphicsMapFlagsWriteDiscard);

    cout << __func__ << ": "<< __LINE__ << ": " << endl;

    // Launch rendering loop
    glutMainLoop();
    cout << __func__ << ": "<< __LINE__ << ": " << endl;
    atexit(bye);

#else // CAMERA
    size_t imageBytes = (size_t)w*h*nc*sizeof(float);

    // Init raw input image array
    // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
    // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
    // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
    convert_mat_to_layered (imgIn, mIn);

    dim3 block(32, 16);
    dim3 grid = make_grid(dim3(w, h, 1), block);

    Timer timer; timer.start();
    // float *d_F, *d_U, *d_vx, *d_vy;
    cudaMalloc(&d_F, imageBytes);
    cudaMalloc(&d_U, imageBytes);
    cudaMalloc(&d_vx, imageBytes);
    cudaMalloc(&d_vy, imageBytes);
    cudaMemcpy(d_F, imgIn, imageBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, imgIn, imageBytes, cudaMemcpyHostToDevice);

    for (int n = 0; n < N; n++) {
        norm_grad<<< grid, block >>>(d_U, d_vx, d_vy, w, h);
        update<<< grid, block >>>(d_U, d_U, d_F, d_vx, d_vy, w, h, lambda, tau);
    }

    cudaMemcpy(imgOut, d_U, imageBytes, cudaMemcpyDeviceToHost);
    cudaFree(d_F);
    cudaFree(d_U);
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

    // close all opencv windows
    cvDestroyAllWindows();

    return 0;
#endif

}

void display()
{
    
    cout << __func__ << ": "<< __LINE__ << ": " << endl;

    float *d_output;
    cudaGraphicsMapResources(1, &outputVBO_CUDA, 0);
    size_t num_bytes; 
    cudaGraphicsResourceGetMappedPointer((void**)&d_output, &num_bytes, outputVBO_CUDA);

    size_t imageBytes = (size_t)w*h*nc*sizeof(float);

    cout << __func__ << ": "<< __LINE__ << ": " << endl;

    // Read a camera image frame every 30 milliseconds:
    // cv::waitKey(30) waits 30 milliseconds for a keyboard input,
    // returns a value <0 if no key is pressed during this time, returns immediately with a value >=0 if a key is pressed
    cv::waitKey(30);
    
    // Get camera image
    camera >> mIn;
    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;

    // Init raw input image array
    // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
    // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
    // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
    convert_mat_to_layered (imgIn, mIn);

    dim3 block(32, 16);
    dim3 grid = make_grid(dim3(w, h, 1), block);

    Timer timer; timer.start();
    // float *d_F, *d_U, *d_vx, *d_vy;
    cudaMalloc(&d_F, imageBytes);
    cudaMalloc(&d_U, imageBytes);
    cudaMalloc(&d_vx, imageBytes);
    cudaMalloc(&d_vy, imageBytes);
    cudaMemcpy(d_F, imgIn, imageBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, imgIn, imageBytes, cudaMemcpyHostToDevice);

    for (int n = 0; n < N; n++) {
        norm_grad<<< grid, block >>>(d_U, d_vx, d_vy, w, h);
        update<<< grid, block >>>(d_output, d_U, d_F, d_vx, d_vy, w, h, lambda, tau);
    }

    // cudaMemcpy(imgOut, d_U, imageBytes, cudaMemcpyDeviceToHost);
    cudaFree(d_F);
    cudaFree(d_U);
    cudaFree(d_vx);
    cudaFree(d_vy);
    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "time: " << t*1000 << " ms" << endl;

    // Unmap buffer object
    cudaGraphicsUnmapResources(1, &outputVBO_CUDA, 0);

    // Render from buffer object
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindBuffer(GL_ARRAY_BUFFER, outputVBO);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_POINTS, 0, w * h);
    glDisableClientState(GL_VERTEX_ARRAY);

    // Swap buffers
    glutSwapBuffers();
    glutPostRedisplay();

}


