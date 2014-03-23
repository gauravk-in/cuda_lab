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

#define GL_GLEXT_PROTOTYPES

#include <GL/gl.h>
#include <GL/glext.h>
#include <GLFW/glfw3.h>
#include "cuda_gl_interop.h"


using namespace std;

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
 * @param U approximation of solution (single-channel)
 * @param F original input image (single-channel)
 * @param vx normalized gradient of U (x-coordinate)
 * @param vy normalized gradient of U (y-coordinate)
 * @param w width of image (pixels)
 * @param h height of image (pixels)
 * @param lambda weight of 'similarity' energy component
 * @param tau update coefficient
 */
#ifdef CAMERA
__global__ void update(float *output, float *U, float *F, float *vx, float *vy,
                       int w, int h, float lambda, float tau)
#else
__global__ void update(float *U, float *F, float *vx, float *vy,
                       int w, int h, float lambda, float tau)
#endif
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
        U[i] += tau * (lambda * d + div_v);

#ifdef CAMERA
        // float u = x / (float)w;
        // float v = y / (float)h;
        // output[i] = make_float4(u, v, U[i], 1.0f);
        output[i] = U[i];
#else        
#endif
    }
}

inline int div_ceil(int n, int b) { return (n + b - 1) / b; }

inline dim3 make_grid(dim3 whole, dim3 block)
{
    return dim3(div_ceil(whole.x, block.x),
                div_ceil(whole.y, block.y),
                div_ceil(whole.z, block.z));
}


// shader for displaying floating-point texture
static const char *shader_code =
    "!!ARBfp1.0\n"
    "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
    "END";

GLuint compileASMShader(GLenum program_type, const char *code)
{
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);

    GLint error_pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

    if (error_pos != -1)
    {
        const GLubyte *error_string;
        error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        cout << "Program error at position: " << (int)error_pos << endl << error_string << endl;
        return 0;
    }

    return program_id;
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
    int repeats = 1;
    getParam("repeats", repeats, argc, argv);
    cout << "repeats: " << repeats << endl;
    
    // load the input image as grayscale if "-gray" is specifed
    bool gray = true;
    // always true: getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;

    // ### Define your own parameters here as needed    
    float lambda = 0.8;
    getParam("lambda", lambda, argc, argv);
    cout << "λ: " << lambda << endl;

    float tau = 0.01;
    getParam("tau", tau, argc, argv);
    cout << "τ: " << tau << endl;

    int N = 2000;
    getParam("N", N, argc, argv);
    cout << "N: " << N << endl;

    // Init camera / Load input image
#ifdef CAMERA

    // Init camera
  	cv::VideoCapture camera(0);
  	if(!camera.isOpened()) { cerr << "ERROR: Could not open camera" << endl; return 1; }
    int camW = 640;
    int camH = 480;
  	camera.set(CV_CAP_PROP_FRAME_WIDTH,camW);
  	camera.set(CV_CAP_PROP_FRAME_HEIGHT,camH);
    // read in first frame to get the dimensions
    cv::Mat mIn;
    camera >> mIn;
    if(gray)
        cvtColor (mIn, mIn, CV_BGR2GRAY);
    
#else
    
    // Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
    cv::Mat mIn = cv::imread(image.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
    // check
    if (mIn.data == NULL) { cerr << "ERROR: Could not load image " << image << endl; return 1; }
    
#endif

    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
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
    float *imgIn  = new float[(size_t)w*h*nc];
    size_t imageBytes = (size_t)w*h*nc*sizeof(float);

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[(size_t)w*h*mOut.channels()];




    // For camera mode: Make a loop to read in camera frames
#ifdef CAMERA

    // KUKU: OpenGL GLFW Code
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    GLFWwindow* window = glfwCreateWindow(640, 480, "Project", NULL, NULL); // Windowed

    glfwMakeContextCurrent(window);

    // KUKU: Interoperability Code  

    GLuint outputVBO;
    struct cudaGraphicsResource* outputVBO_CUDA;
    GLuint texid;   // Texture
    GLuint shader;


    // Explicitly set device 0
    cudaGLSetGLDevice(0);

    // create pixel buffer object
    glGenBuffersARB(1, &outputVBO);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, outputVBO);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, w*h*sizeof(GLfloat), 0, GL_STREAM_DRAW_ARB);

    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    cudaGraphicsGLRegisterBuffer(&outputVBO_CUDA, outputVBO,
                                 cudaGraphicsMapFlagsWriteDiscard);


    // // Create buffer object and register it with CUDA
    // glGenBuffers(1, &outputVBO);
    // glBindBuffer(GL_ARRAY_BUFFER, outputVBO);   
    // unsigned int size = w * h * 4 * sizeof(float);
    // glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    // glBindBuffer(GL_ARRAY_BUFFER, 0);
    // cudaGraphicsGLRegisterBuffer(&outputVBO_CUDA,
    //                              outputVBO,
    //                              cudaGraphicsMapFlagsWriteDiscard);

    // load shader program
    shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);

    // create texture for display
    glGenTextures(1, &texid);
    glBindTexture(GL_TEXTURE_2D, texid);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, w, h, 0, GL_LUMINANCE, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);


    // Read a camera image frame every 30 milliseconds:
    // cv::waitKey(30) waits 30 milliseconds for a keyboard input,
    // returns a value <0 if no key is pressed during this time, returns immediately with a value >=0 if a key is pressed
    while (!glfwWindowShouldClose(window))
    {
    // Get camera image
    camera >> mIn;
    cvtColor(mIn, mIn, CV_BGR2GRAY);
    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;

    float* d_output;
    cudaGraphicsMapResources(1, &outputVBO_CUDA, 0);
    size_t num_bytes; 
    cudaGraphicsResourceGetMappedPointer((void**)&d_output,
                                         &num_bytes,  
                                         outputVBO_CUDA);

#endif

    // Init raw input image array
    // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
    // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
    // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
    convert_mat_to_layered (imgIn, mIn);

    dim3 block(32, 16);
    dim3 grid = make_grid(dim3(w, h, 1), block);

    Timer timer; timer.start();
    float *d_F, *d_U, *d_vx, *d_vy;
    cudaMalloc(&d_F, imageBytes);
    cudaMalloc(&d_U, imageBytes);
    cudaMalloc(&d_vx, imageBytes);
    cudaMalloc(&d_vy, imageBytes);
    cudaMemcpy(d_F, imgIn, imageBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, imgIn, imageBytes, cudaMemcpyHostToDevice);

    for (int n = 0; n < N; n++) {
        norm_grad<<< grid, block >>>(d_U, d_vx, d_vy, w, h);
#ifdef CAMERA
        update<<< grid, block >>>(d_output, d_U, d_F, d_vx, d_vy, w, h, lambda, tau);
#else
        update<<< grid, block >>>(d_U, d_F, d_vx, d_vy, w, h, lambda, tau);
#endif
    }

#ifdef CAMERA
    // Unmap buffer object
    cudaGraphicsUnmapResources(1, &outputVBO_CUDA, 0);
#else
    cudaMemcpy(imgOut, d_U, imageBytes, cudaMemcpyDeviceToHost);
#endif    
    cudaFree(d_F);
    cudaFree(d_U);
    cudaFree(d_vx);
    cudaFree(d_vy);
    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "time: " << t*1000 << " ms" << endl;


#ifdef CAMERA
    // Render from buffer object
    // OpenGL display code path
    {
        glClear(GL_COLOR_BUFFER_BIT);

        // load texture from pbo
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, outputVBO);
        glBindTexture(GL_TEXTURE_2D, texid);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_LUMINANCE, GL_FLOAT, 0);
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

        // fragment program is required to display floating point texture
        glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, shader);
        glEnable(GL_FRAGMENT_PROGRAM_ARB);
        glDisable(GL_DEPTH_TEST);

        glBegin(GL_QUADS);
        {
            glTexCoord2f(0.0f, 0.0f);
            glVertex2f(0.0f, 0.0f);
            glTexCoord2f(1.0f, 0.0f);
            glVertex2f(1.0f, 0.0f);
            glTexCoord2f(1.0f, 1.0f);
            glVertex2f(1.0f, 1.0f);
            glTexCoord2f(0.0f, 1.0f);
            glVertex2f(0.0f, 1.0f);
        }
        glEnd();
        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_FRAGMENT_PROGRAM_ARB);
    }
#else
    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, imgOut);
    showImage("Output", mOut, 100+w+40, 100);
#endif
    // ### Display your own output images here as needed

#ifdef CAMERA
    glfwSwapBuffers(window);
    glfwPollEvents();

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);

    // end of camera loop
    }
    glfwTerminate();
#else
    // wait for key inputs
    cv::waitKey(0);

    // save input and result
    cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
    cv::imwrite("image_result.png",mOut*255.f);
#endif


    // free allocated arrays
    delete[] imgIn;
    delete[] imgOut;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



