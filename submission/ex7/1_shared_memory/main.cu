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
// ### TODO: For every student of your group, please provide here:
// ###
// ### name, email, login username (for example p123)
// ###
// ###


#include "aux.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;

// uncomment to use the camera
//#define CAMERA

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

__global__ void convolution(float *in, float *out, float *kern, int w, int h, int r)
{
    int ksize = 2*r + 1;
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    // Load 'in' to shared memory
    extern __shared__ float s_in[];
    int nThreads = blockDim.x * blockDim.y;
    int threadId = threadIdx.x + blockDim.x * threadIdx.y;
    int V = blockDim.x + 2*r;
    int G = blockDim.y + 2*r;
    int smLength = V * G;
    for (int i = threadId; i < smLength; i += nThreads) {
        int rx = i % V;
        int ry = i / V;
        int cx = clamp<int>(0, blockDim.x*blockIdx.x + rx - r, w-1);
        int cy = clamp<int>(0, blockDim.y*blockIdx.y + ry - r, h-1);
        s_in[i] = in[cx + w*cy];
    }

    __syncthreads();

    // Do the job!
    if (x < w && y < h) {
        float value = 0;
        for (int ky = 0; ky < ksize; ky++) {
            int ry = threadIdx.y + ky;
            for (int kx = 0; kx < ksize; kx++) {
                int rx = threadIdx.x + kx;
                value += kern[kx + ksize*ky] * s_in[rx + V*ry];
            }
        }
        out[x + w*y] = value;
    }
}

inline int div_ceil(int n, int b) { return (n + b - 1) / b; }

inline dim3 make_grid(dim3 whole, dim3 block)
{
    return dim3(div_ceil(whole.x, block.x),
                div_ceil(whole.y, block.y),
                div_ceil(whole.z, block.z));
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
    bool gray = false;
    getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;

    // ### Define your own parameters here as needed    
    float sigma = 2.0;
    getParam("sigma", sigma, argc, argv);
    cout << "sigma: " << sigma << endl;

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

    // Size of the kernel
    int r = ceil(3 * sigma);
    int ksize = 2*r + 1;

    float *kern = new float[ksize * ksize];
    for (int i = 0; i < 2*r+1; i++) {
        double a = i - r;
        for (int j = 0; j < 2*r+1; j++) {
            double b = j - r;
            kern[i*ksize + j] = exp(-(a*a + b*b) / (2 * sigma*sigma))
                                / (2 * M_PI * sigma*sigma);
        }
    }

    float kernMax = 0;
    float kernSum = 0;
    for (int i = 0; i < 2*r+1; i++) {
        for (int j = 0; j < 2*r+1; j++) {
            kernSum += kern[i*ksize + j];
            kernMax = std::max(kernMax, kern[i*ksize + j]);
        }
    }

    float *kernOut = new float[(2*r + 1) * (2*r + 1)];
    for (int i = 0; i < 2*r+1; i++) {
        for (int j = 0; j < 2*r+1; j++) {
            kernOut[i*ksize + j] = kern[i*ksize + j] / kernMax;
            kern[i*ksize + j] /= kernSum;
        }
    }

    cv::Mat mKernOut(ksize, ksize, CV_32F);
    convert_layered_to_mat(mKernOut, kernOut);

    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    float *imgIn  = new float[(size_t)w*h*nc];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[(size_t)w*h*mOut.channels()];




    // For camera mode: Make a loop to read in camera frames
#ifdef CAMERA
    // Read a camera image frame every 30 milliseconds:
    // cv::waitKey(30) waits 30 milliseconds for a keyboard input,
    // returns a value <0 if no key is pressed during this time, returns immediately with a value >=0 if a key is pressed
    while (cv::waitKey(30) < 0)
    {
    // Get camera image
    camera >> mIn;
    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
#endif

    // Init raw input image array
    // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
    // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
    // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
    convert_mat_to_layered (imgIn, mIn);


    dim3 block(16, 16, 1);
    dim3 grid = make_grid(dim3(w, h, 1), block);
    size_t smBytes = (block.x + 2*r) * (block.y + 2*r) * sizeof(float);
    cout << "shared memory: " << smBytes << " bytes" << endl;


    Timer timer; timer.start();
    for (int measurement = 0; measurement < repeats; measurement++) {
//#define CPU
#ifdef CPU
    for (int c = 0; c < nc; c++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float value = 0;
                for (int ky = 0; ky < ksize; ky++) {
                    int cy = clamp(0, y + ky - r, h-1);
                    for (int kx = 0; kx < ksize; kx++) {
                        int cx = clamp(0, x + kx - r, w-1);
                        value += kern[kx + ksize*ky] * imgIn[cx + w*cy + w*h*c];
                    }
                }
                imgOut[x + w*y + w*h*c] = value;
            }
        }
    }
#else
    float *d_in, *d_out, *d_kern;
    size_t pixels = (size_t)w*h;
    size_t nbytes = (size_t)w*h*nc*sizeof(float);
    cudaMalloc(&d_in, nbytes);
    cudaMalloc(&d_out, nbytes);
    cudaMalloc(&d_kern, (size_t)ksize*ksize*sizeof(float));
    cudaMemcpy(d_in, imgIn, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kern, kern, (size_t)ksize*ksize*sizeof(float), cudaMemcpyHostToDevice);
    for (int c = 0; c < nc; c++) {
        convolution<<<grid, block, smBytes>>>(d_in + c*pixels, d_out + c*pixels,
                                              d_kern, w, h, r);
    }
    cudaMemcpy(imgOut, d_out, nbytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_kern);
#endif
    }
    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "time: " << (t / repeats)*1000 << " ms" << endl;






    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, imgOut);
    showImage("Output", mOut, 100+w+40, 100);

    // show kernel
    showImage("Kernel", mKernOut, 100+2*(w+40), 100);

    // ### Display your own output images here as needed

#ifdef CAMERA
    // end of camera loop
    }
#else
    // wait for key inputs
    cv::waitKey(0);
#endif




    // save input and result
    cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
    cv::imwrite("image_result.png",mOut*255.f);

    // free allocated arrays
    delete[] imgIn;
    delete[] imgOut;
    delete[] kern;
    delete[] kernOut;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



