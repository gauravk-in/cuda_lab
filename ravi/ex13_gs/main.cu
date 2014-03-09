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


__device__ __host__ float huber(float s, float epsilon)
{
    return 1.0F / max(epsilon, s);
    //return 1.0F;
    //return expf(-s*s / epsilon) / epsilon;
}

__global__ void compute_g(float *image, float *g, int w, int h, int nc, float epsilon)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < w && y < h) {
//        int b = threadIdx.x + blockDim.x * threadIdx.y;
//        int B = blockDim.x * blockDim.y;

        float G2 = 0.0f;
        for (int c = 0; c < nc; c++) {
            size_t idx = x + (size_t)w*y + (size_t)w*h*c;
            float ux = ((x < w-1) ? (image[idx + 1] - image[idx]) : 0);
            float uy = ((y < h-1) ? (image[idx + w] - image[idx]) : 0);
            G2 += ux*ux + uy*uy;
        }

        g[x + (size_t) w*y] = huber(sqrtf(G2), epsilon);
    }
}

__global__ void rb_gs_update(float *image, float *image_noisy, float *g, int w, int h, int nc, float lambda, float theta, int red_black)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if( ((x + y) % 2) != red_black) {
    	return;
    }

    if (x < w && y < h) {
        for (int c = 0; c < nc; c++) {
            size_t idx = x + (size_t)w*y + (size_t)w*h*c;
            size_t idx_2d = x + (size_t)w*y;
            float temp_uxy = image[idx];
            float gsum_u = (((x+1) < (w) ? 1.0f : 0.0f) * (g[idx_2d + 1]) * (image[idx + 1])) +
    				(((x) > 0 ? 1.0f : 0.0f) * (g[idx_2d - 1]) * ( image[idx - 1])) +
    				(((y+1) < (h) ? 1.0f : 0.0f) * (g[idx_2d + w]) * ( image[idx + w])) +
    				(((y) > 0 ? 1.0f : 0.0f ) * (g[idx_2d - w]) * ( image[idx - w]));

            float gsum = ((x+1) < (w) ? 1.0f : 0.0f) * (g[idx_2d + 1]) +
    				((x) > 0 ? 1.0f : 0.0f) * (g[idx_2d - 1]) +
    				((y+1) < (h) ? 1.0f : 0.0f) * (g[idx_2d + w]) +
    				((y) > 0 ? 1.0f : 0.0f ) * (g[idx_2d - w]);
            float gs_result = ( (2 * image_noisy[idx]) + (lambda * gsum_u)) / ( 2 + (lambda * gsum) );

            // SOR step
            image[idx] = gs_result + theta * ( gs_result - temp_uxy );
        }
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
    float sigma = 0.1;
    getParam("sigma", sigma, argc, argv);
    cout << "σ: " << sigma << endl;

    float lambda = 0.2;
    getParam("lambda", lambda, argc, argv);
    cout << "lambda: " << lambda << endl;

    float theta = 0.8;
    getParam("theta", theta, argc, argv);
    cout << "theta: " << theta << endl;

    float epsilon = 0.01;
    getParam("epsilon", epsilon, argc, argv);
    cout << "ε: " << epsilon << endl;

    int N = 100;
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

    // Add noise to the image here - mIn from here has the noise
    addNoise(mIn, sigma);

    // Init raw input image array
    // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
    // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
    // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
    convert_mat_to_layered (imgIn, mIn);

    dim3 block(32, 16);
    dim3 grid = make_grid(dim3(w, h, 1), block);
    size_t smBytes = (size_t)block.x*block.y*nc*2*sizeof(float);

    Timer timer; timer.start();
    float *d_image, *d_noisy, *d_g;
    cudaMalloc(&d_image, imageBytes);
    cudaMalloc(&d_noisy, imageBytes);
    cudaMalloc(&d_g, w*h*sizeof(float));
    cudaMemcpy(d_image, imgIn, imageBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_noisy, d_image, imageBytes, cudaMemcpyDeviceToDevice);

    for (int n = 0; n < N; n++) {
        compute_g<<< grid, block, smBytes >>>(d_image, d_g, w, h, nc, epsilon);
        rb_gs_update<<< grid, block >>>(d_image, d_noisy, d_g, w, h, nc, lambda, theta, 0);
        rb_gs_update<<< grid, block >>>(d_image, d_noisy, d_g, w, h, nc, lambda, theta, 1);
    }

    cudaMemcpy(imgOut, d_image, imageBytes, cudaMemcpyDeviceToHost);
    cudaFree(d_image);
    cudaFree(d_noisy);
    cudaFree(d_g);
    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "time: " << t*1000 << " ms" << endl;






    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, imgOut);
    showImage("Output", mOut, 100+w+40, 100);

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

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



