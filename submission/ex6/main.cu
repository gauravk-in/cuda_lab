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
// ### Gaurav Kukreja, gaurav.kukreja@tum.de, p058 
// ### Ravikishore Kommajosyula, r.kommajosyula@tum.de, p059
// ### Miklos Homolya, miklos.homolya@tum.de, p057
// ###


#include "aux.h"
#include <iostream>
#include <math.h>
using namespace std;

// uncomment to use the camera
#define CAMERA

#define USING_GPU

    template<typename T>
__device__ T gpu_min(T a, T b)
{
    if (a < b)
        return a;
    else
        return b;
}

    template<typename T>
__device__ T gpu_max(T a, T b)
{
    if (a < b)
        return b;
    else
        return a;
}


// Image Gradient 
__device__ void convolveImage(float* imgIn, float* kernel, float* imgOut, int rad, int w, int h, int nc) 
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int iz = threadIdx.z + blockDim.z * blockIdx.z;

    // Index of the output image, this kernel works on
    int idx = ix + (iy * w) + (iz * w * h);  

    // check limits
    if (idx < w * h * nc)
    { 
        imgOut[idx] = 0;						    // initialize
        float value = 0;
        for(int j = -rad; j < rad; j++)					    // for each row in kernel
        {   
            int iny = gpu_max(0, gpu_min(iy+j, h-1));			    
            for(int i = -rad; i < rad; i++)				    // for each element in the kernel row
            {
                int inx = gpu_max(0, gpu_min(ix+i, w-1));
                int inIdx = inx + (iny * w) + (iz * w * h);		    // Index of Input Image to be multiplied by corresponding element in kernel
                value += imgIn[inIdx] * kernel[i+rad + ((j+rad) * rad)];
            }
        }
        imgOut[idx] = value;
    }
}

__global__ void callKernel(float* imgIn, float* kernel, float* imgOut, int rad, int w, int h, int nc)
{
    convolveImage(imgIn, kernel, imgOut, rad, w, h, nc);    
}

int main(int argc, char **argv)
{
#ifdef USING_GPU
    // Before the GPU can process your kernels, a so called "CUDA context" must be initialized
    // This happens on the very first call to a CUDA function, and takes some time (around half a second)
    // We will do it right here, so that the run time measurements are accurate
    cudaDeviceSynchronize();  CUDA_CHECK;
#endif // USING_GPU



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
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> [-repeats <repeats>] [-gray] [-sigma <sigma>]" << endl << "\t Default Value of sigma = 0.5" << endl; return 1; }
#endif

    // number of computation repetitions to get a better run time measurement
    int repeats = 1;
    getParam("repeats", repeats, argc, argv);
    cout << "repeats: " << repeats << endl;

    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;

    float sigma = 0.5;
    getParam("sigma", sigma, argc, argv);
    cout << "sigma = " << sigma << endl;

    // ### Define your own parameters here as needed    

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
    // ###
    // ###
    // ### TODO: Change the output image format as needed
    // ###
    // ###
    cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
    //cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
    //cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
    // ### Define your own output images here as needed




    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    float *imgIn  = new float[(size_t)w*h*nc];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[(size_t)w*h*mOut.channels()];

    int rad = ceil(3 * sigma); // kernel radius
    int kw = 2 * rad; // kernel width
    float c = 1. / (2. * 3.142857 * sigma * sigma); // constant

    float *kernel =  new float[(size_t) (kw * kw)]; // kernel
    float *kernelOut = new float[(size_t) (kw * kw)]; // kernel to be displayed

    // Computation of Kernel
    float a;
    float b;
    for (int iy = 0; iy < kw; iy++)
    {
        a = iy - rad;
        for (int ix = 0; ix < kw; ix++)
        {
            b = ix - rad;
            kernel[ix + (iy * kw)] = c * exp(-(a*a + b*b) / (2 * sigma*sigma));
        }
    }

    // Normalization of Kernel
    float sum = 0.;
    float kmax = 0.;
    for (int iy = 0; iy < kw; iy++)
    {
        for (int ix = 0; ix < kw; ix++)
        {
            kmax = max(kmax, kernel[ix + (iy * kw)]);
            sum += kernel[ix + (iy * kw)];
        }
    }

    for (int iy = 0; iy < kw; iy++)
    {
        for (int ix = 0; ix < kw; ix++)
        {
            kernelOut[ix + (iy * kw)] = kernel[ix + (iy * kw)] / kmax;
            kernel[ix + (iy * kw)] = kernel[ix + (iy * kw)] / sum;
        }
    }

    // Display Kernel
    cv::Mat cvKernelOut(2*rad, 2*rad, CV_32F);
    convert_layered_to_mat(cvKernelOut, kernelOut);
    showImage("Kernel", cvKernelOut, 100, 10);


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

        Timer timer;
        float t;
        // ###
        // ###
        // ### TODO: Main computation
        // ###
        // ###
#ifdef USING_GPU
        timer.start();

        // Repetitions Loop
        for(int rep = 0; rep < repeats; rep++)
        {
            size_t count = w * h * nc;        

            // Thread Dimensions
            dim3 block = dim3(16, 4, nc);
            dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);

            // Allocating memory on the device
            float *d_imgIn = NULL;
            float *d_imgOut = NULL;
            float *d_kernel = NULL;
            cudaMalloc(&d_imgIn, count * sizeof(float));
            cudaMalloc(&d_imgOut, count * sizeof(float));
            cudaMalloc(&d_kernel, kw * kw * sizeof(float));

            // Copying Input image to device, and initializing result to 0
            cudaMemcpy(d_imgIn, imgIn, count * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_kernel, kernel, kw * kw * sizeof(float), cudaMemcpyHostToDevice);

            // Calling Kernel
            callKernel <<< grid, block >>> (d_imgIn, d_kernel, d_imgOut, rad, w, h, nc);        

            // Copying result back
            cudaMemcpy(imgOut, d_imgOut, count * sizeof(float), cudaMemcpyDeviceToHost);

            CUDA_CHECK;

            // Freeing Memory
            cudaFree(d_imgIn);
            cudaFree(d_kernel);
            cudaFree(d_imgOut);
        }

        timer.end();
        t = timer.get();

#else // USING_GPU
        // CPU Implementation

        timer.start();

        // Repetitions Loop
        for(int rep = 0; rep < repeats; rep++)    
        {
            for(int ix = 0; ix < w; ix++)
            {
                for(int iy = 0; iy < h; iy++)
                {
                    for(int iz = 0; iz < nc; iz++)
                    {
                        int idx = ix + (iy * w) + (iz * w * h);
                        imgOut[idx] = 0;                                                    // initialize
                        float value = 0;
                        for(int j = -rad; j < rad; j++)                                     // for each row in kernel
                        {
                            int iny = max(0, min(iy+j, h-1));
                            for(int i = -rad; i < rad; i++)                                 // for each element in the kernel row
                            {
                                int inx = max(0, min(ix+i, w-1));
                                int inIdx = inx + (iny * w) + (iz * w * h);                 // Index of Input Image to be multiplied by corresponding element in kernel
                                value += imgIn[inIdx] * kernel[i+rad + ((j+rad) * rad)];
                            }
                        }
                        imgOut[idx] = value;
                    }
                }
            }

        }

        timer.end();  
        t = timer.get();  // elapsed time in seconds

#endif

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
    delete[] kernel;
    delete[] kernelOut;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



