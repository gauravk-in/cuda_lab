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
using namespace std;

// uncomment to use the camera
//#define CAMERA

#define USING_GPU

// Image Threshold
__device__ void threshImage(float *in, float *out, float thresh, size_t count)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
        
    if(ix < count)
    {      
        out[ix] = ( in[ix] >= thresh );
    }
}

// Calculate the Intermediate Result required for Threshold
__device__ void intResult(float *in, float *out, int nc, size_t count)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
        
    if(ix < count)
    {
        for(int i = 0; i < nc; i++)
            out[ix] += in[ix + i * count];
        out[ix] = out[ix] / nc;
    }
}


__global__ void callKernel(float *in, float *out, float thresh, int nc, size_t count)
{
    intResult(in, out, nc, count);
    threshImage(out, out, thresh, count);
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

    float thresh = 0.5;


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
    //cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
    //cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
    cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
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
        size_t n_pixels = w * h;
        
        // Thread Dimensions
        dim3 block = dim3(256, 1, 1);
        dim3 grid = dim3((n_pixels + block.x - 1) / block.x, 1, 1);

        // Allocating memory on the device
        float *d_imgIn = NULL;
        float *d_imgOut = NULL;
        cudaMalloc(&d_imgIn, n_pixels * nc * sizeof(float));
        cudaMalloc(&d_imgOut, n_pixels * sizeof(float));
        
        // Copying Input image to device, and initializing result to 0
        cudaMemcpy(d_imgIn, imgIn, n_pixels * nc * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_imgOut, 0, n_pixels * sizeof(float));
        
        // Calling Kernel
        callKernel <<< grid, block >>> (d_imgIn, d_imgOut, thresh, nc, n_pixels);        
        
        // Copying result back
        cudaMemcpy(imgOut, d_imgOut, n_pixels * sizeof(float), cudaMemcpyDeviceToHost);
    
        // Freeing Memory
        cudaFree(d_imgIn);
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
        size_t n_pixels = w * h;
        memset(imgOut, 0, n_pixels * sizeof(float));
        for(size_t idx = 0; idx < n_pixels; idx++)
        {         
            for(int i = 0; i < nc; i++)
                imgOut[idx] += imgIn[idx + i * n_pixels];
            imgOut[idx] = imgOut[idx] / nc;    
            imgOut[idx] = (imgOut[idx] >= thresh);
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

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



