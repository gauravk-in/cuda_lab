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

// Image Gradient 
__device__ void gradImage(float* imgIn, float* forwardDiffX, float* forwardDiffY, float *imgOut, int w, int h, int nc)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;

    float value = 0.; 
    for(int i = 0; i < nc; i++)
    {
	int idx = ix + iy*w + i*w*h;
        forwardDiffX[idx] = (ix < (w-2)) ? (imgIn[idx + 1] - imgIn[idx]) : 0;
        forwardDiffY[idx] = (iy < (h-2)) ? (imgIn[idx + w] - imgIn[idx]) : 0;
        value += pow(forwardDiffX[idx] , 2) +  pow(forwardDiffY[idx] , 2);
    }
    imgOut[ix + (iy * w)] = sqrt(value);
}

__global__ void callKernel(float* imgIn, float* forwardDiffX, float* forwardDiffY, float* imgOut, int w, int h, int nc)
{
    gradImage(imgIn, forwardDiffX, forwardDiffY, imgOut, w, h, nc);    
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
        dim3 block = dim3(32, 8, 1);
        dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y) / block.y, 1);

        // Allocating memory on the device
        float *d_imgIn = NULL;
        float *d_forwardDiffX = NULL;
        float *d_forwardDiffY = NULL;
        float *d_imgOut = NULL;
        cudaMalloc(&d_imgIn, n_pixels * nc * sizeof(float));
        cudaMalloc(&d_forwardDiffX, n_pixels * nc * sizeof(float));
        cudaMalloc(&d_forwardDiffY, n_pixels * nc * sizeof(float));                
        cudaMalloc(&d_imgOut, n_pixels *  sizeof(float));
        
        // Copying Input image to device, and initializing result to 0
        cudaMemcpy(d_imgIn, imgIn, n_pixels * nc * sizeof(float), cudaMemcpyHostToDevice);
        
        // Calling Kernel
        callKernel <<< grid, block >>> (d_imgIn, d_forwardDiffX, d_forwardDiffY, d_imgOut, w, h, nc);        
        
        // Copying result back
        cudaMemcpy(imgOut, d_imgOut, n_pixels * sizeof(float), cudaMemcpyDeviceToHost);
 
	CUDA_CHECK;
 
        // Freeing Memory
        cudaFree(d_imgIn);
	cudaFree(d_forwardDiffX);
	cudaFree(d_forwardDiffY);
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
	float *forwardDiffX = NULL;
	float *forwardDiffY = NULL;

	forwardDiffX = (float*) malloc(sizeof(float) * w * h * nc);
	forwardDiffY = (float*) malloc(sizeof(float) * w * h * nc);

        memset(imgOut, 0, n_pixels * sizeof(float));
	
	for(int ix = 0; ix < w; ix ++)
	    for(int iy = 0; iy < h; iy++)
	    {
		imgOut[ix + (iy * w)] = 0;
	        for(int i = 0; i < nc; i++)
	        {
	            forwardDiffX[ix + (iy * w) + (i * w * h)] = (ix < (w-2)) ? (imgIn[ix + (iy * w) + (i * w * h) + 1] - imgIn[ix + (iy * w) + (i * w * h)]) : 0;
	            forwardDiffY[ix + (iy * w) + (i * w * h)] = (iy < (h-2)) ? (imgIn[ix + ((iy + 1) * w) + (i * w * h)] - imgIn[ix + (iy * w) + (i * w * h)]) : 0; 
	            imgOut[ix + (iy * w)] += pow(forwardDiffX[ix + (iy * w) + (i * w * h)] , 2) +  pow(forwardDiffY[ix + (iy * w) + (i * w * h)] , 2);
	        }
	        imgOut[ix + (iy * w)] = sqrt(imgOut[ix + (iy * w)]);
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



