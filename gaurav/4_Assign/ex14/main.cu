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
// ###
// ###


#include "aux.h"
#include <iostream>
#include <math.h>
using namespace std;

// uncomment to use the camera
//#define CAMERA

#define USING_GPU


// This function finds green colored pixels in the image
// It outputs a bool array mask, which tells if pixel(x, y) is green
// It also sets the green pixel value to (0.5, 0.5, 0.5)
__global__ void findGreen(float* imgIn, float* imgOut, bool* d_mask, size_t n_pixels, int w, int h, int nc)
{
    size_t ix = threadIdx.x + blockDim.x * blockIdx.x;
    size_t iy = threadIdx.y + blockDim.y * blockIdx.y;

    if(ix < w && iy < h && nc == 3)
    {
    	// Only the green Layer
    	size_t b_idx = ix + (iy * w);
    	size_t g_idx = b_idx + (w * h);
    	size_t r_idx = g_idx + (w * h);

    	if(imgIn[g_idx] == 1.0f && imgIn[b_idx] == 0.0f && imgIn[r_idx] == 0.0f)
    	{
    		d_mask[b_idx] = true;
    		imgOut[b_idx] = 0.5f;							// Blue
    		imgOut[g_idx] = 0.5f;							// Green
    		imgOut[r_idx] = 0.5f;							// Red
    	}
    	else
    	{
    		d_mask[b_idx] = false;
    		imgOut[b_idx] = imgIn[b_idx];					// Blue
    		imgOut[g_idx] = imgIn[g_idx];					// Green
    		imgOut[r_idx] = imgIn[r_idx];					// Red
    	}
    }
}

__device__ __host__ float huber(float s, float epsilon)
{
    return 1.0F / max(epsilon, s);
    //return 1.0F;
    //return expf(-s*s / epsilon) / epsilon;
}

__global__ void compute_P(float *image, float *Px, float *Py, int w, int h, int nc, float epsilon)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    extern __shared__ float sh_u[];

    if (x < w && y < h) {
        int b = threadIdx.x + blockDim.x * threadIdx.y;
        int B = blockDim.x * blockDim.y;

        float G2 = 0;
        for (int c = 0; c < nc; c++) {
            int i = x + w*y + w*h*c;
            float ux = ((x < w-1) ? (image[i + 1] - image[i]) : 0);
            float uy = ((y < h-1) ? (image[i + w] - image[i]) : 0);
            sh_u[b + B*c + B*nc*0] = ux;
            sh_u[b + B*c + B*nc*1] = uy;
            G2 += ux*ux + uy*uy;
        }

        float g = huber(sqrtf(G2), epsilon);
        for (int c = 0; c < nc; c++) {
            int i = x + w*y + w*h*c;
            Px[i] = g * sh_u[b + B*c + B*nc*0];
            Py[i] = g * sh_u[b + B*c + B*nc*1];
        }
    }
}

__global__ void divergence_and_update(float *image, float *Px, float *Py, int w, int h, int nc, float tau, bool* mask)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x < w && y < h) {
    	if(mask[x + w*y])
    	{
			for (int c = 0; c < nc; c++) {
				int i = x + w*y + w*h*c;
				float dx_u1 = Px[i] - ((x > 0) ? Px[i - 1] : 0);
				float dy_u2 = Py[i] - ((y > 0) ? Py[i - w] : 0);
				image[i] += tau * (dx_u1 + dy_u2);
			}
    	}
    }
}

// __global__ void callKernel(float* imgIn, float* kernel, float* imgOut, int rad, int w, int h, int nc)
// {
//     convolveImage(imgIn, kernel, imgOut, rad, w, h, nc);    
// }

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

    float sigma = 2.0;
    getParam("sigma", sigma, argc, argv);
    cout << "sigma = " << sigma << endl;

    // ### Define your own parameters here as needed
    float epsilon = 0.05f;
    getParam("epsilon", epsilon, argc, argv);
    cout << "ε: " << epsilon << endl;

    float tau = 0.2 / huber(0, epsilon);
    getParam("tau", tau, argc, argv);
    cout << "τ: " << tau << endl;

    int N = 800;
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
    	size_t count = w * h * nc;

        // Thread Dimensions
        dim3 block = dim3(16, 16, 1);
        dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);
        size_t smBytes = (size_t)block.x*block.y*nc*2*sizeof(float);

        // Allocating memory on the device
        float *d_imgIn = NULL;
//        float *d_imgGreen = NULL;
//        float *d_imgOut = NULL;
        bool *d_mask = NULL;
        float *d_Px = NULL;
        float *d_Py = NULL;
        cudaMalloc(&d_imgIn, count * sizeof(float));
//        cudaMalloc(&d_imgOut, count * sizeof(float));
        cudaMalloc(&d_mask, (n_pixels * sizeof(bool) + 7) / 8);
        cudaMalloc(&d_Px, count * sizeof(float));
        cudaMalloc(&d_Py, count * sizeof(float));
        
        cout << "n_pixels = " << n_pixels << "sizeof(bool)" << sizeof(bool) ;

        // Copying Input image to device, and initializing result to 0
        cudaMemcpy(d_imgIn, imgIn, count * sizeof(float), cudaMemcpyHostToDevice);

        // Calling Kernel
        findGreen <<<grid, block>>> (d_imgIn, d_imgIn, d_mask, count, w, h, nc);

        for (int n = 0; n < N; n++) {
            compute_P<<< grid, block, smBytes >>>(d_imgIn, d_Px, d_Py, w, h, nc, epsilon);
            divergence_and_update<<< grid, block >>>(d_imgIn, d_Px, d_Py, w, h, nc, tau, d_mask);
        }
        
        // Copying result back
        cudaMemcpy(imgOut, d_imgIn, count * sizeof(float), cudaMemcpyDeviceToHost);
 
	    CUDA_CHECK;
 
        // Freeing Memory
        cudaFree(d_imgIn);
//        cudaFree(d_imgOut);
        cudaFree(d_mask);
        cudaFree(d_Px);
        cudaFree(d_Py);
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
	            for(int j = -rad; j <= rad; j++)                                     // for each row in kernel
	            {
	                int iny = max(0, min(iy+j, h-1));
	                for(int i = -rad; i <= rad; i++)                                 // for each element in the kernel row
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

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



