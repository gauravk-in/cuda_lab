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

__device__ __host__ float huber(float s, float epsilon)
{
    return 1.0F / max(epsilon, s);
    //return 1.0F;
    //return expf(-s*s / epsilon) / epsilon;
}

// This function finds green colored pixels in the image
// It outputs a bool array mask, which tells if pixel(x, y) is green
// It also sets the green pixel value to (0.5, 0.5, 0.5)
__global__ void findGreen(float* imgIn, bool* d_mask, size_t n_pixels, int w, int h, int nc)
{
    size_t ix = threadIdx.x + blockDim.x * blockIdx.x;
    size_t iy = threadIdx.y + blockDim.y * blockIdx.y;

    if(ix < w && iy < h && nc == 3)
    {
    	// Only the green Layer
    	size_t b_idx = ix + (size_t)(iy * w);
    	size_t g_idx = b_idx + (size_t)(w * h);
    	size_t r_idx = g_idx + (size_t)(w * h);

    	if(imgIn[g_idx] == 1.0f && imgIn[b_idx] == 0.0f && imgIn[r_idx] == 0.0f)
    	{
    		d_mask[b_idx] = true;
    		imgIn[b_idx] = 0.5f;							// Blue
    		imgIn[g_idx] = 0.5f;							// Green
    		imgIn[r_idx] = 0.5f;							// Red
    	}
    	else
    	{
    		d_mask[b_idx] = false;
    	}
    }
}

__global__ void compute_g(float *image, float *g, int w, int h, int nc, float epsilon)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < w && y < h) {

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

__global__ void rb_gs_update(float *image, float *g, bool* mask, int w, int h, int nc, float theta, int red_black)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if( ((x + y) % 2) != red_black) {
    	return;
    }

    if ( (x < w) && (y < h) && (mask[x + w*y])) {
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
            float gs_result = gsum_u / gsum;

            // SOR step
            image[idx] = gs_result + theta * ( gs_result - temp_uxy );
        }
    }
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

    // ### Define your own parameters here as needed
    float sigma = 0.1;
    getParam("sigma", sigma, argc, argv);
    cout << "σ: " << sigma << endl;

    float theta = 0.7;
    getParam("theta", theta, argc, argv);
    cout << "theta: " << theta << endl;

    float epsilon = 0.01;
    getParam("epsilon", epsilon, argc, argv);
    cout << "ε: " << epsilon << endl;

    int N = 150;
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
        bool *d_mask = NULL;
        float *d_g = NULL;
        cudaMalloc(&d_imgIn, count * sizeof(float));
        cudaMalloc(&d_mask, (n_pixels * sizeof(bool) + 7) / 8);
        cudaMalloc(&d_g, n_pixels * sizeof(float));
        
        cout << "n_pixels = " << n_pixels << "sizeof(bool)" << sizeof(bool) ;

        // Copying Input image to device, and initializing result to 0
        cudaMemcpy(d_imgIn, imgIn, count * sizeof(float), cudaMemcpyHostToDevice);

        // Calling Kernel
        findGreen <<<grid, block>>> (d_imgIn, d_mask, count, w, h, nc);

        for (int n = 0; n < N; n++) {
            compute_g<<< grid, block, smBytes >>>(d_imgIn, d_g, w, h, nc, epsilon);
            rb_gs_update<<< grid, block >>>(d_imgIn, d_g, d_mask, w, h, nc, theta, 0);
            rb_gs_update<<< grid, block >>>(d_imgIn, d_g, d_mask, w, h, nc, theta, 1);
        }
        
        // Copying result back
        cudaMemcpy(imgOut, d_imgIn, count * sizeof(float), cudaMemcpyDeviceToHost);
 
	    CUDA_CHECK;
 
        // Freeing Memory
        cudaFree(d_imgIn);
        cudaFree(d_mask);
        cudaFree(d_g);
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



