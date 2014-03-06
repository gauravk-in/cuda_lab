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
// ### Miklos Homolya, miklos.homolya@tum.de, p056 
// ### Ravikishore Kommajosyula, r.kommajosyula, p057
// ###


#include "aux.h"
#include <iostream>
#include <math.h>
using namespace std;

// uncomment to use the camera
//#define CAMERA

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

__device__ void compute_tensor( float* m11, float* m12, float* m22, float* tensor, int w, int h, int nc) {


}

__device__ void compute_matrices(float* der_x, float* der_y, float* m11, float* m12, float* m22,
								int w, int h, int nc) {
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	int iz = threadIdx.z + blockDim.z * blockIdx.z;



	// Only the first nc (ex. Red) slice 2D id. Make simultaneous
	// updates on this from all threads over different nc.
	size_t idx_2d = ix + (iy * w);

	if (ix < w && iy < h && iz ==0) {
		// store global memory accesses in temporary variables
		float temp_m11 = 0.0f, temp_m22 = 0.0f, temp_m12 = 0.0f;
		for(int i = 0; i < nc; i++)
		{
			size_t idx = ix + (iy * w) + (iz * w * h);
			float temp_dx, temp_dy;
			temp_dx = der_x[idx];
			temp_dy = der_y[idx];

			// add contribution of this 'nc' to the M matrix components
			temp_m11 += temp_dx * temp_dx;
			temp_m12 += temp_dx * temp_dy;
			temp_m22 += temp_dy * temp_dy;
		}
		m11[idx_2d] = temp_m11;
		m12[idx_2d] = temp_m12;
		m22[idx_2d] = temp_m22;
	}
}


__device__ void convolveImage(float *in, float *kern, float *out,
		 	 	 	 	 	  int r, int w, int h, int nc) {
    int ksize = 2*r + 1;
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int c = threadIdx.z + blockDim.z * blockIdx.z;
    if (x < w && y < h && c < nc) {
        float value = 0.0f;
        for (int ky = 0; ky < ksize; ky++) {
            int cy = gpu_max(0, gpu_min(y + ky - r, h-1));
            for (int kx = 0; kx < ksize; kx++) {
                int cx = gpu_max(0, gpu_min(x + kx - r, w-1));
                value += kern[kx + ksize*ky] * in[cx + w*cy + w*h*c];
            }
        }
        out[x + w*y + w*h*c] = value;
    }
}

__global__ void callKernel(float* imgIn, float* kernel, float* imgOut, float* v1, float* v2,
		float* m11, float* m12, float* m22, float* mm11, float* mm12, float* mm22,
		float *diffx_kernel, float *diffy_kernel, int rad, int w, int h, int nc)
{
	convolveImage(imgIn, kernel, imgOut, rad, w, h, nc);
	// compute x derivatives using convolution
	convolveImage(imgOut, diffx_kernel, v1, 1, w, h, nc );
	convolveImage(imgOut, diffy_kernel, v2, 1, w, h, nc );

	// compute the matrices m11, m12 and m22
	compute_matrices(v1, v2, mm11, mm12, mm22, w, h, nc);
	convolveImage(mm11, kernel, m11, rad, w, h, 1 );
	convolveImage(mm12, kernel, m12, rad, w, h, 1 );
	convolveImage(mm22, kernel, m22, rad, w, h, 1 );

	// convolve the matrices m11, m12 and m22

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

	float sigma = 2.0;
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

	// allocate arrays for the M matrix
	float *m11 = new float[(size_t)w*h];
	float *m12 = new float[(size_t)w*h];
	float *m22 = new float[(size_t)w*h];

	int rad = ceil(3 * sigma); // kernel radius
	int kw = 2 * rad + 1; // kernel width
	float c = 1. / (2. * 3.142857 * sigma * sigma); // constant

	cout << "c = " << c << endl;

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

	// Make a new differentiation kernel
	float diffx_kernel[] = { -3, 0, 3, -10, 0, 10, -3, 0, 3 };
	float diffy_kernel[] = { -3, -10, -3, 0, 0, 0, 3, 10, 3 };

	// Display Kernel
	cv::Mat cvKernelOut(kw, kw, CV_32FC1);
	convert_layered_to_mat(cvKernelOut, kernelOut);
	showImage("Kernel", cvKernelOut, 100+2*w+80, 100);

	// add images for m11, m12, m13
	cv::Mat img_m11(h,w,CV_32FC1);    // grayscale
	cv::Mat img_m12(h,w,CV_32FC1);
	cv::Mat img_m22(h,w,CV_32FC1);

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
		timer.start();

		// Repetitions Loop
		for(int rep = 0; rep < repeats; rep++)
		{
			size_t count = (size_t)w * h * nc;

			// Thread Dimensions
			dim3 block = dim3(32, 8, nc);
			dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);

			// Allocating memory on the device
			float *d_imgIn = NULL;
			float *d_imgOut = NULL;
			float *d_v1 = NULL;
			float *d_v2 = NULL;
			float *d_mm11 = NULL;
			float *d_mm12 = NULL;
			float *d_mm22 = NULL;
			float *d_m11 = NULL;
			float *d_m12 = NULL;
			float *d_m22 = NULL;
			float *d_kernel = NULL;
			float *d_diffx_kernel = NULL;
			float *d_diffy_kernel = NULL;
			float *d_tensor = NULL;
			cudaMalloc(&d_imgIn, count * sizeof(float));
			cudaMalloc(&d_imgOut, count * sizeof(float));
			cudaMalloc(&d_kernel, kw * kw * sizeof(float));
			cudaMalloc(&d_diffx_kernel, 3 * 3 * sizeof(float));
			cudaMalloc(&d_diffy_kernel, 3 * 3 * sizeof(float));
			cudaMalloc(&d_v1, count * sizeof(float));
			cudaMalloc(&d_v2, count * sizeof(float));
			cudaMalloc(&d_mm11, w*h * sizeof(float));
			cudaMalloc(&d_mm12, w*h * sizeof(float));
			cudaMalloc(&d_mm22, w*h * sizeof(float));
			cudaMalloc(&d_m11, w*h * sizeof(float));
			cudaMalloc(&d_m12, w*h * sizeof(float));
			cudaMalloc(&d_m22, w*h * sizeof(float));
			cudaMalloc(&d_tensor, w*h*4 * sizeof(float)); // 4 matrix elements of w*h

			// Copying Input image to device, and initializing result to 0
			cudaMemcpy(d_imgIn, imgIn, count * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_kernel, kernel, kw * kw * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_diffx_kernel, diffx_kernel, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_diffy_kernel, diffy_kernel, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

			// Calling gaussian smoothing kernel
			callKernel <<< grid, block >>> (d_imgIn, d_kernel, d_imgOut, d_v1, d_v2,
					d_m11, d_m12, d_m22, d_mm11, d_mm12, d_mm22,
					d_diffx_kernel, d_diffy_kernel, rad, w, h, nc);

			// Copying result back
			cudaMemcpy(m11, d_m11, w * h * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(m12, d_m12, w * h * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(m22, d_m22, w * h * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(imgOut, d_v1, count * sizeof(float), cudaMemcpyDeviceToHost);

			CUDA_CHECK;

			// Freeing Memory
			cudaFree(d_imgIn);
			cudaFree(d_kernel);
			cudaFree(d_diffx_kernel);
			cudaFree(d_diffy_kernel);
			cudaFree(d_imgOut);
			cudaFree(d_v1);
			cudaFree(d_v2);
			cudaFree(d_m11);
			cudaFree(d_m12);
			cudaFree(d_m22);
			cudaFree(d_mm11);
			cudaFree(d_mm12);
			cudaFree(d_mm22);

		}

		timer.end();
		t = timer.get();

		cout << "time: " << t*1000 << " ms" << endl;

		// show input image
		showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

		// show output image: first convert to interleaved opencv format from the layered raw array
		convert_layered_to_mat(mOut, imgOut);
		// showImage("Output", mOut, 100+w+40, 100);

		// ### Display your own output images here as needed
		int factor = 40;
		convert_layered_to_mat(img_m11, m11);
		img_m11 *= factor;
		showImage("M11", img_m11, 100, 400 );
		convert_layered_to_mat(img_m12, m12);
		img_m12 *= factor;
		showImage("M12", img_m12, 100 + w + 40, 400 );
		convert_layered_to_mat(img_m22, m22);
		img_m22 *= factor;
		showImage("M22", img_m22, 100 + w + w + 80 , 400 );


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
	delete[] m11;
	delete[] m12;
	delete[] m22;

	// close all opencv windows
	cvDestroyAllWindows();
	return 0;
}


