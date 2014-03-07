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

__host__ __device__ float absolute_value ( float2 z ) {
	return sqrtf((z.x * z.x) + (z.y * z.y));
}

__host__ __device__ float2 add_complex ( float2 z1, float2 z2 ) {
	return {z1.x + z2.x, z1.y + z2.y };
}

__host__ __device__ float2 square_complex ( float2 z ) {
	return {((z.x*z.x) - (z.y*z.y)), (2.0f * z.x * z.y) };
}


__global__ void callKernel(float* imgOut, int width, int height, float2 center, float radius, int iterations) {
	  int iy = blockIdx.y * blockDim.y + threadIdx.y;  // WIDTH
	  int ix = blockIdx.x * blockDim.x + threadIdx.x;  // HEIGHT
	  int idx = iy * width + ix;
	  if(ix >= width || iy >= height) return;

	  float2 c, z;
	  c.x = ((float)ix / width) * (2.0f * radius) + center.x - radius;
	  c.y = ((float)iy / height) * (2.0f * radius) + center.y - radius;
	  z = c;
	  int n = 0;
	  while( (absolute_value(z) < 2.0f) && (n < iterations))
	  {
		  z = add_complex ( square_complex(z), c);
		  n++;
	  }

	  imgOut[idx] = 1 - (1.0f * n)/iterations;
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

	// ### Define your own parameters here as needed
	float width = 640;
	getParam("width", width, argc, argv);
	cout << "width = " << width << endl;

	float height = 480;
	getParam("height", height, argc, argv);
	cout << "height = " << height<< endl;

	float2 center = {-0.5f, 0.0f};
//	getParam("center", center, argc, argv);
//	cout << "center = " << center.x << ", " << center.y << endl;

	float radius = 1.5f;
	getParam("radius", radius, argc, argv);
	cout << "radius = " << radius << endl;

	int iterations = 100;
	getParam("iterations", iterations, argc, argv);
	cout << "iterations = " << iterations << endl;

	int repeats = 100;
	getParam("repeats", repeats, argc, argv);
	cout << "repeats = " << repeats << endl;

	// Set the output image format
	// ###
	cv::Mat mOut(height, width, CV_32FC1);    // mOut will be a grayscale image, 1 layer

	// Allocate arrays
	// input/output image width: w
	// input/output image height: h
	// input image number of channels: nc
	// output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

	// allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
	float *imgOut = new float[(size_t) (width*height) ];

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
			size_t count = (size_t)width * height;

			// Thread Dimensions
			dim3 block = dim3(32, 8, 1);
			dim3 grid = dim3((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);

			// Allocating memory on the device
			float *d_imgOut = NULL;
			cudaMalloc(&d_imgOut, count * sizeof(float));

			// Calling gaussian smoothing kernel
			callKernel <<< grid, block >>> (d_imgOut, width, height, center, radius, iterations );

			// Copying result back
			cudaMemcpy(imgOut, d_imgOut, count * sizeof(float), cudaMemcpyDeviceToHost);

			CUDA_CHECK;

			// Freeing Memory
			cudaFree(d_imgOut);
		}

		timer.end();
		t = timer.get();

		cout << "time: " << t*1000 << " ms" << endl;

		// show output image: first convert to interleaved opencv format from the layered raw array
		convert_layered_to_mat(mOut, imgOut);
		showImage("Output", mOut, 100, 100);

		// ### Display your own output images here as needed

	// wait for key inputs
	cv::waitKey(0);
	// save input and result
	cv::imwrite("image_result.png",mOut*255.f);

	// free allocated arrays
	delete[] imgOut;

	// close all opencv windows
	cvDestroyAllWindows();
	return 0;
}


