main: main.cu aux.cu aux.h Makefile
	nvcc -o main main.cu aux.cu --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core -lopencv_imgproc -lGL `pkg-config --cflags glfw3` `pkg-config --static --libs glfw3`

