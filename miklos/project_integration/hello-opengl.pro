QT += core gui opengl
CONFIG += debug

TARGET = hello-opengl

TEMPLATE = app

SOURCES += main.cpp glwidget.cpp
HEADERS += glwidget.h kernel.h

CUDA_SOURCES += kernel.cu

LIBS += -lGLEW -lcudart -lcuda -lopencv_core -lopencv_highgui -lopencv_imgproc

QMAKE_LFLAGS += -Wl,-rpath,/usr/lib/nvidia-331-updates -Wl,-rpath,/usr/lib/nvidia-331-updates/tls
# Path to cuda toolkit install
CUDA_DIR      = /usr

# Path to header and libs files
INCLUDEPATH  += $$CUDA_DIR/include
QMAKE_LIBDIR += $$CUDA_DIR/lib64
QMAKE_LIBDIR += /usr/lib/nvidia-331-updates /usr/lib/nvidia-331-updates/tls
# GPU architecture
CUDA_ARCH     = sm_10
NVCCFLAGS     = --compiler-options -use_fast_math --ptxas-options=-v

CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')
 
cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -O3 -arch=$$CUDA_ARCH -c $$NVCCFLAGS $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS ${QMAKE_FILE_NAME}
cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o

# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda
