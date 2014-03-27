#include <iostream>

#include <QApplication>

#include <opencv2/highgui/highgui.hpp>

#include "glwidget.h"

cv::VideoCapture camera;

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    // Init camera
  	camera.open(0);
  	if(!camera.isOpened()) {
        std::cerr << "ERROR: Could not open camera" << std::endl;
        return 1;
    }
    int camW = 640;
    int camH = 480;
  	camera.set(CV_CAP_PROP_FRAME_WIDTH,camW);
  	camera.set(CV_CAP_PROP_FRAME_HEIGHT,camH);
    
    GlWidget w;
    w.show();

    return app.exec();
}
