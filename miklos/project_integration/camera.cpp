#include "camera.h"

#include <opencv2/imgproc/imgproc.hpp>

Camera camera;

bool Camera::init(int device)
{
    capture_.open(device);
    if (!capture_.isOpened())
        return false;

    int camW = 640;
    int camH = 480;
  	capture_.set(CV_CAP_PROP_FRAME_WIDTH, camW);
  	capture_.set(CV_CAP_PROP_FRAME_HEIGHT, camH);

    capture();
    width_  = frame_.cols;
    height_ = frame_.rows;

    return true;
}

void Camera::run()
{
    QMutexLocker locker(&mutex);
    while (true) {  // TODO: never terminates
        frameCopied.wait(&mutex);
        capture();
        emit newFrame();
    }
}

void Camera::capture()
{
    capture_ >> frame_;
    cvtColor(frame_, frame_, CV_BGR2GRAY);

    frame_.convertTo(frame_, CV_32F);
    frame_ /= 255.f;
}
