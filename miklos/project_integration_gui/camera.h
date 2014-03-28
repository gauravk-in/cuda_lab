#ifndef CAMERA_H
#define CAMERA_H

#include <QMutex>
#include <QThread>
#include <QWaitCondition>

#include <opencv2/highgui/highgui.hpp>

class Camera : public QThread {
    Q_OBJECT

public:
    bool init(int device);
    
    size_t width() const { return width_; }
    size_t height() const { return height_; }

    float *data() const {
        return reinterpret_cast<float *>(frame_.data);
    }

    QMutex mutex;
    QWaitCondition frameCopied;

signals:
    void newFrame();

protected:
    void run();

private:
    cv::VideoCapture capture_;
    cv::Mat frame_;
    size_t width_, height_;
    
    void capture();
};

extern Camera camera;

#endif // CAMERA_H
