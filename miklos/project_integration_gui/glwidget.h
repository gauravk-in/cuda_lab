#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QGLWidget>
#include <QGLFunctions>
#include "timer.h"

class GlWidget : public QGLWidget
{
    Q_OBJECT

public:
    explicit GlWidget(QWidget *parent = 0);
    ~GlWidget();
    QSize sizeHint() const;
    float *d_in;
    float lambda;
    float sigma;
    float tau;
    int N;
    float c1;
    float c2;
    Timer timer;
    float fps;

protected:
    void initializeGL();
    void paintGL();

private:
    QGLFunctions gl;

signals:
    void fps_updated(float fps);
};

#endif // GLWIDGET_H
