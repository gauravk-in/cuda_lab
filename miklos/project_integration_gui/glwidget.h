#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QGLWidget>
#include <QGLFunctions>

class GlWidget : public QGLWidget
{
    Q_OBJECT

public:
    explicit GlWidget(QWidget *parent = 0);
    ~GlWidget();
    QSize sizeHint() const;
    float *d_in;

protected:
    void initializeGL();
    void paintGL();

private:
    QGLFunctions gl;
};

#endif // GLWIDGET_H