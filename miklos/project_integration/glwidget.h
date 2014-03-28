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

protected:
    void initializeGL();
    void paintGL();

private:
    QGLFunctions gl;
    float *d_in;
};

#endif // GLWIDGET_H
