#include "camera.h"
#include "glwidget.h"

#include <iostream>
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    GlWidget w;

    // Init camera
    if(!camera.init(0)) {
        std::cerr << "ERROR: Could not open camera" << std::endl;
        return 1;
    }

    QObject::connect(&camera, SIGNAL(newFrame()), &w, SLOT(updateGL()));

    camera.start();
    w.show();

    return app.exec();
}
