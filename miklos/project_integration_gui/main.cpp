//#include "camera.h"
#include "kernel.h"
#include "mainwindow.h"
//#include "glwidget.h"

#include <iostream>
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    MainWindow w;

    w.init();
    w.show();

    return app.exec();
}
