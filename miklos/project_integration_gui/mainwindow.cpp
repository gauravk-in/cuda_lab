#include "mainwindow.h"
#include "camera.h"
#include "ui_mainwindow.h"
#include "kernel.h"

#include <iostream>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

int MainWindow::init()
{
    // Init camera
    if(!camera.init(0)) {
        std::cerr << "ERROR: Could not open camera" << std::endl;
        return 1;
    }

    QObject::connect(&camera, SIGNAL(newFrame()), ui->widget, SLOT(updateGL()));
    allocate_device_memory(ui->widget->d_in, camera.width(), camera.height());

    camera.start();

    return 0;
}
