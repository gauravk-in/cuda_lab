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

    QObject::connect(ui->widget, SIGNAL(fps_updated(float)), this, SLOT(update_fps(float)));

    camera.start();

    return 0;
}

void MainWindow::on_spinBox_valueChanged(int arg1)
{
    ui->widget->N = arg1;
}

void MainWindow::on_doubleSpinBox_3_valueChanged(double arg1)
{
    ui->widget->lambda = arg1;
}

void MainWindow::on_doubleSpinBox_4_valueChanged(double arg1)
{
    ui->widget->c2 = arg1;
}

void MainWindow::on_doubleSpinBox_5_valueChanged(double arg1)
{
    ui->widget->c1 = arg1;
}

void MainWindow::update_fps(float fps)
{
    ui->plainTextEdit->appendPlainText(QString::number((double)fps));
}

