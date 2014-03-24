#include <QApplication>
#include "glwidget.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    GlWidget w;
    w.show();

    return app.exec();
}
