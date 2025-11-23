#include "MainWindow.h"
#include "FrameController.h"

#include <QApplication>
#include <QTimer>

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    MainWindow w;
    FrameController* fc = new FrameController(&w, &w);
    QTimer::singleShot(0, [&]() {
        fc->initialize();
    });
    
    w.setWindowTitle("CVS");
    w.show();
    return a.exec();
}