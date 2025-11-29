#include "MainWindow.h"
#include "FrameController.h"
#include "ConfigController.h"
#include "CalibController.h"

#include <QApplication>
#include <QTimer>

int main(int argc, char* argv[]) {
    QApplication a(argc, argv);
    MainWindow w;

    FrameController* fc = new FrameController(&w, &w);
    ConfigController* cc = new ConfigController(&w);
    CalibController* calib = new CalibController(&w);

    QObject::connect(cc, &ConfigController::initialized, 
        fc, &FrameController::initialize);

    QTimer::singleShot(0, [&]() {
        cc->initialize();
        });

    calib->PDFChain();
    calib->calibrate();

    w.setWindowTitle("CVS");
    w.show();
    return a.exec();
}
