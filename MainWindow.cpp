#include <QApplication>
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QKeyEvent>
#include <QDebug>
#include <QMediaDevices>
#include <QCameraDevice>
#include <QGridLayout>
#include <QGraphicsView>
#include <QAction>
#include <QGraphicsScene>
#include <QMenuBar>
#include <QLabel>
#include <QStatusBar>
#include <QIcon>
#include <QStandardItem>
#include <qfilesystemmodel>
#include <QDir>
#include <QToolButton>
#include <QMutex>
#include "MainWindow.h"
#include "CaptureWorker.h"

MainWindow::MainWindow(QWidget* parent)
: QMainWindow(parent) {
    initUI();
}

MainWindow::~MainWindow() {

}

void MainWindow::initUI() {
    this->setFixedSize(2560, 1440);
    fileMenu_ = menuBar()->addMenu("&Menu");
    createActions();

    // MainWindow 내부의 레이아웃 생성 및 기타 위젯 생성
    QBoxLayout* main_layout = new QHBoxLayout(); // 세로 박스 레이아웃
    QBoxLayout* sub_layout = new QVBoxLayout(); // 중간의 영상영역과 익스플로러 영역을 넣을 가로 박스 레이아웃

    imageLabel_ = new QLabel(this);
    imageLabel_->setScaledContents(false);

    capButton_ = new QToolButton(this);
    capButton_->setText("Start Capture");
    capButton_->setCheckable(true);
    connect(capButton_, &QToolButton::toggled, this,
        [&](bool checked) {
            if (checked) {
                emit startCameraRequest();
                qDebug() << "camera started";
                capButton_->setText("Stop Camera");
            }
            else {
                emit stopCameraRequest();
                qDebug() << "camera stopped";
                capButton_->setText("Start Camera");
            }
        }
    );

    inferButton_ = new QToolButton(this);
    inferButton_->setText("Start Inference");
    inferButton_->setCheckable(true);
    connect(inferButton_, &QToolButton::toggled, this,
        [&](bool checked) {
            if (checked) {
                emit startInferenceRequest();
                qDebug() << "Inference started";
                inferButton_->setText("Stop Inference");
            }
            else {
                emit stopInferenceRequest();
                qDebug() << "Inference stopped";
                inferButton_->setText("Start Inference");
            }
        }
    );

    rightView_ = new QListView();

    main_layout->addWidget(imageLabel_);
    main_layout->addLayout(sub_layout);
    sub_layout->addWidget(capButton_, 0, Qt::AlignHCenter);
    sub_layout->addWidget(inferButton_, 0, Qt::AlignHCenter);
    sub_layout->addWidget(rightView_);
    main_layout->setStretch(0, 4);
    main_layout->setStretch(1, 1);
    sub_layout->setStretch(0, 1);
    sub_layout->setStretch(1, 5);

    // 맨 마지막에 Central Widget 생성 후 여기에 레이아웃을 놓아야 한다.
    QWidget* centralWidget = new QWidget(this);
    centralWidget->setLayout(main_layout);
    setCentralWidget(centralWidget);
}

void MainWindow::createActions() {
    exitAction_ = new QAction("E&xit", this);
    fileMenu_->addAction(exitAction_);

    connect(exitAction_, &QAction::triggered, QApplication::instance(), &QApplication::quit);
}

void MainWindow::updateFrame(cv::Mat mat) {
    dataLock_->lock();
    QImage img(
        mat.data,
        mat.cols,
        mat.rows,
        mat.step,
        QImage::Format_RGB888
    );
    dataLock_->unlock();
    imageLabel_->setPixmap(QPixmap::fromImage(img, Qt::NoFormatConversion));
}