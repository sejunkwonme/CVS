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
    this->setFixedSize(1920, 1080);
    fileMenu_ = menuBar()->addMenu("&Menu");
    createActions();

    // MainWindow 내부의 레이아웃 생성 및 기타 위젯 생성
    QBoxLayout* main_layout = new QHBoxLayout(); // 세로 박스 레이아웃
    QBoxLayout* sub_layout = new QVBoxLayout(); // 중간의 영상영역과 익스플로러 영역을 넣을 가로 박스 레이아웃

    imageScene_ = new QGraphicsScene(this);
    imageView_ = new QGraphicsView(imageScene_);

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

    main_layout->addWidget(imageView_);
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
    // 맨 위 메뉴바의 액션을 생성하여 시그널 슬롯 연결
    cameraInfoAction_ = new QAction("Camera &Information", this);
    fileMenu_->addAction(cameraInfoAction_);
    exitAction_ = new QAction("E&xit", this);
    fileMenu_->addAction(exitAction_);

    // connect the signals and slots
    connect(exitAction_, &QAction::triggered, QApplication::instance(), &QApplication::quit);
    connect(cameraInfoAction_, &QAction::triggered, this, &MainWindow::showCameraInfo);
}

void MainWindow::showCameraInfo() {
    QList<QCameraDevice> cameras = QMediaDevices::videoInputs();
    QString info = QString("Available Cameras: \n");

    foreach(const QCameraDevice &cameraInfo, cameras) {
        info += " - " + cameraInfo.id() + ": ";
        info += cameraInfo.description() + "\n";
    }

    QMessageBox::information(this, "Cameras", info);
}

void MainWindow::updateFrame(cv::Mat mat) {
    dataLock_->lock();
    currentFrame_ = mat;
    dataLock_->unlock();

    QImage frame(
        currentFrame_.data,
        currentFrame_.cols,
        currentFrame_.rows,
        currentFrame_.step,
        QImage::Format_RGB888);
    QPixmap image = QPixmap::fromImage(frame);

    imageScene_->clear();
    imageView_->resetTransform();
    imageScene_->addPixmap(image);
    imageScene_->update();
    imageView_->setSceneRect(image.rect());
}