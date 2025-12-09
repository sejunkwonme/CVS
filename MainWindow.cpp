#include <QApplication>
#include <QPixmap>
#include <QDebug>
#include <QGridLayout>
#include <QAction>
#include <QMenuBar>
#include <QToolButton>
#include <QMutex>
#include "MainWindow.h"

MainWindow::MainWindow(QWidget* parent)
: QMainWindow(parent) {
    initUI();
}

MainWindow::~MainWindow() {

}

void MainWindow::initUI() {
    this->setFixedSize(1024, 768);
    fileMenu_ = menuBar()->addMenu("&Menu");
    createActions();

    QBoxLayout* main_layout = new QHBoxLayout();
    QBoxLayout* sub_layout = new QVBoxLayout();

    imageLabel_ = new QLabel(this);

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
    main_layout->addWidget(imageLabel_, 0, Qt::AlignCenter);
    main_layout->addLayout(sub_layout);
    sub_layout->addWidget(capButton_, 0, Qt::AlignHCenter);
    sub_layout->addWidget(inferButton_, 0, Qt::AlignHCenter);
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
    exitAction_ = new QAction("E&xit", this);
    fileMenu_->addAction(exitAction_);

    // connect the signals and slots
    connect(exitAction_, &QAction::triggered, QApplication::instance(), &QApplication::quit);
}

void MainWindow::updateFrame(cv::Mat mat) {
    dataLock_->lock();
    currentFrame_ = mat.clone();
    dataLock_->unlock();

    QImage frame(
        currentFrame_.data,
        currentFrame_.cols,
        currentFrame_.rows,
        currentFrame_.step,
        QImage::Format_RGB888);

    QPixmap pixmap = QPixmap::fromImage(frame);
    imageLabel_->setPixmap(pixmap);
}