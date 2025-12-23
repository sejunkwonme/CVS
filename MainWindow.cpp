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

    QBoxLayout* main_layout = new QHBoxLayout();
    QBoxLayout* sub_layout = new QVBoxLayout();

    imageLabel_ = new QLabel(this);

    startButton_ = new QToolButton(this);
    startButton_->setText("Start cap & inference");
    startButton_->setCheckable(true);
    connect(startButton_, &QToolButton::toggled, this,
        [&](bool checked) {
            if (checked) {
                emit startCapandInfer();
                qDebug() << "camera started";
                startButton_->setText("Stop Cap and Infer");
            }
            else {
                emit stopCapandInfer();
                qDebug() << "camera stopped";
                startButton_->setText("Start Cap and Infer");
            }
        }
    );
    main_layout->addWidget(imageLabel_, 0, Qt::AlignCenter);
    main_layout->addLayout(sub_layout);
    sub_layout->addWidget(startButton_, 0, Qt::AlignHCenter);
    main_layout->setStretch(0, 4);
    main_layout->setStretch(1, 1);
    sub_layout->setStretch(0, 1);
    sub_layout->setStretch(1, 5);

    // 맨 마지막에 Central Widget 생성 후 여기에 레이아웃을 놓아야 한다.
    QWidget* centralWidget = new QWidget(this);
    centralWidget->setLayout(main_layout);
    setCentralWidget(centralWidget);
}

void MainWindow::refreshFrame(cv::Mat frame) {
    //dataLock_->lock();
    frame.copyTo(currentFrame_);
    //dataLock_->unlock();

    if (!fpsTimer_.isValid()) { fpsTimer_.start(); prevNs_ = fpsTimer_.nsecsElapsed(); }
    else {
        qint64 nowNs = fpsTimer_.nsecsElapsed();
        qint64 dtNs = nowNs - prevNs_;
        prevNs_ = nowNs;
        if (dtNs > 0) qDebug() << "[updateFrame] FPS =" << (1e9 / double(dtNs));
    }

    QImage frame_qimage(
        currentFrame_.data,
        currentFrame_.cols,
        currentFrame_.rows,
        currentFrame_.step,
        QImage::Format_BGR888);

    QPixmap pixmap = QPixmap::fromImage(frame_qimage);
    imageLabel_->setPixmap(pixmap);
}