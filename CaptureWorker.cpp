#include "CaptureWorker.h"
#include "MainWindow.h"

CaptureWorker::CaptureWorker(QObject* parent, std::string pipeline, cv::Mat frame, QMutex* lock)
    : QObject(parent),
    cap_(cv::VideoCapture(pipeline, cv::CAP_GSTREAMER)),
    tmp_(),
    running_(false),
    lock_(lock),
    frame_(frame) {
}

CaptureWorker::~CaptureWorker() {
}

void CaptureWorker::run() {
    if (running_) {
        return;
    }

    if (!cap_.isOpened()) {
        qDebug() << "Camera open failed";
        return;
    }

    running_ = true;

    QMetaObject::invokeMethod(this, "captureOneFrame", Qt::QueuedConnection);
}

void CaptureWorker::captureOneFrame() {
    while (running_) {
        cap_ >> tmp_;

        // tmp_ = NV12 raw buffer (height = 1620, width = 1920)

        cv::Mat rgb;
        cv::cvtColor(tmp_, rgb, cv::COLOR_YUV2RGB_NV12);

        lock_->lock();
        rgb.copyTo(frame_);   // frame_ = RGB Mat
        lock_->unlock();

        emit frameCaptured();
    }
    emit captureFinished();
}

void CaptureWorker::stop() {
    running_ = false;
}