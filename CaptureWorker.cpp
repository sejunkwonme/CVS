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
    if (!running_) {
        cap_.release();
        return;
    }

    if (!cap_.read(tmp_) || tmp_.empty()) {
        running_ = false;
    	cap_.release();
        return;
    }

    cv::Rect roi(
        (tmp_.cols - 448) / 2,
        (tmp_.rows - 448) / 2,
        448, 448
    );

    cv::Mat cropped = tmp_(roi).clone();
    cv::Mat rgb;
    cv::cvtColor(cropped, rgb, cv::COLOR_BGR2RGB);

	lock_->lock();
    rgb.copyTo(frame_);
	lock_->unlock();

    emit frameCaptured();

    QMetaObject::invokeMethod(this, "captureOneFrame", Qt::QueuedConnection);
}

void CaptureWorker::stop() {
    running_ = false;
}