#include "CaptureWorker.h"
#include "MainWindow.h"

CaptureWorker::CaptureWorker(QObject *parent, QMutex* lock)
: QObject(parent), 
running_(false),
datalock_(lock) {
    
}

CaptureWorker::~CaptureWorker() {
	
}

void CaptureWorker::run() {
    if (running_) return;
    running_ = true;

    std::string pipeline =
        "mfvideosrc device-index=" + std::to_string(0) +
        " ! video/x-raw, width=640, height=480, framerate=30/1, auto-focus=1 "
        " ! videoconvert ! appsink";
    cap_ = cv::VideoCapture(pipeline, cv::CAP_GSTREAMER);

    if (!cap_.isOpened()) {
        qDebug() << "Camera open failed";
        return;
    }

    QMetaObject::invokeMethod(this, "processFrame", Qt::QueuedConnection);
}

void CaptureWorker::processFrame() {
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

    /*
    if (inference_) {
        callInference(cropped);
    }
	*/
    cv::Mat rgb;
    cv::cvtColor(cropped, rgb, cv::COLOR_BGR2RGB);


    {
        QMutexLocker locker(datalock_);
        frame_ = rgb;
    }

    emit frameCaptured(&frame_);

    QMetaObject::invokeMethod(this, "processFrame", Qt::QueuedConnection);

}

void CaptureWorker::stop() {
    running_ = false;
}