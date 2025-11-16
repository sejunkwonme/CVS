#include "Capture.h"
#include <QElapsedTimer>
#include <QtConcurrent>
#include <QDebug>
#include "Utilities.h"

using namespace std;

Capture::Capture(int camera, QMutex* lock, QObject* parent)
: running_(false),
inference_(false),
cameraID_(camera),
videoPath(""),
data_lock_(lock),
frame_(),
taking_photo_(false) {

}

Capture::~Capture() {
    
}

/*
void Capture::takePhoto(cv::Mat& frame) {
    QString photo_name = Utilities::newPhotoName();
    QString photo_path = Utilities::getPhotoPath(photo_name, "jpg");
    cv::imwrite(photo_path.toStdString(), frame);
    emit photoTaken(photo_name);
    taking_photo_ = false;
}
*/

void Capture::processFrame() {
    // running 이 false이면 종료
	if (!running_) {
        cap_.release();
        emit capfinished();
        return;
	}

    // 캡처시도후 캡처 안되면 종료
    if (!cap_.read(tmp_) || tmp_.empty()) {
        cap_.release();
        emit capfinished();
        return;
    }

    // 캡처후 중앙_crop
    cv::Rect roi(
        (tmp_.cols - 448) / 2,
        (tmp_.rows - 448) / 2,
        448, 448
    );

    cv::Mat cropped = tmp_(roi).clone();
    cv::Mat rgb;
    cv::cvtColor(cropped, rgb, cv::COLOR_BGR2RGB);

    if (inference_) {
        callInference(rgb);
    }

    {
        QMutexLocker locker(data_lock_);
        frame_ = rgb;
    }

    emit frameCaptured(&frame_);

    // 다음 프레임 재귀적으로 계속 큐에 예약
    QMetaObject::invokeMethod(this, "processFrame", Qt::QueuedConnection);

}

void Capture::start() {
    if (running_) return;
    running_ = true;

    std::string pipeline =
        "mfvideosrc device-index=" + std::to_string(cameraID_) +
        " ! video/x-raw, width=640, height=480, framerate=30/1, auto-focus=1 "
        " ! videoconvert ! appsink";
    cap_ = cv::VideoCapture(pipeline, cv::CAP_GSTREAMER);

    if (!cap_.isOpened()) { // 카메라 열리지 않았을경우 예외처리
        qWarning() << "Camera open failed";
        emit capfinished();
        return;
    }

    // 첫 프레임 요청
    QMetaObject::invokeMethod(this, "processFrame", Qt::QueuedConnection);
}

void Capture::stop() {
    qDebug() << "STOP called in thread:";
    running_ = false;
}

void Capture::startTakePhoto() {
    taking_photo_ = true;
}

void Capture::startInference() {
    inference_ = true;
}

void Capture::stopInference() {
    qDebug() << "stopInference triggered";
    inference_ = false;
    emit inferfinished();
}
