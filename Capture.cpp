#include "Capture.h"
#include <QElapsedTimer>
#include <QtConcurrent>
#include <QDebug>
#include "Utilities.h"

using namespace std;

Capture::Capture(int camera, QMutex* lock, QObject* parent)
: running(false),
cameraID_(camera),
videoPath(""),
data_lock_(lock),
frame_(),
taking_photo_(false) {

}

Capture::~Capture() {
    
}

void Capture::captureLoop() {
    cv::VideoCapture cap(cameraID_, cv::CAP_MSMF);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);

    if (!cap.isOpened()) { // 카메라 열리지 않았을경우 예외처리
        qWarning() << "Camera open failed";
        emit capfinished();
        return;
    }

    cv::Mat tmp;

    while (running) {
        cap >> tmp;
        if (tmp.empty()) break;

        //448x448 로 중앙 크롭하기
        int W = tmp.cols;
        int H = tmp.rows;
        int x_start = (W - 448) / 2;
        int y_start = (H - 448) / 2;
        cv::Rect roi(x_start, y_start, 448, 448);
        cv::Mat cropped = tmp(roi);

        // --- YOLOv1 추론 ---
        //detectObjects(cropped);
        emit callInference(&cropped);

        //메인윈도우에 시그널 보내기
        cv::Mat rgb;
        cv::cvtColor(cropped, rgb, cv::COLOR_BGR2RGB);
        data_lock_->lock();
        frame_ = rgb;
        data_lock_->unlock();
        emit frameCaptured(&frame_);
    }

    cap.release();
    running = false;
    emit capfinished();
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

void Capture::start() {
    if (running) return;
    running = true;
    captureLoop();
}

void Capture::stop() {
    running = false;
}

void Capture::startTakePhoto() {
    taking_photo_ = true;
}