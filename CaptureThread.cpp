#include "CaptureThread.h"
#include <QElapsedTimer>
#include <QtConcurrent>
#include <QDebug>
using namespace std;

#include "Utilities.h"

CaptureThread::CaptureThread(int camera, QMutex* lock) : running(false), cameraID(camera), videoPath(""), data_lock(lock), taking_photo(false) {
}

CaptureThread::CaptureThread(QString videoPath, QMutex* lock) : running(false), cameraID(-1), videoPath(videoPath), data_lock(lock), taking_photo(false) {
}

CaptureThread::~CaptureThread() {
}

void CaptureThread::run() {
    setObjectName("mycapturethread");
    running = true;
    cv::VideoCapture cap(cameraID, cv::CAP_DSHOW);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 30);
    cv::Mat tmp_frame;
    //frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    //frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT)

    classifier = new cv::CascadeClassifier(OPENCV_DATA_DIR "haarcascades/haarcascade_frontalface_default.xml");

    while (running) {
        cap >> tmp_frame;
        if (tmp_frame.empty()) {
            break;
        }

        int W = tmp_frame.cols;
        int H = tmp_frame.rows;

        int x_start = (W - 448) / 2;
        int y_start = (H - 448) / 2;

        cv::Rect roi(x_start, y_start, 448, 448);
        //tmp_frame = tmp_frame(roi);

        detectFaces(tmp_frame);

        if (taking_photo) {
            takePhoto(tmp_frame);
        }

        cvtColor(tmp_frame, tmp_frame, cv::COLOR_BGR2RGB);

        data_lock->lock();
        frame = tmp_frame;
        data_lock->unlock();
        emit frameCaptured(&frame);
    }
    cap.release();
    delete classifier;
    classifier = nullptr;
    running = false;
}

void CaptureThread::takePhoto(cv::Mat& frame) {
    QString photo_name = Utilities::newPhotoName();
    QString photo_path = Utilities::getPhotoPath(photo_name, "jpg");
    cv::imwrite(photo_path.toStdString(), frame);
    emit photoTaken(photo_name);
    taking_photo = false;
}

void CaptureThread::detectFaces(cv::Mat& frame) {
    vector<cv::Rect> faces;
    cv::Mat gray_frame;
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
    classifier->detectMultiScale(gray_frame, faces, 1.3, 5);

    cv::Scalar color = cv::Scalar(0, 0, 255);
    for (size_t i = 0; i < faces.size(); i++) {
        cv::rectangle(frame, faces[i], color, 1);
    }
}
