#include "CaptureWorker.h"
#include "MainWindow.h"

CaptureWorker::CaptureWorker(QObject* parent, std::string pipeline, cv::Mat frame, QMutex* lock, float** ml_image, unsigned char** gui_image)
: QObject(parent),
cap_(cv::VideoCapture(pipeline, cv::CAP_GSTREAMER)),
tmp_(),
running_(false),
lock_(lock),
frame_(frame) {
    ml_image_addr_ = ml_image;
    gui_image_addr_ = gui_image;
    cudaStreamCreateWithFlags(&preProcessStream, cudaStreamNonBlocking);
    cudaMalloc((void**)&d_capture, sizeof(unsigned char) * 640 * 480 * 2);
    cudaEventCreateWithFlags(&preprocess_done_, cudaEventDisableTiming);
}

CaptureWorker::~CaptureWorker() {
    cudaFree(d_capture);
    cudaStreamDestroy(preProcessStream);
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

    cv::Mat processed(448, 448, CV_8UC3);
    while (running_) {
        cap_ >> tmp_;
        cudaMemcpy(d_capture, tmp_.data, sizeof(unsigned char) * 640 * 480 * 2, cudaMemcpyHostToDevice);
        launchPREPROCESS(d_capture, *gui_image_addr_, *ml_image_addr_, preProcessStream);
        cudaEventRecord(preprocess_done_, preProcessStream);

        emit frameCaptured(reinterpret_cast<quintptr>(preprocess_done_));
    }
    emit captureFinished();
}