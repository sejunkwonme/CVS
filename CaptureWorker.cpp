#include "CaptureWorker.h"
#include "MainWindow.h"

CaptureWorker::CaptureWorker(QObject* parent, std::string pipeline, cv::Mat frame, QMutex* lock)
: QObject(parent),
cap_(cv::VideoCapture(pipeline, cv::CAP_GSTREAMER)),
tmp_(),
running_(false),
lock_(lock),
frame_(frame) {
    cudaStreamCreateWithFlags(&preProcessStream, cudaStreamNonBlocking);
    cudaMalloc((void**)&d_capture, sizeof(unsigned char) * 640 * 480 * 2);
    cudaMalloc((void**)&d_gui_image_BGR, sizeof(unsigned char) * 640 * 480 * 3);
    cudaMalloc((void**)&d_ml_image_RGB, sizeof(float) * 640 * 480 * 3);
    cudaMalloc((void**)&d_gui_image_BGR_cropped, sizeof(unsigned char) * 448 * 448 * 3);
    cudaMalloc((void**)&d_ml_image_RGB_cropped, sizeof(float) * 448 * 448 * 3);
}

CaptureWorker::~CaptureWorker() {
    cudaFree(d_capture);
    cudaFree(d_gui_image_BGR);
    cudaFree(d_ml_image_RGB);
    cudaFree(d_gui_image_BGR_cropped);
    cudaFree(d_ml_image_RGB_cropped);
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

    cv::Mat processed(448,448,CV_8UC3);
    while (running_) {
        cap_ >> tmp_;
        cudaMemcpy(d_capture, tmp_.data, sizeof(unsigned char) * 640 * 480 * 2, cudaMemcpyHostToDevice);
        launchYUV2RGB(d_capture, d_gui_image_BGR, d_ml_image_RGB, preProcessStream);
        launchCROP(d_gui_image_BGR, d_ml_image_RGB, d_gui_image_BGR_cropped, d_ml_image_RGB_cropped, preProcessStream);
        cudaStreamSynchronize(preProcessStream);
        cudaMemcpy(processed.data, d_gui_image_BGR_cropped, sizeof(unsigned char) * 448 * 448 * 3, cudaMemcpyDeviceToHost);

        lock_->lock();
        processed.copyTo(frame_);
        lock_->unlock();

        emit frameCaptured(d_ml_image_RGB_cropped);
    }
    emit captureFinished();
}