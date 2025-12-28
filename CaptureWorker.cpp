#include "CaptureWorker.h"
#include "MainWindow.h"

CaptureWorker::CaptureWorker(QObject* parent, std::string pipeline, float** ml_image, unsigned char** gui_image, QMutex* lock)
: QObject(parent),
cap_(cv::VideoCapture(pipeline, cv::CAP_GSTREAMER)),
tmp_(),
running_(false) {
    ml_image_addr_ = ml_image;
    gui_image_addr_ = gui_image;
    cudaStreamCreateWithFlags(&preProcessStream, cudaStreamNonBlocking);
    cudaMalloc((void**)&d_capture, sizeof(unsigned char) * 640 * 480 * 2);
    frame_count_ = 0;
    caplock_ = lock;
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
    caplock_->lock();
    if (!running_) {
        cap_.release();
        return;
    }
    caplock_->unlock();

    while (running_) {
        cap_ >> tmp_;
        cudaMemcpy(d_capture, tmp_.data, sizeof(unsigned char) * 640 * 480 * 2, cudaMemcpyHostToDevice);
        launchPREPROCESS(d_capture, *gui_image_addr_, *ml_image_addr_, preProcessStream);
        //cudaStreamSynchronize(preProcessStream);

        emit frameCaptured(frame_count_);
        frame_count_++;
    }
    emit captureFinished();
}