#include "CaptureController.h"

CaptureController::CaptureController(QObject *parent, cv::Mat frame, QMutex* lock)
: QObject(parent),
captureLock_(lock),
frame_(frame),
worker_(nullptr),
thread_(nullptr) {
	pipeline_ =
		"mfvideosrc device-index=0 "
		"! video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 "
		"! appsink drop=true max-buffers=1 sync=false";
}

CaptureController::~CaptureController() {
    if (worker_) {
        worker_->running_ = false;
    }

    if (thread_) {
        thread_->quit();
        thread_->wait();
        delete thread_;
        thread_ = nullptr;
    }

    worker_ = nullptr;
}

void CaptureController::createCamera() {
	thread_ = new QThread(this);
	worker_ = new CaptureWorker(nullptr, pipeline_, frame_, captureLock_);
	worker_->moveToThread(thread_);

	connect(worker_, &CaptureWorker::captureFinished, thread_, &QThread::quit);
	connect(thread_, &QThread::finished, worker_, &QObject::deleteLater);
	connect(thread_, &QThread::finished, thread_, &QObject::deleteLater);
	connect(worker_, &QObject::destroyed, this, [this]() {
		worker_ = nullptr;
		});
	connect(thread_, &QObject::destroyed, this, [this]() {
		thread_ = nullptr;
		});

	thread_->start();
}

void CaptureController::startCapture() {
	createCamera();
	QMetaObject::invokeMethod(worker_,"run",Qt::QueuedConnection);
}

void CaptureController::stopCapture() {
	worker_->running_ = false;
}