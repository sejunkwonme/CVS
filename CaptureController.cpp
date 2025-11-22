#include "CaptureController.h"

CaptureController::CaptureController(QObject *parent, cv::Mat frame, QMutex* lock)
: QObject(parent),
captureLock_(lock),
frame_(frame) {
	pipeline_ =
		"mfvideosrc device-index=" + std::to_string(0) +
		" ! video/x-raw, width=640, height=480, framerate=30/1, auto-focus=1 "
		" ! videoconvert ! appsink";
}

CaptureController::~CaptureController() {
}

void CaptureController::createCamera() {
	thread_ = new QThread(this);
	worker_ = new CaptureWorker(nullptr, pipeline_, frame_, captureLock_);
	worker_->moveToThread(thread_);
	thread_->start();
}

void CaptureController::destroyCamera() {
	if (!worker_ || !thread_)
		return;
	//worker_->stop();
	worker_->deleteLater();
	thread_->quit();
	thread_->wait();
	thread_->deleteLater();
	thread_ = nullptr;
	worker_ = nullptr;
}

void CaptureController::startCapture() {
	createCamera();
	QMetaObject::invokeMethod(worker_,"run",Qt::QueuedConnection);
}

void CaptureController::stopCapture() {
	QMetaObject::invokeMethod(worker_, "stop", Qt::QueuedConnection);
	destroyCamera();
}