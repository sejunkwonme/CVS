#include "CaptureController.h"

CaptureController::CaptureController(QObject *parent)
: QObject(parent) {
	datalock_ = new QMutex();
}

CaptureController::~CaptureController() {
	
}

void CaptureController::createCamera() {
	thread_ = new QThread(this);
	worker_ = new CaptureWorker(nullptr, datalock_);
	worker_->moveToThread(thread_);
	thread_->start();
	connect(worker_, &CaptureWorker::frameCaptured, this, &CaptureController::frameFromWorker);
}

void CaptureController::destroyCamera() {
	if (!worker_ || !thread_)
		return;
	//worker_->stop();
	thread_->quit();
	thread_->wait();
	worker_->deleteLater();
	thread_->deleteLater();
	thread_ = nullptr;
	worker_ = nullptr;
}

void CaptureController::startCapture() {
	qDebug() << "capture called";
	createCamera();
	QMetaObject::invokeMethod(worker_,"run",Qt::QueuedConnection);
}

void CaptureController::stopCapture() {
	QMetaObject::invokeMethod(worker_, "stop", Qt::QueuedConnection);
	destroyCamera();
}

void CaptureController::frameFromWorker(cv::Mat* data) {
	emit captured(data);
}
