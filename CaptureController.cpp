#include "CaptureController.h"

CaptureController::CaptureController(QObject *parent, float** ml_image, unsigned char** gui_image, QMutex* lock)
: QObject(parent),
captureLock_(lock){
	pipeline_ =
		"mfvideosrc device-index=0 "
		"! video/x-raw,format=YUY2,width=640,height=480,framerate=30/1"
		"! appsink drop=true max-buffers=1 sync=false";
	ml_image_ = ml_image;
	gui_image_ = gui_image;
	createCamera();
}

CaptureController::~CaptureController() {
}

void CaptureController::createCamera() {
	thread_ = new QThread(this);
	worker_ = new CaptureWorker(nullptr, pipeline_, ml_image_, gui_image_, captureLock_);

	//connect(worker_, &CaptureWorker::captureFinished, this, &CaptureController::deleteAfterWait);

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
	QMetaObject::invokeMethod(worker_,"run",Qt::QueuedConnection);
}

void CaptureController::stopCapture() {
	captureLock_->lock();
	worker_->running_ = false;
	captureLock_->unlock();
}

void CaptureController::deleteAfterWait() {
	destroyCamera();
}