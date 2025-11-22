#include "InferenceController.h"

InferenceController::InferenceController(QObject* parent, cv::Mat frame, QMutex* lock)
: QObject(parent),
frame_(frame),
inferLock_(lock),
inferencing_(false) {
	createModel();
}

InferenceController::~InferenceController() {
	destroyModel();
}

void InferenceController::createModel() {
	thread_ = new QThread(this);
	worker_ = new InferenceWorker(nullptr, frame_, inferLock_);
	worker_->moveToThread(thread_);
	thread_->start();
}

void InferenceController::destroyModel() {
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

void InferenceController::startInference() {
	inferencing_ = true;
}

void InferenceController::stopInference() {
	inferencing_ = false;
}

bool InferenceController::state() {
	return inferencing_;
}