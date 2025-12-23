#include "InferenceController.h"

InferenceController::InferenceController(QObject* parent, float** ml_image)
: QObject(parent) {
	cudaMalloc((void**)&middle_image_, sizeof(float) * 1 * 1024 * 14 * 14);
	createModel(ml_image);
}

InferenceController::~InferenceController() {
	destroyModel();
}

void InferenceController::createModel(float** ml_image_addr) {
	thread_ = new QThread(this);
	thread_post_ = new QThread(this);
	worker_ = new InferenceWorker(nullptr, ml_image_addr, middle_image_);
	worker_post_ = new InferenceWorker_Post(nullptr, middle_image_);
	connect(worker_, &InferenceWorker::backboneReady,worker_post_,&InferenceWorker_Post::run);
	worker_->moveToThread(thread_);
	thread_->start();
	worker_post_->moveToThread(thread_post_);
	thread_post_->start();
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

	if (!worker_post_ || !thread_post_)
		return;
	//worker_->stop();
	worker_post_->deleteLater();
	thread_post_->quit();
	thread_post_->wait();
	thread_post_->deleteLater();
	thread_post_ = nullptr;
	worker_post_ = nullptr;
}