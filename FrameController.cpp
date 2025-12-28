#include "FrameController.h"


FrameController::FrameController(QObject* parent, MainWindow* mainW)
: QObject(parent),
lock_(new QMutex()),
mainW_(mainW) {
	frame_[0] = cv::Mat(448, 448, CV_8UC3);
	frame_[1] = cv::Mat(448, 448, CV_8UC3);
	frame_[2] = cv::Mat(448, 448, CV_8UC3);
	frame_[3] = cv::Mat(448, 448, CV_8UC3);
	cudaMalloc((void**)&ml_image_, sizeof(float) * 1 * 3 * 448 * 448);
	cudaMalloc((void**)&gui_image_, sizeof(unsigned char) * 1 * 3 * 448 * 448);
}

FrameController::~FrameController() {
	destroyWorker();
	cudaFree(ml_image_);
	cudaFree(gui_image_);
}

void FrameController::createWorker() {
	thread_ = new QThread(this);
	worker_ = new FrameWorker(nullptr, capC_, inferC_, frame_, gui_image_);
	worker_->moveToThread(thread_);
}

void FrameController::destroyWorker() {
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

void FrameController::initialize() {
	capC_ = new CaptureController(this, &ml_image_, &gui_image_, lock_);
	inferC_ = new InferenceController(this, &ml_image_);
	createWorker();
	connect(mainW_, &MainWindow::startCapandInfer,
		capC_, &CaptureController::startCapture);
	connect(mainW_, &MainWindow::stopCapandInfer,
		capC_, &CaptureController::stopCapture);

	connect(capC_->worker_, &CaptureWorker::frameCaptured,
		worker_, &FrameWorker::writeBuffer);
	connect(capC_->worker_, &CaptureWorker::frameCaptured,
		inferC_->worker_, &InferenceWorker::run);

	connect(inferC_->worker_post_, &InferenceWorker_Post::boundingboxReady,
		worker_, &FrameWorker::renderImage);
	connect(worker_, &FrameWorker::renderCompleted,
		mainW_, &MainWindow::refreshFrame);

	mainW_->dataLock_ = lock_;
	thread_->start();
}