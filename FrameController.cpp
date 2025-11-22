#include "FrameController.h"

FrameController::FrameController(QObject* parent, MainWindow* mainW)
: QObject(parent),
frame_(cv::Mat(448,448,CV_8UC3)),
lock_(new QMutex()),
mainW_(mainW) {
}

FrameController::~FrameController() {

}

void FrameController::createWorker() {
	thread_ = new QThread(this);
	worker_ = new FrameWorker(nullptr, capC_, inferC_);
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

void FrameController::passThroughToGUI() {
	emit frameMade(frame_);
}

void FrameController::initialize() {
	capC_ = new CaptureController(this, frame_, lock_);
	inferC_ = new InferenceController(this, frame_, lock_);
	createWorker();
	connect(mainW_, &MainWindow::startCameraRequest,
		capC_, &CaptureController::startCapture);
	connect(mainW_, &MainWindow::stopCameraRequest,
		capC_, &CaptureController::stopCapture);

	connect(mainW_, &MainWindow::startInferenceRequest,
		inferC_, &InferenceController::startInference);
	connect(mainW_, &MainWindow::stopInferenceRequest,
		inferC_, &InferenceController::stopInference);

	connect(mainW_, &MainWindow::startCameraRequest,
		worker_, &FrameWorker::run);
	connect(worker_, &FrameWorker::frameReady,
		this, &FrameController::passThroughToGUI);
	connect(this, &FrameController::frameMade,
		mainW_, &MainWindow::updateFrame);
	mainW_->dataLock_ = lock_;
	thread_->start();
}