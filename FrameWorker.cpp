#include "FrameWorker.h"

FrameWorker::FrameWorker(QObject* parent, CaptureController* capC, InferenceController* inferC)
: QObject(parent),
capC_(capC),
inferC_(inferC) {
}

FrameWorker::~FrameWorker() {
}

void FrameWorker::run() {
	QEventLoop loop;


    connect(capC_->worker_, &CaptureWorker::frameCaptured,
        this, [&]() {
            if (inferC_->state()) {
                QMetaObject::invokeMethod(
                    inferC_->worker_,
                    "run",
                    Qt::BlockingQueuedConnection
                );
                emit withInference();
            }
            else {
                emit noInference();
            }
        });
	connect(this, &FrameWorker::noInference,
		this, &FrameWorker::finalFrameGenerated);
	connect(this, &FrameWorker::withInference,
		this, &FrameWorker::finalFrameGenerated);
	loop.exec();
}

void FrameWorker::finalFrameGenerated() {
	emit frameReady();
}