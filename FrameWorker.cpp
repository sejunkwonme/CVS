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
        this, [&](quintptr event) {
            if (inferC_->state()) {
                QMetaObject::invokeMethod(
                    inferC_->worker_,
                    "run",
                    Qt::QueuedConnection
                );
                emit withInference(event);
            }
            else {
                emit noInference(event);
            }
        });
    /*
    connect(inferC_->worker_, &InferenceWorker::backboneReady,
        this, [&]() {
                QMetaObject::invokeMethod(
                    inferC_->worker_post_,
                    "run",
                    Qt::QueuedConnection
                );
                emit withInference(event);
        });
        */
	connect(this, &FrameWorker::noInference,
		this, &FrameWorker::finalFrameGenerated);
	connect(this, &FrameWorker::withInference,
		this, &FrameWorker::finalFrameGenerated);
	loop.exec();
}

void FrameWorker::finalFrameGenerated(quintptr event) {
	emit frameReady(event);
}