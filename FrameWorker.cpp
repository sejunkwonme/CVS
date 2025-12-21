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
        this, [&](float* d_ml_image) {
            if (inferC_->state()) {
                QMetaObject::invokeMethod(
                    inferC_->worker_,
                    "run",
                    Qt::BlockingQueuedConnection,
                    Q_ARG(float*, d_ml_image)
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