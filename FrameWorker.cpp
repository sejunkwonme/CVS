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
	static QElapsedTimer totalTimer;
	static bool first = true;

	connect(capC_->worker_, &CaptureWorker::frameCaptured, 
		this, [&]() {

            static QElapsedTimer totalTimer;
            static bool first = true;

            if (first) { totalTimer.start(); first = false; }

            QElapsedTimer t;
            bool didInference = inferC_->state();

            if (didInference) {
                t.start();
                QMetaObject::invokeMethod(
                    inferC_->worker_,
                    "run",
                    Qt::BlockingQueuedConnection
                );
                qDebug() << "[Inference time]" << t.elapsed() << "ms";
                emit withInference();
            }
            else {
                emit noInference();
            }

            qint64 total = totalTimer.restart();
            qDebug() << "[Frame end-to-end latency]" << total << "ms";
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