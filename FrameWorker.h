#pragma once

#include <QObject>
#include <QEventLoop>
#include <QMutex>
#include <opencv2/opencv.hpp>
#include "CaptureController.h"
#include "InferenceController.h"

class FrameWorker  : public QObject {
	Q_OBJECT

public:
	FrameWorker(QObject *parent, CaptureController* capC, InferenceController* inferC);
	~FrameWorker();
	Q_INVOKABLE void run();

private:
	CaptureController* capC_;
	InferenceController* inferC_;

signals:
	void withInference();
	void noInference();
	void frameReady();

public slots:
	void finalFrameGenerated();
};