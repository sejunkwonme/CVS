#pragma once

#include <QObject>
#include <QThread>
#include <QMutex>
#include "InferenceWorker.h"

class InferenceController : public QObject {
	Q_OBJECT

public:
	InferenceController(QObject* parent, cv::Mat frame, QMutex* lock, float** ml_image);
	~InferenceController();
	bool state();
	InferenceWorker* worker_;

private:
	void createModel(float** ml_image);
	void destroyModel();

	QThread* thread_;
	cv::Mat frame_;
	QMutex* inferLock_;
	bool inferencing_;

public slots:
	void startInference();
	void stopInference();
};