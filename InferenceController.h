#pragma once

#include <QObject>
#include <QThread>
#include <QMutex>
#include "InferenceWorker.h"
#include "InferenceWorker_Post.h"

class InferenceController : public QObject {
	Q_OBJECT

public:
	InferenceController(QObject* parent, cv::Mat frame, QMutex* lock, float** ml_image);
	~InferenceController();
	bool state();
	InferenceWorker* worker_;
	InferenceWorker_Post* worker_post_;

private:
	void createModel(float** ml_image);
	void destroyModel();

	QThread* thread_;
	QThread* thread_post_;
	cv::Mat frame_;
	QMutex* inferLock_;
	bool inferencing_;
	float* middle_image_;

public slots:
	void startInference();
	void stopInference();
};