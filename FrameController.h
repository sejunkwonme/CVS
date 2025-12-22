#pragma once

#include <QObject>
#include <opencv2/opencv.hpp>
#include <QThread>
#include <QMutex>
#include "FrameWorker.h"
#include "CaptureController.h"
#include "InferenceController.h"
#include "MainWindow.h"

class FrameController : public QObject {
	Q_OBJECT

public:
	FrameController(QObject *parent, MainWindow *mainW);
	~FrameController();
	void initialize();

private:
	void createWorker();
	void destroyWorker();

	cv::Mat frame_;
	QThread* thread_;
	FrameWorker* worker_;
	QMutex* lock_;
	CaptureController* capC_;
	InferenceController* inferC_;
	MainWindow* mainW_;
	float* ml_image_;

public slots:
	void passThroughToGUI();

signals:
	void frameMade(cv::Mat frame);
	void startWorker();
};