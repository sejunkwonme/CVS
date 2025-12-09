#pragma once

#include <QObject>
#include <QMutex>
#include <QDebug>
#include <opencv2/opencv.hpp>

class CaptureWorker  : public QObject {
	Q_OBJECT

public:
	CaptureWorker(QObject *parent, std::string pieline, cv::Mat frame, QMutex* lock);
	~CaptureWorker();
	bool running_;

private:
	Q_INVOKABLE void run();
	void captureOneFrame();

	cv::VideoCapture cap_;
	cv::Mat tmp_;
	QMutex* lock_;
	cv::Mat frame_;

signals:
	void frameCaptured();
	void captureFinished();
};