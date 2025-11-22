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

private:
	Q_INVOKABLE void run();
	Q_INVOKABLE void captureOneFrame();
	Q_INVOKABLE void stop();

	cv::VideoCapture cap_;
	cv::Mat tmp_;
	bool running_;
	QMutex* lock_;
	cv::Mat frame_;

signals:
	void frameCaptured();
};