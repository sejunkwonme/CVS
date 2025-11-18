#pragma once

#include <QObject>
#include <QMutex>
#include <QDebug>
#include <opencv2/opencv.hpp>

class CaptureWorker  : public QObject {
	Q_OBJECT

public:
	CaptureWorker(QObject *parent, QMutex* lock);
	~CaptureWorker();

private:
	Q_INVOKABLE void run();
	Q_INVOKABLE void processFrame();
	Q_INVOKABLE void stop();

	cv::VideoCapture cap_;
	cv::Mat frame_;
	cv::Mat tmp_;
	bool running_;
	QMutex* datalock_;

signals:
	void frameCaptured(cv::Mat* data);
};