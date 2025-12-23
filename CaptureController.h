#pragma once

#include <QObject>
#include <QThread>
#include <QMutex>
#include "CaptureWorker.h"
#include <opencv2/opencv.hpp>

class CaptureController : public QObject {
	Q_OBJECT

public:
	CaptureController(QObject* parent, cv::Mat frame, QMutex* lock, float** ml_image, unsigned char** gui_image);
	~CaptureController();
	bool state();

	CaptureWorker* worker_;

private:
	void createCamera();
	void destroyCamera();

	QThread* thread_;
	QMutex* captureLock_;
	std::string pipeline_;
	cv::Mat frame_;
	float** ml_image_;
	unsigned char** gui_image_;


public slots:
	void startCapture();
	void stopCapture();
	void deleteAfterWait();

signals:
	void captured(cv::Mat& data);
	void stopped();
};