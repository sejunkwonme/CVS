#pragma once

#include <QObject>
#include <QThread>
#include <QMutex>
#include "CaptureWorker.h"
#include <opencv2/opencv.hpp>

class CaptureController  : public QObject {
	Q_OBJECT

public:
	CaptureController(QObject *parent);
	~CaptureController();

	QMutex* datalock_;

private:
	void createCamera();
	void destroyCamera();

	CaptureWorker* worker_;
	QThread* thread_;

public slots:
	void startCapture();
	void stopCapture();
	void frameFromWorker(cv::Mat* data);

signals:
	void captured(cv::Mat* data);
};