#pragma once

#include <QObject>
#include <QMutex>
#include <QDebug>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "kernel.cuh"

class CaptureWorker  : public QObject {
	Q_OBJECT

public:
	CaptureWorker(QObject *parent, std::string pieline, cv::Mat frame, QMutex* lock);
	~CaptureWorker();
	bool running_;

private:
	Q_INVOKABLE void run();
	Q_INVOKABLE void captureOneFrame();

	cv::VideoCapture cap_;
	cv::Mat tmp_;
	QMutex* lock_;
	cv::Mat frame_;
	cudaStream_t preProcessStream;
	unsigned char* d_capture;
	unsigned char* d_gui_image_BGR;
	float* d_ml_image_RGB;
	unsigned char* d_gui_image_BGR_cropped;
	float* d_ml_image_RGB_cropped;
	
signals:
	void frameCaptured();
	void captureFinished();
};