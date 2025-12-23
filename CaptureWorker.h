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
	CaptureWorker(QObject *parent, std::string pieline, cv::Mat frame, QMutex* lock, float** ml_image, unsigned char** gui_image);
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
	cudaEvent_t preprocess_done_;
	unsigned char* d_capture;
	float** ml_image_addr_;
	unsigned char** gui_image_addr_;
	
signals:
	void frameCaptured(quintptr event);
	void captureFinished();
};