#pragma once

#include <QObject>
#include <QDebug>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <Qmutex>
#include "kernel.cuh"

class CaptureWorker  : public QObject {
Q_OBJECT

public:
	CaptureWorker(QObject *parent, std::string pieline, float** ml_image, unsigned char** gui_image, QMutex* lock);
	~CaptureWorker();
	bool running_;
	QMutex* caplock_;

private:
	Q_INVOKABLE void run();
	Q_INVOKABLE void captureOneFrame();

	cv::VideoCapture cap_;
	cv::Mat tmp_;

	cudaStream_t preProcessStream;
	unsigned char* d_capture;
	float** ml_image_addr_;
	unsigned char** gui_image_addr_;
	uint64_t frame_count_;
	
signals:
	void frameCaptured(uint64_t);
	void captureFinished();
};