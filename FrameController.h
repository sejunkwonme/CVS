#pragma once

#include <QObject>
#include <opencv2/opencv.hpp>
#include <QThread>
#include <QMutex>
#include <cuda_runtime_api.h>
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

	cv::Mat frame_[4];
	QThread* thread_;
	FrameWorker* worker_;
	QMutex* lock_;	
	CaptureController* capC_;
	InferenceController* inferC_;
	MainWindow* mainW_; // 시그널 보내기 위한 메인윈도우 클래스 포인터
	float* ml_image_; //전처리된 ml 이미지 담는 디바이스 포인터
	unsigned char* gui_image_; // 전처리된 gui 이미지 담는 디바이스 포인터

public slots:

signals:
	//void frameMade(quintptr event, unsigned char* gui_image, cv::Mat frame_);
};