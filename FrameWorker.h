#pragma once

#include <QObject>
#include <QEventLoop>
#include <QMutex>
#include <opencv2/opencv.hpp>
#include "CaptureController.h"
#include "InferenceController.h"

class FrameWorker  : public QObject {
	Q_OBJECT

public:
	FrameWorker(QObject *parent, CaptureController* capC, InferenceController* inferC, cv::Mat(&frame)[4], unsigned char* gui_image);
	~FrameWorker();

private:
	CaptureController* capC_;
	InferenceController* inferC_;
	cv::Mat frame_[4];
	unsigned char* gui_image_;

signals:
	void renderCompleted(cv::Mat);
public slots:
	void writeBuffer(uint64_t framecount);
	void renderImage(float*, uint64_t);
};