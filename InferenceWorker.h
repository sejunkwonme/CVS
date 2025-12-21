#pragma once

#include <QObject>
#include "opencv2/opencv.hpp"
#include <onnxruntime_cxx_api.h>
#include <cuda_runtime_api.h> 
#include <QMutex>
#include <QDebug>
#include <QElapsedTimer>
#include <opencv2/dnn/dnn.hpp>
#include "kernel.cuh"

class InferenceWorker  : public QObject
{
	Q_OBJECT

public:
	InferenceWorker(QObject *parent, cv::Mat frame, QMutex* lock);
	~InferenceWorker();

private:
	Q_INVOKABLE void run(float* d_ml_image);

    Ort::Env ort_env_{ ORT_LOGGING_LEVEL_WARNING, "YOLOv1" };
    Ort::Session* ort_session_;
    Ort::SessionOptions session_options_;
    Ort::MemoryInfo memory_info_{ Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU) };
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::string input_name_str_;
    std::string output_name_str_;
    float* dptr_;

    QMutex* inferLock_;
    cv::Mat frame_;
};