#pragma once

#include <QObject>
#include <QDebug>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <cuda_runtime_api.h> 
#include <QElapsedTimer>
#include <opencv2/dnn/dnn.hpp>
#include "kernel.cuh"

class InferenceWorker  : public QObject {
Q_OBJECT

public:
	InferenceWorker(QObject *parent, float** ml_image, float* ml_middle_image);
	~InferenceWorker();

public slots:
    Q_INVOKABLE void run(uint64_t);

private:
    Ort::Env ort_env_{ ORT_LOGGING_LEVEL_WARNING, "YOLOv1-backbone" };
    Ort::Session* ort_session_;
    Ort::SessionOptions session_options_;
    Ort::MemoryInfo memory_info_{ Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU) };
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::string input_name_str_;
    std::string output_name_str_;
    Ort::Value input_gpu_;
    Ort::Value output_gpu_;

    float** ml_image_addr_;
    float* ml_middle_image_;
    Ort::IoBinding* binding_;
    cudaStream_t backboneStream_;
    cudaEvent_t backboneEvent_;

signals:
    void backboneReady(quintptr, uint64_t);
};