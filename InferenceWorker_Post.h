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

class InferenceWorker_Post  : public QObject
{
	Q_OBJECT

public:
	InferenceWorker_Post(QObject *parent, cv::Mat frame, QMutex* lock, float* ml_middle_image);
	~InferenceWorker_Post();
    Q_INVOKABLE void run();

private:
    Ort::Env ort_env_{ ORT_LOGGING_LEVEL_WARNING, "YOLOv1-head" };
    Ort::Session* ort_session_;
    Ort::SessionOptions session_options_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::string input_name_str_;
    std::string output_name_str_;
    Ort::Value input_gpu_;
    Ort::Value output_gpu_;
    Ort::RunOptions run_opts_;

    float* middle_image_;
    float** ml_image_addr_;
    float* d_out_;
    Ort::IoBinding* binding_;
    cudaStream_t head_stream_;

    QMutex* inferLock_;
    cv::Mat frame_;
};