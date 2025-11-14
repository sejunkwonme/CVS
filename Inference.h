#pragma once

#include <QString>
#include <QThread>
#include <QMutex>
#include <onnxruntime_cxx_api.h>
#include <cstdlib> 
#include <cuda_runtime_api.h> 
#include <opencv2/dnn.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/dnn.hpp"

class Inference : public QObject {
    Q_OBJECT
public:
    Inference(QMutex* lock, QObject* parent = nullptr);
    ~Inference();

private:
    void detectObjects(cv::Mat& frame);

    Ort::Env ort_env_{ ORT_LOGGING_LEVEL_WARNING, "YOLOv1" };
    Ort::Session* ort_session_;
    Ort::SessionOptions session_options_;
    Ort::MemoryInfo memory_info_{ Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU) };
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::string input_name_str_;
    std::string output_name_str_;

    QMutex* data_lock_;

public slots:
    void runInference(cv::Mat &frame);
    void finishInference();

signals:
    void InferenceDone();
};