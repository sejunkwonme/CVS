#pragma once

#include <QObject>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <cuda_runtime_api.h>
#include <QDebug>
#include <QMutex>
#include <QElapsedTimer>
#include <opencv2/dnn/dnn.hpp>

class InferenceWorker_Post  : public QObject {
Q_OBJECT

public slots:
    Q_INVOKABLE void run(uint64_t);

public:
	InferenceWorker_Post(QObject *parent, float* ml_middle_image);
	~InferenceWorker_Post();

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
    Ort::IoBinding* binding_;

    cudaStream_t head_stream_;
    float* middle_image_;
    float* d_out_;
    
signals:
    void boundingboxReady(float*, uint64_t);
};