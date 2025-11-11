#ifndef CAPTURETHREAD_H
#define CAPTURETHREAD_H

#include <QString>
#include <QThread>
#include <QMutex>
#include <onnxruntime_cxx_api.h>
#include <cstdlib> 
#include <cuda_runtime_api.h> 

#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/dnn.hpp"


using namespace std;

class CaptureThread : public QThread {
    Q_OBJECT
    public:
        CaptureThread(int camera, QMutex* lock);
        CaptureThread(QString videoPath, QMutex* lock);
        ~CaptureThread();
        void setRunning(bool run) { running = run;};
        void takePhoto() {taking_photo = true;};

    protected:
        void run() override;

    signals:
        void frameCaptured(cv::Mat* data);
        void photoTaken(QString name);

    private:
        bool running;
        int cameraID;
        QString videoPath;
        QMutex* data_lock;
        cv::Mat frame;
        bool taking_photo;

    private:
        // --- YOLOv1 ONNX Runtime 멤버 ---
        Ort::Env ort_env{ ORT_LOGGING_LEVEL_WARNING, "YOLOv1" };
        Ort::Session* ort_session = nullptr;
        Ort::SessionOptions session_options;
        Ort::MemoryInfo memory_info{ Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU) };
        std::vector<const char*> input_names;
        std::vector<const char*> output_names;

        void detectObjects(cv::Mat& frame);  // YOLO 추론 함수
        std::string input_name_str;   // 추가: 문자열 보관(수명 유지)
        std::string output_name_str;  // 추가


    private:
        void takePhoto(cv::Mat& frame);
        void detectFaces(cv::Mat& frame);

    private:
        //face detection
        cv::CascadeClassifier* classifier;

        // video saving
        // int frame_width, frame_height;
};

#endif // CAPTURETHREAD_H
