#ifndef CAPTURETHREAD_H
#define CAPTURETHREAD_H

#include <QObject>
#include <QMutex>
#include <QString>
#include <onnxruntime_cxx_api.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

class Capture : public QObject {
	Q_OBJECT
public:
    Capture(int camera, QMutex* lock, QObject* parent = nullptr);
    Capture(QString videoPath, QMutex* lock);
    ~Capture();
    void setRunning(bool run) { running = run;};
    

public slots:
    void start(); // 캡처루프 시작
    void stop(); // 캡처루프 중지
    void takePhoto() { taking_photo = true; }


signals:
    void frameCaptured(cv::Mat* data); // MainWindow gui 갱신용 시그널
    void photoTaken(QString name); // 사진찍기용 시그널
    void finished(); // 스레드 종료 시그널

private:
    void captureLoop(); //내부 캡쳐루프

private:
    bool running;
    int cameraID;
    QString videoPath;
    QMutex* data_lock = nullptr;
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


    // video saving
    // int frame_width, frame_height;
};

#endif // CAPTURETHREAD_H
