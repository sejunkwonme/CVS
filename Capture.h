#pragma once

#include <QObject>
#include <QMutex>
#include <QString>
#include <onnxruntime_cxx_api.h>
#include <cuda_runtime_api.h> 
#include <opencv2/opencv.hpp>
#include "opencv2/dnn.hpp"

class Capture : public QObject {
	Q_OBJECT
public:
    Capture(int camera, QMutex* lock, QObject* parent = nullptr);
    ~Capture();

private:
    void captureLoop(); //내부 캡쳐루프
    //void takePhoto(cv::Mat& frame); // 사진 찍기

    bool running;
    int cameraID_;
    QString videoPath;
    QMutex* data_lock_;
    cv::Mat frame_;
    bool taking_photo_;

public slots:
    void start(); // 캡처루프 시작
    void stop(); // 캡처루프 중지
    void startTakePhoto(); // 사진찍기 시작

signals:
    void frameCaptured(cv::Mat* data); // MainWindow gui 갱신용 시그널
    void photoTaken(QString name); // 사진찍기 완료 시그널
    void capfinished(); // 캡처 종료 시그널
    void callInference(cv::Mat* data);
};