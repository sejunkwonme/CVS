#ifndef CAPTURETHREAD_H
#define CAPTURETHREAD_H

#include <QString>
#include <QThread>
#include <QMutex>

#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/objdetect.hpp"
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
    void takePhoto(cv::Mat& frame);
    void detectFaces(cv::Mat& frame);

private:
    //face detection
    cv::CascadeClassifier* classifier;

    // video saving
    // int frame_width, frame_height;
};

#endif // CAPTURETHREAD_H
