#pragma once

#include <QMainWindow>
#include <QMutex>
#include <QObject>
#include <QLabel>
#include <QPushButton>
#include <QGraphicsView>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <QMetaType>

class MainWindow : public QMainWindow {
Q_OBJECT
public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();
    QMutex* dataLock_;

private:
    void initUI();
    void createActions();

    // 메인윈도우의 윗부분의 메뉴, 액션 정의
    QMenu* fileMenu_;
    QAction* exitAction_;

    // 카메라 캡처를 표시할 그래픽 객체 정의
    QLabel* imageLabel_;

    // 버튼 정의
    QToolButton* capButton_;
    QToolButton* inferButton_;

    cv::Mat currentFrame_;

    QElapsedTimer fpsTimer_;
    qint64 prevNs_ = -1;

public slots:
    void updateFrame(quintptr event, unsigned char* gui_image, cv::Mat frame);
    void writeFrame(); // 캡처되면 더블버퍼에 프레임을 쓴다
    void renderFrame(); // inferworker_post 가 완료되어 시그널 보내면 이때 과거의 프레임에 박스 그리고 출력한다
signals:
    void startCameraRequest();
    void stopCameraRequest();
    void startInferenceRequest();
    void stopInferenceRequest();
};