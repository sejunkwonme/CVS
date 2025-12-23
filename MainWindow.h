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

    // 카메라 캡처를 표시할 그래픽 객체 정의
    QLabel* imageLabel_;

    // 버튼 정의
    QToolButton* startButton_;

    cv::Mat currentFrame_;

    QElapsedTimer fpsTimer_;
    qint64 prevNs_ = -1;

public slots:
    void refreshFrame(cv::Mat);

signals:
    void startCapandInfer();
    void stopCapandInfer();
};