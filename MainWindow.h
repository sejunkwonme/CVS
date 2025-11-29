#pragma once

#include <QMainWindow>
#include <QMutex>
#include <QObject>
#include <QLabel>
#include <QPushButton>
#include <QGraphicsView>
#include <QListView>
#include <opencv2/opencv.hpp>

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

    // 카메라 캡처를 표시할 QLabel 객체 정의
    QLabel* imageLabel_;

    // 우측의 설정패널창 정의
    QListView* rightView_;

    // 버튼 정의
    QToolButton* capButton_;
    QToolButton* inferButton_;
    QToolButton* shutterButton_;

    cv::Mat currentFrame_;

public slots:
    void updateFrame(cv::Mat mat);
signals:
    void startCameraRequest();
    void stopCameraRequest();
    void startInferenceRequest();
    void stopInferenceRequest();
    void takePhoto();
};
