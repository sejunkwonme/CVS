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
    QAction* cameraInfoAction_;
    QAction* openCameraAction_;
    QAction* exitAction_;

    // 카메라 캡처를 표시할 그래픽 객체 정의
    QGraphicsScene* imageScene_;
    QGraphicsView* imageView_;
    QListView* rightView_;

    // 버튼 정의
    QToolButton* capButton_;
    QToolButton* inferButton_;

    cv::Mat currentFrame_;

public slots:
    void showCameraInfo();
    void updateFrame(cv::Mat mat);
signals:
    void startCameraRequest();
    void stopCameraRequest();
    void startInferenceRequest();
    void stopInferenceRequest();
};
