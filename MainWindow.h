#pragma once

#include <QMainWindow>
#include <QMutex>
#include <QObject>
#include <QLabel>
#include <QPushButton>
#include <QGraphicsView>
#include <opencv2/opencv.hpp>

#include "Capture.h"
#include "FolderModel.h"
#include "FolderView.h"
#include "Inference.h"

class MainWindow : public QMainWindow {
Q_OBJECT
public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

private:
    void initUI();
    void createActions();

    // 메인윈도우의 윗부분의 메뉴, 액션 정의
    QMenu* fileMenu_;
    QAction* cameraInfoAction_;
    QAction* openCameraAction_;
    QAction* exitAction_;

    // 맨아랫부분의 statusbar과 그 안에 쓰일 label 정의
    QStatusBar* mainStatusBar_;
    QLabel* mainStatusLabel_;

    // 카메라 캡처를 표시할 그래픽 객체 정의
    QGraphicsScene* imageScene_;
    QGraphicsView* imageView_;

    // 버튼 정의
    QToolButton* capButton_;
    //QListView* saved_list;

    cv::Mat currentFrame_;

    // for capture thread
    QMutex* data_lock_;
    QThread* captureThread_;
    QThread* inferenceThread_;
    Capture* captureWorker_;
    Inference* inferenceWorker_;
    //QStandardItemModel* list_model;
    FolderModel* explorerModel_;
    FolderView* explorerView_;
    //void populateSavedList();
    
private slots:
    void showCameraInfo();
    //void takePhoto();
    void openCamera();
    void closeCamera();
    void startInference();
    void stopInference();
    void updateFrame(cv::Mat*);
    //void appendSavedPhoto(QString name);
    
    void onFileOpend(const QString& path);
    void onDirectoryEntered(const QString& path);

signals:
    void startCameraRequest();
    void stopcameraRequest();
    void startInferenceRequest();
    void stopInferenceRequest();
};
