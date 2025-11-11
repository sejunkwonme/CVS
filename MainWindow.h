#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMenuBar>
#include <QAction>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QStatusBar>
#include <QLabel>
#include <QListView>
#include <QCheckBox>
#include <QPushButton>
#include <QGraphicsPixmapItem>
#include <QMutex>
#include <QStandardItemModel>

#include "opencv2/opencv.hpp"
#include "CaptureThread.h"


class MainWindow : public QMainWindow {
Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

private:
    void initUI();
    void createActions();
    void populateSavedList();
    
private slots:
    void showCameraInfo();
    void takePhoto();

private:
    // 메인윈도우의 메뉴, 액션 정의
    QMenu* fileMenu;
    QAction* cameraInfoAction;
    QAction* openCameraAction;
    QAction* exitAction;

    // 카메라 캡처를 표시할 그래픽 객체 정의
    QGraphicsScene* imageScene;
    QGraphicsView* imageView;

    // 버튼 정의
    QPushButton* shutterButton;

    // statusbar과 그 안에 쓰일 label 정의
    QStatusBar* mainStatusBar;
    QLabel* mainStatusLabel;

private:
    cv::Mat currentFrame;

    // for capture thread
    QMutex* data_lock;
    CaptureThread* capturer;

    QStandardItemModel *list_model;

private slots:
    void openCamera();
    void updateFrame(cv::Mat*);
    void appendSavedPhoto(QString name);

private:
    QPushButton* testbutton;
};

#endif // MAINWINDOW_H
