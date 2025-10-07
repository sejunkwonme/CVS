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
    QMenu* fileMenu;

    QAction* cameraInfoAction;
    QAction* openCameraAction;
    QAction* exitAction;
    QGraphicsScene* imageScene;
    QGraphicsView* imageView;
    QPushButton* shutterButton;
    QListView* saved_list;
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
};

#endif // MAINWINDOW_H
