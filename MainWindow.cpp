#include <QApplication>
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QKeyEvent>
#include <QDebug>
#include <QMediaDevices>
#include <QCameraDevice>
#include <QGridLayout>
#include <QIcon>
#include <QStandardItem>
#include <qfilesystemmodel>
#include <QDir>
#include "MainWindow.h"
#include "Utilities.h"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), fileMenu(nullptr), capturer(nullptr) {
    initUI();
    data_lock = new QMutex();
}

MainWindow::~MainWindow() {

}

void MainWindow::initUI() {
    // MainWindow 의 크기 변경 및 맨위 메뉴바에서 드롭다운 메뉴 한 개 만들기
    this->setFixedSize(1920, 1080);
    fileMenu = menuBar()->addMenu("&Menu");

    // 맨 밑의 status bar 생성
    mainStatusBar = statusBar();
    mainStatusLabel = new QLabel(mainStatusBar);
    mainStatusBar->addPermanentWidget(mainStatusLabel);
    mainStatusLabel->setText("CVS is Ready");

    // MainWindow 내부의 레이아웃 생성 및 기타 위젯 생성
    QBoxLayout* main_layout = new QBoxLayout(QBoxLayout::TopToBottom); // 세로 박스 레이아웃
    QBoxLayout* cam_sub_layout = new QBoxLayout(QBoxLayout::LeftToRight); // 중간의 영상영역과 익스플로러 영역을 넣을 가로 박스 레이아웃

    shutterButton = new QPushButton(this);
    shutterButton->setText("Take a Photo");

    QBoxLayout* tools_sub_layout = new QBoxLayout(QBoxLayout::LeftToRight); // 버튼 도구들을 담는 서브레이아웃
    
    QListView* topview = new QListView(this);
    imageScene = new QGraphicsScene(this);
    imageView = new QGraphicsView(imageScene);
    QListView* rightview = new QListView(this);
    QListView* bottomView = new QListView(this);
    
    QFileSystemModel* file_model = new QFileSystemModel(this);
    file_model->setRootPath(QDir::rootPath());
    file_model->setFilter(QDir::AllEntries | QDir::NoDotAndDotDot);
    bottomView->setModel(file_model);
    connect(bottomView, &QListView::doubleClicked, this, &FileExplorer::onItemDoubleClicked);

    main_layout->addWidget(topview);
    cam_sub_layout->addWidget(imageView);
    cam_sub_layout->addWidget(rightview);
    tools_sub_layout->addWidget(shutterButton, 0, Qt::AlignHCenter);
    main_layout->addLayout(cam_sub_layout);
    main_layout->addLayout(tools_sub_layout);
    main_layout->addWidget(bottomView);

    main_layout->setStretch(0, 1);
    main_layout->setStretch(1, 4);
    main_layout->setStretch(2, 1);
    main_layout->setStretch(3, 1);
    cam_sub_layout->setStretch(0, 4);
    cam_sub_layout->setStretch(1, 1);

    connect(shutterButton, SIGNAL(clicked(bool)), this, SLOT(takePhoto()));

    saved_list = new QListView(this);
    saved_list->setViewMode(QListView::IconMode);
    saved_list->setResizeMode(QListView::Adjust);
    saved_list->setSpacing(5);
    saved_list->setWrapping(false);
    list_model = new QStandardItemModel(this);
    saved_list->setModel(list_model);
    main_layout->addWidget(saved_list);


    // 맨 위 메뉴 바의 Action 생성한다.
    createActions();

    // 맨 마지막에 Central Widget 생성 후 여기에 레이아웃을 놓아야 한다.
    QWidget* centralWidget = new QWidget(this);
    centralWidget->setLayout(main_layout);
    setCentralWidget(centralWidget);
}

void MainWindow::createActions() {
    // create actions, add them to menus
    cameraInfoAction = new QAction("Camera &Information", this);
    fileMenu->addAction(cameraInfoAction);
    openCameraAction = new QAction("&Open Camera", this);
    fileMenu->addAction(openCameraAction);
    exitAction = new QAction("E&xit", this);
    fileMenu->addAction(exitAction);

    // connect the signals and slots
    connect(exitAction, SIGNAL(triggered(bool)), QApplication::instance(), SLOT(quit()));
    connect(cameraInfoAction, SIGNAL(triggered(bool)), this, SLOT(showCameraInfo()));
    connect(openCameraAction, SIGNAL(triggered(bool)), this, SLOT(openCamera()));
}

void MainWindow::showCameraInfo() {
    QList<QCameraDevice> cameras = QMediaDevices::videoInputs();
    QString info = QString("Available Cameras: \n");

    foreach(const QCameraDevice &cameraInfo, cameras) {
        info += " - " + cameraInfo.id() + ": ";
        info += cameraInfo.description() + "\n";
    }

    QMessageBox::information(this, "Cameras", info);
}

void MainWindow::openCamera() {
    if(capturer != nullptr) {
        // if a thread is already running, stop it
        capturer->setRunning(false);
        disconnect(capturer, &Capture::frameCaptured, this, &MainWindow::updateFrame);
        disconnect(capturer, &Capture::photoTaken, this, &MainWindow::appendSavedPhoto);
        connect(capturer, &Capture::finished, capturer, &Capture::deleteLater);
    }
    // I am using my second camera whose Index is 2.  Usually, the
    // Index of the first camera is 0.
    int camID = 0;
    capturer = new Capture(camID, data_lock);
    connect(capturer, &Capture::frameCaptured, this, &MainWindow::updateFrame);
    connect(capturer, &Capture::photoTaken, this, &MainWindow::appendSavedPhoto);
    capturer->start();
    mainStatusLabel->setText(QString("Capturing Camera %1").arg(camID));
}

void MainWindow::updateFrame(cv::Mat *mat) {
    data_lock->lock();
    currentFrame = *mat;
    data_lock->unlock();

    QImage frame(
        currentFrame.data,
        currentFrame.cols,
        currentFrame.rows,
        currentFrame.step,
        QImage::Format_RGB888);
    QPixmap image = QPixmap::fromImage(frame);

    imageScene->clear();
    imageView->resetTransform();
    imageScene->addPixmap(image);
    imageScene->update();
    imageView->setSceneRect(image.rect());
}

void MainWindow::populateSavedList() {
    QDir dir(Utilities::getDataPath());
    QStringList nameFilters;
    nameFilters << "*.jpg";
    QFileInfoList files = dir.entryInfoList(
        nameFilters, QDir::NoDotAndDotDot | QDir::Files, QDir::Name);

    foreach(QFileInfo photo, files) {
        QString name = photo.baseName();
        QStandardItem *item = new QStandardItem();
        list_model->appendRow(item);
        QModelIndex index = list_model->indexFromItem(item);
        list_model->setData(index, QPixmap(photo.absoluteFilePath()).scaledToHeight(145), Qt::DecorationRole);
        list_model->setData(index, name, Qt::DisplayRole);
    }
}

void MainWindow::appendSavedPhoto(QString name) {
    QString photo_path = Utilities::getPhotoPath(name, "jpg");
    QStandardItem *item = new QStandardItem();
    list_model->appendRow(item);
    QModelIndex index = list_model->indexFromItem(item);
    list_model->setData(index, QPixmap(photo_path).scaledToHeight(145), Qt::DecorationRole);
    list_model->setData(index, name, Qt::DisplayRole);
    saved_list->scrollTo(index);
}

void MainWindow::takePhoto() {
    if(capturer != nullptr) {
        capturer->takePhoto();
    }
}
