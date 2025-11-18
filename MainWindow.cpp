#include <QApplication>
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QKeyEvent>
#include <QDebug>
#include <QMediaDevices>
#include <QCameraDevice>
#include <QGridLayout>
#include <QGraphicsView>
#include <QAction>
#include <QGraphicsScene>
#include <QMenuBar>
#include <QLabel>
#include <QStatusBar>
#include <QIcon>
#include <QStandardItem>
#include <qfilesystemmodel>
#include <QDir>
#include <QToolButton>
#include "MainWindow.h"
#include "CaptureWorker.h"

MainWindow::MainWindow(QWidget* parent)
: QMainWindow(parent),
fileMenu_(nullptr),
cameraInfoAction_(nullptr),
openCameraAction_(nullptr),
exitAction_(nullptr),
mainStatusBar_(nullptr),
mainStatusLabel_(nullptr),
imageScene_(nullptr),
imageView_(nullptr),
capButton_(nullptr),
currentFrame_(),
data_lock_(new QMutex()),
captureThread_(nullptr),
inferenceThread_(nullptr),
captureWorker_(nullptr),
inferenceWorker_(nullptr),
explorerModel_(nullptr),
explorerView_(nullptr) {
    initUI(); // GUI 요소 초기화
    int camID = 0;
    // CaptureController 객체 초기화 후 connect
    CaptureC_ = new CaptureController(this);
    data_lock_ = CaptureC_->datalock_;
    connect(this, &MainWindow::startCameraRequest, CaptureC_, &CaptureController::startCapture);
    connect(this, &MainWindow::stopCameraRequest, CaptureC_, &CaptureController::stopCapture);
    connect(CaptureC_, &CaptureController::captured, this, &MainWindow::updateFrame);
    

    // Inference 객체 초기화 후 connect
}

MainWindow::~MainWindow() {

}

void MainWindow::initUI() {
    // MainWindow 의 크기 변경 및 맨위 메뉴바에서 드롭다운 메뉴 한 개 만들기
    this->setFixedSize(1920, 1080);
    fileMenu_ = menuBar()->addMenu("&Menu");
    // 맨 위 메뉴 바의 Action 생성한다.
    createActions();

    // 맨 밑의 status bar 생성
    mainStatusBar_ = statusBar();
    mainStatusLabel_ = new QLabel(mainStatusBar_);
    mainStatusBar_->addPermanentWidget(mainStatusLabel_);
    mainStatusLabel_->setText("CVS is Ready");

    // MainWindow 내부의 레이아웃 생성 및 기타 위젯 생성
    QBoxLayout* main_layout = new QVBoxLayout(); // 세로 박스 레이아웃
    QBoxLayout* cam_sub_layout = new QHBoxLayout(); // 중간의 영상영역과 익스플로러 영역을 넣을 가로 박스 레이아웃
    QBoxLayout* tools_sub_layout = new QHBoxLayout(); // 버튼 도구들을 담는 서브레이아웃
    QListView* topview = new QListView(this);
    QListView* rightview = new QListView(this);

    imageScene_ = new QGraphicsScene(this);
    imageView_ = new QGraphicsView(imageScene_);

    capButton_ = new QToolButton(this);
    capButton_->setText("Start Capture");
    capButton_->setCheckable(true);
    connect(capButton_, &QToolButton::toggled, this,
        [&](bool checked) {
            if (checked) {
                emit startCameraRequest();
                qDebug() << "camera started";
                capButton_->setText("Stop Camera");
                mainStatusLabel_->setText(QString("Capturing Camera %1").arg(0));
            }
            else {
                emit stopCameraRequest();
                qDebug() << "camera stopped";
                capButton_->setText("Start Camera");
            }
        }
    );
    /*
    inferButton_ = new QToolButton(this);
    inferButton_->setText("Start Inference");
    inferButton_->setCheckable(true);
    connect(inferButton_, &QToolButton::toggled, this,
        [&](bool checked) {
            if (checked) {
                emit startInferenceRequest();
                qDebug() << "Inference started";
                inferButton_->setText("Stop Inference");
            }
            else {
                emit stopInferenceRequest();
                qDebug() << "Inference stopped";
                inferButton_->setText("Start Inference");
            }
        }
    );
	*/

    explorerModel_ = new FolderModel(this);
    explorerView_ = new FolderView(this);
    explorerView_->setModel(explorerModel_);
    connect(explorerView_, &FolderView::fileOpened, this, &MainWindow::onFileOpend);
    connect(explorerView_, &FolderView::directoryEntered, this, &MainWindow::onDirectoryEntered);

    //QListView* bottomView = new QListView(this);

    main_layout->addWidget(topview);
    cam_sub_layout->addWidget(imageView_);
    cam_sub_layout->addWidget(rightview);
    tools_sub_layout->addWidget(capButton_, 0, Qt::AlignHCenter);
    //tools_sub_layout->addWidget(inferButton_, 0, Qt::AlignHCenter);
    main_layout->addLayout(cam_sub_layout);
    main_layout->addLayout(tools_sub_layout);
    main_layout->addWidget(explorerView_);
    main_layout->setStretch(0, 1);
    main_layout->setStretch(1, 4);
    main_layout->setStretch(2, 1);
    main_layout->setStretch(3, 1);
    cam_sub_layout->setStretch(0, 4);
    cam_sub_layout->setStretch(1, 1);

    //connect(shutterButton_, SIGNAL(clicked(bool)), this, SLOT(takePhoto()));
    /*
    saved_list = new QListView(this);
    saved_list->setViewMode(QListView::IconMode);
    saved_list->setResizeMode(QListView::Adjust);
    saved_list->setSpacing(5);
    saved_list->setWrapping(false);
    list_model = new QStandardItemModel(this);
    saved_list->setModel(list_model);
    main_layout->addWidget(saved_list);
	*/

    //connect(this, &MainWindow::startInferenceRequest, this, &MainWindow::startInference);
    //connect(this, &MainWindow::stopInferenceRequest, this, &MainWindow::stopInference);
    // 맨 마지막에 Central Widget 생성 후 여기에 레이아웃을 놓아야 한다.
    QWidget* centralWidget = new QWidget(this);
    centralWidget->setLayout(main_layout);
    setCentralWidget(centralWidget);
}

void MainWindow::createActions() {
    // 맨 위 메뉴바의 액션을 생성하여 시그널 슬롯 연결
    cameraInfoAction_ = new QAction("Camera &Information", this);
    fileMenu_->addAction(cameraInfoAction_);
    openCameraAction_ = new QAction("&Open Camera", this);
    fileMenu_->addAction(openCameraAction_);
    exitAction_ = new QAction("E&xit", this);
    fileMenu_->addAction(exitAction_);

    // connect the signals and slots
    connect(exitAction_, &QAction::triggered, QApplication::instance(), &QApplication::quit);
    connect(cameraInfoAction_, &QAction::triggered, this, &MainWindow::showCameraInfo);
    connect(openCameraAction_, &QAction::triggered, this, &MainWindow::openCamera);
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
    int camID = 0;
    captureThread_ = new QThread(this);
    captureWorker_ = new Capture(camID, data_lock_);
    captureWorker_->moveToThread(captureThread_);

    // 스레드 시작하면 capture 도 같이 시작
    connect(captureThread_,&QThread::started,
        captureWorker_, &Capture::start);

    // GUI 업데이트용 connect
    connect(captureWorker_, &Capture::frameCaptured,
        this, &MainWindow::updateFrame);

    // worker는 스레드 종료 후 삭제
    connect(captureWorker_, &Capture::capfinished,
        captureWorker_, &QObject::deleteLater);

    // thread는 finished() 후 삭제
    connect(captureThread_, &QThread::finished,
        captureThread_, &QObject::deleteLater);

    // 스레드 시작
    captureThread_->start();
    mainStatusLabel_->setText(QString("Capturing Camera %1").arg(camID));
}

void MainWindow::closeCamera() {
    // nullptr 검사
    if (!captureWorker_ || !captureThread_)
        return;

    QMetaObject::invokeMethod(captureWorker_, "stop", Qt::QueuedConnection);
    captureThread_->quit();
    captureThread_->wait();
    qDebug() << "camera off";
    captureThread_ = nullptr;
    captureWorker_ = nullptr;

    mainStatusLabel_->setText("Camera closed");
}

void MainWindow::startInference() {
    //nullptr 검사 둘다 nullptr이면 추론 시작하면 안됨
    if (!captureWorker_ || !captureThread_)
        return;

    int camID = 0;
    inferenceThread_ = new QThread(this);
    inferenceWorker_ = new Inference(data_lock_);

    // movetoThread로 스레드에 QObject 작업 할당
    inferenceWorker_->moveToThread(inferenceThread_);

    // 시그널 슬롯 연결
    connect(inferenceThread_, &QThread::started, captureWorker_, &Capture::startInference);
    connect(captureWorker_, &Capture::callInference, inferenceWorker_, &Inference::runInference, Qt::BlockingQueuedConnection);
    connect(captureWorker_, &Capture::inferfinished, inferenceWorker_, &QObject::deleteLater, Qt::BlockingQueuedConnection);
    connect(inferenceThread_, &QThread::finished, inferenceThread_, &QObject::deleteLater);

    // 스레드 시작
    inferenceThread_->start();
    mainStatusLabel_->setText(QString("Capturing Camera %1").arg(camID));
}

void MainWindow::stopInference() {
    if (!inferenceThread_ || !inferenceWorker_)
        return;

    QMetaObject::invokeMethod(captureWorker_, "stopInference", Qt::BlockingQueuedConnection);
    inferenceThread_->quit();
    inferenceThread_->wait();
    qDebug() << "Inference off";
    inferenceThread_ = nullptr;
    inferenceWorker_ = nullptr;

    mainStatusLabel_->setText("Inference closed");
}

void MainWindow::updateFrame(cv::Mat *mat) {
    data_lock_->lock();
    currentFrame_ = *mat;
    data_lock_->unlock();

    QImage frame(
        currentFrame_.data,
        currentFrame_.cols,
        currentFrame_.rows,
        currentFrame_.step,
        QImage::Format_RGB888);
    QPixmap image = QPixmap::fromImage(frame);

    imageScene_->clear();
    imageView_->resetTransform();
    imageScene_->addPixmap(image);
    imageScene_->update();
    imageView_->setSceneRect(image.rect());
}


/*
void MainWindow::takePhoto() {
    if(captureWorker_ != nullptr) {
        captureWorker_->takePhoto();
    }
}
*/

void MainWindow::onFileOpend(const QString& path) {
    qDebug() << "File open requested:" << path;
}

void MainWindow::onDirectoryEntered(const QString& path) {
    qDebug() << "Entered directory:" << path;
}



/*
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
*/

/*
void MainWindow::appendSavedPhoto(QString name) {
    QString photo_path = Utilities::getPhotoPath(name, "jpg");
    QStandardItem *item = new QStandardItem();
    list_model->appendRow(item);
    QModelIndex index = list_model->indexFromItem(item);
    list_model->setData(index, QPixmap(photo_path).scaledToHeight(145), Qt::DecorationRole);
    list_model->setData(index, name, Qt::DisplayRole);
    saved_list->scrollTo(index);
}
*/