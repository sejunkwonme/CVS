#include "CalibController.h"
#include "aruco_samples_utility.hpp"

CalibController::CalibController(QObject *parent)
: QObject(parent) {
	
}

CalibController::~CalibController() {
	
}

void CalibController::generatePattern() {
    dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_1000);

	boardSize_ = cv::Size(5, 7);
	squareLength_ = 0.03f;   // meter
	markerLength_ = 0.015f;

	board_ = cv::aruco::CharucoBoard::CharucoBoard(
		boardSize_,      // squaresX, squaresY
		squareLength_,        // squareLength (meters)
		markerLength_,        // markerLength
	    dictionary_
    );
}

void CalibController::renderBoardImage(int dpi) {
	double squareLength_mm = squareLength_ * 1000.0;

	double board_mm_width = boardSize_.width * squareLength_mm;
	double board_mm_height = boardSize_.height * squareLength_mm;

	const double mm_per_inch = 25.4;
	int board_px_width = static_cast<int>(board_mm_width * dpi / mm_per_inch);
	int board_px_height = static_cast<int>(board_mm_height * dpi / mm_per_inch);

	board_.generateImage(
		cv::Size(board_px_width, board_px_height),
		boardImage_
	);
}

void CalibController::createPatterPDF() {
    int dpi = 300;
    renderBoardImage(dpi);  // boardImage_ 생성

    // 안전한 복사본 생성 (QImage가 Mat의 메모리를 공유하지 않도록)
    cv::Mat imgCopy;
    boardImage_.copyTo(imgCopy);

    QImage qimg(imgCopy.data,
        imgCopy.cols,
        imgCopy.rows,
        static_cast<int>(imgCopy.step),
        QImage::Format_Grayscale8);

    // PDF Writer 생성
    QPdfWriter writer("charuco_pattern.pdf");
    writer.setPageSize(QPageSize(QPageSize::A4));

    writer.setResolution(dpi);  // very important

    QPainter painter(&writer);

    // PDF DPI와 이미지 DPI가 동일하므로,
    // drawImage를 (0,0)에 그대로 그리면 정확한 물리 크기로 출력됨.

    int pageWidthPX = writer.width();
    int pageHeightPX = writer.height();

    int offsetX = (pageWidthPX - qimg.width()) / 2;
    int offsetY = (pageHeightPX - qimg.height()) / 2;

    painter.drawImage(offsetX, offsetY, qimg);
    painter.end();
}

void CalibController::PDFChain() {
    generatePattern();
    renderBoardImage(300);
    createPatterPDF();
}

void CalibController::calibrate() {
    // === 2) CharucoDetector 생성 ===
    cv::aruco::CharucoParameters charucoParams;
    charucoParams.tryRefineMarkers = true;

    cv::aruco::CharucoDetector charucoDetector(board_, charucoParams);
    QString folderPath = "C:/Users/sjkwon/Pictures/CVS";
    // 사용할 이미지 확장자
    QStringList filters;
    QDir dir(folderPath);
    filters << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp";
    // 해당 확장자만 필터링
    dir.setNameFilters(filters);
    dir.setFilter(QDir::Files | QDir::Readable);

    QStringList imageFiles = dir.entryList();

    QStringList imagePaths;
    for (const QString& file : imageFiles) {
        imagePaths << dir.absoluteFilePath(file);
    }

    // 먼저 detectBoard와 matchImagPoint를 이용해 3D점과 2D점의 쌍을 구한다.
    std::vector<std::vector<cv::Point3f> > allObjectPoints; // 월드좌표계의 모든 3D점
    std::vector<std::vector<cv::Point2f> > allImagePoints; // 이미지상의 2D점
    cv::Size imageSize;

    foreach(const QString& path, imagePaths) {
        cv::Mat image = cv::imread(path.toStdString());
        if (image.empty()) {
            qDebug() << "Failed to read: " << path;
            continue;
        } 

        if (imageSize.width == 0)
            imageSize = image.size();

        std::vector<int> markerIds, charucoIds;
        std::vector<std::vector<cv::Point2f>> markerCorners;
        std::vector<cv::Point2f> charucoCorners;

        //! [interpolateCornersCharuco]
		// detect markers and charuco corners
        charucoDetector.detectBoard(image, charucoCorners, charucoIds, markerCorners, markerIds);
        //! [interpolateCornersCharuco]

        if (charucoIds.size() < 4) {
            qDebug() << path << ": Not enough ChArUco corners";
            continue;
        }

        // === 여기서 중요! Charuco corners → object & image points 생성 ===

        std::vector<cv::Point3f> objPoints;
        std::vector<cv::Point2f> imgPoints;
        board_.matchImagePoints(charucoCorners, charucoIds, objPoints, imgPoints);

        allObjectPoints.push_back(objPoints);
        allImagePoints.push_back(imgPoints);

        qDebug() << path << ": used " << objPoints.size() << " points";
    }

    if (allObjectPoints.size() < 3) {
        qDebug() << "Not enough valid images for calibration.";
    }

    // === 5) calibrateCamera 실행 ===
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distCoeffs = cv::Mat::zeros(8, 1, CV_64F);

    std::vector<cv::Mat> rvecs, tvecs;

    int flags = 0;

    double reprojErr = cv::calibrateCamera(
        allObjectPoints,
        allImagePoints,
        imageSize,
        cameraMatrix,
        distCoeffs,
        rvecs,
        tvecs,
        cv::noArray(), cv::noArray(), cv::noArray(),
        flags,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 200, DBL_EPSILON)
    );

    auto printMat = [](const cv::Mat& m, const QString& name) {
        qDebug() << name;
        for (int r = 0; r < m.rows; r++) {
            QString row;
            for (int c = 0; c < m.cols; c++) {
                row += QString::number(m.at<double>(r, c)) + " ";
            }
            qDebug().noquote() << row;
        }
    };

    qDebug().noquote() << "\n===== Calibration Result =====";
    qDebug().noquote() << "Reprojection error =" << reprojErr;
    printMat(cameraMatrix, "Camera Matrix:");
    printMat(distCoeffs, "Distortion Coeffs:");
}