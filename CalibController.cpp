#include "CalibController.h"


CalibController::CalibController(QObject *parent)
: QObject(parent) {
	
}

CalibController::~CalibController() {
	
}

void CalibController::generatePattern() {
    dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_1000);

	boardSize_ = cv::Size(6, 8);
	squareLength_ = 0.03f;   // meter
	markerLength_ = 0.02f;

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