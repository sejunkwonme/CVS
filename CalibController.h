#pragma once

#include <QObject>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
#include <QImage>
#include <QPdfWriter>
#include <QPainter>
#include <iostream>
#include <vector>

class CalibController  : public QObject{
	Q_OBJECT

public:
	CalibController(QObject *parent);
	~CalibController();

	void generatePattern();
	void renderBoardImage(int dpi);
	void createPatterPDF();
	void PDFChain();
	void drawArucoMarkerVector();

	cv::aruco::Dictionary dictionary_;
	cv::aruco::CharucoBoard board_;
	cv::Mat boardImage_;

	float squareLength_;
	float markerLength_;
	cv::Size boardSize_;
};