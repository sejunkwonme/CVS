#pragma once

#include <QObject>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/aruco/aruco_calib.hpp>
#include <QImage>
#include <QPdfWriter>
#include <QPainter>
#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <QString>
#include <QVector>
#include <QStringList>
#include <QDebug>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <QDir>

class CalibController  : public QObject{
	Q_OBJECT

public:
	CalibController(QObject *parent);
	~CalibController();

	void generatePattern();
	void renderBoardImage(int dpi);
	void createPatterPDF();
	void PDFChain();
	void calibrate();

	cv::aruco::Dictionary dictionary_;
	cv::aruco::CharucoBoard board_;
	cv::Mat boardImage_;

	float squareLength_;
	float markerLength_;
	cv::Size boardSize_;
};