#pragma once

#include <QObject>

class Calibration  : public QObject
{
	Q_OBJECT

public:
	Calibration(QObject *parent);
	~Calibration();
};

