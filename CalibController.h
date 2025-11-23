#pragma once

#include <QObject>

class CalibController  : public QObject{
	Q_OBJECT

public:
	CalibController(QObject *parent);
	~CalibController();
};

