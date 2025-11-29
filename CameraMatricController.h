#pragma once

#include <QObject>

class CameraMatricController  : public QObject
{
	Q_OBJECT

public:
	CameraMatricController(QObject *parent);
	~CameraMatricController();
};

