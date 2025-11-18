#pragma once

#include <QObject>

class Camera  : public QObject
{
	Q_OBJECT

public:
	Camera(QObject *parent);
	~Camera();
};

