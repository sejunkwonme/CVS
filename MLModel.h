#pragma once

#include <QObject>

class MLModel  : public QObject
{
	Q_OBJECT

public:
	MLModel(QObject *parent);
	~MLModel();
};

