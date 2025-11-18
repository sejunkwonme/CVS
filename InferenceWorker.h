#pragma once

#include <QObject>

class InferenceWorker  : public QObject
{
	Q_OBJECT

public:
	InferenceWorker(QObject *parent);
	~InferenceWorker();
};

