#pragma once

#include <QObject>

class InferenceController  : public QObject
{
	Q_OBJECT

public:
	InferenceController(QObject *parent);
	~InferenceController();
};

