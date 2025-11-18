#pragma once

#include <QObject>

class ConfigRepository  : public QObject
{
	Q_OBJECT

public:
	ConfigRepository(QObject *parent);
	~ConfigRepository();
};

