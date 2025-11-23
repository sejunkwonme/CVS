#pragma once

#include <QObject>

class ConfigController : public QObject {
	Q_OBJECT

public:
	ConfigController(QObject *parent);
	~ConfigController();
};