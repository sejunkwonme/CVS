#pragma once

#include <QAbstractItemModel>

class InfoModel  : public QAbstractItemModel
{
	Q_OBJECT

public:
	InfoModel(QObject *parent);
	~InfoModel();
};