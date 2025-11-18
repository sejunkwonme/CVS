#pragma once

#include <QObject>
#include <QFileSystemModel>

class FolderModel  : public QObject
{
	Q_OBJECT

public:
	explicit FolderModel(QObject *parent = nullptr);
	~FolderModel();

	QFileSystemModel* get() const;

private:
	QFileSystemModel* model_;
};