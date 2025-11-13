#pragma once

#include <QWidget>
#include <QListView>
#include <QVBoxLayout>
#include <QLineEdit>
#include <QHBoxLayout>
#include <QPushButton>
#include <QTreeView>
#include <QSplitter>

#include "FolderModel.h"

class FolderView  : public QWidget
{
	Q_OBJECT

public:
	FolderView(QWidget *parent);
	~FolderView();

	void setModel(FolderModel* model);
	
signals:
	void fileOpened(const QString& path);
	void directoryEntered(const QString& path);

private slots:
	void onItemClicked(const QModelIndex& index);
	void onItemDoubleClicked(const QModelIndex& index);
	void onBackButtonClicked();

private:
	QTreeView* leftView_;
	QListView* listView_;
	QSplitter* splitter_;
	QVBoxLayout* mainlayout_;
	QHBoxLayout* topBarLayout_;

	QPushButton* backButton_;
	QLineEdit* pathEdit_;

	FolderModel* model_ = nullptr;

	QModelIndex currentRoot_;
	QList<QModelIndex> historyStack_;
};