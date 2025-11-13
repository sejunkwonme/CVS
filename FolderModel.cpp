#include "FolderModel.h"

FolderModel::FolderModel(QObject *parent)
: QObject(parent) {
	model_ = new QFileSystemModel(this);
	model_->setFilter(QDir::AllEntries | QDir::NoDotAndDotDot);
	model_->setRootPath(QDir::homePath());
}

FolderModel::~FolderModel() {
	
}

QFileSystemModel* FolderModel::get() const {
	return model_;
}
