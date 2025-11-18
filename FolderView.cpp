#include "FolderView.h"

FolderView::FolderView(QWidget *parent) : QWidget(parent) {
	mainlayout_ = new QVBoxLayout(this);
	topBarLayout_ = new QHBoxLayout();

	backButton_ = new QPushButton("<-", this);
	pathEdit_ = new QLineEdit(this);
	pathEdit_->setReadOnly(true);

	topBarLayout_->addWidget(backButton_);
	topBarLayout_->addWidget(pathEdit_);

	// ---- splitter 초기화 ----
	splitter_ = new QSplitter(this);

	leftView_ = new QTreeView(splitter_);
	listView_ = new QListView(splitter_);
	splitter_->setStretchFactor(0, 1);  // leftView
	splitter_->setStretchFactor(1, 3);  // listView

	// 우측 리스트 뷰 설정
	listView_->setStyleSheet(
		"QListView::item:selected {"
		"   background: rgba(56, 116, 242, 128);"
		"   border: 1px solid #3874f2;"
		"   color: white;"
		"}"
	);
	listView_->setViewMode(QListView::IconMode);
	listView_->setIconSize(QSize(32, 32));
	listView_->setResizeMode(QListView::Adjust);
	listView_->setSpacing(10);
	listView_->setSelectionMode(QAbstractItemView::ExtendedSelection);
	listView_->setSelectionRectVisible(true);
	listView_->setDragEnabled(true);

	// ---- 레이아웃에 배치 ----
	mainlayout_->addLayout(topBarLayout_);
	mainlayout_->addWidget(splitter_);
	setLayout(mainlayout_);

	// 뒤로가기 버튼 signal
	connect(backButton_, &QPushButton::clicked,
		this, &FolderView::onBackButtonClicked);
}

FolderView::~FolderView()
{}

void FolderView::setModel(FolderModel* model) {
	this->model_ = model;
	QFileSystemModel* fs = model_->get();
	QModelIndex home = fs->index(QDir::homePath());

	listView_->setModel(fs);
	listView_->setRootIndex(home);
	leftView_->setModel(fs);
	leftView_->setRootIndex(home);
	leftView_->setColumnHidden(1, true);
	leftView_->setColumnHidden(2, true);

	currentRoot_ = home;
	pathEdit_->setText(fs->filePath(home));
	connect(listView_, &QListView::clicked, this, &FolderView::onItemClicked);
	connect(listView_, &QListView::doubleClicked, this, &FolderView::onItemDoubleClicked);

	connect(leftView_, &QTreeView::clicked, this,
		[=](const QModelIndex& index) {
			if (fs->isDir(index)) {
				listView_->setRootIndex(index);
				currentRoot_ = index;
				pathEdit_->setText(fs->filePath(index));
			}
		}
	);
}

void FolderView::onItemClicked(const QModelIndex& index) {
	QFileSystemModel* fs = model_->get();
	QString path = fs->filePath(index);
	qDebug() << "[Clicked]" << path;
}

void FolderView::onItemDoubleClicked(const QModelIndex& index) {
	QFileSystemModel* fs = model_->get();
	QString path = fs->filePath(index);


	if (fs->isDir(index)) {
		historyStack_.push_back(currentRoot_);

		listView_->setRootIndex(index);
		currentRoot_ = index;

		pathEdit_->setText(path);
		emit directoryEntered(fs->filePath(index));
	} else {
		emit fileOpened(fs->filePath(index));
	}
}

void FolderView::onBackButtonClicked() {
	if (historyStack_.isEmpty()) {
		return;
	}

	QModelIndex previous = historyStack_.takeLast();

	listView_->setRootIndex(previous);
	currentRoot_ = previous;

	QFileSystemModel* fs = model_->get();
	QString path = fs->filePath(previous);
	pathEdit_->setText(path);
	emit directoryEntered(path);
}
