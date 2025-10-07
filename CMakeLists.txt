cmake_minimum_required(VERSION 3.7)


project(Facetious VERSION 0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
 
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets Multimedia Concurrent)
find_package(OpenCV REQUIRED)


# include 경로 자동 설정
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
  
qt_add_executable(${PROJECT_NAME}
    main.cpp
    MainWindow.h MainWindow.cpp
    CaptureThread.h CaptureThread.cpp
    Utilities.h Utilities.cpp
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    Qt${QT_VERSION_MAJOR}::Multimedia
    Qt${QT_VERSION_MAJOR}::Concurrent
    ${OpenCV_LIBS}
)
 
# 전처리 매크로 정의
target_compile_definitions(${PROJECT_NAME} PRIVATE
    OPENCV_DATA_DIR="C:/Users/sejunkwon/opencv/data/"
)